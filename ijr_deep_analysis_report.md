# Deep Reverse-Engineering Report for `ocn.py` and `outputs/ijr_summary.txt`

This report is based strictly on executable behavior in `ocn.py` and the generated text summary in `outputs/ijr_summary.txt`. When comments, print strings, or scenario names disagree with the code path that actually runs, this report follows the code.

Two recurring labels are used throughout:

- Implemented: behavior directly visible in the executable code.
- Inferred: likely intent or research motivation suggested by the code structure and the generated results, but not stated explicitly by the implementation.

## 1. Executive Overview

At a high level, this project models a packet-switched interposer network that connects several chiplets over a shared 4x4 mesh of interposer routers. The active experiment layout is a five-chiplet system built by `build_image_layout()` (`ocn.py:239`): four GPU chiplets occupy the four quadrants of the mesh through four boundary routers each, and one CPU chiplet attaches through two boundary routers that share interior interposer routers with the top GPU chiplets.

The simulator is trying to answer a specific systems question: if you keep the interposer relatively lean in VC count, can you recover high-load throughput and avoid pathological dependency cycles by combining adaptive minimal routing with a DRAIN-style escape/drain mechanism, instead of relying only on deterministic routing, turn restrictions, or brute-force many-VC provisioning?

In the context of this codebase, "composable DRAIN for chiplets" means:

- multiple chiplets are composed onto a shared interposer through a configurable `ChipletSpec` mapping;
- the network carries protocol traffic with request/response dependency structure, not just anonymous packets;
- a DRAIN-like mechanism is overlaid on the shared interposer escape VCs using a static per-input-port drain turn-table and a periodic drain scheduler;
- the escape/drain machinery is meant to be independent of which chiplet generated a packet.

That said, the implementation is only partially composable in the strong sense:

- topology composition is supported through `ChipletSpec`;
- protocol composition is approximated through per-class source/ejection queues and REQ->RESP transactions;
- but the actual DRAIN turn-table generator only supports the standard fault-free 4x4 mesh (`_validate_standard_drain_mesh`, `ocn.py:1188`), so the DRAIN control plane is not generalized to arbitrary topologies or faulted meshes.
- BUT it could very easilty and we were able to do it earlier. Just didn't matter for the current tests

## 2. How the Whole System Works

### 2.1 Topology and structural model

The active topology is built in three layers:

- `ChipletSpec` (`ocn.py:149`) describes each chiplet and which interposer routers each of its boundary routers attaches to.
- `BoundaryRouter` (`ocn.py:554`) is the endpoint-facing anchor object for packet source and destination identity.
- `InterposerRouter` (`ocn.py:618`) is the actual switching node inside the 4x4 interposer mesh.

The default image layout created by `build_image_layout()` is:

- `GPU_TL`: boundary routers on `IR(0,0)`, `IR(0,1)`, `IR(1,0)`, `IR(1,1)`
- `GPU_TR`: boundary routers on `IR(0,2)`, `IR(0,3)`, `IR(1,2)`, `IR(1,3)`
- `GPU_BL`: boundary routers on `IR(2,0)`, `IR(2,1)`, `IR(3,0)`, `IR(3,1)`
- `GPU_BR`: boundary routers on `IR(2,2)`, `IR(2,3)`, `IR(3,2)`, `IR(3,3)`
- `CPU`: boundary routers on `IR(1,1)` and `IR(1,2)`, shared with the top GPU quadrants

This produces:

- 16 interposer routers
- 48 directed inter-router links
- 18 boundary routers
- 5 chiplets

An important implementation detail is that multiple chiplets may share the same interposer router. The CPU does exactly that. This matters because the code is not modeling a private router per chiplet; it is modeling composition over a shared interposer fabric.

### 2.2 Router, port, VC, packet, and flit model

The packet transport model is wormhole-like:

- `Packet` (`ocn.py:313`) is the application-visible network unit.
- Each packet is decomposed into `FLITS_PER_PACKET = 4` flits in `Packet.__post_init__`.
- `Flit` (`ocn.py:275`) tracks `flow_id`, worm-head/tail state, and whether the flit is in the escape VC class.
- `VC` (`ocn.py:385`) is a FIFO with wormhole allocation semantics.
- `InputPort` (`ocn.py:488`) contains `num_normal_vcs` normal VCs plus exactly one escape VC.

The relevant wormhole rules are:

- a worm head allocates a VC;
- body flits must follow the already-allocated flow;
- a tail flit releases the VC allocation in `VC.pop()`;
- each router remembers the chosen next hop for a flow in `router.flow_output_table`, so body/tail flits follow the head without recomputing routing every hop.

Each `InterposerRouter` has five input ports:

- `N`, `S`, `E`, `W`: from neighboring routers
- `Down`: from attached boundary routers

Each cycle, at most one flit may leave a router on each output direction. Arbitration is random among eligible contenders.

### 2.3 Boundary routers are lighter-weight than the top-of-file comments suggest

The module header still describes a boundary-router outbound buffering phase, but the current implementation no longer does that.

Implemented behavior:

- boundary routers do not hold outbound VCs;
- injection goes directly into the attached interposer router's `Down` input-port normal VC;
- boundary routers mainly serve as attachment points, source/destination identifiers, and delivery bookkeeping objects.

This is stated explicitly in `BoundaryRouter` and in the injection code path (`_try_inject_packet_into_network`, `ocn.py:2232`).

### 2.4 Simulation cycle, step by step

The outer experiment drivers call three things every cycle:

1. `mesh.inject_random_packets(cycle, rng)`
2. `mesh.step(cycle)`
3. `detector.check(mesh, cycle)`

This means injection happens before router movement, and deadlock detection observes post-step state.

Inside `InterposerMesh.step()` (`ocn.py:2355`), the cycle is:

1. Advance the DRAIN FSM by calling `_advance_drain_fsm()`.
2. If a full drain is active:
   - perform one synchronized DRAIN hop over escape VCs;
   - perform one synchronized escape-flit ejection stage;
   - decrement remaining full-drain hops;
   - service endpoint ejection queues;
   - skip all normal forwarding for this cycle.
3. If a regular drain window is active:
   - perform one synchronized DRAIN hop over escape VCs;
   - perform one synchronized escape-flit ejection stage;
   - decrement remaining regular-drain hops;
   - if `STRICT_REGULAR_DRAIN` is `True`, skip normal forwarding;
   - otherwise continue to normal forwarding.
4. Perform normal router forwarding and normal ejection:
   - inspect the head flit of every VC;
   - compute desired outputs;
   - group contenders by output port;
   - choose at most one winner per output;
   - pop, forward, or eject the winning flit.
5. Service endpoint protocol ejection queues by calling `_service_protocol_ejection_queues()`.

This is already enough to understand a major design choice: DRAIN is not event-triggered by the deadlock detector. It is a periodic scheduler that runs whether or not a deadlock report was generated.

### 2.5 Packet creation, injection, routing, advance, ejection, delivery

#### Packet creation

In the active configuration, `PROTOCOL_MODE_ENABLED = True`, so the traffic source is not legacy single-class traffic. It is REQ-driven protocol traffic:

- `_generate_protocol_request_traffic()` (`ocn.py:2109`) visits each chiplet each cycle.
- For each chiplet, it asks `current_injection_rate(cycle, rng)`.
- If a Bernoulli trial succeeds, it creates a REQ packet to a random different chiplet.
- It creates a `TransactionRecord` and enqueues the packet in the source chiplet's per-class injection queue if capacity and outstanding limits allow.

The source queue and outstanding limit checks are governed by `ProtocolConfig` (`ocn.py:178`).

#### Injection into the interposer

Actual network injection is handled by `_try_inject_packet_into_network()` (`ocn.py:2232`):

- one queued packet per chiplet may be injected per cycle in protocol mode;
- the packet is tried against the source chiplet's candidate boundary routers;
- the code prefers the packet's assigned source BR, then randomizes the rest;
- injection succeeds only if a normal VC in the attached interposer router's `Down` input port is unallocated and has room for all 4 flits at once;
- on success, all 4 flits are pushed immediately into that `Down` VC in the same cycle.

This is a strong modeling choice. Injection is packet-atomic into a router input VC, not serialized flit-by-flit across a boundary-router pipeline.

#### Routing in the network

Routing depends on the scenario:

- `xy_next_hop()` for deterministic column-first routing
- `yx_next_hop()` for deterministic row-first routing
- `_minimal_directions()` plus random choice for fully adaptive minimal routing
- `_turn_restricted_directions()` for West-First turn restriction

The routing entry points are:

- `compute_next_hop()` (`ocn.py:1803`)
- `compute_next_hop_for_packet()` (`ocn.py:1841`)

For worm heads:

- the router computes a next hop;
- it decides whether the flow stays in normal VCs or enters escape;
- if the flit is not a singleton tail, it records the chosen `(next_router, use_escape)` in `flow_output_table`.

For body and tail flits:

- if the flow has an entry in `flow_output_table`, the flit follows it;
- if the entry is missing, the flit was separated from its original worm by a DRAIN split and is promoted into a new singleton worm head/tail so it can route independently.

#### Escape VC entry

Escape entry is decided hop-by-hop for worm heads in `step()`:

- only non-escape heads may newly enter escape;
- if `escape_entry_prob > 0`, the router may choose escape probabilistically;
- or it may force escape when the normal next hop is blocked and the escape VC can accept the flit;
- once a flow enters escape, `flit.in_escape_vc` becomes `True` and the flow remains in escape thereafter.

There is no path from escape back to normal.

#### Ejection and endpoint delivery

When a flit reaches the destination interposer router:

- it competes for the router's `Down` output and is ejected one flit at a time;
- the packet reserves a destination ejection-queue slot before the first flit ejects;
- when all 4 flits are ejected, `_finalize_packet_delivery()` (`ocn.py:2065`) records delivery at the boundary router and chiplet.

In protocol mode, delivery is not the end of the transaction:

- the delivered packet is first placed into a per-class endpoint ejection queue;
- `_service_protocol_ejection_queues()` (`ocn.py:2192`) drains those queues with fixed priority;
- RESP packets are consumed greedily every cycle;
- REQ packets are consumed only if a corresponding RESP can be generated or the code decides to pop the REQ that cycle.

REQ handling is therefore closed-loop:

- external offered traffic creates REQs;
- consumed REQs may generate internal RESP packets through `_try_generate_response_for_request()` (`ocn.py:2151`);
- the transaction completes only when the RESP is consumed at the original requester.

### 2.6 Deadlock detection and how it is actually used

The deadlock detector is `DeadlockDetector` (`ocn.py:2786`).

Implemented detection method:

- build a wait-for graph over routers using `_build_wait_for_graph()` (`ocn.py:2905`);
- add an edge `R_a -> R_b` if `R_a` has a blocked head flit whose next required resource is at `R_b`;
- for adaptive routing, a head is considered truly blocked only if all minimal directions are blocked;
- detect cycles in this functional graph with `_detect_cycles()` (`ocn.py:2995`);
- report a deadlock when a cycle persists for `persist_cycles` consecutive checks. The default is `1`.

Important usage detail:

- the experiment drivers call `detector.check(mesh, cycle)` every cycle;
- but the resulting `DeadlockReport` is not used to trigger DRAIN, change routing, or annotate the saved summary;
- `outputs/ijr_summary.txt` contains no deadlock counts.

So the detector is implemented and exercised, but only as an observer. In this codebase, DRAIN is periodic control, not detector-triggered control.

## 3. Features of the DRAIN Design

### 3.1 What is actually DRAIN-related in the code

The real DRAIN-related mechanisms are:

- one escape VC per input port (`InputPort.escape_vc`);
- periodic drain modes managed by a finite-state machine inside `InterposerMesh`;
- a statically built per-input-port DRAIN turn-table (`DrainTurnBinding`, `ocn.py:364`);
- globally synchronized escape-VC shift stages during drain windows;
- periodic full-drain windows;
- pre-drain cycles that freeze some new admissions before drain starts;
- flit splitting and retagging into singleton sub-worms during synchronized drain hops.

This is more than naming. The DRAIN behavior is in executable control logic, especially:

- `_compute_standard_drain_turn_cycle()` (`ocn.py:1227`)
- `_begin_pre_drain()` (`ocn.py:1443`)
- `_start_drain_window()` (`ocn.py:1464`)
- `_advance_drain_fsm()` (`ocn.py:1487`)
- `_perform_regular_drain_hop()` (`ocn.py:1540`)
- `_perform_drain_ejection_stage()` (`ocn.py:1653`)

### 3.2 Escape VCs

Implemented behavior:

- every router input port has exactly one escape VC in addition to its normal VCs;
- packets are injected only into normal VCs in the `Down` port;
- packets may enter escape later during routing;
- once in escape, they stay in escape.

What escape means in this code depends on the scenario:

- in `random_adaptive_with_drain`, escape VCs are mostly just a separate buffer class during normal cycles, then participate in synchronized DRAIN operations during drain windows;
- in `random_adaptive_turn_restricted`, escape VCs exist, but all routing decisions already use West-First in the actual implementation.

### 3.3 The static drain turn-table

The DRAIN turn-table is not hard-coded by hand. It is generated algorithmically at mesh construction time.

What `_compute_standard_drain_turn_cycle()` does:

- enumerates all 48 directed cardinal inter-router channels in the 4x4 mesh;
- orders outgoing edges at each router in `N, E, S, W` order;
- performs a DFS-based traversal that yields a closed cycle over directed edges;
- converts that edge cycle into a per-input-port binding:
  - source router
  - source input port
  - forced output direction
  - successor router
  - successor input port

From the code alone, the resulting cycle has `48` entries, exactly matching the `48` directed cardinal channels in a 4x4 mesh.

Interpretation:

- the DRAIN control plane is channel-centric, not packet-centric;
- each cardinal input port gets one unique forced output in the drain cycle;
- during a synchronized drain stage, any escape head flit at that input port may be shifted along that precomputed successor relation.

### 3.4 Regular drain windows

Regular drain windows are controlled by:

- `drain_period`
- `drain_window_hops`
- `STRICT_REGULAR_DRAIN`

Implemented behavior:

- every `drain_period` cycles, the mesh enters PRE_DRAIN, then DRAIN;
- while in regular DRAIN mode, each cycle runs one synchronized escape-VC drain hop and one synchronized escape ejection stage;
- `_drain_hops_remaining` counts down from `drain_window_hops`.

Default values in the checked-in code are:

- `DRAIN_PERIOD = 500`
- `DRAIN_WINDOW_HOPS = 1`
- `STRICT_REGULAR_DRAIN = False`

For the `ijr_summary.txt` run, the effective period is not fixed. It is adapted per load point in `sweep_injection_rates()` as:

- `max(1, round(drain_period_c / rate^drain_period_alpha))`
- with `drain_period_c = 0.2`
- and `drain_period_alpha = 2.1`

That is why the summary shows very large periods at low load and much shorter periods at high load.

### 3.5 Pre-drain cycles

`PRE_DRAIN_CYCLES` is implemented, not just named.

Implemented behavior during PRE_DRAIN:

- `self._freeze_escape_admission = True`, so new escape entry is disabled;
- new network injection is blocked by `_injection_blocked_this_cycle()`;
- new normal worm-head routing is also suppressed in `step()`;
- already-routed body/tail flits may still advance if they have a valid `flow_output_table` entry.

So PRE_DRAIN is more than "freeze escape admission". It also suppresses new head-driven route acquisition in the normal VC path.

This is an approximation. The code comments explicitly say PRE_DRAIN is modeling a short settling period because the simulator does not model in-flight links or hardware timing in enough detail to represent a more literal pre-drain phase.

### 3.6 Full drain windows

Full drain windows are implemented by:

- `FULL_DRAIN_EVERY_N_WINDOWS = 20`
- `_full_drain_hops_remaining = len(self._drain_turn_cycle)`

Implemented behavior:

- every 20th drain window becomes a full drain instead of a regular drain;
- a full drain lasts for one full traversal of the 48-entry drain cycle;
- during full drain, injection is blocked and normal forwarding is skipped entirely;
- only synchronized escape circulation and escape ejection happen that cycle.

This is materially stronger than a regular drain window.

### 3.7 Drain period

`drain_period` is the interval between drain-window triggers.

Implemented in code:

- if `cycle > 0 and cycle % self.drain_period == 0`, `_begin_pre_drain(cycle)` is called.

In the active `ijr_summary.txt` run, the effective per-rate periods were:

- rate `0.001`: `399052`
- rate `0.006`: `9266`
- rate `0.011`: `2595`
- rate `0.016`: `1181`
- rate `0.021`: `667`
- rate `0.026`: `426`

So the experiment intentionally drains more frequently as offered load rises.

### 3.8 Drain window hops

`drain_window_hops` is the number of synchronized drain cycles per regular drain window.

Implemented behavior:

- one drain hop means one escape-flit shift opportunity along the static drain cycle, plus one drain-specific ejection stage;
- regular windows count down `_drain_hops_remaining`;
- the default used here is `1`, which makes regular drains very short but frequent at high load.

There is no dedicated sweep in `__main__` over `drain_window_hops`, but the parameter is exposed and used by all DRAIN sweep helpers.

### 3.9 Full-drain frequency

`full_drain_every_n_windows` determines how often a regular drain window is replaced by a full drain window.

Implemented behavior:

- regular windows increment `_drain_window_count`;
- when `_drain_window_count % full_drain_every_n_windows == 0`, the code starts `FULL_DRAIN` instead of `DRAIN`.

There is a dedicated `sweep_full_drain_window()` helper (`ocn.py:3640`) for this parameter, although it is commented out in `__main__`.

### 3.10 Strict regular drain vs escape-first drain

This distinction is implemented through the global `STRICT_REGULAR_DRAIN`.

When `STRICT_REGULAR_DRAIN = False`:

- regular drain happens first;
- then normal forwarding continues in the same cycle;
- escape flits do not also participate in normal forwarding that cycle.

When `STRICT_REGULAR_DRAIN = True`:

- the cycle performs only the regular drain hop and drain ejection stage;
- normal forwarding is skipped for that cycle.

In the checked-in run, the flag is `False`, so the active behavior is escape-first regular drain, not escape-only regular drain.

### 3.11 Synchronized vs staggered behavior

What is truly implemented:

- a globally synchronized drain shift over escape VCs.

What is not truly implemented:

- per-router staggered drain phasing.

There is a global variable named `stagger_drain`, but it is only used in `sweep_injection_rates()` as a gate on whether the sweep uses the inverse-power-law `drain_period` formula. It does not create a staggered router schedule.

### 3.12 What makes this design "composable"

Supported by implementation:

- chiplet composition through arbitrary `ChipletSpec` lists;
- shared interposer routers across chiplets;
- protocol composition through source/ejection/outstanding constraints;
- drain bindings that operate on router ports rather than chiplet identities.

Not supported by implementation:

- automatic DRAIN synthesis for arbitrary topologies;
- DRAIN over faulted topologies;
- arbitrary mesh sizes;
- chiplet-specific drain policies.

So "composable" is supported in the endpoint-to-interposer composition sense, but only partially in the topology-generality sense.

## 4. Actual Methodology

### 4.1 Experimental system under test

The run that generated `outputs/ijr_summary.txt` comes from the `__main__` block near the end of `ocn.py`.

Actual run configuration:

| Parameter | Actual value used by the active `__main__` path |
| --- | --- |
| Layout | 4 corner GPUs + 1 central CPU via `build_image_layout()` |
| Injection-rate sweep | `0.001` to `0.026` in steps of `0.005` |
| Seeds averaged | `4` |
| Injection cycles per run | `20000` |
| Cooldown cycles per run | `10000` |
| Traffic mode | `NEW_PROTOCOL_REQ_RESP` |
| Base traffic seed | `42` |
| Base routing seed | `0` |
| DRAIN period law | `max(1, round(0.2 / rate^2.1))` |
| `drain_window_hops` | `1` |
| `pre_drain_cycles` | `1` |
| `full_drain_every_n_windows` | `20` |

One important caveat: the print statement in `__main__` still says "Sweeping INJECTION_RATE from 0.05 to 0.50 (step 0.025, averaged over 5 seeds)", but that is stale. The actual function call immediately below it uses the smaller `0.001` to `0.026` range and `num_seeds=4`, and `ijr_summary.txt` clearly reflects the actual call.

### 4.2 Traffic generation model

There are two endpoint traffic behaviors:

- `CPUChiplet.current_injection_rate()` returns a fixed Bernoulli probability each cycle.
- `GPUChiplet.current_injection_rate()` alternates between burst and quiet phases.

Default active rates are defined relative to the swept CPU rate:

- CPU: `rate`
- GPU burst: `4 * rate`
- GPU quiet: `rate / 10`
- GPU burst duration: `20` cycles
- GPU quiet duration: `80` cycles

The GPU chiplets start in the quiet phase so they are not all aligned in burst mode at time zero.

One subtle but important code-backed point: the comment says the default GPU parameters give roughly 4x higher average throughput than a CPU chiplet. That is not what the arithmetic implies for a single GPU chiplet. The long-run average GPU injection probability is:

- `(20 * 4r + 80 * 0.1r) / 100 = 0.88r`

So the current defaults make each individual GPU slightly lower than the CPU on average, not 4x higher. The aggregate GPU traffic is still much larger because there are four GPUs.

### 4.3 Protocol mode and message classes

The active workload is protocol mode, not legacy traffic:

- `PROTOCOL_MODE_ENABLED = True`
- source class: `REQ`
- sink class: `RESP`
- additional defined but unused classes: `DATA`, `CTRL`

Protocol handling in the current experiments is:

- external traffic offers only REQs;
- a consumed REQ may generate a RESP back to the original requester;
- `DATA` and `CTRL` exist structurally but remain zero in `ijr_summary.txt`.

The summary file confirms this:

- every scenario row shows nonzero `REQ` and `RESP`;
- every row shows `DATA` and `CTRL` attempted, injected, delivered all equal to zero.

### 4.4 Transaction model

Transactions are explicit:

- each new REQ creates a `TransactionRecord`;
- the RESP keeps the original `transaction_id` and `request_packet_id`;
- transaction latency is measured from REQ creation to RESP consumption at the original requester.

This is why the summary reports both packet statistics and transaction-completion statistics.

The simulation is therefore evaluating two related but distinct forms of progress:

- packet transport progress
- end-to-end protocol transaction progress

### 4.5 Outstanding limits and endpoint queues

The endpoint-side protocol model is deliberately packet-level and coarse:

- injection queue capacity per class: `32`
- ejection queue capacity per class: `32`
- outstanding limit per class:
  - `REQ = 16`
  - `RESP = 16`
  - `DATA = 8`
  - `CTRL = 8`

REQ generation is blocked if the source chiplet is at either:

- injection-queue capacity, or
- outstanding limit for that message class.

Packet ejection into a destination chiplet is also gated by ejection-queue capacity through packet-level reservation.

This models endpoint backpressure and protocol dependence, but not at hardware-accurate credit granularity.

### 4.6 Injection and ejection service policy

Protocol mode adds specific queue service rules:

- one queued source packet per chiplet may enter the network per cycle;
- ejection queues are serviced in `RESP, REQ, DATA, CTRL` order;
- `RESP` is drained greedily every cycle before any other class;
- for other classes, the code consumes at most one packet per chiplet per cycle.

That policy is important. It biases the endpoint model toward keeping responses drainable so transactions can complete.

### 4.7 Randomness and seeded control

The experiments are reproducible:

- traffic uses seeded `random.Random(...)` instances;
- routing decisions use a separate seeded `random.Random(...)` instance inside each `InterposerMesh`;
- multi-seed sweeps offset both seeds by `1000` per repetition.

This is good methodology for two reasons:

- it stabilizes the reported curves by averaging over several runs;
- it decouples traffic randomness from routing randomness.

### 4.8 Warmup, active, and cooldown phases

Implemented run phases:

- no warmup phase
- active injection phase
- cooldown phase with no new external traffic

For the `ijr_summary.txt` run:

- active: `20000` cycles
- cooldown: `10000` cycles

The cooldown phase still allows:

- source queues to keep injecting already-generated packets;
- in-flight packets to keep moving;
- endpoint ejection queues to keep draining;
- responses to keep being generated from already-delivered requests.

### 4.9 What comparisons are actually being made

The main comparison is across six scenarios inside `run_simulation()` (`ocn.py:3101`):

- `xy`
- `yx`
- `random_adaptive_without_drain`
- `random_adaptive_with_drain`
- `random_adaptive_turn_restricted`
- `shortest_path`

This is not only a routing comparison. It is also a resource-budget comparison, because the VC count changes across scenarios.

## 5. Overall File Structure, Code Organization, and Algorithms

### 5.1 File map

| Code object | Location | Role |
| --- | --- | --- |
| `ChipletSpec` | `ocn.py:149` | Declares chiplet type, BR count, and BR-to-IR attachment points |
| `ProtocolConfig` | `ocn.py:178` | Endpoint queue capacities, outstanding limits, service order, protocol mode |
| `TransactionRecord` | `ocn.py:212` | Tracks REQ->RESP transactions |
| `build_image_layout()` | `ocn.py:239` | Builds the active five-chiplet image-processing style layout |
| `Flit` | `ocn.py:275` | Flit metadata including flow identity and escape membership |
| `Packet` | `ocn.py:313` | Packet object, broken into 4 flits |
| `DrainTurnBinding` | `ocn.py:364` | One static DRAIN binding from input-port escape VC to successor channel |
| `VC` | `ocn.py:385` | Wormhole VC with allocation semantics |
| `InputPort` | `ocn.py:488` | Per-direction port with normal VCs plus one escape VC |
| `BoundaryRouter` | `ocn.py:554` | Endpoint attachment/delivery bookkeeping object |
| `InterposerRouter` | `ocn.py:618` | 5-port router with input-buffered flit forwarding |
| `Chiplet`, `CPUChiplet`, `GPUChiplet` | `ocn.py:705`, `868`, `890` | Endpoint traffic behavior and protocol queues |
| `InterposerMesh` | `ocn.py:942` | Topology construction, injection, routing, DRAIN, stepping, statistics |
| `DeadlockReport`, `DeadlockDetector` | `ocn.py:2763`, `2786` | Wait-for-graph deadlock observation |
| `run_simulation()` | `ocn.py:3101` | Per-scenario experiment driver |
| `sweep_injection_rates()` | `ocn.py:3295` | Main rate sweep used for `ijr_summary.txt` |
| Summary/plot writers | `ocn.py:3588` onward | Save figures and text summaries |

### 5.2 Main algorithms

| Algorithm or mechanism | Primary function(s) | Actual behavior |
| --- | --- | --- |
| XY routing | `xy_next_hop()` | Deterministic minimal path, X then Y |
| YX routing | `yx_next_hop()` | Deterministic minimal path, Y then X |
| Random adaptive routing | `compute_next_hop()` + `_minimal_directions()` | Chooses randomly among minimal directions |
| Turn-restricted routing | `_turn_restricted_directions()` | West-First minimal routing |
| "Shortest path" baseline | `run_simulation()` scenario table | Same random adaptive minimal routing, but with 40 normal VCs |
| Escape VC admission | `step()` | Probabilistic or forced-on-blocked transition from normal to escape |
| Static DRAIN cycle synthesis | `_compute_standard_drain_turn_cycle()` | Builds 48-entry per-port cycle over directed mesh channels |
| Synchronized drain hop | `_perform_regular_drain_hop()` | Moves one escape head per eligible bound input port, with split-and-retag semantics |
| Drain ejection | `_perform_drain_ejection_stage()` | Allows one escape flit per router to eject during drain stages |
| Deadlock detection | `_build_wait_for_graph()`, `_detect_cycles()`, `check()` | Router-level wait-for-graph cycle detection |

### 5.3 A key algorithmic subtlety: DRAIN splits worms

The most consequential algorithmic detail in the whole file is inside `_perform_regular_drain_hop()`.

When a synchronized DRAIN hop moves one escape flit out of a VC:

- the moved flit is turned into a singleton worm with a fresh `flow_id`;
- any remaining sibling flits still in the source escape VC are retagged with a new source-side `flow_id`;
- the first and last remaining flits are re-marked as head and tail as needed.

This means DRAIN does not preserve packet contiguity. It explicitly fragments a packet into smaller independently routed worms.

That design choice is central to the observed latency behavior.

## 6. What Exactly Is Being Evaluated

### 6.1 Scenario comparison table

| Scenario in summary | Routing policy | DRAIN enabled | Normal VCs per port | Escape VC usage | What it represents | Important caveat |
| --- | --- | --- | --- | --- | --- | --- |
| `XY (deadlock-free)` | `ROUTING_XY` | No | 2 | Present structurally, unused (`escape_prob=0`) | Deterministic deadlock-free baseline | Uses 2 normal VCs, not 1 |
| `YX (deadlock-free)` | `ROUTING_YX` | No | 2 | Present structurally, unused | Second deterministic deadlock-free baseline | Same resource budget as XY |
| `Random Adaptive (no DRAIN)` | `ROUTING_RANDOM_ADAPTIVE` | No | 2 | Present structurally, unused | Fully adaptive minimal baseline without deadlock protection | Not deadlock-free by construction |
| `Random Adaptive + DRAIN` | `ROUTING_RANDOM_ADAPTIVE` | Yes | 1 | Active; flows may enter escape probabilistically and then participate in periodic drain scheduling | Main proposed DRAIN-style design | Escape routing is not separately restricted outside drain windows |
| `Random Adaptive + Turn Restrictions` | `ROUTING_RANDOM_ADAPTIVE_TR` | No | 1 | Present and may still be used probabilistically | Deadlock-free turn-restricted baseline | Actual code routes all packets West-First, not just escape packets |
| `Shortest Path (40 VCs)` | `ROUTING_RANDOM_ADAPTIVE` | No | 40 | Present structurally, unused | High-VC abundance baseline approximating minimal routing with little VC contention | Not a distinct routing algorithm despite the label |

### 6.2 Metrics table

| Metric | Where computed | Meaning |
| --- | --- | --- |
| `attempted_injections` | `InterposerMesh` counters | Packets offered by the source/protocol logic before successful network entry |
| `injected` | `delivered_stats()` | Packets that actually entered the network |
| `failed_injections` | `delivered_stats()` | Packets that were offered but rejected before entering the network |
| `acceptance_rate` | `delivered_stats()` | `injected / attempted_injections` |
| `delivered` | `_packet_stats()` | Delivered packets only; excludes dropped and still-undelivered packets |
| `avg_latency` | `_packet_stats()` | Average `delivered_cycle - created_cycle` over delivered packets only |
| `avg_hops` | `_packet_stats()` | Average interposer hops over delivered packets |
| `throughput` | `run_simulation()` | Delivered packets normalized by run length and chiplet count |
| `transactions started` | `transaction_stats()` | REQ transactions created |
| `transactions completed` | `transaction_stats()` | REQ->RESP exchanges completed at the requester |
| `completion_rate` | `transaction_stats()` | `completed / started` |
| `avg_completion_latency` | `transaction_stats()` | Average cycles from REQ creation to RESP completion |
| `in_flight_flits` / `in_flight_packets_equiv` | `delivered_stats()` | Residual backlog after cooldown; computed but not printed in `ijr_summary.txt` |

Two interpretation cautions matter:

- `avg_latency` only covers delivered packets, so a scenario can look latency-stable while silently dropping or failing to inject many packets.
- packet throughput and transaction completion are different. A scenario can deliver many packets but still have poor end-to-end completion if REQ/RESP coupling suffers.

### 6.3 Sweeps implemented in the file

Implemented experiment surfaces in `ocn.py`:

- injection-rate sweep: `sweep_injection_rates()`
- drain-period sweep: `sweep_drain_window()`
- full-drain-frequency sweep: `sweep_full_drain_window()`
- escape-probability sweep: `sweep_escape_prob()`
- fault-count sweep: `sweep_fault_count()`
- per-message-class analysis: `plot_message_class_sweep()`
- transaction completion analysis: `plot_transaction_completion_sweep_all_scenarios()`
- REQ/RESP latency analysis: `plot_req_resp_latency_sweep_all_scenarios()`

Notably absent as a dedicated sweep:

- no standalone sweep helper varies `drain_window_hops`;
- no helper varies `STRICT_REGULAR_DRAIN`;
- no helper varies `pre_drain_cycles` directly, though both parameters are exposed.

### 6.4 Fault evaluation status

The file includes a fault-count sweep, but there is a major implementation limit:

- `inject_faults()` explicitly raises `NotImplementedError` if `drain_enabled=True` and `num_faults > 0`.

So fault sweeps are only meaningfully supported for non-DRAIN or zero-fault DRAIN cases in the current code. The evaluation surface exists, but "DRAIN under faults" is not actually implemented.

## 7. Code-backed Findings

This section is limited to what is directly supported by the generated `ijr_summary.txt` plus hard facts from `ocn.py`.

### 7.1 The high-VC baseline is the throughput ceiling in this study

`Shortest Path (40 VCs)` is the dominant scenario in delivered packets and acceptance across all nontrivial loads:

- at rate `0.011`: `1933` delivered, `0` failed injections, `avg_latency = 6.60`
- at rate `0.016`: `2972` delivered, `0` failed, `completion_rate = 0.9945`
- at rate `0.026`: `4579` delivered, `0` failed, `completion_rate = 0.9937`

This tells you that, in this simulator, abundant VC provisioning overwhelms the bottlenecks that dominate the constrained scenarios.

### 7.2 Among the constrained designs, DRAIN is the highest-throughput choice at high load

Starting around the mid-to-high load regime, `Random Adaptive + DRAIN` clearly delivers the most packets among the leaner designs:

- at rate `0.016`:
  - DRAIN: `2883` delivered
  - XY: `1663`
  - YX: `1358`
  - no DRAIN adaptive: `865`
  - turn-restricted: `1630`
- at rate `0.021`:
  - DRAIN: `2790`
  - XY: `1250`
  - YX: `1337`
  - no DRAIN adaptive: `956`
  - turn-restricted: `1297`
- at rate `0.026`:
  - DRAIN: `3229`
  - XY: `1052`
  - YX: `1015`
  - no DRAIN adaptive: `718`
  - turn-restricted: `883`

The same pattern appears in transaction completion:

- at `0.016`, DRAIN completes `1439` transactions with `0.9906` completion rate
- XY completes `788` with `0.8141`
- YX completes `628` with `0.7737`
- no DRAIN adaptive completes `408` with `0.7134`
- turn-restricted completes `775` with `0.8010`

### 7.3 DRAIN pays a severe latency penalty, especially in the middle of the sweep

`Random Adaptive + DRAIN` is not a free win. It has the worst latency by a large margin at several load points:

- rate `0.006`: `avg_latency = 10.91`, while XY/YX/TR are around `6.27` to `6.32`
- rate `0.011`: `avg_latency = 45.15`, while XY/YX/TR are around `6.37` to `6.50`
- rate `0.016`: `avg_latency = 22.23`, still far above the other constrained scenarios

Transaction latency shows the same effect:

- DRAIN `avg_completion_latency = 55.01` at `0.011`
- the others at that point are roughly `12.75` to `13.20`

So the summary supports a strong throughput/completion advantage for DRAIN at high load, but also a strong latency cost.

### 7.4 Random adaptive without DRAIN is consistently poor once load rises

The unconstrained adaptive baseline degrades fastest among the constrained designs:

- at `0.011`: only `905` delivered, `455` failed injections
- at `0.016`: `865` delivered, `905` failed
- at `0.026`: `718` delivered, `1868` failed

Its completion rates are also weak:

- `0.7741` at `0.011`
- `0.7134` at `0.016`
- `0.7309` at `0.026`

In this simulator, adaptivity by itself does not rescue a low-VC design.

### 7.5 Turn-restricted routing is a competent safe baseline, but it does not match DRAIN under heavy load

`Random Adaptive + Turn Restrictions` usually beats the no-DRAIN adaptive case and is competitive with deterministic routing at moderate load:

- at `0.011`: `1667` delivered, close to YX's `1683`
- at `0.016`: `1630` delivered, close to XY's `1663`

But at higher load it remains far below DRAIN:

- at `0.026`: TR delivers `883` vs DRAIN's `3229`

Its completion rate also collapses more than DRAIN at the top of the sweep:

- TR at `0.026`: `0.6553`
- DRAIN at `0.026`: `0.9215`

### 7.6 XY and YX remain low-latency but saturate early

The deterministic baselines have the lowest average packet latency among the constrained designs almost everywhere:

- XY stays near `6.09` to `6.87`
- YX stays near `6.10` to `7.06`

But they give up throughput and completion much earlier than DRAIN:

- XY `completion_rate` falls to `0.7198` by `0.026`
- YX falls to `0.7117`

YX is slightly better than XY at `0.011`, but the advantage is not uniform across the sweep.

### 7.7 CPU traffic is consistently lower-latency than GPU traffic, and DRAIN amplifies that gap

From the per-type statistics in the summary:

- under XY at `0.011`, CPU `avg_latency = 5.77`, GPU `6.55`
- under DRAIN at `0.011`, CPU `6.27`, GPU `54.36`
- under DRAIN at `0.016`, CPU `15.29`, GPU `24.11`

This is one of the strongest asymmetries in the output file. Whatever is hurting DRAIN latency is disproportionately affecting GPU-originated traffic.

### 7.8 REQ and RESP are both visible, and RESP is usually slightly slower

Across most non-DRAIN scenarios:

- REQ latency is a bit lower than RESP latency
- example at `0.026` under XY:
  - REQ `6.51`
  - RESP `7.30`
- example at `0.026` under shortest-path:
  - REQ `6.97`
  - RESP `7.75`

Under DRAIN at high load, REQ and RESP latencies become similarly inflated:

- at `0.026`:
  - REQ `10.52`
  - RESP `10.34`

### 7.9 The summary does not provide deadlock evidence, even though the detector ran

Hard code fact:

- `DeadlockDetector.check()` is called every cycle in the main sweeps.

Hard output fact:

- `outputs/ijr_summary.txt` does not report deadlock count, deadlock cycles, or deadlock history.

So there is no code-backed claim available from the summary about how often each scenario deadlocked in the observed runs. The code supports such observation internally, but the chosen output format does not expose it.

## 8. Likely Interpretation

This section combines the code structure with the observed results to explain what the project is likely trying to show.

### 8.1 Likely research claim

The likely claim is not that DRAIN minimizes latency. It is that DRAIN allows a low-VC, adaptive interposer design to preserve throughput and end-to-end transaction completion under load much better than:

- plain deterministic deadlock-free routing,
- plain adaptive routing without recovery,
- or a turn-restricted deadlock-free baseline.

The summary supports that claim in the throughput and completion dimensions.

### 8.2 Why DRAIN likely helps throughput

The code gives DRAIN three advantages that the simpler baselines do not share:

- adaptive minimal routing outside drain windows;
- an extra escape VC class beyond the single normal VC;
- periodic synchronized escape circulation and ejection that can break up stuck escape occupancy patterns.

That combination likely lets the network keep admitting and eventually delivering traffic when the deterministic and turn-restricted baselines are already strongly injection-limited.

### 8.3 Why DRAIN likely hurts latency so much

The latency penalty is probably not coming from a single cause. The code suggests a stack of latency-inflating effects:

- PRE_DRAIN suppresses new injection and new head-driven forwarding.
- FULL_DRAIN completely suppresses normal forwarding.
- Regular drain windows preempt escape VCs from normal forwarding for that cycle.
- Most importantly, `_perform_regular_drain_hop()` fragments packets into singleton sub-worms by retagging the moved flit and the remaining siblings separately.

That last mechanism can significantly increase packet completion time because:

- a packet's flits can be separated in space and time;
- each singleton then competes independently for forwarding and ejection;
- packet completion waits for all 4 flits to arrive.

That interpretation is strongly consistent with the summary's pattern:

- DRAIN has near-best injection acceptance at high load,
- but its packet and transaction latency can become enormous at intermediate drain periods.

### 8.4 Why the worst DRAIN latency appears in the middle of the sweep

The dynamic drain-period formula is the likely reason.

At low load:

- the period is so large that DRAIN almost never activates, so latency stays near the non-DRAIN baselines.

At very high load:

- the period becomes short enough that drain activity is frequent and the system seems to maintain strong acceptance and better eventual completion.

In the middle:

- DRAIN is active often enough to fragment traffic and pause normal activity,
- but perhaps not often enough to keep escape-state buildup tightly controlled.

That is the most plausible explanation for the striking `0.011` and `0.016` latency spikes.

### 8.5 DRAIN vs deterministic routing

Likely interpretation:

- deterministic routing gives short, stable paths and low latency, but its constrained-path nature saturates quickly;
- DRAIN keeps a much larger fraction of the offered load alive at high rates, but pays for that with much longer time-in-system per packet.

So DRAIN looks like a throughput-oriented mechanism, not a latency-oriented one.

### 8.6 DRAIN vs random adaptive without DRAIN

Likely interpretation:

- unrestricted minimal adaptivity with very few VCs is too fragile in this model;
- it does not maintain injection acceptance under load;
- the DRAIN overlay is what turns that adaptive routing from unstable to useful.

This is one of the clearest messages in the results.

### 8.7 DRAIN vs turn-restricted routing

Likely interpretation:

- turn restriction is simpler, safer, and much easier on latency;
- but it gives up routing freedom and therefore leaves throughput on the table at high load;
- DRAIN preserves more of adaptive routing's load-balancing benefit, which is why it wins so strongly in delivered packets and completion rate once the system gets busy.

### 8.8 DRAIN vs the many-VC baseline

Likely interpretation:

- if abundant VC provisioning is allowed, this simulator says that the resource-rich baseline wins outright;
- DRAIN is interesting because it tries to recover similar load tolerance without paying the 40-VC cost.

So the real value proposition of the DRAIN design here is not "better than 40 VCs". It is "much better than low-VC deadlock-safe or low-VC uncontrolled adaptive designs."

## 9. Strengths, Limitations, and Assumptions

### 9.1 What the simulator models well

- explicit flit/packet/VC wormhole interactions
- per-input-port buffering and arbitration
- adaptive vs deterministic routing tradeoffs
- chiplet composition over a shared interposer
- endpoint protocol backpressure and REQ->RESP dependency loops
- a real, executable periodic DRAIN scheduler
- reproducible multi-seed sweeps and several analysis surfaces

### 9.2 Important simplifications

- no intra-chiplet network is modeled
- links are not modeled as explicit in-flight pipelines
- boundary routers have no outbound buffering in the current implementation
- packet injection is atomic into an interposer `Down` VC if enough room exists
- endpoint flow control is packet-level queue reservation, not flit-credit flow control
- all packets are fixed-size 4-flit packets
- only REQ external traffic is generated in the active experiments
- `DATA` and `CTRL` classes are defined but unused in this workload

### 9.3 Important implementation limits

- DRAIN only supports the standard fault-free 4x4 mesh
- DRAIN-enabled fault injection is explicitly unimplemented
- the deadlock detector is not coupled back into control
- the summary output omits deadlock statistics even though the detector runs
- no warmup phase is used before measurement
- the "shortest_path" scenario is really a many-VC adaptive baseline, not a separate routing algorithm

### 9.4 Places where naming or comments overstate what is implemented

- The module header still describes an older boundary-router pipeline model.
- The `__main__` print string describes a different sweep range and seed count than the actual function call.
- The comment for `ROUTING_RANDOM_ADAPTIVE_TR` suggests a hybrid where normal VCs are fully adaptive and only escape VC uses turn restriction.
- The actual `compute_next_hop_for_packet()` implementation instead routes all packets West-First in that mode.

This is why the code has to be treated as authoritative over comments.

### 9.5 What could affect interpretation of the results

- Because `avg_latency` only covers delivered packets, scenarios with large injection failure can still appear latency-competitive.
- Because the workload is closed-loop REQ->RESP, network load is shaped by both source injection and protocol completion behavior.
- Because DRAIN periodically fragments worms into singleton sub-worms, some of the observed latency cost may be an artifact of this specific approximation rather than an intrinsic property of all DRAIN-like schemes.
- Because full drain is fixed to the 48-step drain-cycle length, the control cost is topology-sized rather than queue-sized.

## 10. Final Synthesis

The core design idea in this project is to evaluate whether a chiplet interposer can keep the VC budget small, preserve adaptive minimal routing, and still remain robust under load by adding an escape-VC plus periodic DRAIN-style control layer. The implementation does this with a 4x4 interposer mesh, explicit flits and VCs, a packet-level REQ->RESP protocol model, and a static 48-entry per-input-port drain turn-table that is exercised through regular and full drain windows.

Methodologically, the simulator is not just comparing routing algorithms. It is comparing routing-plus-resource-control regimes: deterministic deadlock-free routing, unrestricted adaptive routing, turn-restricted deadlock-free routing, a DRAIN-enabled low-VC adaptive design, and an abundant-VC baseline. The active evaluation uses seeded multi-run injection-rate sweeps with cooldown, reports per-type and per-message-class statistics, and tracks end-to-end transaction completion.

The likely main conclusion is sharp: in this model, the DRAIN design is not the lowest-latency option, but it is the strongest low-VC/high-load design. It dramatically improves delivered traffic and transaction completion relative to deterministic, unrestricted adaptive, and turn-restricted low-VC baselines once the offered load rises. The price is complexity and, in the current implementation, sometimes very large latency inflation. The many-VC baseline still wins outright when cost is ignored, so the real argument for this project is that DRAIN can recover much of that robustness without resorting to extreme VC provisioning.
