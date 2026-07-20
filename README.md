# Composable DRAIN for Chiplet-based Systems

**Authors:** Yash Gangavarapu, Danilo Rincon Camacho

An interposer-level extension of [DRAIN](https://doi.org/10.1109/hpca47549.2020.00044) that removes inter-chiplet deadlocks in active-interposer chiplet systems, implemented as a Python-based NoC simulator.

---

## Overview

Modern chiplet-based systems interconnect independently designed chiplets through an active silicon interposer. Even when each chiplet is internally deadlock-free, cyclic resource dependencies can still emerge **across chiplet boundaries**, leading to inter-chip network deadlocks. Existing chiplet deadlock-avoidance techniques (e.g. turn restrictions at boundary routers) typically trade away routing flexibility or require expensive extra buffering to stay correct.

**Composable DRAIN** takes a different approach. Rather than proactively restricting routing to prevent cycles, it lets deadlocks happen and periodically drains packets along a precomputed cyclic escape path at the interposer level — a subactive recovery mechanism, extending the original [DRAIN](https://doi.org/10.1109/hpca47549.2020.00044) framework from single-chip NoCs to multi-chiplet systems.

This keeps each chiplet's internal NoC fully independent (no global knowledge required) while guaranteeing global forward progress at the point where chiplets actually meet: the active interposer.

## Key Idea

- Assume each chiplet's internal NoC is already deadlock-free.
- Add an escape virtual channel (E-VC) at each interposer router.
- Compute an offline drain path spanning the interposer mesh.
- During periodic drain windows, force escape-domain packets to advance one hop along this path, breaking any cyclic wait regardless of where it formed.
- Because the drain path is computed ahead of time and spans all interposer links, **no runtime global deadlock detection is needed**.

## Architecture

The simulated baseline is an accelerated processing unit (APU) composed of:

- **4 GPU chiplets** — one per quadrant of a 4×4 active interposer mesh, 4 boundary routers each
- **1 CPU chiplet** — 2 boundary routers, attached to the two upper-interior interposer routers (shared with the top GPU quadrants)

Each interposer router has 5 input/output ports (`NORTH`, `SOUTH`, `EAST`, `WEST`, `DOWN`), where `DOWN` connects to a chiplet's boundary router. Each input port holds a configurable number of normal VCs plus exactly one escape VC (8-flit capacity each), with wormhole flow control (4 flits/packet) and a max of 1 flit per output port per cycle.

## Features

- **Wormhole flow control** with explicit header/body/tail flit modeling, including correct handling of **flit separation during DRAIN** (upstream flits promoted into newly routable heads when a drain event moves them off the original downstream path).
- **Multiple routing modes** for comparison:
  - Deterministic XY / YX (dimension-ordered, deadlock-free by construction)
  - Fully random adaptive (best path diversity, *not* deadlock-free under load)
  - Turn-restricted adaptive (West-First, deadlock-free by construction, reduced path diversity)
  - **Composable DRAIN** (adaptive routing + escape VC + periodic drain scheduling)
  - Idealized "Shortest Path" baseline (40 VCs, impractically large, deadlock-free upper bound)
- **Protocol-aware traffic model**: external traffic is generated as `REQ` packets; delivered `REQ`s may generate `RESP` packets back to the requester. Transactions are tracked end-to-end (creation → completion), not just packet delivery.
- **Realistic traffic generation**: uniform Bernoulli injection for CPU traffic; bursty burst/quiet-phase injection for GPU traffic.
- **Configurable DRAIN scheduler**: drain period, drain window hop count, pre-drain length, escape-entry probability, and full-drain interval are all tunable.
- **Fault injection**: random bidirectional interposer link faults, with the drain path automatically recomputed over the surviving topology.

## Research Questions

1. Can extending DRAIN provide composable global deadlock freedom across chiplet-based systems?
2. What is the performance impact of interconnect-level DRAIN on latency, throughput, and tail behavior compared to modular routing and other chiplet deadlock-avoidance methods?
3. Do local DRAIN regions with different drain windows (tuned per chiplet type — CPU vs. GPU) perform better than a single global DRAIN?

## Results Summary

- **Deadlock removal:** Without DRAIN, random adaptive routing saturates around 1400 packets injected, with a growing share of failed/undelivered packets once injection rate exceeds ~0.005 packets/cycle/chiplet — clear evidence of deadlock. With DRAIN, injected packets equal delivered packets, and delivery scales linearly with injection rate.
- **Throughput under load:** DRAIN saturates around 0.07 packets/cycle/chiplet, well beyond YX (~0.02) and Turn Restrictions (~0.03), because it preserves full adaptive routing flexibility in the common case and only pays a recovery cost during scheduled drain windows.
- **Transaction completion:** At an injection rate of 0.016 packets/cycle/chiplet, DRAIN completed 1439 transactions at a 99.06% completion rate, versus 81.41% (XY), 77.37% (YX), 80.10% (turn-restricted), and 71.34% (adaptive, no DRAIN).
- **Fault tolerance:** A single faulty interposer link degrades Turn Restrictions' delivered packets by ~80%. DRAIN degrades far more gracefully, not reaching an 80% reduction until around 8 faulty links.
- **Cost:** These gains come with a real latency penalty — periodic forced movement, escape-domain transitions, and worm fragmentation increase packet residence time even as throughput and completion improve. DRAIN is best understood as a **throughput- and completion-oriented** deadlock recovery mechanism, not a latency-minimizing one.

## Assumptions & Scope

- Each chiplet's internal NoC is assumed to be internally deadlock-free (only inter-chiplet deadlock is addressed).
- Boundary routers are exposed to at least one escape VC; escape-domain packets never re-enter normal VCs.
- The full interposer topology is known, so the drain path can be computed offline with negligible runtime cost.
- All drain windows are synchronized across boundary routers.
- Current evaluation covers a single **global** interposer-level DRAIN; regional/hierarchical DRAIN (different drain windows per chiplet type) is proposed but not yet implemented.
- The protocol model uses a simplified REQ/RESP message-class pair rather than a full coherence protocol.

## Limitations & Future Work

- Implement full regional/hierarchical DRAIN with per-region drain windows tuned to CPU vs. GPU traffic characteristics.
- Extend the protocol-aware model toward a more complete coherence protocol with richer message-class dependencies.
- Develop a more principled, load-adaptive drain-period scheduling policy rather than the current fixed inverse-square-law heuristic.

*This project was developed as a research exploration into subactive, composable deadlock removal for chiplet-based systems.*
