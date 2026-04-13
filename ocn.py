#Interpreter Path: /Users/danilorincon04/venvs/mnist/bin/python
#Activate virtual env source ~/venvs/mnist/bin/activate
"""
chiplet_interchiplet_sim.py
---------------------------
Foundation classes for simulating inter-chiplet packet communication on a
4×4 interposer mesh.

Topology
--------
  - 16 interposer nodes arranged in a 4×4 grid.
  - Each interposer node has one InterposerRouter.
  - Each InterposerRouter connects to exactly one CPU Chiplet through one
    BoundaryRouter (the chiplet's single boundary router).
  - Bidirectional links connect all cardinal (N/S/E/W) interposer neighbors.

Traffic model
-------------
  - Chiplets only *generate* and *receive* inter-chiplet packets.
  - No intra-chiplet routing is modelled.
  - Routing through the interposer uses XY (dimension-ordered) routing:
    traverse the column (X) dimension first, then the row (Y) dimension.

Simulation cycle
----------------
  Phase 1  Boundary-router outbound VCs → attached interposer router VCs.
  Phase 2  Each interposer router advances its head packet one hop.
           Packets that reach their destination interposer router are ejected
           to the destination boundary router and delivered to the chiplet.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─── Global defaults ──────────────────────────────────────────────────────────

MESH_ROWS          = 4
MESH_COLS          = 4
NUM_NORMAL_VCS     = 1   # normal VCs per router port; escape VC is always 1 extra
FLITS_PER_PACKET   = 4   # every packet is decomposed into this many flits
VC_FLIT_CAPACITY   = 8   # every VC holds up to this many flits
DRAIN_ESCAPE_ENTRY_PROB           = 0.1
TURN_RESTRICTED_ESCAPE_ENTRY_PROB = 0.1
ESCAPE_ENTRY_PROB  = DRAIN_ESCAPE_ENTRY_PROB  # default for single-mesh runs
DRAIN_PERIOD       =  500
DRAIN_WINDOW_HOPS  = 1
PRE_DRAIN_CYCLES   = 1 # FLITS_PER_PACKET # paper says -  maximum packet size supported in the networ
FULL_DRAIN_EVERY_N_WINDOWS = 20   # perform a full drain once every N regular drain windows
STRICT_REGULAR_DRAIN = False      # False = escape-first priority drain, True = escape-only regular drain
INJECTION_RATE     = .025#0.006

DRAIN_DEBUG_PRINTS = True
DRAIN_DEBUG_PRINT_PERIOD = 200

# ─── Chiplet type identifiers ─────────────────────────────────────────────────

CHIPLET_CPU = "CPU"
CHIPLET_GPU = "GPU"

# CPU chiplet defaults
CPU_VC_CAPACITY      = VC_FLIT_CAPACITY   # capacity in flits (uniform across all VCs)
CPU_INJECTION_RATE   = INJECTION_RATE     # uniform Bernoulli per cycle

# GPU chiplet defaults
GPU_VC_CAPACITY      = VC_FLIT_CAPACITY   # same flit capacity for all VCs
GPU_BURST_RATE       = 4 * CPU_INJECTION_RATE            # ~8× CPU rate during a burst window
GPU_QUIET_RATE       = CPU_INJECTION_RATE/10            # near-idle between bursts
GPU_BURST_CYCLES     = 20              # cycles per burst
GPU_QUIET_CYCLES     = 80              # quiet cycles between bursts

# ─── Routing algorithm identifiers ────────────────────────────────────────────

ROUTING_XY               = "XY"
# Traverse columns (X) first, then rows (Y).  Deadlock-free.

ROUTING_YX               = "YX"
# Traverse rows (Y) first, then columns (X).  Deadlock-free.

ROUTING_RANDOM_ADAPTIVE  = "RANDOM_ADAPTIVE"
# At each hop randomly choose among ALL directions that reduce Manhattan
# distance to the destination (fully adaptive, minimal paths only).
# NOT deadlock-free: cyclic channel dependencies can form under load.

ROUTING_RANDOM_ADAPTIVE_TR = "RANDOM_ADAPTIVE_TR"
# Hybrid mode:
# - Normal VCs use fully random adaptive minimal routing.
# - Escape VC uses turn-restricted adaptive routing (West-First).
# - Packets may enter escape VC probabilistically and cannot return.

DRAIN_MODE_NORMAL = "NORMAL"
DRAIN_MODE_PRE_DRAIN = "PRE_DRAIN"
DRAIN_MODE_DRAIN = "DRAIN"
DRAIN_MODE_FULL_DRAIN = "FULL_DRAIN"


# ═══════════════════════════════════════════════════════════════════════════════
# Chiplet specification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChipletSpec:
    """
    Describes one chiplet and which interposer router each of its boundary
    routers connects to.

    Parameters
    ----------
    name         : Unique chiplet name (e.g. "GPU_TL", "CPU").
    chiplet_type : ``CHIPLET_CPU`` or ``CHIPLET_GPU``.
    br_ir_coords : List of (row, col) coordinates of the interposer router
                   each boundary router connects to.  The list length equals
                   the number of boundary routers for this chiplet.
                   Multiple chiplets may share the same IR coordinate — their
                   BRs will both be registered on that IR.
    vc_capacity  : VC capacity for this chiplet's BRs and their attached IRs.
                   Defaults to the type's standard capacity if None.
    """
    name:         str
    chiplet_type: str
    br_ir_coords: List[Tuple[int, int]]
    vc_capacity:  Optional[int] = None   # None → use type default

    def resolved_vc_capacity(self) -> int:
        if self.vc_capacity is not None:
            return self.vc_capacity
        return GPU_VC_CAPACITY if self.chiplet_type == CHIPLET_GPU else CPU_VC_CAPACITY


def build_image_layout() -> List[ChipletSpec]:
    """
    Build the 5-chiplet layout shown in the architecture diagram:

        4×4 interposer mesh (rows 0-3, cols 0-3)

        IR(0,0) IR(0,1) | IR(0,2) IR(0,3)
        IR(1,0) IR(1,1)*| IR(1,2)*IR(1,3)
        ────────────────┼────────────────
        IR(2,0) IR(2,1) | IR(2,2) IR(2,3)
        IR(3,0) IR(3,1) | IR(3,2) IR(3,3)

    GPU_TL (top-left)  : 4 BRs → IR(0,0), IR(0,1), IR(1,0), IR(1,1)
    GPU_TR (top-right) : 4 BRs → IR(0,2), IR(0,3), IR(1,2), IR(1,3)
    GPU_BL (bot-left)  : 4 BRs → IR(2,0), IR(2,1), IR(3,0), IR(3,1)
    GPU_BR (bot-right) : 4 BRs → IR(2,2), IR(2,3), IR(3,2), IR(3,3)
    CPU    (center)    : 2 BRs → IR(1,1)*, IR(1,2)*
                         (* shared with GPU_TL and GPU_TR respectively)

    All 16 IRs are covered; the CPU piggybacks on the inner IRs of the two
    top GPU chiplets without expanding the mesh.
    """
    return [
        ChipletSpec("GPU_TL", CHIPLET_GPU, [(0,0),(0,1),(1,0),(1,1)]),
        ChipletSpec("GPU_TR", CHIPLET_GPU, [(0,2),(0,3),(1,2),(1,3)]),
        ChipletSpec("GPU_BL", CHIPLET_GPU, [(2,0),(2,1),(3,0),(3,1)]),
        ChipletSpec("GPU_BR", CHIPLET_GPU, [(2,2),(2,3),(3,2),(3,3)]),
        ChipletSpec("CPU",    CHIPLET_CPU, [(1,1),(1,2)]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Flit
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Flit:
    """
    One flit of a wormhole packet.

    Each packet is decomposed into ``FLITS_PER_PACKET`` flits:
      - flit_idx == 0                         → header flit: performs routing
                                                and VC allocation at each hop.
      - 0 < flit_idx < FLITS_PER_PACKET - 1  → body flit:  follows the header.
      - flit_idx == FLITS_PER_PACKET - 1      → tail flit:  last flit; the VC's
                                                allocation is released after it
                                                departs (tracked dynamically).

    ``is_worm_head`` is True for the original header flit (flit_idx == 0) AND
    for the leading flit of any sub-worm created by a DRAIN split.  A worm-head
    flit can claim a free (unallocated) VC at the next router; body flits must
    follow an already-allocated VC.

    ``is_worm_tail`` is True for the original tail flit (flit_idx ==
    FLITS_PER_PACKET - 1) AND for the last flit of any DRAIN-split sub-worm.
    When a worm-tail flit departs a VC the VC's allocation is released, and
    when it departs a router the router's flow-output-table entry is cleared.

    ``in_escape_vc`` is set when the header flit's escape decision is made and
    is propagated to all sibling flits still in the boundary-router queue.
    """
    parent_packet: "Packet"
    flit_idx:      int        # position in the original packet (0 … FLITS_PER_PACKET-1)
    flow_id:       int        # unique worm identifier; may change on DRAIN split
    is_worm_head:  bool = False   # True → this flit performs routing & VC alloc
    is_worm_tail:  bool = False   # True → this flit releases VC & routing-table on depart
    in_escape_vc:  bool = False   # True → this flit belongs to the escape path


# ═══════════════════════════════════════════════════════════════════════════════
# Packet
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Packet:
    """One inter-chiplet packet travelling through the interposer."""

    packet_id:           int
    src_chiplet:         str            # e.g. "CPU_0_0"
    dst_chiplet:         str            # e.g. "CPU_2_3"
    src_boundary_router: str            # ID of the injecting boundary router
    dst_boundary_router: str            # ID of the ejecting boundary router
    created_cycle:       int
    current_node:        str  = ""      # ID of the node currently holding the header flit
    hops:                int  = 0       # interposer hops taken by the header flit
    delivered_cycle:     Optional[int] = None
    flits_delivered:     int  = 0       # number of this packet's flits that have been ejected
    flits:               List = field(default_factory=list)   # populated in __post_init__

    def __post_init__(self) -> None:
        if not self.flits:
            self.flits = [
                Flit(
                    parent_packet=self,
                    flit_idx=i,
                    flow_id=self.packet_id,
                    is_worm_head=(i == 0),
                    is_worm_tail=(i == FLITS_PER_PACKET - 1),
                )
                for i in range(FLITS_PER_PACKET)
            ]


# ═══════════════════════════════════════════════════════════════════════════════
# Link
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Link:
    """
    Unidirectional link between two network nodes (identified by string IDs).
    Links are created by InterposerMesh during topology construction and are
    used for topology inspection and statistics; actual packet movement is
    modelled through VC enqueue/dequeue rather than per-link in-flight queues.
    """

    src:     str
    dst:     str
    latency: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Virtual Channel (VC)
# ═══════════════════════════════════════════════════════════════════════════════

class VC:
    """
    FIFO virtual channel buffer with a fixed flit capacity.

    Wormhole semantics
    ------------------
    A VC is "allocated" to a worm (identified by ``allocated_flow_id``) when a
    worm-head flit enters.  Subsequent body/tail flits of the SAME flow can enter
    freely as long as there is space.  Flits of a DIFFERENT flow are rejected
    until the current worm completely departs.

    The allocation is released dynamically in ``pop()``: after a flit is
    dequeued, if no remaining flits in the queue share its flow_id the
    ``allocated_flow_id`` is cleared.  This means the last flit of any worm
    (original tail OR leading flit of a DRAIN-split singleton) correctly frees
    the VC without requiring an explicit ``is_worm_tail`` flag.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity:          int           = capacity
        self.fifo_queue:        List[Flit]    = []
        self.allocated_flow_id: Optional[int] = None   # wormhole lock

    # ── predicates ────────────────────────────────────────────────────────────

    def can_push(self) -> bool:
        return len(self.fifo_queue) < self.capacity

    def can_accept(self, flit: Flit) -> bool:
        """
        Return True if ``flit`` may enter this VC under wormhole rules.

        Worm heads can claim a VC that is unallocated OR that is empty with a
        stale allocation (the previous worm's tail already departed but the
        allocation was not cleared yet — it will be overwritten on push).
        Body/tail flits follow the currently allocated flow only.
        """
        if not self.can_push():
            return False
        if flit.is_worm_head:
            return self.allocated_flow_id is None or self.is_empty()
        return self.allocated_flow_id == flit.flow_id

    def can_pop(self) -> bool:
        return len(self.fifo_queue) > 0

    def is_full(self)  -> bool:
        return len(self.fifo_queue) == self.capacity

    def is_empty(self) -> bool:
        return len(self.fifo_queue) == 0

    def occupancy(self) -> int:
        return len(self.fifo_queue)

    # ── operations ────────────────────────────────────────────────────────────

    def push(self, flit: Flit) -> bool:
        """Enqueue flit under wormhole rules; returns True on success."""
        if not self.can_accept(flit):
            return False
        if flit.is_worm_head:
            self.allocated_flow_id = flit.flow_id
        self.fifo_queue.append(flit)
        return True

    def force_push(self, flit: Flit) -> None:
        """Push flit to the FRONT without wormhole checks (used for push-back)."""
        self.fifo_queue.insert(0, flit)
        if self.allocated_flow_id is None:
            self.allocated_flow_id = flit.flow_id

    def pop(self) -> Flit:
        """
        Dequeue and return the head flit.

        ``allocated_flow_id`` is cleared ONLY when the worm-tail flit departs
        (``flit.is_worm_tail == True``).  For non-tail flits the allocation
        deliberately persists: sibling body flits may still be in transit at
        upstream routers and need to see the allocation intact when they arrive
        in subsequent cycles.  For non-tail departures the allocation is cleared
        lazily when the next worm head pushes into this VC.
        """
        flit = self.fifo_queue.pop(0)
        if flit.is_worm_tail:
            self.allocated_flow_id = None
        return flit

    def peek(self) -> Optional[Flit]:
        """Return the head flit without removing it."""
        return self.fifo_queue[0] if self.fifo_queue else None

    def __repr__(self) -> str:
        return (
            f"VC(capacity={self.capacity}, occupancy={self.occupancy()}, "
            f"flow={self.allocated_flow_id})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Input Port
# ═══════════════════════════════════════════════════════════════════════════════

class InputPort:
    """
    Directional input buffer for one port of an InterposerRouter.

    Each InterposerRouter has five input ports: N, S, E, W, and Down (where
    Down receives flits from the attached BoundaryRouter).  Each port holds
    ``num_normal_vcs`` normal VCs and one escape VC.  Flits arriving from a
    given direction are buffered here until they are forwarded toward their
    destination or ejected.

    Wormhole semantics are enforced at the VC level (see ``VC.can_accept``).
    Port-level helpers check downstream capacity before committing a transfer.
    """

    def __init__(self, direction: str, vc_capacity: int, num_normal_vcs: int) -> None:
        self.direction:   str      = direction
        self.normal_vcs:  List[VC] = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc:   VC       = VC(vc_capacity)

    def all_vcs(self) -> List[Tuple[VC, bool]]:
        """Return all VCs as (vc, is_escape) pairs; normal VCs first."""
        return [(vc, False) for vc in self.normal_vcs] + [(self.escape_vc, True)]

    def can_accept(self, flit: "Flit", use_escape: bool) -> bool:
        """True if at least one eligible VC can accept this flit."""
        if use_escape:
            return self.escape_vc.can_accept(flit)
        return any(vc.can_accept(flit) for vc in self.normal_vcs)

    def enqueue(
        self,
        flit:       "Flit",
        use_escape: bool,
        rng:        Optional[random.Random] = None,
    ) -> bool:
        """Enqueue flit into the appropriate VC; random normal-VC selection."""
        if use_escape:
            return self.escape_vc.push(flit)
        if rng is None:
            rng = random
        order = list(range(len(self.normal_vcs)))
        if len(order) > 1:
            first = rng.randrange(len(order))
            order = [first] + [i for i in order if i != first]
        for i in order:
            if self.normal_vcs[i].push(flit):
                return True
        return False

    def has_escape_flit(self) -> bool:
        return not self.escape_vc.is_empty()

    def total_occupancy(self) -> int:
        return (sum(vc.occupancy() for vc in self.normal_vcs)
                + self.escape_vc.occupancy())

    def __repr__(self) -> str:
        return (f"InputPort({self.direction!r}, "
                f"normal={[vc.occupancy() for vc in self.normal_vcs]}, "
                f"escape={self.escape_vc.occupancy()})")


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary Router
# ═══════════════════════════════════════════════════════════════════════════════

class BoundaryRouter:
    """
    Connects a Chiplet to its attached InterposerRouter.

    Outbound path  (chiplet → interposer):
        Packets are injected directly into the IR's Down input-port VCs by
        ``InterposerMesh.inject_random_packets()``.  The BR itself holds no
        outbound buffers; it serves only as an address/identity anchor for the
        source and destination of each packet.

    Inbound path   (interposer → chiplet):
        ``InterposerMesh.step()`` calls ``receive_packet()`` when all flits of
        a packet have been ejected at the destination IR.
    """

    def __init__(
        self,
        router_id:    str,
        chiplet_name: str,
        vc_capacity:  int,
    ) -> None:
        self.router_id:    str = router_id
        self.chiplet_name: str = chiplet_name
        self.vc_capacity:  int = vc_capacity

        # ID of the interposer router this boundary router feeds into
        self.attached_ir: Optional[str] = None

        # Tracking
        self.injected_packets: List[Packet] = []   # all packets ever injected
        self.received_packets: List[Packet] = []   # all packets ever delivered

    # ── inbound (interposer → chiplet) ────────────────────────────────────────

    def receive_packet(self, packet: Packet, cycle: int) -> None:
        """Record packet delivery; called once all flits have been ejected."""
        packet.current_node    = self.router_id
        packet.delivered_cycle = cycle
        self.received_packets.append(packet)

    def __repr__(self) -> str:
        return (
            f"BoundaryRouter({self.router_id!r}, "
            f"chiplet={self.chiplet_name!r}, "
            f"ir={self.attached_ir!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Interposer Router
# ═══════════════════════════════════════════════════════════════════════════════

class InterposerRouter:
    """
    One node in the 4×4 interposer mesh.

    Buffer layout (per-port model)
    --------------------------------
    Each router has five named input ports:
      "N", "S", "E", "W"  – receive flits from the corresponding neighbour IR.
      "Down"               – receives flits from the attached BoundaryRouter(s).

    Each InputPort holds ``num_normal_vcs`` normal VCs and one escape VC.
    Flits arriving from a direction are buffered in that port's VCs until they
    are forwarded (or ejected) in a later cycle.

    Output-port arbitration (enforced by InterposerMesh.step())
    -----------------------------------------------------------
    Each cycle, at most ONE flit may leave the router via each output direction
    (N/S/E/W/Down).  Multiple input-port VCs may compete for the same output
    direction; one is chosen at random (the others retry next cycle).

    Wormhole semantics
    ------------------
    When a worm-head flit is forwarded it records
        flow_output_table[flow_id] = (next_rid, use_escape)
    Body/tail flits of the same flow look up this table at the same router
    (they always arrive via the same input port as the head, since the upstream
    router sends them all via the same output direction).
    """

    PORT_NAMES: List[str] = ["N", "S", "E", "W", "Down"]

    def __init__(
        self,
        router_id:      str,
        coordinate:     Tuple[int, int],
        vc_capacity:    int,
        num_normal_vcs: int,
    ) -> None:
        self.router_id:      str              = router_id
        self.coordinate:     Tuple[int, int]  = coordinate
        self.vc_capacity:    int              = vc_capacity
        self.num_normal_vcs: int              = num_normal_vcs

        # One InputPort per direction (including Down = from boundary router)
        self.input_ports: Dict[str, InputPort] = {
            d: InputPort(d, vc_capacity, num_normal_vcs)
            for d in self.PORT_NAMES
        }

        # Topology: direction ("N"/"S"/"E"/"W") → neighbour router_id or None
        self.ir_neighbors: Dict[str, Optional[str]] = {}

        # IDs of boundary routers attached to this interposer router
        self.attached_boundary_routers: List[str] = []

        # flow_id → (next_rid, use_escape): set by head flit, followed by body/tail
        self.flow_output_table: Dict[int, Tuple[str, bool]] = {}

    # ── topology helpers ──────────────────────────────────────────────────────

    def add_ir_neighbor(self, direction: str, ir_id: Optional[str]) -> None:
        self.ir_neighbors[direction] = ir_id

    def add_boundary_router(self, br_id: str) -> None:
        self.attached_boundary_routers.append(br_id)

    # ── convenience ───────────────────────────────────────────────────────────

    def has_flit(self) -> bool:
        return self.total_occupancy() > 0

    def total_occupancy(self) -> int:
        return sum(port.total_occupancy() for port in self.input_ports.values())

    def __repr__(self) -> str:
        r, c = self.coordinate
        return (
            f"InterposerRouter({self.router_id!r}, "
            f"coord=({r},{c}), "
            f"occupancy={self.total_occupancy()})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Chiplet
# ═══════════════════════════════════════════════════════════════════════════════

class Chiplet:
    """
    Base class for all chiplet types.

    Each chiplet:
      - *Generates* packets destined for other chiplets and injects them
        through its boundary routers.
      - *Receives* packets delivered by the interposer network.

    Subclasses (CPUChiplet, GPUChiplet) override ``current_injection_rate``
    to implement their specific traffic patterns.
    """

    chiplet_type: str = "BASE"   # overridden in subclasses

    def __init__(self, name: str, mesh_coord: Tuple[int, int]) -> None:
        self.name:       str             = name
        self.mesh_coord: Tuple[int, int] = mesh_coord   # (row, col) in the chiplet grid

        # Set by InterposerMesh during construction — one entry per boundary router
        self.boundary_routers: List[BoundaryRouter] = []

        # All packets delivered to this chiplet
        self.received_packets: List[Packet] = []

    # ── setup ─────────────────────────────────────────────────────────────────

    def add_boundary_router(self, br: BoundaryRouter) -> None:
        """Attach a boundary router (called during mesh construction)."""
        self.boundary_routers.append(br)

    # ── traffic model ─────────────────────────────────────────────────────────

    def current_injection_rate(self, cycle: int, rng: random.Random) -> float:
        """
        Return the injection probability for this cycle.

        Base implementation returns 0; subclasses override with their
        traffic patterns (uniform for CPU, bursty for GPU).
        The `rng` argument may be used to update stochastic burst state.
        """
        return 0.0

    # ── packet generation ─────────────────────────────────────────────────────

    def generate_packet(
        self,
        packet_id:   int,
        dst_chiplet: str,
        dst_br_id:   str,
        cycle:       int,
        rng:         random.Random,
    ) -> Optional[Packet]:
        """
        Legacy helper that constructs a packet anchored to one source BR.

        The current simulator injects directly in
        ``InterposerMesh.inject_random_packets()`` by writing flits into the
        attached interposer router's ``Down`` input-port VCs. Boundary routers
        no longer hold outbound VC state, so this helper does not enqueue.
        """
        if not self.boundary_routers:
            return None

        brs = list(self.boundary_routers)
        rng.shuffle(brs)
        br = brs[0]
        return Packet(
            packet_id=packet_id,
            src_chiplet=self.name,
            dst_chiplet=dst_chiplet,
            src_boundary_router=br.router_id,
            dst_boundary_router=dst_br_id,
            created_cycle=cycle,
            current_node=br.router_id,
        )

    # ── packet reception ──────────────────────────────────────────────────────

    def receive_packet(self, packet: Packet) -> None:
        self.received_packets.append(packet)

    # ── stats helpers ─────────────────────────────────────────────────────────

    def num_received(self) -> int:
        return len(self.received_packets)

    def __repr__(self) -> str:
        br_ids = [br.router_id for br in self.boundary_routers]
        return f"{self.chiplet_type}Chiplet({self.name!r}, coord={self.mesh_coord}, brs={br_ids})"


class CPUChiplet(Chiplet):
    """
    CPU chiplet — uniform Bernoulli traffic.

    Each cycle independently injects with probability ``injection_rate``.
    """

    chiplet_type = CHIPLET_CPU

    def __init__(
        self,
        name:           str,
        mesh_coord:     Tuple[int, int],
        injection_rate: float = CPU_INJECTION_RATE,
    ) -> None:
        super().__init__(name, mesh_coord)
        self.injection_rate: float = injection_rate

    def current_injection_rate(self, cycle: int, rng: random.Random) -> float:
        return self.injection_rate


class GPUChiplet(Chiplet):
    """
    GPU chiplet — bursty traffic.

    Traffic alternates between a high-rate burst window and a low-rate quiet
    window.  The chiplet tracks which phase it is in and how many cycles remain
    in that phase.  Within each cycle the injection probability is either
    ``burst_rate`` or ``quiet_rate``.

    Default parameters give ~4× higher average throughput than a CPU chiplet.
    """

    chiplet_type = CHIPLET_GPU

    def __init__(
        self,
        name:         str,
        mesh_coord:   Tuple[int, int],
        burst_rate:   float = GPU_BURST_RATE,
        quiet_rate:   float = GPU_QUIET_RATE,
        burst_cycles: int   = GPU_BURST_CYCLES,
        quiet_cycles: int   = GPU_QUIET_CYCLES,
    ) -> None:
        super().__init__(name, mesh_coord)
        self.burst_rate:   float = burst_rate
        self.quiet_rate:   float = quiet_rate
        self.burst_cycles: int   = burst_cycles
        self.quiet_cycles: int   = quiet_cycles

        # Start in a quiet phase so not all GPUs burst simultaneously
        self._in_burst:     bool = False
        self._phase_remaining: int = quiet_cycles

    def current_injection_rate(self, cycle: int, rng: random.Random) -> float:
        """
        Advance burst/quiet state machine by one cycle and return the
        current injection rate.
        """
        self._phase_remaining -= 1
        if self._phase_remaining <= 0:
            # Flip phase
            self._in_burst = not self._in_burst
            self._phase_remaining = (
                self.burst_cycles if self._in_burst else self.quiet_cycles
            )
        return self.burst_rate if self._in_burst else self.quiet_rate


# ═══════════════════════════════════════════════════════════════════════════════
# Interposer Mesh
# ═══════════════════════════════════════════════════════════════════════════════

class InterposerMesh:
    """
    Fixed rows×cols interposer mesh with an arbitrary set of chiplets.

    Topology
    --------
    The interposer always has exactly ``rows × cols`` InterposerRouters
    arranged in a grid.  Chiplets are described by a list of
    ``ChipletSpec`` objects, each specifying which IR coordinates its
    boundary routers connect to.  Multiple chiplets may share the same IR.

    Use ``build_image_layout()`` to get the default 5-chiplet layout from
    the architecture diagram (4 corner GPUs + 1 central CPU sharing 2 IRs
    with the top GPUs).

    Traffic
    -------
    CPU chiplets inject uniform Bernoulli traffic; GPU chiplets are bursty.
    Each chiplet's ``current_injection_rate()`` is called every cycle.
    """

    DIRECTIONS: Dict[str, Tuple[int, int]] = {
        "N": (-1,  0),
        "S": ( 1,  0),
        "E": ( 0,  1),
        "W": ( 0, -1),
    }
    OPPOSITE: Dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}
    # Maps output direction → input port name at the downstream router
    OPPOSITE_DIR: Dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def __init__(
        self,
        rows:              int              = MESH_ROWS,
        cols:              int              = MESH_COLS,
        chiplet_specs:     Optional[List[ChipletSpec]] = None,
        num_normal_vcs:    int              = NUM_NORMAL_VCS,
        routing_mode:      str              = ROUTING_XY,
        routing_seed:      int              = 0,
        drain_enabled:     bool             = False,
        escape_entry_prob: float            = ESCAPE_ENTRY_PROB,
        drain_period:      int              = DRAIN_PERIOD,
        drain_window_hops: int              = DRAIN_WINDOW_HOPS,
        full_drain_every_n_windows: int     = FULL_DRAIN_EVERY_N_WINDOWS,
        pre_drain_cycles:  int              = PRE_DRAIN_CYCLES,
        cpu_injection_rate: float           = CPU_INJECTION_RATE,
        gpu_burst_rate:    float            = GPU_BURST_RATE,
        gpu_quiet_rate:    float            = GPU_QUIET_RATE,
        gpu_burst_cycles:  int              = GPU_BURST_CYCLES,
        gpu_quiet_cycles:  int              = GPU_QUIET_CYCLES,
    ) -> None:
        self.rows:             int  = rows
        self.cols:             int  = cols
        self.num_normal_vcs:   int  = num_normal_vcs
        self.routing_mode:     str  = routing_mode
        self.drain_enabled:    bool = drain_enabled
        self.escape_entry_prob: float = max(0.0, min(1.0, escape_entry_prob))
        self.drain_period:      int   = max(1, drain_period)
        self.drain_window_hops: int   = max(1, drain_window_hops)
        self.full_drain_every_n_windows: int = max(1, full_drain_every_n_windows)
        self.pre_drain_cycles:  int   = max(0, pre_drain_cycles)
        self._drain_mode: str = DRAIN_MODE_NORMAL
        self._freeze_escape_admission: bool = False
        self._pre_drain_cycles_remaining: int = 0
        self._drain_hops_remaining: int = 0
        self._drain_window_count:   int = 0   # number of regular drain windows fired so far
        self._full_drain_hops_remaining: int = 0  # >0 while a full drain is in progress
        self.cpu_injection_rate: float = cpu_injection_rate
        self.gpu_burst_rate:     float = gpu_burst_rate
        self.gpu_quiet_rate:     float = gpu_quiet_rate
        self.gpu_burst_cycles:   int   = gpu_burst_cycles
        self.gpu_quiet_cycles:   int   = gpu_quiet_cycles

        # Default: image-layout (4 corner GPUs + 1 central CPU)
        if chiplet_specs is None:
            chiplet_specs = build_image_layout()
        self.chiplet_specs: List[ChipletSpec] = chiplet_specs

        # Validate that all IR coordinates fit within the mesh
        for spec in chiplet_specs:
            for r, c in spec.br_ir_coords:
                if not (0 <= r < rows and 0 <= c < cols):
                    raise ValueError(
                        f"Chiplet '{spec.name}' BR coord ({r},{c}) is outside "
                        f"the {rows}×{cols} mesh."
                    )

        # Per-IR VC capacity: take the max over all chiplets that attach to it.
        # (Shared IRs used by both GPU and CPU get GPU capacity.)
        self._ir_vc_cap: Dict[Tuple[int,int], int] = {}
        for spec in chiplet_specs:
            vc = spec.resolved_vc_capacity()
            for coord in spec.br_ir_coords:
                self._ir_vc_cap[coord] = max(self._ir_vc_cap.get(coord, 0), vc)
        # IRs with no chiplet attached use CPU default
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in self._ir_vc_cap:
                    self._ir_vc_cap[(r, c)] = CPU_VC_CAPACITY

        # Dedicated RNG for routing decisions (adaptive only).
        self._routing_rng: random.Random = random.Random(routing_seed)

        self.routers:          Dict[str, InterposerRouter] = {}
        self.chiplets:         Dict[str, Chiplet]          = {}
        self.boundary_routers: Dict[str, BoundaryRouter]   = {}
        self.links:            List[Link]                  = []

        # Unified ID counter: packet IDs and DRAIN-split flow IDs share one
        # namespace so they never collide.
        self._packet_counter:       int = 0
        self._attempted_injections: int = 0
        self._successful_injections: int = 0
        self._failed_injections:    int = 0   # attempted but dropped (all BRs full)
        self._attempted_injections_by_type: Dict[str, int] = {}
        self._successful_injections_by_type: Dict[str, int] = {}

        self._build_routers()
        self._build_links()
        self._attach_chiplets()
        self._drain_cycle: List[str] = self._compute_drain_cycle()
        self._drain_next: Dict[str, str] = {
            self._drain_cycle[i]: self._drain_cycle[(i + 1) % len(self._drain_cycle)]
            for i in range(len(self._drain_cycle))
        }

    # ── ID helpers ────────────────────────────────────────────────────────────

    def router_id(self, row: int, col: int) -> str:
        """IR identifier for mesh position (row, col)."""
        return f"IR_{row}_{col}"

    def br_id(self, chiplet_name: str, br_idx: int) -> str:
        """Boundary-router identifier: chiplet name + BR index."""
        return f"BR_{chiplet_name}_{br_idx}"

    # ── construction ──────────────────────────────────────────────────────────

    def _build_routers(self) -> None:
        """
        Create one InterposerRouter per mesh position (rows × cols).
        Each IR's VC capacity is the maximum over all chiplets that attach to it.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                rid    = self.router_id(r, c)
                vc_cap = self._ir_vc_cap[(r, c)]
                self.routers[rid] = InterposerRouter(
                    router_id=rid,
                    coordinate=(r, c),
                    vc_capacity=vc_cap,
                    num_normal_vcs=self.num_normal_vcs,
                )

    def _build_links(self) -> None:
        """Wire up cardinal neighbours on the rows × cols interposer grid."""
        for r in range(self.rows):
            for c in range(self.cols):
                rid    = self.router_id(r, c)
                router = self.routers[rid]
                for direction, (dr, dc) in self.DIRECTIONS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        nid = self.router_id(nr, nc)
                        router.add_ir_neighbor(direction, nid)
                        self.links.append(Link(src=rid, dst=nid))
                    else:
                        router.add_ir_neighbor(direction, None)

    def _attach_chiplets(self) -> None:
        """
        Instantiate each chiplet from its ChipletSpec, create its boundary
        routers, and register them with the appropriate interposer routers.
        Multiple chiplets may share the same IR.
        """
        for spec in self.chiplet_specs:
            vc_cap = spec.resolved_vc_capacity()

            if spec.chiplet_type == CHIPLET_GPU:
                chiplet: Chiplet = GPUChiplet(
                    name=spec.name,
                    mesh_coord=(0, 0),   # positional coord not used for routing
                    burst_rate=self.gpu_burst_rate,
                    quiet_rate=self.gpu_quiet_rate,
                    burst_cycles=self.gpu_burst_cycles,
                    quiet_cycles=self.gpu_quiet_cycles,
                )
            else:
                chiplet = CPUChiplet(
                    name=spec.name,
                    mesh_coord=(0, 0),
                    injection_rate=self.cpu_injection_rate,
                )

            for br_idx, (ir_r, ir_c) in enumerate(spec.br_ir_coords):
                bid = self.br_id(spec.name, br_idx)
                rid = self.router_id(ir_r, ir_c)

                br = BoundaryRouter(
                    router_id=bid,
                    chiplet_name=spec.name,
                    vc_capacity=vc_cap,
                )
                br.attached_ir = rid

                chiplet.add_boundary_router(br)
                self.routers[rid].add_boundary_router(bid)
                self.boundary_routers[bid] = br

            self.chiplets[spec.name] = chiplet

    def _are_neighbors(self, a: str, b: str) -> bool:
        ra, ca = self.routers[a].coordinate
        rb, cb = self.routers[b].coordinate
        return abs(ra - rb) + abs(ca - cb) == 1

    def _direction_to(self, src_rid: str, dst_rid: str) -> Optional[str]:
        """Return the output direction from src to its neighbor dst, or None."""
        for d, nid in self.routers[src_rid].ir_neighbors.items():
            if nid == dst_rid:
                return d
        return None

    def _compute_drain_cycle(self) -> List[str]:
        """
        Offline computation of a Hamiltonian cycle over the interposer routers.
        Escape-VC packets follow this cycle during DRAIN windows.
        """
        nodes = list(self.routers.keys())
        if not nodes:
            return []

        start = self.router_id(0, 0)
        path: List[str] = [start]
        visited = {start}

        neighbor_map: Dict[str, List[str]] = {}
        for rid, router in self.routers.items():
            neighbor_map[rid] = [
                nid for nid in router.ir_neighbors.values() if nid is not None
            ]

        def dfs(cur: str) -> bool:
            if len(path) == len(nodes):
                return self._are_neighbors(path[-1], start)

            # Warnsdorff-style ordering: try low-degree-remaining neighbors first.
            candidates = [n for n in neighbor_map[cur] if n not in visited]
            candidates.sort(
                key=lambda n: sum(1 for m in neighbor_map[n] if m not in visited)
            )
            for nxt in candidates:
                visited.add(nxt)
                path.append(nxt)
                if dfs(nxt):
                    return True
                path.pop()
                visited.remove(nxt)
            return False

        if not dfs(start):
            # Fallback: deterministic ring over sorted IDs (may include non-adjacent
            # jumps, but keeps behavior defined if cycle search fails).
            return sorted(nodes)
        return path

    def inject_faults(self, num_faults: int, rng: random.Random) -> List[Tuple[str, str]]:
        """
        Randomly disable ``num_faults`` bidirectional links on the interposer.

        Each fault removes one undirected edge by setting ``ir_neighbors`` to
        ``None`` in both endpoint routers.  Afterwards the drain Hamiltonian
        cycle is recomputed on the surviving topology so DRAIN windows can
        still make progress wherever connectivity allows.

        Parameters
        ----------
        num_faults : number of links to disable (clamped to available links).
        rng        : seeded Random instance so fault placement is reproducible.

        Returns
        -------
        List of (rid_a, rid_b) pairs for the disabled edges.
        """
        # Collect all undirected edges once.
        edges: List[Tuple[str, str, str]] = []  # (rid_a, rid_b, dir_a→b)
        seen: set = set()
        for rid, router in self.routers.items():
            for direction, nid in router.ir_neighbors.items():
                if nid is not None:
                    key = tuple(sorted([rid, nid]))
                    if key not in seen:
                        seen.add(key)
                        edges.append((rid, nid, direction))

        num_faults = min(num_faults, len(edges))
        if num_faults == 0:
            return []

        chosen_indices = rng.sample(range(len(edges)), num_faults)
        disabled: List[Tuple[str, str]] = []
        for i in chosen_indices:
            rid_a, rid_b, dir_a = edges[i]
            # Find the reverse direction.
            dir_b = next(
                d for d, n in self.routers[rid_b].ir_neighbors.items() if n == rid_a
            )
            self.routers[rid_a].ir_neighbors[dir_a] = None
            self.routers[rid_b].ir_neighbors[dir_b] = None
            disabled.append((rid_a, rid_b))

        # Recompute drain cycle on the surviving topology.
        self._drain_cycle = self._compute_drain_cycle()
        self._drain_next = {
            self._drain_cycle[i]: self._drain_cycle[(i + 1) % len(self._drain_cycle)]
            for i in range(len(self._drain_cycle))
        }
        return disabled

    def _drain_active(self) -> bool:
        return self.drain_enabled and self._drain_mode in (
            DRAIN_MODE_DRAIN,
            DRAIN_MODE_FULL_DRAIN,
        )

    def _pre_drain_active(self) -> bool:
        return self.drain_enabled and self._drain_mode == DRAIN_MODE_PRE_DRAIN

    def _print_drain_debug(self, cycle: int, message: str, force: bool = False) -> None:
        if DRAIN_DEBUG_PRINTS and (force or (cycle % max(1, DRAIN_DEBUG_PRINT_PERIOD) == 1)):
            print(f"  cycle={cycle} mode={self._drain_mode} {message}")

    def _set_drain_mode(self, mode: str) -> None:
        """Update the drain FSM mode and whether new escape admissions are allowed."""
        self._drain_mode = mode
        self._freeze_escape_admission = mode != DRAIN_MODE_NORMAL

    def _begin_pre_drain(self, cycle: int) -> None:
        """
        Enter PRE_DRAIN before the actual drain window begins.

        In this model links are not represented with explicit in-flight state,
        so PRE_DRAIN is implemented as a short period that freezes new escape-VC
        admissions before the regular/full drain starts.
        """
        if not self.drain_enabled:
            return
        if self.pre_drain_cycles <= 0:
            self._start_drain_window(cycle)
            return
        self._pre_drain_cycles_remaining = self.pre_drain_cycles
        self._set_drain_mode(DRAIN_MODE_PRE_DRAIN)
        self._print_drain_debug(
            cycle,
            f"enter pre_drain: freeze_escape_admission=1 cycles={self._pre_drain_cycles_remaining}",
            force=True,
        )

    def _start_drain_window(self, cycle: int) -> None:
        """Start either a regular DRAIN window or a periodic FULL_DRAIN window."""
        if not self.drain_enabled:
            return
        self._drain_window_count += 1
        if self._drain_window_count % self.full_drain_every_n_windows == 0:
            self._full_drain_hops_remaining = len(self._drain_cycle)
            self._drain_hops_remaining = 0
            self._set_drain_mode(DRAIN_MODE_FULL_DRAIN)
            self._print_drain_debug(
                cycle,
                f"enter full_drain: hops_remaining={self._full_drain_hops_remaining}",
                force=True,
            )
        else:
            self._drain_hops_remaining = self.drain_window_hops
            self._set_drain_mode(DRAIN_MODE_DRAIN)
            self._print_drain_debug(
                cycle,
                f"enter drain: hops_remaining={self._drain_hops_remaining}",
                force=True,
            )

    def _advance_drain_fsm(self, cycle: int) -> None:
        """Advance the drain scheduler once per cycle before flit movement."""
        if not self.drain_enabled:
            return

        if self._drain_mode == DRAIN_MODE_PRE_DRAIN:
            self._pre_drain_cycles_remaining -= 1
            if self._pre_drain_cycles_remaining <= 0:
                self._start_drain_window(cycle)
            return

        if self._drain_mode == DRAIN_MODE_DRAIN:
            if self._drain_hops_remaining <= 0:
                self._set_drain_mode(DRAIN_MODE_NORMAL)
                self._print_drain_debug(cycle, "exit drain", force=True)
            return

        if self._drain_mode == DRAIN_MODE_FULL_DRAIN:
            if self._full_drain_hops_remaining <= 0:
                self._set_drain_mode(DRAIN_MODE_NORMAL)
                self._print_drain_debug(cycle, "exit full_drain", force=True)
            return

        if cycle > 0 and cycle % self.drain_period == 0:
            self._begin_pre_drain(cycle)

    def _perform_regular_drain_hop(self, cycle: int) -> int:
        """
        Move escape-domain flits during a regular DRAIN window using
        output-port arbitration.

        Every input-port escape VC that has a head flit may request the router's
        drain-path output port. At most one winner is selected per output port
        (matching the normal forwarding model), then all winners are checked
        against downstream escape-VC availability and committed simultaneously.

        The current offline drain cycle assigns one drain-next router per
        source router, so all escape requests within one router typically
        contend for the same output port. This helper still removes the older
        "first input port only" shortcut and keeps arbitration output-centric.
        
        
        
        """
        if not self._drain_cycle:
            return 0

        # For each router input port that has an escape flit, build a request
        # for that router's drain-path output port.
        output_groups: Dict[Tuple[str, str], List[Tuple[str, str, str, str]]] = defaultdict(list)
        candidate_count = 0
        for src_rid, router in self.routers.items():
            dst_rid = self._drain_next.get(src_rid)
            if dst_rid is None:
                continue
            out_dir = self._direction_to(src_rid, dst_rid)
            if out_dir is None:
                continue
            dst_port_name = self.OPPOSITE_DIR[out_dir]
            for src_port_name, port in router.input_ports.items():
                if not port.has_escape_flit():
                    continue
                output_groups[(src_rid, out_dir)].append(
                    (src_rid, src_port_name, dst_rid, dst_port_name)
                )
                candidate_count += 1

        if candidate_count == 0:
            return 0

        # Arbitrate one winner per output port.
        winners: List[Tuple[str, str, str, str]] = []
        for _output_key, candidates in output_groups.items():
            winners.append(self._routing_rng.choice(candidates))

        # Determine which winners can actually advance based on downstream
        # escape-VC capacity/allocation.
        # dst_port_name = OPPOSITE_DIR[out_dir] where out_dir = direction to drain-next.
        will_depart_rids = {src_rid for src_rid, _, _, _ in winners}
        movable: List[Tuple[str, str, str, str]] = []
        for src_rid, src_port_name, dst_rid, dst_port_name in winners:
            dst_esc = self.routers[dst_rid].input_ports[dst_port_name].escape_vc
            # Account for a flit that may itself depart from dst this cycle.
            projected_occ = dst_esc.occupancy() - (1 if dst_rid in will_depart_rids else 0)
            dst_free = projected_occ < dst_esc.capacity
            dst_free_alloc = (
                dst_esc.allocated_flow_id is None
                or dst_rid in will_depart_rids
            )
            if dst_free and dst_free_alloc:
                movable.append((src_rid, src_port_name, dst_rid, dst_port_name))

        # Stage pops before committing.
        staged: List[Tuple[str, str, str, str, Flit]] = []
        for src_rid, src_port_name, dst_rid, dst_port_name in movable:
            src_esc = self.routers[src_rid].input_ports[src_port_name].escape_vc
            flit = src_esc.pop()
            staged.append((src_rid, src_port_name, dst_rid, dst_port_name, flit))

        moves = 0
        for src_rid, src_port_name, dst_rid, dst_port_name, flit in staged:
            src_router = self.routers[src_rid]
            src_esc = src_router.input_ports[src_port_name].escape_vc
            old_flow = flit.flow_id

            # ── DRAIN split: remaining flits of same flow in source escape VC ─
            remaining = [f for f in src_esc.fifo_queue if f.flow_id == old_flow]
            if remaining:
                new_flow_src = self._alloc_flow_id()
                for f in remaining:
                    f.flow_id = new_flow_src
                remaining[0].is_worm_head = True
                if not remaining[-1].is_worm_tail:
                    remaining[-1].is_worm_tail = True
                src_esc.allocated_flow_id = new_flow_src
                src_router.flow_output_table.pop(old_flow, None)

            # ── Moved flit becomes a fresh solo worm at the destination ────
            new_flow_dst = self._alloc_flow_id()
            flit.flow_id = new_flow_dst
            flit.is_worm_head = True
            flit.is_worm_tail = True
            flit.in_escape_vc = True

            dst_esc = self.routers[dst_rid].input_ports[dst_port_name].escape_vc
            dst_esc.push(flit)
            flit.parent_packet.current_node = dst_rid
            if flit.flit_idx == 0:
                flit.parent_packet.hops += 1
            moves += 1

        blocked = candidate_count - len(movable)
        self._print_drain_debug(
            cycle,
            f"regular_drain_hop: candidates={candidate_count} winners={len(winners)} "
            f"forwarded={len(movable)} blocked={blocked}",
            force = True
        )

        return moves

    # ── routing ───────────────────────────────────────────────────────────────

    def xy_next_hop(self, current_rid: str, dst_rid: str) -> Optional[str]:
        """
        XY routing: traverse columns (X) before rows (Y).
        Returns the ID of the next interposer router, or None if already at dst.
        """
        if current_rid == dst_rid:
            return None
        cr, cc = self.routers[current_rid].coordinate
        dr, dc = self.routers[dst_rid].coordinate

        if cc < dc:
            return self.routers[current_rid].ir_neighbors.get("E")
        if cc > dc:
            return self.routers[current_rid].ir_neighbors.get("W")
        if cr < dr:
            return self.routers[current_rid].ir_neighbors.get("S")
        return self.routers[current_rid].ir_neighbors.get("N")    # cr > dr

    def xy_path(self, src_rid: str, dst_rid: str) -> List[str]:
        """
        Return the ordered list of interposer router IDs on the XY path
        from src_rid to dst_rid (inclusive of both endpoints).
        """
        path    = [src_rid]
        current = src_rid
        while current != dst_rid:
            nxt = self.xy_next_hop(current, dst_rid)
            if nxt is None:
                break
            path.append(nxt)
            current = nxt
        return path

    def yx_next_hop(self, current_rid: str, dst_rid: str) -> Optional[str]:
        """
        YX routing: traverse rows (Y) first, then columns (X).
        Deadlock-free (mirrors XY but in the opposite dimension order).
        """
        if current_rid == dst_rid:
            return None
        cr, cc = self.routers[current_rid].coordinate
        dr, dc = self.routers[dst_rid].coordinate

        if cr < dr:
            return self.routers[current_rid].ir_neighbors.get("S")
        if cr > dr:
            return self.routers[current_rid].ir_neighbors.get("N")
        if cc < dc:
            return self.routers[current_rid].ir_neighbors.get("E")
        return self.routers[current_rid].ir_neighbors.get("W")

    def _minimal_directions(self, current_rid: str, dst_rid: str) -> List[str]:
        """
        Return all cardinal directions that strictly reduce Manhattan distance
        to dst_rid.  Used by adaptive routing to find the set of valid next hops.

        Example: if the packet must go East and South, returns ["E", "S"].
        If only East is needed, returns ["E"].
        """
        cr, cc = self.routers[current_rid].coordinate
        dr, dc = self.routers[dst_rid].coordinate
        dirs: List[str] = []
        if cc < dc: dirs.append("E")
        if cc > dc: dirs.append("W")
        if cr < dr: dirs.append("S")
        if cr > dr: dirs.append("N")
        return dirs

    def _turn_restricted_directions(self, current_rid: str, dst_rid: str) -> List[str]:
        """
        West-First turn restriction on minimal routes.

        If destination is to the West, the packet must move West now.
        Once West movement is no longer needed, the packet may choose
        adaptively among the remaining minimal directions (E/N/S).
        """
        cr, cc = self.routers[current_rid].coordinate
        dr, dc = self.routers[dst_rid].coordinate

        if cc > dc:
            return ["W"]

        dirs: List[str] = []
        if cc < dc:
            dirs.append("E")
        if cr < dr:
            dirs.append("S")
        if cr > dr:
            dirs.append("N")
        return dirs

    def compute_next_hop(self, current_rid: str, dst_rid: str) -> Optional[str]:
        """
        Unified next-hop computation.  Dispatches to the appropriate algorithm
        based on self.routing_mode.

        ROUTING_XY              – deterministic, always deadlock-free.
        ROUTING_YX              – deterministic, always deadlock-free.
        ROUTING_RANDOM_ADAPTIVE – randomly picks among all minimal directions;
                                  NOT deadlock-free under load.
        ROUTING_RANDOM_ADAPTIVE_TR – turn-restricted adaptive (West-First),
                                     deadlock-free by construction.
        """
        if current_rid == dst_rid:
            return None

        if self.routing_mode == ROUTING_XY:
            return self.xy_next_hop(current_rid, dst_rid)

        if self.routing_mode == ROUTING_YX:
            return self.yx_next_hop(current_rid, dst_rid)

        if self.routing_mode == ROUTING_RANDOM_ADAPTIVE:
            dirs = self._minimal_directions(current_rid, dst_rid)
            if not dirs:
                return None
            chosen = self._routing_rng.choice(dirs)
            return self.routers[current_rid].ir_neighbors.get(chosen)

        if self.routing_mode == ROUTING_RANDOM_ADAPTIVE_TR:
            dirs = self._turn_restricted_directions(current_rid, dst_rid)
            if not dirs:
                return None
            chosen = self._routing_rng.choice(dirs)
            return self.routers[current_rid].ir_neighbors.get(chosen)

        # Fallback to XY for unknown modes
        return self.xy_next_hop(current_rid, dst_rid)

    def compute_next_hop_for_packet(
        self,
        current_rid: str,
        dst_rid: str,
        is_escape: bool = False,
    ) -> Optional[str]:
        """
        Per-packet next hop. In ROUTING_RANDOM_ADAPTIVE_TR, ALL packets
        (normal and escape VCs) follow West-First turn restrictions so that
        the entire network is deadlock-free by construction.
        """
        if current_rid == dst_rid:
            return None

        if self.routing_mode != ROUTING_RANDOM_ADAPTIVE_TR:
            return self.compute_next_hop(current_rid, dst_rid)

        dirs = self._turn_restricted_directions(current_rid, dst_rid)
        if not dirs:
            return None
        chosen = self._routing_rng.choice(dirs)
        return self.routers[current_rid].ir_neighbors.get(chosen)

    def all_minimal_hops_blocked(self, current_rid: str, dst_rid: str) -> bool:
        """
        Return True only when EVERY minimal-path direction from current_rid
        toward dst_rid cannot accept a new worm head.

        Under wormhole flow control a VC is locked to one flow at a time.  A
        4-flit packet occupies at most 4 of 8 VC slots, so ``vc.is_full()``
        is never true and cannot be used as the blocking predicate.  Instead a
        VC blocks a new worm head when it is already allocated to another flow
        AND still holds flits of that flow (non-empty).

        Used by the deadlock detector's wait-for graph:
        - For deterministic routing (XY/YX) there is only one minimal direction,
          so this reduces to a single check.
        - For adaptive routing a packet is not truly blocked unless ALL
          minimal directions are occupied; checking just one would give false
          positives.
        """
        def _port_blocks_new_head(port: InputPort) -> bool:
            """True when ALL normal VCs in this input port block a new worm head."""
            return all(
                vc.allocated_flow_id is not None and not vc.is_empty()
                for vc in port.normal_vcs
            )

        if self.routing_mode in (ROUTING_XY, ROUTING_YX):
            nxt = self.compute_next_hop(current_rid, dst_rid)
            if nxt is None:
                return False
            out_dir = self._direction_to(current_rid, nxt)
            if out_dir is None:
                return False
            in_port = self.routers[nxt].input_ports[self.OPPOSITE_DIR[out_dir]]
            return _port_blocks_new_head(in_port)

        # Adaptive variants: blocked only when ALL valid next-hop input ports
        # are fully occupied by other worms.
        dirs = (
            self._turn_restricted_directions(current_rid, dst_rid)
            if self.routing_mode == ROUTING_RANDOM_ADAPTIVE_TR
            else self._minimal_directions(current_rid, dst_rid)
        )
        if not dirs:
            return False
        for direction in dirs:
            nxt = self.routers[current_rid].ir_neighbors.get(direction)
            if nxt is None:
                continue
            in_port = self.routers[nxt].input_ports[self.OPPOSITE_DIR[direction]]
            if not _port_blocks_new_head(in_port):
                return False   # at least one direction has a free VC → not blocked
        return True   # every minimal direction is occupied → truly blocked

    # ── traffic injection ─────────────────────────────────────────────────────

    def _next_packet_id(self) -> int:
        pid = self._packet_counter
        self._packet_counter += 1
        return pid

    def _alloc_flow_id(self) -> int:
        """Allocate a fresh flow ID for DRAIN-split sub-worms."""
        return self._next_packet_id()

    def inject_random_packets(
        self,
        cycle: int,
        rng:   random.Random,
    ) -> int:
        """
        Each chiplet independently decides whether to inject a packet this cycle.

        Injection model (direct-to-IR):
          Flits bypass the boundary-router staging buffer and are written
          directly into the attached IR's Down input-port VCs at injection time.
          A normal VC in the Down port is eligible if it is unallocated AND has
          at least ``FLITS_PER_PACKET`` free slots.  Source BRs are tried in
          random order; the first eligible IR Down-port VC wins.
          If no eligible VC exists across all source BRs, the injection fails
          and ``_failed_injections`` is incremented.

        Returns the count of successfully injected packets this cycle.
        During PRE_DRAIN and FULL_DRAIN, injection is suppressed.
        """
        pre_drain_starts_this_cycle = (
            self.drain_enabled
            and self._drain_mode == DRAIN_MODE_NORMAL
            and cycle > 0
            and cycle % self.drain_period == 0
        )
        if self._full_drain_active() or self._pre_drain_active() or pre_drain_starts_this_cycle:
            return 0
        chiplet_names = list(self.chiplets.keys())
        injected = 0
        for src_name, chiplet in self.chiplets.items():
            rate = chiplet.current_injection_rate(cycle, rng)
            if rng.random() >= rate:
                continue
            self._attempted_injections += 1
            self._attempted_injections_by_type[chiplet.chiplet_type] = (
                self._attempted_injections_by_type.get(chiplet.chiplet_type, 0) + 1
            )
            dst_name    = rng.choice([n for n in chiplet_names if n != src_name])
            dst_chiplet = self.chiplets[dst_name]
            if not dst_chiplet.boundary_routers:
                continue
            dst_br = rng.choice(dst_chiplet.boundary_routers)

            # Try source BRs in random order; push all flits into the first
            # eligible Down-port normal VC found at the attached IR.
            src_brs = list(chiplet.boundary_routers)
            rng.shuffle(src_brs)
            success = False
            for src_br in src_brs:
                ir        = self.routers[src_br.attached_ir]   # type: ignore[index]
                down_port = ir.input_ports["Down"]
                eligible  = [
                    vc for vc in down_port.normal_vcs
                    if vc.allocated_flow_id is None
                    and (vc.capacity - vc.occupancy()) >= FLITS_PER_PACKET
                ]
                if not eligible:
                    continue
                target_vc = rng.choice(eligible)
                pkt = Packet(
                    packet_id=self._next_packet_id(),
                    src_chiplet=src_name,
                    dst_chiplet=dst_name,
                    src_boundary_router=src_br.router_id,
                    dst_boundary_router=dst_br.router_id,
                    created_cycle=cycle,
                    current_node=ir.router_id,
                )
                for flit in pkt.flits:
                    target_vc.push(flit)
                src_br.injected_packets.append(pkt)
                injected += 1
                self._successful_injections += 1
                self._successful_injections_by_type[chiplet.chiplet_type] = (
                    self._successful_injections_by_type.get(chiplet.chiplet_type, 0) + 1
                )
                success = True
                break

            if not success:
                self._failed_injections += 1
        return injected

    # ── DRAIN scheduling ──────────────────────────────────────────────────────

    def _full_drain_active(self) -> bool:
        return self.drain_enabled and self._drain_mode == DRAIN_MODE_FULL_DRAIN

    # ── simulation step ───────────────────────────────────────────────────────

    def step(self, cycle: int) -> int:
        """
        Advance the network by one cycle at flit granularity under wormhole
        flow control.  Returns the number of flit movements that occurred
        (useful for deadlock detection).

        Phase 1 – Boundary-router outbound VCs → attached interposer router.
                  One flit per boundary router per cycle.

                  Escape decision (header flits only):
                    If the flit is a worm head and not yet assigned to escape,
                    a Bernoulli trial decides whether it enters the escape VC.
                    On success, ALL remaining sibling flits still in the BR
                    outbound queue have ``in_escape_vc`` set to True so they
                    follow the header into the escape path.

        Phase 2a – Synchronized DRAIN circulation for escape-VC flits
                   (only when a DRAIN window is active).

        Phase 2b – Normal flit forwarding through the interposer.

                  Worm-head flits: run routing, decide escape probability,
                    record next hop in ``router.flow_output_table``.
                  Body/tail flits: look up ``flow_output_table`` to find the
                    pre-recorded next hop; if the entry is absent the flit was
                    separated from its header (DRAIN split) and is promoted to
                    a new worm head so it can route independently.

                  Ejection: when a flit reaches its destination IR, it is
                    delivered.  The parent Packet is considered delivered once
                    all ``FLITS_PER_PACKET`` flits have been ejected.
        """
        moves = 0
        self._advance_drain_fsm(cycle)

        # ── Full drain: freeze injection + normal-VC forwarding ───────────
        # During a full drain the Hamiltonian cycle runs one hop per cycle
        # until every escape-VC flit has visited all routers and can eject.
        if self._full_drain_active():
            hop_moves = self._perform_regular_drain_hop(cycle)
            moves += hop_moves
            self._full_drain_hops_remaining -= 1
            # Eject escape flits sitting at their destination IR.
            ejected = 0
            for rid, router in self.routers.items():
                for port in router.input_ports.values():
                    flit = port.escape_vc.peek()
                    if flit is None:
                        continue
                    dst_br_id = flit.parent_packet.dst_boundary_router
                    dst_ir_id = self.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]
                    if rid == dst_ir_id:
                        port.escape_vc.pop()
                        pkt = flit.parent_packet
                        pkt.flits_delivered += 1
                        if pkt.flits_delivered == FLITS_PER_PACKET:
                            dst_br = self.boundary_routers[dst_br_id]
                            dst_br.receive_packet(pkt, cycle)
                            self.chiplets[dst_br.chiplet_name].receive_packet(pkt)
                        moves += 1
                        ejected += 1
            self._print_drain_debug(
                cycle,
                f"full_drain_step: hop_moves={hop_moves} ejected={ejected} hops_remaining={self._full_drain_hops_remaining}",
                force=True
            )
            return moves

        # ── Phase 2a: synchronized DRAIN circulation for escape VCs ───────
        drain_this_cycle = self._drain_active()
        if drain_this_cycle and self._drain_mode == DRAIN_MODE_DRAIN:
            moves += self._perform_regular_drain_hop(cycle)
            self._drain_hops_remaining -= 1
            if STRICT_REGULAR_DRAIN:
                self._print_drain_debug(
                    cycle,
                    "strict_regular_drain: normal forwarding skipped",
                    force=True,
                )
                return moves

        # ── Phase 2b: per-port flit forwarding with output-port arbitration ─
        #
        # For each (router, input_port, VC) triple, determine the head flit's
        # desired output direction and whether the downstream port can accept it.
        # Candidates are grouped by (src_rid, output_direction).  At most ONE
        # flit may use each output port per cycle; a winner is chosen at random
        # among eligible candidates (losers retry next cycle).
        #
        # Candidate tuple: (port_name, vc, is_escape_vc, flit, nxt_rid, use_escape_next)
        # output_direction "Down" = eject to boundary router (no downstream check).

        output_groups: Dict[Tuple[str, str], List] = defaultdict(list)

        for rid, router in self.routers.items():
            for port_name, port in router.input_ports.items():
                for vc, is_escape_vc in port.all_vcs():
                    flit = vc.peek()
                    if flit is None:
                        continue
                    if is_escape_vc and drain_this_cycle:
                        continue   # escape flits handled by DRAIN this cycle

                    dst_br_id = flit.parent_packet.dst_boundary_router
                    dst_ir_id = self.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]

                    # ── Eject candidate ───────────────────────────────────
                    if rid == dst_ir_id:
                        output_groups[(rid, "Down")].append(
                            (port_name, vc, is_escape_vc, flit, None, False)
                        )
                        continue

                    # ── Forward candidate ─────────────────────────────────
                    if flit.is_worm_head:
                        use_escape_next = is_escape_vc
                        if self._pre_drain_active() and not use_escape_next:
                            # PRE_DRAIN freezes new normal worm-head VC grabs.
                            continue
                        nxt_rid = self.compute_next_hop_for_packet(
                            rid, dst_ir_id, is_escape=use_escape_next
                        )
                        if nxt_rid is None:
                            continue
                        out_dir = self._direction_to(rid, nxt_rid)
                        if out_dir is None:
                            continue
                        in_port_name = self.OPPOSITE_DIR[out_dir]
                        nxt_port     = self.routers[nxt_rid].input_ports[in_port_name]

                        # Escape/normal VC selection (non-escape flits only):
                        # 1. Flip coin (or force if normal VC full).
                        # 2. If coin says escape but escape VC full → fall back to normal.
                        # 3. If both full → skip (retry next cycle).
                        allow_new_escape = not self._freeze_escape_admission
                        if (
                            allow_new_escape
                            and not use_escape_next
                            and self.escape_entry_prob > 0
                        ):
                            normal_blocked = not nxt_port.can_accept(flit, use_escape=False)
                            want_escape    = (normal_blocked
                                             or self._routing_rng.random() < self.escape_entry_prob)
                            if want_escape and nxt_port.can_accept(flit, use_escape=True):
                                use_escape_next = True

                        if not nxt_port.can_accept(flit, use_escape=use_escape_next):
                            continue   # blocked — wait

                        output_groups[(rid, out_dir)].append(
                            (port_name, vc, is_escape_vc, flit, nxt_rid, use_escape_next)
                        )

                    else:
                        # Body/tail: follow head's recorded routing decision.
                        entry = router.flow_output_table.get(flit.flow_id)
                        if entry is None:
                            # Stranded by DRAIN split → promote to new worm head.
                            if self._pre_drain_active() and not is_escape_vc:
                                continue
                            flit.is_worm_head = True
                            flit.is_worm_tail = True
                            use_escape_next   = is_escape_vc
                            nxt_rid = self.compute_next_hop_for_packet(
                                rid, dst_ir_id, is_escape=use_escape_next
                            )
                            if nxt_rid is None:
                                flit.is_worm_head = False
                                flit.is_worm_tail = False
                                continue
                            out_dir = self._direction_to(rid, nxt_rid)
                            if out_dir is None:
                                flit.is_worm_head = False
                                flit.is_worm_tail = False
                                continue
                            in_port_name = self.OPPOSITE_DIR[out_dir]
                            nxt_port     = self.routers[nxt_rid].input_ports[in_port_name]
                            if not nxt_port.can_accept(flit, use_escape=use_escape_next):
                                flit.is_worm_head = False
                                flit.is_worm_tail = False
                                continue
                            output_groups[(rid, out_dir)].append(
                                (port_name, vc, is_escape_vc, flit, nxt_rid, use_escape_next)
                            )
                        else:
                            nxt_rid, use_escape_next = entry
                            out_dir = self._direction_to(rid, nxt_rid)
                            if out_dir is None:
                                continue
                            in_port_name = self.OPPOSITE_DIR[out_dir]
                            nxt_port     = self.routers[nxt_rid].input_ports[in_port_name]
                            if not nxt_port.can_accept(flit, use_escape=use_escape_next):
                                continue   # body/tail blocked — wait
                            output_groups[(rid, out_dir)].append(
                                (port_name, vc, is_escape_vc, flit, nxt_rid, use_escape_next)
                            )

        # ── Arbitrate and commit ──────────────────────────────────────────
        for (rid, out_dir), candidates in output_groups.items():
            # Pick one winner at random among all eligible candidates.
            winner = self._routing_rng.choice(candidates)
            port_name, vc, _is_esc, flit, nxt_rid, use_escape_next = winner
            router = self.routers[rid]

            if out_dir == "Down":
                # Eject: deliver flit to the boundary router.
                vc.pop()
                pkt = flit.parent_packet
                pkt.flits_delivered += 1
                if pkt.flits_delivered == FLITS_PER_PACKET:
                    dst_br_id = pkt.dst_boundary_router
                    dst_br    = self.boundary_routers[dst_br_id]
                    dst_br.receive_packet(pkt, cycle)
                    self.chiplets[dst_br.chiplet_name].receive_packet(pkt)
                # Clear routing table if this is the worm tail.
                if flit.is_worm_tail:
                    router.flow_output_table.pop(flit.flow_id, None)
                moves += 1
            else:
                # Forward to next router's input port.
                vc.pop()
                if use_escape_next and not flit.in_escape_vc:
                    flit.in_escape_vc = True

                in_port_name = self.OPPOSITE_DIR[out_dir]
                nxt_port     = self.routers[nxt_rid].input_ports[in_port_name]
                nxt_port.enqueue(flit, use_escape=use_escape_next, rng=self._routing_rng)

                if flit.is_worm_head:
                    flit.parent_packet.current_node = nxt_rid
                    flit.parent_packet.hops += 1
                    if not flit.is_worm_tail:
                        router.flow_output_table[flit.flow_id] = (nxt_rid, use_escape_next)
                    else:
                        router.flow_output_table.pop(flit.flow_id, None)
                elif flit.is_worm_tail:
                    router.flow_output_table.pop(flit.flow_id, None)

                moves += 1

        return moves

    # ── topology queries ──────────────────────────────────────────────────────
    # NOT used in the core simulation loop, but useful for testing, debugging, and statistics
    def get_router(self, row: int, col: int) -> InterposerRouter:
        return self.routers[self.router_id(row, col)]

    def get_chiplet(self, chiplet_name: str) -> Chiplet:
        return self.chiplets[chiplet_name]

    def get_boundary_router(self, br_id: str) -> BoundaryRouter:
        return self.boundary_routers[br_id]

    # ── network-wide statistics ───────────────────────────────────────────────

    def all_in_flight(self) -> int:
        """
        Total flits currently buffered inside the network.

        In the current model packets inject directly into interposer-router
        input ports, so boundary routers no longer contribute outbound state.
        Divide by ``FLITS_PER_PACKET`` to get the packet-equivalent count.
        """
        total = 0
        for router in self.routers.values():
            total += router.total_occupancy()
        return total

    def all_in_flight_packets_equiv(self) -> float:
        """Packet-equivalent backlog derived from the flit backlog."""
        return self.all_in_flight() / float(FLITS_PER_PACKET)

    def _packet_stats(self, packets: List[Packet]) -> Dict[str, float]:
        """Summarize latency and hop counts for a packet collection."""
        total_delivered = 0
        total_latency   = 0
        total_hops      = 0
        for pkt in packets:
            total_delivered += 1
            if pkt.delivered_cycle is not None:
                total_latency += pkt.delivered_cycle - pkt.created_cycle
            total_hops += pkt.hops
        avg_lat  = total_latency / total_delivered if total_delivered else 0.0
        avg_hops = total_hops    / total_delivered if total_delivered else 0.0
        return {
            "delivered":   total_delivered,
            "avg_latency": round(avg_lat, 2),
            "avg_hops":    round(avg_hops, 2),
        }

    def delivered_stats(self) -> Dict[str, float]:
        """Aggregate injection, delivery, latency, and backlog statistics."""
        delivered_packets = [
            pkt
            for chiplet in self.chiplets.values()
            for pkt in chiplet.received_packets
        ]
        stats = self._packet_stats(delivered_packets)
        acceptance_rate = (
            self._successful_injections / self._attempted_injections
            if self._attempted_injections
            else 0.0
        )
        stats.update({
            "attempted_injections": self._attempted_injections,
            "injected": self._successful_injections,
            "failed_injections": self._failed_injections,
            "acceptance_rate": round(acceptance_rate, 4),
            "in_flight_flits": self.all_in_flight(),
            "in_flight_packets_equiv": round(self.all_in_flight_packets_equiv(), 4),
        })
        return stats

    def delivered_stats_by_type(self) -> Dict[str, Dict[str, float]]:
        """Aggregate stats by source chiplet type."""
        packets_by_type: Dict[str, List[Packet]] = {}
        chiplet_counts_by_type: Dict[str, int] = {}

        for chiplet in self.chiplets.values():
            chiplet_counts_by_type[chiplet.chiplet_type] = (
                chiplet_counts_by_type.get(chiplet.chiplet_type, 0) + 1
            )

        for chiplet in self.chiplets.values():
            for pkt in chiplet.received_packets:
                src_type = self.chiplets[pkt.src_chiplet].chiplet_type
                packets_by_type.setdefault(src_type, []).append(pkt)

        all_types = set(chiplet_counts_by_type) | set(self._attempted_injections_by_type)
        stats_by_type: Dict[str, Dict[str, float]] = {}
        for chiplet_type in sorted(all_types):
            attempted = self._attempted_injections_by_type.get(chiplet_type, 0)
            injected = self._successful_injections_by_type.get(chiplet_type, 0)
            acceptance_rate = injected / attempted if attempted else 0.0
            type_stats = self._packet_stats(packets_by_type.get(chiplet_type, []))
            type_stats.update({
                "attempted_injections": attempted,
                "injected": injected,
                "failed_injections": attempted - injected,
                "acceptance_rate": round(acceptance_rate, 4),
                "chiplet_count": chiplet_counts_by_type.get(chiplet_type, 0),
            })
            stats_by_type[chiplet_type] = type_stats
        return stats_by_type

    def __repr__(self) -> str:
        cpu_count = sum(1 for s in self.chiplet_specs if s.chiplet_type == CHIPLET_CPU)
        gpu_count = sum(1 for s in self.chiplet_specs if s.chiplet_type == CHIPLET_GPU)
        return (
            f"InterposerMesh(mesh={self.rows}×{self.cols}, "
            f"{len(self.chiplets)} chiplets [{cpu_count} CPUs, {gpu_count} GPUs], "
            f"{len(self.routers)} IRs, "
            f"{len(self.boundary_routers)} BRs, "
            f"{len(self.links)} directed links)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Deadlock Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeadlockReport:
    """
    Result of one deadlock detection check.

    detected        True if a deadlock was identified this cycle.
    cycle           The simulation cycle at which the check was made.
    method          Which method triggered the report:
                      "wait_for_cycle"  – a cycle was found in the wait-for graph
                      "none"            – no deadlock detected
    router_cycles   List of deadlock cycles found in the wait-for graph; each
                    inner list is an ordered sequence of router IDs that form a
                    cyclic dependency (R0 waits for R1, …, Rn waits for R0).
    blocked_packets Packet IDs of the head packets sitting at the routers
                    identified as being in a deadlock cycle.
    """

    detected:        bool
    cycle:           int
    method:          str
    router_cycles:   List[List[str]] = field(default_factory=list)
    blocked_packets: List[int]       = field(default_factory=list)


class DeadlockDetector:
    """
    Detects routing-level deadlocks in the interposer interconnect.

    A routing-level deadlock occurs when a set of packets form a cyclic
    resource dependence: each packet holds a VC buffer and is waiting for
    a buffer held by the next packet in the cycle, so none can make progress.

    Detection method – Wait-for graph (WFG) cycle detection
    ────────────────────────────────────────────────────────
    Each cycle, a directed *wait-for graph* is constructed over the interposer
    routers:

        Edge  R_a → R_b  exists when
            • R_a has a head packet whose next hop is R_b, AND
            • ALL minimal-direction VCs at R_b are completely full
              (for adaptive routing this means every valid next hop is blocked,
               not just the one randomly chosen this cycle).

    Each blocked router has at most one outgoing edge, so this is a functional
    graph (out-degree ≤ 1).  Cycle detection runs in O(|V|) time by following
    chains until either a repeated node (cycle) or an unblocked node is found.

    A cycle in the WFG means the routers in the cycle are mutually blocked by
    true circular resource dependence → deadlock confirmed.

    Usage
    ─────
        detector = DeadlockDetector()
        for cycle in range(total_cycles):
            mesh.inject_random_packets(cycle, rng)
            mesh.step(cycle)
            report = detector.check(mesh, cycle)
            if report.detected:
                # handle / log / trigger DRAIN ...
    """

    def __init__(
        self,
        persist_cycles: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        persist_cycles : int
            Number of consecutive cycles a WFG cycle must persist before it
            is reported as a confirmed deadlock.  Setting this to 1 reports
            every cycle immediately; higher values filter transient HOL
            blocking that self-resolves.
        """
        self.persist_cycles:   int = persist_cycles
        self._total_deadlocks: int = 0
        self._history:         List[DeadlockReport] = []
        # Tracks how many consecutive cycles each frozen cycle signature has
        # been observed: frozenset(router_ids) → consecutive count
        self._cycle_streak: Dict[frozenset, int] = {}

    # ── public interface ──────────────────────────────────────────────────────

    def check(
        self,
        mesh:  "InterposerMesh",
        cycle: int,
    ) -> DeadlockReport:
        """
        Build the wait-for graph, detect cycles, and return a DeadlockReport.

        Parameters
        ----------
        mesh  : The InterposerMesh being simulated.
        cycle : Current simulation cycle (for reporting).
        """
        # ── Wait-for graph cycle detection ────────────────────────────────
        wfg    = self._build_wait_for_graph(mesh)
        cycles = self._detect_cycles(wfg)

        # Update persistence streaks: increment seen cycles, drop unseen ones
        current_sigs = {frozenset(c) for c in cycles}
        new_streak: Dict[frozenset, int] = {}
        for sig in current_sigs:
            new_streak[sig] = self._cycle_streak.get(sig, 0) + 1
        self._cycle_streak = new_streak

        # Report only cycles that JUST reached the persist threshold (first
        # confirmation) — avoids re-reporting the same deadlock every cycle.
        newly_confirmed = [c for c in cycles
                           if self._cycle_streak.get(frozenset(c), 0) == self.persist_cycles]

        if newly_confirmed:
            blocked_pkt_ids = self._blocked_packet_ids(mesh, newly_confirmed)
            report = DeadlockReport(
                detected=True,
                cycle=cycle,
                method="wait_for_cycle",
                router_cycles=newly_confirmed,
                blocked_packets=blocked_pkt_ids,
            )
            self._total_deadlocks += 1
            self._history.append(report)
            return report

        return DeadlockReport(
            detected=False,
            cycle=cycle,
            method="none",
        )

    @property
    def total_deadlocks(self) -> int:
        """Cumulative number of deadlock events detected across the simulation."""
        return self._total_deadlocks

    @property
    def history(self) -> List[DeadlockReport]:
        """All DeadlockReports where detected=True."""
        return list(self._history)

    # ── wait-for graph construction ───────────────────────────────────────────

    def _build_wait_for_graph(
        self,
        mesh: "InterposerMesh",
    ) -> Dict[str, str]:
        """
        Build the wait-for graph over interposer routers.

        For each InterposerRouter R:
          1. Peek at R's head packet P (skip if empty).
          2. Determine P's destination IR (dst_ir).
          3. If R == dst_ir the packet can eject; R is not blocked.
          4. Compute next_hop = xy_next_hop(R, dst_ir).
          5. Check whether next_hop's target VC has room:
               - escape VC if P.in_escape_vc else any normal VC.
          6. If no room: record  R.router_id → next_hop  in the graph.

        Returns a dict { blocked_router_id : next_hop_router_id }.
        Each blocked router has exactly one outgoing edge (XY is deterministic),
        so this is a functional graph (out-degree ≤ 1).
        """
        wait_for: Dict[str, str] = {}

        for rid, router in mesh.routers.items():
            if rid in wait_for:
                continue   # already found a blocked flit for this router
            # Scan all input ports and all VCs — check each head-of-line flit.
            found_edge = False
            for port in router.input_ports.values():
                if found_edge:
                    break
                for vc, is_escape in port.all_vcs():
                    flit = vc.peek()
                    if flit is None:
                        continue

                    dst_br_id = flit.parent_packet.dst_boundary_router
                    dst_ir_id = mesh.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]

                    if rid == dst_ir_id:
                        continue   # can eject — not a deadlock contributor

                    # ── Determine whether this flit is blocked ────────────
                    if is_escape:
                        nxt_rid = mesh.compute_next_hop_for_packet(
                            rid, dst_ir_id, is_escape=True
                        )
                        if nxt_rid is None:
                            continue
                        out_dir = mesh._direction_to(rid, nxt_rid)
                        if out_dir is None:
                            continue
                        in_port_name = mesh.OPPOSITE_DIR[out_dir]
                        esc     = mesh.routers[nxt_rid].input_ports[in_port_name].escape_vc
                        blocked = esc.allocated_flow_id is not None and not esc.is_empty()
                    elif flit.is_worm_head:
                        # Head can use any minimal direction — blocked only when ALL are occupied.
                        blocked = mesh.all_minimal_hops_blocked(rid, dst_ir_id)
                        nxt_rid = mesh.compute_next_hop_for_packet(rid, dst_ir_id, is_escape=False)
                    else:
                        entry = router.flow_output_table.get(flit.flow_id)
                        if entry is None:
                            continue   # stranded; will be promoted — not a deadlock
                        nxt_rid, use_esc = entry
                        out_dir = mesh._direction_to(rid, nxt_rid)
                        if out_dir is None:
                            continue
                        in_port_name = mesh.OPPOSITE_DIR[out_dir]
                        nxt_port     = mesh.routers[nxt_rid].input_ports[in_port_name]
                        if use_esc:
                            esc     = nxt_port.escape_vc
                            blocked = esc.allocated_flow_id is not None and not esc.is_empty()
                        else:
                            blocked = not any(v.can_accept(flit) for v in nxt_port.normal_vcs)

                    if blocked:
                        # Add one WFG edge for this router.
                        if is_escape:
                            nxt_rid = mesh.compute_next_hop_for_packet(rid, dst_ir_id, is_escape=True)
                        elif not flit.is_worm_head:
                            entry   = router.flow_output_table.get(flit.flow_id)
                            nxt_rid = entry[0] if entry else None
                        if nxt_rid is not None:
                            wait_for[rid] = nxt_rid
                        found_edge = True
                        break   # one edge per router is enough

        return wait_for

    # ── cycle detection on a functional graph ─────────────────────────────────

    def _detect_cycles(
        self,
        wait_for: Dict[str, str],
    ) -> List[List[str]]:
        """
        Find all cycles in the functional wait-for graph in O(|V|) time.

        Because each node has out-degree ≤ 1 (each blocked router waits for
        exactly one next router), we chase each chain until:
          • we reach a node not in wait_for  → chain ends, no cycle, or
          • we revisit a node seen in the *current* chain → cycle found, or
          • we reach a node already fully processed → no new cycle from here.

        Returns a list of cycles, where each cycle is an ordered list of
        router IDs [R0, R1, …, Rk] such that
            R0 waits for R1, R1 waits for R2, …, Rk waits for R0.
        """
        processed: set  = set()   # nodes whose full chain has been explored
        cycles:    List[List[str]] = []

        for start in wait_for:
            if start in processed:
                continue

            # Follow the chain from `start`, recording the path
            path:         List[str]      = []
            path_index:   Dict[str, int] = {}   # node → position in path
            node = start

            while True:
                if node in processed:
                    # Already explored — no new cycle through this chain
                    break

                if node in path_index:
                    # Revisiting a node in the current path → cycle detected
                    cycle_start = path_index[node]
                    cycles.append(path[cycle_start:])
                    break

                path_index[node] = len(path)
                path.append(node)

                if node not in wait_for:
                    # Chain ends: node is not blocked → no cycle from here
                    break

                node = wait_for[node]

            # Mark every node in this path as fully processed
            processed.update(path)

        return cycles

    # ── helper: collect blocked packet IDs ───────────────────────────────────

    def _blocked_packet_ids(
        self,
        mesh:   "InterposerMesh",
        cycles: List[List[str]],
    ) -> List[int]:
        """
        Return the packet_ids of the head packets sitting at each router
        that is identified as part of a deadlock cycle.
        """
        pkt_ids: List[int] = []
        seen_routers: set  = set()
        for cycle_routers in cycles:
            for rid in cycle_routers:
                if rid in seen_routers:
                    continue
                seen_routers.add(rid)
                router = mesh.routers[rid]
                # Find the first non-empty VC head flit across all input ports.
                for port in router.input_ports.values():
                    for vc, _ in port.all_vcs():
                        f = vc.peek()
                        if f is not None:
                            pkt_ids.append(f.parent_packet.packet_id)
                            break
                    else:
                        continue
                    break
        return pkt_ids

    # ── diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of all detected deadlocks."""
        if not self._history:
            return "DeadlockDetector: no deadlocks detected."
        lines = [f"DeadlockDetector: {self._total_deadlocks} deadlock(s) detected."]
        # for r in self._history:
        #     lines.append(
        #         f"  cycle={r.cycle:4d}  method={r.method:<16s}"
        #         f"  cycles={[' → '.join(c) for c in r.router_cycles]}"
        #         f"  stall={r.stall_cycles}"
        #     )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ═══════════════════════════════════════════════════════════════════════════════

COOLDOWN_CYCLES = 10000   # injection-free cycles appended after each run


def run_simulation(
    cycles:             int                        = 200,
    cooldown_cycles:    int                        = COOLDOWN_CYCLES,
    chiplet_specs:      Optional[List[ChipletSpec]] = None,
    cpu_injection_rate: Optional[float]            = None,
    seed:               Optional[int]              = None,
    routing_seed:       Optional[int]              = None,
    drain_escape_entry_prob: float                 = DRAIN_ESCAPE_ENTRY_PROB,
    turn_restricted_escape_entry_prob: float       = TURN_RESTRICTED_ESCAPE_ENTRY_PROB,
    drain_period:       int                        = DRAIN_PERIOD,
    drain_window_hops:  int                        = DRAIN_WINDOW_HOPS,
    full_drain_every_n_windows: int               = FULL_DRAIN_EVERY_N_WINDOWS,
    pre_drain_cycles:   int                        = PRE_DRAIN_CYCLES,
    verbose:            bool                       = False,
) -> Dict[str, Tuple[Dict[str, Any], DeadlockDetector]]:
    """
    Run routing-algorithm comparison scenarios for `cycles` cycles, followed
    by `cooldown_cycles` injection-free cycles so in-flight packets can drain.

    Parameters
    ----------
    cycles             : Number of simulation cycles per scenario (with injection).
    cooldown_cycles    : Extra cycles after injection stops; packets already in
                         the network continue routing until delivered or stuck.
                         Defaults to ``COOLDOWN_CYCLES`` (500).
    chiplet_specs      : List of ChipletSpec objects describing the system.
                         Defaults to ``build_image_layout()`` (4 corner GPUs
                         + 1 central CPU sharing 2 IRs with the top GPUs).
    cpu_injection_rate : Injection probability per cycle for CPU chiplets.
                         GPU rates are scaled from this value per run:
                           burst = 4 * cpu_injection_rate
                           quiet = cpu_injection_rate / 10
    seed               : RNG seed for traffic injection (None → random).
    routing_seed       : RNG seed for routing decisions (None → random).
    drain_escape_entry_prob : Per-hop escape-entry probability used by
                              ``random_adaptive_with_drain``.
    turn_restricted_escape_entry_prob : Per-hop escape-entry probability used by
                                        ``random_adaptive_turn_restricted``.
    drain_period       : DRAIN window triggered every N cycles.
    drain_window_hops  : Synchronized DRAIN hops per window.
    full_drain_every_n_windows : Run a FULL_DRAIN once every N regular drain windows.
    pre_drain_cycles   : Cycles spent freezing new escape admissions before a
                         drain window begins.
    verbose            : Print a line each time a deadlock is detected.

    Returns
    -------
    Dict mapping scenario label → (stats dict, DeadlockDetector).
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()
    effective_cpu_rate = CPU_INJECTION_RATE if cpu_injection_rate is None else cpu_injection_rate
    results: Dict[str, Tuple[Dict[str, Any], DeadlockDetector]] = {}
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 32)
    if routing_seed is None:
        routing_seed = random.SystemRandom().randrange(1 << 32)

    scenarios = [
        # (label,                              routing_mode,              drain,  escape_prob,       num_vcs,        seed_offset)
        ("xy",                                 ROUTING_XY,                False,  0.0,               2,              2),
        ("yx",                                 ROUTING_YX,                False,  0.0,               2,              2),
        ("random_adaptive_without_drain",      ROUTING_RANDOM_ADAPTIVE,   False,  0.0,               2,              2),
        ("random_adaptive_with_drain",         ROUTING_RANDOM_ADAPTIVE,   True,   drain_escape_entry_prob, NUM_NORMAL_VCS, 2),
        ("random_adaptive_turn_restricted",    ROUTING_RANDOM_ADAPTIVE_TR,False,  turn_restricted_escape_entry_prob, NUM_NORMAL_VCS, 2),
        ("shortest_path",                      ROUTING_RANDOM_ADAPTIVE,   False,  0.0,               40,             2),
    ]
    for (
        label,
        routing_mode,
        drain_enabled,
        scenario_escape_entry_prob,
        scenario_num_vcs,
        seed_offset,
    ) in scenarios:
        rng  = random.Random(seed + seed_offset)
        mesh = InterposerMesh(
            chiplet_specs=chiplet_specs,
            routing_mode=routing_mode,
            routing_seed=routing_seed + seed_offset,
            drain_enabled=drain_enabled,
            escape_entry_prob=scenario_escape_entry_prob,
            drain_period=drain_period,
            drain_window_hops=drain_window_hops,
            full_drain_every_n_windows=full_drain_every_n_windows,
            pre_drain_cycles=pre_drain_cycles,
            num_normal_vcs=scenario_num_vcs,
            cpu_injection_rate=effective_cpu_rate,
            gpu_burst_rate=4 * effective_cpu_rate,
            gpu_quiet_rate=effective_cpu_rate / 10.0,
        )

        # Safety check: ensure CPU chiplets are using the intended swept rate.
        for c in mesh.chiplets.values():
            if isinstance(c, CPUChiplet):
                assert abs(c.injection_rate - effective_cpu_rate) < 1e-12
        detector = DeadlockDetector()

        # ── Injection phase ───────────────────────────────────────────────
        for cycle in range(cycles):
            mesh.inject_random_packets(cycle, rng)
            mesh.step(cycle)
            detector.check(mesh, cycle)

        # ── Cooldown phase: no new packets injected ───────────────────────
        for cycle in range(cycles, cycles + cooldown_cycles):
            mesh.step(cycle)
            detector.check(mesh, cycle)

        stats = mesh.delivered_stats()
        chiplet_count = max(1, len(mesh.chiplets))
        stats["throughput"] = round(stats["delivered"] / (cycles * chiplet_count), 6)

        per_type = mesh.delivered_stats_by_type()
        for chiplet_type, type_stats in per_type.items():
            type_chiplet_count = max(1, int(type_stats.get("chiplet_count", 0)))
            type_stats["throughput"] = round(
                type_stats["delivered"] / (cycles * type_chiplet_count), 6
            )
        stats["per_type"] = per_type

        injected_by_chiplet: Dict[str, int] = {name: 0 for name in mesh.chiplets.keys()}
        for br in mesh.boundary_routers.values():
            for pkt in br.injected_packets:
                injected_by_chiplet[pkt.src_chiplet] += 1
        stats["injected_by_chiplet"] = injected_by_chiplet  # type: ignore[index]

        results[label] = (stats, detector)

    return results


SCENARIO_LABELS = (
    "xy",
    "yx",
    "random_adaptive_without_drain",
    "random_adaptive_with_drain",
    "random_adaptive_turn_restricted",
    "shortest_path",
)

SCENARIO_DISPLAY = {
    "xy":                              "XY (deadlock-free)",
    "yx":                              "YX (deadlock-free)",
    "random_adaptive_without_drain":   "Random Adaptive (no DRAIN)",
    "random_adaptive_with_drain":      "Random Adaptive + DRAIN",
    "random_adaptive_turn_restricted": "Random Adaptive + Turn Restrictions",
    "shortest_path":                   "Shortest Path (40 VCs)",
}


def sweep_injection_rates(
    chiplet_specs:      Optional[List[ChipletSpec]] = None,
    cycles:             int   = 5000,
    cooldown_cycles:    int   = COOLDOWN_CYCLES,
    rate_min:           float = 0.10,
    rate_max:           float = 1.00,
    rate_step:          float = 0.05,
    seed:               int   = 42,
    routing_seed:       int   = 0,
    drain_escape_entry_prob: float = DRAIN_ESCAPE_ENTRY_PROB,
    turn_restricted_escape_entry_prob: float = TURN_RESTRICTED_ESCAPE_ENTRY_PROB,
    drain_period:       int   = DRAIN_PERIOD,
    drain_window_hops:  int   = DRAIN_WINDOW_HOPS,
    full_drain_every_n_windows: int = FULL_DRAIN_EVERY_N_WINDOWS,
    pre_drain_cycles:   int   = PRE_DRAIN_CYCLES,
    num_seeds:          int   = 5,
    drain_period_c:     Optional[float] = 0.2,
    drain_period_alpha: float           = 2.0,
) -> Dict[str, Dict[float, Dict[str, int]]]:
    """
    Sweep CPU INJECTION_RATE and collect packet counts per scenario.

    Each rate point is averaged over ``num_seeds`` independent RNG seeds to
    reduce variance (especially important at low injection rates where each run
    injects too few packets for stable single-run statistics).

    Each seed runs ``cycles`` injection cycles followed by ``cooldown_cycles``
    injection-free cycles so in-flight packets can finish.

    drain_period_c     : If set, the drain period for the DRAIN scenario is
                         computed as ``max(1, round(c / rate^alpha))`` at each
                         rate point (steeper inverse: drain much more frequently
                         at high load).  Set to None to use the fixed
                         ``drain_period`` argument instead.
    drain_period_alpha : Exponent for the inverse power law (default 2.0).

    Returns
    -------
    sweep[label][rate] = {"injected": int, "delivered": int, "avg_latency": float}
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()

    rates = [
        round(rate_min + i * rate_step, 10)
        for i in range(int(round((rate_max - rate_min) / rate_step)) + 1)
        if rate_min + i * rate_step <= rate_max + 1e-9
    ]

    sweep: Dict[str, Dict[float, Dict[str, int]]] = {label: {} for label in SCENARIO_LABELS}
    total = len(rates)
    for idx, rate in enumerate(rates):
        effective_drain_period = (
            max(1, round(drain_period_c / (rate ** drain_period_alpha)))
            if drain_period_c is not None
            else drain_period
        )
        # effective_drain_period = drain_period
        print(f"  rate={rate:.6f}, drain_period={effective_drain_period} ({idx + 1}/{total})", flush=True)

        # Accumulate totals across multiple seeds, then average.
        accum: Dict[str, Dict[str, float]] = {
            label: {"injected": 0.0, "delivered": 0.0, "failed_injections": 0.0, "avg_latency": 0.0}
            for label in SCENARIO_LABELS
        }
        for seed_offset in range(num_seeds):
            results = run_simulation(
                cycles=cycles,
                cooldown_cycles=cooldown_cycles,
                chiplet_specs=chiplet_specs,
                cpu_injection_rate=rate,
                seed=seed + seed_offset * 1000,
                routing_seed=routing_seed + seed_offset * 1000,
                drain_escape_entry_prob=drain_escape_entry_prob,
                turn_restricted_escape_entry_prob=turn_restricted_escape_entry_prob,
                drain_period=effective_drain_period,
                drain_window_hops=drain_window_hops,
                full_drain_every_n_windows=full_drain_every_n_windows,
                pre_drain_cycles=pre_drain_cycles,
            )
            for label in SCENARIO_LABELS:
                stats = results[label][0]
                accum[label]["injected"]          += stats["injected"]
                accum[label]["delivered"]         += stats["delivered"]
                accum[label]["failed_injections"] += stats["failed_injections"]
                accum[label]["avg_latency"]       += float(stats["avg_latency"])

        for label in SCENARIO_LABELS:
            sweep[label][rate] = {
                "injected":          int(round(accum[label]["injected"]          / num_seeds)),
                "delivered":         int(round(accum[label]["delivered"]         / num_seeds)),
                "failed_injections": int(round(accum[label]["failed_injections"] / num_seeds)),
                "avg_latency":       accum[label]["avg_latency"] / num_seeds,
            }
    return sweep


def sweep_drain_window(
    chiplet_specs:      Optional[List[ChipletSpec]] = None,
    cycles:             int   = 100_000,
    cooldown_cycles:    int   = COOLDOWN_CYCLES,
    cpu_injection_rate: float = 0.01,
    drain_period_min:   int   = 100,
    drain_period_max:   int   = 5000,
    drain_period_step:  int   = 100,
    drain_window_hops:  int   = DRAIN_WINDOW_HOPS,
    full_drain_every_n_windows: int = FULL_DRAIN_EVERY_N_WINDOWS,
    pre_drain_cycles:   int   = PRE_DRAIN_CYCLES,
    drain_escape_entry_prob: float = DRAIN_ESCAPE_ENTRY_PROB,
    seed:               int   = 42,
    routing_seed:       int   = 0,
    num_seeds:          int   = 3,
) -> Dict[int, Dict[str, int]]:
    """
    Sweep drain_period for the ``random_adaptive_with_drain`` scenario only.

    Returns
    -------
    result[drain_period] = {"injected": int, "delivered": int}
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()

    periods = list(range(drain_period_min, drain_period_max + 1, drain_period_step))
    result: Dict[int, Dict[str, int]] = {}
    total = len(periods)

    for idx, period in enumerate(periods):
        print(f"  drain_period={period} ({idx + 1}/{total})", flush=True)
        accum = {"injected": 0.0, "delivered": 0.0, "failed_injections": 0.0, "avg_latency": 0.0}

        for seed_offset in range(num_seeds):
            rng = random.Random(seed + seed_offset * 1000)
            mesh = InterposerMesh(
                chiplet_specs=chiplet_specs,
                routing_mode=ROUTING_RANDOM_ADAPTIVE,
                routing_seed=routing_seed + seed_offset * 1000,
                drain_enabled=True,
                escape_entry_prob=drain_escape_entry_prob,
                drain_period=period,
                drain_window_hops=drain_window_hops,
                full_drain_every_n_windows=full_drain_every_n_windows,
                pre_drain_cycles=pre_drain_cycles,
                num_normal_vcs=NUM_NORMAL_VCS,
                cpu_injection_rate=cpu_injection_rate,
                gpu_burst_rate=4 * cpu_injection_rate,
                gpu_quiet_rate=cpu_injection_rate / 10.0,
            )
            detector = DeadlockDetector()

            for cycle in range(cycles):
                mesh.inject_random_packets(cycle, rng)
                mesh.step(cycle)
                detector.check(mesh, cycle)

            for cycle in range(cycles, cycles + cooldown_cycles):
                mesh.step(cycle)
                detector.check(mesh, cycle)

            stats = mesh.delivered_stats()
            accum["injected"]          += int(stats["injected"])
            accum["delivered"]         += int(stats["delivered"])
            accum["failed_injections"] += int(stats["failed_injections"])
            accum["avg_latency"]       += float(stats["avg_latency"])

        result[period] = {
            "injected":          int(round(accum["injected"]          / num_seeds)),
            "delivered":         int(round(accum["delivered"]         / num_seeds)),
            "failed_injections": int(round(accum["failed_injections"] / num_seeds)),
            "avg_latency":       accum["avg_latency"] / num_seeds,
        }

    return result


def plot_drain_window_sweep(
    result:     Dict[int, Dict[str, int]],
    out_prefix: str  = "drain_window_sweep",
    show:       bool = True,
) -> None:
    """
    Plot drain_period vs # packets injected and # packets delivered
    for the random_adaptive_with_drain scenario.
    """
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    periods     = sorted(result.keys())
    injected    = [result[p]["injected"]    for p in periods]
    delivered   = [result[p]["delivered"]   for p in periods]
    avg_latency = [result[p].get("avg_latency", 0.0) for p in periods]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    ax1.plot(periods, injected,  marker="o", linewidth=2, color="steelblue",
             label="Injected")
    ax1.plot(periods, delivered, marker="s", linewidth=2, linestyle="--",
             color="darkorange", label="Delivered")
    ax1.set_ylabel("# Packets")
    ax1.set_title("Random Adaptive + DRAIN\nInjected & Delivered vs DRAIN Window Period")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(periods, avg_latency, marker="^", linewidth=2, color="seagreen",
             label="Avg Latency")
    ax2.set_xlabel("DRAIN Period (cycles between DRAIN windows)")
    ax2.set_ylabel("Avg Packet Latency (cycles)")
    ax2.set_title("Average Latency vs DRAIN Window Period")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()

    out_file = f"{out_prefix}.png"
    fig.savefig(out_file, dpi=150)
    print(f"  Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def sweep_full_drain_window(
    chiplet_specs:      Optional[List[ChipletSpec]] = None,
    cycles:             int   = 20000,
    cooldown_cycles:    int   = COOLDOWN_CYCLES,
    cpu_injection_rate: float = 0.01,
    full_drain_every_values: Optional[List[int]] = None,
    drain_period:       int   = DRAIN_PERIOD,
    drain_window_hops:  int   = DRAIN_WINDOW_HOPS,
    pre_drain_cycles:   int   = PRE_DRAIN_CYCLES,
    drain_escape_entry_prob: float = DRAIN_ESCAPE_ENTRY_PROB,
    seed:               int   = 42,
    routing_seed:       int   = 0,
    num_seeds:          int   = 3,
) -> Dict[int, Dict[str, int]]:
    """
    Sweep FULL_DRAIN_EVERY_N_WINDOWS for the ``random_adaptive_with_drain``
    scenario only.

    Returns
    -------
    result[full_drain_every_n_windows] = {"injected": int, "delivered": int}
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()
    if full_drain_every_values is None:
        full_drain_every_values = [5, 10, 20, 50]

    values = [max(1, int(v)) for v in full_drain_every_values]
    result: Dict[int, Dict[str, int]] = {}
    total = len(values)

    for idx, every_n in enumerate(values):
        print(f"  full_drain_every_n_windows={every_n} ({idx + 1}/{total})", flush=True)
        accum = {"injected": 0.0, "delivered": 0.0, "failed_injections": 0.0, "avg_latency": 0.0}

        for seed_offset in range(num_seeds):
            rng = random.Random(seed + seed_offset * 1000)
            mesh = InterposerMesh(
                chiplet_specs=chiplet_specs,
                routing_mode=ROUTING_RANDOM_ADAPTIVE,
                routing_seed=routing_seed + seed_offset * 1000,
                drain_enabled=True,
                escape_entry_prob=drain_escape_entry_prob,
                drain_period=drain_period,
                drain_window_hops=drain_window_hops,
                full_drain_every_n_windows=every_n,
                pre_drain_cycles=pre_drain_cycles,
                num_normal_vcs=NUM_NORMAL_VCS,
                cpu_injection_rate=cpu_injection_rate,
                gpu_burst_rate=4 * cpu_injection_rate,
                gpu_quiet_rate=cpu_injection_rate / 10.0,
            )
            detector = DeadlockDetector()

            for cycle in range(cycles):
                mesh.inject_random_packets(cycle, rng)
                mesh.step(cycle)
                detector.check(mesh, cycle)

            for cycle in range(cycles, cycles + cooldown_cycles):
                mesh.step(cycle)
                detector.check(mesh, cycle)

            stats = mesh.delivered_stats()
            accum["injected"]          += int(stats["injected"])
            accum["delivered"]         += int(stats["delivered"])
            accum["failed_injections"] += int(stats["failed_injections"])
            accum["avg_latency"]       += float(stats["avg_latency"])

        result[every_n] = {
            "injected":          int(round(accum["injected"]          / num_seeds)),
            "delivered":         int(round(accum["delivered"]         / num_seeds)),
            "failed_injections": int(round(accum["failed_injections"] / num_seeds)),
            "avg_latency":       accum["avg_latency"] / num_seeds,
        }

    return result


def plot_full_drain_window_sweep(
    result:     Dict[int, Dict[str, int]],
    out_prefix: str  = "full_drain_window_sweep",
    show:       bool = True,
) -> None:
    """
    Plot FULL_DRAIN_EVERY_N_WINDOWS vs # packets injected and # packets delivered
    for the random_adaptive_with_drain scenario.
    """
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    values      = sorted(result.keys())
    injected    = [result[v]["injected"]    for v in values]
    delivered   = [result[v]["delivered"]   for v in values]
    avg_latency = [result[v].get("avg_latency", 0.0) for v in values]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    ax1.plot(values, injected,  marker="o", linewidth=2, color="steelblue",
             label="Injected")
    ax1.plot(values, delivered, marker="s", linewidth=2, linestyle="--",
             color="darkorange", label="Delivered")
    ax1.set_ylabel("# Packets")
    ax1.set_title("Random Adaptive + DRAIN\nInjected & Delivered vs FULL_DRAIN_EVERY_N_WINDOWS")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(values, avg_latency, marker="^", linewidth=2, color="seagreen",
             label="Avg Latency")
    ax2.set_xlabel("FULL_DRAIN_EVERY_N_WINDOWS")
    ax2.set_ylabel("Avg Packet Latency (cycles)")
    ax2.set_title("Average Latency vs FULL_DRAIN_EVERY_N_WINDOWS")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()

    out_file = f"{out_prefix}.png"
    fig.savefig(out_file, dpi=150)
    print(f"  Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def sweep_escape_prob(
    chiplet_specs:       Optional[List[ChipletSpec]] = None,
    cycles:              int   = 50_000,
    cooldown_cycles:     int   = COOLDOWN_CYCLES,
    cpu_injection_rate:  float = CPU_INJECTION_RATE,
    prob_min:            float = 0.0,
    prob_max:            float = 1.0,
    prob_step:           float = 0.05,
    drain_period:        int   = DRAIN_PERIOD,
    drain_window_hops:   int   = DRAIN_WINDOW_HOPS,
    pre_drain_cycles:    int   = PRE_DRAIN_CYCLES,
    seed:                int   = 42,
    routing_seed:        int   = 0,
    num_seeds:           int   = 3,
) -> Dict[float, Dict[str, int]]:
    """
    Sweep drain_escape_entry_prob for the ``random_adaptive_with_drain`` scenario.

    Returns
    -------
    result[prob] = {"injected": int, "delivered": int}
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()

    probs = [
        round(prob_min + i * prob_step, 10)
        for i in range(int(round((prob_max - prob_min) / prob_step)) + 1)
        if prob_min + i * prob_step <= prob_max + 1e-9
    ]
    result: Dict[float, Dict[str, int]] = {}
    total = len(probs)

    for idx, prob in enumerate(probs):
        print(f"  escape_entry_prob={prob:.2f} ({idx + 1}/{total})", flush=True)
        accum = {"injected": 0.0, "delivered": 0.0, "failed_injections": 0.0, "avg_latency": 0.0}

        for seed_offset in range(num_seeds):
            rng = random.Random(seed + seed_offset * 1000)
            mesh = InterposerMesh(
                chiplet_specs=chiplet_specs,
                routing_mode=ROUTING_RANDOM_ADAPTIVE,
                routing_seed=routing_seed + seed_offset * 1000,
                drain_enabled=True,
                escape_entry_prob=prob,
                drain_period=drain_period,
                drain_window_hops=drain_window_hops,
                pre_drain_cycles=pre_drain_cycles,
                num_normal_vcs=NUM_NORMAL_VCS,
                cpu_injection_rate=cpu_injection_rate,
                gpu_burst_rate=4 * cpu_injection_rate,
                gpu_quiet_rate=cpu_injection_rate / 10.0,
            )
            detector = DeadlockDetector()

            for cycle in range(cycles):
                mesh.inject_random_packets(cycle, rng)
                mesh.step(cycle)
                detector.check(mesh, cycle)

            for cycle in range(cycles, cycles + cooldown_cycles):
                mesh.step(cycle)
                detector.check(mesh, cycle)

            stats = mesh.delivered_stats()
            accum["injected"]          += int(stats["injected"])
            accum["delivered"]         += int(stats["delivered"])
            accum["failed_injections"] += int(stats["failed_injections"])
            accum["avg_latency"]       += float(stats["avg_latency"])

        result[prob] = {
            "injected":          int(round(accum["injected"]          / num_seeds)),
            "delivered":         int(round(accum["delivered"]         / num_seeds)),
            "failed_injections": int(round(accum["failed_injections"] / num_seeds)),
            "avg_latency":       accum["avg_latency"] / num_seeds,
        }

    return result


def plot_escape_prob_sweep(
    result:     Dict[float, Dict[str, int]],
    out_prefix: str  = "escape_prob_sweep",
    show:       bool = True,
) -> None:
    """
    Plot drain_escape_entry_prob vs # packets injected and # packets delivered
    for the random_adaptive_with_drain scenario.
    """
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    probs       = sorted(result.keys())
    injected    = [result[p]["injected"]    for p in probs]
    delivered   = [result[p]["delivered"]   for p in probs]
    avg_latency = [result[p].get("avg_latency", 0.0) for p in probs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    ax1.plot(probs, injected,  marker="o", linewidth=2, color="steelblue",
             label="Injected")
    ax1.plot(probs, delivered, marker="s", linewidth=2, linestyle="--",
             color="darkorange", label="Delivered")
    ax1.set_ylabel("# Packets")
    ax1.set_title("Random Adaptive + DRAIN\nInjected & Delivered vs Escape Entry Probability")
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(probs, avg_latency, marker="^", linewidth=2, color="seagreen",
             label="Avg Latency")
    ax2.set_xlabel("Escape Entry Probability")
    ax2.set_ylabel("Avg Packet Latency (cycles)")
    ax2.set_title("Average Latency vs Escape Entry Probability")
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()

    out_file = f"{out_prefix}.png"
    fig.savefig(out_file, dpi=150)
    print(f"  Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def plot_injection_sweep(
    sweep:      Dict[str, Dict[float, Dict[str, int]]],
    out_prefix: str  = "injection_rate_sweep",
    show:       bool = True,
) -> None:
    """
    Plot INJECTION_RATE vs packet counts.

    Each figure shows injected, delivered, and failed_injections curves so
    acceptance and rejection behavior are visible directly.

    Outputs per call:
      - one figure per scenario  (<prefix>_<label>.png)
      - combined random-adaptive comparison  (<prefix>_random_adaptive_comparison.png)
      - random_adaptive_with_drain vs XY     (<prefix>_random_adaptive_with_drain_vs_xy.png)
      - random_adaptive_with_drain vs YX     (<prefix>_random_adaptive_with_drain_vs_yx.png)
      - multi-scenario comparison            (<prefix>_scenario_comparison.png)
    """
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    # ── helper: extract injected and delivered series for one label ───────────
    def _series(label: str) -> Tuple[List[float], List[int], List[int], List[int]]:
        rates     = sorted(sweep[label].keys())
        injected  = [sweep[label][r]["injected"]  for r in rates]
        delivered = [sweep[label][r]["delivered"] for r in rates]
        failed    = [sweep[label][r]["failed_injections"] for r in rates]
        return rates, injected, delivered, failed

    # Use exact sweep bounds for every figure's x-axis.
    all_sweep_rates = [r for label_data in sweep.values() for r in label_data.keys()]
    x_min = min(all_sweep_rates)
    x_max = max(all_sweep_rates)

    # ── per-scenario figures ──────────────────────────────────────────────────
    for label in SCENARIO_LABELS:
        rates, injected, delivered, failed = _series(label)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rates, injected,  marker="o", linewidth=2,
                color="steelblue",  label="Injected")
        ax.plot(rates, delivered, marker="s", linewidth=2, linestyle="--",
                color="darkorange", label="Delivered")
        ax.plot(rates, failed, marker="^", linewidth=2, linestyle="-.",
                color="firebrick", label="Failed Injections")
        ax.set_xlabel("INJECTION_RATE")
        ax.set_ylabel("# Packets")
        ax.set_title(
            f"{SCENARIO_DISPLAY.get(label, label)}\n"
            "Injected, Delivered & Failed Injections vs INJECTION_RATE"
        )
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        out_file = f"{out_prefix}_{label}.png"
        fig.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")
        if show:
            plt.show()
        plt.close(fig)

    # ── helper: two-scenario comparison figure ────────────────────────────────
    def _plot_pair(a: str, b: str, suffix: str, title: str) -> None:
        if a not in sweep or b not in sweep:
            return
        rates_a, inj_a, del_a, fail_a = _series(a)
        rates_b, inj_b, del_b, fail_b = _series(b)

        fig, ax = plt.subplots(figsize=(8, 5))
        disp_a = SCENARIO_DISPLAY.get(a, a)
        disp_b = SCENARIO_DISPLAY.get(b, b)
        ax.plot(rates_a, inj_a, marker="o", linewidth=2,
                color="darkorange", label=f"{disp_a} – Injected")
        ax.plot(rates_a, del_a, marker="o", linewidth=2, linestyle="--",
                color="darkorange", label=f"{disp_a} – Delivered")
        ax.plot(rates_a, fail_a, marker="o", linewidth=2, linestyle="-.",
                color="darkorange", alpha=0.8, label=f"{disp_a} – Failed")
        ax.plot(rates_b, inj_b, marker="s", linewidth=2,
                color="steelblue",  label=f"{disp_b} – Injected")
        ax.plot(rates_b, del_b, marker="s", linewidth=2, linestyle="--",
                color="steelblue",  label=f"{disp_b} – Delivered")
        ax.plot(rates_b, fail_b, marker="s", linewidth=2, linestyle="-.",
                color="steelblue", alpha=0.8, label=f"{disp_b} – Failed")
        ax.set_xlabel("INJECTION_RATE")
        ax.set_ylabel("# Packets")
        ax.set_title(f"{title}\nInjected, Delivered & Failed vs INJECTION_RATE")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_file = f"{out_prefix}_{suffix}.png"
        fig.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")
        if show:
            plt.show()
        plt.close(fig)

    # Combined random-adaptive comparison
    _plot_pair(
        "random_adaptive_without_drain",
        "random_adaptive_with_drain",
        "random_adaptive_comparison",
        "Random Adaptive Comparison",
    )
    _plot_pair(
        "random_adaptive_with_drain",
        "xy",
        "random_adaptive_with_drain_vs_xy",
        "Random Adaptive + DRAIN vs XY",
    )
    _plot_pair(
        "random_adaptive_with_drain",
        "yx",
        "random_adaptive_with_drain_vs_yx",
        "Random Adaptive + DRAIN vs YX",
    )

    # Multi-scenario comparison split into two graphs:
    #  1) injected only
    #  2) delivered only
    combo_labels = [
        "random_adaptive_with_drain",
        "random_adaptive_without_drain",
        "yx",
        "random_adaptive_turn_restricted",
        "shortest_path",
    ]
    if all(label in sweep for label in combo_labels):
        colors  = ["darkorange", "steelblue", "seagreen", "slategray", "firebrick"]
        markers = ["o", "s", "^", "x", "D"]
        fig_inj, ax_inj = plt.subplots(figsize=(9, 6))
        fig_del, ax_del = plt.subplots(figsize=(9, 6))
        fig_fail, ax_fail = plt.subplots(figsize=(9, 6))

        for label, color, marker in zip(combo_labels, colors, markers):
            rates, injected, delivered, failed = _series(label)
            disp = SCENARIO_DISPLAY.get(label, label)
            ax_inj.plot(
                rates,
                injected,
                marker=marker,
                linewidth=2,
                color=color,
                label=disp,
            )
            ax_fail.plot(
                rates,
                failed,
                marker=marker,
                linewidth=2,
                color=color,
                label=disp,
            )
            ax_del.plot(
                rates,
                delivered,
                marker=marker,
                linewidth=2,
                color=color,
                label=disp,
            )

        ax_inj.set_xlabel("INJECTION_RATE")
        ax_inj.set_ylabel("# Packets Injected")
        ax_inj.set_title("Scenario Comparison\nInjected vs INJECTION_RATE")
        ax_inj.set_xlim(x_min, x_max)
        ax_inj.grid(True, linestyle="--", alpha=0.5)
        ax_inj.legend(fontsize=9)
        fig_inj.tight_layout()

        out_inj = f"{out_prefix}_scenario_comparison_injected.png"
        fig_inj.savefig(out_inj, dpi=150)
        print(f"  Saved: {out_inj}")
        if show:
            plt.show()
        plt.close(fig_inj)

        ax_del.set_xlabel("INJECTION_RATE")
        ax_del.set_ylabel("# Packets Delivered")
        ax_del.set_title("Scenario Comparison\nDelivered vs INJECTION_RATE")
        ax_del.set_xlim(x_min, x_max)
        ax_del.grid(True, linestyle="--", alpha=0.5)
        ax_del.legend(fontsize=9)
        fig_del.tight_layout()

        out_del = f"{out_prefix}_scenario_comparison_delivered.png"
        fig_del.savefig(out_del, dpi=150)
        print(f"  Saved: {out_del}")
        if show:
            plt.show()
        plt.close(fig_del)

        ax_fail.set_xlabel("INJECTION_RATE")
        ax_fail.set_ylabel("# Failed Injections")
        ax_fail.set_title("Scenario Comparison\nFailed Injections vs INJECTION_RATE")
        ax_fail.set_xlim(x_min, x_max)
        ax_fail.grid(True, linestyle="--", alpha=0.5)
        ax_fail.legend(fontsize=9)
        fig_fail.tight_layout()

        out_fail = f"{out_prefix}_scenario_comparison_failed_injections.png"
        fig_fail.savefig(out_fail, dpi=150)
        print(f"  Saved: {out_fail}")
        if show:
            plt.show()
        plt.close(fig_fail)


def plot_latency_sweep_all_scenarios(
    sweep:      Dict[str, Dict[float, Dict[str, int]]],
    out_prefix: str  = "injection_rate_sweep",
    show:       bool = True,
) -> None:
    """Plot INJECTION_RATE vs average packet latency for all scenarios."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    all_sweep_rates = [r for label_data in sweep.values() for r in label_data.keys()]
    x_min = min(all_sweep_rates)
    x_max = max(all_sweep_rates)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["mediumpurple", "seagreen", "steelblue", "darkorange", "slategray", "firebrick"]
    markers = ["P", "^", "s", "o", "x", "D"]
    for label, color, marker in zip(SCENARIO_LABELS, colors, markers):
        if label not in sweep:
            continue
        rates = sorted(sweep[label].keys())
        lat = [float(sweep[label][r]["avg_latency"]) for r in rates]
        ax.plot(rates, lat, marker=marker, linewidth=2, color=color, label=SCENARIO_DISPLAY.get(label, label))

    ax.set_xlabel("INJECTION_RATE")
    ax.set_ylabel("Avg Packet Latency (cycles)")
    ax.set_title("Scenario Comparison\nAvg Packet Latency vs INJECTION_RATE")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_file = f"{out_prefix}_latency_all_scenarios.png"
    fig.savefig(out_file, dpi=150)
    print(f"  Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def sweep_fault_count(
    chiplet_specs:       Optional[List[ChipletSpec]] = None,
    fault_counts:        List[int]  = None,
    cycles:              int   = 300_000,
    cooldown_cycles:     int   = COOLDOWN_CYCLES,
    cpu_injection_rate:  float = 0.002,
    drain_period:        int   = 500,
    drain_window_hops:   int   = DRAIN_WINDOW_HOPS,
    pre_drain_cycles:    int   = PRE_DRAIN_CYCLES,
    escape_entry_prob:   float = 0.1,
    seed:                int   = 42,
    routing_seed:        int   = 0,
    num_seeds:           int   = 3,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Sweep number of randomly-injected link faults on the interposer mesh and
    measure average packet latency (low-load regime).

    Two fault-tolerant scenarios are tested:
      - ``random_adaptive_with_drain``      (escape-VC + periodic DRAIN windows)
      - ``random_adaptive_turn_restricted`` (West-First turn-restricted routing)

    After each fault set is placed, the drain Hamiltonian cycle is recomputed
    on the surviving topology so DRAIN windows can still make progress wherever
    connectivity allows.  Adaptive routing automatically avoids severed links.

    Parameters
    ----------
    fault_counts        : list of fault counts to sweep, e.g. [0, 1, 4, 8, 12].
    cycles              : injection cycles per run (300 000 by default).
    cooldown_cycles     : drain-only cycles after injection stops.
    cpu_injection_rate  : per-cycle Bernoulli injection rate (low-load: 0.002).
    drain_period        : cycles between DRAIN windows (500).
    drain_window_hops   : escape-VC hops per DRAIN window.
    pre_drain_cycles    : cycles spent freezing new escape admissions before a
                          drain window begins.
    escape_entry_prob   : probability a packet enters the escape VC per hop (0.1).
    seed                : base RNG seed for traffic and fault placement.
    routing_seed        : base RNG seed for routing decisions.
    num_seeds           : independent seeds to average over.

    Returns
    -------
    result[scenario][num_faults] = {
        "avg_latency": float,
        "delivered":   int,
        "injected":    int,
    }
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()
    if fault_counts is None:
        fault_counts = [0, 1, 4, 8, 12]

    scenarios = [
        ("random_adaptive_with_drain",       ROUTING_RANDOM_ADAPTIVE,    True,  escape_entry_prob),
        ("random_adaptive_turn_restricted",  ROUTING_RANDOM_ADAPTIVE_TR, False, escape_entry_prob),
    ]

    result: Dict[str, Dict[int, Dict[str, float]]] = {
        label: {} for label, _, _, _ in scenarios
    }
    total = len(fault_counts) * len(scenarios)
    done  = 0

    for num_faults in fault_counts:
        for label, routing_mode, drain_enabled, esc_prob in scenarios:
            done += 1
            print(
                f"  fault_count={num_faults}, scenario={label} ({done}/{total})",
                flush=True,
            )
            accum = {"injected": 0.0, "delivered": 0.0, "avg_latency": 0.0}

            for seed_offset in range(num_seeds):
                fault_rng   = random.Random(seed + seed_offset * 1000)
                traffic_rng = random.Random(seed + seed_offset * 1000 + 1)

                mesh = InterposerMesh(
                    chiplet_specs=chiplet_specs,
                    routing_mode=routing_mode,
                    routing_seed=routing_seed + seed_offset * 1000,
                    drain_enabled=drain_enabled,
                    escape_entry_prob=esc_prob,
                    drain_period=drain_period,
                    drain_window_hops=drain_window_hops,
                    pre_drain_cycles=pre_drain_cycles,
                    num_normal_vcs=NUM_NORMAL_VCS,
                    cpu_injection_rate=cpu_injection_rate,
                    gpu_burst_rate=4 * cpu_injection_rate,
                    gpu_quiet_rate=cpu_injection_rate / 10.0,
                )

                if num_faults > 0:
                    mesh.inject_faults(num_faults, fault_rng)

                detector = DeadlockDetector()

                for cycle in range(cycles):
                    mesh.inject_random_packets(cycle, traffic_rng)
                    mesh.step(cycle)
                    detector.check(mesh, cycle)

                for cycle in range(cycles, cycles + cooldown_cycles):
                    mesh.step(cycle)
                    detector.check(mesh, cycle)

                stats = mesh.delivered_stats()
                accum["injected"]    += int(stats["injected"])
                accum["delivered"]   += int(stats["delivered"])
                accum["avg_latency"] += float(stats["avg_latency"])

            result[label][num_faults] = {
                "injected":    int(round(accum["injected"]    / num_seeds)),
                "delivered":   int(round(accum["delivered"]   / num_seeds)),
                "avg_latency": accum["avg_latency"] / num_seeds,
            }

    return result


def plot_fault_sweep(
    result:     Dict[str, Dict[int, Dict[str, float]]],
    out_prefix: str  = "fault_sweep",
    show:       bool = True,
) -> None:
    """
    Plot number of faulty links vs average packet latency for the drain and
    turn-restricted scenarios.

    Outputs
    -------
    <out_prefix>_latency.png   – latency comparison for both scenarios
    <out_prefix>_delivered.png – delivered packet count comparison
    """
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    scenario_style = {
        "random_adaptive_with_drain": {
            "label":  "Random Adaptive + DRAIN",
            "color":  "darkorange",
            "marker": "o",
        },
        "random_adaptive_turn_restricted": {
            "label":  "Random Adaptive Turn-Restricted",
            "color":  "steelblue",
            "marker": "s",
        },
    }

    # ── latency figure ────────────────────────────────────────────────────────
    fig_lat, ax_lat = plt.subplots(figsize=(9, 6))
    for scenario, style in scenario_style.items():
        if scenario not in result:
            continue
        faults  = sorted(result[scenario].keys())
        latency = [result[scenario][f]["avg_latency"] for f in faults]
        ax_lat.plot(
            faults, latency,
            marker=style["marker"], linewidth=2, color=style["color"],
            label=style["label"],
        )
    ax_lat.set_xlabel("Number of Faulty Links")
    ax_lat.set_ylabel("Avg Packet Latency (cycles)")
    ax_lat.set_title("Low-Load Avg Latency vs Number of Faulty Links\n"
                     "(injection_rate=0.002, drain_period=500, escape_prob=0.1)")
    ax_lat.grid(True, linestyle="--", alpha=0.5)
    ax_lat.legend(fontsize=10)
    fig_lat.tight_layout()
    out_lat = f"{out_prefix}_latency.png"
    fig_lat.savefig(out_lat, dpi=150)
    print(f"  Saved: {out_lat}")
    if show:
        plt.show()
    plt.close(fig_lat)

    # ── delivered figure ──────────────────────────────────────────────────────
    fig_del, ax_del = plt.subplots(figsize=(9, 6))
    for scenario, style in scenario_style.items():
        if scenario not in result:
            continue
        faults    = sorted(result[scenario].keys())
        delivered = [result[scenario][f]["delivered"] for f in faults]
        ax_del.plot(
            faults, delivered,
            marker=style["marker"], linewidth=2, color=style["color"],
            label=style["label"],
        )
    ax_del.set_xlabel("Number of Faulty Links")
    ax_del.set_ylabel("# Packets Delivered")
    ax_del.set_title("Packets Delivered vs Number of Faulty Links\n"
                     "(injection_rate=0.002, drain_period=500, escape_prob=0.1)")
    ax_del.grid(True, linestyle="--", alpha=0.5)
    ax_del.legend(fontsize=10)
    fig_del.tight_layout()
    out_del = f"{out_prefix}_delivered.png"
    fig_del.savefig(out_del, dpi=150)
    print(f"  Saved: {out_del}")
    if show:
        plt.show()
    plt.close(fig_del)


if __name__ == "__main__":
    LABELS = SCENARIO_LABELS

    # ── Image layout: 4 corner GPUs + 1 central CPU (sharing IRs with top GPUs)
    specs = build_image_layout()
    print("=" * 70)
    print("Image layout: 4 corner GPU chiplets + 1 central CPU chiplet")
    print("  GPU_TL: 4 BRs → IR(0,0)(0,1)(1,0)(1,1)")
    print("  GPU_TR: 4 BRs → IR(0,2)(0,3)(1,2)(1,3)")
    print("  GPU_BL: 4 BRs → IR(2,0)(2,1)(3,0)(3,1)")
    print("  GPU_BR: 4 BRs → IR(2,2)(2,3)(3,2)(3,3)")
    print("  CPU:    2 BRs → IR(1,1) [shared w/ GPU_TL], IR(1,2) [shared w/ GPU_TR]")
    print("  Interposer: 4×4 = 16 IRs (unchanged)")
    print("=" * 70)
    # results = run_simulation(cycles=300000, chiplet_specs=specs)
    # for label in LABELS:
    #     stats, det = results[label]
    #     print(f"\n[{label}] \ninjected={stats['injected']}  "
    #           f"delivered={stats['delivered']}  "
    #           f"avg_latency={stats['avg_latency']}  "
    #           f"avg_hops={stats['avg_hops']}  "
    #           f"failed_injections={stats['failed_injections']}")
    #     injected_by_chiplet = stats.get("injected_by_chiplet", {})
    #     if isinstance(injected_by_chiplet, dict):
    #         print("  Injected by chiplet:")
    #         for chiplet_name in sorted(injected_by_chiplet.keys()):
    #             print(f"    {chiplet_name}: {injected_by_chiplet[chiplet_name]}")
    #     print(f"  {det.summary()}")

    print("Sweeping INJECTION_RATE from 0.05 to 0.50 (step 0.025, averaged over 5 seeds) ...")
    sweep = sweep_injection_rates(
        chiplet_specs=specs,
        cycles=20000,
        rate_min=0.01,
        rate_max=0.031,
        rate_step=0.005,
        seed=42,
        routing_seed=0,
        num_seeds=3,
    )
    print("Generating sweep plots ...")
    plot_injection_sweep(sweep, out_prefix="injection_rate_sweep", show=False)
    plot_latency_sweep_all_scenarios(sweep, out_prefix="injection_rate_sweep", show=False)
    print()


    # print("Sweeping DRAIN window period from 100 to 5000 (step 100) ...")
    # drain_sweep = sweep_drain_window(
    #     chiplet_specs=specs,
    #     cycles=300000,
    #     cpu_injection_rate=0.006,
    #     drain_period_min=0,
    #     drain_period_max=10000,
    #     drain_period_step=1000,
    #     seed=42,
    #     routing_seed=0,
    #     num_seeds=5,
    # )
    # print("Generating DRAIN window sweep plot ...")
    # plot_drain_window_sweep(drain_sweep, out_prefix="drain_window_sweep", show=False)
    # print()
    
    # print("Sweeping FULL_DRAIN_EVERY_N_WINDOWS")
    # full_drain_sweep = sweep_full_drain_window(
    #     chiplet_specs=specs,
    #     cycles=200000,
    #     cpu_injection_rate=0.01,
    #     full_drain_every_values= [5, 10, 20, 50],
    #     drain_period=80,
    #     drain_window_hops=1,
    #     pre_drain_cycles=2,
    #     seed=42,
    #     routing_seed=0,
    #     num_seeds=2,
    # )
    # print("Generating FULL DRAIN window sweep plot ...")
    # plot_full_drain_window_sweep(full_drain_sweep, out_prefix="full_drain_window_sweep", show=False)
    # print()

    # print("Sweeping drain_escape_entry_prob from 0.0 to 1.0 (step 0.05) ...")
    # escape_prob_sweep = sweep_escape_prob(
    #     chiplet_specs=specs,
    #     cycles=1000000,
    #     prob_min=0.0,
    #     prob_max=.5,
    #     prob_step=0.05,
    #     seed=42,
    #     routing_seed=0,
    #     num_seeds=5,
    # )
    # print("Generating escape probability sweep plot ...")
    # plot_escape_prob_sweep(escape_prob_sweep, out_prefix="escape_prob_sweep", show=False)
    # print()

    # print("Sweeping faulty link counts [0, 1, 4, 8, 12] ...")
    # fault_result = sweep_fault_count(
    #     chiplet_specs=specs,
    #     fault_counts=[0, 1, 2, 4, 6, 8, 12],
    #     cycles=300000,
    #     cpu_injection_rate=0.001,
    #     drain_period=500,
    #     escape_entry_prob=0.1,
    #     seed=42,
    #     routing_seed=0,
    #     num_seeds=5,
    # )
    # print("Generating fault sweep plots ...")
    # plot_fault_sweep(fault_result, out_prefix="fault_sweep", show=False)
    # print()
