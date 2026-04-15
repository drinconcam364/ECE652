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
stagger_drain = 1
DRAIN_WINDOW_HOPS  = 1
PRE_DRAIN_CYCLES   = 1 # FLITS_PER_PACKET # paper says -  maximum packet size supported in the networ
FULL_DRAIN_EVERY_N_WINDOWS = 20   # perform a full drain once every N regular drain windows
STRICT_REGULAR_DRAIN = False      # False = escape-first priority drain, True = escape-only regular drain
INJECTION_RATE     = .025#0.006

COOLDOWN_CYCLES = 10000   # injection-free cycles appended after each run

DRAIN_DEBUG_PRINTS = False
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

# ─── Protocol/message-class identifiers ──────────────────────────────────────

MESSAGE_CLASS_REQ  = "REQ"
MESSAGE_CLASS_RESP = "RESP"
MESSAGE_CLASS_DATA = "DATA"
MESSAGE_CLASS_CTRL = "CTRL"

MESSAGE_CLASSES: Tuple[str, ...] = (
    MESSAGE_CLASS_REQ,
    MESSAGE_CLASS_RESP,
    MESSAGE_CLASS_DATA,
    MESSAGE_CLASS_CTRL,
)

PROTOCOL_SERVICE_ORDER: Tuple[str, ...] = (
    MESSAGE_CLASS_RESP,
    MESSAGE_CLASS_REQ,
    MESSAGE_CLASS_DATA,
    MESSAGE_CLASS_CTRL,
)

# Top-level switch for legacy vs protocol traffic generation.
# False = legacy single-class traffic, True = REQ->RESP protocol traffic.
PROTOCOL_MODE_ENABLED = True

## iffy --
PROTOCOL_INJECTION_QUEUE_CAPACITY_PER_CLASS = 32
PROTOCOL_EJECTION_QUEUE_CAPACITY_PER_CLASS = 32
PROTOCOL_OUTSTANDING_LIMIT_PER_CLASS: Dict[str, int] = {
    MESSAGE_CLASS_REQ: 16,
    MESSAGE_CLASS_RESP: 16,
    MESSAGE_CLASS_DATA: 8,
    MESSAGE_CLASS_CTRL: 8,
}


def _zero_message_class_map(default_value: int = 0) -> Dict[str, int]:
    """Return a fresh per-message-class map."""
    return {message_class: default_value for message_class in MESSAGE_CLASSES}


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


@dataclass
class ProtocolConfig:
    """
    Packet-level approximation of endpoint protocol queues and dependencies.

    This intentionally models the DRAIN paper's protocol-deadlock concepts at a
    coarse packet granularity rather than as hardware-accurate credit or
    shift-register logic.
    """

    enabled: bool = PROTOCOL_MODE_ENABLED
    injection_queue_capacity_per_class: int = PROTOCOL_INJECTION_QUEUE_CAPACITY_PER_CLASS
    ejection_queue_capacity_per_class: int = PROTOCOL_EJECTION_QUEUE_CAPACITY_PER_CLASS
    outstanding_limit_per_class: Dict[str, int] = field(
        default_factory=lambda: dict(PROTOCOL_OUTSTANDING_LIMIT_PER_CLASS)
    )
    source_message_classes: Tuple[str, ...] = (MESSAGE_CLASS_REQ,)
    sink_message_classes: Tuple[str, ...] = (MESSAGE_CLASS_RESP,)
    service_order: Tuple[str, ...] = PROTOCOL_SERVICE_ORDER
    request_generates_response: bool = True

    def outstanding_limit(self, message_class: str) -> int:
        """Return the configured outstanding packet limit for one class."""
        return max(
            0,
            int(
                self.outstanding_limit_per_class.get(
                    message_class,
                    self.injection_queue_capacity_per_class,
                )
            ),
        )


@dataclass
class TransactionRecord:
    """Track one REQ→RESP transaction in protocol mode."""

    transaction_id: int
    request_packet_id: int
    request_created_cycle: int
    request_src_chiplet: str
    request_dst_chiplet: str
    response_packet_id: Optional[int] = None
    response_created_cycle: Optional[int] = None
    completed_cycle: Optional[int] = None


def build_default_protocol_config() -> ProtocolConfig:
    """Construct a fresh protocol config from the top-level default variables."""
    return ProtocolConfig(
        enabled=PROTOCOL_MODE_ENABLED,
        injection_queue_capacity_per_class=PROTOCOL_INJECTION_QUEUE_CAPACITY_PER_CLASS,
        ejection_queue_capacity_per_class=PROTOCOL_EJECTION_QUEUE_CAPACITY_PER_CLASS,
        outstanding_limit_per_class=dict(PROTOCOL_OUTSTANDING_LIMIT_PER_CLASS),
        source_message_classes=(MESSAGE_CLASS_REQ,),
        sink_message_classes=(MESSAGE_CLASS_RESP,),
        service_order=PROTOCOL_SERVICE_ORDER,
        request_generates_response=True,
    )


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
    message_class:       str  = MESSAGE_CLASS_REQ
    transaction_id:      Optional[int] = None
    request_packet_id:   Optional[int] = None
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


@dataclass(frozen=True)
class DrainTurnBinding:
    """
    One static DRAIN turn-table entry.

    The binding is keyed by the source router and the input port whose escape
    VC currently holds the head flit. It specifies the unique output direction
    that port must use during a DRAIN shift, along with the exact successor
    router/input-port pair reached by that move.
    """

    src_router: str
    src_input_port: str
    out_dir: str
    dst_router: str
    dst_input_port: str


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
        self.injected_packets_by_class: Dict[str, List[Packet]] = {
            message_class: [] for message_class in MESSAGE_CLASSES
        }
        self.received_packets_by_class: Dict[str, List[Packet]] = {
            message_class: [] for message_class in MESSAGE_CLASSES
        }

    def record_injected_packet(self, packet: Packet) -> None:
        """Track one packet injection at this BR."""
        self.injected_packets.append(packet)
        self.injected_packets_by_class[packet.message_class].append(packet)

    # ── inbound (interposer → chiplet) ────────────────────────────────────────

    def receive_packet(self, packet: Packet, cycle: int) -> None:
        """Record packet delivery; called once all flits have been ejected."""
        packet.current_node    = self.router_id
        packet.delivered_cycle = cycle
        self.received_packets.append(packet)
        self.received_packets_by_class[packet.message_class].append(packet)

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
        self.received_packets_by_class: Dict[str, List[Packet]] = {
            message_class: [] for message_class in MESSAGE_CLASSES
        }

        # Packet-level protocol queues.
        self.injection_queues: Dict[str, List[Packet]] = {
            message_class: [] for message_class in MESSAGE_CLASSES
        }
        self.ejection_queues: Dict[str, List[Packet]] = {
            message_class: [] for message_class in MESSAGE_CLASSES
        }
        self.outstanding_packets_by_class: Dict[str, int] = _zero_message_class_map()

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
        self.received_packets_by_class[packet.message_class].append(packet)

    def enqueue_injection_packet(self, packet: Packet, capacity: int) -> bool:
        """Append a packet to the per-class source queue if space remains."""
        queue = self.injection_queues[packet.message_class]
        if len(queue) >= capacity:
            return False
        queue.append(packet)
        return True

    def peek_next_injection_packet(
        self,
        service_order: Tuple[str, ...] = PROTOCOL_SERVICE_ORDER,
    ) -> Optional[Packet]:
        """Return the highest-priority pending source packet, if any."""
        for message_class in service_order:
            queue = self.injection_queues[message_class]
            if queue:
                return queue[0]
        return None

    def pop_next_injection_packet(
        self,
        service_order: Tuple[str, ...] = PROTOCOL_SERVICE_ORDER,
    ) -> Optional[Packet]:
        """Pop the highest-priority pending source packet, if any."""
        for message_class in service_order:
            queue = self.injection_queues[message_class]
            if queue:
                return queue.pop(0)
        return None

    def can_accept_ejection_packet(
        self,
        message_class: str,
        capacity: int,
        reserved_slots: int = 0,
    ) -> bool:
        """
        Check whether one more delivered packet may enter this class queue.

        Reservations are counted separately so packets can eject flit-by-flit
        once the class queue has committed to holding the completed packet.
        """
        return len(self.ejection_queues[message_class]) + reserved_slots < capacity

    def enqueue_ejection_packet(self, packet: Packet) -> None:
        """Append a fully ejected packet to the destination class queue."""
        self.ejection_queues[packet.message_class].append(packet)

    def peek_ejection_packet(self, message_class: str) -> Optional[Packet]:
        """Inspect the oldest delivered packet for one message class."""
        queue = self.ejection_queues[message_class]
        return queue[0] if queue else None

    def pop_ejection_packet(self, message_class: str) -> Optional[Packet]:
        """Remove the oldest delivered packet for one message class."""
        queue = self.ejection_queues[message_class]
        return queue.pop(0) if queue else None

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
        protocol_config:   Optional[ProtocolConfig] = None,
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
        self.protocol_config: ProtocolConfig = (
            protocol_config if protocol_config is not None else ProtocolConfig()
        )

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
        self._transaction_counter:  int = 0
        self._attempted_injections: int = 0
        self._successful_injections: int = 0
        self._failed_injections:    int = 0   # attempted but dropped (all BRs full)
        self._attempted_injections_by_type: Dict[str, int] = {}
        self._successful_injections_by_type: Dict[str, int] = {}
        self._failed_injections_by_type: Dict[str, int] = {}
        self._attempted_injections_by_message_class: Dict[str, int] = _zero_message_class_map()
        self._successful_injections_by_message_class: Dict[str, int] = _zero_message_class_map()
        self._failed_injections_by_message_class: Dict[str, int] = _zero_message_class_map()
        self._transactions_started: int = 0
        self._transactions_completed: int = 0
        self._completed_transaction_latency_total: int = 0
        self._transaction_records: Dict[int, TransactionRecord] = {}
        self._packet_ejection_reservations: Dict[int, str] = {}
        self._reserved_ejection_slots: Dict[str, Dict[str, int]] = {}

        self._build_routers()
        self._build_links()
        self._attach_chiplets()
        self._reserved_ejection_slots = {
            chiplet_name: _zero_message_class_map()
            for chiplet_name in self.chiplets.keys()
        }
        self._drain_turn_cycle: List[DrainTurnBinding] = []
        self._drain_turn_table: Dict[Tuple[str, str], DrainTurnBinding] = {}
        if self.drain_enabled:
            self._drain_turn_cycle = self._compute_standard_drain_turn_cycle()
            self._drain_turn_table = {
                (binding.src_router, binding.src_input_port): binding
                for binding in self._drain_turn_cycle
            }
            self._validate_drain_turn_table()

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

    def _validate_standard_drain_mesh(self) -> None:
        """
        Ensure the topology matches the standard fault-free 4x4 interposer mesh.

        The first per-port full-shift DRAIN implementation intentionally targets
        only the fully connected 4x4 mesh. Faulted topologies or other mesh
        sizes must not silently fall back to a weaker drain model.
        """
        if self.rows != 4 or self.cols != 4:
            raise ValueError(
                "Per-port DRAIN currently supports only the standard fault-free 4x4 mesh."
            )

        for r in range(self.rows):
            for c in range(self.cols):
                rid = self.router_id(r, c)
                router = self.routers[rid]
                expected_neighbors = {
                    "N": self.router_id(r - 1, c) if r > 0 else None,
                    "S": self.router_id(r + 1, c) if r < self.rows - 1 else None,
                    "E": self.router_id(r, c + 1) if c < self.cols - 1 else None,
                    "W": self.router_id(r, c - 1) if c > 0 else None,
                }
                if router.ir_neighbors != expected_neighbors:
                    raise ValueError(
                        "Per-port DRAIN currently supports only the standard fault-free 4x4 mesh."
                    )

    def _directed_inter_router_edges(self) -> List[Tuple[str, str, str]]:
        """Return every directed cardinal inter-router channel exactly once."""
        edges: List[Tuple[str, str, str]] = []
        for rid in sorted(self.routers.keys()):
            router = self.routers[rid]
            for out_dir in ("N", "E", "S", "W"):
                dst_rid = router.ir_neighbors.get(out_dir)
                if dst_rid is not None:
                    edges.append((rid, out_dir, dst_rid))
        return edges

    def _compute_standard_drain_turn_cycle(self) -> List[DrainTurnBinding]:
        """
        Compute one deterministic cycle over all directed inter-router channels.

        The cycle is built once at construction time and reused for every drain
        window. Each directed channel corresponds to one bound input port at its
        destination router, giving a per-input-port drain turn-table.
        """
        self._validate_standard_drain_mesh()

        outgoing: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for src_rid, out_dir, dst_rid in self._directed_inter_router_edges():
            outgoing[src_rid].append((src_rid, out_dir, dst_rid))

        direction_order = {"N": 0, "E": 1, "S": 2, "W": 3}
        for rid in outgoing:
            outgoing[rid].sort(key=lambda edge: direction_order[edge[1]])

        edge_cycle_reversed: List[Tuple[str, str, str]] = []

        def dfs(src_rid: str) -> None:
            while outgoing[src_rid]:
                edge = outgoing[src_rid].pop(0)
                dfs(edge[2])
                edge_cycle_reversed.append(edge)

        start_rid = self.router_id(0, 0)
        dfs(start_rid)
        edge_cycle = list(reversed(edge_cycle_reversed))
        expected_edge_count = len(self._directed_inter_router_edges())
        if len(edge_cycle) != expected_edge_count:
            raise ValueError(
                "Failed to build a complete per-port DRAIN cycle for the 4x4 mesh."
            )

        bindings: List[DrainTurnBinding] = []
        edge_count = len(edge_cycle)
        for idx, (src_rid, out_dir, dst_rid) in enumerate(edge_cycle):
            nxt_src_rid, nxt_out_dir, nxt_dst_rid = edge_cycle[(idx + 1) % edge_count]
            if dst_rid != nxt_src_rid:
                raise ValueError("Drain edge cycle is not closed correctly.")
            src_input_port = self.OPPOSITE_DIR[out_dir]
            dst_input_port = self.OPPOSITE_DIR[nxt_out_dir]
            bindings.append(
                DrainTurnBinding(
                    src_router=dst_rid,
                    src_input_port=src_input_port,
                    out_dir=nxt_out_dir,
                    dst_router=nxt_dst_rid,
                    dst_input_port=dst_input_port,
                )
            )
        return bindings

    def _validate_drain_turn_table(self) -> None:
        """Sanity-check the static per-port drain turn-table."""
        if not self._drain_turn_cycle:
            raise ValueError("Per-port DRAIN requires a non-empty drain turn-table.")

        directed_edges = self._directed_inter_router_edges()
        expected_keys = {
            (dst_rid, self.OPPOSITE_DIR[out_dir])
            for src_rid, out_dir, dst_rid in directed_edges
        }
        actual_keys = set(self._drain_turn_table.keys())
        if actual_keys != expected_keys:
            raise ValueError("Per-port DRAIN turn-table does not cover all directed channels.")

        for binding in self._drain_turn_cycle:
            router = self.routers[binding.src_router]
            if binding.src_input_port not in ("N", "S", "E", "W"):
                raise ValueError("Drain turn-table must bind only cardinal input ports.")
            dst_neighbor = router.ir_neighbors.get(binding.out_dir)
            if dst_neighbor != binding.dst_router:
                raise ValueError("Drain turn-table references a non-neighbor drain output.")
            if self.OPPOSITE_DIR[binding.out_dir] != binding.dst_input_port:
                raise ValueError("Drain turn-table destination input port is inconsistent.")

        router_outputs: Dict[str, Dict[str, str]] = defaultdict(dict)
        for binding in self._drain_turn_cycle:
            prev = router_outputs[binding.src_router].get(binding.src_input_port)
            if prev is not None and prev != binding.out_dir:
                raise ValueError("Drain turn-table contains duplicate bindings for an input port.")
            router_outputs[binding.src_router][binding.src_input_port] = binding.out_dir

        for rid, mapping in router_outputs.items():
            if len(set(mapping.values())) != len(mapping):
                raise ValueError(
                    f"Drain turn-table does not assign distinct outputs per input port at {rid}."
                )

        cycle_len = len(self._drain_turn_cycle)
        for idx, binding in enumerate(self._drain_turn_cycle):
            next_binding = self._drain_turn_cycle[(idx + 1) % cycle_len]
            if binding.dst_router != next_binding.src_router:
                raise ValueError("Drain turn-table cycle is not globally closed.")
            if binding.dst_input_port != next_binding.src_input_port:
                raise ValueError("Drain turn-table cycle does not match successor input ports.")

    def _compute_drain_cycle(self) -> List[str]:
        """
        Legacy router-level drain-cycle helper retained for reference.

        Active DRAIN behavior now uses the static per-port drain turn-table
        built by ``_compute_standard_drain_turn_cycle()`` instead.
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
        ``None`` in both endpoint routers.

        The current per-port DRAIN implementation is intentionally limited to
        the standard fault-free 4x4 mesh, so DRAIN-enabled fault injection is
        rejected explicitly rather than silently falling back to a weaker path.

        Parameters
        ----------
        num_faults : number of links to disable (clamped to available links).
        rng        : seeded Random instance so fault placement is reproducible.

        Returns
        -------
        List of (rid_a, rid_b) pairs for the disabled edges.
        """
        if self.drain_enabled and num_faults > 0:
            raise NotImplementedError(
                "Per-port DRAIN currently supports only the standard fault-free 4x4 mesh; "
                "DRAIN-enabled fault injection is not implemented."
            )

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
            self._full_drain_hops_remaining = len(self._drain_turn_cycle)
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

    def _drain_dest_can_accept(
        self,
        binding: DrainTurnBinding,
        departing_ports: set,
    ) -> bool:
        """
        Check whether one synchronized DRAIN shift may push into the successor
        escape VC under the current split-based approximation.
        """
        dst_esc = self.routers[binding.dst_router].input_ports[binding.dst_input_port].escape_vc
        dst_key = (binding.dst_router, binding.dst_input_port)
        dst_departs = dst_key in departing_ports

        projected_occ = dst_esc.occupancy() - (1 if dst_departs else 0)
        if projected_occ < 0 or projected_occ >= dst_esc.capacity:
            return False

        if dst_esc.allocated_flow_id is None or dst_esc.is_empty():
            return True

        if not dst_departs:
            return False

        # Preserve the current VC-lock approximation: an incoming singleton may
        # reuse the destination escape VC only when that VC empties completely.
        return dst_esc.occupancy() == 1

    def _perform_regular_drain_hop(self, cycle: int) -> int:
        """
        Perform one globally synchronized per-port DRAIN shift stage.

        Every cardinal input-port escape VC covered by the static drain
        turn-table may advance simultaneously through its bound output,
        subject to downstream escape-VC availability.
        """
        if not self._drain_turn_cycle:
            return 0

        for rid, router in self.routers.items():
            if router.input_ports["Down"].has_escape_flit():
                raise AssertionError(
                    "Per-port DRAIN assumes Down.escape_vc remains empty; Down is used only for ejection."
                )

        candidates: List[DrainTurnBinding] = []
        departing_ports: set = set()
        for binding in self._drain_turn_cycle:
            src_port = self.routers[binding.src_router].input_ports[binding.src_input_port]
            if not src_port.has_escape_flit():
                continue
            flit = src_port.escape_vc.peek()
            if flit is None:
                continue
            dst_br_id = flit.parent_packet.dst_boundary_router
            dst_ir_id = self.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]
            if binding.src_router == dst_ir_id:
                continue
            candidates.append(binding)
            departing_ports.add((binding.src_router, binding.src_input_port))

        if not candidates:
            return 0

        # A destination port can only be treated as freeing one slot if its
        # own source-side move is also part of the synchronized shift. Start
        # from the optimistic candidate set and prune until the movable set is
        # self-consistent.
        movable = list(candidates)
        while True:
            movable_departures = {
                (binding.src_router, binding.src_input_port)
                for binding in movable
            }
            new_movable = [
                binding
                for binding in candidates
                if self._drain_dest_can_accept(binding, movable_departures)
            ]
            if len(new_movable) == len(movable) and all(
                a is b for a, b in zip(new_movable, movable)
            ):
                break
            movable = new_movable

        staged: List[Tuple[DrainTurnBinding, Flit]] = []
        for binding in movable:
            src_esc = self.routers[binding.src_router].input_ports[binding.src_input_port].escape_vc
            staged.append((binding, src_esc.pop()))

        # Phase 1: finalize all source-side split/retag state before any
        # destination VC receives a new flit. This preserves the intended
        # globally synchronized shift semantics when a VC is both a source and
        # a destination in the same DRAIN cycle.
        moves = 0
        for binding, flit in staged:
            src_router = self.routers[binding.src_router]
            src_esc = src_router.input_ports[binding.src_input_port].escape_vc
            old_flow = flit.flow_id

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

        # Phase 2: commit all destination pushes after every source VC has been
        # updated for this synchronized shift step.
        for binding, flit in staged:
            new_flow_dst = self._alloc_flow_id()
            flit.flow_id = new_flow_dst
            flit.is_worm_head = True
            flit.is_worm_tail = True
            flit.in_escape_vc = True

            dst_esc = self.routers[binding.dst_router].input_ports[binding.dst_input_port].escape_vc
            if not dst_esc.push(flit):
                raise AssertionError(
                    "Synchronized DRAIN commit violated escape-VC capacity/allocation: "
                    f"{binding.src_router}.{binding.src_input_port} -> "
                    f"{binding.dst_router}.{binding.dst_input_port}"
                )
            flit.parent_packet.current_node = binding.dst_router
            if flit.flit_idx == 0:
                flit.parent_packet.hops += 1
            moves += 1

        blocked = len(candidates) - len(movable)
        self._print_drain_debug(
            cycle,
            f"regular_drain_hop: candidates={len(candidates)} forwarded={len(movable)} blocked={blocked}",
            force=True,
        )

        return moves

    def _perform_drain_ejection_stage(self, cycle: int) -> int:
        """
        Perform one synchronized drain-ejection stage through router Down outputs.

        At most one escape flit may eject per router in this stage.
        """
        candidates_by_router: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for rid, router in self.routers.items():
            for port_name, port in router.input_ports.items():
                flit = port.escape_vc.peek()
                if flit is None:
                    continue
                dst_br_id = flit.parent_packet.dst_boundary_router
                dst_ir_id = self.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]
                if rid == dst_ir_id:
                    candidates_by_router[rid].append((rid, port_name))

        if not candidates_by_router:
            return 0

        winners: List[Tuple[str, str]] = []
        for rid, candidates in candidates_by_router.items():
            eligible = []
            for candidate in candidates:
                port_name = candidate[1]
                flit = self.routers[rid].input_ports[port_name].escape_vc.peek()
                if flit is not None and self._can_start_packet_ejection(flit.parent_packet):
                    eligible.append(candidate)
            if eligible:
                winners.append(self._routing_rng.choice(eligible))

        staged: List[Tuple[str, str, Packet]] = []
        for rid, port_name in winners:
            flit = self.routers[rid].input_ports[port_name].escape_vc.peek()
            if flit is None:
                continue
            pkt = flit.parent_packet
            if self._reserve_packet_ejection(pkt):
                staged.append((rid, port_name, pkt))

        moves = 0
        for rid, port_name, pkt in staged:
            esc = self.routers[rid].input_ports[port_name].escape_vc
            esc.pop()
            pkt.flits_delivered += 1
            if pkt.flits_delivered == FLITS_PER_PACKET:
                self._finalize_packet_delivery(pkt, cycle)
            moves += 1

        self._print_drain_debug(
            cycle,
            f"drain_ejection_stage: candidates={sum(len(v) for v in candidates_by_router.values())} "
            f"ejected={moves}",
            force=True,
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

    def _next_transaction_id(self) -> int:
        """Allocate a fresh protocol transaction ID."""
        txn_id = self._transaction_counter
        self._transaction_counter += 1
        return txn_id

    def _protocol_enabled(self) -> bool:
        return self.protocol_config.enabled

    def _record_packet_attempt(self, packet: Packet) -> None:
        """Account for one newly offered packet before network injection."""
        src_type = self.chiplets[packet.src_chiplet].chiplet_type
        self._attempted_injections += 1
        self._attempted_injections_by_type[src_type] = (
            self._attempted_injections_by_type.get(src_type, 0) + 1
        )
        self._attempted_injections_by_message_class[packet.message_class] += 1

    def _record_packet_drop(self, packet: Packet) -> None:
        """Account for one packet offer rejected before entering the network."""
        src_type = self.chiplets[packet.src_chiplet].chiplet_type
        self._failed_injections += 1
        self._failed_injections_by_type[src_type] = (
            self._failed_injections_by_type.get(src_type, 0) + 1
        )
        self._failed_injections_by_message_class[packet.message_class] += 1

    def _record_packet_network_injection(self, packet: Packet) -> None:
        """Account for one packet entering the interposer network."""
        src_type = self.chiplets[packet.src_chiplet].chiplet_type
        self._successful_injections += 1
        self._successful_injections_by_type[src_type] = (
            self._successful_injections_by_type.get(src_type, 0) + 1
        )
        self._successful_injections_by_message_class[packet.message_class] += 1

    def _build_packet(
        self,
        src_chiplet: str,
        dst_chiplet: str,
        cycle: int,
        rng: random.Random,
        message_class: str = MESSAGE_CLASS_REQ,
        transaction_id: Optional[int] = None,
        request_packet_id: Optional[int] = None,
        src_boundary_router: Optional[str] = None,
        dst_boundary_router: Optional[str] = None,
    ) -> Optional[Packet]:
        """Construct a packet and choose default boundary routers if needed."""
        src_chiplet_obj = self.chiplets[src_chiplet]
        dst_chiplet_obj = self.chiplets[dst_chiplet]
        if not src_chiplet_obj.boundary_routers or not dst_chiplet_obj.boundary_routers:
            return None

        if src_boundary_router is None:
            src_boundary_router = rng.choice(src_chiplet_obj.boundary_routers).router_id
        if dst_boundary_router is None:
            dst_boundary_router = rng.choice(dst_chiplet_obj.boundary_routers).router_id

        return Packet(
            packet_id=self._next_packet_id(),
            src_chiplet=src_chiplet,
            dst_chiplet=dst_chiplet,
            src_boundary_router=src_boundary_router,
            dst_boundary_router=dst_boundary_router,
            created_cycle=cycle,
            current_node=src_boundary_router,
            message_class=message_class,
            transaction_id=transaction_id,
            request_packet_id=request_packet_id,
        )

    def _candidate_source_brs(
        self,
        packet: Packet,
        rng: random.Random,
    ) -> List[BoundaryRouter]:
        """Order candidate source BRs, preferring the packet's assigned BR."""
        chiplet = self.chiplets[packet.src_chiplet]
        ordered: List[BoundaryRouter] = []
        preferred = packet.src_boundary_router
        for br in chiplet.boundary_routers:
            if br.router_id == preferred:
                ordered.append(br)
                break

        others = [br for br in chiplet.boundary_routers if br.router_id != preferred]
        rng.shuffle(others)
        ordered.extend(others)
        return ordered

    def _can_accept_protocol_source_packet(
        self,
        src_chiplet: str,
        message_class: str,
    ) -> bool:
        """Check queue capacity and per-class outstanding limits."""
        chiplet = self.chiplets[src_chiplet]
        queue = chiplet.injection_queues[message_class]
        if len(queue) >= self.protocol_config.injection_queue_capacity_per_class:
            return False
        outstanding_limit = self.protocol_config.outstanding_limit(message_class)
        if chiplet.outstanding_packets_by_class[message_class] >= outstanding_limit:
            return False
        return True

    def _enqueue_protocol_source_packet(self, packet: Packet) -> bool:
        """Place a new packet into a source chiplet's per-class injection queue."""
        chiplet = self.chiplets[packet.src_chiplet]
        ok = chiplet.enqueue_injection_packet(
            packet,
            self.protocol_config.injection_queue_capacity_per_class,
        )
        if ok:
            chiplet.outstanding_packets_by_class[packet.message_class] += 1
        return ok

    def _mark_packet_delivered(self, packet: Packet) -> None:
        """Release one outstanding packet slot at the source chiplet."""
        src_chiplet = self.chiplets[packet.src_chiplet]
        current = src_chiplet.outstanding_packets_by_class.get(packet.message_class, 0)
        src_chiplet.outstanding_packets_by_class[packet.message_class] = max(0, current - 1)

    def _can_start_packet_ejection(self, packet: Packet) -> bool:
        """
        Check whether the destination chiplet can accept this packet's ejection.

        The queue reservation is packet-level, which is a deliberate
        approximation of endpoint flow control rather than a flit-credit model.
        """
        if not self._protocol_enabled():
            return True
        if packet.packet_id in self._packet_ejection_reservations:
            return True
        if packet.flits_delivered != 0:
            return False

        dst_chiplet = self.chiplets[packet.dst_chiplet]
        reserved_slots = self._reserved_ejection_slots[packet.dst_chiplet][packet.message_class]
        return dst_chiplet.can_accept_ejection_packet(
            packet.message_class,
            self.protocol_config.ejection_queue_capacity_per_class,
            reserved_slots=reserved_slots,
        )

    def _reserve_packet_ejection(self, packet: Packet) -> bool:
        """Reserve one destination queue slot for a packet before first ejection."""
        if not self._protocol_enabled():
            return True
        if packet.packet_id in self._packet_ejection_reservations:
            return True
        if not self._can_start_packet_ejection(packet):
            return False

        self._packet_ejection_reservations[packet.packet_id] = packet.dst_chiplet
        self._reserved_ejection_slots[packet.dst_chiplet][packet.message_class] += 1
        return True

    def _finalize_packet_delivery(self, packet: Packet, cycle: int) -> None:
        """
        Complete packet ejection into the destination endpoint model.

        In protocol mode the packet is first placed in a per-class ejection
        queue, then consumed later according to protocol rules.
        """
        dst_br_id = packet.dst_boundary_router
        dst_br = self.boundary_routers[dst_br_id]
        dst_chiplet = self.chiplets[dst_br.chiplet_name]

        dst_br.receive_packet(packet, cycle)
        if self._protocol_enabled():
            reserved_owner = self._packet_ejection_reservations.pop(packet.packet_id, None)
            if reserved_owner is not None:
                self._reserved_ejection_slots[reserved_owner][packet.message_class] = max(
                    0,
                    self._reserved_ejection_slots[reserved_owner][packet.message_class] - 1,
                )
            dst_chiplet.enqueue_ejection_packet(packet)
        dst_chiplet.receive_packet(packet)
        self._mark_packet_delivered(packet)

    def _generate_protocol_request_traffic(
        self,
        cycle: int,
        rng: random.Random,
    ) -> None:
        """Create external REQ packets and queue them at their source chiplets."""
        chiplet_names = list(self.chiplets.keys())
        for src_name, chiplet in self.chiplets.items():
            rate = chiplet.current_injection_rate(cycle, rng)
            if rng.random() >= rate:
                continue

            dst_name = rng.choice([name for name in chiplet_names if name != src_name])
            transaction_id = self._next_transaction_id()
            pkt = self._build_packet(
                src_chiplet=src_name,
                dst_chiplet=dst_name,
                cycle=cycle,
                rng=rng,
                message_class=MESSAGE_CLASS_REQ,
                transaction_id=transaction_id,
            )
            if pkt is None:
                continue

            self._record_packet_attempt(pkt)
            if not self._can_accept_protocol_source_packet(src_name, pkt.message_class):
                self._record_packet_drop(pkt)
                continue
            if not self._enqueue_protocol_source_packet(pkt):
                self._record_packet_drop(pkt)
                continue

            self._transactions_started += 1
            self._transaction_records[transaction_id] = TransactionRecord(
                transaction_id=transaction_id,
                request_packet_id=pkt.packet_id,
                request_created_cycle=cycle,
                request_src_chiplet=src_name,
                request_dst_chiplet=dst_name,
            )

    def _try_generate_response_for_request(
        self,
        request_packet: Packet,
        cycle: int,
        rng: random.Random,
    ) -> bool:
        """Create a RESP packet when a delivered REQ is protocol-consumed."""
        if not self.protocol_config.request_generates_response:
            return True
        if request_packet.transaction_id is None:
            return True

        response_src = request_packet.dst_chiplet
        response_dst = request_packet.src_chiplet
        if not self._can_accept_protocol_source_packet(response_src, MESSAGE_CLASS_RESP):
            return False

        response_packet = self._build_packet(
            src_chiplet=response_src,
            dst_chiplet=response_dst,
            cycle=cycle,
            rng=rng,
            message_class=MESSAGE_CLASS_RESP,
            transaction_id=request_packet.transaction_id,
            request_packet_id=request_packet.packet_id,
            src_boundary_router=request_packet.dst_boundary_router,
            dst_boundary_router=request_packet.src_boundary_router,
        )
        if response_packet is None:
            return False

        self._record_packet_attempt(response_packet)
        if not self._enqueue_protocol_source_packet(response_packet):
            return False

        record = self._transaction_records.get(request_packet.transaction_id)
        if record is not None:
            record.response_packet_id = response_packet.packet_id
            record.response_created_cycle = cycle
        return True

    def _service_protocol_ejection_queues(
        self,
        cycle: int,
        rng: random.Random,
    ) -> None:
        """
        Consume endpoint queues in fixed message-class priority order.

        RESP is treated as a sink class and is consumed greedily every cycle so
        that it remains eventually drainable even when REQ queues are backlogged.
        """
        if not self._protocol_enabled():
            return

        for chiplet in self.chiplets.values():
            while chiplet.peek_ejection_packet(MESSAGE_CLASS_RESP) is not None:
                response_packet = chiplet.pop_ejection_packet(MESSAGE_CLASS_RESP)
                if response_packet is None:
                    break
                if response_packet.transaction_id is not None:
                    record = self._transaction_records.get(response_packet.transaction_id)
                    if record is not None and record.completed_cycle is None:
                        record.completed_cycle = cycle
                        self._transactions_completed += 1
                        self._completed_transaction_latency_total += (
                            cycle - record.request_created_cycle
                        )

            for message_class in self.protocol_config.service_order:
                if message_class == MESSAGE_CLASS_RESP:
                    continue
                packet = chiplet.peek_ejection_packet(message_class)
                if packet is None:
                    continue
                if message_class == MESSAGE_CLASS_REQ:
                    if not self._try_generate_response_for_request(packet, cycle, rng):
                        continue
                chiplet.pop_ejection_packet(message_class)
                break

    def _try_inject_packet_into_network(
        self,
        packet: Packet,
        rng: random.Random,
    ) -> bool:
        """Try to place all flits of one packet into an attached IR Down VC."""
        for src_br in self._candidate_source_brs(packet, rng):
            ir = self.routers[src_br.attached_ir]   # type: ignore[index]
            down_port = ir.input_ports["Down"]
            eligible = [
                vc for vc in down_port.normal_vcs
                if vc.allocated_flow_id is None
                and (vc.capacity - vc.occupancy()) >= FLITS_PER_PACKET
            ]
            if not eligible:
                continue

            target_vc = rng.choice(eligible)
            packet.src_boundary_router = src_br.router_id
            packet.current_node = ir.router_id
            for flit in packet.flits:
                target_vc.push(flit)
            src_br.record_injected_packet(packet)
            self._record_packet_network_injection(packet)
            return True
        return False

    def _injection_blocked_this_cycle(self, cycle: int) -> bool:
        """
        Preserve the current DRAIN behavior for new network injections.

        PRE_DRAIN and FULL_DRAIN remain injection-free, and the cycle that
        enters PRE_DRAIN also suppresses same-cycle network admission.
        """
        pre_drain_starts_this_cycle = (
            self.drain_enabled
            and self._drain_mode == DRAIN_MODE_NORMAL
            and cycle > 0
            and cycle % self.drain_period == 0
        )
        return (
            self._full_drain_active()
            or self._pre_drain_active()
            or pre_drain_starts_this_cycle
        )

    def inject_random_packets(
        self,
        cycle: int,
        rng:   random.Random,
        generate_new_traffic: bool = True,
    ) -> int:
        """
        Inject traffic into the network for one cycle.

        Legacy mode:
          Flits bypass the boundary-router staging buffer and are written
          directly into the attached IR's Down input-port VCs at injection time.
          A normal VC in the Down port is eligible if it is unallocated AND has
          at least ``FLITS_PER_PACKET`` free slots.  Source BRs are tried in
          random order; the first eligible IR Down-port VC wins.

        Protocol mode:
          New REQ packets are generated into per-chiplet source queues, then one
          queued packet per chiplet may enter the interposer each cycle.  During
          cooldown, callers can set ``generate_new_traffic=False`` to stop new
          REQ creation while still draining the source queues.

        Returns the count of successfully injected packets this cycle.
        During PRE_DRAIN and FULL_DRAIN, injection is suppressed.
        """
        if self._protocol_enabled():
            if generate_new_traffic:
                self._generate_protocol_request_traffic(cycle, rng)
            if self._injection_blocked_this_cycle(cycle):
                return 0

            injected = 0
            for chiplet in self.chiplets.values():
                packet = chiplet.peek_next_injection_packet(self.protocol_config.service_order)
                if packet is None:
                    continue
                if not self._try_inject_packet_into_network(packet, rng):
                    continue
                chiplet.pop_next_injection_packet(self.protocol_config.service_order)
                injected += 1
            return injected

        if not generate_new_traffic:
            return 0
        if self._injection_blocked_this_cycle(cycle):
            return 0

        chiplet_names = list(self.chiplets.keys())
        injected = 0
        for src_name, chiplet in self.chiplets.items():
            rate = chiplet.current_injection_rate(cycle, rng)
            if rng.random() >= rate:
                continue
            dst_name    = rng.choice([n for n in chiplet_names if n != src_name])
            pkt = self._build_packet(
                src_chiplet=src_name,
                dst_chiplet=dst_name,
                cycle=cycle,
                rng=rng,
                message_class=MESSAGE_CLASS_REQ,
            )
            if pkt is None:
                continue
            self._record_packet_attempt(pkt)
            if self._try_inject_packet_into_network(pkt, rng):
                injected += 1
                continue
            self._record_packet_drop(pkt)
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

        Phase 2a – One globally synchronized per-port DRAIN shift stage for
                   escape-VC flits, followed by a same-cycle drain ejection
                   stage through ``Down`` when a DRAIN window is active.

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
        # During a full drain, one globally synchronized per-port DRAIN shift
        # stage runs per cycle, followed by a same-cycle drain ejection stage.
        if self._full_drain_active():
            hop_moves = self._perform_regular_drain_hop(cycle)
            ejected = self._perform_drain_ejection_stage(cycle)
            moves += hop_moves + ejected
            self._full_drain_hops_remaining -= 1
            self._service_protocol_ejection_queues(cycle, self._routing_rng)
            self._print_drain_debug(
                cycle,
                f"full_drain_step: hop_moves={hop_moves} ejected={ejected} hops_remaining={self._full_drain_hops_remaining}",
                force=True
            )
            return moves

        # ── Phase 2a: synchronized DRAIN circulation for escape VCs ───────
        drain_this_cycle = self._drain_active()
        if drain_this_cycle and self._drain_mode == DRAIN_MODE_DRAIN:
            hop_moves = self._perform_regular_drain_hop(cycle)
            ejected = self._perform_drain_ejection_stage(cycle)
            moves += hop_moves + ejected
            self._drain_hops_remaining -= 1
            if STRICT_REGULAR_DRAIN:
                self._print_drain_debug(
                    cycle,
                    f"strict_regular_drain: hop_moves={hop_moves} ejected={ejected} normal forwarding skipped",
                    force=True,
                )
                self._service_protocol_ejection_queues(cycle, self._routing_rng)
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
            router = self.routers[rid]

            if out_dir == "Down":
                eligible = [
                    candidate
                    for candidate in candidates
                    if self._can_start_packet_ejection(candidate[3].parent_packet)
                ]
                if not eligible:
                    continue
                winner = self._routing_rng.choice(eligible)
            else:
                winner = self._routing_rng.choice(candidates)

            port_name, vc, _is_esc, flit, nxt_rid, use_escape_next = winner

            if out_dir == "Down":
                # Eject: deliver flit to the boundary router.
                pkt = flit.parent_packet
                if not self._reserve_packet_ejection(pkt):
                    continue
                vc.pop()
                pkt.flits_delivered += 1
                if pkt.flits_delivered == FLITS_PER_PACKET:
                    self._finalize_packet_delivery(pkt, cycle)
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

        self._service_protocol_ejection_queues(cycle, self._routing_rng)
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

    def delivered_stats(self) -> Dict[str, Any]:
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
        stats["per_message_class"] = self.delivered_stats_by_message_class()
        if self._protocol_enabled():
            stats["transactions"] = self.transaction_stats()
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
            failed = self._failed_injections_by_type.get(chiplet_type, 0)
            acceptance_rate = injected / attempted if attempted else 0.0
            type_stats = self._packet_stats(packets_by_type.get(chiplet_type, []))
            type_stats.update({
                "attempted_injections": attempted,
                "injected": injected,
                "failed_injections": failed,
                "acceptance_rate": round(acceptance_rate, 4),
                "chiplet_count": chiplet_counts_by_type.get(chiplet_type, 0),
            })
            stats_by_type[chiplet_type] = type_stats
        return stats_by_type

    def delivered_stats_by_message_class(self) -> Dict[str, Dict[str, float]]:
        """Aggregate delivered and injection statistics for each message class."""
        packets_by_message_class: Dict[str, List[Packet]] = {
            message_class: [] for message_class in MESSAGE_CLASSES
        }
        for chiplet in self.chiplets.values():
            for packet in chiplet.received_packets:
                packets_by_message_class[packet.message_class].append(packet)

        stats_by_message_class: Dict[str, Dict[str, float]] = {}
        for message_class in MESSAGE_CLASSES:
            attempted = self._attempted_injections_by_message_class.get(message_class, 0)
            injected = self._successful_injections_by_message_class.get(message_class, 0)
            failed = self._failed_injections_by_message_class.get(message_class, 0)
            acceptance_rate = injected / attempted if attempted else 0.0
            class_stats = self._packet_stats(packets_by_message_class[message_class])
            class_stats.update({
                "attempted_injections": attempted,
                "injected": injected,
                "failed_injections": failed,
                "acceptance_rate": round(acceptance_rate, 4),
            })
            stats_by_message_class[message_class] = class_stats
        return stats_by_message_class

    def transaction_stats(self) -> Dict[str, float]:
        """Return REQ→RESP transaction completion statistics."""
        completion_rate = (
            self._transactions_completed / self._transactions_started
            if self._transactions_started
            else 0.0
        )
        avg_completion_latency = (
            self._completed_transaction_latency_total / self._transactions_completed
            if self._transactions_completed
            else 0.0
        )
        return {
            "started": self._transactions_started,
            "completed": self._transactions_completed,
            "completion_rate": round(completion_rate, 4),
            "avg_completion_latency": round(avg_completion_latency, 2),
        }

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
    protocol_config:    Optional[ProtocolConfig]   = None,
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
    protocol_config    : Optional protocol/message-class configuration.
    verbose            : Print a line each time a deadlock is detected.

    Returns
    -------
    Dict mapping scenario label → (stats dict, DeadlockDetector).
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()
    if protocol_config is None:
        protocol_config = build_default_protocol_config()
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
            protocol_config=protocol_config,
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
            mesh.inject_random_packets(cycle, rng, generate_new_traffic=False)
            mesh.step(cycle)
            detector.check(mesh, cycle)

        stats = mesh.delivered_stats()
        chiplet_count = max(1, len(mesh.chiplets))
        stats["throughput"] = round(stats["delivered"] / (cycles * chiplet_count), 6)
        for message_class, class_stats in stats["per_message_class"].items():
            class_stats["throughput"] = round(
                class_stats["delivered"] / (cycles * chiplet_count),
                6,
            )

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


def _empty_message_class_metric_map() -> Dict[str, Dict[str, float]]:
    """Return a fresh accumulator for per-message-class sweep metrics."""
    return {
        message_class: {
            "attempted_injections": 0.0,
            "injected": 0.0,
            "failed_injections": 0.0,
            "delivered": 0.0,
            "throughput": 0.0,
            "avg_latency": 0.0,
        }
        for message_class in MESSAGE_CLASSES
    }


def _empty_chiplet_type_metric_map() -> Dict[str, Dict[str, float]]:
    """Return a fresh accumulator for per-chiplet-type sweep metrics."""
    return {
        chiplet_type: {
            "attempted_injections": 0.0,
            "injected": 0.0,
            "failed_injections": 0.0,
            "delivered": 0.0,
            "throughput": 0.0,
            "avg_latency": 0.0,
            "acceptance_rate": 0.0,
            "chiplet_count": 0.0,
        }
        for chiplet_type in (CHIPLET_CPU, CHIPLET_GPU)
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
    protocol_config:    Optional[ProtocolConfig] = None,
    num_seeds:          int   = 5,
    drain_period_c:     Optional[float] = 0.2,
    drain_period_alpha: float           = 2.1,
) -> Dict[str, Dict[float, Dict[str, Any]]]:
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
    if protocol_config is None:
        protocol_config = build_default_protocol_config()

    rates = [
        round(rate_min + i * rate_step, 10)
        for i in range(int(round((rate_max - rate_min) / rate_step)) + 1)
        if rate_min + i * rate_step <= rate_max + 1e-9
    ]

    sweep: Dict[str, Dict[float, Dict[str, Any]]] = {label: {} for label in SCENARIO_LABELS}
    total = len(rates)
    for idx, rate in enumerate(rates):
        effective_drain_period = (
            max(1, round((drain_period_c / (rate ** drain_period_alpha))))
            if drain_period_c is not None and stagger_drain == 1
            else drain_period
        )
        # effective_drain_period = drain_period
        print(f"  rate={rate:.6f}, drain_period={effective_drain_period} ({idx + 1}/{total})", flush=True)

        # Accumulate totals across multiple seeds, then average.
        accum: Dict[str, Dict[str, Any]] = {
            label: {
                "injected": 0.0,
                "delivered": 0.0,
                "failed_injections": 0.0,
                "avg_latency": 0.0,
                "per_message_class": _empty_message_class_metric_map(),
                "per_type": _empty_chiplet_type_metric_map(),
                "transactions": {
                    "started": 0.0,
                    "completed": 0.0,
                    "completion_rate": 0.0,
                    "avg_completion_latency": 0.0,
                },
            }
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
                protocol_config=protocol_config,
            )
            for label in SCENARIO_LABELS:
                stats = results[label][0]
                accum[label]["injected"]          += stats["injected"]
                accum[label]["delivered"]         += stats["delivered"]
                accum[label]["failed_injections"] += stats["failed_injections"]
                accum[label]["avg_latency"]       += float(stats["avg_latency"])

                per_type = stats.get("per_type", {})
                for chiplet_type in (CHIPLET_CPU, CHIPLET_GPU):
                    type_stats = per_type.get(chiplet_type, {})
                    accum[label]["per_type"][chiplet_type]["attempted_injections"] += float(
                        type_stats.get("attempted_injections", 0)
                    )
                    accum[label]["per_type"][chiplet_type]["injected"] += float(
                        type_stats.get("injected", 0)
                    )
                    accum[label]["per_type"][chiplet_type]["failed_injections"] += float(
                        type_stats.get("failed_injections", 0)
                    )
                    accum[label]["per_type"][chiplet_type]["delivered"] += float(
                        type_stats.get("delivered", 0)
                    )
                    accum[label]["per_type"][chiplet_type]["throughput"] += float(
                        type_stats.get("throughput", 0.0)
                    )
                    accum[label]["per_type"][chiplet_type]["avg_latency"] += float(
                        type_stats.get("avg_latency", 0.0)
                    )
                    accum[label]["per_type"][chiplet_type]["acceptance_rate"] += float(
                        type_stats.get("acceptance_rate", 0.0)
                    )
                    accum[label]["per_type"][chiplet_type]["chiplet_count"] += float(
                        type_stats.get("chiplet_count", 0)
                    )

                per_message_class = stats.get("per_message_class", {})
                for message_class in MESSAGE_CLASSES:
                    class_stats = per_message_class.get(message_class, {})
                    accum[label]["per_message_class"][message_class]["attempted_injections"] += float(
                        class_stats.get("attempted_injections", 0)
                    )
                    accum[label]["per_message_class"][message_class]["injected"] += float(
                        class_stats.get("injected", 0)
                    )
                    accum[label]["per_message_class"][message_class]["failed_injections"] += float(
                        class_stats.get("failed_injections", 0)
                    )
                    accum[label]["per_message_class"][message_class]["delivered"] += float(
                        class_stats.get("delivered", 0)
                    )
                    accum[label]["per_message_class"][message_class]["throughput"] += float(
                        class_stats.get("throughput", 0.0)
                    )
                    accum[label]["per_message_class"][message_class]["avg_latency"] += float(
                        class_stats.get("avg_latency", 0.0)
                    )

                transactions = stats.get("transactions")
                if isinstance(transactions, dict):
                    accum[label]["transactions"]["started"] += float(transactions.get("started", 0))
                    accum[label]["transactions"]["completed"] += float(transactions.get("completed", 0))
                    accum[label]["transactions"]["completion_rate"] += float(
                        transactions.get("completion_rate", 0.0)
                    )
                    accum[label]["transactions"]["avg_completion_latency"] += float(
                        transactions.get("avg_completion_latency", 0.0)
                    )

        for label in SCENARIO_LABELS:
            averaged_stats: Dict[str, Any] = {
                "injected":          int(round(accum[label]["injected"]          / num_seeds)),
                "delivered":         int(round(accum[label]["delivered"]         / num_seeds)),
                "failed_injections": int(round(accum[label]["failed_injections"] / num_seeds)),
                "avg_latency":       accum[label]["avg_latency"] / num_seeds,
                "drain_period":      int(effective_drain_period),
            }
            averaged_stats["per_type"] = {}
            for chiplet_type in (CHIPLET_CPU, CHIPLET_GPU):
                type_accum = accum[label]["per_type"][chiplet_type]
                averaged_stats["per_type"][chiplet_type] = {
                    "attempted_injections": int(round(type_accum["attempted_injections"] / num_seeds)),
                    "injected": int(round(type_accum["injected"] / num_seeds)),
                    "failed_injections": int(round(type_accum["failed_injections"] / num_seeds)),
                    "delivered": int(round(type_accum["delivered"] / num_seeds)),
                    "throughput": type_accum["throughput"] / num_seeds,
                    "avg_latency": type_accum["avg_latency"] / num_seeds,
                    "acceptance_rate": type_accum["acceptance_rate"] / num_seeds,
                    "chiplet_count": int(round(type_accum["chiplet_count"] / num_seeds)),
                }
            averaged_stats["per_message_class"] = {}
            for message_class in MESSAGE_CLASSES:
                class_accum = accum[label]["per_message_class"][message_class]
                averaged_stats["per_message_class"][message_class] = {
                    "attempted_injections": int(round(class_accum["attempted_injections"] / num_seeds)),
                    "injected": int(round(class_accum["injected"] / num_seeds)),
                    "failed_injections": int(round(class_accum["failed_injections"] / num_seeds)),
                    "delivered": int(round(class_accum["delivered"] / num_seeds)),
                    "throughput": class_accum["throughput"] / num_seeds,
                    "avg_latency": class_accum["avg_latency"] / num_seeds,
                }

            transactions = accum[label]["transactions"]
            if any(transactions.values()):
                averaged_stats["transactions"] = {
                    "started": int(round(transactions["started"] / num_seeds)),
                    "completed": int(round(transactions["completed"] / num_seeds)),
                    "completion_rate": transactions["completion_rate"] / num_seeds,
                    "avg_completion_latency": transactions["avg_completion_latency"] / num_seeds,
                }
            sweep[label][rate] = averaged_stats
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
    protocol_config:    Optional[ProtocolConfig] = None,
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
    if protocol_config is None:
        protocol_config = build_default_protocol_config()

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
                protocol_config=protocol_config,
            )
            detector = DeadlockDetector()

            for cycle in range(cycles):
                mesh.inject_random_packets(cycle, rng)
                mesh.step(cycle)
                detector.check(mesh, cycle)

            for cycle in range(cycles, cycles + cooldown_cycles):
                mesh.inject_random_packets(cycle, rng, generate_new_traffic=False)
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
    protocol_config:    Optional[ProtocolConfig] = None,
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
    if protocol_config is None:
        protocol_config = build_default_protocol_config()
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
                protocol_config=protocol_config,
            )
            detector = DeadlockDetector()

            for cycle in range(cycles):
                mesh.inject_random_packets(cycle, rng)
                mesh.step(cycle)
                detector.check(mesh, cycle)

            for cycle in range(cycles, cycles + cooldown_cycles):
                mesh.inject_random_packets(cycle, rng, generate_new_traffic=False)
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
    protocol_config:     Optional[ProtocolConfig] = None,
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
    if protocol_config is None:
        protocol_config = build_default_protocol_config()

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
                protocol_config=protocol_config,
            )
            detector = DeadlockDetector()

            for cycle in range(cycles):
                mesh.inject_random_packets(cycle, rng)
                mesh.step(cycle)
                detector.check(mesh, cycle)

            for cycle in range(cycles, cycles + cooldown_cycles):
                mesh.inject_random_packets(cycle, rng, generate_new_traffic=False)
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
    sweep:      Dict[str, Dict[float, Dict[str, Any]]],
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


def plot_transaction_completion_sweep_all_scenarios(
    sweep: Dict[str, Dict[float, Dict[str, Any]]],
    out_prefix: str = "injection_rate_sweep",
    show: bool = True,
) -> None:
    """Plot INJECTION_RATE vs transaction completion rate for all scenarios."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    available_labels = [
        label for label in SCENARIO_LABELS
        if label in sweep
        and any("transactions" in sweep[label][rate] for rate in sweep[label])
    ]
    if not available_labels:
        return

    all_sweep_rates = [r for label in available_labels for r in sweep[label].keys()]
    x_min = min(all_sweep_rates)
    x_max = max(all_sweep_rates)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["mediumpurple", "seagreen", "steelblue", "darkorange", "slategray", "firebrick"]
    markers = ["P", "^", "s", "o", "x", "D"]
    for label, color, marker in zip(available_labels, colors, markers):
        rates = sorted(sweep[label].keys())
        completion_rates = [
            float(
                sweep[label][rate]
                .get("transactions", {})
                .get("completion_rate", 0.0)
            )
            for rate in rates
        ]
        ax.plot(
            rates,
            completion_rates,
            marker=marker,
            linewidth=2,
            color=color,
            label=SCENARIO_DISPLAY.get(label, label),
        )

    ax.set_xlabel("INJECTION_RATE")
    ax.set_ylabel("Transaction Completion Rate")
    ax.set_title("Scenario Comparison\nTransaction Completion Rate vs INJECTION_RATE")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_file = f"{out_prefix}_transaction_completion_all_scenarios.png"
    fig.savefig(out_file, dpi=150)
    print(f"  Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def plot_req_resp_latency_sweep_all_scenarios(
    sweep: Dict[str, Dict[float, Dict[str, Any]]],
    out_prefix: str = "injection_rate_sweep",
    show: bool = True,
) -> None:
    """Plot INJECTION_RATE vs REQ/RESP average latency for all scenarios."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    available_labels = [
        label for label in SCENARIO_LABELS
        if label in sweep
        and any("per_message_class" in sweep[label][rate] for rate in sweep[label])
    ]
    if not available_labels:
        return

    all_sweep_rates = [r for label in available_labels for r in sweep[label].keys()]
    x_min = min(all_sweep_rates)
    x_max = max(all_sweep_rates)

    fig, ax = plt.subplots(figsize=(10, 6))
    base_colors = {
        "xy": "mediumpurple",
        "yx": "seagreen",
        "random_adaptive_without_drain": "steelblue",
        "random_adaptive_with_drain": "darkorange",
        "random_adaptive_turn_restricted": "slategray",
        "shortest_path": "firebrick",
    }
    markers = {
        "xy": "P",
        "yx": "^",
        "random_adaptive_without_drain": "s",
        "random_adaptive_with_drain": "o",
        "random_adaptive_turn_restricted": "x",
        "shortest_path": "D",
    }

    plotted_any = False
    for label in available_labels:
        rates = sorted(sweep[label].keys())
        req_latency = [
            float(
                sweep[label][rate]
                .get("per_message_class", {})
                .get(MESSAGE_CLASS_REQ, {})
                .get("avg_latency", 0.0)
            )
            for rate in rates
        ]
        resp_latency = [
            float(
                sweep[label][rate]
                .get("per_message_class", {})
                .get(MESSAGE_CLASS_RESP, {})
                .get("avg_latency", 0.0)
            )
            for rate in rates
        ]
        req_delivered = [
            int(
                sweep[label][rate]
                .get("per_message_class", {})
                .get(MESSAGE_CLASS_REQ, {})
                .get("delivered", 0)
            )
            for rate in rates
        ]
        resp_delivered = [
            int(
                sweep[label][rate]
                .get("per_message_class", {})
                .get(MESSAGE_CLASS_RESP, {})
                .get("delivered", 0)
            )
            for rate in rates
        ]

        color = base_colors.get(label, "black")
        marker = markers.get(label, "o")
        display = SCENARIO_DISPLAY.get(label, label)

        if any(req_delivered):
            plotted_any = True
            ax.plot(
                rates,
                req_latency,
                marker=marker,
                linewidth=2,
                color=color,
                label=f"{display} – REQ",
            )
        if any(resp_delivered):
            plotted_any = True
            ax.plot(
                rates,
                resp_latency,
                marker=marker,
                linewidth=2,
                linestyle="--",
                color=color,
                alpha=0.9,
                label=f"{display} – RESP",
            )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("INJECTION_RATE")
    ax.set_ylabel("Avg Packet Latency (cycles)")
    ax.set_title("Scenario Comparison\nREQ vs RESP Avg Latency vs INJECTION_RATE")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_file = f"{out_prefix}_req_resp_latency_all_scenarios.png"
    fig.savefig(out_file, dpi=150)
    print(f"  Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def write_injection_sweep_summary_text(
    sweep: Dict[str, Dict[float, Dict[str, Any]]],
    out_file: str = "injection_rate_sweep_summary.txt",
    protocol_config: Optional[ProtocolConfig] = None,
) -> None:
    """Write a human-readable injection-rate sweep summary to a text file."""
    if protocol_config is not None:
        traffic_mode = (
            "NEW_PROTOCOL_REQ_RESP"
            if protocol_config.enabled
            else "LEGACY_SINGLE_CLASS"
        )
    else:
        traffic_mode = "LEGACY_SINGLE_CLASS"
        for label_data in sweep.values():
            for stats in label_data.values():
                if "transactions" in stats:
                    traffic_mode = "NEW_PROTOCOL_REQ_RESP"
                    break
            if traffic_mode == "NEW_PROTOCOL_REQ_RESP":
                break

    lines: List[str] = []
    lines.append("Injection Rate Sweep Summary")
    lines.append("=" * 72)
    lines.append(f"Traffic Mode: {traffic_mode}")
    if traffic_mode == "LEGACY_SINGLE_CLASS":
        lines.append("  Legacy mode: existing single-class packet injection workload.")
    else:
        lines.append("  New protocol mode: REQ source traffic with generated RESP replies.")
    lines.append("")
    for label in SCENARIO_LABELS:
        if label not in sweep:
            continue
        lines.append(f"Scenario: {SCENARIO_DISPLAY.get(label, label)}")
        lines.append("-" * 72)
        for rate in sorted(sweep[label].keys()):
            stats = sweep[label][rate]
            lines.append(
                "  rate={rate:.6f}  injected={injected}  delivered={delivered}  "
                "failed={failed}  avg_latency={avg_latency:.2f}  drain_period={drain_period}".format(
                    rate=rate,
                    injected=int(stats.get("injected", 0)),
                    delivered=int(stats.get("delivered", 0)),
                    failed=int(stats.get("failed_injections", 0)),
                    avg_latency=float(stats.get("avg_latency", 0.0)),
                    drain_period=int(stats.get("drain_period", DRAIN_PERIOD)),
                )
            )
            per_type = stats.get("per_type", {})
            for chiplet_type in (CHIPLET_CPU, CHIPLET_GPU):
                type_stats = per_type.get(chiplet_type, {})
                lines.append(
                    "    {chiplet_type:<4} attempted={attempted:>6}  injected={injected:>6}  "
                    "failed={failed:>6}  delivered={delivered:>6}  throughput={throughput:>8.6f}  "
                    "avg_latency={avg_latency:>8.2f}  acceptance={acceptance:>6.4f}  count={count}".format(
                        chiplet_type=chiplet_type,
                        attempted=int(type_stats.get("attempted_injections", 0)),
                        injected=int(type_stats.get("injected", 0)),
                        failed=int(type_stats.get("failed_injections", 0)),
                        delivered=int(type_stats.get("delivered", 0)),
                        throughput=float(type_stats.get("throughput", 0.0)),
                        avg_latency=float(type_stats.get("avg_latency", 0.0)),
                        acceptance=float(type_stats.get("acceptance_rate", 0.0)),
                        count=int(type_stats.get("chiplet_count", 0)),
                    )
                )
            per_message_class = stats.get("per_message_class", {})
            for message_class in MESSAGE_CLASSES:
                class_stats = per_message_class.get(message_class, {})
                lines.append(
                    "    {message_class:<4} attempted={attempted:>6}  injected={injected:>6}  "
                    "failed={failed:>6}  delivered={delivered:>6}  throughput={throughput:>8.6f}  "
                    "avg_latency={avg_latency:>8.2f}".format(
                        message_class=message_class,
                        attempted=int(class_stats.get("attempted_injections", 0)),
                        injected=int(class_stats.get("injected", 0)),
                        failed=int(class_stats.get("failed_injections", 0)),
                        delivered=int(class_stats.get("delivered", 0)),
                        throughput=float(class_stats.get("throughput", 0.0)),
                        avg_latency=float(class_stats.get("avg_latency", 0.0)),
                    )
                )
            transactions = stats.get("transactions")
            if isinstance(transactions, dict):
                lines.append(
                    "    transactions started={started}  completed={completed}  "
                    "completion_rate={completion_rate:.4f}  avg_completion_latency={avg_completion_latency:.2f}".format(
                        started=int(transactions.get("started", 0)),
                        completed=int(transactions.get("completed", 0)),
                        completion_rate=float(transactions.get("completion_rate", 0.0)),
                        avg_completion_latency=float(
                            transactions.get("avg_completion_latency", 0.0)
                        ),
                    )
                )
        lines.append("")

    with open(out_file, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    print(f"  Saved: {out_file}")


def write_drain_turn_table_text(
    mesh: InterposerMesh,
    out_file: str = "drain_turn_table.txt",
) -> None:
    """Write the static per-port DRAIN turn-table to a text file."""
    lines: List[str] = []
    lines.append("Per-Port DRAIN Turn Table")
    lines.append("=" * 72)
    lines.append(
        "Static table: built once at mesh construction and reused for every DRAIN window."
    )
    lines.append(f"entries={len(mesh._drain_turn_cycle)}")
    lines.append("")

    if not mesh._drain_turn_cycle:
        lines.append("DRAIN disabled: no turn-table present.")
    else:
        lines.append(
            "idx  src_router  src_in  out  dst_router  dst_in  successor_key"
        )
        lines.append("-" * 72)
        for idx, binding in enumerate(mesh._drain_turn_cycle):
            successor = mesh._drain_turn_cycle[(idx + 1) % len(mesh._drain_turn_cycle)]
            lines.append(
                f"{idx:>3}  "
                f"{binding.src_router:<9}  "
                f"{binding.src_input_port:<6}  "
                f"{binding.out_dir:<3}  "
                f"{binding.dst_router:<9}  "
                f"{binding.dst_input_port:<6}  "
                f"({successor.src_router},{successor.src_input_port})"
            )

    with open(out_file, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    print(f"  Saved: {out_file}")


def plot_message_class_sweep(
    sweep: Dict[str, Dict[float, Dict[str, Any]]],
    scenario_label: str = "random_adaptive_with_drain",
    out_prefix: str = "injection_rate_sweep",
    show: bool = True,
) -> None:
    """
    Plot per-message-class packet counts and latency for one scenario.

    The packet-count figure shows injected and delivered packets for each class.
    The latency figure shows the average delivered-packet latency by class.
    """
    if scenario_label not in sweep or not sweep[scenario_label]:
        return

    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    rates = sorted(sweep[scenario_label].keys())
    colors = {
        MESSAGE_CLASS_REQ: "steelblue",
        MESSAGE_CLASS_RESP: "darkorange",
        MESSAGE_CLASS_DATA: "seagreen",
        MESSAGE_CLASS_CTRL: "firebrick",
    }
    markers = {
        MESSAGE_CLASS_REQ: "o",
        MESSAGE_CLASS_RESP: "s",
        MESSAGE_CLASS_DATA: "^",
        MESSAGE_CLASS_CTRL: "D",
    }

    fig_packets, ax_packets = plt.subplots(figsize=(9, 6))
    fig_latency, ax_latency = plt.subplots(figsize=(9, 6))

    for message_class in MESSAGE_CLASSES:
        injected = [
            sweep[scenario_label][rate]
            .get("per_message_class", {})
            .get(message_class, {})
            .get("injected", 0)
            for rate in rates
        ]
        delivered = [
            sweep[scenario_label][rate]
            .get("per_message_class", {})
            .get(message_class, {})
            .get("delivered", 0)
            for rate in rates
        ]
        latencies = [
            float(
                sweep[scenario_label][rate]
                .get("per_message_class", {})
                .get(message_class, {})
                .get("avg_latency", 0.0)
            )
            for rate in rates
        ]
        if not any(injected) and not any(delivered):
            continue

        ax_packets.plot(
            rates,
            injected,
            marker=markers[message_class],
            linewidth=2,
            color=colors[message_class],
            label=f"{message_class} Injected",
        )
        ax_packets.plot(
            rates,
            delivered,
            marker=markers[message_class],
            linewidth=2,
            linestyle="--",
            color=colors[message_class],
            alpha=0.85,
            label=f"{message_class} Delivered",
        )
        ax_latency.plot(
            rates,
            latencies,
            marker=markers[message_class],
            linewidth=2,
            color=colors[message_class],
            label=message_class,
        )

    title = SCENARIO_DISPLAY.get(scenario_label, scenario_label)
    ax_packets.set_xlabel("INJECTION_RATE")
    ax_packets.set_ylabel("# Packets")
    ax_packets.set_title(f"{title}\nPer-Message-Class Packets vs INJECTION_RATE")
    ax_packets.grid(True, linestyle="--", alpha=0.5)
    ax_packets.legend(fontsize=9)
    fig_packets.tight_layout()

    ax_latency.set_xlabel("INJECTION_RATE")
    ax_latency.set_ylabel("Avg Packet Latency (cycles)")
    ax_latency.set_title(f"{title}\nPer-Message-Class Avg Packet Latency vs INJECTION_RATE")
    ax_latency.grid(True, linestyle="--", alpha=0.5)
    ax_latency.legend(fontsize=9)
    fig_latency.tight_layout()

    packet_file = f"{out_prefix}_{scenario_label}_message_class_packets.png"
    latency_file = f"{out_prefix}_{scenario_label}_message_class_latency.png"
    fig_packets.savefig(packet_file, dpi=150)
    fig_latency.savefig(latency_file, dpi=150)
    print(f"  Saved: {packet_file}")
    print(f"  Saved: {latency_file}")
    if show:
        plt.show()
    plt.close(fig_packets)
    plt.close(fig_latency)


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
    protocol_config:     Optional[ProtocolConfig] = None,
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

    Fault sweeps remain supported for non-DRAIN scenarios.  The current
    per-port DRAIN implementation is intentionally limited to the standard
    fault-free 4x4 mesh, so DRAIN-enabled fault sweeps with nonzero faults
    raise an explicit unsupported-topology error.

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
    if protocol_config is None:
        protocol_config = build_default_protocol_config()
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
                    protocol_config=protocol_config,
                )

                if num_faults > 0:
                    mesh.inject_faults(num_faults, fault_rng)

                detector = DeadlockDetector()

                for cycle in range(cycles):
                    mesh.inject_random_packets(cycle, traffic_rng)
                    mesh.step(cycle)
                    detector.check(mesh, cycle)

                for cycle in range(cycles, cycles + cooldown_cycles):
                    mesh.inject_random_packets(cycle, traffic_rng, generate_new_traffic=False)
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

    out_prefix = "outputs/ijr"
    print("Sweeping INJECTION_RATE from 0.05 to 0.50 (step 0.025, averaged over 5 seeds) ...")
    sweep = sweep_injection_rates(
        chiplet_specs=specs,
        cycles=20000,
        rate_min=0.001,
        rate_max=0.026, #0.031,
        rate_step=0.005,
        seed=42, # 42,
        routing_seed=0,
        num_seeds=4,
    )
    print("Generating sweep plots ...")
    plot_injection_sweep(sweep, out_prefix=out_prefix, show=False)
    plot_latency_sweep_all_scenarios(sweep, out_prefix=out_prefix, show=False)
    plot_transaction_completion_sweep_all_scenarios(
        sweep,
        out_prefix=out_prefix,
        show=False,
    )
    plot_req_resp_latency_sweep_all_scenarios(
        sweep,
        out_prefix=out_prefix,
        show=False,
    )
    text_out_file = f"{out_prefix}_summary.txt"
    write_injection_sweep_summary_text(
        sweep,
        out_file=text_out_file,
    )
    drain_table_out_file = f"{out_prefix}_drain_turn_table.txt"
    drain_table_mesh = InterposerMesh(
        chiplet_specs=specs,
        routing_mode=ROUTING_RANDOM_ADAPTIVE,
        routing_seed=0,
        drain_enabled=True,
    )
    write_drain_turn_table_text(
        drain_table_mesh,
        out_file=drain_table_out_file,
    )
    
    plot_message_class_sweep(sweep, scenario_label = "random_adaptive_with_drain", out_prefix = out_prefix, show=False)
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
