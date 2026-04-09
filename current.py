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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── Global defaults ──────────────────────────────────────────────────────────

MESH_ROWS          = 4
MESH_COLS          = 4
NUM_NORMAL_VCS     = 1   # normal VCs per router port; escape VC is always 1 extra
DRAIN_ESCAPE_ENTRY_PROB           = 0.3
TURN_RESTRICTED_ESCAPE_ENTRY_PROB = 0.3
ESCAPE_ENTRY_PROB  = DRAIN_ESCAPE_ENTRY_PROB  # default for single-mesh runs
DRAIN_PERIOD       = 50
DRAIN_WINDOW_HOPS  = 1
INJECTION_RATE     = 0.75

# ─── Chiplet type identifiers ─────────────────────────────────────────────────

CHIPLET_CPU = "CPU"
CHIPLET_GPU = "GPU"

# CPU chiplet defaults
CPU_VC_CAPACITY      = 8
CPU_INJECTION_RATE   = INJECTION_RATE   # uniform Bernoulli per cycle

# GPU chiplet defaults (4× more VC capacity than CPU, bursty traffic)
GPU_VC_CAPACITY      = 32              # 4× CPU_VC_CAPACITY
GPU_BURST_RATE       = 8 * INJECTION_RATE            # ~8× CPU rate during a burst window
GPU_QUIET_RATE       = INJECTION_RATE/10            # near-idle between bursts
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
    current_node:        str  = ""      # ID of the node currently holding the packet
    hops:                int  = 0       # interposer hops taken so far
    in_escape_vc:        bool = False   # True once packet has entered the escape VC
    delivered_cycle:     Optional[int] = None


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
    """FIFO virtual channel buffer with a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity:    int          = capacity
        self.fifo_queue:  List[Packet] = []

    # ── predicates ────────────────────────────────────────────────────────────

    def can_push(self) -> bool:
        return len(self.fifo_queue) < self.capacity

    def can_pop(self) -> bool:
        return len(self.fifo_queue) > 0

    def is_full(self)  -> bool:
        return len(self.fifo_queue) == self.capacity

    def is_empty(self) -> bool:
        return len(self.fifo_queue) == 0

    def occupancy(self) -> int:
        return len(self.fifo_queue)

    # ── operations ────────────────────────────────────────────────────────────

    def push(self, packet: Packet) -> bool:
        """Enqueue packet; returns True on success, False if full."""
        if not self.can_push():
            return False
        self.fifo_queue.append(packet)
        return True

    def pop(self) -> Packet:
        """Dequeue and return the head packet (caller must check can_pop first)."""
        return self.fifo_queue.pop(0)

    def peek(self) -> Optional[Packet]:
        """Return the head packet without removing it."""
        return self.fifo_queue[0] if self.fifo_queue else None

    def __repr__(self) -> str:
        return f"VC(capacity={self.capacity}, occupancy={self.occupancy()})"


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary Router
# ═══════════════════════════════════════════════════════════════════════════════

class BoundaryRouter:
    """
    Connects a Chiplet to its attached InterposerRouter.

    Outbound path  (chiplet → interposer):
        Chiplet calls inject_packet() → packet enters a normal VC.
        InterposerMesh.step() drains these VCs into the interposer router.

    Inbound path   (interposer → chiplet):
        InterposerMesh.step() calls receive_packet() when a packet arrives at
        the destination interposer router; the boundary router records it and
        the chiplet's receive_packet() is called.
    """

    def __init__(
        self,
        router_id:      str,
        chiplet_name:   str,
        vc_capacity:    int,
        num_normal_vcs: int,
    ) -> None:
        self.router_id:      str = router_id
        self.chiplet_name:   str = chiplet_name
        self.vc_capacity:    int = vc_capacity
        self.num_normal_vcs: int = num_normal_vcs

        # Outbound buffers: packets waiting to enter the interposer
        self.normal_vcs: List[VC] = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc:  VC       = VC(vc_capacity)

        # ID of the interposer router this boundary router feeds into
        self.attached_ir: Optional[str] = None

        # Tracking
        self.injected_packets: List[Packet] = []   # all packets ever injected
        self.received_packets: List[Packet] = []   # all packets ever delivered

    # ── outbound (chiplet → interposer) ───────────────────────────────────────

    def inject_packet(
        self,
        packet: Packet,
        rng: Optional[random.Random] = None,
    ) -> bool:
        """
        Try to enqueue packet into a normal VC.
        If multiple normal VCs exist, choose a random first choice so packets
        are split evenly in expectation; then fall back to other VCs if needed.
        Returns True on success; False if all normal VCs are full.
        """
        if not self.normal_vcs:
            return False

        if rng is None:
            rng = random

        order = list(range(len(self.normal_vcs)))
        if len(order) > 1:
            first = rng.randrange(len(order))
            order = [first] + [i for i in order if i != first]

        for i in order:
            vc = self.normal_vcs[i]
            if vc.push(packet):
                packet.current_node = self.router_id
                self.injected_packets.append(packet)
                return True
        return False

    def has_outbound_packet(self) -> bool:
        return (
            any(not vc.is_empty() for vc in self.normal_vcs)
            or not self.escape_vc.is_empty()
        )

    def pop_outbound_packet(self) -> Optional[Packet]:
        """
        Remove and return the head packet from the first non-empty VC.
        Normal VCs are checked before the escape VC.
        """
        for vc in self.normal_vcs:
            if vc.can_pop():
                return vc.pop()
        if self.escape_vc.can_pop():
            return self.escape_vc.pop()
        return None

    def push_back_outbound(self, packet: Packet) -> None:
        """Re-insert a packet that could not be forwarded (IR buffers full)."""
        target = self.escape_vc if packet.in_escape_vc else self.normal_vcs[0]
        target.fifo_queue.insert(0, packet)   # put back at the head

    # ── inbound (interposer → chiplet) ────────────────────────────────────────

    def receive_packet(self, packet: Packet, cycle: int) -> None:
        """Record arrival; called by the interposer when a packet is ejected."""
        packet.current_node  = self.router_id
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

    Holds VC buffers for packets in transit.  The InterposerMesh drives packet
    movement each cycle; the router itself is a passive data structure that
    exposes enqueue/dequeue helpers and topology information.

    Buffer layout
    -------------
      normal_vcs  – standard VCs; packets route freely.
      escape_vc   – one escape VC per router; supports DRAIN-style deadlock
                    removal (packets in the escape VC follow a forced path
                    when drain is active).
    """

    def __init__(
        self,
        router_id:      str,
        coordinate:     Tuple[int, int],
        vc_capacity:    int,
        num_normal_vcs: int,
    ) -> None:
        self.router_id:      str              = router_id
        self.coordinate:     Tuple[int, int]  = coordinate     # (row, col)
        self.vc_capacity:    int              = vc_capacity
        self.num_normal_vcs: int              = num_normal_vcs

        # Packet buffers
        self.normal_vcs: List[VC] = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc:  VC       = VC(vc_capacity)

        # Topology: direction ("N"/"S"/"E"/"W") → neighbour router_id or None
        self.ir_neighbors: Dict[str, Optional[str]] = {}

        # IDs of boundary routers attached to this interposer router
        self.attached_boundary_routers: List[str] = []

    # ── topology helpers ──────────────────────────────────────────────────────

    def add_ir_neighbor(self, direction: str, ir_id: Optional[str]) -> None:
        self.ir_neighbors[direction] = ir_id

    def add_boundary_router(self, br_id: str) -> None:
        self.attached_boundary_routers.append(br_id)

    # ── buffer helpers ────────────────────────────────────────────────────────

    def enqueue(
        self,
        packet: Packet,
        use_escape: bool = False,
        rng: Optional[random.Random] = None,
    ) -> bool:
        """
        Try to enqueue packet.  If use_escape, target the escape VC;
        otherwise try a random first-choice normal VC (if multiple exist), then
        fall back to the others.
        Returns True on success.
        """
        if use_escape:
            return self.escape_vc.push(packet)

        if rng is None:
            rng = random

        order = list(range(len(self.normal_vcs)))
        if len(order) > 1:
            first = rng.randrange(len(order))
            order = [first] + [i for i in order if i != first]

        for i in order:
            vc = self.normal_vcs[i]
            if vc.push(packet):
                return True
        return False

    def has_packet(self) -> bool:
        return (
            any(not vc.is_empty() for vc in self.normal_vcs)
            or not self.escape_vc.is_empty()
        )

    def peek_head(self) -> Optional[Tuple[Packet, bool]]:
        """
        Return (head_packet, is_from_escape_vc) from the first non-empty VC
        without removing it.  Normal VCs are checked before escape VC.
        Returns None if all VCs are empty.
        """
        for vc in self.normal_vcs:
            if vc.can_pop():
                return vc.peek(), False
        if self.escape_vc.can_pop():
            return self.escape_vc.peek(), True
        return None

    def pop_head(self) -> Optional[Tuple[Packet, bool]]:
        """
        Remove and return (head_packet, is_from_escape_vc) from the first
        non-empty VC.  Normal VCs are checked before the escape VC.
        Returns None if all VCs are empty.
        """
        for vc in self.normal_vcs:
            if vc.can_pop():
                return vc.pop(), False
        if self.escape_vc.can_pop():
            return self.escape_vc.pop(), True
        return None

    def total_occupancy(self) -> int:
        return (
            sum(vc.occupancy() for vc in self.normal_vcs)
            + self.escape_vc.occupancy()
        )

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
        Try to inject a packet through one of this chiplet's boundary routers
        (chosen at random for load balancing).
        Returns the Packet on success, None if all boundary-router VCs are full.
        """
        if not self.boundary_routers:
            return None

        brs = list(self.boundary_routers)
        rng.shuffle(brs)
        for br in brs:
            packet = Packet(
                packet_id=packet_id,
                src_chiplet=self.name,
                dst_chiplet=dst_chiplet,
                src_boundary_router=br.router_id,
                dst_boundary_router=dst_br_id,
                created_cycle=cycle,
                current_node=br.router_id,
            )
            if br.inject_packet(packet, rng=rng):
                return packet
        return None   # all boundary routers full

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
        self._drain_hops_remaining: int = 0
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

        self._packet_counter: int = 0

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
                    num_normal_vcs=self.num_normal_vcs,
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

    def _drain_active(self) -> bool:
        return self.drain_enabled and self._drain_hops_remaining > 0

    def _perform_drain_hop(self) -> int:
        """
        Move escape-VC head packets one hop along the offline DRAIN cycle.
        Movement is synchronized: all eligible routers move in the same cycle.
        """
        if not self._drain_cycle:
            return 0

        # Sources that currently have at least one escape packet.
        candidates = [
            rid for rid, router in self.routers.items() if router.escape_vc.can_pop()
        ]
        if not candidates:
            return 0

        # A destination can accept an incoming drain packet if occupancy after
        # its own departure (if any) still has room.
        will_depart = set(candidates)
        movable: List[str] = []
        for src in candidates:
            dst = self._drain_next[src]
            dst_router = self.routers[dst]
            projected = dst_router.escape_vc.occupancy() - (1 if dst in will_depart else 0)
            if projected < dst_router.escape_vc.capacity:
                movable.append(src)

        staged: List[Tuple[str, Packet]] = []
        for src in movable:
            popped = self.routers[src].escape_vc.pop()
            staged.append((self._drain_next[src], popped))

        moves = 0
        for dst, packet in staged:
            self.routers[dst].escape_vc.push(packet)
            packet.current_node = dst
            packet.hops += 1
            moves += 1
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
            dirs = self._minimal_directions(current_rid, dst_rid)
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
        toward dst_rid has a full target VC.

        Used by the deadlock detector's wait-for graph:
        - For deterministic routing (XY/YX) there is only one minimal direction,
          so this reduces to a single check.
        - For adaptive routing a packet is not truly blocked unless ALL
          minimal directions are full; checking just one would give false
          positives.
        """
        if self.routing_mode in (ROUTING_XY, ROUTING_YX):
            nxt = self.compute_next_hop(current_rid, dst_rid)
            if nxt is None:
                return False
            nxt_router = self.routers[nxt]
            return all(vc.is_full() for vc in nxt_router.normal_vcs)

        # Adaptive variants: use the same direction set the router actually uses
        # so a packet is only considered blocked when all its valid next hops are full.
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
            nxt_router = self.routers[nxt]
            if any(not vc.is_full() for vc in nxt_router.normal_vcs):
                return False   # at least one direction has room → not blocked
        return True   # every minimal direction is full → truly blocked

    # ── traffic injection ─────────────────────────────────────────────────────

    def _next_packet_id(self) -> int:
        pid = self._packet_counter
        self._packet_counter += 1
        return pid

    def inject_random_packets(
        self,
        cycle: int,
        rng:   random.Random,
    ) -> int:
        """
        Each chiplet independently decides whether to inject a packet this
        cycle using its own traffic model:
          - CPUChiplet: uniform Bernoulli with its ``injection_rate``.
          - GPUChiplet: bursty — high rate during burst windows, low rate
            between bursts; the chiplet advances its state machine here.

        The destination chiplet is chosen uniformly at random (never self),
        and one of the destination chiplet's BRs is picked at random.

        Returns the count of successfully injected packets this cycle.
        """
        chiplet_names = list(self.chiplets.keys())
        injected = 0
        for src_name, chiplet in self.chiplets.items():
            rate = chiplet.current_injection_rate(cycle, rng)
            if rng.random() >= rate:
                continue
            dst_name    = rng.choice([n for n in chiplet_names if n != src_name])
            dst_chiplet = self.chiplets[dst_name]
            if not dst_chiplet.boundary_routers:
                continue
            dst_br = rng.choice(dst_chiplet.boundary_routers)
            pkt = chiplet.generate_packet(
                packet_id=self._next_packet_id(),
                dst_chiplet=dst_name,
                dst_br_id=dst_br.router_id,
                cycle=cycle,
                rng=rng,
            )
            if pkt is not None:
                injected += 1
        return injected

    # ── DRAIN scheduling ──────────────────────────────────────────────────────

    def trigger_drain(self) -> None:
        """Start a DRAIN window in which escape-VC packets follow the offline cycle."""
        if not self.drain_enabled:
            return
        self._drain_hops_remaining = self.drain_window_hops

    # ── simulation step ───────────────────────────────────────────────────────

    def step(self, cycle: int) -> int:
        """
        Advance the network by one cycle.  Returns the number of packet
        movements that occurred (useful for deadlock detection).

        Phase 1 – Boundary router outbound VCs → attached interposer router.
                  One packet per boundary router per cycle.

        Phase 2 – Router movement:
                  - During a DRAIN window, escape-VC packets move first using
                    the offline DRAIN cycle (synchronized circulation).
                  - Non-escape packets always use normal routing_mode.
                  - Escape packets outside DRAIN windows also use routing_mode.
        """
        moves = 0
        if self.drain_enabled and cycle > 0 and cycle % self.drain_period == 0:
            self.trigger_drain()

        # ── Phase 1: BR outbound → IR ─────────────────────────────────────
        for br in self.boundary_routers.values():
            if not br.has_outbound_packet():
                continue
            ir = self.routers[br.attached_ir]      # type: ignore[index]
            packet = br.pop_outbound_packet()
            if packet is None:
                continue
            use_escape = packet.in_escape_vc
            if not use_escape and self._routing_rng.random() < self.escape_entry_prob:
                packet.in_escape_vc = True
                use_escape = True
            if ir.enqueue(packet, use_escape=use_escape, rng=self._routing_rng):
                packet.current_node = ir.router_id
                packet.hops += 1
                moves += 1
            else:
                # IR buffers full; return packet to the front of the BR queue
                br.push_back_outbound(packet)

        # ── Phase 2a: synchronized DRAIN circulation for escape VCs ───────
        drain_this_cycle = self._drain_active()
        if drain_this_cycle:
            moves += self._perform_drain_hop()
            self._drain_hops_remaining -= 1

        # ── Phase 2b: hop through the interposer (normal routing) ─────────
        # Stage all transfers before committing to prevent double-movement.
        transfers: List[Tuple[str, Packet, bool]] = []   # (dst_rid, pkt, escape)

        for rid, router in self.routers.items():
            result = router.peek_head()
            if result is None:
                continue
            packet, is_escape = result
            if is_escape and drain_this_cycle:
                # Escape packets are handled by the DRAIN circulation step.
                continue

            dst_br_id = packet.dst_boundary_router
            dst_ir_id = self.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]

            if rid == dst_ir_id:
                # ── Eject: packet has reached its destination IR ───────────
                router.pop_head()
                dst_br = self.boundary_routers[dst_br_id]
                dst_br.receive_packet(packet, cycle)
                self.chiplets[dst_br.chiplet_name].receive_packet(packet)
                moves += 1

            else:
                # ── Forward: compute next hop and check capacity ───────────
                nxt_rid = self.compute_next_hop_for_packet(
                    rid, dst_ir_id, is_escape=is_escape
                )
                if nxt_rid is None:
                    continue   # should not happen in a connected mesh
                nxt_router = self.routers[nxt_rid]
                use_escape_next = is_escape
                if not use_escape_next and self._routing_rng.random() < self.escape_entry_prob:
                    use_escape_next = True
                can_fit = (
                    not nxt_router.escape_vc.is_full()
                    if use_escape_next
                    else any(not vc.is_full() for vc in nxt_router.normal_vcs)
                )
                if can_fit:
                    router.pop_head()
                    if use_escape_next and not packet.in_escape_vc:
                        packet.in_escape_vc = True
                    transfers.append((nxt_rid, packet, use_escape_next))
                # else: head-of-line blocked; packet stays put this cycle

        # Commit all staged transfers
        for nxt_rid, packet, is_escape in transfers:
            self.routers[nxt_rid].enqueue(packet, use_escape=is_escape, rng=self._routing_rng)
            packet.current_node = nxt_rid
            packet.hops += 1
            moves += 1

        return moves

    # ── topology queries ──────────────────────────────────────────────────────

    def get_router(self, row: int, col: int) -> InterposerRouter:
        return self.routers[self.router_id(row, col)]

    def get_chiplet(self, row: int, col: int) -> Chiplet:
        return self.chiplets[self.chiplet_name(row, col)]

    def get_boundary_router(self, row: int, chip_col: int, br_idx: int = 0) -> BoundaryRouter:
        return self.boundary_routers[self.br_id(row, chip_col, br_idx)]

    # ── network-wide statistics ───────────────────────────────────────────────

    def all_in_flight(self) -> int:
        """Total packets currently inside the network (IRs + BRs)."""
        total = 0
        for router in self.routers.values():
            total += router.total_occupancy()
        for br in self.boundary_routers.values():
            total += sum(vc.occupancy() for vc in br.normal_vcs)
            total += br.escape_vc.occupancy()
        return total

    def delivered_stats(self) -> Dict[str, float]:
        """Aggregate latency and hop-count statistics across all delivered packets."""
        total_injected  = sum(len(br.injected_packets) for br in self.boundary_routers.values())
        total_delivered = 0
        total_latency   = 0
        total_hops      = 0
        for chiplet in self.chiplets.values():
            for pkt in chiplet.received_packets:
                total_delivered += 1
                if pkt.delivered_cycle is not None:
                    total_latency += pkt.delivered_cycle - pkt.created_cycle
                total_hops += pkt.hops
        avg_lat  = total_latency / total_delivered if total_delivered else 0.0
        avg_hops = total_hops    / total_delivered if total_delivered else 0.0
        return {
            "injected":     total_injected,
            "delivered":    total_delivered,
            "avg_latency":  round(avg_lat,  2),
            "avg_hops":     round(avg_hops, 2),
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
            mesh.inject_random_packets(cycle, rate, rng)
            moves  = mesh.step(cycle)
            report = detector.check(mesh, cycle, moves)
            if report.detected:
                # handle / log / trigger DRAIN ...
    """

    def __init__(
        self,
        persist_cycles: int = 3,
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
            result = router.peek_head()
            if result is None:
                continue

            packet, is_escape = result
            dst_br_id = packet.dst_boundary_router
            dst_ir_id = mesh.boundary_routers[dst_br_id].attached_ir  # type: ignore[index]

            # Packet can eject at this router — not blocking
            if rid == dst_ir_id:
                continue

            # Use all_minimal_hops_blocked so that adaptive routing packets are
            # only counted as blocked when every valid next-hop direction is full
            # (not just the one randomly chosen this cycle).
            if is_escape:
                nxt_rid = mesh.compute_next_hop_for_packet(
                    rid, dst_ir_id, is_escape=True
                )
                blocked  = nxt_rid is not None and mesh.routers[nxt_rid].escape_vc.is_full()
            else:
                blocked = mesh.all_minimal_hops_blocked(rid, dst_ir_id)

            if blocked:
                # Record which router this one is waiting for.
                # For adaptive routing, use compute_next_hop for the WFG edge
                # (a representative direction; the stall counter is the reliable
                # signal when all directions are equally blocked).
                nxt_rid = mesh.compute_next_hop_for_packet(
                    rid, dst_ir_id, is_escape=is_escape
                )
                if nxt_rid is not None:
                    wait_for[rid] = nxt_rid

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
                result = mesh.routers[rid].peek_head()
                if result is not None:
                    pkt_ids.append(result[0].packet_id)
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
    chiplet_specs:      Optional[List[ChipletSpec]] = None,
    cpu_injection_rate: float                      = CPU_INJECTION_RATE,
    seed:               Optional[int]              = None,
    routing_seed:       Optional[int]              = None,
    drain_escape_entry_prob: float                 = DRAIN_ESCAPE_ENTRY_PROB,
    turn_restricted_escape_entry_prob: float       = TURN_RESTRICTED_ESCAPE_ENTRY_PROB,
    drain_period:       int                        = DRAIN_PERIOD,
    drain_window_hops:  int                        = DRAIN_WINDOW_HOPS,
    verbose:            bool                       = False,
) -> Dict[str, Tuple[Dict[str, float], DeadlockDetector]]:
    """
    Run routing-algorithm comparison scenarios for `cycles` cycles.

    Parameters
    ----------
    cycles             : Number of simulation cycles per scenario.
    chiplet_specs      : List of ChipletSpec objects describing the system.
                         Defaults to ``build_image_layout()`` (4 corner GPUs
                         + 1 central CPU sharing 2 IRs with the top GPUs).
    cpu_injection_rate : Injection probability per cycle for CPU chiplets.
                         GPU chiplets use their own bursty logic.
    seed               : RNG seed for traffic injection (None → random).
    routing_seed       : RNG seed for routing decisions (None → random).
    drain_escape_entry_prob : Per-hop escape-entry probability used by
                              ``random_adaptive_with_drain``.
    turn_restricted_escape_entry_prob : Per-hop escape-entry probability used by
                                        ``random_adaptive_turn_restricted``.
    drain_period       : DRAIN window triggered every N cycles.
    drain_window_hops  : Synchronized DRAIN hops per window.
    verbose            : Print a line each time a deadlock is detected.

    Returns
    -------
    Dict mapping scenario label → (stats dict, DeadlockDetector).
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()
    results: Dict[str, Tuple[Dict[str, float], DeadlockDetector]] = {}
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 32)
    if routing_seed is None:
        routing_seed = random.SystemRandom().randrange(1 << 32)

    scenarios = [
        # (label,                              routing_mode,              drain,  escape_prob,       num_vcs,        seed_offset)
        ("xy",                                 ROUTING_XY,                False,  0.0,               2,              0),
        ("yx",                                 ROUTING_YX,                False,  0.0,               2,              1),
        ("random_adaptive_without_drain",      ROUTING_RANDOM_ADAPTIVE,   False,  0.0,               2,              2),
        ("random_adaptive_with_drain",         ROUTING_RANDOM_ADAPTIVE,   True,   drain_escape_entry_prob, NUM_NORMAL_VCS, 2),
        ("random_adaptive_turn_restricted",    ROUTING_RANDOM_ADAPTIVE_TR,False,  turn_restricted_escape_entry_prob, NUM_NORMAL_VCS, 3),
        ("shortest_path",                      ROUTING_RANDOM_ADAPTIVE,   False,  0.0,               40,             4),
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
            num_normal_vcs=scenario_num_vcs,
            cpu_injection_rate=cpu_injection_rate,
        )
        detector = DeadlockDetector()

        for cycle in range(cycles):
            mesh.inject_random_packets(cycle, rng)
            mesh.step(cycle)
            report = detector.check(mesh, cycle)

        results[label] = (mesh.delivered_stats(), detector)

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
    "shortest_path":                   "Shortest Path (20 VCs)",
}


def sweep_injection_rates(
    chiplet_specs:      Optional[List[ChipletSpec]] = None,
    cycles:             int   = 5000,
    rate_min:           float = 0.10,
    rate_max:           float = 1.00,
    rate_step:          float = 0.05,
    seed:               int   = 42,
    routing_seed:       int   = 0,
    drain_escape_entry_prob: float = DRAIN_ESCAPE_ENTRY_PROB,
    turn_restricted_escape_entry_prob: float = TURN_RESTRICTED_ESCAPE_ENTRY_PROB,
    drain_period:       int   = DRAIN_PERIOD,
    drain_window_hops:  int   = DRAIN_WINDOW_HOPS,
) -> Dict[str, Dict[float, int]]:
    """
    Sweep CPU INJECTION_RATE and collect injected-packet counts per scenario.

    Returns
    -------
    sweep[label][rate] = injected_packet_count
    """
    if chiplet_specs is None:
        chiplet_specs = build_image_layout()

    rates = [
        round(rate_min + i * rate_step, 10)
        for i in range(int(round((rate_max - rate_min) / rate_step)) + 1)
        if rate_min + i * rate_step <= rate_max + 1e-9
    ]

    sweep: Dict[str, Dict[float, int]] = {label: {} for label in SCENARIO_LABELS}
    total = len(rates)
    for idx, rate in enumerate(rates):
        print(f"  rate={rate:.2f} ({idx + 1}/{total})", flush=True)
        results = run_simulation(
            cycles=cycles,
            chiplet_specs=chiplet_specs,
            cpu_injection_rate=rate,
            seed=seed,
            routing_seed=routing_seed,
            drain_escape_entry_prob=drain_escape_entry_prob,
            turn_restricted_escape_entry_prob=turn_restricted_escape_entry_prob,
            drain_period=drain_period,
            drain_window_hops=drain_window_hops,
        )
        for label in SCENARIO_LABELS:
            sweep[label][rate] = int(results[label][0]["injected"])
    return sweep


def plot_injection_sweep(
    sweep:      Dict[str, Dict[float, int]],
    out_prefix: str  = "injection_rate_sweep",
    show:       bool = True,
) -> None:
    """
    Plot INJECTION_RATE vs # injected packets.

    Outputs:
    - one figure per scenario
    - one combined random-adaptive comparison figure
    - one random_adaptive_with_drain vs XY figure
    - one random_adaptive_with_drain vs YX figure
    - one multi-scenario comparison figure
    """
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})

    for label in SCENARIO_LABELS:
        rates = sorted(sweep[label].keys())
        injected_counts = [sweep[label][rate] for rate in rates]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rates, injected_counts, marker="o", linewidth=2, color="steelblue")
        ax.set_xlabel("INJECTION_RATE")
        ax.set_ylabel("# Packets Injected")
        ax.set_title(
            f"{SCENARIO_DISPLAY.get(label, label)}\n"
            "# Packets Injected vs INJECTION_RATE"
        )
        ax.set_xlim(min(rates) - 0.02, max(rates) + 0.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        out_file = f"{out_prefix}_{label}.png"
        fig.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")
        if show:
            plt.show()
        plt.close(fig)

    # Combined random-adaptive comparison on one graph
    ra_wo = "random_adaptive_without_drain"
    ra_w  = "random_adaptive_with_drain"
    if ra_wo in sweep and ra_w in sweep:
        rates_wo = sorted(sweep[ra_wo].keys())
        rates_w  = sorted(sweep[ra_w].keys())
        counts_wo = [sweep[ra_wo][rate] for rate in rates_wo]
        counts_w  = [sweep[ra_w][rate] for rate in rates_w]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            rates_wo,
            counts_wo,
            marker="o",
            linewidth=2,
            color="steelblue",
            label=SCENARIO_DISPLAY.get(ra_wo, ra_wo),
        )
        ax.plot(
            rates_w,
            counts_w,
            marker="s",
            linewidth=2,
            color="darkorange",
            label=SCENARIO_DISPLAY.get(ra_w, ra_w),
        )
        ax.set_xlabel("INJECTION_RATE")
        ax.set_ylabel("# Packets Injected")
        ax.set_title("Random Adaptive Comparison\n# Packets Injected vs INJECTION_RATE")
        ax.set_xlim(min(rates_wo + rates_w) - 0.02, max(rates_wo + rates_w) + 0.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        out_file = f"{out_prefix}_random_adaptive_comparison.png"
        fig.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_pair(a: str, b: str, suffix: str, title: str) -> None:
        if a not in sweep or b not in sweep:
            return
        rates_a = sorted(sweep[a].keys())
        rates_b = sorted(sweep[b].keys())
        counts_a = [sweep[a][rate] for rate in rates_a]
        counts_b = [sweep[b][rate] for rate in rates_b]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            rates_a,
            counts_a,
            marker="o",
            linewidth=2,
            color="darkorange",
            label=SCENARIO_DISPLAY.get(a, a),
        )
        ax.plot(
            rates_b,
            counts_b,
            marker="s",
            linewidth=2,
            color="steelblue",
            label=SCENARIO_DISPLAY.get(b, b),
        )
        ax.set_xlabel("INJECTION_RATE")
        ax.set_ylabel("# Packets Injected")
        ax.set_title(f"{title}\n# Packets Injected vs INJECTION_RATE")
        ax.set_xlim(min(rates_a + rates_b) - 0.02, max(rates_a + rates_b) + 0.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        out_file = f"{out_prefix}_{suffix}.png"
        fig.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")
        if show:
            plt.show()
        plt.close(fig)

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

    # Requested multi-scenario comparison on one graph
    combo_labels = [
        "random_adaptive_with_drain",
        "random_adaptive_without_drain",
        "yx",
        "shortest_path",
        "random_adaptive_turn_restricted",
    ]
    if all(label in sweep for label in combo_labels):
        colors = ["darkorange", "steelblue", "seagreen", "firebrick", "slategray"]
        markers = ["o", "s", "^", "D", "x"]
        fig, ax = plt.subplots(figsize=(9, 6))
        all_rates: List[float] = []

        for label, color, marker in zip(combo_labels, colors, markers):
            rates = sorted(sweep[label].keys())
            counts = [sweep[label][rate] for rate in rates]
            all_rates.extend(rates)
            ax.plot(
                rates,
                counts,
                marker=marker,
                linewidth=2,
                color=color,
                label=SCENARIO_DISPLAY.get(label, label),
            )

        ax.set_xlabel("INJECTION_RATE")
        ax.set_ylabel("# Packets Injected")
        ax.set_title("Scenario Comparison\n# Packets Injected vs INJECTION_RATE")
        ax.set_xlim(min(all_rates) - 0.02, max(all_rates) + 0.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        out_file = f"{out_prefix}_scenario_comparison.png"
        fig.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")
        if show:
            plt.show()
        plt.close(fig)


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
    # results = run_simulation(cycles=50000, chiplet_specs=specs)
    # for label in LABELS:
    #     stats, det = results[label]
    #     print(f"\n[{label}] \ninjected={stats['injected']}  "
    #           f"delivered={stats['delivered']}  "
    #           f"avg_latency={stats['avg_latency']}  "
    #           f"avg_hops={stats['avg_hops']}")
    #     print(f"  {det.summary()}")

    print("Sweeping INJECTION_RATE from 0.10 to 1.00 (step 0.05) ...")
    sweep = sweep_injection_rates(
        chiplet_specs=specs,
        cycles=5000,
        rate_min=0.10,
        rate_max=1.00,
        rate_step=0.05,
        seed=42,
        routing_seed=0,
    )
    print("Generating sweep plots ...")
    plot_injection_sweep(sweep, out_prefix="injection_rate_sweep", show=False)
    print()
