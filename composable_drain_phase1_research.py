from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import csv
import random


MESHDIM = (4, 4)
VCCAPACITY = 8
NUMBOUNDARYROUTERS = 2


@dataclass
class Packet:
    packet_id: int
    src_chiplet: str
    dst_chiplet: str
    src_boundary_router: str
    dst_boundary_router: str
    created_cycle: int
    current_node: str = ""
    hops: int = 0
    in_escape_vc: bool = False
    ever_entered_escape: bool = False
    delivered_cycle: Optional[int] = None
    route: List[str] = field(default_factory=list)
    route_index: int = 0
    blocked_cycles: int = 0
    times_promoted: int = 0


@dataclass
class Link:
    src: str
    dst: str
    latency: int = 1


@dataclass
class ExperimentConfig:
    scenario_name: str = "unnamed_scenario"
    rows: int = 4
    columns: int = 4
    vc_capacity: int = 2
    num_normal_vcs: int = 1
    enable_drain: bool = True
    drain_interval: int = 6
    drain_duration: int = 2
    drain_block_threshold: int = 4
    drain_stall_threshold: int = 3
    hard_promotion_threshold: int = 8
    deadlock_no_progress_threshold: int = 12
    max_cycles: int = 120
    drain_extra_cycles: int = 20
    retry_failed_injections: bool = True


class VC:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.fifo_queue: List[Packet] = []

    def can_push(self) -> bool:
        return len(self.fifo_queue) < self.capacity

    def push(self, packet: Packet) -> bool:
        if not self.can_push():
            return False
        self.fifo_queue.append(packet)
        return True

    def can_pop(self) -> bool:
        return len(self.fifo_queue) > 0

    def pop(self) -> Packet:
        return self.fifo_queue.pop(0)

    def peek(self) -> Optional[Packet]:
        return self.fifo_queue[0] if self.fifo_queue else None

    def occupancy(self) -> int:
        return len(self.fifo_queue)

    def is_full(self) -> bool:
        return len(self.fifo_queue) == self.capacity

    def is_empty(self) -> bool:
        return len(self.fifo_queue) == 0


class BoundaryRouter:
    def __init__(self, router_id: str, chiplet_name: str, vc_capacity: int, num_normal_vcs: int):
        self.router_id = router_id
        self.chiplet_name = chiplet_name
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        self.normal_vcs = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc = VC(vc_capacity)
        self.injected_packets: List[Packet] = []
        self.received_packets: List[Packet] = []
        self.attached_ir: Optional[str] = None

    def inject_packet(self, packet: Packet) -> bool:
        for vc in self.normal_vcs:
            if vc.push(packet):
                packet.current_node = self.router_id
                self.injected_packets.append(packet)
                return True
        return False

    def receive_packet(self, packet: Packet, cycle: int):
        packet.current_node = self.router_id
        packet.delivered_cycle = cycle
        self.received_packets.append(packet)


class InterposerRouter:
    def __init__(self, router_id: str, coordinate: Tuple[int, int], vc_capacity: int, num_normal_vcs: int):
        self.router_id = router_id
        self.coordinate = coordinate
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        self.normal_vcs = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc = VC(vc_capacity)
        self.attached_boundary_routers: List[str] = []
        self.ir_neighbors: Dict[str, Optional[str]] = {}

    def add_ir_neighbor(self, direction: str, ir_neighbor_id: Optional[str]):
        self.ir_neighbors[direction] = ir_neighbor_id

    def add_boundary_router(self, boundary_router_id: str):
        self.attached_boundary_routers.append(boundary_router_id)


class Chiplet:
    def __init__(
        self,
        name: str,
        chiplet_type: str,
        num_boundary_routers: int,
        vc_capacity: int,
        num_normal_vcs: int,
        mesh_coord: Tuple[int, int],
    ):
        self.name = name
        self.chiplet_type = chiplet_type
        self.mesh_coord = mesh_coord
        self.boundary_routers: Dict[str, BoundaryRouter] = {}
        self.create_boundary_routers(num_boundary_routers, vc_capacity, num_normal_vcs)

    def create_boundary_routers(self, num_boundary_routers: int, vc_capacity: int, num_normal_vcs: int):
        for i in range(num_boundary_routers):
            local_id = f"BR_{i}"
            full_id = f"{self.name}:{local_id}"
            boundary_router = BoundaryRouter(full_id, self.name, vc_capacity, num_normal_vcs)
            self.boundary_routers[full_id] = boundary_router

    def pick_boundary_router(self) -> BoundaryRouter:
        return random.choice(list(self.boundary_routers.values()))


class CPUChiplet(Chiplet):
    def __init__(self, name, num_boundary_routers, vc_capacity, num_normal_vcs, mesh_coord):
        super().__init__(name, "CPU", num_boundary_routers, vc_capacity, num_normal_vcs, mesh_coord)


class GPUChiplet(Chiplet):
    def __init__(self, name, num_boundary_routers, vc_capacity, num_normal_vcs, mesh_coord):
        super().__init__(name, "GPU", num_boundary_routers, vc_capacity, num_normal_vcs, mesh_coord)


class InterposerMesh:
    def __init__(self, rows: int, columns: int, vc_capacity: int, num_normal_vcs: int):
        self.rows = rows
        self.columns = columns
        self.routers: Dict[str, InterposerRouter] = {}
        self.links: List[Link] = []
        self.create_interposer_routers(vc_capacity, num_normal_vcs)
        self.create_mesh()

    def make_router_id(self, row: int, column: int) -> str:
        return f"IR_{row}_{column}"

    def create_interposer_routers(self, vc_capacity: int, num_normal_vcs: int):
        for row in range(self.rows):
            for column in range(self.columns):
                router_id = self.make_router_id(row, column)
                self.routers[router_id] = InterposerRouter(
                    router_id,
                    (row, column),
                    vc_capacity,
                    num_normal_vcs,
                )

    def create_mesh(self):
        for row in range(self.rows):
            for column in range(self.columns):
                router_id = self.make_router_id(row, column)
                router = self.routers[router_id]

                north = self.make_router_id(row - 1, column) if row > 0 else None
                south = self.make_router_id(row + 1, column) if row < self.rows - 1 else None
                west = self.make_router_id(row, column - 1) if column > 0 else None
                east = self.make_router_id(row, column + 1) if column < self.columns - 1 else None

                router.add_ir_neighbor("N", north)
                router.add_ir_neighbor("S", south)
                router.add_ir_neighbor("W", west)
                router.add_ir_neighbor("E", east)

                for neighbor in (north, south, west, east):
                    if neighbor is not None:
                        self.links.append(Link(router_id, neighbor))

    def get_interposer_router(self, row: int, column: int) -> InterposerRouter:
        return self.routers[self.make_router_id(row, column)]

    def xy_path(self, src_ir: str, dst_ir: str) -> List[str]:
        src_row, src_col = self.routers[src_ir].coordinate
        dst_row, dst_col = self.routers[dst_ir].coordinate
        current_row, current_col = src_row, src_col
        path = [src_ir]

        while current_col != dst_col:
            current_col += 1 if current_col < dst_col else -1
            path.append(self.make_router_id(current_row, current_col))

        while current_row != dst_row:
            current_row += 1 if current_row < dst_row else -1
            path.append(self.make_router_id(current_row, current_col))

        return path

    def snake_drain_cycle(self) -> List[str]:
        order = []
        for row in range(self.rows):
            cols = range(self.columns) if row % 2 == 0 else range(self.columns - 1, -1, -1)
            for col in cols:
                order.append(self.make_router_id(row, col))
        return order


class System:
    def __init__(
        self,
        rows: int,
        columns: int,
        vc_capacity: int,
        num_normal_vcs: int,
        enable_drain: bool = True,
        drain_interval: int = 10,
        drain_duration: int = 3,
        drain_block_threshold: int = 4,
        drain_stall_threshold: int = 3,
        hard_promotion_threshold: int = 8,
        deadlock_no_progress_threshold: int = 12,
    ):
        self.mesh = InterposerMesh(rows, columns, vc_capacity, num_normal_vcs)
        self.chiplets: Dict[str, Chiplet] = {}
        self.boundary_routers: Dict[str, BoundaryRouter] = {}
        self.packet_counter = 0
        self.cycle = 0
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        self.completed_packets: List[Packet] = []

        self.injection_attempts = 0
        self.injected_packet_count = 0
        self.failed_injection_count = 0

        self.enable_drain = enable_drain
        self.drain_interval = drain_interval
        self.drain_duration = drain_duration
        self.drain_block_threshold = drain_block_threshold
        self.drain_stall_threshold = drain_stall_threshold
        self.hard_promotion_threshold = hard_promotion_threshold

        self.no_progress_cycles = 0
        self.total_no_progress_cycles = 0
        self.deadlock_detected = False
        self.deadlock_no_progress_threshold = deadlock_no_progress_threshold

        self.promoted_packets_count = 0
        self.last_cycle_moves = 0
        self.last_cycle_promotions = 0
        self.last_cycle_blocked_heads = 0

        self.global_drain_cycle = self.mesh.snake_drain_cycle()
        self.global_drain_successor = {
            self.global_drain_cycle[i]: self.global_drain_cycle[(i + 1) % len(self.global_drain_cycle)]
            for i in range(len(self.global_drain_cycle))
        }

    def is_drain_cycle(self) -> bool:
        if not self.enable_drain:
            return False
        return (self.cycle % self.drain_interval) < self.drain_duration

    def add_chiplet(self, chiplet_name: str, chiplet_type: str, num_boundary_routers: int, chiplet_coord: Tuple[int, int]):
        if chiplet_type == "CPU":
            chiplet = CPUChiplet(chiplet_name, num_boundary_routers, self.vc_capacity, self.num_normal_vcs, chiplet_coord)
        elif chiplet_type == "GPU":
            chiplet = GPUChiplet(chiplet_name, num_boundary_routers, self.vc_capacity, self.num_normal_vcs, chiplet_coord)
        else:
            raise ValueError(f"Unknown chiplet type: {chiplet_type}")

        self.chiplets[chiplet_name] = chiplet

        attached_ir = self.mesh.make_router_id(*chiplet_coord)
        for br_id, br in chiplet.boundary_routers.items():
            br.attached_ir = attached_ir
            self.boundary_routers[br_id] = br
            self.mesh.routers[attached_ir].add_boundary_router(br_id)

    def create_default_chiplet_layout(self):
        self.add_chiplet("CPU0", "CPU", 1, (0, 0))
        self.add_chiplet("CPU1", "CPU", 1, (0, 3))
        self.add_chiplet("GPU0", "GPU", 1, (3, 0))
        self.add_chiplet("GPU1", "GPU", 1, (3, 3))

    def get_next_packet_id(self) -> int:
        packet_id = self.packet_counter
        self.packet_counter += 1
        return packet_id

    def build_normal_route(self, src_br: str, dst_br: str) -> List[str]:
        src_ir = self.boundary_routers[src_br].attached_ir
        dst_ir = self.boundary_routers[dst_br].attached_ir
        mesh_path = self.mesh.xy_path(src_ir, dst_ir)
        return [src_br] + mesh_path + [dst_br]

    def create_packet(self, src_chiplet: str, dst_chiplet: str, created_cycle: Optional[int] = None) -> Optional[Packet]:
        self.injection_attempts += 1
        created_cycle = self.cycle if created_cycle is None else created_cycle
        src_br = self.chiplets[src_chiplet].pick_boundary_router()
        dst_br = self.chiplets[dst_chiplet].pick_boundary_router()

        packet = Packet(
            packet_id=self.get_next_packet_id(),
            src_chiplet=src_chiplet,
            dst_chiplet=dst_chiplet,
            src_boundary_router=src_br.router_id,
            dst_boundary_router=dst_br.router_id,
            created_cycle=created_cycle,
            current_node=src_br.router_id,
        )
        packet.route = self.build_normal_route(src_br.router_id, dst_br.router_id)

        if src_br.inject_packet(packet):
            self.injected_packet_count += 1
            return packet

        self.failed_injection_count += 1
        return None

    def seed_packet_on_route(
        self,
        src_chiplet: str,
        dst_chiplet: str,
        current_node: str,
        route: List[str],
        route_index: int,
        use_escape: bool = False,
    ) -> bool:
        """
        Seeds a packet directly into a chosen node/VC.
        This is mainly for controlled synthetic experiments that create a cyclic dependency.
        """
        self.injection_attempts += 1
        src_br = self.chiplets[src_chiplet].pick_boundary_router()
        dst_br = self.chiplets[dst_chiplet].pick_boundary_router()

        packet = Packet(
            packet_id=self.get_next_packet_id(),
            src_chiplet=src_chiplet,
            dst_chiplet=dst_chiplet,
            src_boundary_router=src_br.router_id,
            dst_boundary_router=dst_br.router_id,
            created_cycle=self.cycle,
            current_node=current_node,
            route=list(route),
            route_index=route_index,
            in_escape_vc=use_escape,
        )

        if current_node in self.boundary_routers:
            router = self.boundary_routers[current_node]
        else:
            router = self.mesh.routers[current_node]

        target_vc = router.escape_vc if use_escape else router.normal_vcs[0]

        if target_vc.can_push():
            target_vc.push(packet)
            self.injected_packet_count += 1
            return True

        self.failed_injection_count += 1
        return False

    def _all_vcs(self):
        for br in self.boundary_routers.values():
            for vc in br.normal_vcs:
                yield br.router_id, False, vc
            yield br.router_id, True, br.escape_vc

        for ir in self.mesh.routers.values():
            for vc in ir.normal_vcs:
                yield ir.router_id, False, vc
            yield ir.router_id, True, ir.escape_vc

    def in_flight_packet_count(self) -> int:
        total = 0
        for _, _, vc in self._all_vcs():
            total += len(vc.fifo_queue)
        return total

    def network_empty(self) -> bool:
        return self.in_flight_packet_count() == 0

    def _packet_next_node(self, packet: Packet) -> Optional[str]:
        if self.enable_drain and packet.in_escape_vc:
            current = packet.current_node
            if current in self.boundary_routers:
                return self.boundary_routers[current].attached_ir
            if current == self.boundary_routers[packet.dst_boundary_router].attached_ir:
                return packet.dst_boundary_router
            return self.global_drain_successor[current]

        if packet.route_index + 1 >= len(packet.route):
            return None
        return packet.route[packet.route_index + 1]

    def _get_target_vc(self, packet: Packet, dst_node_id: str) -> Optional[VC]:
        if dst_node_id in self.boundary_routers:
            return None

        dst_router = self.mesh.routers[dst_node_id]
        if packet.in_escape_vc:
            if dst_router.escape_vc.can_push():
                return dst_router.escape_vc
            return None

        for vc in dst_router.normal_vcs:
            if vc.can_push():
                return vc
        return None

    def _promote_to_escape(self) -> int:
        if not self.is_drain_cycle():
            return 0

        promoted = 0
        routers = list(self.boundary_routers.values()) + list(self.mesh.routers.values())

        for router in routers:
            if not router.escape_vc.can_push():
                continue

            best_candidate_index = None
            best_packet = None

            for i, vc in enumerate(router.normal_vcs):
                pkt = vc.peek()
                if pkt is None:
                    continue

                eligible = (
                    pkt.blocked_cycles >= self.hard_promotion_threshold
                    or (
                        pkt.blocked_cycles >= self.drain_block_threshold
                        and self.no_progress_cycles >= self.drain_stall_threshold
                    )
                )

                if not eligible:
                    continue

                if best_packet is None or pkt.blocked_cycles > best_packet.blocked_cycles:
                    best_packet = pkt
                    best_candidate_index = i

            if best_packet is not None:
                router.normal_vcs[best_candidate_index].pop()
                best_packet.in_escape_vc = True
                best_packet.ever_entered_escape = True
                best_packet.times_promoted += 1
                best_packet.blocked_cycles = 0
                router.escape_vc.push(best_packet)
                promoted += 1

        self.promoted_packets_count += promoted
        return promoted

    def step(self):
        promotions = self._promote_to_escape()
        proposals = []

        for node_id, is_escape, vc in list(self._all_vcs()):
            pkt = vc.peek()
            if pkt is None:
                continue
            next_node = self._packet_next_node(pkt)
            if next_node is None:
                continue
            proposals.append((node_id, is_escape, vc, pkt, next_node))

        proposals.sort(
            key=lambda item: (
                0 if item[1] else 1,     # escape VC packets first
                -item[3].blocked_cycles, # then most-blocked packets
                item[3].packet_id,
            )
        )

        reserved = set()
        executed = []
        blocked_heads = 0

        for node_id, is_escape, vc, pkt, next_node in proposals:
            if next_node in self.boundary_routers:
                key = ("eject", next_node)
                if key in reserved:
                    pkt.blocked_cycles += 1
                    blocked_heads += 1
                    continue
                reserved.add(key)
                executed.append((node_id, is_escape, vc, pkt, next_node, None))
                continue

            target_vc = self._get_target_vc(pkt, next_node)
            if target_vc is None:
                pkt.blocked_cycles += 1
                blocked_heads += 1
                continue

            key = ("vc", id(target_vc))
            if key in reserved:
                pkt.blocked_cycles += 1
                blocked_heads += 1
                continue

            reserved.add(key)
            executed.append((node_id, is_escape, vc, pkt, next_node, target_vc))

        for node_id, is_escape, src_vc, pkt, next_node, target_vc in executed:
            src_vc.pop()
            pkt.current_node = next_node
            pkt.hops += 1
            pkt.blocked_cycles = 0

            if not pkt.in_escape_vc:
                pkt.route_index += 1

            if next_node in self.boundary_routers:
                self.boundary_routers[next_node].receive_packet(pkt, self.cycle + 1)
                self.completed_packets.append(pkt)
            else:
                target_vc.push(pkt)

        moves = len(executed)
        in_flight = self.in_flight_packet_count()

        if in_flight > 0 and moves == 0:
            self.no_progress_cycles += 1
            self.total_no_progress_cycles += 1
        else:
            self.no_progress_cycles = 0

        if in_flight > 0 and self.no_progress_cycles >= self.deadlock_no_progress_threshold:
            self.deadlock_detected = True

        self.last_cycle_moves = moves
        self.last_cycle_promotions = promotions
        self.last_cycle_blocked_heads = blocked_heads
        self.cycle += 1

        return {
            "cycle": self.cycle,
            "moves": moves,
            "promotions": promotions,
            "blocked_heads": blocked_heads,
            "in_flight": in_flight,
        }

    def run(self, cycles: int):
        history = []
        for _ in range(cycles):
            history.append(self.step())
        return history

    def stats(self) -> Dict[str, Optional[float]]:
        total_injected = self.injected_packet_count
        delivered = len(self.completed_packets)
        avg_latency = None
        max_latency = None
        avg_hops = None
        throughput = None
        escape_packet_fraction = None

        if delivered:
            latencies = [p.delivered_cycle - p.created_cycle for p in self.completed_packets]
            avg_latency = sum(latencies) / delivered
            max_latency = max(latencies)
            avg_hops = sum(p.hops for p in self.completed_packets) / delivered

        if self.cycle:
            throughput = delivered / self.cycle

        if total_injected:
            packets_that_ever_entered_escape = sum(1 for p in self.completed_packets if p.ever_entered_escape)
            for _, _, vc in self._all_vcs():
                packets_that_ever_entered_escape += sum(1 for p in vc.fifo_queue if p.ever_entered_escape)
            escape_packet_fraction = packets_that_ever_entered_escape / total_injected

        return {
            "cycle": self.cycle,
            "drain_enabled": self.enable_drain,
            "total_injection_attempts": self.injection_attempts,
            "total_injected_packets": total_injected,
            "failed_injections": self.failed_injection_count,
            "delivered_packets": delivered,
            "in_flight_packets": self.in_flight_packet_count(),
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "avg_hops": avg_hops,
            "throughput": throughput,
            "drain_mode": self.is_drain_cycle(),
            "escape_packet_fraction": escape_packet_fraction,
            "promoted_packets_count": self.promoted_packets_count,
            "no_progress_cycles": self.no_progress_cycles,
            "total_no_progress_cycles": self.total_no_progress_cycles,
            "deadlock_detected": self.deadlock_detected,
        }


def build_schedule_burst(start_cycle: int, flows: List[Tuple[str, str]], count_per_flow: int, spread: int = 0):
    schedule = defaultdict(list)

    if spread <= 0:
        for _ in range(count_per_flow):
            for src, dst in flows:
                schedule[start_cycle].append((src, dst))
    else:
        current_cycle = start_cycle
        for _ in range(count_per_flow):
            for src, dst in flows:
                schedule[current_cycle].append((src, dst))
                current_cycle += spread

    return schedule


def build_schedule_periodic(
    start_cycle: int,
    end_cycle: int,
    period: int,
    flows: List[Tuple[str, str]],
    packets_per_flow: int = 1,
):
    schedule = defaultdict(list)
    for cyc in range(start_cycle, end_cycle + 1, period):
        for _ in range(packets_per_flow):
            for src, dst in flows:
                schedule[cyc].append((src, dst))
    return schedule


def merge_schedules(*schedules):
    merged = defaultdict(list)
    for schedule in schedules:
        for cycle, flows in schedule.items():
            merged[cycle].extend(flows)
    return merged


def build_system_from_config(config: ExperimentConfig) -> System:
    system = System(
        rows=config.rows,
        columns=config.columns,
        vc_capacity=config.vc_capacity,
        num_normal_vcs=config.num_normal_vcs,
        enable_drain=config.enable_drain,
        drain_interval=config.drain_interval,
        drain_duration=config.drain_duration,
        drain_block_threshold=config.drain_block_threshold,
        drain_stall_threshold=config.drain_stall_threshold,
        hard_promotion_threshold=config.hard_promotion_threshold,
        deadlock_no_progress_threshold=config.deadlock_no_progress_threshold,
    )
    system.create_default_chiplet_layout()
    return system


def run_scheduled_experiment(config: ExperimentConfig, schedule, verbose: bool = False) -> System:
    system = build_system_from_config(config)
    pending_injections = deque()

    max_scheduled_cycle = max(schedule.keys()) if schedule else -1
    total_cycles = max(config.max_cycles, max_scheduled_cycle + 1 + config.drain_extra_cycles)

    for cyc in range(total_cycles):
        for src, dst in schedule.get(cyc, []):
            pending_injections.append((src, dst, cyc))

        attempts_this_cycle = len(pending_injections)
        for _ in range(attempts_this_cycle):
            src, dst, created_cycle = pending_injections.popleft()
            pkt = system.create_packet(src, dst, created_cycle=created_cycle)
            if pkt is None and config.retry_failed_injections:
                pending_injections.append((src, dst, created_cycle))

        cycle_result = system.step()

        if verbose:
            print(
                f"cycle={cycle_result['cycle']:3d} "
                f"moves={cycle_result['moves']:2d} "
                f"promotions={cycle_result['promotions']:2d} "
                f"blocked={cycle_result['blocked_heads']:2d} "
                f"in_flight={cycle_result['in_flight']:2d}"
            )

        if cyc >= max_scheduled_cycle and not pending_injections and system.network_empty():
            break

    return system


def build_seeded_cyclic_dependency_system(enable_drain: bool) -> System:
    """
    Creates a synthetic, explicit cyclic dependency in the interposer.
    Without DRAIN, the four packets remain deadlocked.
    With DRAIN, the same cycle is broken and the packets eventually complete.
    """
    system = System(
        rows=4,
        columns=4,
        vc_capacity=1,
        num_normal_vcs=1,
        enable_drain=enable_drain,
        drain_interval=3,
        drain_duration=1,
        drain_block_threshold=2,
        drain_stall_threshold=2,
        hard_promotion_threshold=4,
        deadlock_no_progress_threshold=5,
    )
    system.create_default_chiplet_layout()

    routes = [
        (
            "CPU0",
            "CPU1",
            "IR_0_0",
            ["CPU0:BR_0", "IR_0_0", "IR_0_1", "IR_1_1", "IR_1_0", "IR_0_0", "IR_0_1", "IR_0_2", "IR_0_3", "CPU1:BR_0"],
            1,
        ),
        (
            "CPU1",
            "GPU1",
            "IR_0_1",
            ["CPU1:BR_0", "IR_0_1", "IR_1_1", "IR_1_0", "IR_0_0", "IR_0_1", "IR_0_2", "IR_0_3", "IR_1_3", "IR_2_3", "IR_3_3", "GPU1:BR_0"],
            1,
        ),
        (
            "GPU1",
            "GPU0",
            "IR_1_1",
            ["GPU1:BR_0", "IR_1_1", "IR_1_0", "IR_0_0", "IR_0_1", "IR_1_1", "IR_2_1", "IR_3_1", "IR_3_0", "GPU0:BR_0"],
            1,
        ),
        (
            "GPU0",
            "CPU0",
            "IR_1_0",
            ["GPU0:BR_0", "IR_1_0", "IR_0_0", "IR_0_1", "IR_1_1", "IR_1_0", "IR_0_0", "CPU0:BR_0"],
            1,
        ),
    ]

    for src_chiplet, dst_chiplet, current_node, route, route_index in routes:
        seeded = system.seed_packet_on_route(
            src_chiplet=src_chiplet,
            dst_chiplet=dst_chiplet,
            current_node=current_node,
            route=route,
            route_index=route_index,
            use_escape=False,
        )
        if not seeded:
            raise RuntimeError("Failed to seed cyclic dependency packet. The test setup is inconsistent.")

    return system


def run_seeded_cyclic_dependency_demo(enable_drain: bool, max_cycles: int = 40) -> System:
    system = build_seeded_cyclic_dependency_system(enable_drain=enable_drain)
    for _ in range(max_cycles):
        system.step()
        if system.network_empty():
            break
    return system


def compare_drain_modes(config: ExperimentConfig, schedule):
    off_config = ExperimentConfig(**asdict(config))
    on_config = ExperimentConfig(**asdict(config))
    off_config.enable_drain = False
    on_config.enable_drain = True

    off_system = run_scheduled_experiment(off_config, schedule)
    on_system = run_scheduled_experiment(on_config, schedule)

    return {
        "drain_off": off_system.stats(),
        "drain_on": on_system.stats(),
    }


def write_results_csv(results: List[Dict], filepath: str):
    if not results:
        return

    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def pretty_print_stats(title: str, stats: Dict):
    print(title)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def demo_normal_xy_scenario():
    config = ExperimentConfig(
        scenario_name="normal_xy_baseline",
        rows=4,
        columns=4,
        vc_capacity=2,
        num_normal_vcs=1,
        enable_drain=True,
        drain_interval=6,
        drain_duration=2,
        max_cycles=120,
        drain_extra_cycles=20,
    )

    flows = [
        ("CPU0", "GPU1"),
        ("GPU0", "CPU1"),
    ]
    schedule = build_schedule_periodic(
        start_cycle=0,
        end_cycle=30,
        period=2,
        flows=flows,
        packets_per_flow=1,
    )

    return compare_drain_modes(config, schedule)


def demo_seeded_deadlock_scenario():
    no_drain_system = run_seeded_cyclic_dependency_demo(enable_drain=False, max_cycles=40)
    with_drain_system = run_seeded_cyclic_dependency_demo(enable_drain=True, max_cycles=40)

    return {
        "drain_off": no_drain_system.stats(),
        "drain_on": with_drain_system.stats(),
    }


if __name__ == "__main__":
    print("=== Normal XY baseline scenario ===")
    normal_results = demo_normal_xy_scenario()
    pretty_print_stats("DRAIN OFF", normal_results["drain_off"])
    pretty_print_stats("DRAIN ON", normal_results["drain_on"])

    print("=== Seeded cyclic-dependency deadlock scenario ===")
    deadlock_results = demo_seeded_deadlock_scenario()
    pretty_print_stats("DRAIN OFF", deadlock_results["drain_off"])
    pretty_print_stats("DRAIN ON", deadlock_results["drain_on"])

    csv_rows = []
    row = {"scenario": "normal_xy_baseline", "mode": "drain_off"}
    row.update(normal_results["drain_off"])
    csv_rows.append(row)

    row = {"scenario": "normal_xy_baseline", "mode": "drain_on"}
    row.update(normal_results["drain_on"])
    csv_rows.append(row)

    row = {"scenario": "seeded_cyclic_deadlock", "mode": "drain_off"}
    row.update(deadlock_results["drain_off"])
    csv_rows.append(row)

    row = {"scenario": "seeded_cyclic_deadlock", "mode": "drain_on"}
    row.update(deadlock_results["drain_on"])
    csv_rows.append(row)

    # write_results_csv(csv_rows, "outputs/phase1_experiment_results.csv")
    # print("Wrote CSV summary to outputs/phase1_experiment_results.csv")
