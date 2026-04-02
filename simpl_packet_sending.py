from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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


@dataclass
class Link:
    src: str
    dst: str
    latency: int = 1


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
    ):
        self.mesh = InterposerMesh(rows, columns, vc_capacity, num_normal_vcs)
        self.chiplets: Dict[str, Chiplet] = {}
        self.boundary_routers: Dict[str, BoundaryRouter] = {}
        self.packet_counter = 0
        self.cycle = 0
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        self.completed_packets: List[Packet] = []
        self.injected_packet_count = 0
        self.enable_drain = enable_drain
        self.drain_interval = drain_interval
        self.drain_duration = drain_duration

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
        return None

    def _all_vcs(self):
        for br in self.boundary_routers.values():
            for vc in br.normal_vcs:
                yield br.router_id, False, vc
            yield br.router_id, True, br.escape_vc

        for ir in self.mesh.routers.values():
            for vc in ir.normal_vcs:
                yield ir.router_id, False, vc
            yield ir.router_id, True, ir.escape_vc

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
            return dst_router.escape_vc

        for vc in dst_router.normal_vcs:
            if vc.can_push():
                return vc
        return None

    def _promote_to_escape(self):
        if not self.is_drain_cycle():
            return

        for br in self.boundary_routers.values():
            if br.escape_vc.can_push():
                for vc in br.normal_vcs:
                    pkt = vc.peek()
                    if pkt is not None:
                        vc.pop()
                        pkt.in_escape_vc = True
                        pkt.ever_entered_escape = True
                        br.escape_vc.push(pkt)
                        break

        for ir in self.mesh.routers.values():
            if ir.escape_vc.can_push():
                for vc in ir.normal_vcs:
                    pkt = vc.peek()
                    if pkt is not None:
                        vc.pop()
                        pkt.in_escape_vc = True
                        pkt.ever_entered_escape = True
                        ir.escape_vc.push(pkt)
                        break

    def step(self):
        self._promote_to_escape()
        proposals = []

        for node_id, is_escape, vc in list(self._all_vcs()):
            pkt = vc.peek()
            if pkt is None:
                continue
            next_node = self._packet_next_node(pkt)
            if next_node is None:
                continue
            proposals.append((node_id, is_escape, vc, pkt, next_node))

        reserved = set()
        executed = []
        for node_id, is_escape, vc, pkt, next_node in proposals:
            if next_node in self.boundary_routers:
                key = ("eject", next_node)
                if key in reserved:
                    continue
                reserved.add(key)
                executed.append((node_id, is_escape, vc, pkt, next_node, None))
            else:
                target_vc = self._get_target_vc(pkt, next_node)
                if target_vc is None:
                    continue
                key = ("vc", id(target_vc))
                if key in reserved:
                    continue
                reserved.add(key)
                executed.append((node_id, is_escape, vc, pkt, next_node, target_vc))

        for node_id, is_escape, src_vc, pkt, next_node, target_vc in executed:
            src_vc.pop()
            pkt.current_node = next_node
            pkt.hops += 1

            if not pkt.in_escape_vc:
                pkt.route_index += 1

            if next_node in self.boundary_routers:
                self.boundary_routers[next_node].receive_packet(pkt, self.cycle + 1)
                self.completed_packets.append(pkt)
            else:
                target_vc.push(pkt)

        self.cycle += 1

    def run(self, cycles: int):
        for _ in range(cycles):
            self.step()

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
            "total_injected_packets": total_injected,
            "delivered_packets": delivered,
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "avg_hops": avg_hops,
            "throughput": throughput,
            "drain_mode": 
                self.is_drain_cycle(),
            "escape_packet_fraction": escape_packet_fraction,
        }


if __name__ == "__main__":
    rows = 4
    columns = 4
    vc_capacity = 8
    num_normal_vcs = 2
    drain_interval = 5
    drain_duration = 2

    enable_drain = False

    # This is the full system setup and specs of scenario 
    system = System(
        rows=rows,
        columns=columns,
        vc_capacity=vc_capacity,
        num_normal_vcs=num_normal_vcs,
        enable_drain=enable_drain,
        drain_interval=drain_interval,
        drain_duration=drain_duration,
    )
    
    # create a nxm mesh interposer with y-flit VCs and z normal VCs per router,
    system.create_default_chiplet_layout()

    # inject X packets from CPU0 to GPU1 and X from GPU0 to CPU1 to create some initial traffic in the system.
    for _ in range(24):
        system.create_packet("CPU0", "GPU1")
        system.create_packet("GPU0", "CPU1")

    # sending packets from CPU0 to GPU1 and from GPU0 to CPU1, 
    # then run the system for 25 cycles and print stats and completed packets
    system.run(25)

    
    stats = system.stats()
    print(f"Run metrics after {stats['cycle']} cycles:")
    print(f"  DRAIN enabled: {stats['drain_enabled']}")
    print(f"  Total injected packets: {stats['total_injected_packets']}")
    print(f"  Delivered packets: {stats['delivered_packets']}")
    print(f"  Average latency: {stats['avg_latency']}")
    print(f"  Max latency: {stats['max_latency']}")
    print(f"  Average hops: {stats['avg_hops']}")
    print(f"  Throughput: {stats['throughput']}")
    print(f"  Escape-mode fraction: {stats['escape_packet_fraction']}")
    for packet in system.completed_packets:
        print(
            f"Packet {packet.packet_id}: {packet.src_chiplet} -> {packet.dst_chiplet}, "
            f"hops={packet.hops}, latency={packet.delivered_cycle - packet.created_cycle}, "
            f"escape={packet.ever_entered_escape}"
        )
