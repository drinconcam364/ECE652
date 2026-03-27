from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random


MESHDIM = [4,4]
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
    current_node: str = ''
    hops: int = 0
    in_escape_vc: bool = False
    
@dataclass
class Link:
    src: str
    dst: str
    latency: int = 1 # set latency as constant 1 for now

class Chiplet:
    def __init__(self, name, chiplet_type, num_boundary_routers, vc_capcity, num_normal_vcs, mesh_coord):
        self.name = name
        self.chiplet_type = chiplet_type
        self.boundary_routers: Dict[str, BoundaryRouter] = {}
        self.create_boundary_routers(num_boundary_routers, vc_capcity, num_normal_vcs)
        self.mesh_coord = mesh_coord
    
    def create_boundary_routers(self, num_boundary_routers, vc_capacity, num_normal_vcs):
        for i in range(num_boundary_routers):
            boundary_router_id = f"BR_{i}"
            boundary_router = BoundaryRouter(boundary_router_id, self.chiplet_name, vc_capacity, num_normal_vcs)
            self.boundary_routers[boundary_router.router_id] = boundary_router
        
    def pick_boundary_router(self):
        return random.choice(list(self.boundary_routers.values()))
    
    def create_packet(self, packet_id, dst_chiplet, dst_boundary_router):
        src_boundary_router = self.pick_boundary_router()
        packet = Packet ( packet_id = packet_id,
                        src_chiplet = self.name,
                        dst_chiplet = dst_chiplet,
                        dst_boundary_router = dst_boundary_router,
                        created_cycle = 0,
                        current_node = src_boundary_router.router_id
                    )
        successfully_injected = src_boundary_router.inject_packet(packet)
        if successfully_injected:
            return packet
        return None
        
class CPUChiplet(Chiplet):
    def __init__(self, name, num_boundary_routers):
        super().__init__(name, "CPU", num_boundary_routers)

class GPUChiplet(Chiplet):
    def __init__(self, name, num_boundary_routers):
        super().__init__(name, "GPU", num_boundary_routers)
        
class BoundaryRouter:
    def __init__(self, router_id, chiplet_name, vc_capacity, num_normal_vcs):
        self.router_id = router_id
        self.chiplet_name = chiplet_name
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        self.normal_vcs = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc = VC(vc_capacity)
        self.injected_packets: List[Packet] = []
        self.received_packets: List[Packet] = []
    
    def inject_packet(self, packet): # May need boolean return
        for vc in self.normal_vcs:
            if not vc.is_full():
                vc.push(packet)
                packet.current_node = self.router_id
                self.injected_packets.append(packet)
                return True
        return False
    
    def receive_packet(self, packet):
        packet.current_node = self.router_id
        self.received_packets.append(packet)
        
class InterposerRouter:
    def __init__(self, router_id, coordinate, vc_capacity, num_normal_vcs):
        self.router_id = router_id
        self.coordinate = coordinate
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        self.normal_vcs = [VC(vc_capacity) for _ in range(num_normal_vcs)]
        self.escape_vc = VC(vc_capacity)
        self.attatched_boundary_routers = List[str]
        self.ir_neighbors = {}
    
    def add_ir_neighbor(self,direction, ir_neighbor_id):
        self.ir_neighbors[direction] = ir_neighbor_id
    
    def add_boundary_router(self, boundary_router_id):
        self.attatched_boundary_routers.append(boundary_router_id)
        
class InterposerMesh:
    def __init__(self, rows, columns, vc_capacity, num_normal_vcs):
        self.rows = rows
        self.columns = columns
        self.routers = {}
        self.links: List[Link]= []
        self.create_interposer_routers(vc_capacity, num_normal_vcs)
        self.create_mesh()
        
    def make_router_id(self, row, column):
        return f"IR_{row}_{column}"
        
    def create_interposer_routers(self, vc_capacity, num_normal_vcs):
        for row in range(self.rows):
            for column in range(self.columns):
                coordinate = [row, column]
                router_id = self.make_router_id(row, column)
                self.routers[router_id] = InterposerRouter(
                    router_id,
                    coordinate,
                    vc_capacity,
                    num_normal_vcs
                )
    
    def create_mesh(self):
        for row in range(self.rows):
            for column in range(self.columns):
                router_id = self.make_router_id(row, column)
                router = self.routers[router_id]
                if row > 0:
                    north = self.make_router_id(row - 1, column)
                    router.add_ir_neighbor('N', north)
                    self.links.append(Link(router_id, north))
                else: router.add_ir_neighbor('N', None)
                if row < self.rows - 1:
                    south = self.make_router_id(row + 1, column)
                    router.add_ir_neighbor('S', south)
                    self.links.append(Link(router_id, south))
                else: router.add_ir_neighbor("S", None)   
                if column > 0:
                    west = self.make_router_id(row, column -1)
                    router.add_ir_neighbor('W', west)
                    self.links.append(Link(router_id, west))
                else: router.add_ir_neighbor("W", None)      
                if column < self.columns -1:
                    east = self.make_router_id(row, column + 1)
                    router.add_ir_neighbor('E', east)       
                    self.links.append(Link(router_id, east))
                else: router.add_ir_neighbor('E', None)
    
    def get_interposer_router(self, row, column):
        router_id = self.make_router_id(row, column)
        return self.routers(router_id)
    
    def find_path(self, src, dst):
        src_router = self.routers[src]
        dst_router = self.routers[dst]
        src_row, src_column = src_router.coord
        dst_row, dst_column = dst_router.coord
        current_row, current_column = src_router.coord
        path = [src]
        while current_column != dst_column:
            if current_column > dst_column: current_column -= 1
            else: current_column += 1
            path.append(self.make_router_id(current_row, current_column))
        while current_row != dst_row:
            if current_row > dst_row: current_row -= 1
            else: current_row += 1
            path.append(self.make_router_id(current_row, current_column))
        return path
                
class VC:
    def __init__(self, capacity):
        self.capacity = capacity
        self.fifo_queue = []   
    def can_push(self):
        if len(self.fifo_queue) < self.capacity: return True
        return False 
    def push(self, packet): self.queue.append(packet)      
    def can_pop(self):
        if len(self.fifo_queue) > 0: return True
        return False  
    def pop(self): return self.fifo_queue.pop(0)
    def occupancy(self): return len(self.fifo_queue)
    def is_full(self): return len(self.fifo_queue) == self.capacity
    def is_empty(self): return len(self.fifo_queue) == 0
    

class System:
    def __init__(self, rows, columns, vc_capacity, num_normal_vcs):
        self.mesh = InterposerMesh(rows, columns, vc_capacity, num_normal_vcs)
        self.chiplets: Dict[str, Chiplet] = {}
        self.boundary_routers: Dict[str, BoundaryRouter] = {}
        self.packet_counter = 0
        self.vc_capacity = vc_capacity
        self.num_normal_vcs = num_normal_vcs
        
    def add_chiplet(self, chiplet_name, chiplet_type, num_boundary_routers, vc_capacity, num_normal_vcs, chiplet_coord):
        chiplet = None
        if chiplet_type == "CPU": chiplet = CPUChiplet(chiplet_name, num_boundary_routers, vc_capacity, num_normal_vcs, chiplet_coord)
        if chiplet_type == "GPU": chiplet = GPUChiplet(chiplet_name, num_boundary_routers, vc_capacity, num_normal_vcs, chiplet_coord)
        self.chiplets[chiplet_name] = chiplet
        
    def create_default_chiplet_layout(self):
        cpu0 = self.add_chiplet("CPU0", "CPU", 1, self.vc_capacity, self.num_normal_vcs, [0,0])
        cpu1 = self.add_chiplet("CPU1", "CPU", 1, self.vc_capacity, self.num_normal_vcs, [0,1])
        cpu2 = self.add_chiplet("CPU2", "CPU", 1, self.vc_capacity, self.num_normal_vcs, [1,0])
        cpu3 = self.add_chiplet("CPU3", "CPU", 1, self.vc_capacity, self.num_normal_vcs, [1,1])
    
    def get_next_packet_id(self):
        packet_id = self.packet_counter
        self.packet_counter += 1
        return packet_id
    
        
        
        