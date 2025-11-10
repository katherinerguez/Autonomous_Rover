from typing import Dict, List, Tuple
from copy import deepcopy


terrain_consumption = {
    'plano': {'energy_consumption': 1.0},
    'rocas': {'energy_consumption': 2.0},
    'arena': {'energy_consumption': 1.5},
    'dunas': {'energy_consumption': 2.5},
}

def rc_id(r: int, c: int, width: int) -> int:
    return r * width + c

def id_rc(nid: int, width: int) -> Tuple[int,int]:
    return divmod(nid, width)

class Node:
    def __init__(self, nid: int, x: float = 0.0, y: float = 0.0, terrain='plano'):
        self.id = nid
        self.x = x
        self.y = y
        self.terrain= terrain

class Edge:
    def __init__(self, u: int, v: int, distance: float, terrain: str):
        self.u = u
        self.v = v
        self.distance = distance
        self.terrain = terrain


class Environment:
    def __init__(self, grid, cell_size_m = 1.0):
        self.h = len(grid)
        self.w = len(grid[0]) if self.h>0 else 0
        self.cell_size = cell_size_m
        self.nodes: Dict[int, Node] = {}
        self.adj: Dict[int, List[Tuple[int, Edge]]] = {}
        self.build(grid)
        self.base_id=0
    
    def build(self, grid):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(self.h):
            for c in range(self.w):
                val = grid[r][c]
                if val is None or val == 'X':  
                    continue
                
                terrain = val if isinstance(val, str) else 'plano'
                nid = rc_id(r, c, self.w)
                self.nodes[nid] = Node(nid, r, c, terrain)
                self.adj[nid] = []

        for nid, node in list(self.nodes.items()):
            for dr, dc in moves:
                nr, nc = node.row + dr, node.col + dc
                
                if not (0 <= nr < self.h and 0 <= nc < self.w):
                    continue
                
                nval = grid[nr][nc]
                if nval is None or nval == 'X':
                    continue
                
                neighbor_id = rc_id(nr, nc, self.w)
                
                if dr == 0 or dc == 0: 
                    dist = self.cell_size
                else:  
                    dist = self.cell_size * (2 ** 0.5)
                
                terrain = self.nodes[neighbor_id].terrain
                edge = Edge(nid, neighbor_id, dist, terrain)
                self.adj[nid].append((neighbor_id, edge))
    


class State:
    def __init__(self, pos: int, battery: float, memory: float, terrain: str):
        self.pos = pos
        self.battery = battery
        self.memory = memory
        self.route = []
        self.terrain= terrain
    def clone(self) -> 'State':
        return deepcopy(self)
    def recargar(self, env: Environment, max_battery: float = 25.0) -> bool:
        # recarga solo si está en la base
        if self.pos == env.base_id:
            self.battery = max_battery
            return True
        return False

    def transmitir(self, env: Environment,  poi: List[int], transmit_cost: float = 3.0) -> Tuple[bool, float]:
        node = env.nodes.get(self.pos)
        if node is None:
            return False, 0.0
        if type(node) is int: 
            node_rc=id_rc(node, 5)
            if node_rc in poi:
                self.memory=0.0
        
            return True, transmit_cost
        
    def can_consume(self, cost: float) -> bool:
        return cost <= (self.battery )

    def consume(self, cost: float):
        self.battery -= cost

    def move(self, next_id: int, cost: float) -> bool:
        if not self.can_consume(cost):
            return False
        self.consume(cost)
        self.pos = next_id
        self.route.append(next_id)
        return True