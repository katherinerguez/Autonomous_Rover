import heapq
import math
from typing import Dict, List, Tuple, Optional
from environment import Environment, Edge, terrain_consumption, rc_id, id_rc

def cost(edge: Edge, consumption_map=terrain_consumption):
    consumption_terrain = consumption_map.get(edge.terrain, consumption_map['plano'])
    energy = consumption_terrain['energy_consumption'] * edge.distance
    return energy

def heuristic(env: Environment, a: int, b: int) -> float:
    na, nb = env.nodes[a], env.nodes[b]
    return math.hypot(na.x - nb.x, na.y - nb.y)

def dijkstra(env: Environment, start_rc: Tuple[int,int], goal_rc: Tuple[int,int],  use_heuristic: bool = False):
    start = rc_id(start_rc[0], start_rc[1], env.w)
    goal = rc_id(goal_rc[0], goal_rc[1], env.w)
    if start not in env.nodes or goal not in env.nodes:
        return None, float('inf')

    dist = {nid: float('inf') for nid in env.nodes}
    prev = {}
    dist[start] = 0.0
    if use_heuristic:
        f_score = {nid: float('inf') for nid in env.nodes}
        f_score[start] = heuristic(env, start, goal)
        pq = [(f_score[start], start)]
    else:
        pq = [(0.0, start)]

    while pq:
        if use_heuristic:
            current_f, u = heapq.heappop(pq)
            if current_f > f_score[u]:
                continue
        else:
            cur_cost, u = heapq.heappop(pq)
            if cur_cost > dist[u]:
                continue
        if u == goal:
            break
        for v, edge in env.adj.get(u, []):
            energy = cost(edge)
            alt = cur_cost + energy
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                if use_heuristic:
                    f_score[v] = alt + heuristic(env, v, goal)
                    heapq.heappush(pq, (f_score[v], v))
                else:
                    heapq.heappush(pq, (alt, v))
    
    if dist[goal] == float('inf'):
        return None, float('inf')

    path_ids = []
    node = goal
    while node != start:
        path_ids.append(node)
        node = prev[node]
    path_ids.append(start)
    path_ids.reverse()
    path_rc = [id_rc(nid, env.w) for nid in path_ids]
    return path_rc, dist[goal]
