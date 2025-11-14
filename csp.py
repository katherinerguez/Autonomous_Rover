from typing import Dict, List, Optional
from environment import Environment

def estimate_min_tour_cost(base_id: int, selected: List[int], dist_matrix: Dict[int, Dict[int, Optional[float]]], action_cost: float) -> float:
    if not selected:
        return 0.0
    cur = base_id
    cost = 0.0
    unvis = set(selected)
    while unvis:
        nxt = min(unvis, key=lambda p: dist_matrix.get(cur, {}).get(p, float('inf')))
        d = dist_matrix.get(cur, {}).get(nxt)
        if d is None:
            return float('inf')
        cost += d
        cur = nxt
        unvis.remove(nxt)
    back = dist_matrix.get(cur, {}).get(base_id)
    if back is None:
        return float('inf')
    cost += back + len(selected) * action_cost
    return cost

def csp_select_pois_by_battery_memory(env: 'Environment', candidates: List[int], dist_matrix: Dict[int, Dict[int, Optional[float]]],
                                      max_battery: float, max_memory: float) -> List[int]:
    order = sorted(candidates, key=lambda p: dist_matrix.get(env.base_id, {}).get(p, float('inf')))
    
    best_solution: List[int] = []
    best_count = 0

    def backtrack(i: int, selected: List[int]):
        nonlocal best_solution, best_count
        
        est = estimate_min_tour_cost(env.base_id, selected, dist_matrix)
        if est <= max_battery and len(selected) > best_count:
            best_solution = selected.copy()
            best_count = len(selected)
        if i >= len(order):
            return
        
        p = order[i]
        # quick reachability check
        if dist_matrix.get(env.base_id, {}).get(p) is not None:
            selected.append(p)
            # forward check: if still possible to be feasible
            est2 = estimate_min_tour_cost(env.base_id, selected, dist_matrix)
            if est2 <= max_battery:
                backtrack(i+1, selected)
            selected.pop()
        backtrack(i+1, selected)

    backtrack(0, [])
    return best_solution