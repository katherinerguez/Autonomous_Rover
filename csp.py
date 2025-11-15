from typing import Dict, List, Optional
from environment import Environment, id_rc

def estimate_min_tour_cost(env: 'Environment', selected: List[int], dist_matrix: Dict[int, Dict[int, Optional[float]]]) -> float:
    if not selected:
        return 0.0
    cur = id_rc(env.base_id, env.w)
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
    back = dist_matrix.get(cur, {}).get(cur)
    if back is None:
        return float('inf')
    cost += back 
    return cost

def csp_select_pois_by_battery_memory(env: 'Environment', candidates: List[int], dist_matrix: Dict[int, Dict[int, Optional[float]]],
                                      max_battery: float, max_memory: float) -> List[int]:
    order = sorted(candidates, key=lambda p: dist_matrix.get(env.base_id, {}).get(p, float('inf')))
    
    best_solution: List[int] = []
    best_count = 0
    base=id_rc(env.base_id, env.w)
    def backtrack(i: int, selected: List[int]):
        nonlocal best_solution, best_count
        
        est = estimate_min_tour_cost(env, selected, dist_matrix)
        
        if selected and est <= max_battery and len(selected) > best_count:
            best_solution = selected.copy()
            best_count = len(selected)
        if i >= len(order):
            return
        
        p = order[i]
        
        selected.append(p)
        est2 = estimate_min_tour_cost(env, selected, dist_matrix)
        if est2 <= max_battery:
            backtrack(i + 1, selected)
        selected.pop()
        backtrack(i + 1, selected)
    backtrack(0, [])
    if not best_solution:
        for p in order:
            est1 = estimate_min_tour_cost(env, [p], dist_matrix)
            if est1 <= max_battery and 1 <= max_memory:
                return [p]
    
    return best_solution