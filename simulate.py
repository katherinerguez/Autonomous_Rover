from typing import Dict, List, Tuple, Optional
from environment import State, Environment

def simulate(state: 'State', base_id: int, tour: List[int],
             dist_matrix: Dict[int, Dict[int, Optional[float]]],
             path_matrix: Dict[int, Dict[int, Optional[List[Tuple[int,int]]]]],
             env: Environment,
             action_cost: float = 0.0,
             transmit_cost: Optional[float] = None,
             safety_margin: float = 0.0) -> Tuple[bool, State, List[Dict], float, float]:
    s = state.clone()
    profile: List[Dict] = []
    total_cost = 0.0

    for p in tour:
        d = dist_matrix.get(s.pos, {}).get(p)
        if d is None:
            return False, s, profile, s.battery, float('inf')
        if not s.move_to(p, d, safety_margin):
            return False, s, profile, s.battery, float('inf')
        total_cost += d
        profile.append({'from': None, 'to': p, 'cost': d, 'path': path_matrix.get(s.pos, {}).get(p)})

        if action_cost:
            if not s.perform_action(action_cost, memory_cost=0.0, safety=safety_margin):
                return False, s, profile, s.battery, float('inf')
            total_cost += action_cost
            profile.append({'action': 'sample', 'node': p, 'cost': action_cost})

        if transmit_cost is not None:
            ok_tx, tx_cost = s.transmitir(env, transmit_cost, require_poi=True, safety_margin=safety_margin)
            if not ok_tx:
                return False, s, profile, s.battery, float('inf')
            total_cost += tx_cost
            profile.append({'action': 'transmit', 'node': p, 'cost': tx_cost})

    dback = dist_matrix.get(s.pos, {}).get(base_id)
    if dback is None:
        return False, s, profile, s.battery, float('inf')
    if not s.move(base_id, dback, safety_margin):
        return False, s, profile, s.battery, float('inf')
    total_cost += dback
    profile.append({'from': None, 'to': base_id, 'cost': dback, 'path': path_matrix.get(s.pos, {}).get(base_id)})

    return True, s, profile, s.battery, total_cost