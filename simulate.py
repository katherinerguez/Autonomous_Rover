from typing import Dict, List, Tuple, Optional
from environment import State, Environment

def simulate(state: 'State', tour: List[int],
             dist_matrix: Dict[int, Dict[int, Optional[float]]],
             path_matrix: Dict[int, Dict[int, Optional[List[Tuple[int,int]]]]],
             env: Environment,
             transmit_cost: Optional[float] = None,
             ) -> Tuple[bool, State, List[Dict], float, float]:
    s = state.clone()
    profile: List[Dict] = []
    total_cost = 0.0

    for p in tour:
        d = dist_matrix.get(s.pos, {}).get(p)
        if d is None:
            return False, s, profile, s.battery, float('inf')
        if not s.move(p, d):
            return False, s, profile, s.battery, float('inf')
        total_cost += d
        profile.append({'from': None, 'to': p, 'cost': d, 'path': path_matrix.get(s.pos, {}).get(p)})

        if transmit_cost is not None:
            ok_tx, tx_cost = s.transmitir(env, poi, transmit_cost)
            if not ok_tx:
                return False, s, profile, s.battery, float('inf')
            total_cost += tx_cost
            profile.append({'accion': 'transmitir', 'node': p, 'cost': tx_cost})

    dback = dist_matrix.get(s.pos, {}).get(env.base_id)
    if dback is None:
        return False, s, profile, s.battery, float('inf')
    if not s.move(env.base_id, dback):
        return False, s, profile, s.battery, float('inf')
    total_cost += dback
    profile.append({'from': None, 'to': env.base_id, 'cost': dback, 'path': path_matrix.get(s.pos, {}).get(env.base_id)})

    return True, s, profile, s.battery, total_cost