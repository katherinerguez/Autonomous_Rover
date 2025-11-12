from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import random
from environment import Environment, State
from simulate import simulate

class Individual:
    def __init__(self, bitmask: List[int], perm: List[int]):
        self.bitmask = bitmask  # 0/1 per candidate index
        self.perm = perm        # permutation of candidate ids

def decode(ind: Individual, pool: List[int], index_map: Dict[int,int]) -> List[int]:
    return [p for p in ind.perm if ind.bitmask[index_map[p]] == 1]

def init_population(pop_size: int, pool: List[int]) -> List[Individual]:
    pop = []
    for _ in range(pop_size):
        perm = pool[:]
        random.shuffle(perm)
        bitmask = [random.choice([0,1]) for _ in pool]
        # ensure at least one selected sometimes
        if sum(bitmask) == 0:
            bitmask[random.randrange(len(bitmask))] = 1
        pop.append(Individual(bitmask, perm))
    return pop

def tournament_select(pop: List[Individual], fitnesses: List[float], k: int = 3) -> Individual:
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return deepcopy(pop[best])

def uniform_crossover_bitmask(b1: List[int], b2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(b1)
    c1, c2 = b1[:], b2[:]
    for i in range(n):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def order_crossover_perm(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(parent_a, parent_b):
        child = [None]*n
        child[a:b+1] = parent_a[a:b+1]
        pos = (b+1) % n
        for x in parent_b[b+1:] + parent_b[:b+1]:
            if x not in child:
                child[pos] = x
                pos = (pos+1) % n
        return child
    return ox(p1,p2), ox(p2,p1)

def mutate(ind: Individual, pm_bit: float = 0.05, pm_perm: float = 0.1):
    # flip bits
    for i in range(len(ind.bitmask)):
        if random.random() < pm_bit:
            ind.bitmask[i] = 1 - ind.bitmask[i]
    # perm mutation: swap
    if random.random() < pm_perm:
        i,j = random.sample(range(len(ind.perm)), 2)
        ind.perm[i], ind.perm[j] = ind.perm[j], ind.perm[i]

def evaluate_individual(ind: Individual, base_id: int, pool: List[int], index_map: Dict[int,int],
                        dist_matrix: Dict[int, Dict[int, Optional[float]]],
                        path_matrix: Dict[int, Dict[int, Optional[List[Tuple[int,int]]]]],
                        env: Environment, init_state: State,
                        action_cost: float, transmit_cost: Optional[float],
                        safety_margin: float, alpha: float = 100.0, beta: float = 1.0, penalty: float = 1e6) -> float:
    tour = decode(ind, pool, index_map)
    feasible, _, _, _, total = simulate(init_state, base_id, tour, dist_matrix, path_matrix, env, action_cost, transmit_cost, safety_margin)
    if not feasible:
        return -penalty - total
    return alpha * len(tour) - beta * total

def repair_by_battery(ind: Individual, base_id: int, pool: List[int], index_map: Dict[int,int],
                      dist_matrix: Dict[int, Dict[int, Optional[float]]],
                      path_matrix: Dict[int, Dict[int, Optional[List[Tuple[int,int]]]]],
                      env: Environment, init_state: State, action_cost: float, transmit_cost: Optional[float],
                      safety_margin: float):
    # remove POIs until feasible; remove the one with worst marginal improvement
    while True:
        tour = decode(ind, pool, index_map)
        feasible, _, _, _, total = simulate(init_state, base_id, tour, dist_matrix, path_matrix, env, action_cost, transmit_cost, safety_margin)
        if feasible:
            return
        if not tour:
            return
        best_gain = -float('inf')
        worst_p = None
        # compute marginal cost of removing each p
        for p in tour:
            # create candidate with p removed
            new_mask = ind.bitmask[:]
            new_mask[index_map[p]] = 0
            cand = Individual(new_mask, ind.perm)
            f, _, _, _, tot = simulate(init_state, base_id, decode(cand, pool, index_map), dist_matrix, path_matrix, env, action_cost, transmit_cost, safety_margin)
            if f:
                worst_p = p
                break
            gain = total - tot
            if gain > best_gain:
                best_gain = gain
                worst_p = p
        if worst_p is None:
            return
        ind.bitmask[index_map[worst_p]] = 0

def ga_main(base_id: int, pool: List[int], dist_matrix: Dict[int, Dict[int, Optional[float]]],
            path_matrix: Dict[int, Dict[int, Optional[List[Tuple[int,int]]]]],
            env: Environment, init_state: State,
            pop_size: int = 50, gens: int = 200,
            action_cost: float = 0.0, transmit_cost: Optional[float] = None,
            safety_margin: float = 0.0) -> Tuple[Individual, float]:
    index_map = {p:i for i,p in enumerate(pool)}
    pop = init_population(pop_size, pool)
    best_ind = None
    best_fit = -float('inf')
    for g in range(gens):
        fitnesses = [evaluate_individual(ind, base_id, pool, index_map, dist_matrix, path_matrix, env, init_state, action_cost, transmit_cost, safety_margin) for ind in pop]
       
        for i, f in enumerate(fitnesses):
            if f > best_fit:
                best_fit = f
                best_ind = deepcopy(pop[i])
      
        new_pop = []
        elite_count = max(1, pop_size // 20)
        elite_idxs = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)[:elite_count]
        for idx in elite_idxs:
            new_pop.append(deepcopy(pop[idx]))
       
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)
            b1, b2 = uniform_crossover_bitmask(p1.bitmask, p2.bitmask)
            perm1, perm2 = order_crossover_perm(p1.perm, p2.perm)
            c1 = Individual(b1, perm1)
            c2 = Individual(b2, perm2)
            mutate(c1); mutate(c2)
            # repair
            repair_by_battery(c1, base_id, pool, index_map, dist_matrix, path_matrix, env, init_state, action_cost, transmit_cost, safety_margin)
            repair_by_battery(c2, base_id, pool, index_map, dist_matrix, path_matrix, env, init_state, action_cost, transmit_cost, safety_margin)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = new_pop
    return best_ind, best_fit