"""Baseline DSE methods for the shadow comparison (paper experiment).

Compared against the new Pareto MO-MCTS on the SAME architecture design space and
objectives, without touching the live scalar pipeline:

  * ``random_search``       — naive baseline: sample feasible architectures.
  * ``weighted_sum_search`` — the OLD approach: collapse the objective vector to a
    single scalar via a (requirement-derived) weight vector and return the single
    best. Demonstrates the structural limitation pain point C names — it hands the
    engineer ONE design and hides the trade-off front.

Both respect operator feasibility (so requirement-driven constraints apply equally).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .mo_mcts import Objectives, Operator, ParetoArchive, State, dominates


def _sample_feasible(operators: Sequence[Operator], ctx: object, rng: random.Random) -> State:
    """Sample a feasible architecture, building it incrementally so cross-operator
    constraints (state-aware feasibility) apply — same coherent space all methods
    search, for a fair comparison."""
    state: State = {}
    for op in operators:
        feasible = [v for v in op.variants if op.feasible(v, ctx, state)]
        if feasible:
            state[op.point_id] = rng.choice(feasible)
    return state


def random_search(
    operators: Sequence[Operator],
    objective_fn: Callable[[State, object], Objectives],
    ctx: object,
    objective_names: Sequence[str],
    reference: Sequence[float],
    n_evals: int,
    random_seed: Optional[int] = None,
) -> ParetoArchive:
    """Sample ``n_evals`` feasible architectures; return their Pareto archive."""
    rng = random.Random(random_seed)
    archive = ParetoArchive(objective_names, reference)
    for _ in range(n_evals):
        s = _sample_feasible(operators, ctx, rng)
        archive.add(s, objective_fn(s, ctx))
    return archive


def weighted_sum_search(
    operators: Sequence[Operator],
    objective_fn: Callable[[State, object], Objectives],
    ctx: object,
    weights: Dict[str, float],
    n_evals: int,
    random_seed: Optional[int] = None,
) -> Tuple[State, Objectives]:
    """The old scalar approach: maximise a weighted sum, return the single best."""
    rng = random.Random(random_seed)
    best: Optional[Tuple[State, Objectives]] = None
    best_score = float("-inf")
    for _ in range(n_evals):
        s = _sample_feasible(operators, ctx, rng)
        o = objective_fn(s, ctx)
        score = sum(weights.get(k, 0.0) * v for k, v in o.items())
        if score > best_score:
            best_score, best = score, (s, o)
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# NSGA-II — the classic multi-objective baseline
# ---------------------------------------------------------------------------


def _fast_nondominated_sort(vecs: List[Tuple[float, ...]]) -> List[List[int]]:
    n = len(vecs)
    dominated: List[List[int]] = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(vecs[p], vecs[q]):
                dominated[p].append(q)
            elif dominates(vecs[q], vecs[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def _crowding_distance(front: List[int], vecs: List[Tuple[float, ...]], m: int) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for obj in range(m):
        order = sorted(front, key=lambda i: vecs[i][obj])
        dist[order[0]] = dist[order[-1]] = float("inf")
        lo, hi = vecs[order[0]][obj], vecs[order[-1]][obj]
        span = (hi - lo) or 1.0
        for k in range(1, len(order) - 1):
            dist[order[k]] += (vecs[order[k + 1]][obj] - vecs[order[k - 1]][obj]) / span
    return dist


def nsga2(
    operators: Sequence[Operator],
    objective_fn: Callable[[State, object], Objectives],
    ctx: object,
    objective_names: Sequence[str],
    reference: Sequence[float],
    pop_size: int = 8,
    generations: int = 6,
    mutation_rate: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> ParetoArchive:
    """NSGA-II over the discrete operator-variant chromosome (per-gene feasibility
    is independent of other genes, so crossover/mutation preserve feasibility).
    Budget ≈ pop_size × (generations + 1) evaluations. Returns the final front."""
    rng = random.Random(random_seed)
    names = list(objective_names)
    m = len(names)
    points = [op.point_id for op in operators]
    op_by_point = {op.point_id: op for op in operators}
    mut = mutation_rate if mutation_rate is not None else 1.0 / max(1, len(points))

    def evaluate(state: State):
        o = objective_fn(state, ctx)
        return state, o, tuple(o[n] for n in names)

    def mutate(state: State) -> State:
        child = dict(state)
        for pid in points:
            if rng.random() < mut:
                feas = [v for v in op_by_point[pid].variants if op_by_point[pid].feasible(v, ctx)]
                if feas:
                    child[pid] = rng.choice(feas)
        return child

    def crossover(a: State, b: State) -> State:
        return {pid: (a if rng.random() < 0.5 else b).get(pid) for pid in points}

    pop = [evaluate(_sample_feasible(operators, ctx, rng)) for _ in range(pop_size)]

    for _ in range(generations):
        vecs = [e[2] for e in pop]
        fronts = _fast_nondominated_sort(vecs)
        rank = {i: r for r, fr in enumerate(fronts) for i in fr}
        crowd: Dict[int, float] = {}
        for fr in fronts:
            crowd.update(_crowding_distance(fr, vecs, m))

        def tournament() -> State:
            i, j = rng.randrange(len(pop)), rng.randrange(len(pop))
            if rank[i] != rank[j]:
                win = i if rank[i] < rank[j] else j
            else:
                win = i if crowd[i] >= crowd[j] else j
            return pop[win][0]

        offspring = []
        while len(offspring) < pop_size:
            child = mutate(crossover(tournament(), tournament()))
            offspring.append(evaluate(child))

        combined = pop + offspring
        cvecs = [e[2] for e in combined]
        cfronts = _fast_nondominated_sort(cvecs)
        new_pop: List = []
        for fr in cfronts:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop += [combined[i] for i in fr]
            else:
                d = _crowding_distance(fr, cvecs, m)
                for i in sorted(fr, key=lambda i: d[i], reverse=True):
                    if len(new_pop) >= pop_size:
                        break
                    new_pop.append(combined[i])
                break
        pop = new_pop

    archive = ParetoArchive(names, reference)
    for state, o, _ in pop:
        archive.add(state, o)
    return archive
