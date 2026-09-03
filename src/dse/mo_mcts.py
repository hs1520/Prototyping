"""Multi-objective (Pareto-aware) MCTS - bilevel outer-layer prototype.

Replaces the scalar weighted-sum reward of ``mcts.py`` for architecture search:
state = the resolved variation-point choices {point_id: variant}; action =
resolve one operator's variation point to a feasible variant; reward = the
hypervolume gain a rollout contributes to the global Pareto archive (MO-MCTS,
hypervolume-driven; Wang & Sebag 2012 lineage); output = the global Pareto front
rather than one best configuration. Objectives come from a callable returning a
dict of values, all maximised. Operators are duck-typed: ``point_id``,
``variants``, ``feasible(variant, ctx)`` and ``resolve(variant)``. Decoupled from
the live scalar pipeline in ``mcts.py`` so the existing flow is unaffected during
the M1/M2 refactor.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

Objectives = Dict[str, float]
State = Dict[str, str]


class Operator(Protocol):
    point_id: str

    @property
    def variants(self) -> List[str]: ...
    # ``state`` is the partially-resolved architecture so far, enabling
    # cross-operator feasibility (e.g. sensing channels >= redundancy channels).
    def feasible(self, variant: str, ctx, state: Optional["State"] = None) -> bool: ...
    def resolve(self, variant: str) -> str: ...


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if a Pareto-dominates b (>= on all, > on at least one)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _nondominated(points: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    keep: List[Tuple[float, ...]] = []
    for p in points:
        if any(dominates(q, p) for q in points if q is not p):
            continue
        if p not in keep:
            keep.append(p)
    return keep


def hypervolume_nd(points: Sequence[Tuple[float, ...]], reference: Sequence[float]) -> float:
    """N-dimensional hypervolume above ``reference`` (maximisation), via HSO."""
    ref = tuple(reference)
    pts = [tuple(p) for p in points if all(p[i] > ref[i] for i in range(len(ref)))]

    def rec(ps: List[Tuple[float, ...]], r: Tuple[float, ...]) -> float:
        if not ps:
            return 0.0
        d = len(r)
        if d == 1:
            return max(p[0] for p in ps) - r[0]
        ps = sorted(ps, key=lambda p: p[-1], reverse=True)
        vol = 0.0
        for i in range(len(ps)):
            lower = ps[i + 1][-1] if i + 1 < len(ps) else r[-1]
            height = ps[i][-1] - lower
            if height <= 0:
                continue
            proj = _nondominated([p[:-1] for p in ps[: i + 1]])
            vol += rec(proj, r[:-1]) * height
        return vol

    return rec(pts, ref)


class ParetoArchive:
    """Maintains the non-dominated set and its hypervolume w.r.t. a reference."""

    def __init__(self, objective_names: Sequence[str], reference: Sequence[float]):
        self.names = list(objective_names)
        self.reference = tuple(reference)
        if len(self.reference) != len(self.names):
            raise ValueError("reference must have one coordinate per objective")
        self.members: List[Tuple[State, Objectives]] = []

    def _vec(self, obj: Objectives) -> Tuple[float, ...]:
        return tuple(obj[n] for n in self.names)

    def add(self, state: State, obj: Objectives) -> bool:
        """Add a point; drop any it dominates."""
        v = self._vec(obj)
        for _, existing in self.members:
            if dominates(self._vec(existing), v) or self._vec(existing) == v:
                return False
        self.members = [
            (s, o) for s, o in self.members if not dominates(v, self._vec(o))
        ]
        self.members.append((dict(state), dict(obj)))
        return True

    def hypervolume(self) -> float:
        """Hypervolume of the front above the reference (maximisation, any D)."""
        return hypervolume_nd([self._vec(o) for _, o in self.members], self.reference)


@dataclass
class MONode:
    state: State
    parent: Optional["MONode"] = None
    children: List["MONode"] = field(default_factory=list)
    untried: List[Tuple[str, str]] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.avg_reward
        return self.avg_reward + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )


class MultiObjectiveMCTS:
    def __init__(
        self,
        operators: Sequence[Operator],
        objective_fn: Callable[[State, object], Objectives],
        ctx: object,
        objective_names: Sequence[str],
        reference: Sequence[float],
        exploration_constant: float = math.sqrt(2),
        random_seed: Optional[int] = None,
    ):
        self.operators = list(operators)
        self.objective_fn = objective_fn
        self.ctx = ctx
        self.c = exploration_constant
        self.rng = random.Random(random_seed)
        self.archive = ParetoArchive(objective_names, reference)
        self.root = MONode(state={}, untried=self._actions({}))

    def _actions(self, state: State) -> List[Tuple[str, str]]:
        acts: List[Tuple[str, str]] = []
        for op in self.operators:
            if op.point_id in state:
                continue
            for v in op.variants:
                if op.feasible(v, self.ctx, state):
                    acts.append((op.point_id, v))
        return acts

    def _is_terminal(self, state: State) -> bool:
        return all(op.point_id in state for op in self.operators)

    def search(self, iterations: int = 100) -> ParetoArchive:
        for _ in range(iterations):
            node = self._select(self.root)
            if node.untried:
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        return self.archive

    def _select(self, node: MONode) -> MONode:
        while not node.untried and node.children:
            node = max(node.children, key=lambda n: n.ucb(self.c))
        return node

    def _expand(self, node: MONode) -> MONode:
        point_id, variant = node.untried.pop(
            self.rng.randrange(len(node.untried))
        )
        child_state = dict(node.state)
        child_state[point_id] = variant
        child = MONode(
            state=child_state,
            parent=node,
            untried=self._actions(child_state),
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MONode) -> float:
        state = dict(node.state)
        remaining = self._actions(state)
        while not self._is_terminal(state) and remaining:
            point_id, variant = self.rng.choice(remaining)
            state[point_id] = variant
            remaining = self._actions(state)

        # cross-operator feasibility can leave a branch with no completion; an incomplete
        # architecture is not scored (objective_fn expects all points set).
        if not self._is_terminal(state):
            return 0.0

        obj = self.objective_fn(state, self.ctx)
        before = self.archive.hypervolume()
        self.archive.add(state, obj)
        after = self.archive.hypervolume()
        return after - before

    def _backpropagate(self, node: Optional[MONode], reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


def forest_search(
    operators: Sequence[Operator],
    objective_fn: Callable[[State, object], Objectives],
    ctx: object,
    objective_names: Sequence[str],
    reference: Sequence[float],
    n_trees: int = 4,
    iterations: int = 30,
    seeds: Optional[Sequence[int]] = None,
) -> ParetoArchive:
    """Multi-seed forest search (Item E): run ``n_trees`` independent MO-MCTS trees and
    merge their fronts into one Pareto archive.

    Each tree explores a different trajectory, so the union covers more of the front
    than a single tree of the same budget. Total budget = n_trees x iterations; also
    an ablation axis (forest vs single-tree). With LLM-generated variation skeletons
    (optional step 3) the per-tree seeds would be distinct seed architectures.
    """
    seeds = list(seeds) if seeds is not None else list(range(n_trees))
    merged = ParetoArchive(objective_names, reference)
    for s in seeds:
        tree = MultiObjectiveMCTS(
            operators, objective_fn, ctx, objective_names, reference, random_seed=s
        )
        front = tree.search(iterations=iterations)
        for state, obj in front.members:
            merged.add(state, obj)
    return merged
