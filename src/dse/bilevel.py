"""Bilevel coupling: outer MO-MCTS evaluation delegates to inner Bayesian opt.

``BilevelEvaluator`` scores a discrete architecture by the best continuous
parameterisation the inner BO finds for it. It plugs into
``MultiObjectiveMCTS(objective_fn=...)`` and caches inner results per architecture,
so repeated rollouts of the same structure do not re-run BO (cross-architecture
warm-start is M3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from .inner_bo import BayesianOptimizer, BOResult
from .mo_mcts import Objectives, State

InnerObjective = Callable[[State, float, object], float]
OuterMap = Callable[[State, float, float, object], Objectives]


@dataclass
class BilevelEvaluator:
    inner_bounds: Tuple[float, float]
    inner_objective: InnerObjective
    outer_map: OuterMap
    n_init: int = 3
    n_iter: int = 8
    random_seed: Optional[int] = 0
    _cache: Dict[Tuple, Tuple[float, float]] = field(default_factory=dict)
    inner_runs: int = 0

    def _key(self, state: State) -> Tuple:
        return tuple(sorted(state.items()))

    def __call__(self, state: State, ctx: object) -> Objectives:
        key = self._key(state)
        if key not in self._cache:
            self.inner_runs += 1
            bo = BayesianOptimizer(
                bounds=self.inner_bounds,
                objective=lambda x: self.inner_objective(state, x, ctx),
                n_init=self.n_init,
                n_iter=self.n_iter,
                random_seed=self.random_seed,
            )
            res: BOResult = bo.optimize()
            self._cache[key] = (res.best_x, res.best_y)
        best_x, best_y = self._cache[key]
        return self.outer_map(state, best_x, best_y, ctx)

    def best_param(self, state: State) -> Optional[float]:
        """The inner-optimal continuous value found for a resolved architecture."""
        hit = self._cache.get(self._key(state))
        return hit[0] if hit else None

    def cached_states(self):
        """Yield every resolved architecture the inner layer has evaluated."""
        for key in self._cache:
            yield dict(key)
