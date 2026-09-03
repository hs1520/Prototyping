"""End-to-end bilevel test: outer MCTS (discrete) + inner BO (continuous).

Outer chooses architecture (redundancy x topology); the inner Bayesian
optimizer tunes control_frequency_hz to each architecture's optimum. Backs
contribution #2: each layer uses the search matched to its geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.dse.bilevel import BilevelEvaluator
from src.dse.inner_bo import BayesianOptimizer
from src.dse.mo_mcts import MultiObjectiveMCTS, dominates
from src.dse.operators import CATALOG, DecomposeController, RedundantizeComponent


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85
    part_count: int = 5
    allow_distributed: bool = True
    controller_reliability: float = 0.9


# control_frequency optimum depends on topology: distributed takes a faster loop
def _f_opt(state) -> float:
    return 80.0 if state["topology"] == "distributed" else 40.0


def _inner_objective(state, f, ctx):
    return math.exp(-((f - _f_opt(state)) / 20.0) ** 2)


def _outer_map(state, best_f, best_perf, ctx):
    channels = CATALOG[state["arbitration"]][1]
    nodes = 1 if state["topology"] == "centralised" else 3
    block_r = 1.0 - (1.0 - ctx.channel_reliability) ** channels
    ctrl_r = (
        ctx.controller_reliability
        if state["topology"] == "centralised"
        else 1.0 - (1.0 - ctx.controller_reliability) ** nodes
    )
    units = channels + nodes
    return {
        "reliability": block_r * ctrl_r * best_perf,
        "cost_efficiency": 1.0 - (units - 2) / 4.0,
    }


def _make():
    evaluator = BilevelEvaluator(
        inner_bounds=(10.0, 120.0),
        inner_objective=_inner_objective,
        outer_map=_outer_map,
        random_seed=0,
    )
    mcts = MultiObjectiveMCTS(
        operators=[RedundantizeComponent(), DecomposeController()],
        objective_fn=evaluator,
        ctx=Ctx(),
        objective_names=["reliability", "cost_efficiency"],
        reference=[0.0, 0.0],
        random_seed=7,
    )
    return mcts, evaluator


def test_inner_bo_finds_optimum():
    mcts, ev = _make()
    mcts.search(iterations=120)
    for state in ev.cached_states():
        f = ev.best_param(state)
        target = _f_opt(state)
        assert abs(f - target) < 8.0, f"{state}: f*={f} expected ~{target}"


def test_front_non_dominated():
    mcts, _ = _make()
    front = mcts.search(iterations=120)
    vecs = [(o["reliability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_inner_runs_cached():
    mcts, ev = _make()
    mcts.search(iterations=120)
    assert ev.inner_runs <= 6
    assert ev.inner_runs < 120


def test_bo_beats_random_equal_budget():
    def perf(f):
        return math.exp(-((f - 80.0) / 20.0) ** 2)

    bo = BayesianOptimizer((10, 120), perf, n_init=3, n_iter=12, random_seed=0)
    bo_best = bo.optimize().best_y
    import random as _r

    rng = _r.Random(0)
    rand_best = max(perf(rng.uniform(10, 120)) for _ in range(15))
    assert bo_best >= rand_best
