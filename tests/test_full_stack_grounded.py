"""Capstone: outer MO-MCTS + inner BO + grounded (executed-model) safety eval.

The outer layer searches architectures, the inner BO tunes control_frequency per
architecture, and reliability comes from executing each resolved model under
fault injection (grounded_eval).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.dse.bilevel import BilevelEvaluator
from src.dse.grounded_eval import grounded_safety
from src.dse.mo_mcts import MultiObjectiveMCTS, dominates
from src.dse.operators import DecomposeController, RedundantizeComponent

_RED = RedundantizeComponent()


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85
    part_count: int = 5
    allow_distributed: bool = True


def _f_opt(state) -> float:
    return 80.0 if state["topology"] == "distributed" else 40.0


def _inner_objective(state, f, ctx):
    return math.exp(-((f - _f_opt(state)) / 20.0) ** 2)


def _outer_map(state, best_f, best_perf, ctx):
    g = grounded_safety(_RED.resolve(state["arbitration"], with_fanin=True),
                        channel_reliability=ctx.channel_reliability)
    nodes = 1 if state["topology"] == "centralised" else 3
    units = g.channels + nodes
    return {
        "reliability": g.reliability * best_perf,
        "cost_efficiency": 1.0 - (units - 2) / 4.0,
    }


def _make():
    ev = BilevelEvaluator(
        inner_bounds=(10.0, 120.0),
        inner_objective=_inner_objective,
        outer_map=_outer_map,
        random_seed=0,
    )
    mcts = MultiObjectiveMCTS(
        operators=[RedundantizeComponent(), DecomposeController()],
        objective_fn=ev,
        ctx=Ctx(),
        objective_names=["reliability", "cost_efficiency"],
        reference=[0.0, 0.0],
        random_seed=7,
    )
    return mcts, ev


def test_front_non_dominated():
    mcts, _ = _make()
    front = mcts.search(iterations=120)
    vecs = [(o["reliability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_reliability_grounded():
    mcts, _ = _make()
    front = mcts.search(iterations=120)
    by_red = {}
    for s, o in front.members:
        by_red.setdefault(s["arbitration"], o["reliability"])
    assert grounded_safety(_RED.resolve("triple", with_fanin=True)).reliability > grounded_safety(_RED.resolve("single")).reliability


def test_inner_bo_cached_and_bounded():
    mcts, ev = _make()
    mcts.search(iterations=120)
    assert ev.inner_runs <= 6
