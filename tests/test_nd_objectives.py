"""N-dimensional objectives: 3-objective Pareto search over the architecture space.

Demonstrates item B: real model-derived objectives (reliability, safety_integrity)
plus cost, driven by the generalised N-D hypervolume — not the toy 2-D placeholder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.dse.grounded_eval import grounded_objectives
from src.dse.mo_mcts import MultiObjectiveMCTS, dominates, hypervolume_nd
from src.dse.operators import RedundantizeComponent

_RED = RedundantizeComponent()


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85


def _objectives(state, ctx):
    sysml = _RED.resolve(state["arbitration"], with_fanin=True)
    obj = grounded_objectives(sysml, ctx.channel_reliability)
    channels = {"single": 1, "dual": 2, "triple": 3}[state["arbitration"]]
    obj["cost_efficiency"] = 1.0 - (channels - 1) / 2.0
    return obj


def _make():
    return MultiObjectiveMCTS(
        operators=[_RED],
        objective_fn=_objectives,
        ctx=Ctx(),
        objective_names=["reliability", "safety_integrity", "cost_efficiency"],
        reference=[0.0, 0.0, 0.0],
        random_seed=5,
    )


# ── N-D hypervolume sanity ─────────────────────────────────────────────────

def test_hypervolume_matches_2d_and_inclusion_exclusion():
    assert abs(hypervolume_nd([(0.85, 1.0), (0.9775, 0.667), (0.99663, 0.333)], (0, 0)) - 0.9414) < 1e-2
    assert hypervolume_nd([(1, 1, 1)], (0, 0, 0)) == 1.0
    # two 3-D boxes, inclusion-exclusion: 0.5 + 0.25 - 0.125 = 0.625
    assert abs(hypervolume_nd([(1, 1, 0.5), (0.5, 0.5, 1)], (0, 0, 0)) - 0.625) < 1e-9


# ── 3-objective search ─────────────────────────────────────────────────────

def test_three_objective_front_is_non_dominated():
    front = _make().search(iterations=60)
    vecs = [(o["reliability"], o["safety_integrity"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_three_objective_hypervolume_positive():
    front = _make().search(iterations=60)
    assert front.hypervolume() > 0.0


def test_safety_integrity_is_orthogonal_to_reliability():
    """A redundant model with a broken voting chain keeps redundancy reliability
    but loses safety integrity — the two objectives are independent."""
    real = _RED.resolve("triple", with_fanin=True)
    fake = re.sub(
        r"attribute failedChannels : Integer =[^;]+;",
        "attribute failedChannels : Integer = 0;",
        real,
        flags=re.DOTALL,
    )
    g_real = grounded_objectives(real)
    g_fake = grounded_objectives(fake)
    assert g_real["reliability"] == g_fake["reliability"]          # same redundancy depth
    assert g_fake["safety_integrity"] < g_real["safety_integrity"]  # integrity collapses
