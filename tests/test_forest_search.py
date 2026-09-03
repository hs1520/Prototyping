"""Tests for multi-seed forest search (Item E) plus the ablation finding.

Forest correctness is asserted, superiority is not: on this small enumerable
architecture space a single tree matches or beats a forest at equal budget.
Ablation result recorded in docs/DSE_REDESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.dse.mo_mcts import MultiObjectiveMCTS, dominates, forest_search
from src.dse.operators import (
    AddRedundantSensor,
    DecomposeController,
    ReplaceInterfaceProtocol,
    RedundantizeComponent,
)
from src.dse.operators.protocol import CATALOG as PROTO
from src.dse.operators.redundantize import kofn_reliability
from src.dse.operators.sensing import CATALOG as SENSE

_OPS = [RedundantizeComponent(), DecomposeController(), AddRedundantSensor(), ReplaceInterfaceProtocol()]
_NAMES = ["capability", "cost_efficiency"]
_REF = [0.0, 0.0]


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85
    part_count: int = 5
    allow_distributed: bool = True
    max_sensors: int = 3
    allowed_protocols = None


def _obj(s, c):
    masked = {"single": 0, "dual": 1, "triple": 1}[s["arbitration"]]
    rel = kofn_reliability(SENSE[s["sensing"]], masked, c.channel_reliability)
    units = (
        {"single": 1, "dual": 2, "triple": 3}[s["arbitration"]]
        + SENSE[s["sensing"]]
        + (1 if s["topology"] == "centralised" else 3)
    )
    return {"capability": 0.6 * rel + 0.4 * PROTO[s["protocol"]][2],
            "cost_efficiency": 1.0 - (units - 3) / 9.0}


def test_forest_front_nondominated():
    front = forest_search(_OPS, _obj, Ctx(), _NAMES, _REF, n_trees=4, iterations=10)
    assert front.hypervolume() > 0.0
    vecs = [(o["capability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_forest_beats_one_tree():
    iters = 10
    forest = forest_search(_OPS, _obj, Ctx(), _NAMES, _REF, n_trees=3, iterations=iters,
                           seeds=[0, 1, 2])
    one_tree = MultiObjectiveMCTS(_OPS, _obj, Ctx(), _NAMES, _REF, random_seed=0).search(iters)
    assert forest.hypervolume() >= one_tree.hypervolume() - 1e-9


def test_forest_seeds_deterministic():
    a = forest_search(_OPS, _obj, Ctx(), _NAMES, _REF, n_trees=3, iterations=8, seeds=[1, 2, 3])
    b = forest_search(_OPS, _obj, Ctx(), _NAMES, _REF, n_trees=3, iterations=8, seeds=[1, 2, 3])
    assert abs(a.hypervolume() - b.hypervolume()) < 1e-12
