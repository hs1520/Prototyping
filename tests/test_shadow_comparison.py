from __future__ import annotations

from dataclasses import dataclass


from src.dse.baselines import nsga2, random_search, weighted_sum_search
from src.dse.mo_mcts import MultiObjectiveMCTS, dominates
from src.dse.operators import (
    AddRedundantSensor,
    DecomposeController,
    ReplaceInterfaceProtocol,
    RedundantizeComponent,
)
from src.dse.operators.protocol import CATALOG as PROTO
from src.dse.operators.redundantize import kofn_reliability
from src.dse.operators.sensing import CATALOG as SENSE

_OPS = [
    RedundantizeComponent(),
    DecomposeController(),
    AddRedundantSensor(),
    ReplaceInterfaceProtocol(),
]
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


def _obj(state, ctx):
    masked = {"single": 0, "dual": 1, "triple": 1}[state["arbitration"]]
    rel = kofn_reliability(SENSE[state["sensing"]], masked, ctx.channel_reliability)
    interop = PROTO[state["protocol"]][2]
    units = (
        {"single": 1, "dual": 2, "triple": 3}[state["arbitration"]]
        + SENSE[state["sensing"]]
        + (1 if state["topology"] == "centralised" else 3)
    )
    return {"capability": 0.6 * rel + 0.4 * interop, "cost_efficiency": 1.0 - (units - 3) / 9.0}


def test_weighted_sum_vs_front():
    ctx = Ctx()
    ws_state, ws_obj = weighted_sum_search(
        _OPS, _obj, ctx, {"capability": 0.5, "cost_efficiency": 0.5}, n_evals=300, random_seed=1
    )
    front = MultiObjectiveMCTS(_OPS, _obj, ctx, _NAMES, _REF, random_seed=7).search(iterations=300)
    assert isinstance(ws_state, dict)
    assert len(front.members) >= 2


def test_weighted_sum_not_dominated():
    ctx = Ctx()
    ws_state, ws_obj = weighted_sum_search(
        _OPS, _obj, ctx, {"capability": 0.5, "cost_efficiency": 0.5}, n_evals=300, random_seed=1
    )
    front = MultiObjectiveMCTS(_OPS, _obj, ctx, _NAMES, _REF, random_seed=7).search(iterations=300)
    front_vecs = [(o["capability"], o["cost_efficiency"]) for _, o in front.members]
    ws_vec = (ws_obj["capability"], ws_obj["cost_efficiency"])
    assert not any(dominates(fv, ws_vec) for fv in front_vecs)


def _mean(xs):
    return sum(xs) / len(xs)


def test_mo_mcts_beats_random():
    """Averaged over seeds at a small budget, MO-MCTS reaches higher hypervolume than
    random search.

    Per-seed it can tie on tiny budgets, so no per-seed claim is made.
    """
    ctx = Ctx()
    seeds = range(1, 7)
    mo = [MultiObjectiveMCTS(_OPS, _obj, ctx, _NAMES, _REF, random_seed=s).search(iterations=15).hypervolume() for s in seeds]
    rs = [random_search(_OPS, _obj, ctx, _NAMES, _REF, n_evals=15, random_seed=s).hypervolume() for s in seeds]
    assert _mean(mo) > _mean(rs)


def test_front_hypervolume_beats_point():
    ctx = Ctx()
    from src.dse.mo_mcts import hypervolume_nd

    ws_state, ws_obj = weighted_sum_search(
        _OPS, _obj, ctx, {"capability": 0.5, "cost_efficiency": 0.5}, n_evals=300, random_seed=1
    )
    front = MultiObjectiveMCTS(_OPS, _obj, ctx, _NAMES, _REF, random_seed=7).search(iterations=300)
    single_hv = hypervolume_nd([(ws_obj["capability"], ws_obj["cost_efficiency"])], _REF)
    assert front.hypervolume() >= single_hv


def test_nsga2_front_nondominated():
    ctx = Ctx()
    front = nsga2(_OPS, _obj, ctx, _NAMES, _REF, pop_size=8, generations=6, random_seed=1)
    assert front.hypervolume() > 0.0
    vecs = [(o["capability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_nsga2_and_mo_mcts_converge():
    ctx = Ctx()
    ng = nsga2(_OPS, _obj, ctx, _NAMES, _REF, pop_size=8, generations=8, random_seed=1)
    mo = MultiObjectiveMCTS(_OPS, _obj, ctx, _NAMES, _REF, random_seed=7).search(iterations=200)
    assert abs(ng.hypervolume() - mo.hypervolume()) < 1e-6


def test_mo_mcts_competitive_with_nsga2():
    ctx = Ctx()
    seeds = range(1, 7)
    mo = [MultiObjectiveMCTS(_OPS, _obj, ctx, _NAMES, _REF, random_seed=s).search(iterations=24).hypervolume() for s in seeds]
    ng = [nsga2(_OPS, _obj, ctx, _NAMES, _REF, pop_size=6, generations=3, random_seed=s).hypervolume() for s in seeds]
    assert _mean(mo) >= _mean(ng)
