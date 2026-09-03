from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85
    part_count: int = 5
    allow_distributed: bool = True
    max_sensors: int = 3
    allowed_protocols = None


def _objectives(state, ctx):
    masked = {"single": 0, "dual": 1, "triple": 1}[state["arbitration"]]
    rel = kofn_reliability(SENSE[state["sensing"]], masked, ctx.channel_reliability)
    interop = PROTO[state["protocol"]][2]
    capability = 0.6 * rel + 0.4 * interop
    units = (
        {"single": 1, "dual": 2, "triple": 3}[state["arbitration"]]
        + SENSE[state["sensing"]]
        + (1 if state["topology"] == "centralised" else 3)
    )
    return {"capability": capability, "cost_efficiency": 1.0 - (units - 3) / 9.0}


def _make():
    return MultiObjectiveMCTS(
        operators=_OPS,
        objective_fn=_objectives,
        ctx=Ctx(),
        objective_names=["capability", "cost_efficiency"],
        reference=[0.0, 0.0],
        random_seed=11,
    )


def test_terminal_states_four_points():
    front = _make().search(iterations=300)
    for state, _ in front.members:
        assert set(state) == {"arbitration", "topology", "sensing", "protocol"}


def test_front_non_dominated():
    front = _make().search(iterations=300)
    vecs = [(o["capability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_front_spans_tradeoff():
    front = _make().search(iterations=300)
    caps = [o["capability"] for _, o in front.members]
    costs = [o["cost_efficiency"] for _, o in front.members]
    assert max(caps) - min(caps) > 0.05
    assert max(costs) - min(costs) > 0.05


def test_front_members_valid_sysml():
    from src.simulation.syntax_checker import check_syntax

    red, dec, sen, pro = _OPS
    front = _make().search(iterations=200)
    for state, _ in front.members:
        assert not check_syntax(red.resolve(state["arbitration"], with_fanin=True)).has_errors
        assert not check_syntax(dec.resolve(state["topology"], with_wiring=True)).has_errors
        assert not check_syntax(sen.resolve(state["sensing"])).has_errors
        assert not check_syntax(pro.resolve(state["protocol"])).has_errors
