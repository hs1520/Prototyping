from __future__ import annotations

from dataclasses import dataclass

from src.dse.mo_mcts import MultiObjectiveMCTS, dominates
from src.dse.operators import CATALOG, DecomposeController, RedundantizeComponent
from src.dse.operators.redundantize import parallel_reliability
from src.simulation.syntax_checker import check_syntax


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85
    part_count: int = 5
    allow_distributed: bool = True
    controller_reliability: float = 0.9


def _objectives(state, ctx):
    channels = CATALOG[state["arbitration"]][1]
    nodes = 1 if state["topology"] == "centralised" else 3
    block_r = parallel_reliability(channels, ctx.channel_reliability)
    ctrl_r = (
        ctx.controller_reliability
        if state["topology"] == "centralised"
        else 1.0 - (1.0 - ctx.controller_reliability) ** nodes
    )
    units = channels + nodes
    return {
        "reliability": block_r * ctrl_r,
        "cost_efficiency": 1.0 - (units - 2) / 4.0,
    }


def _make(ctx=None, seed=7) -> MultiObjectiveMCTS:
    return MultiObjectiveMCTS(
        operators=[RedundantizeComponent(), DecomposeController()],
        objective_fn=_objectives,
        ctx=ctx or Ctx(),
        objective_names=["reliability", "cost_efficiency"],
        reference=[0.0, 0.0],
        random_seed=seed,
    )


def _front_keys(front):
    return {(s["arbitration"], s["topology"]) for s, _ in front.members}


def test_terminal_states_both_operators():
    front = _make().search(iterations=120)
    for state, _ in front.members:
        assert "arbitration" in state and "topology" in state


def test_front_non_dominated():
    front = _make().search(iterations=120)
    vecs = [(o["reliability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a)


def test_dominated_combo_excluded():
    front = _make().search(iterations=150)
    assert ("single", "distributed") not in _front_keys(front)


def test_extremes_on_front():
    front = _make().search(iterations=150)
    keys = _front_keys(front)
    assert ("single", "centralised") in keys
    assert ("triple", "distributed") in keys


def test_front_members_valid_sysml():
    red, dec = RedundantizeComponent(), DecomposeController()
    front = _make().search(iterations=120)
    for state, _ in front.members:
        assert not check_syntax(red.resolve(state["arbitration"], with_fanin=True)).has_errors
        assert not check_syntax(dec.resolve(state["topology"], with_wiring=True)).has_errors


def test_infeasible_collapses_topology():
    front = _make(ctx=Ctx(allow_distributed=False)).search(iterations=80)
    assert all(s["topology"] == "centralised" for s, _ in front.members)
