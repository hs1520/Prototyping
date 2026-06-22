"""Integration test: RedundantizeComponent wired into multi-objective MCTS.

Demonstrates the bilevel outer layer end-to-end:
  operator (variation resolution) → MO-MCTS → Pareto front,
with every front member's resolved SysML validated through Syside.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.dse.mo_mcts import MultiObjectiveMCTS, ParetoArchive, dominates
from src.dse.operators import CATALOG, RedundantizeComponent
from src.dse.operators.redundantize import parallel_reliability
from src.simulation.syntax_checker import check_syntax


@dataclass
class Ctx:
    num_sensors: int = 3
    is_safety_critical: bool = True
    channel_reliability: float = 0.85


def _objectives(state, ctx):
    """Two maximised objectives from the resolved redundancy choice."""
    variant = state["arbitration"]
    channels = CATALOG[variant][1]
    return {
        "reliability": parallel_reliability(channels, ctx.channel_reliability),
        "cost_efficiency": 1.0 - (channels - 1) / 3.0,
    }


def _make() -> MultiObjectiveMCTS:
    op = RedundantizeComponent()
    return MultiObjectiveMCTS(
        operators=[op],
        objective_fn=_objectives,
        ctx=Ctx(),
        objective_names=["reliability", "cost_efficiency"],
        reference=[0.0, 0.0],
        random_seed=42,
    )


def test_hypervolume_sweep_is_correct():
    arc = ParetoArchive(["a", "b"], [0.0, 0.0])
    arc.add({"x": "1"}, {"a": 0.85, "b": 1.0})
    arc.add({"x": "2"}, {"a": 0.9775, "b": 0.667})
    arc.add({"x": "3"}, {"a": 0.99663, "b": 0.333})
    # strips: 0.85*1.0 + 0.1275*0.667 + 0.01913*0.333 ≈ 0.9414
    assert abs(arc.hypervolume() - 0.9414) < 1e-2


def test_dominates():
    assert dominates([1.0, 1.0], [0.5, 0.5])
    assert not dominates([1.0, 0.4], [0.5, 0.5])  # trade-off → no domination


def test_front_contains_all_three_variants():
    mcts = _make()
    front = mcts.search(iterations=50)
    variants = {s["arbitration"] for s, _ in front.members}
    # single (cheap, low reliability), dual, triple (reliable, costly) — all on the front
    assert variants == {"single", "dual", "triple"}


def test_front_is_non_dominated():
    mcts = _make()
    front = mcts.search(iterations=50)
    vecs = [(o["reliability"], o["cost_efficiency"]) for _, o in front.members]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not dominates(b, a), "front member is dominated"


def test_every_front_member_resolves_to_valid_sysml():
    """Valid-by-construction holds inside the search loop, not just in unit tests."""
    op = RedundantizeComponent()
    mcts = _make()
    front = mcts.search(iterations=50)
    for state, _ in front.members:
        sysml = op.resolve(state["arbitration"], with_fanin=True)
        result = check_syntax(sysml)
        assert not result.has_errors, result.short_summary()


def test_hypervolume_positive_after_search():
    mcts = _make()
    front = mcts.search(iterations=50)
    assert front.hypervolume() > 0.0


def test_infeasible_variants_pruned_when_few_sensors():
    """With 1 sensor, dual/triple are infeasible → front collapses to single."""
    op = RedundantizeComponent()
    mcts = MultiObjectiveMCTS(
        operators=[op],
        objective_fn=_objectives,
        ctx=Ctx(num_sensors=1),
        objective_names=["reliability", "cost_efficiency"],
        reference=[0.0, 0.0],
        random_seed=1,
    )
    front = mcts.search(iterations=20)
    assert {s["arbitration"] for s, _ in front.members} == {"single"}
