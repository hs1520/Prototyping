from __future__ import annotations

import pytest

from src.dse.operators import DecomposeController
from src.simulation.syntax_checker import check_syntax


@pytest.fixture
def op() -> DecomposeController:
    return DecomposeController()


def _assert_parses(sysml_text: str) -> None:
    result = check_syntax(sysml_text)
    assert not result.has_errors, result.short_summary() + "\n" + result.format_for_llm()


def test_variants_and_node_counts(op: DecomposeController) -> None:
    assert op.variants == ["centralised", "distributed"]
    assert op.nodes("centralised") == 1
    assert op.nodes("distributed") == 3


def test_skeleton_parses(op: DecomposeController) -> None:
    _assert_parses(op.declare_skeleton())


@pytest.mark.parametrize("variant", ["centralised", "distributed"])
def test_resolve_parses(op: DecomposeController, variant: str) -> None:
    _assert_parses(op.resolve(variant))


def test_distributed_wiring_connects(op: DecomposeController) -> None:
    text = op.resolve("distributed", with_wiring=True)
    _assert_parses(text)
    assert text.count("connect coordinator.toNode") == 3


def test_resolve_sets_controller_nodes(op: DecomposeController) -> None:
    assert "controllerNodes : Integer = 1" in op.resolve("centralised")
    assert "controllerNodes : Integer = 3" in op.resolve("distributed")


def test_resolve_unknown_variant_raises(op: DecomposeController) -> None:
    with pytest.raises(ValueError):
        op.resolve("federated")


def test_preconditions(op: DecomposeController) -> None:
    assert op.preconditions("centralised", part_count=1)
    assert op.preconditions("distributed", part_count=5)
    assert not op.preconditions("distributed", part_count=2)
    assert not op.preconditions("distributed", part_count=5, allow_distributed=False)
