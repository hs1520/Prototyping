"""Tests for the #1 AddRedundantSensor + InsertFusionNode operator."""
from __future__ import annotations

import pytest

from src.dse.operators import AddRedundantSensor
from src.simulation.syntax_checker import check_syntax


@pytest.fixture
def op() -> AddRedundantSensor:
    return AddRedundantSensor()


def _assert_parses(text: str) -> None:
    r = check_syntax(text)
    assert not r.has_errors, r.short_summary() + "\n" + r.format_for_llm()


def test_variants_and_sensor_counts(op: AddRedundantSensor) -> None:
    assert op.variants == ["single", "dual", "triple"]
    assert op.sensors("single") == 1
    assert op.sensors("dual") == 2
    assert op.sensors("triple") == 3


def test_skeleton_parses(op: AddRedundantSensor) -> None:
    _assert_parses(op.declare_skeleton())


@pytest.mark.parametrize("variant", ["single", "dual", "triple"])
def test_resolve_parses(op: AddRedundantSensor, variant: str) -> None:
    _assert_parses(op.resolve(variant))


@pytest.mark.parametrize("variant", ["dual", "triple"])
def test_no_dangling_sensors(op: AddRedundantSensor, variant: str) -> None:
    """Every redundant sensor is wired into the fusion node (the legacy bug fix)."""
    text = op.resolve(variant)
    assert "FusionNode" in text
    assert text.count("connect s") == op.sensors(variant)


def test_single_has_no_fusion(op: AddRedundantSensor) -> None:
    text = op.resolve("single")
    assert "FusionNode" not in text
    assert text.count("connect s") == 0


def test_preconditions_sensor_budget(op: AddRedundantSensor) -> None:
    assert op.preconditions("triple", max_sensors=3)
    assert not op.preconditions("triple", max_sensors=2)
    assert op.preconditions("dual", max_sensors=2)
    assert op.preconditions("single", max_sensors=1)


def test_resolve_unknown_variant_raises(op: AddRedundantSensor) -> None:
    with pytest.raises(ValueError):
        op.resolve("quad")
