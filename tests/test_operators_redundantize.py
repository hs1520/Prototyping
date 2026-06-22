"""Tests for the #2 RedundantizeComponent architecture operator.

Every emitted SysML fragment is validated through the real Syside syntax
checker — the operator's contract is "valid-by-construction", so any output
that fails to parse is a hard failure.
"""
from __future__ import annotations

import pytest

from src.dse.operators import RedundantizeComponent
from src.simulation.syntax_checker import check_syntax


@pytest.fixture
def op() -> RedundantizeComponent:
    return RedundantizeComponent(target_part="SafetyMonitor")


def _assert_parses(sysml_text: str) -> None:
    result = check_syntax(sysml_text)
    assert not result.has_errors, (
        f"expected 0 syntax errors, got: {result.short_summary()}\n"
        + result.format_for_llm()
    )


def test_variants_and_channel_counts(op: RedundantizeComponent) -> None:
    assert op.variants == ["single", "dual", "triple"]
    assert op.channels("single") == 1
    assert op.channels("dual") == 2
    assert op.channels("triple") == 3


def test_skeleton_parses(op: RedundantizeComponent) -> None:
    """The variation-point skeleton (all variants open) must be valid SysML v2."""
    _assert_parses(op.declare_skeleton())


@pytest.mark.parametrize("variant", ["single", "dual", "triple"])
def test_resolve_parses(op: RedundantizeComponent, variant: str) -> None:
    """Each resolved (variant-bound) concrete model must parse cleanly."""
    _assert_parses(op.resolve(variant))


@pytest.mark.parametrize("variant", ["dual", "triple"])
def test_resolve_with_fanin_parses(op: RedundantizeComponent, variant: str) -> None:
    """Redundant sensors wired into distinct channel input ports = legal fan-in."""
    text = op.resolve(variant, with_fanin=True)
    _assert_parses(text)
    # Every redundant sensor is connected (no dangling sensor — the legacy bug)
    assert text.count("connect s") == op.channels(variant)


def test_resolve_sets_redundancy_channels(op: RedundantizeComponent) -> None:
    assert "redundancyChannels : Integer = 3" in op.resolve("triple")
    assert "redundancyChannels : Integer = 2" in op.resolve("dual")


def test_resolve_unknown_variant_raises(op: RedundantizeComponent) -> None:
    with pytest.raises(ValueError):
        op.resolve("quadruple")


def test_preconditions_safety_and_sensor_coupling(op: RedundantizeComponent) -> None:
    # non-safety-critical → only single is admissible
    assert op.preconditions("single", num_sensors=5, is_safety_critical=False)
    assert not op.preconditions("triple", num_sensors=5, is_safety_critical=False)
    # safety-critical → variant needs enough independent sensor channels
    assert op.preconditions("triple", num_sensors=3, is_safety_critical=True)
    assert not op.preconditions("triple", num_sensors=2, is_safety_critical=True)
    assert op.preconditions("dual", num_sensors=2, is_safety_critical=True)
    assert not op.preconditions("dual", num_sensors=1, is_safety_critical=True)
