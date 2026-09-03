"""Tests for grounded safety evaluation (causal fault chain over the executed model).

The full causal model needs the platform wiring (sensors in, failsafe output out),
so evaluations use resolve(with_fanin=True).
"""
from __future__ import annotations

import re

import pytest

from src.dse.grounded_eval import grounded_safety
from src.dse.operators import RedundantizeComponent


@pytest.fixture
def op() -> RedundantizeComponent:
    return RedundantizeComponent()


def _real(op, variant):
    return op.resolve(variant, with_fanin=True)


def test_channel_depth_read_from_model(op: RedundantizeComponent) -> None:
    assert grounded_safety(op.resolve("single")).channels == 1
    assert grounded_safety(_real(op, "dual")).channels == 2
    assert grounded_safety(_real(op, "triple")).channels == 3


def test_redundant_models_causally_complete(op: RedundantizeComponent) -> None:
    for v in ("dual", "triple"):
        g = grounded_safety(_real(op, v))
        assert g.guard_grounded and g.output_connected and g.has_failsafe
        assert g.causal_complete


def test_faults_masked_from_guard(op: RedundantizeComponent) -> None:
    assert grounded_safety(_real(op, "dual")).faults_masked == 1
    assert grounded_safety(_real(op, "triple")).faults_masked == 1


def test_reliability_voting_order(op: RedundantizeComponent) -> None:
    """TMR is 2oo3 (masks 1), so at high per-channel reliability it is less reliable
    than dual 1oo2, which masks 1 with fewer channels. Expected ordering:
    single < triple < dual.
    """
    single = grounded_safety(op.resolve("single")).reliability
    dual = grounded_safety(_real(op, "dual")).reliability
    triple = grounded_safety(_real(op, "triple")).reliability
    assert single < triple < dual
    assert grounded_safety(_real(op, "triple")).faults_masked == 1


def test_free_guard_variable_caught(op: RedundantizeComponent) -> None:
    real = _real(op, "triple")
    fake = re.sub(
        r"attribute failedChannels : Integer =[^;]+;",
        "attribute failedChannels : Integer = 0;",
        real,
        flags=re.DOTALL,
    )
    g = grounded_safety(fake)
    assert not g.guard_grounded
    assert not g.causal_complete
    assert g.reliability < grounded_safety(real).reliability * 0.7


def test_dangling_failsafe_caught(op: RedundantizeComponent) -> None:
    real = _real(op, "triple")
    fake = real.replace(
        "        connect monitor.arbitration.overrideCmd to controller.cmdIn;\n", ""
    )
    g = grounded_safety(fake)
    assert not g.output_connected
    assert not g.causal_complete
    assert g.reliability < grounded_safety(real).reliability * 0.7


def test_missing_fault_transition_caught(op: RedundantizeComponent) -> None:
    real = _real(op, "triple")
    fake = re.sub(r"transition n2f[^;]+;", "", real, flags=re.DOTALL)
    g = grounded_safety(fake)
    assert not g.has_failsafe
    assert not g.causal_complete
    assert g.reliability < grounded_safety(real).reliability * 0.7
