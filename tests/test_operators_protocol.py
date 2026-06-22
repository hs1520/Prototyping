"""Tests for the #3 ReplaceInterfaceProtocol operator."""
from __future__ import annotations

import pytest

from src.dse.operators import ReplaceInterfaceProtocol
from src.dse.operators.protocol import CATALOG
from src.simulation.syntax_checker import check_syntax


@pytest.fixture
def op() -> ReplaceInterfaceProtocol:
    return ReplaceInterfaceProtocol()


def _assert_parses(text: str) -> None:
    r = check_syntax(text)
    assert not r.has_errors, r.short_summary() + "\n" + r.format_for_llm()


def test_variants(op: ReplaceInterfaceProtocol) -> None:
    assert set(op.variants) == {"mavlink", "can", "ethernet"}


def test_skeleton_parses(op: ReplaceInterfaceProtocol) -> None:
    _assert_parses(op.declare_skeleton())


@pytest.mark.parametrize("variant", ["mavlink", "can", "ethernet"])
def test_resolve_parses(op: ReplaceInterfaceProtocol, variant: str) -> None:
    _assert_parses(op.resolve(variant))


@pytest.mark.parametrize("variant", ["mavlink", "can", "ethernet"])
def test_data_ports_typed_by_chosen_protocol(op: ReplaceInterfaceProtocol, variant: str) -> None:
    signal = CATALOG[variant][0]
    text = op.resolve(variant)
    assert f"out port telemetry : {signal}" in text
    assert f"in port command : {signal}" in text


@pytest.mark.parametrize("variant", ["mavlink", "can", "ethernet"])
def test_power_ports_untouched(op: ReplaceInterfaceProtocol, variant: str) -> None:
    """Domain separation: power ports keep PowerPort, never a protocol signal."""
    text = op.resolve(variant)
    assert "in port power : PowerPort;" in text


def test_interop_weight_ordering(op: ReplaceInterfaceProtocol) -> None:
    assert op.interop("ethernet") > op.interop("mavlink") > op.interop("can")


def test_preconditions_allowed_set(op: ReplaceInterfaceProtocol) -> None:
    assert op.preconditions("mavlink")
    assert op.preconditions("mavlink", allowed=["mavlink", "can"])
    assert not op.preconditions("ethernet", allowed=["mavlink", "can"])


def test_resolve_unknown_variant_raises(op: ReplaceInterfaceProtocol) -> None:
    with pytest.raises(ValueError):
        op.resolve("zigbee")
