"""Payload family enters datasheet closure scope (closure admission rule).

The rule: a family enters CLOSURE_SCOPE_FAMILIES iff it is computable from
manufacturer data with ZERO assumed constants. Payload capacity qualifies:
realized metrics are computed CARRYING the rated delivery payload, and hover
throttle comes from bench-curve interpolation. The evidence was always there
(the matcher's hover_throttle interface check) — this routes it to the
requirement instead of leaving the requirement unassigned.

Honesty checks: the estimator has no throttle model (estimator_value stays 0),
and attitude/behaviour clauses in the same requirement are NOT claimed.
"""
from __future__ import annotations

import dataclasses

from src.dse.physics_estimator import DesignInputs
from src.realization.bottom_up import RealizedMetrics
from src.realization.closure import close_the_loop
from src.realization.closure_types import (
    CLOSURE_SCOPE_FAMILIES,
    requirement_verdicts,
)

_DESIGN = DesignInputs(
    payload_mass_kg=1.5, battery_capacity_mah=16000.0, battery_cells=6,
    rotor_count=6, rotor_radius_m=0.2032,
)

_ENDURANCE = "REQ-PERF-002: The system shall sustain continuous flight for a minimum of 25 minutes at maximum rated payload."
_MTOW = "REQ-CONS-003: The maximum take-off mass of the system shall not exceed 8.0 kg."
_PAYLOAD_30 = ("REQ-FUNC-003: The system shall transport payloads with a gross mass of up "
               "to 1.5 kg while maintaining a hover throttle margin of at least 30 percent "
               "and roll and pitch RMS within 1.0 degree.")
_PAYLOAD_PLAIN = ("REQ-FUNC-003: The system shall transport payloads with a gross mass of "
                  "up to 1.5 kg.")


def _metrics(hover_throttle: float) -> RealizedMetrics:
    # Filter kwargs by the dataclass's actual fields so the fixture works both
    # before and after the voltage-derating fields land on RealizedMetrics.
    want = dict(
        total_mass_kg=5.54, hover_thrust_per_motor_g=923.0,
        hover_current_per_motor_a=4.1, hover_throttle=hover_throttle,
        total_hover_current_a=24.6, twr_max=2.14, endurance_min=31.2,
        range_m=0.0, cost=5.54, pack_voltage_v=22.2, voltage_ratio=0.925,
        derated_max_thrust_per_motor_g=1975.0,
    )
    fields = {f.name for f in dataclasses.fields(RealizedMetrics)}
    return RealizedMetrics(**{k: v for k, v in want.items() if k in fields})


def test_payload_family_is_in_closure_scope():
    assert "payload" in CLOSURE_SCOPE_FAMILIES


def test_payload_verdict_uses_bench_margin_and_states_its_boundary():
    verdicts = requirement_verdicts(
        _DESIGN, _metrics(hover_throttle=0.54),
        [_ENDURANCE, _MTOW, _PAYLOAD_30])
    by_fam = {v.family: v for v in verdicts}
    v = by_fam["payload"]
    assert v.scope == "closure"
    assert v.met is True                          # margin 0.46 >= 0.30
    assert abs(v.realized_value - 0.46) < 1e-9
    assert abs(v.target - 0.30) < 1e-9
    assert v.estimator_value == 0.0               # estimator has no throttle model
    assert "not covered at the datasheet tier" in v.note  # RMS clause not claimed


def test_stricter_margin_than_achieved_fails_the_verdict():
    strict = _PAYLOAD_30.replace("at least 30 percent", "at least 60 percent")
    verdicts = requirement_verdicts(
        _DESIGN, _metrics(hover_throttle=0.54), [_ENDURANCE, _MTOW, strict])
    v = {x.family: x for x in verdicts}["payload"]
    assert v.met is False                         # margin 0.46 < 0.60


def test_no_explicit_margin_falls_back_to_hovers_within_curve():
    verdicts = requirement_verdicts(
        _DESIGN, _metrics(hover_throttle=0.54), [_ENDURANCE, _MTOW, _PAYLOAD_PLAIN])
    v = {x.family: x for x in verdicts}["payload"]
    assert v.met is True and v.target == 0.0
    assert "no explicit margin" in v.note


def test_end_to_end_closed_still_holds_with_payload_scope():
    report = close_the_loop(_DESIGN, [], [_ENDURANCE, _MTOW, _PAYLOAD_30])
    assert report.verdict in ("CLOSED", "CLOSED_AFTER_RESIZE")
    payload = [v for v in report.per_requirement if v.family == "payload"]
    assert payload and payload[0].met is True


def test_end_to_end_unachievable_margin_blocks_closure_with_named_attribution():
    # 95 % margin needs hover throttle <= 0.05 — no real combo achieves it.
    impossible = _PAYLOAD_30.replace("at least 30 percent", "at least 95 percent")
    report = close_the_loop(_DESIGN, [], [_ENDURANCE, _MTOW, impossible])
    assert report.verdict == "INFEASIBLE_REALIZATION"
    assert any("payload" in c.detail for c in report.failed_checks)
