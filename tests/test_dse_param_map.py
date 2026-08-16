"""Layer 1: DSE settable-family → ArduPilot .parm mapping + static L1 validation.

Settable families (speed) map to a real parameter with unit conversion and a range
check; emergent families (mass/endurance/range) are skipped — they belong to L2.
See memory `sitl-family-param-mapping`.
"""
from __future__ import annotations

from src.sitl.parameter_projection import (
    check_settable_parameters,
    settable_parm_lines,
    validate_settable_parameters,
)

_MODEL = """package Drone {
    part def QuadProp {
        attribute speedMps : Real = 20.0;
        attribute enduranceMinutes : Real = 35.0;
        attribute massKg : Real = 2.5;
    }
}"""


def test_speed_maps_to_wpnav_speed_with_unit_conversion():
    by_fam = {r.family: r for r in check_settable_parameters(_MODEL)}
    assert "speed" in by_fam
    r = by_fam["speed"]
    assert r.param_name == "WPNAV_SPEED"
    assert r.param_value == 2000.0          # 20 m/s × 100 = 2000 cm/s
    assert r.source_attr == "speedMps"      # traceable to the model attribute


def test_emergent_families_are_skipped():
    fams = {r.family for r in check_settable_parameters(_MODEL)}
    assert "mass" not in fams and "time" not in fams   # can't be set → not L1 params


def test_parm_lines_only_include_settable_in_range():
    lines = settable_parm_lines(_MODEL)
    assert lines == ["WPNAV_SPEED          2000.0"]


def test_l1_passes_when_in_range():
    ok, results = validate_settable_parameters(_MODEL)
    assert ok and all(r.ok for r in results)


def test_l1_fails_out_of_range():
    bad = "package P { part def X { attribute speedMps : Real = 50.0; } }"  # 5000 cm/s > 2000
    ok, results = validate_settable_parameters(bad)
    assert not ok
    assert "out of range" in results[0].message


def test_no_settable_families_is_empty():
    none = "package P { part def X { attribute massKg : Real = 2.0; } }"
    assert check_settable_parameters(none) == []
    assert validate_settable_parameters(none) == (True, [])
