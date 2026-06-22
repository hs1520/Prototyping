"""Multirotor physics estimator (DSE→SITL bridge): design inputs → emergent metrics.

All-up mass is EMERGENT (frame base + battery pack + payload), so a bigger battery
pays a weight penalty — the real battery-vs-endurance-vs-cost trade-off. These tests
pin physical sanity (realistic magnitude + correct monotonic responses), not exact values.
"""
from __future__ import annotations

from src.dse.physics_estimator import (
    DesignInputs,
    disk_area_m2,
    endurance_min,
    estimate,
    hover_power_w,
    range_m,
    total_mass_kg,
)

_BASE = DesignInputs(
    payload_mass_kg=0.5, battery_capacity_mah=5000, battery_cells=4,
    rotor_count=4, rotor_radius_m=0.13,
)


def test_all_up_mass_emerges_from_base_battery_payload():
    m = total_mass_kg(_BASE)
    assert 1.5 < m < 2.5   # ~1.0 frame + ~0.5 battery(5Ah4S) + 0.5 payload


def test_typical_quad_endurance_is_realistic():
    assert 8.0 <= endurance_min(_BASE) <= 30.0


def test_heavier_payload_reduces_endurance():
    heavy = DesignInputs(2.0, 5000, 4, 4, 0.13)
    assert endurance_min(heavy) < endurance_min(_BASE)


def test_bigger_battery_increases_endurance_in_normal_range():
    bigger = DesignInputs(0.5, 8000, 4, 4, 0.13)
    assert endurance_min(bigger) > endurance_min(_BASE)


def test_bigger_battery_pays_a_weight_penalty():
    # the key trade-off mechanism: more capacity → heavier all-up mass → costlier
    bigger = DesignInputs(0.5, 12000, 4, 4, 0.13)
    assert total_mass_kg(bigger) > total_mass_kg(_BASE)


def test_more_disk_area_lowers_hover_power():
    assert hover_power_w(2.0, disk_area_m2(4, 0.18)) < hover_power_w(2.0, disk_area_m2(4, 0.10))


def test_range_scales_with_cruise_speed():
    hover = DesignInputs(0.5, 5000, 4, 4, 0.13, cruise_speed_mps=0.0)
    cruising = DesignInputs(0.5, 5000, 4, 4, 0.13, cruise_speed_mps=15.0)
    assert range_m(hover) == 0.0 and range_m(cruising) > 0.0


def test_estimate_reports_all_metrics():
    m = estimate(_BASE)
    assert {"endurance_min", "range_m", "total_mass_kg", "electrical_power_w", "cruise_speed_mps"} <= set(m)


def test_zero_area_is_safe():
    assert endurance_min(DesignInputs(0.5, 5000, 4, 0, 0.13)) == 0.0


def test_bigger_rotor_is_heavier():
    # rotor sizing now has a mass cost (motors+props+arms ∝ disk area)
    small = total_mass_kg(DesignInputs(0.5, 5000, 4, 4, 0.13))
    big = total_mass_kg(DesignInputs(0.5, 5000, 4, 4, 0.25))
    assert big > small


def test_rotor_sizing_has_internal_optimum():
    # bigger rotor = more efficient but heavier → endurance peaks at a middle radius,
    # NOT the largest (so the outer search won't collapse to max rotor)
    base = endurance_min(DesignInputs(0.5, 5000, 4, 4, 0.13))
    mid = endurance_min(DesignInputs(0.5, 5000, 4, 4, 0.16))
    huge = endurance_min(DesignInputs(0.5, 5000, 4, 4, 0.30))
    assert mid >= base and mid > huge
