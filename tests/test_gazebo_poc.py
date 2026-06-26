"""Datasheet-grounded endurance cross-check (Gazebo hi-fi PoC, stage 1): real motor+prop curve
replaces the estimator's lumped FOM, giving an independent endurance bracket."""
from __future__ import annotations

import pytest

from gazebo_poc.component_data import MN5008_KV340_18x61
from gazebo_poc.datasheet_endurance import datasheet_endurance


def test_curve_interp_endpoints_and_monotonic():
    m = MN5008_KV340_18x61
    # clamps below the first row
    assert m.interp_at_thrust(500.0) == (m.curve[0].current_a, m.curve[0].power_w)
    # interpolates between rows (1238g..1541g) → current between 5.12 and 6.93
    cur, pwr = m.interp_at_thrust(1390.0)
    assert 5.12 < cur < 6.93 and 120.0 < pwr < 162.0


def test_thrust_beyond_max_cannot_hover():
    with pytest.raises(ValueError):
        MN5008_KV340_18x61.interp_at_thrust(MN5008_KV340_18x61.max_thrust_g() + 1.0)


def test_datasheet_endurance_is_sane():
    r = datasheet_endurance(total_mass_kg=5.5, rotor_count=4, battery_capacity_mah=16000)
    assert r.twr_max > 1.0                              # flyable
    assert 10.0 < r.endurance_min < 60.0               # plausible multirotor hover endurance
    assert r.total_hover_current_a > 4 * r.hover_current_per_motor_a - 1e-6  # + avionics load
    assert r.source_url.startswith("https://")         # cited


def test_overweight_design_raises():
    # 4×4215 g ≈ 16.9 kg max thrust → 20 kg can't be hovered by a quad of this motor
    with pytest.raises(ValueError):
        datasheet_endurance(total_mass_kg=20.0, rotor_count=4, battery_capacity_mah=16000)
