"""Datasheet-grounded endurance cross-check (Gazebo hi-fi PoC, stage 1): real motor+prop curve
replaces the estimator's lumped FOM, giving an independent endurance bracket."""
from __future__ import annotations

from pathlib import Path

import pytest

from gazebo_poc.component_data import MN5008_KV340_18x61
from gazebo_poc.datasheet_endurance import datasheet_endurance
from gazebo_poc.sdf_generator import generate_sdf, multirotor_inertia

_TEMPLATES = Path("gazebo_poc/templates")
_HAS_TEMPLATES = (_TEMPLATES / "all_models" / "iris_with_gimbal" / "model.sdf").exists()


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


def test_inertia_is_physical_for_heavy_quad():
    # a 5.5 kg, 0.19 m quad should have Ixx in the ~0.1-0.3 range (NOT iris's tiny 0.008)
    ixx, iyy, izz = multirotor_inertia(5.5, 4, 0.19)
    assert ixx == iyy                      # symmetric
    assert izz > ixx                       # yaw inertia largest (all motors in plane)
    assert 0.05 < ixx < 0.4                # physical for this size/mass
    # heavier → more inertia
    assert multirotor_inertia(8.0, 4, 0.19)[0] > ixx


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent (regenerate via docker cp)")
def test_generate_sdf_injects_mass_inertia_area(tmp_path):
    g = generate_sdf(5.5, 4, 0.19, template_dir=_TEMPLATES, out_dir=tmp_path)
    so, gm = g.standoffs_path.read_text(), g.gimbal_path.read_text()
    assert "<mass>5.5000</mass>" in so          # our mass, not iris 1.5
    assert "<mass>1.5</mass>" not in so
    assert f"<ixx>{g.inertia[0]:.6f}</ixx>" in so
    assert gm.count(f"<area>{0.002 * g.area_scale:.6f}</area>") == 8   # all rotors scaled
    assert abs(g.area_scale - 5.5 / 1.5) < 1e-9                        # preserve iris T/W


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_non_quad_not_supported():
    with pytest.raises(NotImplementedError):
        generate_sdf(5.5, 6, 0.19, template_dir=_TEMPLATES, out_dir=Path("/tmp/x"))
