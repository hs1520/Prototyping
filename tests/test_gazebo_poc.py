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


def test_throttle_at_thrust_interp():
    m = MN5008_KV340_18x61
    assert m.throttle_at_thrust(500.0) == m.curve[0].throttle      # clamp low
    t = m.throttle_at_thrust(1390.0)                               # between 45% and 50%
    assert 0.45 < t < 0.50


def test_cross_validate_consistent_and_inconsistent():
    from gazebo_poc.cross_validate import cross_validate
    cv = cross_validate(5.5, 4, 16000, gazebo_stable=True)
    assert cv.consistent and cv.datasheet_within_envelope
    assert abs(cv.hover_thrust_per_rotor_g - 5.5 * 1000 / 4) < 1.0
    assert 20 < cv.endurance_min < 45
    # unstable sim → not consistent even if power is fine
    assert not cross_validate(5.5, 4, 16000, gazebo_stable=False).consistent
    # overweight → outside motor envelope → not consistent
    assert not cross_validate(20.0, 4, 16000, gazebo_stable=True).datasheet_within_envelope


def test_prop_hover_rpm_physical():
    from gazebo_poc.prop_theory import prop_hover_rpm
    # higher thrust → higher RPM; bigger prop → lower RPM for same thrust
    assert prop_hover_rpm(20, 0.38) > prop_hover_rpm(10, 0.38)
    assert prop_hover_rpm(13.5, 0.46) < prop_hover_rpm(13.5, 0.38)


def test_rpm_cross_check_passes_at_design_diameter():
    from gazebo_poc.prop_theory import rpm_cross_check
    # Gazebo-measured 4204 RPM for the 5.5 kg quad at the DESIGN diameter (0.19 m radius)
    c = rpm_cross_check(gazebo_rpm=4204, mass_kg=5.5, rotor_count=4, rotor_radius_m=0.19)
    assert abs(c.diameter_m - 0.38) < 1e-9          # design diameter, not 18"
    assert c.within_ct_band                          # agrees with real-prop theory
    assert 0.10 <= c.ct_implied <= 0.13              # implied Ct is physically realistic
    assert abs(c.pct_diff) < 8.0                     # within a few %


def test_rpm_cross_check_flags_wrong_diameter_error():
    from gazebo_poc.prop_theory import rpm_cross_check
    # the earlier 2900-RPM (18" prop) figure is NOT consistent with the 15" design rotor
    c = rpm_cross_check(gazebo_rpm=2900, mass_kg=5.5, rotor_count=4, rotor_radius_m=0.19)
    assert not c.within_ct_band                       # would imply an unphysical Ct


def test_forward_flight_power_is_u_shaped():
    from gazebo_poc.forward_flight import power_at_speed
    hover = power_at_speed(5.5, 4, 0.19, 0.01).power_w
    mid = power_at_speed(5.5, 4, 0.19, 12.0).power_w
    fast = power_at_speed(5.5, 4, 0.19, 25.0).power_w
    assert mid < hover and mid < fast               # U-shape: cheaper to cruise than hover/fast
    # induced drops with speed, parasite grows
    assert power_at_speed(5.5,4,0.19,15).induced_w < power_at_speed(5.5,4,0.19,5).induced_w
    assert power_at_speed(5.5,4,0.19,15).parasite_w > power_at_speed(5.5,4,0.19,5).parasite_w


def test_range_estimate_sane():
    from gazebo_poc.forward_flight import range_estimate
    r = range_estimate(5.5, 4, 0.19, 16000, 6)
    assert r.best_range_speed_mps > r.best_endurance_speed_mps   # always, physically
    assert r.min_power_w < r.hover_power_w                        # cruise cheaper than hover
    assert 5 < r.range_km < 100                                   # plausible for this class


def test_power_coefficient_and_mechanical_power():
    from gazebo_poc.prop_theory import power_coefficient, mechanical_power_w
    assert 0.03 < power_coefficient() < 0.06          # Cp from Ct+FOM, realistic
    # 4 rotors at the measured hover RPM ≈ the analytical hover power (~600 W)
    assert 500 < 4 * mechanical_power_w(4204, 0.38) < 700


def test_drag_area_backout_roundtrips():
    from gazebo_poc.forward_flight import power_at_speed, effective_drag_area_from_power
    p = power_at_speed(5.5, 4, 0.19, 20.0, drag_area=0.08).power_w
    f = effective_drag_area_from_power(p, 20.0, 5.5, 4, 0.19)
    assert abs(f - 0.08) < 0.005                       # measured power → recovers drag area


def test_gazebo_verify_skips_without_design():
    from gazebo_poc.gazebo_verify import summary_line, verify_recommended_design
    v = verify_recommended_design(None)
    assert v["status"] == "skipped"
    assert "skipped" in summary_line(v)


def test_gazebo_verify_skips_non_quad():
    from src.dse.domain_objective import DesignInputs
    from gazebo_poc.gazebo_verify import verify_recommended_design
    d = DesignInputs(payload_mass_kg=1.5, battery_capacity_mah=16000, battery_cells=6,
                     rotor_count=3, rotor_radius_m=0.19, cruise_speed_mps=0.0)
    assert verify_recommended_design(d)["status"] == "skipped"   # tri unsupported (quad/hexa/octa only)


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_hexa_sdf_structure(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
    so, gm, fc = generate_multirotor_sdf(6.0, 6, 0.19, multirotor_inertia(6.0, 6, 0.19),
                                         0.008, _TEMPLATES, tmp_path)
    s, g = so.read_text(), gm.read_text()
    assert fc == 2                                       # ArduCopter HEXA
    assert s.count("<link name='rotor_") == 6           # 6 rotor links
    assert g.count("gz-sim-lift-drag-system") == 12     # 2 blades each
    assert g.count("<control channel=") == 6
    # 3 CCW (+838) + 3 CW (-838) — matches ArduCopter HEXA-X spin pattern
    assert g.count("<multiplier>838</multiplier>") == 3
    assert g.count("<multiplier>-838</multiplier>") == 3
    md.parseString(s); md.parseString(g)                # both well-formed


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_octa_sdf_structure(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
    so, gm, fc = generate_multirotor_sdf(7.0, 8, 0.19, multirotor_inertia(7.0, 8, 0.19),
                                         0.009, _TEMPLATES, tmp_path)
    s, g = so.read_text(), gm.read_text()
    assert fc == 3                                       # ArduCopter OCTA
    assert s.count("<link name='rotor_") == 8
    assert g.count("gz-sim-lift-drag-system") == 16
    assert g.count("<control channel=") == 8
    assert g.count("<multiplier>838</multiplier>") == 4    # 4 CCW + 4 CW (yaw-balanced)
    assert g.count("<multiplier>-838</multiplier>") == 4
    md.parseString(s); md.parseString(g)
