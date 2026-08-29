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
    assert g.count("<multiplier>838.0</multiplier>") == 3
    assert g.count("<multiplier>-838.0</multiplier>") == 3
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
    assert g.count("<multiplier>838.0</multiplier>") == 4    # 4 CCW + 4 CW (yaw-balanced)
    assert g.count("<multiplier>-838.0</multiplier>") == 4
    md.parseString(s); md.parseString(g)


def test_real_motor_calibration():
    from gazebo_poc.prop_theory import calibrated_area, calibrated_max_rad_s
    from gazebo_poc.component_data import MN5008_KV340_18x61 as motor
    # Ct-matched area for the design diameter is iris-scale and positive
    a = calibrated_area(0.38)
    assert 0.005 < a < 0.012
    # max rotor speed set so full throttle = real motor max thrust
    w = calibrated_max_rad_s(a, motor.max_thrust_g() / 1000.0 * 9.81)
    assert 600 < w < 900                       # realistic, below iris's over-powered 838 default
    # bigger prop → larger Ct-matched area
    assert calibrated_area(0.46) > calibrated_area(0.38)


def test_thrust_diagnostics_separates_sdf_margin_from_controller_failure():
    from gazebo_poc.prop_theory import calibrated_area, calibrated_max_rad_s
    from gazebo_poc.run_flight import _parm_text, _thrust_diagnostics
    area = calibrated_area(0.381)
    max_rad_s = calibrated_max_rad_s(area, 1880.0 / 1000.0 * 9.81)
    d = _thrust_diagnostics(4.678, 4, area, max_rad_s, 1880.0)

    assert d["sdf_can_hover"]
    assert 1.5 < d["model_twr"] < 1.7
    assert 0.7 < d["hover_rad_s_fraction_of_max"] < 0.9
    assert "MOT_THST_HOVER 0.638" in _parm_text(frame_class=1, hover_throttle=0.638)
    actuator_parms = _parm_text(
        frame_class=1, hover_throttle=0.638, gripper_servo=7, parachute_servo=8
    )
    assert "SERVO7_FUNCTION 28" in actuator_parms
    assert "SERVO8_FUNCTION 27" in actuator_parms
    assert "CHUTE_DELAY_MS 0" in actuator_parms


def test_wind_effects_working_point_is_lumped_and_explicit():
    from gazebo_poc.run_flight import (
        _lidar_scan_min,
        _obstacle_requirement_met,
        _obstacle_world_text,
        _vehicle_spawn_heading_deg,
        _wind_force_scale,
        _wind_world_text,
    )

    scale = _wind_force_scale(4.678, 15.0, 0.05)
    expected_force = 4.678 * scale * 15.0
    assert expected_force == pytest.approx(0.5 * 1.2041 * 0.05 * 15.0 ** 2)

    world = _wind_world_text("<sdf><world name='iris_runway'></world></sdf>", scale)
    assert "gz::sim::systems::WindEffects" in world
    assert f"{scale:.9f}" in world
    stock = (
        "<sdf><world name='iris_runway'><include>"
        "<uri>model://iris_with_gimbal</uri>"
        "<pose degrees='true'>0 0 0.195 0 0 90</pose>"
        "</include></world></sdf>"
    )
    assert _vehicle_spawn_heading_deg(stock) == 90.0
    obstacle = _obstacle_world_text(stock)
    assert 'model name="gazebo_test_obstacle"' in obstacle
    assert '<pose degrees="true">0.000 25.000 10 0 0 90.000</pose>' in obstacle
    assert "<size>2 6 20</size>" in obstacle
    assert _lidar_scan_min([float("inf")] * 61) == 15.0
    assert _lidar_scan_min([float("inf"), 7.2, 5.4]) == 5.4
    assert _obstacle_requirement_met(True, True, True, 6.0, 5.0, 5.0, 5.0)
    assert not _obstacle_requirement_met(True, True, True, 6.0, 5.0, 4.999, 5.0)
    # "initiate before 5 m" does not imply a minimum-clearance invariant.
    assert _obstacle_requirement_met(True, True, True, 6.0, 5.0, 3.0, None)
    assert not _obstacle_requirement_met(True, True, True, 4.9, 5.0, 4.9, None)


def test_payload_model_uses_requested_mass_and_physical_inertia(tmp_path):
    from gazebo_poc.run_flight import _parse_model_z, _prepare_payload_model

    path = _prepare_payload_model(tmp_path, 1.5)
    text = path.read_text()
    assert "<mass>1.500000</mass>" in text
    assert "<ixx>0.001250000</ixx>" in text
    assert _parse_model_z("Pose [ XYZ [1.0 2.0 9.75] RPY [0 0 0] ]") == 9.75
    assert _parse_model_z(
        "Model: [49]\n  - Pose [ XYZ (m) ] [ RPY (rad) ]:\n"
        "    [-0.000000 0.000000 10.949999]\n    [0 0 0]\n"
    ) == 10.949999


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_quad_via_multirotor_calibrated_structure(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
    so, gm, fc = generate_multirotor_sdf(5.5, 4, 0.19, multirotor_inertia(5.5, 4, 0.19),
                                         0.00783, _TEMPLATES, tmp_path, max_rotor_rad_s=745.0)
    s, g = so.read_text(), gm.read_text()
    assert fc == 1                                       # QUAD via the unified path
    assert s.count("<link name='rotor_") == 4
    assert g.count("gz-sim-lift-drag-system") == 8
    assert g.count("<control channel=") == 4
    assert g.count("<multiplier>745.0</multiplier>") == 2     # calibrated max rotor speed
    assert g.count("<multiplier>-745.0</multiplier>") == 2    # 2 CCW + 2 CW
    md.parseString(s); md.parseString(g)


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_generated_airframe_can_enable_wind_and_physical_payload_release(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf

    so, gm, _ = generate_multirotor_sdf(
        3.178, 4, 0.1905, multirotor_inertia(4.678, 4, 0.1905),
        0.00783, _TEMPLATES, tmp_path, max_rotor_rad_s=500.0,
        enable_wind=True, payload_release=True, parachute_deploy=True,
        forward_lidar=True,
    )
    standoffs, gimbal = so.read_text(), gm.read_text()
    assert "<enable_wind>true</enable_wind>" in standoffs
    assert '<control channel="6">' in gimbal
    assert "gz-sim-detachable-joint-system" in gimbal
    assert "<child_model>payload_box</child_model>" in gimbal
    assert '<control channel="7">' in gimbal
    assert "ParachutePlugin" in gimbal
    assert "<child_model>parachute_small</child_model>" in gimbal
    assert 'sensor name="forward_lidar" type="gpu_lidar"' in standoffs
    assert "<topic>/forward_lidar</topic>" in standoffs
    md.parseString(standoffs)
    md.parseString(gimbal)


def test_redundancy_requirement_detection():
    from gazebo_poc.gazebo_verify import _redundancy_req
    reqs = ["REQ-PERF-001: cruise at 15 m/s.",
            "REQ-SAFE-007: maintain controlled flight following the failure of a single "
            "propulsion unit (one motor inoperative)."]
    assert _redundancy_req(reqs) == "REQ-SAFE-007"
    assert _redundancy_req(["REQ-PERF-001: cruise."]) is None
    # the parachute req ("critical propulsion subsystem failure") must NOT be mistaken for it
    assert _redundancy_req(["REQ-SAFE-005: deploy the parachute within 0.5s of detecting a "
                            "critical propulsion subsystem failure."]) is None


def test_motor_failure_upgrades_redundancy_req_to_flight_verified():
    from src.dse.requirement_coverage import classify_requirement_coverage, FLIGHT_VERIFIED
    model = """package D {
        requirement def REQ_SAFE_007 { doc /* single motor failure */ }
        part def Drone { satisfy requirement REQ_SAFE_007; }
    }"""
    reqs = ["REQ-SAFE-007: maintain controlled flight following the failure of a single "
            "propulsion unit."]
    # without Gazebo → not flight-verified
    assert classify_requirement_coverage(model, reqs).get("REQ-SAFE-007") != FLIGHT_VERIFIED
    # Gazebo confirmed 1-motor-out controllable → flight-verified
    gv = {"status": "ok", "motor_failure_tolerant": True, "redundancy_req": "REQ-SAFE-007"}
    assert classify_requirement_coverage(model, reqs, gazebo=gv)["REQ-SAFE-007"] == FLIGHT_VERIFIED
    # lost control → NOT flight-verified
    gv2 = {"status": "ok", "motor_failure_tolerant": False, "redundancy_req": "REQ-SAFE-007"}
    assert classify_requirement_coverage(model, reqs, gazebo=gv2)["REQ-SAFE-007"] != FLIGHT_VERIFIED
    # inconclusive (flight errored) → also NOT flight-verified (a transient can't earn the green)
    gv3 = {"status": "ok", "motor_failure_tolerant": None, "redundancy_req": "REQ-SAFE-007"}
    assert classify_requirement_coverage(model, reqs, gazebo=gv3)["REQ-SAFE-007"] != FLIGHT_VERIFIED


def test_attitude_rms_deg_centres_each_axis_on_its_own_mean():
    import math
    from gazebo_poc.run_flight import _attitude_rms_deg

    # constant offset (forward-dash trim pitch) must NOT count as deviation
    samples = [(0.0, math.radians(-8.0))] * 10
    rms = _attitude_rms_deg(samples)
    assert rms["n"] == 10
    assert rms["rms_deg"] == 0.0

    # symmetric ±1° square wave about the mean → RMS exactly 1°
    one = math.radians(1.0)
    samples = [(one, 0.0), (-one, 0.0)] * 8
    rms = _attitude_rms_deg(samples)
    assert abs(rms["roll_rms_deg"] - 1.0) < 1e-9
    assert abs(rms["pitch_rms_deg"]) < 1e-9
    assert abs(rms["rms_deg"] - 1.0) < 1e-9


def test_attitude_rms_deg_needs_at_least_two_samples():
    from gazebo_poc.run_flight import _attitude_rms_deg

    assert _attitude_rms_deg([])["rms_deg"] is None
    assert _attitude_rms_deg([(0.1, 0.2)])["rms_deg"] is None


def test_attitude_sampler_dedupes_on_time_boot_ms():
    from types import SimpleNamespace

    from gazebo_poc.run_flight import _AttitudeSampler

    att = SimpleNamespace(roll=0.01, pitch=0.02, time_boot_ms=1000)
    m = SimpleNamespace(messages={"ATTITUDE": att})
    sampler = _AttitudeSampler()
    sampler.sample(m)
    sampler.sample(m)                       # same message → deduped
    assert len(sampler.samples) == 1

    m.messages["ATTITUDE"] = SimpleNamespace(roll=0.03, pitch=0.04,
                                             time_boot_ms=1100)
    sampler.sample(m)
    assert len(sampler.samples) == 2
    assert sampler.samples[-1] == (0.03, 0.04)

    sampler.sample(SimpleNamespace(messages={}))   # no ATTITUDE yet → no-op
    assert len(sampler.samples) == 2
