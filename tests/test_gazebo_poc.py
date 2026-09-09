from __future__ import annotations

from pathlib import Path

import pytest

from gazebo_poc.component_data import MN5008_KV340_18x61
from gazebo_poc.datasheet_endurance import datasheet_endurance
from gazebo_poc.sdf_generator import generate_sdf, multirotor_inertia

_TEMPLATES = Path("gazebo_poc/templates")
_HAS_TEMPLATES = (_TEMPLATES / "all_models" / "iris_with_gimbal" / "model.sdf").exists()


def test_curve_interp_endpoints():
    m = MN5008_KV340_18x61
    assert m.interp_at_thrust(500.0) == (m.curve[0].current_a, m.curve[0].power_w)
    cur, pwr = m.interp_at_thrust(1390.0)
    assert 5.12 < cur < 6.93 and 120.0 < pwr < 162.0


def test_thrust_beyond_max_cannot_hover():
    with pytest.raises(ValueError):
        MN5008_KV340_18x61.interp_at_thrust(MN5008_KV340_18x61.max_thrust_g() + 1.0)


def test_datasheet_endurance_sane():
    r = datasheet_endurance(total_mass_kg=5.5, rotor_count=4, battery_capacity_mah=16000)
    assert r.twr_max > 1.0
    assert 10.0 < r.endurance_min < 60.0
    assert r.total_hover_current_a > 4 * r.hover_current_per_motor_a - 1e-6
    assert r.source_url.startswith("https://")


def test_overweight_design_raises():
    # 4x4215 g ~ 16.9 kg max thrust -> 20 kg can't be hovered by a quad of this motor
    with pytest.raises(ValueError):
        datasheet_endurance(total_mass_kg=20.0, rotor_count=4, battery_capacity_mah=16000)


def test_inertia_physical_heavy_quad():
    # a 5.5 kg, 0.19 m quad should have Ixx ~0.1-0.3 (iris's 0.008 is too small)
    ixx, iyy, izz = multirotor_inertia(5.5, 4, 0.19)
    assert ixx == iyy
    assert izz > ixx
    assert 0.05 < ixx < 0.4
    assert multirotor_inertia(8.0, 4, 0.19)[0] > ixx


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent (regenerate via docker cp)")
def test_sdf_injects_mass_inertia(tmp_path):
    g = generate_sdf(5.5, 4, 0.19, template_dir=_TEMPLATES, out_dir=tmp_path)
    so, gm = g.standoffs_path.read_text(), g.gimbal_path.read_text()
    assert "<mass>5.5000</mass>" in so
    assert "<mass>1.5</mass>" not in so
    assert f"<ixx>{g.inertia[0]:.6f}</ixx>" in so
    assert gm.count(f"<area>{0.002 * g.area_scale:.6f}</area>") == 8
    assert abs(g.area_scale - 5.5 / 1.5) < 1e-9


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_non_quad_not_supported():
    with pytest.raises(NotImplementedError):
        generate_sdf(5.5, 6, 0.19, template_dir=_TEMPLATES, out_dir=Path("/tmp/x"))


def test_throttle_at_thrust_interp():
    m = MN5008_KV340_18x61
    assert m.throttle_at_thrust(500.0) == m.curve[0].throttle
    t = m.throttle_at_thrust(1390.0)
    assert 0.45 < t < 0.50


def test_cross_validate_consistency():
    from gazebo_poc.cross_validate import cross_validate
    cv = cross_validate(5.5, 4, 16000, gazebo_stable=True)
    assert cv.consistent and cv.datasheet_within_envelope
    assert abs(cv.hover_thrust_per_rotor_g - 5.5 * 1000 / 4) < 1.0
    assert 20 < cv.endurance_min < 45
    assert not cross_validate(5.5, 4, 16000, gazebo_stable=False).consistent
    assert not cross_validate(20.0, 4, 16000, gazebo_stable=True).datasheet_within_envelope


def test_prop_hover_rpm_physical():
    from gazebo_poc.prop_theory import prop_hover_rpm
    assert prop_hover_rpm(20, 0.38) > prop_hover_rpm(10, 0.38)
    assert prop_hover_rpm(13.5, 0.46) < prop_hover_rpm(13.5, 0.38)


def test_rpm_check_design_diameter():
    from gazebo_poc.prop_theory import rpm_cross_check
    c = rpm_cross_check(gazebo_rpm=4204, mass_kg=5.5, rotor_count=4, rotor_radius_m=0.19)
    assert abs(c.diameter_m - 0.38) < 1e-9
    assert c.within_ct_band
    assert 0.10 <= c.ct_implied <= 0.13
    assert abs(c.pct_diff) < 8.0


def test_rpm_check_wrong_diameter():
    from gazebo_poc.prop_theory import rpm_cross_check
    c = rpm_cross_check(gazebo_rpm=2900, mass_kg=5.5, rotor_count=4, rotor_radius_m=0.19)
    assert not c.within_ct_band                       # would imply an unphysical Ct


def test_forward_power_u_shaped():
    from gazebo_poc.forward_flight import power_at_speed
    hover = power_at_speed(5.5, 4, 0.19, 0.01).power_w
    mid = power_at_speed(5.5, 4, 0.19, 12.0).power_w
    fast = power_at_speed(5.5, 4, 0.19, 25.0).power_w
    assert mid < hover and mid < fast
    assert power_at_speed(5.5,4,0.19,15).induced_w < power_at_speed(5.5,4,0.19,5).induced_w
    assert power_at_speed(5.5,4,0.19,15).parasite_w > power_at_speed(5.5,4,0.19,5).parasite_w


def test_range_estimate_sane():
    from gazebo_poc.forward_flight import range_estimate
    r = range_estimate(5.5, 4, 0.19, 16000, 6)
    assert r.best_range_speed_mps > r.best_endurance_speed_mps
    assert r.min_power_w < r.hover_power_w
    assert 5 < r.range_km < 100


def test_power_coefficient_mech_power():
    from gazebo_poc.prop_theory import power_coefficient, mechanical_power_w
    assert 0.03 < power_coefficient() < 0.06
    assert 500 < 4 * mechanical_power_w(4204, 0.38) < 700


def test_drag_area_backout_roundtrips():
    from gazebo_poc.forward_flight import power_at_speed, effective_drag_area_from_power
    p = power_at_speed(5.5, 4, 0.19, 20.0, drag_area=0.08).power_w
    f = effective_drag_area_from_power(p, 20.0, 5.5, 4, 0.19)
    assert abs(f - 0.08) < 0.005


def test_verify_skips_no_design():
    from gazebo_poc.gazebo_verify import summary_line, verify_recommended_design
    v = verify_recommended_design(None)
    assert v["status"] == "skipped"
    assert "skipped" in summary_line(v)


def test_verify_skips_non_quad():
    from src.dse.domain_objective import DesignInputs
    from gazebo_poc.gazebo_verify import verify_recommended_design
    d = DesignInputs(payload_mass_kg=1.5, battery_capacity_mah=16000, battery_cells=6,
                     rotor_count=3, rotor_radius_m=0.19, cruise_speed_mps=0.0)
    assert verify_recommended_design(d)["status"] == "skipped"


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_hexa_sdf_structure(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
    so, gm, fc = generate_multirotor_sdf(6.0, 6, 0.19, multirotor_inertia(6.0, 6, 0.19),
                                         0.008, _TEMPLATES, tmp_path)
    s, g = so.read_text(), gm.read_text()
    assert fc == 2
    assert s.count("<link name='rotor_") == 6
    assert g.count("gz-sim-lift-drag-system") == 12
    assert g.count("<control channel=") == 6
    # 3 CCW (+838) + 3 CW (-838) - matches ArduCopter HEXA-X spin pattern
    assert g.count("<multiplier>838.0</multiplier>") == 3
    assert g.count("<multiplier>-838.0</multiplier>") == 3
    md.parseString(s); md.parseString(g)


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_octa_sdf_structure(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
    so, gm, fc = generate_multirotor_sdf(7.0, 8, 0.19, multirotor_inertia(7.0, 8, 0.19),
                                         0.009, _TEMPLATES, tmp_path)
    s, g = so.read_text(), gm.read_text()
    assert fc == 3
    assert s.count("<link name='rotor_") == 8
    assert g.count("gz-sim-lift-drag-system") == 16
    assert g.count("<control channel=") == 8
    assert g.count("<multiplier>838.0</multiplier>") == 4
    assert g.count("<multiplier>-838.0</multiplier>") == 4
    md.parseString(s); md.parseString(g)


def test_real_motor_calibration():
    from gazebo_poc.prop_theory import calibrated_area, calibrated_max_rad_s
    from gazebo_poc.component_data import MN5008_KV340_18x61 as motor
    a = calibrated_area(0.38)
    assert 0.005 < a < 0.012
    w = calibrated_max_rad_s(a, motor.max_thrust_g() / 1000.0 * 9.81)
    assert 600 < w < 900
    assert calibrated_area(0.46) > calibrated_area(0.38)


def test_thrust_diagnostics_margin():
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


def test_wind_force_scale_lumped():
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


def test_payload_model_mass_inertia(tmp_path):
    from gazebo_poc.run_flight import (
        _parse_model_xyz,
        _parse_model_z,
        _prepare_payload_model,
    )

    path = _prepare_payload_model(tmp_path, 1.5)
    text = path.read_text()
    assert "<mass>1.500000</mass>" in text
    assert "<ixx>0.001250000</ixx>" in text
    assert _parse_model_z("Pose [ XYZ [1.0 2.0 9.75] RPY [0 0 0] ]") == 9.75
    assert _parse_model_z(
        "Model: [49]\n  - Pose [ XYZ (m) ] [ RPY (rad) ]:\n"
        "    [-0.000000 0.000000 10.949999]\n    [0 0 0]\n"
    ) == 10.949999
    assert _parse_model_xyz(
        "Model: [49]\n  - Pose [ XYZ (m) ] [ RPY (rad) ]:\n"
        "    [-0.250000 1.500000 10.949999]\n    [0 0 0]\n"
    ) == (-0.25, 1.5, 10.949999)


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_quad_sdf_structure(tmp_path):
    import xml.dom.minidom as md
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
    so, gm, fc = generate_multirotor_sdf(5.5, 4, 0.19, multirotor_inertia(5.5, 4, 0.19),
                                         0.00783, _TEMPLATES, tmp_path, max_rotor_rad_s=745.0)
    s, g = so.read_text(), gm.read_text()
    assert fc == 1
    assert s.count("<link name='rotor_") == 4
    assert g.count("gz-sim-lift-drag-system") == 8
    assert g.count("<control channel=") == 4
    assert g.count("<multiplier>745.0</multiplier>") == 2
    assert g.count("<multiplier>-745.0</multiplier>") == 2
    md.parseString(s); md.parseString(g)


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent")
def test_airframe_wind_payload_release(tmp_path):
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
    # the parachute req ("critical propulsion subsystem failure") is not it
    assert _redundancy_req(["REQ-SAFE-005: deploy the parachute within 0.5s of detecting a "
                            "critical propulsion subsystem failure."]) is None


def test_motor_failure_flight_verified():
    from src.dse.requirement_coverage import classify_requirement_coverage, FLIGHT_VERIFIED
    model = """package D {
        requirement def REQ_SAFE_007 { doc /* single motor failure */ }
        part def Drone { satisfy requirement REQ_SAFE_007; }
    }"""
    reqs = ["REQ-SAFE-007: maintain controlled flight following the failure of a single "
            "propulsion unit."]
    assert classify_requirement_coverage(model, reqs).get("REQ-SAFE-007") != FLIGHT_VERIFIED
    gv = {"status": "ok", "motor_failure_tolerant": True, "redundancy_req": "REQ-SAFE-007"}
    assert classify_requirement_coverage(model, reqs, gazebo=gv)["REQ-SAFE-007"] == FLIGHT_VERIFIED
    gv2 = {"status": "ok", "motor_failure_tolerant": False, "redundancy_req": "REQ-SAFE-007"}
    assert classify_requirement_coverage(model, reqs, gazebo=gv2)["REQ-SAFE-007"] != FLIGHT_VERIFIED
    # inconclusive (flight errored) -> not flight-verified either
    gv3 = {"status": "ok", "motor_failure_tolerant": None, "redundancy_req": "REQ-SAFE-007"}
    assert classify_requirement_coverage(model, reqs, gazebo=gv3)["REQ-SAFE-007"] != FLIGHT_VERIFIED


def test_jitter_excludes_offset():
    import math
    from gazebo_poc.run_flight import _attitude_rms_deg

    samples = [(0.0, math.radians(-8.0))] * 10
    rms = _attitude_rms_deg(samples)
    assert rms["n"] == 10
    assert rms["jitter_rms_about_window_mean_deg"] == 0.0

    one = math.radians(1.0)
    samples = [(one, 0.0), (-one, 0.0)] * 8
    rms = _attitude_rms_deg(samples)
    assert abs(rms["roll_rms_deg"] - 1.0) < 1e-9
    assert abs(rms["pitch_rms_deg"]) < 1e-9
    assert abs(rms["jitter_rms_about_window_mean_deg"] - 1.0) < 1e-9


def test_rms_against_commanded_attitude():
    """"Attitude deviations within 0.5 degree RMS" bounds error from the commanded
    attitude.

    Centring on the window's own mean measures jitter around whatever it settled
    at, so a steady 12 deg tracking error scores 0.003 deg.
    """
    import math
    from gazebo_poc.run_flight import _attitude_rms_deg

    bias = math.radians(12.0)
    samples = [(t * 0.1, bias + 0.0001 * (t % 2), bias + 0.0001 * (t % 2), 0.0, 0.0)
               for t in range(40)]
    rms = _attitude_rms_deg(samples)

    assert rms["jitter_rms_about_window_mean_deg"] < 0.01
    assert abs(rms["rms_about_command_deg"] - 12.0) < 0.01
    assert abs(rms["mean_roll_error_deg"] - 12.0) < 0.01
    assert rms["rms_deg"] == rms["rms_about_command_deg"]
    assert rms["command_samples"] == 40


def test_no_command_no_rms():
    import math
    from gazebo_poc.run_flight import _attitude_rms_deg

    samples = [(t * 0.1, math.radians(12.0), math.radians(12.0)) for t in range(20)]
    rms = _attitude_rms_deg(samples)

    assert rms["jitter_rms_about_window_mean_deg"] == 0.0
    assert rms["rms_about_command_deg"] is None
    assert rms["rms_deg"] is None
    assert rms["command_samples"] == 0


def test_rms_needs_two_samples():
    from gazebo_poc.run_flight import _attitude_rms_deg

    assert _attitude_rms_deg([])["rms_deg"] is None
    assert _attitude_rms_deg([(0.1, 0.2)])["rms_deg"] is None


def test_sampler_dedupes_by_time():
    from types import SimpleNamespace

    from gazebo_poc.run_flight import _AttitudeSampler

    att = SimpleNamespace(roll=0.01, pitch=0.02, time_boot_ms=1000)
    m = SimpleNamespace(messages={"ATTITUDE": att})
    sampler = _AttitudeSampler()
    sampler.sample(m)
    sampler.sample(m)
    assert len(sampler.samples) == 1

    m.messages["ATTITUDE"] = SimpleNamespace(roll=0.03, pitch=0.04,
                                             time_boot_ms=1100)
    sampler.sample(m)
    assert len(sampler.samples) == 2
    # samples are timestamped so attitude RMS can be restricted to the window
    # whose speed was certified steady
    stamp, roll, pitch, target_roll, target_pitch = sampler.samples[-1]
    assert (roll, pitch) == (0.03, 0.04)
    assert target_roll is None and target_pitch is None
    assert stamp > 0
    assert sampler.between(stamp, stamp) == [sampler.samples[-1]]

    sampler.sample(SimpleNamespace(messages={}))
    assert len(sampler.samples) == 2
