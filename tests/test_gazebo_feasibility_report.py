from __future__ import annotations

import json

from examples import run_gazebo_feasibility as rgf
from src.prototyping.artifact_provenance import build_run_provenance


def test_planned_gazebo_requirements_are_extracted_by_requirement_text():
    planned = rgf._planned_gazebo_reqs([
        "REQ-SAFE-007: maintain controlled flight following one motor inoperative.",
        "REQ-PERF-005: maintain cruise speed in a 12 m/s headwind.",
        "REQ-FUNC-002: detect obstacles and initiate collision avoidance.",
        "REQ-PERF-002: sustain flight for a minimum of 20 minutes.",
    ])

    by_id = {item["req_id"]: item["check"] for item in planned}
    assert by_id == {
        "REQ-SAFE-007": "single_motor_out",
        "REQ-PERF-005": "wind_condition",
        "REQ-FUNC-002": "obstacle_avoidance",
    }


def test_numeric_wind_and_actuation_targets_are_preserved():
    planned = rgf._planned_gazebo_reqs([
        "REQ-PERF-004: maintain a minimum forward ground speed of 2 m/s when "
        "operating in sustained headwinds of up to 15 m/s.",
        "REQ-PERF-005: payload release actuation shall complete within 2.0 seconds.",
    ])
    by_check = {item["check"]: item for item in planned}

    assert by_check["wind_condition"]["wind_mps"] == 15.0
    assert by_check["wind_condition"]["min_groundspeed_mps"] == 2.0
    assert by_check["timed_actuation"]["max_delay_s"] == 2.0


def test_build_report_reuses_unchanged_pass_without_launching_gazebo(
    tmp_path, monkeypatch
):
    model = """package D {
        requirement def REQ_SAFE_007 {
            doc /* Maintain controlled flight with one motor inoperative. */
        }
        part def Drone;
    }"""
    design = {
        "battery_capacity_mah": 10000,
        "battery_cells": 6,
        "rotor_count": 6,
        "rotor_radius_m": 0.2,
        "payload_mass_kg": 1.5,
        "cruise_speed_mps": 18.0,
    }
    realization = {"chosen": {"combo": "M", "pack": "B", "frame": "F"}}

    def run(run_id):
        value = {
            "requirements": ["REQ-SAFE-007: one motor inoperative"],
            "recommended_design_inputs": design,
            "realization": realization,
            "requirement_impact": {
                "invalidated_requirement_ids": ["REQ_OTHER_001"]
            },
        }
        value["artifact_provenance"] = build_run_provenance(
            model_sysml=model,
            recommended_design=design,
            realization=realization,
            requirements=value["requirements"],
            parm_text=None,
            run_id=run_id,
        )
        return value

    previous = tmp_path / "previous"
    current = tmp_path / "current"
    output = tmp_path / "output"
    previous.mkdir()
    current.mkdir()
    output.mkdir()
    previous_run = run("previous-run")
    current_run = run("current-run")
    (previous / "realization_run.json").write_text(
        json.dumps(previous_run), encoding="utf-8"
    )
    (previous / "final_model.sysml").write_text(model, encoding="utf-8")
    (previous / "gazebo_feasibility_report.json").write_text(json.dumps({
        "status": "PASS",
        "source_provenance": previous_run["artifact_provenance"],
        "req_results": [{
            "req_id": "REQ-SAFE-007",
            "check": "single_motor_out",
            "status": "PASS",
            "message": "stable",
        }],
        "gazebo_result": {"hover_stable": True},
    }), encoding="utf-8")
    (current / "realization_run.json").write_text(
        json.dumps(current_run), encoding="utf-8"
    )
    (current / "final_model.sysml").write_text(model, encoding="utf-8")

    monkeypatch.setattr(rgf, "INPUT", current)
    monkeypatch.setattr(rgf, "SYSML_PATH", current / "final_model.sysml")
    monkeypatch.setattr(rgf, "RUN_JSON", current / "realization_run.json")
    monkeypatch.setattr(rgf, "REPORT_JSON", output / "report.json")
    monkeypatch.setattr(rgf, "REPORT_MD", output / "report.md")
    monkeypatch.setattr(rgf, "latest_output_dir", lambda: previous)
    monkeypatch.setattr(
        rgf,
        "_run_live_gazebo",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Gazebo must not launch")
        ),
    )

    report = rgf.build_report(dry_run=False, include_single_motor_out=True)

    assert report["reused_from_run_id"] == "previous-run"
    assert report["req_results"][0]["evidence_reused"] is True
    assert report["source_provenance"] == current_run["artifact_provenance"]
    assert report["evidence_origin_provenance"] == previous_run["artifact_provenance"]




def test_req_results_add_evidence_only_for_the_implemented_gazebo_check():
    planned = [
        {"req_id": "REQ-SAFE-007", "check": "single_motor_out", "message": "needs dynamics"},
        {"req_id": "REQ-FUNC-002", "check": "obstacle_avoidance", "message": "needs contact physics"},
    ]
    runs = [{"return_code": 0, "stable": True, "attitude_rms_deg": 0.31,
             "hover_alt_m": 10.0, "hover_throttle_pct": 50.0}]
    live = {"motor_failure_req": "REQ-SAFE-007", "motor_failure_tolerant": True,
            "motor_failure_runs": runs, "motor_failure_attempts": 1,
            "motor_failure_attitude_rms_deg": 0.31,
            "motor_failure_attitude_limit_deg": 5.0}

    results = rgf._req_results(live, planned)

    by_id = {item["req_id"]: item for item in results}
    assert by_id["REQ-SAFE-007"]["status"] == "PARTIAL"
    assert by_id["REQ-SAFE-007"]["criterion_evaluation"] == "PASS"
    assert by_id["REQ-FUNC-002"]["status"] == "PLANNED"


def test_req_results_suspend_motor_out_and_keep_subchecks_partial():
    planned = [
        {"req_id": "REQ-SAFE-007", "check": "single_motor_out", "message": "redundancy"},
        {"req_id": "REQ-PERF-004", "check": "wind_condition", "message": "wind",
         "wind_mps": 15.0, "min_groundspeed_mps": 2.0},
        {"req_id": "REQ-PERF-005", "check": "timed_actuation", "message": "release",
         "max_delay_s": 2.0, "requirement_text": "payload release within 2 seconds"},
    ]
    live = {
        "wind_test_complete": True,
        "wind_groundspeed_mps": 5.5,
        "wind_headwind_alignment": 0.99,
        "wind_groundspeed_steady_state": {
            "steady": True, "reason": "plateaued", "drift_fraction": 0.03,
            "duration_s": 9.9, "samples": 97,
        },
        "wind_drag_area_m2": 0.057437,
        "payload_release_commanded": True,
        "payload_observer_available": True,
        "payload_release_detected": True,
        "payload_release_delay_s": 0.42,
    }

    results = rgf._req_results(live, planned)
    by_id = {item["req_id"]: item for item in results}
    assert by_id["REQ-SAFE-007"]["status"] == "SUSPENDED"
    # the headwind now acts through geometry-derived quadratic drag on airspeed
    # and the ground speed held is certified steady, so this closes
    assert by_id["REQ-PERF-004"]["status"] == "PASS"
    assert "the verdict survives unless the true f exceeds" in by_id["REQ-PERF-004"]["message"]
    assert by_id["REQ-PERF-005"]["status"] == "PARTIAL"
    assert rgf._overall_status({"return_code": 0, "hover_stable": True}, results, False) == "PARTIAL"


def test_missing_payload_observer_is_inconclusive_not_false_failure():
    planned = [{
        "req_id": "REQ-PERF-005",
        "check": "timed_actuation",
        "message": "release",
        "max_delay_s": 2.0,
        "requirement_text": "payload release within 2 seconds",
    }]
    live = {
        "payload_release_commanded": True,
        "payload_observer_available": False,
        "payload_release_detected": False,
        "payload_release_delay_s": None,
    }

    result = rgf._req_results(live, planned)[0]
    assert result["status"] == "INCONCLUSIVE"


def test_payload_timing_never_uses_parachute_requirement_when_order_reverses():
    planned = [
        {"req_id": "REQ-SAFE-005", "check": "timed_actuation",
         "requirement_text": "deploy ballistic recovery parachute within 0.5 seconds"},
        {"req_id": "REQ-PERF-005", "check": "timed_actuation",
         "requirement_text": "mechanical payload release within 2.0 seconds"},
    ]

    assert rgf._payload_timing_req(planned)["req_id"] == "REQ-PERF-005"
    assert rgf._parachute_timing_req(planned)["req_id"] == "REQ-SAFE-005"


def test_position_and_parachute_subchecks_remain_partial_not_full_green():
    planned = [
        {"req_id": "REQ-FUNC-005", "check": "positional_release",
         "requirement_text": "release payload within 1 metre", "max_error_m": 1.0},
        {"req_id": "REQ-SAFE-005", "check": "timed_actuation",
         "requirement_text": "deploy ballistic recovery parachute within 0.5 seconds",
         "max_delay_s": 0.5},
    ]
    live = {
        "payload_release_commanded": True,
        "payload_release_detected": True,
        "payload_release_position_error_m": 0.42,
        "payload_release_position_met": True,
        "payload_release_position_basis": "payload_ground_truth_at_separation",
        "trigger_truth_error_m": 0.42,
        "separation_truth_error_m": 3.1,
        "delivery_abort_inactive": True,
        "parachute_commanded": True,
        "parachute_observer_available": True,
        "parachute_model_observed": True,
        "parachute_deploy_delay_s": 0.31,
    }

    by_id = {r["req_id"]: r for r in rgf._req_results(live, planned)}
    assert by_id["REQ-FUNC-005"]["status"] == "PARTIAL"
    assert by_id["REQ-SAFE-005"]["status"] == "PARTIAL"


def test_obstacle_chain_can_pass_only_with_real_lidar_and_avoidance_response():
    planned = [{
        "req_id": "REQ-FUNC-002",
        "check": "obstacle_avoidance",
        "requirement_text": "detect obstacles within 15m and avoid before 5m",
    }]
    live = {
        "obstacle_req": "REQ-FUNC-002",
        "obstacle_lidar_available": True,
        "obstacle_detected_within_15m": True,
        "obstacle_min_distance_m": 5.2,
        "obstacle_final_groundspeed_mps": 0.4,
        "obstacle_response_observed": True,
        "obstacle_avoidance_met": True,
    }

    result = rgf._req_results(live, planned)[0]
    assert result["status"] == "PASS"
    assert "DISTANCE_SENSOR" in result["message"]

    live["obstacle_avoidance_met"] = False
    live["obstacle_min_distance_m"] = 3.0
    assert rgf._req_results(live, planned)[0]["status"] == "FAIL"


def test_gazebo_design_uses_realized_phase8_values_when_available():
    run = {
        "recommended_design_inputs": {
            "payload_mass_kg": 1.5,
            "battery_capacity_mah": 16015.0,
            "battery_cells": 4,
            "rotor_count": 4,
            "rotor_radius_m": 0.15,
            "cruise_speed_mps": 0.0,
        },
        "realization": {
            "chosen": {
                "combo": "T-Motor MN3508 KV380 + P15x5 (6S)",
                "total_mass_kg": 4.2,
                "hover_throttle": 0.64,
                "design_drift": [
                    {"name": "rotor_radius_m", "realized": 0.1905},
                    {"name": "battery_capacity_mah", "realized": 16000.0},
                    {"name": "battery_cells", "realized": 6.0},
                ],
            }
        },
    }

    design = rgf._gazebo_design_from_run(run)

    assert design["mass_kg"] == 4.2
    assert design["payload_mass_kg"] == 1.5
    assert design["rotor_radius_m"] == 0.1905
    assert design["battery_capacity_mah"] == 16000.0
    assert design["battery_cells"] == 6.0
    assert design["max_thrust_g"] == 1880.0
    assert design["hover_throttle"] == 0.64


def test_overall_status_distinguishes_uncalibrated_from_dynamics_infeasible():
    assert rgf._overall_status(
        {"return_code": 8, "hover_stable": False},
        [],
        dry_run=False,
    ) == "GAZEBO_MODEL_UNCALIBRATED"
    assert rgf._overall_status(
        {"return_code": 7, "hover_stable": False},
        [],
        dry_run=False,
    ) == "DYNAMICS_INFEASIBLE"
    assert rgf._overall_status(
        {"return_code": 0, "hover_stable": True},
        [{"status": "PLANNED"}],
        dry_run=False,
    ) == "PARTIAL"


def test_flight_quality_checks_route_with_numeric_limits():
    planned = rgf._planned_gazebo_reqs([
        "REQ-PERF-001: The system shall maintain roll and pitch attitude "
        "deviations within 0.5 degree RMS during steady cruise flight at all "
        "authorised speeds. [V: Gazebo attitude logging / HIL]",
        "REQ-PERF-003: The system shall achieve a cruise airspeed of at least "
        "18 m/s in nil-wind, level-flight conditions.",
        "REQ-FUNC-003: The system shall transport payloads with a gross mass "
        "of up to 1.5 kg while maintaining a hover throttle margin of at "
        "least 30 percent and roll and pitch RMS within 1.0 degree.",
    ])
    by_check = {item["check"]: item for item in planned}
    assert by_check["cruise_attitude"]["req_id"] == "REQ-PERF-001"
    assert by_check["cruise_attitude"]["max_rms_deg"] == 0.5
    assert by_check["cruise_speed"]["req_id"] == "REQ-PERF-003"
    assert by_check["cruise_speed"]["min_speed_mps"] == 18.0
    assert by_check["payload_attitude"]["req_id"] == "REQ-FUNC-003"
    assert by_check["payload_attitude"]["max_rms_deg"] == 1.0
    assert by_check["payload_attitude"]["min_margin_pct"] == 30.0


_QUALITY_REQS = [
    "REQ-PERF-001: The system shall maintain roll and pitch attitude "
    "deviations within 0.5 degree RMS during steady cruise flight at all "
    "authorised speeds.",
    "REQ-PERF-003: The system shall achieve a cruise airspeed of at least "
    "18 m/s in nil-wind, level-flight conditions.",
    "REQ-FUNC-003: The system shall transport payloads with a gross mass of "
    "up to 1.5 kg while maintaining a hover throttle margin of at least 30 "
    "percent and roll and pitch RMS within 1.0 degree.",
]

_STEADY = {"steady": True, "reason": "plateaued", "drift_fraction": 0.021,
           "duration_s": 10.0, "samples": 99}
_TRENDING = {"steady": False, "reason": "still trending", "drift_fraction": 0.48,
             "duration_s": 9.0, "samples": 89}

_QUALITY_LIVE = {
    "nilwind_dash_speed_mps": 21.3,
    "nilwind_dash_peak_mps": 23.9,
    "nilwind_dash_pitch_rc": 1100,
    "nilwind_dash_steady_state": _STEADY,
    "cruise_sweep_speeds_mps": [10.1, 15.6, 19.9, 21.3],
    "cruise_sweep_payload_attached": True,
    "transport_windows": [
        {"label": "hover", "attitude_rms_deg": 0.44,
         "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
        {"label": "cruise@rc1420", "attitude_rms_deg": 0.21, "speed_mps": 10.1,
         "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
        {"label": "cruise@rc1330", "attitude_rms_deg": 0.31, "speed_mps": 15.6,
         "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
        {"label": "cruise@rc1220", "attitude_rms_deg": 0.28, "speed_mps": 19.9,
         "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
    ],
    "cruise_payload_attached": True,
    "cruise_payload_attachment_distance_m": 0.8,
    "cruise_attitude_rms_deg": 0.31,
    "cruise_attitude_roll_rms_deg": 0.22,
    "cruise_attitude_pitch_rms_deg": 0.31,
    "cruise_attitude_samples": 88,
    "cruise_attitude_mean_speed_mps": 19.7,
    "cruise_attitude_points": 4,
    "cruise_attitude_speed_span_mps": [10.1, 21.3],
    "hover_attitude_rms_deg": 0.44,
    "hover_attitude_samples": 132,
    "hover_attitude_with_payload": True,
    "hover_payload_attached": True,
    "hover_payload_attachment_distance_m": 0.8,
    "hover_throttle_pct": 38.0,
    "payload_mass_kg": 1.5,
}


def test_flight_quality_checks_judge_pass_partial_and_scope_caveats():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    results = {r["check"]: r for r in rgf._req_results(_QUALITY_LIVE, planned)}

    assert results["cruise_speed"]["status"] == "PASS"
    assert "HELD" in results["cruise_speed"]["message"]
    assert "certified steady points" in results["cruise_speed"]["message"]

    # a swept envelope may close "at all authorised speeds", but this run
    # reported no per-point RMS, so nothing shows the worst case sits at the
    # boundary rather than between two samples
    assert results["cruise_attitude"]["status"] == "PARTIAL"
    assert "WORST of 4 certified steady speed points" in results["cruise_attitude"]["message"]
    assert "authorised-speed envelope is not defined" in results["cruise_attitude"]["message"]
    assert "no per-point RMS was reported" in results["cruise_attitude"]["message"]

    # the sweep is flown with the payload aboard, so it is a transport cruise
    assert results["payload_attitude"]["status"] == "PASS"
    assert "margin" in results["payload_attitude"]["message"]
    assert ("window(s) observed carrying the payload"
            in results["payload_attitude"]["message"])


def test_a_speed_that_never_plateaued_cannot_be_reported():
    """The 2026-08-30 defect: a still-accelerating dash reported as a cruise speed."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, nilwind_dash_steady_state=_TRENDING)
    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    # 21.3 m/s clears the 18 m/s requirement, but it was never held
    assert results["cruise_speed"]["status"] == "INCONCLUSIVE"
    assert "still trending" in results["cruise_speed"]["message"]


def test_a_single_speed_point_still_cannot_claim_all_authorised_speeds():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, cruise_attitude_points=1,
                cruise_attitude_speed_span_mps=[19.7, 19.7],
                # one carrying window is not an envelope, however many the
                # sweep flew — transport is judged on what was observed loaded
                transport_windows=[
                    {"label": "hover", "attitude_rms_deg": 0.44,
                     "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
                ])
    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    assert results["cruise_attitude"]["status"] == "PARTIAL"
    assert "not swept" in results["cruise_attitude"]["message"]
    # and without a swept transport cruise the payload claim stays PARTIAL too
    assert results["payload_attitude"]["status"] == "PARTIAL"
    assert ("fewer than 3 windows observed carrying"
            in results["payload_attitude"]["message"])


#: A sweep whose variation across speed is tiny next to the headroom under the
#: bound — the shape of the real 2026-08-31 measurement.
_FLAT_SWEEP = {
    "cruise_attitude_swept_speeds_mps": [10.1, 15.6, 19.9, 21.3],
    "cruise_attitude_swept_rms_deg": [0.022, 0.026, 0.029, 0.031],
    "cruise_attitude_rms_deg": 0.031,
}


def test_a_flat_sweep_to_the_demonstrated_ceiling_closes_all_authorised_speeds():
    """No declared envelope, and no finite sweep can enumerate one. What can be
    shown is that nothing outside the sweep plausibly breaches the bound: the
    top of the sweep is the fastest speed the vehicle HELD, and the sweep's own
    variation across speed is tiny next to the headroom. Demanding more makes
    the verdict unreachable by construction."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    assert result["status"] == "PASS"
    assert "no authorised envelope is declared" in result["message"]
    assert "the fastest speed the vehicle HELD" in result["message"]
    assert "to breach the bound" in result["message"]


def test_a_demonstrated_envelope_still_fails_when_the_bound_is_exceeded():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    live["cruise_attitude_swept_rms_deg"] = [0.022, 0.026, 0.029, 0.91]
    live["cruise_attitude_rms_deg"] = 0.91
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    # a breach the vehicle actually flew is a FAIL, not a scope caveat
    assert result["status"] == "FAIL"


def test_a_small_dip_in_the_sweep_does_not_block_closure():
    """The real 2026-08-31 sweep dipped 0.0032 deg between two points against a
    0.5 deg limit. A monotonicity gate refused a 15x-margin result over that;
    the margin argument does not care about the shape, only the size."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    live["cruise_attitude_swept_rms_deg"] = [0.0252, 0.0220, 0.0304, 0.0321]
    live["cruise_attitude_rms_deg"] = 0.0321
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    assert result["status"] == "PASS"


def test_a_sweep_that_varies_as_much_as_its_headroom_cannot_close():
    """Wide variation next to a thin margin is exactly the case where an
    unsampled point could be the one that breaches."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    live["cruise_attitude_swept_rms_deg"] = [0.05, 0.14, 0.09, 0.19]
    live["cruise_attitude_rms_deg"] = 0.19
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    assert result["status"] == "PARTIAL"
    assert "could\n" not in result["message"]
    assert "plausibly breach the bound" in result["message"]


def test_a_worst_point_near_the_bound_closes_nothing_however_flat():
    """A perfectly flat sweep at 0.49 deg under a 0.5 deg limit has no room to
    extrapolate into, and flatness is not evidence that it does."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    live["cruise_attitude_swept_rms_deg"] = [0.49, 0.49, 0.49, 0.49]
    live["cruise_attitude_rms_deg"] = 0.49
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    assert result["status"] == "PARTIAL"
    assert "98% of the bound" in result["message"]


def test_a_sweep_short_of_the_held_ceiling_cannot_speak_for_the_envelope():
    """Stopping at 19.9 m/s when the vehicle held 21.3 leaves real authorised
    speeds above everything measured."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    live["cruise_attitude_swept_speeds_mps"] = [10.1, 15.6, 18.4, 19.9]
    live["cruise_attitude_speed_span_mps"] = [10.1, 19.9]
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    assert result["status"] == "PARTIAL"
    assert "stops short of the fastest speed the vehicle held" in result["message"]


def test_a_declared_envelope_still_outranks_the_demonstrated_one():
    """A model that names its authorised range is judged against that range,
    not against what the airframe happened to reach."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, **_FLAT_SWEEP)
    live["cruise_authorised_speed_range_mps"] = [10.0, 24.0]   # beyond the sweep
    live["cruise_authorised_speed_source"] = "generated_model"
    result = {r["check"]: r for r in rgf._req_results(live, planned)}["cruise_attitude"]

    assert result["status"] == "PARTIAL"
    assert "does not cover authorised range" in result["message"]


def test_cruise_attitude_closes_only_when_model_authority_range_is_covered():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(
        _QUALITY_LIVE,
        cruise_authorised_speed_range_mps=[11.0, 20.0],
        cruise_authorised_speed_source="generated_model",
    )

    result = {r["check"]: r for r in rgf._req_results(
        live, planned,
    )}["cruise_attitude"]

    assert result["status"] == "PASS"
    assert "generated_model" in result["message"]


def test_flight_quality_checks_fail_on_violated_limits():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE)
    live.update({
        "nilwind_dash_speed_mps": 12.4,     # held, but below 18
        "cruise_attitude_rms_deg": 0.9,     # above 0.5
        "hover_throttle_pct": 80.0,         # margin 20 < 30
        "cruise_authorised_speed_range_mps": [11.0, 20.0],
        "cruise_authorised_speed_source": "generated_model",
    })
    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    assert results["cruise_speed"]["status"] == "FAIL"
    assert results["cruise_attitude"]["status"] == "FAIL"
    assert results["payload_attitude"]["status"] == "FAIL"


def test_flight_quality_checks_stay_planned_without_measurements():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    results = {r["check"]: r for r in rgf._req_results({}, planned)}
    assert results["cruise_speed"]["status"] == "PLANNED"
    assert results["cruise_attitude"]["status"] == "PLANNED"
    assert results["payload_attitude"]["status"] == "PLANNED"


def test_payload_attitude_requires_payload_actually_attached():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE)
    live["hover_payload_attached"] = False
    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    # hover attitude measured without the payload must not judge FUNC-003
    assert results["payload_attitude"]["status"] == "PLANNED"


def test_payload_configuration_flags_cannot_substitute_for_attachment_truth():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE)
    live.pop("hover_payload_attached")
    live.pop("cruise_payload_attached")

    results = {r["check"]: r for r in rgf._req_results(live, planned)}

    assert live["hover_attitude_with_payload"] is True
    assert live["cruise_sweep_payload_attached"] is True
    assert results["payload_attitude"]["status"] == "PLANNED"


def test_coordinate_chain_delay_supersedes_command_delay():
    planned = rgf._planned_gazebo_reqs([
        "REQ-PERF-005: The mechanical payload release actuation shall complete "
        "within 2.0 seconds from the moment the delivery coordinate condition "
        "is satisfied.",
    ])
    live = {
        "payload_release_commanded": True,
        "payload_release_detected": True,
        "payload_observer_available": True,
        "payload_release_delay_s": 0.68,
        "payload_coordinate_to_separation_delay_s": 0.91,
    }
    result = rgf._req_results(live, planned)[0]
    assert result["check"] == "timed_actuation"
    assert result["status"] == "PARTIAL"
    assert "delivery-coordinate condition satisfied" in result["message"]
    assert "harness" in result["message"]

    live["payload_coordinate_to_separation_delay_s"] = 2.5  # over the 2 s limit
    result = rgf._req_results(live, planned)[0]
    assert result["status"] == "FAIL"

    del live["payload_coordinate_to_separation_delay_s"]
    result = rgf._req_results(live, planned)[0]
    assert result["status"] == "PARTIAL"
    assert "coordinate-condition detection was not exercised" in result["message"]


def test_obstacle_scenario_is_actually_flown_when_the_requirement_states_it(monkeypatch):
    """It stopped being flown once and nobody noticed for months: the planner
    kept recording the requirement while no scenario ever produced evidence."""
    calls = []

    from gazebo_poc import run_flight

    def fake_main(**kwargs):
        calls.append(kwargs)
        run_flight.LAST_RESULT.clear()
        run_flight.LAST_RESULT.update(
            {"hover_stable": True}
            if not kwargs.get("obstacle_avoidance")
            else {"obstacle_req": True, "obstacle_avoidance_met": True,
                  "obstacle_lidar_available": True,
                  "obstacle_min_distance_m": 5.44}
        )
        return 0

    monkeypatch.setattr(run_flight, "main", fake_main)
    planned = rgf._planned_gazebo_reqs([
        "REQ-FUNC-002: When approaching a stationary collision threat directly "
        "ahead within the forward sensor field of view at a closing speed no "
        "greater than 1.5 m/s, the system shall execute an avoidance manoeuvre "
        "following threat detection no later than 15 metres, maintaining an "
        "airframe-to-obstacle separation of at least 5 metres."
    ])
    design = {"mass_kg": 5.54, "rotor_radius_m": 0.2032, "rotor_count": 6,
              "battery_capacity_mah": 16000.0, "payload_mass_kg": 0.0}
    live = rgf._run_live_gazebo(design, planned)

    obstacle_calls = [c for c in calls if c.get("obstacle_avoidance")]
    assert len(obstacle_calls) == 1, "the obstacle scenario was never flown"
    # the envelope is the requirement's own, not an invented one
    assert obstacle_calls[0]["obstacle_detection_range_m"] == 15.0
    assert obstacle_calls[0]["obstacle_min_separation_m"] == 5.0
    assert obstacle_calls[0]["obstacle_approach_speed_mps"] == 1.5
    assert live["obstacle_req"] is True

    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    assert results["obstacle_avoidance"]["status"] == "PASS"


def test_obstacle_is_not_flown_when_the_requirement_omits_the_envelope(monkeypatch):
    calls = []

    from gazebo_poc import run_flight

    def fake_main(**kwargs):
        calls.append(kwargs)
        run_flight.LAST_RESULT.clear()
        run_flight.LAST_RESULT.update({"hover_stable": True})
        return 0

    monkeypatch.setattr(run_flight, "main", fake_main)
    planned = rgf._planned_gazebo_reqs([
        "REQ-FUNC-002: The system shall avoid obstacles."   # no numbers at all
    ])
    assert planned[0]["contract_ready"] is False
    assert planned[0]["semantic_gaps"]
    rgf._run_live_gazebo({"mass_kg": 5.54, "rotor_radius_m": 0.2032, "rotor_count": 6,
                      "battery_capacity_mah": 16000.0, "payload_mass_kg": 0.0}, planned)
    assert not [c for c in calls if c.get("obstacle_avoidance")], (
        "refused envelope must not be invented by flying a default scenario"
    )


def test_navigation_campaign_is_separate_from_the_primary_cruise_flight(monkeypatch):
    calls = []
    from gazebo_poc import run_flight

    def fake_main(**kwargs):
        calls.append(kwargs)
        run_flight.LAST_RESULT.clear()
        if kwargs.get("navigation_accuracy"):
            run_flight.LAST_RESULT.update({
                "cep_samples": 8,
                "cep_m": 0.4,
                "takeoff_command_accepted": True,
                "takeoff_method": "guided_nav_takeoff",
            })
        else:
            run_flight.LAST_RESULT.update({
                "hover_stable": True,
                "nilwind_dash_speed_mps": 19.3,
            })
        return 0

    monkeypatch.setattr(run_flight, "main", fake_main)
    planned = rgf._planned_gazebo_reqs([
        _NAV_REQ,
        "REQ-PERF-003: achieve cruise airspeed of at least 18 m/s in nil wind.",
    ])
    design = {
        "mass_kg": 5.54, "rotor_radius_m": 0.2032, "rotor_count": 6,
        "battery_capacity_mah": 16000.0, "payload_mass_kg": 0.0,
    }

    live = rgf._run_live_gazebo(design, planned)

    assert len(calls) == 2
    assert calls[0].get("navigation_accuracy") is None
    assert calls[1]["navigation_accuracy"] is True
    assert live["nilwind_dash_speed_mps"] == 19.3
    assert live["cep_samples"] == 8


_RELEASE_PLANNED = [
    {"req_id": "REQ-PERF-005", "check": "timed_actuation", "message": "release",
     "max_delay_s": 2.0, "requirement_text": "payload release within 2 seconds"},
    {"req_id": "REQ-FUNC-005", "check": "positional_release", "message": "release",
     "max_error_m": 1.0, "requirement_text": "release within 1.0 metre"},
]
_RELEASE_LIVE = {
    # measured on the authoritative design: the estimator was accurate, and the
    # payload still separated 3.8 m out because 1.8 s elapsed while the vehicle
    # kept flying
    "delivery_abort_inactive": True,
    "trigger_estimated_error_m": 0.978,
    "trigger_truth_error_m": 0.854,
    "separation_truth_error_m": 3.824,
    "trigger_to_separation_s": 1.798,
    "payload_release_commanded": True,
    "payload_observer_available": True,
    "payload_release_detected": True,
    "payload_release_delay_s": 0.69,
    "payload_coordinate_to_separation_delay_s": 1.33,
    "payload_release_position_error_m": 0.765,
    "payload_release_position_met": True,
    "payload_release_position_basis": "payload_ground_truth_at_separation",
}
_MODEL_DECISION = [{
    "owner_part": "PayloadMechanism", "machine": "PayloadReleaseBehavior",
    "event": "DeliveryCoordinateSatisfied", "from_state": "Locked",
    "to_state": "Releasing", "action": "onReleasing",
    "action_definition": "actuateRelease",
    "decided_by": "generated model",
}]


def test_a_harness_triggered_release_stays_partial():
    """Physics evidence about Gazebo is not evidence about the generated model."""
    results = {r["check"]: r for r in rgf._req_results(
        dict(_RELEASE_LIVE, payload_release_decided_by="harness"), _RELEASE_PLANNED)}
    assert results["timed_actuation"]["status"] == "PARTIAL"
    assert results["positional_release"]["status"] == "PARTIAL"
    assert "not generated mission-logic ownership" in results["positional_release"]["message"]


def test_the_estimators_own_figure_cannot_substitute_for_the_trigger_truth():
    """The clause closes on where the vehicle TRULY was at the trigger. A system
    scoring itself against its own estimate of that moment proves nothing, so
    an unobserved truth is INCONCLUSIVE even when the estimate looks good."""
    live = dict(_RELEASE_LIVE)
    live.pop("trigger_truth_error_m")

    result = {r["check"]: r for r in rgf._req_results(
        live, _RELEASE_PLANNED,
    )}["positional_release"]

    assert result["status"] == "INCONCLUSIVE"
    assert "true position at the trigger was not observed" in result["message"]
    assert "cannot stand in for it" in result["message"]


def test_an_observed_trigger_position_outside_tolerance_is_a_failure():
    """Over the bound with a valid measurement is a FAIL, never INCONCLUSIVE."""
    live = dict(
        _RELEASE_LIVE,
        trigger_truth_error_m=1.4,
        payload_release_decided_by="generated model",
        payload_release_decisions=_MODEL_DECISION,
    )

    result = {r["check"]: r for r in rgf._req_results(
        live, _RELEASE_PLANNED,
    )}["positional_release"]

    assert result["status"] == "FAIL"


def test_a_model_owned_release_closes_and_names_the_machine():
    live = dict(_RELEASE_LIVE,
                payload_release_decided_by="generated model",
                payload_release_decisions=_MODEL_DECISION)
    results = {r["check"]: r for r in rgf._req_results(live, _RELEASE_PLANNED)}

    assert results["timed_actuation"]["status"] == "PASS"
    assert results["positional_release"]["status"] == "PASS"
    for check in ("timed_actuation", "positional_release"):
        message = results[check]["message"]
        assert "the GENERATED model owned the decision" in message
        # the evidence names what actually fired, not just that something did
        assert "PayloadMechanism.PayloadReleaseBehavior Locked->Releasing" in message
        assert "firing onReleasing" in message


def test_an_unrelated_model_action_cannot_close_release_requirements():
    wrong_decision = [{
        **_MODEL_DECISION[0],
        "action_definition": "lockPayload",
    }]
    live = dict(
        _RELEASE_LIVE,
        payload_release_decided_by="generated model",
        payload_release_decisions=wrong_decision,
    )

    results = {r["check"]: r for r in rgf._req_results(live, _RELEASE_PLANNED)}

    assert results["timed_actuation"]["status"] == "PARTIAL"
    assert results["positional_release"]["status"] == "PARTIAL"


def test_parachute_ownership_does_not_silently_claim_precedence():
    planned = [{"req_id": "REQ-SAFE-005", "check": "timed_actuation",
                "message": "parachute", "max_delay_s": 0.5,
                "requirement_text": "deploy the ballistic recovery parachute within 0.5 seconds"}]
    live = {
        "parachute_commanded": True,
        "parachute_observer_available": True,
        "parachute_model_observed": True,
        "parachute_deploy_delay_s": 0.0516,
        "parachute_max_delay_s": 0.5,
        "parachute_decided_by": "generated model",
        "parachute_decisions": [{
            "owner_part": "SafetyMonitor", "machine": "ParachuteDeploymentBehavior",
            "event": "CriticalPropulsionFailure", "from_state": "Monitoring",
            "to_state": "DeployingParachute", "action": "onDeployingParachute",
            "action_definition": "deployParachute",
            "decided_by": "generated model",
        }],
        "parachute_precedence_status": "verified",
        "parachute_precedence_description": (
            "winner deployParachute fired=True; no competing response fired"
        ),
        "parachute_precedence_competing_actions": [
            "initiateArmingInhibit", "initiateEmergencyLand", "initiateBatteryRtb",
        ],
        "parachute_precedence_competing_actions_fired": [],
    }
    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    message = results["parachute_deploy_timing"]["message"]
    assert results["parachute_deploy_timing"]["status"] == "PASS"
    assert "the GENERATED model owned the decision" in message
    # SAFE-005 also asks for precedence over all other safety responses, which
    # deploying on cue does not demonstrate
    assert "precedence over other safety responses is a separate" in message
    assert results["safety_precedence"]["status"] == "PASS"
    assert "simultaneously active" in results["safety_precedence"]["message"]


_INHIBITION_REQ = (
    "REQ-SAFE-006: The system shall maintain the payload in the mechanically "
    "locked state whenever a delivery-abort condition is active, regardless of "
    "geographic proximity to the delivery waypoint."
)


def test_an_inhibition_requirement_is_planned_as_its_own_driven_scenario():
    planned = rgf._planned_gazebo_reqs([_INHIBITION_REQ])
    assert planned[0]["check"] == "delivery_abort_inhibition"
    assert "physically observed, not inferred" in planned[0]["message"]


def test_releasing_while_the_abort_is_active_fails_the_inhibition():
    """The 2026-08-30 model does exactly this, and REQ-SAFE-006 was marked
    verified: two unguarded machines, no arbitration between them."""
    planned = rgf._planned_gazebo_reqs([_INHIBITION_REQ])
    live = {
        "abort_inhibition_req": "REQ-SAFE-006",
        "abort_inhibition_abort_active": True,
        "abort_inhibition_observer_available": True,
        "abort_inhibition_release_detected": True,
        "abort_inhibition_release_guards": [],
        "abort_inhibition_z_before_m": 8.799,
        "abort_inhibition_z_after_m": 8.291,
        "abort_inhibition_decisions": [{
            "machine": "PayloadReleaseBehavior", "from_state": "Locked",
            "to_state": "Releasing", "action": "onReleasing",
        }],
    }
    result = {r["check"]: r for r in rgf._req_results(live, planned)}
    row = result["delivery_abort_inhibition"]
    assert row["status"] == "FAIL"
    assert "SEPARATED anyway" in row["message"]
    assert "carries no guard" in row["message"]


def test_a_payload_that_stays_attached_passes_the_inhibition():
    planned = rgf._planned_gazebo_reqs([_INHIBITION_REQ])
    live = {
        "abort_inhibition_req": "REQ-SAFE-006",
        "abort_inhibition_abort_active": True,
        "abort_inhibition_observer_available": True,
        "abort_inhibition_release_detected": False,
        "abort_inhibition_release_guards": [],
        "abort_inhibition_decisions": [],
    }
    result = {r["check"]: r for r in rgf._req_results(live, planned)}
    assert result["delivery_abort_inhibition"]["status"] == "PASS"
    assert "remained attached" in result["delivery_abort_inhibition"]["message"]


def test_no_payload_observer_makes_the_inhibition_inconclusive_not_a_pass():
    """Without an observer, 'no separation seen' is not evidence of inhibition."""
    planned = rgf._planned_gazebo_reqs([_INHIBITION_REQ])
    live = {
        "abort_inhibition_req": "REQ-SAFE-006",
        "abort_inhibition_abort_active": True,
        "abort_inhibition_observer_available": False,
        "abort_inhibition_release_detected": False,
    }
    result = {r["check"]: r for r in rgf._req_results(live, planned)}
    assert result["delivery_abort_inhibition"]["status"] == "INCONCLUSIVE"


_NAV_REQ = ("REQ-FUNC-001: The system shall autonomously navigate to designated "
            "GPS waypoints with a circular error probable (CEP) of less than 1.0 metre.")


def test_a_stick_flown_rig_reports_no_cep_and_says_why():
    """A blank row says nothing; a stated, checkable limitation says a lot."""
    planned = rgf._planned_gazebo_reqs([_NAV_REQ])
    assert planned[0]["max_cep_m"] == 1.0

    row = rgf._req_results({
        "takeoff_command_accepted": False,
        "takeoff_command_result": 4,          # MAV_RESULT_FAILED
        "takeoff_method": "guided_nav_takeoff_then_alt_hold_fallback",
    }, planned)[0]

    assert row["status"] == "INCONCLUSIVE"
    assert "REJECTED MAV_CMD_NAV_TAKEOFF" in row["message"]
    assert "flown by RC stick in ALT_HOLD" in row["message"]


def test_cep_is_only_reported_from_autonomous_flight():
    planned = rgf._planned_gazebo_reqs([_NAV_REQ])

    # a CEP measured while the rig was stick-flown must not be reported
    stick = rgf._req_results({
        "takeoff_command_accepted": False,
        "takeoff_method": "guided_nav_takeoff_then_alt_hold_fallback",
        "cep_m": 0.4,
    }, planned)[0]
    assert stick["status"] == "INCONCLUSIVE"

    autonomous = rgf._req_results({
        "takeoff_command_accepted": True,
        "takeoff_method": "guided_nav_takeoff",
        "cep_m": 0.4, "cep_samples": 8,
        "cep_horizontal_error_source": "sim_gps_glitch_xy_campaign",
        "cep_horizontal_error_injected_m": 2.0,
        "cep_raw_gnss_truth_error_m": 1.8,
    }, planned)[0]
    assert autonomous["status"] == "PASS"
    assert "CEP 0.4 m over 8 commanded global waypoints" in autonomous["message"]

    # the same flight on a noise-free GNSS is a floor, not a CEP
    idealised = rgf._req_results({
        "takeoff_command_accepted": True,
        "takeoff_method": "guided_nav_takeoff",
        "cep_m": 0.4, "cep_samples": 8, "cep_raw_gnss_scatter_m": 0.02,
    }, planned)[0]
    assert idealised["status"] == "INCONCLUSIVE"


def test_one_motor_out_cannot_pass_on_altitude_alone():
    """Repeated runs of the same configuration held 1.17 m, 9.93 m and 6.27 m,
    while attitude RMS stayed 13-20 deg in every one. A verdict resting on the
    signal that does not reproduce is a coin flip on a safety requirement."""
    planned = [{"req_id": "REQ-SAFE-007", "check": "single_motor_out",
                "message": "redundancy"}]

    unmeasured = rgf._req_results(
        {"motor_failure_req": "REQ-SAFE-007", "motor_failure_tolerant": True},
        planned, include_single_motor_out=True)[0]
    assert unmeasured["status"] == "INCONCLUSIVE"
    assert "attitude was not measured" in unmeasured["message"]

    wobbling_run = {"return_code": 0, "stable": False,
                    "attitude_rms_deg": 13.46, "hover_alt_m": 9.93,
                    "hover_throttle_pct": 35.0}
    wobbling = rgf._req_results({
        "motor_failure_req": "REQ-SAFE-007",
        "motor_failure_tolerant": False,
        "motor_failure_runs": [wobbling_run],
        "motor_failure_attempts": 1,
        "motor_failure_attitude_rms_deg": 13.46,
        "motor_failure_attitude_limit_deg": 5.0,
        "motor_failure_hover_alt_m": 9.93,
        "motor_failure_hover_throttle_pct": 35.0,
    }, planned, include_single_motor_out=True)[0]
    assert wobbling["status"] == "PARTIAL"
    assert wobbling["criterion_evaluation"] == "FAIL"
    assert "unaccepted engineering interpretation" in wobbling["message"]
    assert "13.46 deg" in wobbling["message"]


def test_hover_stability_requires_attitude_tracking():
    from gazebo_poc.run_flight import _HOVER_ATTITUDE_RMS_LIMIT_DEG

    # an order of magnitude above the 0.5 deg RMS asked of steady cruise, and
    # far below tilt authority — the observed 13-20 deg is outside any choice
    # in that band, so the verdict does not hinge on the exact value
    assert 2.0 <= _HOVER_ATTITUDE_RMS_LIMIT_DEG <= 10.0


def test_a_cep_from_a_noise_free_gnss_is_not_a_navigation_cep():
    """The rig now flies waypoints autonomously and the error is tiny — which
    is exactly when a floor is easiest to mistake for a result. Raw GPS and the
    fused estimate agree to 0.02 m and an injected 2 m SIM_GPS1_NOISE changes
    nothing, so no GNSS error is represented at all."""
    planned = rgf._planned_gazebo_reqs([_NAV_REQ])
    row = rgf._req_results({
        "takeoff_command_accepted": True,
        "takeoff_method": "guided_nav_takeoff",
        "cep_m": 0.0079, "cep_samples": 8, "cep_max_error_m": 0.0131,
        "cep_raw_gnss_scatter_m": 0.022, "cep_gps_noise_m": 2.0,
    }, planned)[0]

    assert row["status"] == "INCONCLUSIVE"
    assert "NOT a navigation CEP" in row["message"]
    assert "the requirement stays open" in row["message"]
    # the measurement is still reported — it is a real number about the control loop
    assert row["cep_m"] == 0.0079
    assert "flew 8 commanded waypoints autonomously" in row["message"]


def test_cep_closes_once_gnss_error_is_actually_present():
    planned = rgf._planned_gazebo_reqs([_NAV_REQ])
    row = rgf._req_results({
        "takeoff_command_accepted": True,
        "takeoff_method": "guided_nav_takeoff",
        "cep_m": 0.42, "cep_samples": 8, "cep_max_error_m": 0.7,
        "cep_horizontal_error_source": "sim_gps_glitch_xy_campaign",
        "cep_horizontal_error_injected_m": 2.0,
        "cep_raw_gnss_truth_error_m": 1.8,
    }, planned)[0]
    assert row["status"] == "PASS"
    assert "GNSS error represented" in row["message"]

    # and a real CEP over the limit is a real failure, not an excuse
    over = rgf._req_results({
        "takeoff_command_accepted": True,
        "takeoff_method": "guided_nav_takeoff",
        "cep_m": 1.6, "cep_samples": 8, "cep_max_error_m": 2.4,
        "cep_horizontal_error_source": "sim_gps_glitch_xy_campaign",
        "cep_horizontal_error_injected_m": 2.0,
        "cep_raw_gnss_truth_error_m": 1.8,
    }, planned)[0]
    assert over["status"] == "FAIL"


def test_incomplete_horizontal_error_campaign_cannot_report_cep():
    planned = rgf._planned_gazebo_reqs([_NAV_REQ])
    row = rgf._req_results({
        "takeoff_command_accepted": True,
        "takeoff_method": "guided_nav_takeoff",
        "cep_m": 0.42, "cep_samples": 7, "cep_max_error_m": 0.7,
        "cep_horizontal_error_source": "sim_gps_glitch_xy_campaign",
        "cep_horizontal_error_injected_m": 2.0,
        "cep_raw_gnss_truth_error_m": 1.8,
    }, planned)[0]

    assert row["status"] == "INCONCLUSIVE"
    assert "7 of 8" in row["message"]


def _motor_out_live(runs, passes=None):
    measured = [r for r in runs if r.get("attitude_rms_deg") is not None]
    worst = max(measured, key=lambda r: r["attitude_rms_deg"]) if measured else {}
    stable = [r for r in runs if r.get("stable")]
    return {
        "motor_failure_req": "REQ-SAFE-007",
        "motor_failure_runs": runs,
        "motor_failure_attempts": len(runs),
        "motor_failure_passes": len(stable) if passes is None else passes,
        "motor_failure_tolerant": len(stable) == len(runs),
        "motor_failure_attitude_rms_deg": worst.get("attitude_rms_deg"),
        "motor_failure_attitude_limit_deg": 5.0,
        "motor_failure_hover_alt_m": worst.get("hover_alt_m"),
        "motor_failure_hover_throttle_pct": worst.get("hover_throttle_pct"),
    }


_MOTOR_OUT_PLANNED = [{"req_id": "REQ-SAFE-007", "check": "single_motor_out",
                       "message": "redundancy"}]


def test_a_redundancy_that_holds_only_sometimes_is_unguaranteed_not_incapable():
    """These are different claims and a safety report must not conflate them."""
    runs = [
        {"return_code": 0, "stable": True,  "attitude_rms_deg": 1.59,  "hover_alt_m": 10.00, "hover_throttle_pct": 52},
        {"return_code": 0, "stable": False, "attitude_rms_deg": 13.46, "hover_alt_m": 9.93,  "hover_throttle_pct": 35},
        {"return_code": 0, "stable": False, "attitude_rms_deg": 19.56, "hover_alt_m": 6.27,  "hover_throttle_pct": 96},
        {"return_code": 0, "stable": True,  "attitude_rms_deg": 2.10,  "hover_alt_m": 10.00, "hover_throttle_pct": 51},
        {"return_code": 0, "stable": False, "attitude_rms_deg": 15.00, "hover_alt_m": 1.17,  "hover_throttle_pct": 78},
    ]
    row = rgf._req_results(_motor_out_live(runs), _MOTOR_OUT_PLANNED,
                           include_single_motor_out=True)[0]
    assert row["status"] == "PARTIAL"
    assert row["criterion_evaluation"] == "FAIL"
    assert row["criterion"]["source"] == "engineering_judgement"
    assert row["criterion"]["accepted_for_requirement"] is False
    assert row["sensitivity"] == [
        {"interpretation": "attitude RMS <= 5 deg", "passed_runs": 2, "total_runs": 5},
        {"interpretation": "flight completed without scenario termination", "passed_runs": 5, "total_runs": 5},
    ]
    assert "engineering interpretation" in row["message"]
    assert "2 of 5 one-motor-out flights" in row["message"]
    # every flight's numbers are in the evidence, not just the worst
    for value in ("1.59", "13.46", "19.56", "2.10", "15.00"):
        assert value in row["message"]


def test_failing_engineering_interpretation_does_not_become_requirement_fail():
    runs = [{"return_code": 0, "stable": False, "attitude_rms_deg": 12.0 + i, "hover_alt_m": 2.0,
             "hover_throttle_pct": 90} for i in range(5)]
    row = rgf._req_results(_motor_out_live(runs), _MOTOR_OUT_PLANNED,
                           include_single_motor_out=True)[0]
    assert row["status"] == "PARTIAL"
    assert row["criterion_evaluation"] == "FAIL"
    assert row["sensitivity"][1] == {
        "interpretation": "flight completed without scenario termination",
        "passed_runs": 5,
        "total_runs": 5,
    }


def test_passing_engineering_interpretation_does_not_become_requirement_pass():
    runs = [{"return_code": 0, "stable": True, "attitude_rms_deg": 0.4, "hover_alt_m": 10.0,
             "hover_throttle_pct": 50} for _ in range(5)]
    row = rgf._req_results(_motor_out_live(runs), _MOTOR_OUT_PLANNED,
                           include_single_motor_out=True)[0]
    assert row["status"] == "PARTIAL"
    assert row["criterion_evaluation"] == "PASS"
    assert row["criterion"]["accepted_for_requirement"] is False

    # one bad flight out of five is enough to withhold the claim
    runs[2] = {"return_code": 0, "stable": False, "attitude_rms_deg": 14.0, "hover_alt_m": 3.0,
               "hover_throttle_pct": 88}
    degraded = rgf._req_results(_motor_out_live(runs), _MOTOR_OUT_PLANNED,
                                include_single_motor_out=True)[0]
    assert degraded["status"] == "PARTIAL"
    assert degraded["criterion_evaluation"] == "FAIL"


def test_the_repeat_count_is_declared_not_incidental():
    assert rgf._MOTOR_OUT_REPEATS >= 3


def test_a_cruise_window_flown_after_release_is_not_transport_evidence():
    """The defect this replaced: attachment was a flag set before takeoff, so a
    window measured after separation counted as transport evidence."""
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, transport_windows=[
        {"label": "hover", "attitude_rms_deg": 0.44,
         "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
        {"label": "cruise@rc1420", "attitude_rms_deg": 0.21, "speed_mps": 10.1,
         "attachment": {"observed": True, "distance_m": 0.15, "vertical_separation_m": 0.15}},
        # released: the box is on the ground, the vehicle is at altitude
        {"label": "cruise@rc1330", "attitude_rms_deg": 0.31, "speed_mps": 15.6,
         "attachment": {"observed": True, "distance_m": 880.0, "vertical_separation_m": 9.2}},
        {"label": "cruise@rc1220", "attitude_rms_deg": 0.28, "speed_mps": 19.9,
         "attachment": {"observed": True, "distance_m": 1250.0, "vertical_separation_m": 9.2}},
    ])

    row = {r["check"]: r for r in rgf._req_results(live, planned)}["payload_attitude"]

    assert row["status"] == "PARTIAL"
    assert "2 window(s) excluded as observed unloaded" in row["message"]
    assert "cruise@rc1330" in row["message"] and "cruise@rc1220" in row["message"]
    assert "fewer than 3 windows observed carrying" in row["message"]


def test_a_run_that_never_looked_cannot_claim_transport():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE, transport_windows=[
        {"label": "hover", "attitude_rms_deg": 0.44,
         "attachment": {"observed": False, "distance_m": None, "vertical_separation_m": None}},
    ])

    row = {r["check"]: r for r in rgf._req_results(live, planned)}["payload_attitude"]

    assert row["status"] == "PARTIAL"
    assert "attachment was never observed" in row["message"]


def test_a_release_during_an_abort_cannot_close_the_position_clause():
    """REQ-FUNC-005 is a conjunction — within 1.0 m AND no abort active. Only
    the distance was ever checked, so a release that happened while an abort
    was active would have closed it."""
    live = dict(_RELEASE_LIVE,
                payload_release_decided_by="generated model",
                payload_release_decisions=_MODEL_DECISION,
                delivery_abort_inactive=False)

    row = {r["check"]: r for r in rgf._req_results(live, _RELEASE_PLANNED)}["positional_release"]

    assert row["status"] == "PARTIAL"
    assert "second conjunct is unverified" in row["message"]


def test_an_unstated_abort_state_is_not_proof_that_none_was_active():
    live = dict(_RELEASE_LIVE,
                payload_release_decided_by="generated model",
                payload_release_decisions=_MODEL_DECISION)
    live.pop("delivery_abort_inactive")

    row = {r["check"]: r for r in rgf._req_results(live, _RELEASE_PLANNED)}["positional_release"]

    assert row["status"] == "PARTIAL"
    assert "NOT shown inactive" in row["message"]


def test_the_clause_closes_on_trigger_truth_and_reports_the_separation_miss():
    """They differ, and which one closed the clause has to be visible."""
    live = dict(_RELEASE_LIVE,
                payload_release_decided_by="generated model",
                payload_release_decisions=_MODEL_DECISION)

    row = {r["check"]: r for r in rgf._req_results(live, _RELEASE_PLANNED)}["positional_release"]

    assert row["status"] == "PASS"
    # the clause closes on where the vehicle TRULY was at the trigger
    assert row["trigger_truth_error_m"] == 0.854
    # the estimator's own figure is reported but never decides
    assert row["trigger_estimated_error_m"] == 0.978
    # and the separation miss is mandatory in the evidence even though the
    # requirement does not bound it
    assert row["separation_truth_error_m"] == 3.824
    assert "truly 0.854 m from the designated waypoint" in row["message"]
    assert "rather than the estimator's own figure" in row["message"]
    assert "separated 3.824 m from the waypoint" in row["message"]
    assert "approach speed times the actuation delay" in row["message"]


_ABORT_REQS = [
    "REQ-SAFE-006: The system shall maintain the payload in the mechanically "
    "locked state whenever a delivery-abort condition is active, regardless of "
    "geographic proximity to the delivery waypoint.",
]

_ABORT_LIVE = {
    "abort_inhibition_req": "REQ-SAFE-006",
    "abort_inhibition_observer_available": True,
    "abort_inhibition_abort_active": True,
    "abort_inhibition_release_detected": True,
    "abort_inhibition_z_before_m": 10.0953,
    "abort_inhibition_z_after_m": 9.57917,
}


def _abort(**overrides):
    planned = rgf._planned_gazebo_reqs(_ABORT_REQS)
    # Guards MEASURED as absent by default; tri-state contract reserves a
    # missing/None value for "the release identity never resolved".
    live = {**_ABORT_LIVE, "abort_inhibition_release_guards": [], **overrides}
    return {r["check"]: r for r in rgf._req_results(live, planned)}[
        "delivery_abort_inhibition"]


def test_an_unguarded_release_that_separates_during_an_abort_is_a_failure():
    """The measured defect, reproduced at three tiers."""
    result = _abort()
    assert result["status"] == "FAIL"
    assert "carries no guard" in result["message"]


def test_a_guard_the_run_never_fed_cannot_convict_the_model():
    """An unset boolean reads FALSE, so a correctly guarded model releases
    exactly like an unguarded one. Calling that a model defect would blame the
    model for a condition the harness never told it about — and would make the
    guard fix look applied while the verdict stayed red."""
    result = _abort(
        abort_inhibition_release_guards=["deliveryAbortActive == false"],
        abort_inhibition_flags_unbound=["deliveryAbortActive"],
        abort_inhibition_flags_raised={},
    )
    assert result["status"] == "INCONCLUSIVE"
    assert "says nothing about the model" in result["message"]


def test_a_guard_that_was_fed_and_fired_anyway_is_a_model_defect():
    result = _abort(
        abort_inhibition_release_guards=["deliveryAbortActive == false"],
        abort_inhibition_flags_unbound=[],
        abort_inhibition_flags_raised={"deliveryAbortActive": True},
    )
    assert result["status"] == "FAIL"
    assert "fired regardless" in result["message"]


def test_a_fed_guard_that_held_the_payload_closes_the_requirement():
    result = _abort(
        abort_inhibition_release_detected=False,
        abort_inhibition_release_guards=["deliveryAbortActive == false"],
        abort_inhibition_flags_unbound=[],
        abort_inhibition_flags_raised={"deliveryAbortActive": True},
    )
    assert result["status"] == "PASS"
    assert "the inhibition holds" in result["message"]


def test_a_payload_that_stayed_put_with_no_guard_is_not_credited_to_inhibition():
    """Nothing held it back; it merely did not go. A PASS here would credit the
    model with logic it does not contain."""
    result = _abort(abort_inhibition_release_detected=False)
    assert "not attributable to inhibition logic" in result["message"]


def test_an_unresolved_release_identity_is_not_measured_not_unguarded():
    """The run3 false verdict, pinned: when the harness could not resolve the
    release identity on this model, whether the path is guarded was never
    measured — INCONCLUSIVE and said so, never 'carries no guard'."""
    result = _abort(abort_inhibition_release_guards=None)
    assert result["status"] == "INCONCLUSIVE"
    assert "NOT MEASURED" in result["message"]
    assert "carries no guard" not in result["message"]
