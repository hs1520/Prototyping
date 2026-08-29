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




def test_req_results_upgrade_only_the_implemented_gazebo_check():
    planned = [
        {"req_id": "REQ-SAFE-007", "check": "single_motor_out", "message": "needs dynamics"},
        {"req_id": "REQ-FUNC-002", "check": "obstacle_avoidance", "message": "needs contact physics"},
    ]
    live = {"motor_failure_req": "REQ-SAFE-007", "motor_failure_tolerant": True}

    results = rgf._req_results(live, planned)

    by_id = {item["req_id"]: item for item in results}
    assert by_id["REQ-SAFE-007"]["status"] == "PASS"
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
        "payload_release_commanded": True,
        "payload_observer_available": True,
        "payload_release_detected": True,
        "payload_release_delay_s": 0.42,
    }

    results = rgf._req_results(live, planned)
    by_id = {item["req_id"]: item for item in results}
    assert by_id["REQ-SAFE-007"]["status"] == "SUSPENDED"
    assert by_id["REQ-PERF-004"]["status"] == "PARTIAL"
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
        "payload_release_position_error_m": 0.42,
        "payload_release_position_met": True,
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

_QUALITY_LIVE = {
    "nilwind_dash_speed_mps": 21.3,
    "nilwind_dash_peak_mps": 23.9,
    "cruise_attitude_rms_deg": 0.31,
    "cruise_attitude_roll_rms_deg": 0.22,
    "cruise_attitude_pitch_rms_deg": 0.31,
    "cruise_attitude_samples": 88,
    "cruise_attitude_mean_speed_mps": 19.7,
    "hover_attitude_rms_deg": 0.44,
    "hover_attitude_samples": 132,
    "hover_attitude_with_payload": True,
    "hover_throttle_pct": 38.0,
    "payload_mass_kg": 1.5,
}


def test_flight_quality_checks_judge_pass_partial_and_scope_caveats():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    results = {r["check"]: r for r in rgf._req_results(_QUALITY_LIVE, planned)}

    assert results["cruise_speed"]["status"] == "PASS"
    assert "nil-wind" in results["cruise_speed"]["message"]

    assert results["cruise_attitude"]["status"] == "PARTIAL"
    assert "not swept" in results["cruise_attitude"]["message"]

    assert results["payload_attitude"]["status"] == "PARTIAL"
    assert "margin" in results["payload_attitude"]["message"]
    assert "carry-hover" in results["payload_attitude"]["message"]


def test_flight_quality_checks_fail_on_violated_limits():
    planned = rgf._planned_gazebo_reqs(_QUALITY_REQS)
    live = dict(_QUALITY_LIVE)
    live.update({
        "nilwind_dash_speed_mps": 12.4,     # below 18
        "cruise_attitude_rms_deg": 0.9,     # above 0.5
        "hover_throttle_pct": 80.0,         # margin 20 < 30
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
    live["hover_attitude_with_payload"] = False
    results = {r["check"]: r for r in rgf._req_results(live, planned)}
    # hover attitude measured without the payload must not judge FUNC-003
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
