from __future__ import annotations

from examples import run_gazebo_feasibility as rgf


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
