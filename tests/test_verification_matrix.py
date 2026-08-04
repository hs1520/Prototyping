"""Verification strategy matrix: SITL-unmapped ≠ unverified.

Pins the tier-assignment rules so "unassigned" stays the true honest gap:
requirements verified at datasheet/behavioral tiers, inspection-only compliance
items, and Gazebo-planned physics must not be lumped into one "unmapped" bucket.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.prototyping.verification_matrix import build_matrix, summarize, to_json, to_markdown
from src.prototyping.verification_obligations import compile_verification_obligations
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

_MODEL = """package D {
    requirement def REQ_SAFE_003 {
        doc /* GCS link loss for more than 10 seconds shall trigger safe landing. */
    }
    requirement def REQ_PERF_002 {
        doc /* The system shall sustain flight for a minimum of 20 minutes. */
    }
    requirement def REQ_CONS_002 {
        doc /* Enclosures shall meet a minimum IP54 ingress-protection rating. */
    }
    requirement def REQ_FUNC_002 {
        doc /* Detect obstacles and initiate collision avoidance manoeuvres. */
    }
    requirement def REQ_OPER_001 {
        doc /* Operate in sequential phases: STANDBY then CRUISE then LANDING. */
    }
    requirement def REQ_MISC_001 {
        doc /* The airframe paint shall be blue. */
    }
    part def SafetyMonitor {
        attribute commLossTime : Real = 0.0;
        satisfy requirement REQ_SAFE_003;
        satisfy requirement REQ_PERF_002;
        satisfy requirement REQ_CONS_002;
        satisfy requirement REQ_FUNC_002;
        satisfy requirement REQ_OPER_001;
        satisfy requirement REQ_MISC_001;
        state def Monitor {
            state nominal;
            state lost;
            transition initial then nominal;
            transition gcs first nominal if commLossTime > 10.0 then lost;
        }
    }
}"""

_REALIZATION = {
    "per_requirement": [
        {"req_id": "REQ-PERF-002", "family": "time", "scope": "closure",
         "target": 20.0, "realized_value": 31.8, "met": True},
    ],
}


def _rows():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    return {r.req_id: r for r in build_matrix(
        model, _REALIZATION, linker,
        l2_results=[{"req_id": "REQ_SAFE_003", "passed": True}],
    )}


def test_matrix_assigns_each_requirement_class_to_the_right_tier():
    rows = _rows()

    gcs = rows["REQ_SAFE_003"]
    assert "l2_sitl" in gcs.tiers and "behavioral_sim" in gcs.tiers
    assert gcs.status == "verified"
    assert any("behavior-chain" in e for e in gcs.evidence)

    endurance = rows["REQ_PERF_002"]
    assert endurance.tiers == ("datasheet",)
    assert endurance.status == "verified"  # SITL-unmapped but datasheet-verified

    ip54 = rows["REQ_CONS_002"]
    assert ip54.tiers == ("inspection_analysis",)
    assert ip54.status == "out-of-sim-scope"

    obstacle = rows["REQ_FUNC_002"]
    assert obstacle.tiers == ("gazebo_deferred",)
    assert obstacle.status == "planned"

    phases = rows["REQ_OPER_001"]
    assert "behavioral_sim" in phases.tiers
    assert phases.status == "verified"

    mystery = rows["REQ_MISC_001"]
    assert mystery.tiers == ()
    assert mystery.status == "unassigned"


def test_matrix_summary_and_markdown_surface_the_honest_gap():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    rows = build_matrix(model, _REALIZATION, linker)

    s = summarize(rows)
    assert s["total"] == 6
    assert s["by_status"]["unassigned"] == 1
    assert s["unassigned_req_ids"] == ["REQ_MISC_001"]

    md = to_markdown(rows)
    assert "Unassigned (honest gap)" in md
    assert "REQ_MISC_001" in md


def test_matrix_does_not_mark_planned_l2_as_verified_without_execution_result():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)

    row = {r.req_id: r for r in build_matrix(model, _REALIZATION, linker)}["REQ_SAFE_003"]

    assert "l2_sitl_planned" in row.tiers
    assert "l2_sitl" not in row.tiers
    assert row.status == "partial"  # behavioral model evidence exists; native SITL remains planned
    assert any("planned, not executed" in e for e in row.evidence)


def test_matrix_marks_executed_l2_pass_and_fail_from_results():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)

    passed = {r.req_id: r for r in build_matrix(
        model, _REALIZATION, linker,
        l2_results=[{"req_id": "REQ-SAFE-003", "passed": True}],
    )}["REQ_SAFE_003"]
    failed = {r.req_id: r for r in build_matrix(
        model, _REALIZATION, linker,
        l2_results=[{"req_id": "REQ_SAFE_003", "passed": False}],
    )}["REQ_SAFE_003"]

    assert "l2_sitl" in passed.tiers and passed.status == "verified"
    assert "l2_sitl_failed" in failed.tiers and failed.status == "failed"


def test_matrix_never_marks_an_unmet_datasheet_result_verified():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    realization = {
        "per_requirement": [{
            "req_id": "REQ-PERF-002", "family": "time", "scope": "closure",
            "target": 20.0, "realized_value": 15.0, "met": False,
        }],
    }

    row = {r.req_id: r for r in build_matrix(model, realization, linker)}["REQ_PERF_002"]

    assert row.status == "failed"
    assert "datasheet_failed" in row.tiers
    assert "datasheet" not in row.tiers


def test_matrix_never_marks_an_unmet_forward_flight_result_verified():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    realization = {
        "per_requirement": [{
            "req_id": "REQ-FUNC-002", "family": "range", "scope": "forward_flight",
            "target": 10.0, "realized_value": 5.0, "met": False,
        }],
    }

    row = {r.req_id: r for r in build_matrix(model, realization, linker)}["REQ_FUNC_002"]

    assert row.status == "failed"
    assert "forward_flight_failed" in row.tiers
    assert "forward_flight" not in row.tiers


def test_matrix_requires_an_l1_validation_result_before_marking_l1_verified():
    model = build_lite_model(
        """package D {
            requirement def REQ_PERF_006 { doc /* Control loop rate shall be at least 10 Hz. */ }
            part Drone { satisfy requirement REQ_PERF_006; }
        }""",
        model_name="D",
    )
    spec = SimpleNamespace(
        req_id="REQ_PERF_006", tier="L1",
        params=[SimpleNamespace(param_name="SCHED_LOOP_RATE")],
    )
    linker = SimpleNamespace(
        _req_texts={"REQ_PERF_006": "Control loop rate shall be at least 10 Hz."},
        _satisfy_map={"REQ_PERF_006": ["Drone"]},
        _guard_assignment={},
        generate_test_specs=lambda: [spec],
    )

    planned = build_matrix(model, None, linker)[0]
    passed = build_matrix(
        model, None, linker,
        l1_results=[{"req_id": "REQ_PERF_006", "passed": True}],
    )[0]
    failed = build_matrix(
        model, None, linker,
        l1_results=[{"req_id": "REQ_PERF_006", "passed": False}],
    )[0]

    assert planned.tiers == ("l1_param_planned",) and planned.status == "planned"
    assert passed.tiers == ("l1_param",) and passed.status == "verified"
    assert failed.tiers == ("l1_param_failed",) and failed.status == "failed"


def test_matrix_does_not_treat_serial_protocol_as_postflight_report_evidence():
    model = build_lite_model(
        """package D {
            requirement def REQ_FUNC_008 {
                doc /* The system shall transmit a post-flight health report to
                the GCS within 5.0 seconds of landing completion. */
            }
            part def CommunicationSystem {
                attribute encryptionKeyLength : Real = 256.0;
                action def transmitHealthReport { }
                satisfy requirement REQ_FUNC_008;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    rows = build_matrix(
        model, None, linker,
        l1_results=[{"req_id": "REQ_FUNC_008", "passed": True}],
    )

    assert rows[0].status == "unassigned"
    assert rows[0].tiers == ()


def test_matrix_does_not_treat_unrelated_phase_machine_as_report_evidence():
    model = build_lite_model(
        """package D {
            requirement def REQ_FUNC_008 {
                doc /* Transmit a post-flight health report within 5 seconds
                after landing completion. */
            }
            part def FlightController {
                action def transmitHealthReport { }
                state def FlightPhaseMachine {
                    state Cruise;
                    state Land;
                    transition initial then Cruise;
                    transition finish first Cruise accept CmdToLand then Land;
                }
                satisfy requirement REQ_FUNC_008;
            }
        }""",
        model_name="D",
    )
    row = build_matrix(model, None, RequirementLinker(model))[0]

    assert row.status == "unassigned"
    assert "behavioral_sim" not in row.tiers
    assert any("response action is not produced" in item for item in row.evidence)


def test_matrix_accepts_reachable_postflight_report_action():
    model = build_lite_model(
        """package D {
            requirement def REQ_FUNC_008 {
                doc /* Transmit a post-flight health report within 5 seconds
                after landing completion. */
            }
            part def FlightController {
                attribute maxHealthReportLatency : Real = 5.0 [s];
                attribute currentHealthReportLatency : Real = 0.0 [s];
                action def transmitHealthReport { }
                assert constraint healthReportLatencyBound {
                    currentHealthReportLatency <= maxHealthReportLatency
                }
                state def FlightPhaseMachine {
                    state Cruise;
                    state LandingComplete;
                    state ReportSent {
                        entry action report : transmitHealthReport;
                    }
                    transition initial then Cruise;
                    transition completeLanding
                        first Cruise
                        accept CmdToLand
                        then LandingComplete;
                    transition sendReport
                        first LandingComplete
                        accept CmdToReport
                        then ReportSent;
                }
                satisfy requirement REQ_FUNC_008;
            }
        }""",
        model_name="D",
    )
    row = build_matrix(model, None, RequirementLinker(model))[0]

    assert row.status == "partial"
    assert "behavioral_sim" in row.tiers
    assert {item.kind: item.status for item in row.obligations} == {
        "behavior": "verified",
        "response_time": "unverified",
    }


def test_compound_payload_requirement_requires_every_mandatory_clause():
    model = build_lite_model(
        """package D {
            requirement def REQ_FUNC_003 {
                doc /* The system shall transport payloads with a gross mass of
                up to 1.5 kg while maintaining a hover throttle margin of at
                least 30 percent and roll and pitch RMS within 1.0 degree. */
            }
            part Drone { satisfy requirement REQ_FUNC_003; }
        }""",
        model_name="D",
    )
    realization = {"per_requirement": [{
        "req_id": "REQ-FUNC-003", "family": "payload", "scope": "closure",
        "target": 0.3, "realized_value": 0.46, "met": True,
        "note": "attitude/behaviour clauses are not covered at the datasheet tier",
    }]}

    row = build_matrix(model, realization, RequirementLinker(model))[0]
    by_kind = {item.kind: item.status for item in row.obligations}

    assert row.status == "partial"
    assert by_kind == {
        "behavior": "verified",
        "payload": "verified",
        "hover_throttle_margin": "verified",
        "attitude_rms": "unverified",
    }
    payload = to_json([row])["rows"][0]
    assert payload["obligations"][3]["kind"] == "attitude_rms"
    assert payload["obligations"][3]["evidence"] == []


def test_behavioral_pass_cannot_verify_a_position_accuracy_threshold():
    model_text = _MODEL.replace(
        "Operate in sequential phases: STANDBY then CRUISE then LANDING.",
        "Operate in sequential phases: STANDBY then CRUISE then LANDING, "
        "with a CEP of less than 1.0 metre.",
    )
    model = build_lite_model(model_text, model_name="D")

    row = {item.req_id: item for item in build_matrix(
        model, _REALIZATION, RequirementLinker(model)
    )}["REQ_OPER_001"]

    assert row.status == "partial"
    assert {item.kind: item.status for item in row.obligations} == {
        "behavior": "verified",
        "position_accuracy": "unverified",
    }


def test_obligation_compiler_keeps_percent_and_temperature_limits():
    battery = compile_verification_obligations(
        "REQ_SAFE_001",
        "The system shall return when battery state-of-charge reaches 25%.",
    )
    temperature = compile_verification_obligations(
        "REQ_PERF_008",
        "The system shall operate across an ambient temperature range of "
        "-10 °C to +45 °C.",
    )
    timeout = compile_verification_obligations(
        "REQ_SAFE_003",
        "The system shall land when the uplink has been absent for more than "
        "10 consecutive seconds.",
    )

    assert [item.kind for item in battery] == ["behavior", "battery_threshold"]
    assert [item.kind for item in temperature] == [
        "behavior", "temperature", "temperature",
    ]
    assert [item.kind for item in timeout] == ["behavior", "response_time"]


def test_matrix_marks_trace_blocked_requirements():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_003 {
                doc /* The system shall enter RTL when the GCS link is lost. */
            }
            part def SafetyMonitor {
                action def deployParachute { }
                attribute propulsionCriticalFailure : Boolean;
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
                satisfy requirement REQ_SAFE_003;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    rows = {r.req_id: r for r in build_matrix(model, None, linker)}

    assert rows["REQ_SAFE_003"].status == "blocked"
    assert any("TRACE blocked" in e for e in rows["REQ_SAFE_003"].evidence)


def test_matrix_marks_mixed_verified_and_gazebo_deferred_as_partial():
    model = build_lite_model(
        """package D {
            requirement def REQ_PERF_005 {
                doc /* The system shall maintain cruise speed in a 12 m/s headwind. */
            }
            part Drone {
                satisfy requirement REQ_PERF_005;
            }
        }""",
        model_name="D",
    )
    realization = {
        "per_requirement": [
            {"req_id": "REQ-PERF-005", "family": "speed", "scope": "forward_flight",
             "target": 8.0, "realized_value": 13.5, "met": True},
        ],
    }
    linker = RequirementLinker(model)
    rows = {r.req_id: r for r in build_matrix(model, realization, linker)}

    row = rows["REQ_PERF_005"]
    assert "forward_flight" in row.tiers
    assert "gazebo_deferred" in row.tiers
    assert row.status == "partial"


def test_matrix_consumes_gazebo_pass_and_fail_results():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_007 {
                doc /* The system shall maintain controlled flight following the failure of a single propulsion unit. */
            }
            requirement def REQ_FUNC_002 {
                doc /* Detect obstacles and initiate collision avoidance manoeuvres. */
            }
            part Drone {
                satisfy requirement REQ_SAFE_007;
                satisfy requirement REQ_FUNC_002;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    gazebo = {
        "req_results": [
            {"req_id": "REQ-SAFE-007", "check": "single_motor_out", "status": "PASS",
             "message": "stable one-motor-out hover"},
            {"req_id": "REQ-FUNC-002", "check": "obstacle_avoidance", "status": "FAIL",
             "message": "collision occurred"},
        ]
    }
    rows = {r.req_id: r for r in build_matrix(model, None, linker, gazebo=gazebo)}

    assert rows["REQ_SAFE_007"].status == "verified"
    assert rows["REQ_SAFE_007"].tiers == ("gazebo",)
    assert any("Gazebo PASS" in e for e in rows["REQ_SAFE_007"].evidence)

    assert rows["REQ_FUNC_002"].status == "failed"
    assert "gazebo_failed" in rows["REQ_FUNC_002"].tiers
    assert any("Gazebo FAIL" in e for e in rows["REQ_FUNC_002"].evidence)


def test_matrix_preserves_partial_gazebo_evidence_without_false_green():
    model = build_lite_model(
        """package D {
            requirement def REQ_PERF_004 {
                doc /* Maintain 2 m/s ground speed in a 15 m/s headwind. */
            }
            part Drone { satisfy requirement REQ_PERF_004; }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    gazebo = {"req_results": [{
        "req_id": "REQ-PERF-004",
        "check": "wind_condition",
        "status": "PARTIAL",
        "message": "closed-loop flight passed with lumped drag calibration",
    }]}

    row = {r.req_id: r for r in build_matrix(model, None, linker, gazebo=gazebo)}["REQ_PERF_004"]
    assert row.status == "partial"
    assert "gazebo_partial" in row.tiers
    assert "gazebo_deferred" in row.tiers
    assert any("Gazebo partial" in e for e in row.evidence)
