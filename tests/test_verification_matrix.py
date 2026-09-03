"""Verification strategy matrix: SITL-unmapped != unverified.

Pins the tier-assignment rules so "unassigned" stays the real gap: requirements
verified at datasheet/behavioral tiers, inspection-only compliance items, and
Gazebo-planned physics stay out of one "unmapped" bucket.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.prototyping.verification_matrix import build_matrix, summarize, to_json, to_markdown
from src.prototyping.verification_obligations import compile_verification_obligations
from src.sitl.requirement_linker import (
    RequirementEvidenceBundle,
    RequirementLinker,
)
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
        model, _REALIZATION, linker.compile_evidence(),
        l2_results=[{"req_id": "REQ_SAFE_003", "passed": True}],
    )}


def test_tiers_assigned_per_class():
    rows = _rows()

    gcs = rows["REQ_SAFE_003"]
    assert "l2_sitl" in gcs.tiers and "behavioral_sim" in gcs.tiers
    assert gcs.status == "verified"
    assert any("behavior-chain" in e for e in gcs.evidence)

    endurance = rows["REQ_PERF_002"]
    assert endurance.tiers == ("datasheet",)
    assert endurance.status == "verified"

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


def test_summary_shows_honest_gap():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    rows = build_matrix(model, _REALIZATION, linker.compile_evidence())

    s = summarize(rows)
    assert s["total"] == 6
    assert s["by_status"]["unassigned"] == 1
    assert s["unassigned_req_ids"] == ["REQ_MISC_001"]

    md = to_markdown(rows)
    assert "Unassigned (honest gap)" in md
    assert "REQ_MISC_001" in md


def test_planned_l2_not_verified():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)

    row = {r.req_id: r for r in build_matrix(model, _REALIZATION, linker.compile_evidence())}["REQ_SAFE_003"]

    assert "l2_sitl_planned" in row.tiers
    assert "l2_sitl" not in row.tiers
    assert row.status == "partial"
    assert any("planned, not executed" in e for e in row.evidence)


def test_executed_l2_pass_and_fail():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)

    passed = {r.req_id: r for r in build_matrix(
        model, _REALIZATION, linker.compile_evidence(),
        l2_results=[{"req_id": "REQ-SAFE-003", "passed": True}],
    )}["REQ_SAFE_003"]
    failed = {r.req_id: r for r in build_matrix(
        model, _REALIZATION, linker.compile_evidence(),
        l2_results=[{"req_id": "REQ_SAFE_003", "passed": False}],
    )}["REQ_SAFE_003"]

    assert "l2_sitl" in passed.tiers and passed.status == "verified"
    assert "l2_sitl_failed" in failed.tiers and failed.status == "failed"


def test_inconclusive_l2_partial():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)

    row = {r.req_id: r for r in build_matrix(
        model, _REALIZATION, linker.compile_evidence(),
        l2_results=[{
            "req_id": "REQ-SAFE-003",
            "passed": False,
            "conclusive": False,
        }],
    )}["REQ_SAFE_003"]

    assert "l2_sitl_inconclusive" in row.tiers
    assert row.status == "partial"


def test_unmet_datasheet_fails():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    realization = {
        "per_requirement": [{
            "req_id": "REQ-PERF-002", "family": "time", "scope": "closure",
            "target": 20.0, "realized_value": 15.0, "met": False,
        }],
    }

    row = {r.req_id: r for r in build_matrix(model, realization, linker.compile_evidence())}["REQ_PERF_002"]

    assert row.status == "failed"
    assert "datasheet_failed" in row.tiers
    assert "datasheet" not in row.tiers


def test_unmet_forward_flight_fails():
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    realization = {
        "per_requirement": [{
            "req_id": "REQ-FUNC-002", "family": "range", "scope": "forward_flight",
            "target": 10.0, "realized_value": 5.0, "met": False,
        }],
    }

    row = {r.req_id: r for r in build_matrix(model, realization, linker.compile_evidence())}["REQ_FUNC_002"]

    assert row.status == "failed"
    assert "forward_flight_failed" in row.tiers
    assert "forward_flight" not in row.tiers


def test_l1_needs_validation_result():
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
    evidence = RequirementEvidenceBundle(
        model_digest=__import__("hashlib").sha256(
            (model.to_sysml_text() or "").encode("utf-8")
        ).hexdigest(),
        requirement_texts={
            "REQ_PERF_006": "Control loop rate shall be at least 10 Hz."
        },
        satisfying_parts={"REQ_PERF_006": ("Drone",)},
        guard_assignments={},
        test_specs=(spec,),
        resolved_params=(),
        parm_file="",
        traceability_mismatches=(),
        coverage={},
    )

    planned = build_matrix(model, None, evidence)[0]
    passed = build_matrix(
        model, None, evidence,
        l1_results=[{"req_id": "REQ_PERF_006", "passed": True}],
    )[0]
    failed = build_matrix(
        model, None, evidence,
        l1_results=[{"req_id": "REQ_PERF_006", "passed": False}],
    )[0]

    assert planned.tiers == ("l1_param_planned",) and planned.status == "planned"
    assert passed.tiers == ("l1_param",) and passed.status == "verified"
    assert failed.tiers == ("l1_param_failed",) and failed.status == "failed"


def test_serial_not_report_evidence():
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
        model, None, linker.compile_evidence(),
        l1_results=[{"req_id": "REQ_FUNC_008", "passed": True}],
    )

    assert rows[0].status == "unassigned"
    assert rows[0].tiers == ()


def test_phase_machine_not_evidence():
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
    row = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]

    assert row.status == "unassigned"
    assert "behavioral_sim" not in row.tiers
    assert any("response action is not produced" in item for item in row.evidence)


def test_reachable_report_accepted():
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
    row = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]

    assert row.status == "partial"
    assert "behavioral_sim" in row.tiers
    assert {item.kind: item.status for item in row.obligations} == {
        "behavior": "verified",
        "response_time": "unverified",
    }


def test_compound_req_needs_clauses():
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

    row = build_matrix(model, realization, RequirementLinker(model).compile_evidence())[0]
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


def test_behavioral_pass_not_accuracy():
    model_text = _MODEL.replace(
        "Operate in sequential phases: STANDBY then CRUISE then LANDING.",
        "Operate in sequential phases: STANDBY then CRUISE then LANDING, "
        "with a CEP of less than 1.0 metre.",
    )
    model = build_lite_model(model_text, model_name="D")

    row = {item.req_id: item for item in build_matrix(
        model, _REALIZATION, RequirementLinker(model).compile_evidence()
    )}["REQ_OPER_001"]

    assert row.status == "partial"
    assert {item.kind: item.status for item in row.obligations} == {
        "behavior": "verified",
        "position_accuracy": "unverified",
    }


def test_percent_and_temp_limits_kept():
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


def test_trace_blocked_marked():
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
    rows = {r.req_id: r for r in build_matrix(model, None, linker.compile_evidence())}

    assert rows["REQ_SAFE_003"].status == "blocked"
    assert any("TRACE blocked" in e for e in rows["REQ_SAFE_003"].evidence)


def test_gazebo_deferred_mix_partial():
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
    rows = {r.req_id: r for r in build_matrix(model, realization, linker.compile_evidence())}

    row = rows["REQ_PERF_005"]
    assert "forward_flight" in row.tiers
    assert "gazebo_deferred" in row.tiers
    assert row.status == "partial"


def test_gazebo_pass_and_fail():
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
    rows = {r.req_id: r for r in build_matrix(model, None, linker.compile_evidence(), gazebo=gazebo)}

    assert rows["REQ_SAFE_007"].status == "partial"
    assert "gazebo_partial" in rows["REQ_SAFE_007"].tiers
    assert any("legacy PASS downgraded" in e for e in rows["REQ_SAFE_007"].evidence)

    assert rows["REQ_FUNC_002"].status == "failed"
    assert "gazebo_failed" in rows["REQ_FUNC_002"].tiers
    assert any("Gazebo FAIL" in e for e in rows["REQ_FUNC_002"].evidence)


def test_partial_gazebo_no_false_green():
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

    row = {r.req_id: r for r in build_matrix(model, None, linker.compile_evidence(), gazebo=gazebo)}["REQ_PERF_004"]
    assert row.status == "partial"
    assert "gazebo_partial" in row.tiers
    assert "gazebo_deferred" in row.tiers
    assert any("Gazebo partial" in e for e in row.evidence)


def test_motor_out_criterion_kept():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_007 {
                doc /* The system shall maintain controlled flight following the
                       failure of a single propulsion unit. */
            }
            part Drone { satisfy requirement REQ_SAFE_007; }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    gazebo = {"req_results": [{
        "req_id": "REQ-SAFE-007",
        "check": "single_motor_out",
        "status": "PARTIAL",
        "message": "2 of 5 met the engineering interpretation",
        "criterion": {
            "metric": "attitude_rms_deg",
            "operator": "<=",
            "threshold": 5.0,
            "unit": "deg",
            "source": "engineering_judgement",
            "basis": "separates large-amplitude wobble",
            "accepted_for_requirement": False,
        },
        "sensitivity": [
            {"interpretation": "attitude RMS <= 5 deg", "passed_runs": 2,
             "total_runs": 5},
            {"interpretation": "flight completed without scenario termination",
             "passed_runs": 5, "total_runs": 5},
        ],
    }]}

    row = {r.req_id: r for r in build_matrix(
        model, None, linker.compile_evidence(), gazebo=gazebo,
    )}["REQ_SAFE_007"]
    obligation = row.obligations[0]

    assert row.status == "partial"
    assert obligation.status == "partial"
    assert obligation.criteria[0].source.value == "engineering_judgement"
    assert obligation.criteria[0].accepted_for_requirement is False
    assert [(item.passed_runs, item.total_runs) for item in obligation.sensitivity] == [
        (2, 5),
        (5, 5),
    ]


def test_parachute_timing_and_precedence():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_005 {
                doc /* Deploy the parachute within 0.5 seconds of critical
                       propulsion failure, taking precedence over all other
                       safety responses. */
            }
            part Drone { satisfy requirement REQ_SAFE_005; }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    gazebo = {"req_results": [
        {"req_id": "REQ-SAFE-005", "check": "parachute_deploy_timing",
         "status": "PASS", "message": "physical deployment in 0.04 s",
         "action_definition_observed": "deployParachute"},
        {"req_id": "REQ-SAFE-005", "check": "safety_precedence",
         "status": "PASS", "message": "all declared competitors suppressed"},
    ]}

    row = {r.req_id: r for r in build_matrix(
        model, None, linker.compile_evidence(), gazebo=gazebo,
    )}["REQ_SAFE_005"]

    assert row.status == "verified"
    assert {obligation.kind.value: obligation.status for obligation in row.obligations} == {
        "behavior": "verified",
        "response_time": "verified",
        "precedence": "verified",
    }


def test_pass_closes_stated_bar():
    """The discriminator is whether an acceptance bar exists to judge against, not
    whether the runner attached a criterion object.

    "within 0.5 seconds" is the requirement's own bar, so a PASS applied it;
    "maintain controlled flight" states none, so a PASS there used a definition of
    its own that cannot be inspected.
    """
    from src.prototyping.verification_obligations import states_acceptance_threshold

    assert states_acceptance_threshold(
        "Deploy the parachute within 0.5 seconds of critical propulsion failure."
    )
    assert not states_acceptance_threshold(
        "The system shall maintain controlled flight following the failure of a "
        "single propulsion unit."
    )

    def _row(doc, req_id, check):
        model = build_lite_model(
            f"""package D {{
                requirement def {req_id} {{ doc /* {doc} */ }}
                part Drone {{ satisfy requirement {req_id}; }}
            }}""",
            model_name="D",
        )
        gazebo = {"req_results": [{
            "req_id": req_id.replace("_", "-"), "check": check,
            "status": "PASS", "message": "observed",
        }]}
        return {r.req_id: r for r in build_matrix(
            model, None, RequirementLinker(model).compile_evidence(), gazebo=gazebo,
        )}[req_id]

    stated = _row("Deploy the parachute within 0.5 seconds of critical "
                  "propulsion failure.", "REQ_SAFE_005", "parachute_deploy_timing")
    assert stated.status == "verified"
    assert all("legacy PASS downgraded" not in e for e in stated.evidence)

    unstated = _row("The system shall maintain controlled flight following the "
                    "failure of a single propulsion unit.",
                    "REQ_SAFE_007", "single_motor_out")
    assert unstated.status == "partial"
    assert any("legacy PASS downgraded" in e for e in unstated.evidence)
    assert any("no verification criterion declared" in e for e in unstated.evidence)


def _row(req_id, text, statuses):
    from src.prototyping.verification_matrix import MatrixRow
    from src.prototyping.verification_obligations import ObligationKind, ObligationResult
    return MatrixRow(
        req_id=req_id, text=text, tiers=(), methods=(), status="partial",
        obligations=tuple(
            ObligationResult(obligation_id=f"{req_id}_{i}", clause=text,
                             kind=ObligationKind.BEHAVIOR, status=st)
            for i, st in enumerate(statuses)
        ),
    )


def test_out_of_scope_excluded():
    """Scoring a simulation stack on an IP54 ingress rating measures nothing about the
    stack.

    Both denominators are published so a reader can see which one a claim rests on.
    """
    s = summarize([
        _row("REQ-CONS-002",
             "All enclosures shall meet a minimum IP54 ingress-protection "
             "rating. [V: inspection / ingress test]",
             ["out-of-sim-scope"]),
        _row("REQ-PERF-003",
             "The system shall achieve a cruise airspeed of at least 18 m/s.",
             ["verified", "verified"]),
    ])

    assert s["obligations_total"] == 3
    assert s["obligations_out_of_sim_scope"] == 1
    assert s["obligations_in_sim_scope"] == 2
    assert s["obligations_verified"] == 2
    assert s["out_of_sim_scope_by_rule"] == {"enclosure_ingress": ["REQ-CONS-002"]}
    assert s["out_of_sim_scope_without_a_rule"] == []
    # the [V:] tag corroborates that rule; it is not a second way in
    assert s["out_of_sim_scope_corroborated_by_requirement"] == ["REQ-CONS-002"]


def test_untagged_req_same_rule():
    """A requirement with no [V:] tag is decided by the same rule table, which names
    the observable the stack lacks.

    One criterion, so the denominator cannot be widened by hand.
    """
    s = summarize([
        _row("REQ-INTF-001",
             "The system shall exchange telemetry with the GCS using the "
             "MAVLink v2.0 protocol over an AES-256 encrypted RF channel.",
             ["verified", "out-of-sim-scope"]),
    ])

    assert s["out_of_sim_scope_by_rule"] == {"cryptography": ["REQ-INTF-001"]}
    assert s["out_of_sim_scope_without_a_rule"] == []
    assert s["out_of_sim_scope_corroborated_by_requirement"] == []


def test_both_denominators_in_markdown():
    md = to_markdown([
        _row("REQ-CONS-004",
             "The system shall comply with EASA UAS Category C operational "
             "regulations. [V: inspection / regulatory audit]",
             ["out-of-sim-scope"]),
        _row("REQ-PERF-003", "cruise airspeed of at least 18 m/s", ["verified"]),
    ])

    assert "all clauses: 1/2" in md
    assert "simulation-reachable clauses only: 1/1" in md
    assert "excluded by rule `regulatory_conformance`" in md
    assert "no flight observable stands in for it" in md
    assert "REQ-CONS-004" in md
    assert "corroborated by the requirement's own [V:] method: REQ-CONS-004" in md


def test_ruleless_exclusion_unauditable():
    s = summarize([
        _row("REQ-MYST-001", "The system shall be good.", ["out-of-sim-scope"]),
    ])
    assert s["out_of_sim_scope_without_a_rule"] == ["REQ-MYST-001"]
    assert s["out_of_sim_scope_by_rule"] == {}

    md = to_markdown([
        _row("REQ-MYST-001", "The system shall be good.", ["out-of-sim-scope"]),
    ])
    assert "EXCLUDED WITH NO RULE (unauditable): REQ-MYST-001" in md


def test_rules_name_distinct_observables():
    from src.prototyping.verification_obligations import NON_SIMULABLE_RULES

    names = [n for n, _, _ in NON_SIMULABLE_RULES]
    reasons = [r for _, _, r in NON_SIMULABLE_RULES]
    terms = [t for _, ts, _ in NON_SIMULABLE_RULES for t in ts]
    assert len(set(names)) == len(names)
    assert len(set(reasons)) == len(reasons)
    # no term may be claimed by two rules, or the exclusion reason is ambiguous
    assert len(set(terms)) == len(terms)


# ── A2: tier input contract ────────────────────────────────────────
# run3's archived report carried a realization dict with per_requirement
# projected away, and the matrix reported three requirements as "unassigned" -
# a runner artefact blamed on the model.


def _rows_with(realization):
    model = build_lite_model(_MODEL, model_name="D")
    linker = RequirementLinker(model)
    return {r.req_id: r for r in build_matrix(
        model, realization, linker.compile_evidence(),
    )}


def test_missing_per_requirement_gap():
    """A realization dict without per_requirement (run3's archived shape): tier-less
    rows read evidence-input-missing, attributable and distinct from the ontology
    gap.
    """
    run3_shaped = {
        "verdict": "CLOSED", "summary": "…", "chosen": None,
        "forward_flight_ok": True, "rank_preservation": {}, "resize_note": "",
    }
    rows = _rows_with(run3_shaped)
    assert rows["REQ_MISC_001"].status == "evidence-input-missing"
    # The endurance requirement loses its datasheet tier with the input (the run3
    # symptom) and does not read as a model gap either.
    assert rows["REQ_PERF_002"].status == "evidence-input-missing"

    from src.prototyping.verification_matrix import summarize, to_markdown
    all_rows = list(_rows_with(run3_shaped).values())
    s = summarize(all_rows)
    assert "REQ_MISC_001" in s["evidence_input_missing_req_ids"]
    assert s["unassigned_req_ids"] == []
    md = to_markdown(all_rows)
    assert "runner gap — NOT a model finding" in md


def test_empty_per_requirement_unassigned():
    rows = _rows_with({"per_requirement": []})
    assert rows["REQ_MISC_001"].status == "unassigned"


def test_no_realization_unassigned():
    rows = _rows_with(None)
    assert rows["REQ_MISC_001"].status == "unassigned"


def test_run_report_keeps_tier_input():
    import inspect
    from src.app.pipeline import PrototypingPipeline
    source = inspect.getsource(PrototypingPipeline.build_run_report)
    assert '"per_requirement": realization.get("per_requirement")' in source
