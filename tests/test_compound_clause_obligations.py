"""A compound requirement asserts more than one thing, and its halves can have
different verification means.

REQ-INTF-001 asks for MAVLink v2.0 over an AES-256 encrypted channel: SITL
tests the protocol, nothing in simulation tests the encryption. As one
obligation over the whole sentence, the untestable half pulled the testable
one out of scope, and evidence for the testable half would have appeared to
close the encryption clause too.
"""
from __future__ import annotations

from src.prototyping.verification_obligations import (
    EvidenceCapability,
    EvidenceClaim,
    CriterionEvaluation,
    CriterionSource,
    INSPECTION_TERMS,
    ObligationKind,
    VerificationCriterion,
    compile_verification_obligations,
    evaluate_evidence,
    split_capability_and_medium,
)

_INTF_001 = (
    "The system shall exchange telemetry and mission commands with the GCS "
    "using the MAVLink v2.0 protocol over an AES-256 encrypted RF channel."
)


def test_capability_and_medium_split():
    obligations = compile_verification_obligations("REQ_INTF_001", _INTF_001)

    assert len(obligations) == 2
    capability, medium = obligations
    assert "MAVLink v2.0 protocol" in capability.clause
    assert "encrypted" not in capability.clause.lower()
    assert "AES-256 encrypted RF channel" in medium.clause
    assert medium.obligation_id.endswith("M")


def test_protocol_closes_only_protocol():
    obligations = compile_verification_obligations("REQ_INTF_001", _INTF_001)
    sitl = EvidenceClaim(
        description="L2 wire-level MAVLink v2 conformance (PASS)",
        status="verified",
        capabilities=frozenset({EvidenceCapability.WIRE_PROTOCOL_OBSERVED}),
        clause_exclude_terms=frozenset(INSPECTION_TERMS),
    )
    inspection = EvidenceClaim(
        description="inspection/analysis item",
        status="out-of-sim-scope",
        applies_to_matching_clauses=True,
        clause_terms=frozenset(INSPECTION_TERMS),
    )

    results = {r.obligation_id: r for r in
               evaluate_evidence(obligations, [sitl, inspection])}
    capability, medium = obligations

    assert results[capability.obligation_id].status == "verified"
    assert results[medium.obligation_id].status == "out-of-sim-scope"
    assert "MAVLink" in results[capability.obligation_id].evidence[0]
    assert all("MAVLink" not in e for e in results[medium.obligation_id].evidence)


def test_inspection_not_every_clause():
    obligations = compile_verification_obligations("REQ_INTF_001", _INTF_001)
    inspection = EvidenceClaim(
        description="inspection/analysis item",
        status="out-of-sim-scope",
        applies_to_matching_clauses=True,
        clause_terms=frozenset(INSPECTION_TERMS),
    )
    results = {r.obligation_id: r for r in evaluate_evidence(obligations, [inspection])}
    capability, medium = obligations

    assert results[medium.obligation_id].status == "out-of-sim-scope"
    assert results[capability.obligation_id].status == "unverified"


def test_condition_not_medium():
    assert split_capability_and_medium(
        "The system shall operate across an ambient temperature range of "
        "-10 °C to +45 °C."
    ) is None


def test_untestable_in_clause_no_split():
    """REQ-FUNC-004: 'a continuously encrypted bidirectional data link' - the
    encryption qualifies the link itself, and no word-level cut separates them
    without inventing a requirement."""
    assert split_capability_and_medium(
        "The system shall maintain a continuously encrypted bidirectional data "
        "link with the GCS for telemetry upload and mission command download "
        "throughout the operational flight envelope."
    ) is None


def test_means_not_medium():
    """"authenticate ... using an AES challenge" - the AES challenge is the
    authentication, not a channel it runs over. Splitting it left a testable
    "authenticate every operator command" that a protocol-level test could close
    while the mechanism went untested. A medium is separable; a means is not.
    """
    assert split_capability_and_medium(
        "The system shall authenticate every operator command and reject "
        "unsigned packets using an AES challenge over the uplink."
    ) is None

    # still splits: the channel is a medium and MAVLink v2 conformance
    # over it is a separate capability
    split = split_capability_and_medium(_INTF_001)
    assert split is not None
    capability, medium = split
    assert "MAVLink v2.0 protocol" in capability
    assert "AES-256 encrypted RF channel" in medium


def test_substring_term_no_split():
    assert split_capability_and_medium(
        "The system shall serve cached query results to the operator console "
        "using a materialized database view over a replicated store."
    ) is None


def test_conformance_qualifier_not_medium():
    assert split_capability_and_medium(
        "The system shall broadcast remote identification and real-time spatial "
        "positioning data in accordance with the ASTM F3411-22 standard."
    ) is None


def test_inspection_terms_match_words():
    assert split_capability_and_medium(
        "The system shall log every accepted mission command and parameter "
        "update using a materialized database view."
    ) is None


def test_no_untestable_term_no_split():
    obligations = compile_verification_obligations(
        "REQ_PERF_003",
        "The system shall achieve a cruise airspeed of at least 18 m/s in "
        "nil-wind, level-flight conditions using the primary propulsion system.",
    )
    assert [o.kind for o in obligations] == ["behavior", "speed"]


def test_abort_safe_state_inhibition():
    abort_lock = compile_verification_obligations(
        "REQ_SAFE_006",
        "The system shall maintain the payload in the mechanically locked state "
        "whenever a delivery-abort condition is active, regardless of geographic "
        "proximity to the delivery waypoint.",
    )
    controlled_flight = compile_verification_obligations(
        "REQ_SAFE_007",
        "The system shall maintain controlled flight following the failure of a "
        "single propulsion unit.",
    )

    assert abort_lock[0].kind is ObligationKind.INHIBITION
    assert controlled_flight[0].kind is ObligationKind.CONTROLLED_FLIGHT


def test_capabilities_close_matching_only():
    inhibition = compile_verification_obligations(
        "REQ_SAFE_006",
        "The system shall maintain the payload in the mechanically locked state "
        "whenever a delivery-abort condition is active.",
    )
    protocol = compile_verification_obligations(
        "REQ_INTF_001",
        "The system shall exchange telemetry and mission commands using the "
        "MAVLink v2 protocol.",
    )

    servo_only = EvidenceClaim(
        description="servo output reached release PWM",
        status="verified",
        capabilities=frozenset({EvidenceCapability.ACTUATOR_COMMAND_OBSERVED}),
    )
    inhibited = EvidenceClaim(
        description="abort active and physical payload remained attached",
        status="verified",
        capabilities=frozenset({EvidenceCapability.PHYSICAL_INHIBITION_OBSERVED}),
    )
    wire = EvidenceClaim(
        description="MAVLink v2 frames observed in both directions",
        status="verified",
        capabilities=frozenset({EvidenceCapability.WIRE_PROTOCOL_OBSERVED}),
    )

    assert evaluate_evidence(inhibition, [servo_only])[0].status == "unverified"
    assert evaluate_evidence(inhibition, [inhibited])[0].status == "verified"
    assert evaluate_evidence(protocol, [wire])[0].status == "verified"


def test_timing_not_close_precedence():
    obligations = compile_verification_obligations(
        "REQ_SAFE_005",
        "The system shall deploy the ballistic recovery parachute within 0.5 "
        "seconds of detecting a critical propulsion subsystem failure during "
        "flight, taking precedence over all other safety responses.",
    )
    timing = EvidenceClaim(
        description="failure event to physical deployment measured at 0.2 s",
        status="verified",
        capabilities=frozenset({
            EvidenceCapability.BEHAVIOR_OBSERVED,
            EvidenceCapability.RESPONSE_TIME_MEASURED,
        }),
    )

    results = evaluate_evidence(obligations, [timing])

    assert [result.kind for result in results] == [
        ObligationKind.BEHAVIOR,
        ObligationKind.RESPONSE_TIME,
        ObligationKind.PRECEDENCE,
    ]
    assert [result.status for result in results] == [
        "verified", "verified", "unverified",
    ]


def test_unaccepted_criterion_conditional():
    obligations = compile_verification_obligations(
        "REQ_SAFE_007",
        "The system shall maintain controlled flight following the failure of "
        "a single propulsion unit.",
    )
    criterion = VerificationCriterion(
        metric="attitude_rms_deg",
        operator="<=",
        threshold=5.0,
        unit="deg",
        source=CriterionSource.ENGINEERING_JUDGEMENT,
        basis="separates attitude tracking from large-amplitude wobble",
        accepted_for_requirement=False,
    )
    claim = EvidenceClaim(
        description="two of five flights met the 5 degree interpretation",
        status="failed",
        capabilities=frozenset({EvidenceCapability.CONTROLLED_FLIGHT_OBSERVED}),
        criterion=criterion,
        sensitivity=(
            CriterionEvaluation("attitude RMS <= 5 deg", 2, 5),
            CriterionEvaluation("flight completed without termination", 5, 5),
        ),
    )

    result = evaluate_evidence(obligations, [claim])[0]

    assert result.status == "partial"
    assert result.criteria == (criterion,)
    assert result.sensitivity[0].passed_runs == 2
    assert result.sensitivity[1].passed_runs == 5


def test_l2_closes_measured_kinds():
    from src.prototyping.verification_matrix import _L2_EVIDENCE_CAPABILITIES

    assert _L2_EVIDENCE_CAPABILITIES == {
        "noop": frozenset(),
        "wait_mode": frozenset({EvidenceCapability.MODE_TRANSITION_OBSERVED}),
        "assert_arm_rejected": frozenset({EvidenceCapability.PHYSICAL_INHIBITION_OBSERVED}),
        "assert_sensor_unhealthy": frozenset({EvidenceCapability.SENSOR_STATE_OBSERVED}),
        "assert_servo_pwm": frozenset({EvidenceCapability.ACTUATOR_COMMAND_OBSERVED}),
        "assert_mavlink_v2_link": frozenset({EvidenceCapability.WIRE_PROTOCOL_OBSERVED}),
        "assert_waypoint_update_latency": frozenset({
            EvidenceCapability.ACTIVE_ROUTE_CHANGE_OBSERVED,
            EvidenceCapability.RESPONSE_TIME_MEASURED,
        }),
    }


def test_timed_l2_closes_numeric_clause():
    obligations = compile_verification_obligations(
        "REQ_FUNC_006",
        "The system shall incorporate a revised waypoint sequence into the "
        "active flight plan within 1.0 second of receiving a valid "
        "waypoint-modification command from the GCS.",
    )
    assert [o.kind for o in obligations] == [
        ObligationKind.ACTIVE_ROUTE_CHANGE,
        ObligationKind.RESPONSE_TIME,
    ]

    timed = EvidenceClaim(
        description="L2 noop→assert_waypoint_update_latency (PASS)",
        status="verified",
        capabilities=frozenset({
            EvidenceCapability.ACTIVE_ROUTE_CHANGE_OBSERVED,
            EvidenceCapability.RESPONSE_TIME_MEASURED,
        }),
    )
    results = evaluate_evidence(obligations, [timed])
    assert [r.status for r in results] == ["verified", "verified"]

    untimed = EvidenceClaim(
        description="L2 disconnect_gcs→wait_mode (PASS)",
        status="verified",
        capabilities=frozenset({EvidenceCapability.MODE_TRANSITION_OBSERVED}),
    )
    results = evaluate_evidence(obligations, [untimed])
    assert [r.status for r in results] == ["unverified", "unverified"]
