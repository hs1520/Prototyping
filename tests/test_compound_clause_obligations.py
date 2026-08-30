"""A compound requirement asserts more than one thing, and its halves can have
different verification means.

REQ-INTF-001 asks for MAVLink v2.0 *over an AES-256 encrypted channel*. SITL
tests the protocol; nothing in simulation tests the encryption. While that was
one obligation over the whole sentence, the untestable half dragged the testable
one out of scope — and had the testable half been tested, its evidence would
have appeared to close the encryption clause too. Both directions are wrong.
"""
from __future__ import annotations

from src.prototyping.verification_obligations import (
    EvidenceClaim,
    INSPECTION_TERMS,
    compile_verification_obligations,
    evaluate_obligations,
    split_capability_and_medium,
)

_INTF_001 = (
    "The system shall exchange telemetry and mission commands with the GCS "
    "using the MAVLink v2.0 protocol over an AES-256 encrypted RF channel."
)


def test_capability_and_untestable_medium_become_separate_obligations():
    obligations = compile_verification_obligations("REQ_INTF_001", _INTF_001)

    assert len(obligations) == 2
    capability, medium = obligations
    assert "MAVLink v2.0 protocol" in capability.clause
    assert "encrypted" not in capability.clause.lower()
    assert "AES-256 encrypted RF channel" in medium.clause
    assert medium.obligation_id.endswith("M")


def test_a_protocol_test_closes_the_protocol_half_and_not_the_encrypted_half():
    obligations = compile_verification_obligations("REQ_INTF_001", _INTF_001)
    sitl = EvidenceClaim(
        description="L2 wire-level MAVLink v2 conformance (PASS)",
        status="verified",
        kinds=frozenset({"behavior"}),
        clause_exclude_terms=frozenset(INSPECTION_TERMS),
    )
    inspection = EvidenceClaim(
        description="inspection/analysis item",
        status="out-of-sim-scope",
        all_obligations=True,
        clause_terms=frozenset(INSPECTION_TERMS),
    )

    results = {r.obligation_id: r for r in
               evaluate_obligations(obligations, [sitl, inspection])}
    capability, medium = obligations

    assert results[capability.obligation_id].status == "verified"
    assert results[medium.obligation_id].status == "out-of-sim-scope"
    # and the evidence does not cross over
    assert "MAVLink" in results[capability.obligation_id].evidence[0]
    assert all("MAVLink" not in e for e in results[medium.obligation_id].evidence)


def test_an_inspection_finding_no_longer_stamps_every_clause():
    """The defect: one 'encrypted' put the whole requirement out of scope."""
    obligations = compile_verification_obligations("REQ_INTF_001", _INTF_001)
    inspection = EvidenceClaim(
        description="inspection/analysis item",
        status="out-of-sim-scope",
        all_obligations=True,
        clause_terms=frozenset(INSPECTION_TERMS),
    )
    results = {r.obligation_id: r for r in evaluate_obligations(obligations, [inspection])}
    capability, medium = obligations

    assert results[medium.obligation_id].status == "out-of-sim-scope"
    # the protocol clause is simply unverified — open, not declared untestable
    assert results[capability.obligation_id].status == "unverified"


# --------------------------------------------------------------------------
# the splitter declines far more often than it fires
# --------------------------------------------------------------------------

def test_a_condition_is_not_a_medium():
    """'operate across an ambient temperature range' is one obligation. Splitting
    it would leave 'The system shall operate', which asserts nothing."""
    assert split_capability_and_medium(
        "The system shall operate across an ambient temperature range of "
        "-10 °C to +45 °C."
    ) is None


def test_an_untestable_term_inside_the_main_clause_does_not_split():
    """REQ-FUNC-004: 'a continuously encrypted bidirectional data link' — the
    encryption qualifies the link itself, and no word-level cut separates them
    without inventing a requirement."""
    assert split_capability_and_medium(
        "The system shall maintain a continuously encrypted bidirectional data "
        "link with the GCS for telemetry upload and mission command download "
        "throughout the operational flight envelope."
    ) is None


def test_a_conformance_qualifier_is_not_a_medium():
    """'in accordance with the ASTM F3411-22 standard' says HOW the same
    capability must behave, not what it runs over."""
    assert split_capability_and_medium(
        "The system shall broadcast remote identification and real-time spatial "
        "positioning data in accordance with the ASTM F3411-22 standard."
    ) is None


def test_a_requirement_with_no_untestable_term_is_never_split():
    obligations = compile_verification_obligations(
        "REQ_PERF_003",
        "The system shall achieve a cruise airspeed of at least 18 m/s in "
        "nil-wind, level-flight conditions using the primary propulsion system.",
    )
    assert [o.kind for o in obligations] == ["behavior", "speed"]


def test_an_l2_check_closes_only_the_kinds_it_actually_measured():
    """A timed check may close a response_time clause; a mode-change check that
    never looked at a threshold may not."""
    from src.prototyping.verification_matrix import _L2_CLOSES_KINDS

    assert _L2_CLOSES_KINDS["assert_waypoint_update_latency"] == {
        "behavior", "response_time",
    }
    # anything undeclared falls back to the behaviour clause alone
    assert "wait_mode" not in _L2_CLOSES_KINDS
    assert "assert_servo_pwm" not in _L2_CLOSES_KINDS


def test_a_timed_l2_result_closes_the_numeric_clause_it_measured():
    obligations = compile_verification_obligations(
        "REQ_FUNC_006",
        "The system shall incorporate a revised waypoint sequence into the "
        "active flight plan within 1.0 second of receiving a valid "
        "waypoint-modification command from the GCS.",
    )
    assert [o.kind for o in obligations] == ["behavior", "response_time"]

    timed = EvidenceClaim(
        description="L2 noop→assert_waypoint_update_latency (PASS)",
        status="verified",
        kinds=frozenset({"behavior", "response_time"}),
    )
    results = evaluate_obligations(obligations, [timed])
    assert [r.status for r in results] == ["verified", "verified"]

    # the same result from a check that measured no interval closes only behaviour
    untimed = EvidenceClaim(
        description="L2 disconnect_gcs→wait_mode (PASS)",
        status="verified",
        kinds=frozenset({"behavior"}),
    )
    results = evaluate_obligations(obligations, [untimed])
    assert [r.status for r in results] == ["verified", "unverified"]
