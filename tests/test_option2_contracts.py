from __future__ import annotations

from src.prototyping.contract_types import INCOMPLETE, READY, UNSUPPORTED
from src.prototyping.requirement_contracts import (
    build_contract_bundle,
    build_requirement_contract,
)


def test_timed_parachute_contract_preserves_concept_not_platform_command():
    contract = build_requirement_contract(
        "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
        "within 0.5 seconds of detecting a critical propulsion subsystem failure "
        "during flight."
    )

    assert contract.completeness == READY
    obligation = contract.obligations[0]
    assert obligation.kind == "timed_response"
    assert obligation.trigger.concept == "critical_propulsion_failure"
    assert obligation.response.concept == "deploy_parachute"
    assert obligation.criterion.value == 0.5
    assert "CMD_" not in str(contract.to_dict())
    assert all(item.source_span is not None for item in obligation.provenance)


def test_compound_startup_requirement_becomes_two_obligations():
    contract = build_requirement_contract(
        "REQ-SAFE-010: If the sensor self-test fails, the system shall prevent "
        "arming and alert the GCS."
    )

    assert contract.completeness == READY
    assert {item.response.concept for item in contract.obligations} == {
        "prevent_arming", "alert_gcs",
    }
    assert {item.obligation_id for item in contract.obligations} == {
        "REQ_SAFE_010.O1", "REQ_SAFE_010.O2",
    }


def test_supported_incomplete_and_unsupported_are_distinct():
    incomplete = build_requirement_contract(
        "REQ-FUNC-002: The system shall detect obstacles and avoid collisions."
    )
    unsupported = build_requirement_contract(
        "REQ-INTF-002: The system shall accept RTCM 3.3 corrections."
    )

    assert incomplete.completeness == INCOMPLETE
    assert unsupported.completeness == UNSUPPORTED


def test_obstacle_contract_uses_behavioral_mvp_evidence_tier():
    contract = build_requirement_contract(
        "REQ-FUNC-002: When approaching a stationary collision threat directly "
        "ahead within the forward sensor field of view at a closing speed no "
        "greater than 1.5 m/s, the system shall execute an avoidance manoeuvre "
        "following threat detection no later than 15 metres, maintaining an "
        "airframe-to-obstacle separation of at least 5 metres."
    )

    assert contract.completeness == READY
    assert contract.obligations[0].verification_intent.preferred_tier == "behavioral"


def test_release_contract_uses_waypoint_trigger_and_negative_abort_qualifier():
    contract = build_requirement_contract(
        "REQ-FUNC-005: The system shall release the payload when the current "
        "geographic position is within 1.0 metre of the designated delivery "
        "waypoint and no delivery-abort condition is active."
    )

    trigger = contract.obligations[0].trigger
    assert trigger.concept == "delivery_waypoint_proximity"
    assert trigger.value == 1.0
    assert trigger.qualifiers == ("delivery_abort_inactive",)


def test_bundle_retains_one_result_for_every_requirement():
    bundle = build_contract_bundle([
        "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute within 0.5 seconds.",
        "REQ-PERF-001: The system shall maintain attitude within 0.5 degrees RMS.",
    ])

    assert len(bundle.contracts) == 2
    assert bundle.by_req_id()["REQ_SAFE_005"].completeness == READY
    assert bundle.by_req_id()["REQ_PERF_001"].completeness == UNSUPPORTED


def test_contract_digest_and_source_preserve_stakeholder_whitespace_exactly():
    source = (
        "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute\n"
        "within 0.5 seconds."
    )
    contract = build_requirement_contract(source)

    assert contract.source_text == source
    from src.prototyping.contract_types import source_digest
    assert contract.source_digest == source_digest(source)


def test_single_motor_out_is_design_invariant_not_emergency_action_contract():
    contract = build_requirement_contract(
        "REQ-SAFE-007: The system shall maintain controlled flight following "
        "the failure of any single propulsion motor."
    )

    obligation = contract.obligations[0]
    assert contract.completeness == READY
    assert obligation.kind == "state_invariant"
    assert obligation.trigger.concept == "single_motor_failure"
    assert obligation.response.concept == "maintain_controlled_flight"
    assert obligation.verification_intent.preferred_tier == "gazebo"


def test_payload_release_actuator_default_is_a_lock_not_release_obligation():
    contract = build_requirement_contract(
        "REQ-SAFE-008: The payload-release actuator shall default to the "
        "mechanically locked state upon power-on, before any arming or flight "
        "authorisation."
    )

    assert contract.completeness == READY
    assert len(contract.obligations) == 1
    obligation = contract.obligations[0]
    assert obligation.trigger.concept == "power_on"
    assert obligation.response.concept == "lock_payload"
    assert contract.envelope.operating_states == ("pre_arm",)


def test_compound_contingency_does_not_silently_drop_trigger_branches():
    contract = build_requirement_contract(
        "REQ-FUNC-007: The system shall execute a return-to-base trajectory upon "
        "detection of a non-critical contingency (GCS link loss, geofence breach, "
        "or battery state-of-charge at the return threshold) that does not require "
        "immediate landing."
    )

    assert contract.completeness == INCOMPLETE
    assert contract.obligations[0].trigger.concept == "compound_contingency"
    assert contract.obligations[0].trigger.qualifiers == (
        "gcs_link_absent", "geofence_breach", "battery_state_of_charge",
    )
    assert "must not select only one branch" in contract.gaps[0]
    assert any("cross-reference" in gap for gap in contract.gaps)
    assert any("no-immediate-landing" in gap for gap in contract.gaps)


def test_timed_payload_actuation_is_a_supported_contract_family():
    contract = build_requirement_contract(
        "REQ-PERF-005: The mechanical payload release actuation shall complete "
        "within 2.0 seconds from the moment the delivery coordinate condition "
        "is satisfied."
    )

    assert contract.completeness == READY
    obligation = contract.obligations[0]
    assert obligation.kind == "timed_actuation"
    assert obligation.trigger.concept == "delivery_coordinate_condition_satisfied"
    assert obligation.response.concept == "release_payload"
    assert obligation.criterion.value == 2.0
    assert obligation.verification_intent.preferred_tier == "behavioral"


def test_source_safety_004_has_only_the_inhibit_obligation():
    contract = build_requirement_contract(
        "REQ-SAFE-004: The system shall not transition to the armed or airborne "
        "state if any onboard sensor reports a failure during the power-on "
        "self-test sequence."
    )

    assert contract.completeness == READY
    assert len(contract.obligations) == 1
    assert contract.obligations[0].response.concept == "prevent_arming"


def test_unchecked_safety_precedence_clauses_are_preserved_as_qualifiers():
    battery = build_requirement_contract(
        "REQ-SAFE-001: The system shall initiate an autonomous return-to-base "
        "sequence when the battery state-of-charge reaches 25%, unless a "
        "higher-priority safety response is already in progress."
    )
    parachute = build_requirement_contract(
        "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
        "within 0.5 seconds of detecting a critical propulsion subsystem failure "
        "during flight, taking precedence over all other safety responses."
    )

    assert battery.obligations[0].response.qualifiers == (
        "unless_higher_priority_safety_response_in_progress",
    )
    assert parachute.obligations[0].response.qualifiers == (
        "takes_precedence_over_other_safety_responses",
    )
