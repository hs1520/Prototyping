from __future__ import annotations

import pytest

from src.prototyping.platform_semantics import command_matches, platform_binding
from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.safety_patterns import (
    render_generation_guidance,
    render_step_guidance,
    select_patterns,
)


def test_pattern_selection_is_semantic_not_requirement_id_based():
    bundle = build_contract_bundle([
        "REQ-SAFE-099: During flight, critical propulsion failure shall deploy "
        "the parachute within 0.5 seconds."
    ])
    bindings = select_patterns(bundle)

    assert {item.pattern_id for item in bindings} == {
        "TriggeredFailsafeResponse", "TimedEmergencyActuation",
    }


def test_generation_guidance_contains_only_requested_requirement_slice():
    bundle = build_contract_bundle([
        "REQ-SAFE-005: During flight, critical propulsion failure shall deploy "
        "the parachute within 0.5 seconds.",
        "REQ-SAFE-006: The system shall maintain the payload in the mechanically "
        "locked state whenever a delivery-abort condition is active.",
    ])
    bindings = select_patterns(bundle)
    guidance = render_generation_guidance(
        bundle, bindings, req_ids=["REQ-SAFE-005"]
    )

    assert "REQ_SAFE_005.O1" in guidance
    assert "TimedEmergencyActuation" in guidance
    assert "REQ_SAFE_006" not in guidance
    assert "MAV_CMD_DO_PARACHUTE" in guidance


def test_b1_contract_guidance_does_not_require_safety_patterns():
    bundle = build_contract_bundle([
        "REQ-FUNC-006: The system shall incorporate a revised waypoint sequence "
        "within 1.0 second of receiving a valid waypoint-modification command."
    ])

    guidance = render_generation_guidance(bundle, ())

    assert "REQ_FUNC_006.O1" in guidance
    assert "response=revise_waypoint_sequence" in guidance
    assert "Apply " not in guidance


def test_power_on_locked_default_selects_locked_until_authorised_release():
    bundle = build_contract_bundle([
        "REQ-SAFE-008: The payload-release actuator shall default to the "
        "mechanically locked state upon power-on."
    ])

    assert {item.pattern_id for item in select_patterns(bundle)} == {
        "LockedUntilAuthorisedRelease"
    }


def test_each_generation_step_receives_only_its_semantic_slice():
    bundle = build_contract_bundle([
        "REQ-SAFE-005: During flight, critical propulsion failure shall deploy "
        "the parachute within 0.5 seconds."
    ])
    bindings = select_patterns(bundle)

    parts = render_step_guidance(bundle, bindings, step="parts")
    interfaces = render_step_guidance(bundle, bindings, step="interfaces")
    behavior = render_step_guidance(bundle, bindings, step="behavior")
    assembly = render_step_guidance(bundle, bindings, step="assembly")

    assert "trigger_variable=propulsionCriticalFailure" in parts
    assert "entry action" not in parts
    assert "observation_concept=parachute_deployed" in interfaces
    assert "endpoint_authority=architecture" in interfaces
    assert "canonical_commands=MAV_CMD_DO_PARACHUTE" in interfaces
    assert "platform_semantic_tag=PARACHUTE_DEPLOY" in interfaces
    assert "Apply TimedEmergencyActuation" in behavior
    assert "source_digest=" in assembly
    assert "criterion=response_latency <= 0.5 s" in assembly


def test_unknown_generation_guidance_step_is_rejected():
    bundle = build_contract_bundle([
        "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
        "within 0.5 seconds."
    ])

    with pytest.raises(ValueError, match="unknown generation guidance step"):
        render_step_guidance(bundle, (), step="architecture")


@pytest.mark.parametrize(
    ("response_concept", "canonical_command"),
    [
        ("deploy_parachute", "MAV_CMD_DO_PARACHUTE"),
        ("lock_payload", "MAV_CMD_DO_GRIPPER"),
        ("release_payload", "MAV_CMD_DO_GRIPPER"),
        ("return_to_base", "MAV_CMD_NAV_RETURN_TO_LAUNCH"),
        ("controlled_landing", "MAV_CMD_NAV_LAND"),
    ],
)
def test_official_canonical_platform_commands_match_the_checker(
    response_concept: str, canonical_command: str
):
    binding = platform_binding(response_concept)

    assert binding is not None
    assert canonical_command in binding.canonical_commands
    assert command_matches(binding, canonical_command)


def test_platform_bindings_stay_compatible_with_the_sitl_catalogue():
    """Guard against silent drift between bindings and the SITL catalogue.

    Every SITL-tier binding must reference a semantic tag the ArduPilot
    catalogue actually implements; otherwise a repaired/passing model would
    map to an oracle that can never execute.
    """
    from src.prototyping.platform_semantics import validate_catalogue_bindings

    assert validate_catalogue_bindings() == []
