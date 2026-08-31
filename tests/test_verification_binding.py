"""PLAN-tier requirement→identity bindings, pinned against run3's real plan.

The four vocabulary layers the harness held (event spellings, action-def
spellings, sent-command substrings) are all derivable from the frozen plan;
these tests pin that derivation on the committed run3 evidence bundle."""
from __future__ import annotations

import json
from pathlib import Path

from src.simulation.verification_binding import (
    RequirementBinding,
    binding_for,
    plan_bindings,
)

_BUNDLE = (
    Path(__file__).parent.parent
    / "examples" / "output" / "run3_authoritative_20260831"
)


def _run3_bindings():
    report = json.loads((_BUNDLE / "run_report.json").read_text())
    return plan_bindings(report["whole_model_generation_plan"])


def test_release_identities_come_from_the_plan_not_the_harness():
    """Layer 1+2: the harness guessed 'DeliveryCoordinateSatisfied' and
    'actuateRelease'. The plan states what run3 declares — attributing the
    release behavior to REQ_PERF_005 (release latency), with REQ_FUNC_005
    holding the command route. Bindings mirror the plan's own attribution;
    they never re-guess it."""
    bindings = _run3_bindings()
    release = binding_for(bindings, "REQ_PERF_005")
    assert release.behavior == "PayloadReleaseBehavior"
    assert release.trigger_events == ("DeliveryCoordinateConditionSatisfied",)
    assert release.response_actions == ("releasePayload",)
    assert ("releaseConditionMet", "not deliveryAbortConditionActive") in (
        release.transition_guards
    )
    delivery = binding_for(bindings, "REQ_FUNC_005")
    assert ("FlightController", "deliveryPayloadCmd",
            "PayloadMechanism", "deliveryPayloadCmd") in delivery.route


def test_parachute_route_and_action_come_from_the_plan():
    """Layer 3+4: 'deployParachute' and the CHUTE-substring command check
    both dissolve — the plan names the action and the command's port route."""
    b = binding_for(_run3_bindings(), "REQ_SAFE_005")
    assert "deployBallisticRecoveryParachute" in b.response_actions
    assert b.trigger_events == ("CriticalPropulsionSubsystemFailure",)
    assert "recoveryCmd" in b.route_ports
    assert ("SafetyMonitor", "recoveryCmd", "RecoverySystem", "recoveryCmd") \
        in b.route


def test_inhibition_binding_carries_the_guard_verbatim():
    """Identity only: the binding reports the guard; judging it is the
    criteria's job, and a resolver must never substitute a passing element."""
    b = binding_for(_run3_bindings(), "REQ_SAFE_006")
    assert b.behavior == "PayloadLockBehavior"
    guards = dict(b.transition_guards)
    assert guards.get("unlockCommandReceived") == (
        "not deliveryAbortConditionActive"
    )


def test_requirement_id_spelling_is_normalised():
    bindings = _run3_bindings()
    assert binding_for(bindings, "REQ-FUNC-005") is binding_for(
        bindings, "REQ_FUNC_005"
    )


def test_no_plan_yields_no_bindings_never_a_verdict():
    assert plan_bindings(None) == {}
    assert plan_bindings({"behaviors": "not-a-list"}) == {}


def test_causal_path_only_requirements_still_get_a_route():
    b = binding_for(_run3_bindings(), "REQ_SAFE_003")
    assert b.route  # uplink loss path: CommunicationSystem → SafetyMonitor → FC
    assert "uplinkCommStatus" in b.route_ports
