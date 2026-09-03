"""PLAN-tier requirement->identity bindings, pinned against run3's plan.

The vocabulary layers the harness held (event spellings, action-def spellings,
sent-command substrings) are derivable from the frozen plan; these pin that
derivation on the committed run3 evidence bundle.
"""
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


def test_release_identities_from_plan():
    """Bindings mirror the plan's attribution instead of re-guessing it.

    The harness guessed 'DeliveryCoordinateSatisfied' and 'actuateRelease'; run3's
    plan attributes the release behavior to REQ_PERF_005 (release latency), with
    REQ_FUNC_005 holding the command route.
    """
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


def test_parachute_route_from_plan():
    b = binding_for(_run3_bindings(), "REQ_SAFE_005")
    assert "deployBallisticRecoveryParachute" in b.response_actions
    assert b.trigger_events == ("CriticalPropulsionSubsystemFailure",)
    assert "recoveryCmd" in b.route_ports
    assert ("SafetyMonitor", "recoveryCmd", "RecoverySystem", "recoveryCmd") \
        in b.route


def test_inhibition_guard_verbatim():
    b = binding_for(_run3_bindings(), "REQ_SAFE_006")
    assert b.behavior == "PayloadLockBehavior"
    guards = dict(b.transition_guards)
    assert guards.get("unlockCommandReceived") == (
        "not deliveryAbortConditionActive"
    )


def test_requirement_id_normalised():
    bindings = _run3_bindings()
    assert binding_for(bindings, "REQ-FUNC-005") is binding_for(
        bindings, "REQ_FUNC_005"
    )


def test_no_plan_no_bindings():
    assert plan_bindings(None) == {}
    assert plan_bindings({"behaviors": "not-a-list"}) == {}


def test_causal_path_gets_route():
    b = binding_for(_run3_bindings(), "REQ_SAFE_003")
    assert b.route
    assert "uplinkCommStatus" in b.route_ports
