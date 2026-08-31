"""The generated model owns the decision; the harness only actuates it.

Every payload/parachute result in the 2026-08-30 authoritative run said so of
itself: "the coordinate condition was evaluated by the harness, not by generated
mission logic". These tests hold the boundary that removes that caveat, and pin
the arbitration failure driving the model straight away exposed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gazebo_poc.model_mission import ModelAction, ModelDrivenMission

_MODEL = """
package Drone {
    item def DeliveryCoordinateSatisfied;
    item def AbortConditionActive;

    part def PayloadMechanism {
        attribute isLocked : Boolean = true;
        action def actuateRelease {}
        action def lockPayload {}

        state def PayloadReleaseBehavior {
            entry; then Locked;
            state Locked;
            state Releasing {
                entry action onReleasing : actuateRelease;
            }
            transition toReleasing
                first Locked
                accept DeliveryCoordinateSatisfied
                then Releasing;
        }

        state def DeliveryAbortBehavior {
            entry; then Monitoring;
            state Monitoring;
            state Aborted {
                entry action onAborted : lockPayload;
            }
            transition toAborted
                first Monitoring
                accept AbortConditionActive
                then Aborted;
        }
    }
}
"""


def test_the_model_fires_its_own_transition_and_action():
    mission = ModelDrivenMission(_MODEL)
    assert mission.state_of("PayloadMechanism.PayloadReleaseBehavior") == "Locked"

    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert [d.action for d in fired] == ["onReleasing"]
    assert [d.action_definition for d in fired] == ["actuateRelease"]
    assert mission.performed(fired, ModelAction.RELEASE_PAYLOAD)
    assert fired[0].from_state == "Locked" and fired[0].to_state == "Releasing"
    assert fired[0].as_dict()["decided_by"] == "generated model"
    assert mission.state_of("PayloadMechanism.PayloadReleaseBehavior") == "Releasing"


def test_an_unrelated_action_on_the_same_event_does_not_authorize_release():
    model = _MODEL.replace(
        "entry action onReleasing : actuateRelease;",
        "entry action onReleasing : lockPayload;",
    )
    mission = ModelDrivenMission(model)

    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert [decision.action for decision in fired] == ["onReleasing"]
    assert [decision.action_definition for decision in fired] == ["lockPayload"]
    assert not mission.performed(fired, ModelAction.RELEASE_PAYLOAD)


def test_an_event_the_model_ignores_fires_nothing():
    """An empty result must mean the harness does nothing — that is the point of
    asking the model instead of deciding for it."""
    mission = ModelDrivenMission(_MODEL)
    assert mission.offer("SomeEventTheModelNeverDeclared", time=1.0) == ()
    assert mission.decisions == []
    # and a second delivery event finds the machine already past Locked
    mission.offer("DeliveryCoordinateSatisfied", time=1.0)
    assert mission.offer("DeliveryCoordinateSatisfied", time=2.0) == ()


def test_accepted_events_come_from_the_model_not_a_hardcoded_list():
    mission = ModelDrivenMission(_MODEL)
    assert mission.accepted_events() == (
        "AbortConditionActive", "DeliveryCoordinateSatisfied",
    )
    assert mission.handles("AbortConditionActive")
    assert not mission.handles("CriticalPropulsionFailure")


def test_competing_actions_are_discovered_from_the_named_model_machine():
    mission = ModelDrivenMission(_MODEL)

    assert mission.action_definitions_for_machine("DeliveryAbortBehavior") == (
        "lockPayload",
    )


def test_provenance_names_what_was_executed():
    mission = ModelDrivenMission(_MODEL)
    prov = mission.provenance()
    assert len(prov["model_sha256"]) == 64
    assert "PayloadMechanism.PayloadReleaseBehavior" in prov["machines"]
    assert prov["decision_owner"] == "generated model state machines"


def test_unguarded_release_is_not_inhibited_by_an_active_abort():
    """REQ-SAFE-006 shape: locked "whenever a delivery-abort condition is
    active, regardless of proximity". The two behaviours are independent
    machines and the release transition carries no guard, so the abort cannot
    inhibit it. Driving the model is what exposes this; structural checks see
    two well-formed state machines."""
    mission = ModelDrivenMission(_MODEL)

    mission.offer("AbortConditionActive", time=1.0)
    assert mission.state_of("PayloadMechanism.DeliveryAbortBehavior") == "Aborted"

    fired = mission.offer("DeliveryCoordinateSatisfied", time=2.0)

    assert [d.action for d in fired] == ["onReleasing"], (
        "if this ever stops firing, the generated model gained the arbitration "
        "REQ-SAFE-006 asks for and the inhibition scenario should be re-judged"
    )


# --------------------------------------------------------------------------
# the authoritative model itself
# --------------------------------------------------------------------------

_AUTHORITATIVE = Path("examples/output/latest/final_model.sysml")


@pytest.mark.skipif(not _AUTHORITATIVE.exists(), reason="no authoritative run on disk")
def test_authoritative_model_is_drivable_end_to_end():
    mission = ModelDrivenMission(_AUTHORITATIVE.read_text(encoding="utf-8"))

    for event, machine, action in (
        ("DeliveryCoordinateSatisfied", "PayloadReleaseBehavior", "onReleasing"),
        ("CriticalPropulsionFailure", "ParachuteDeploymentBehavior",
         "onDeployingParachute"),
    ):
        assert mission.handles(event)
        fired = [d for d in mission.offer(event, time=1.0) if d.machine == machine]
        assert [d.action for d in fired] == [action], (
            f"{machine} did not own its own {event} response"
        )


def test_guard_only_machines_are_loaded_and_drivable():
    """A machine whose transitions are all guards is still driven: offer() steps
    every machine, and a guard fires on the variables it is given. Filtering
    them out hid SafetyArbiter, which expresses REQ-SAFE-005's precedence
    entirely in guards — so the precedence check saw nothing to take precedence
    over and returned inconclusive forever."""
    model = """
package Drone {
    part def SafetyMonitor {
        attribute hazard : Boolean = false;
        attribute winning : Boolean = false;
        action def competingResponse {}

        state def Arbiter {
            entry; then Nominal;
            state Nominal;
            state Competing {
                entry action onCompeting : competingResponse;
            }
            transition toCompeting
                first Nominal
                if hazard
                and not winning
                then Competing;
        }
    }
}
"""
    mission = ModelDrivenMission(model)
    assert "SafetyMonitor.Arbiter" in mission.provenance()["machines"]
    assert mission.action_definitions_for_machine("Arbiter") == ("competingResponse",)

    # the guard fires on variables, under an event the model never declares
    fired = mission.offer("__control__", time=0.0,
                          variables={"hazard": True, "winning": False})
    assert [d.action_definition for d in fired] == ["competingResponse"]

    # and the winning condition suppresses it — precedence, expressed as a guard
    suppressed = ModelDrivenMission(model).offer(
        "__control__", time=0.0, variables={"hazard": True, "winning": True})
    assert suppressed == ()
