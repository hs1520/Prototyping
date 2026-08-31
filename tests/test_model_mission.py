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


_GUARDED_RELEASE = """
package GuardedRelease {
    item def DeliveryCoordinateSatisfied;

    part def PayloadMechanism {
        attribute deliveryAbortActive : Boolean;

        state def PayloadReleaseBehavior {
            entry; then Locked;

            state Locked;
            state Releasing {
                entry action onReleasing : actuateRelease;
            }

            transition toReleasing
                first Locked
                accept DeliveryCoordinateSatisfied
                if not deliveryAbortActive
                then Releasing;
        }
        action def actuateRelease {}
    }
}
"""


def test_an_unbound_guard_flag_makes_a_guarded_model_behave_as_unguarded():
    """Measured: with no variables bound, `not deliveryAbortActive` still fired
    the release. An unset flag reads FALSE, so adding the guard to the model
    changes nothing the harness can see — the fix would have looked applied and
    the requirement would still have failed, blaming the model."""
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    assert mission.boolean_guard_attributes() == ("deliveryAbortActive",)
    assert mission.unlatched_boolean_attributes() == ("deliveryAbortActive",)


def test_offering_the_abort_raises_the_flag_the_model_itself_declared():
    """The flag is found by matching the offered event's words against names the
    MODEL declared, so a model that calls it something else is still checkable
    and this harness holds no requirement's spellings."""
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)

    assert mission.conditions == {"deliveryAbortActive": True}
    assert mission.latched == [(0.0, "AbortConditionActive", "deliveryAbortActive")]
    assert mission.unlatched_boolean_attributes() == ()


def test_an_active_abort_inhibits_the_release_the_model_would_otherwise_fire():
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)
    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert fired == ()


def test_an_ordinary_delivery_still_releases():
    """The guard must not inhibit the ordinary case. It did, briefly: matching
    the event against the flag on any shared word raised the abort flag from
    the DELIVERY event itself, because both names carry "delivery"."""
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    fired = mission.offer("DeliveryCoordinateSatisfied", time=0.0)

    assert [d.action for d in fired] == ["onReleasing"]
    assert mission.conditions == {}


def test_a_condition_stays_raised_for_later_events():
    """"Whenever an abort is active" is a standing state, not an instant."""
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)
    mission.offer("SomethingElse", time=1.0)

    assert mission.conditions == {"deliveryAbortActive": True}
    assert mission.offer("DeliveryCoordinateSatisfied", time=2.0) == ()


def test_an_explicit_caller_value_overrides_a_latched_one():
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)
    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0,
                          variables={"deliveryAbortActive": False})

    assert [d.action for d in fired] == ["onReleasing"]


_RENAMED_EVENTS = """
package RenamedEvents {
    item def DeliveryCoordinateConditionSatisfied;
    item def CriticalPropulsionSubsystemFailure;
    item def SinglePropulsionUnitFailure;
    item def PayloadReleaseCommand;

    part def PayloadMechanism {
        attribute deliveryAbortConditionActive : Boolean;

        state def PayloadReleaseBehavior {
            entry; then Locked;

            state Locked;
            state Releasing {
                entry action onReleasing : releasePayload;
            }

            transition releaseConditionMet
                first Locked
                accept DeliveryCoordinateConditionSatisfied
                if not deliveryAbortConditionActive
                then Releasing;
        }
        action def releasePayload {}
    }
}
"""


def test_a_model_may_name_its_own_events_and_still_be_driven():
    """run3's model accepts `DeliveryCoordinateConditionSatisfied` where the
    harness offered `DeliveryCoordinateSatisfied`. Nothing fired, and the
    evidence would have read "the generated logic declined to release" — a
    harness spelling reported as a model defect."""
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)

    resolved, how = mission.resolve_event("DeliveryCoordinateSatisfied")
    assert resolved == "DeliveryCoordinateConditionSatisfied"
    assert "semantic match" in how

    fired = mission.offer("DeliveryCoordinateSatisfied", time=0.0)
    assert [d.action for d in fired] == ["onReleasing"]
    assert mission.resolutions[0][1:3] == (
        "DeliveryCoordinateSatisfied", "DeliveryCoordinateConditionSatisfied")


def test_a_declared_name_is_used_verbatim_and_not_renamed():
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    resolved, how = mission.resolve_event("DeliveryCoordinateConditionSatisfied")
    assert resolved == "DeliveryCoordinateConditionSatisfied"
    assert how == "declared verbatim"
    mission.offer("DeliveryCoordinateConditionSatisfied", time=0.0)
    assert mission.resolutions == []


def test_an_event_the_model_never_declares_is_recorded_not_guessed():
    """"The model was never asked" and "the model was asked and declined" are
    different findings. A scenario resting on an unresolved event proves
    nothing, so it must not look like a refusal."""
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    resolved, why = mission.resolve_event("AbortConditionActive")

    assert resolved is None
    assert "no declared event covers" in why


def test_a_condition_expressed_as_a_flag_is_not_an_unasked_model():
    """run3 has no abort EVENT at all — the delivery abort is a standing
    boolean its guards read. "No event matched" would read as "the model was
    never told", when the latched flag told it. Only an offer that resolves to
    nothing AND raises nothing established nothing."""
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    mission.offer("AbortConditionActive", time=0.0)

    assert mission.unresolved == []
    assert mission.condition_only == [
        (0.0, "AbortConditionActive", ("deliveryAbortConditionActive",))]
    assert mission.conditions == {"deliveryAbortConditionActive": True}
    # and the condition it established really does inhibit the release
    assert mission.offer("DeliveryCoordinateSatisfied", time=1.0) == ()


def test_an_offer_that_establishes_nothing_is_the_one_that_is_flagged():
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    mission.offer("SomethingNobodyModelled", time=0.0)

    assert mission.condition_only == []
    assert mission.unresolved and mission.unresolved[0][1] == "SomethingNobodyModelled"
    assert mission.conditions == {}


def test_an_event_that_says_something_else_does_not_match():
    """Matching is on the scenario's words being covered, not on overlap: a
    single motor failure is not a critical subsystem failure."""
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    assert mission.resolve_event("CriticalPropulsionFailure")[0] is None
    # ...and with the critical event declared, it resolves to that one only
    text = _RENAMED_EVENTS.replace(
        "transition releaseConditionMet",
        "transition criticalFailure\n                first Locked\n"
        "                accept CriticalPropulsionSubsystemFailure\n"
        "                then Releasing;\n\n            transition releaseConditionMet",
    )
    other = ModelDrivenMission(model_text=text)
    assert other.resolve_event("CriticalPropulsionFailure")[0] == (
        "CriticalPropulsionSubsystemFailure")
    assert other.resolve_event("SinglePropulsionFailure")[0] is None
