"""The generated model owns the decision; the harness actuates it.

Every payload/parachute result in the 2026-08-30 authoritative run carried the
caveat "the coordinate condition was evaluated by the harness, not by generated
mission logic". These tests hold that boundary and pin the arbitration failure
driving the model exposed.
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


def test_model_fires_transition_and_action():
    mission = ModelDrivenMission(_MODEL)
    assert mission.state_of("PayloadMechanism.PayloadReleaseBehavior") == "Locked"

    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert [d.action for d in fired] == ["onReleasing"]
    assert [d.action_definition for d in fired] == ["actuateRelease"]
    assert mission.performed(fired, ModelAction.RELEASE_PAYLOAD)
    assert fired[0].from_state == "Locked" and fired[0].to_state == "Releasing"
    assert fired[0].as_dict()["decided_by"] == "generated model"
    assert mission.state_of("PayloadMechanism.PayloadReleaseBehavior") == "Releasing"


def test_renamed_release_matched_by_role():
    """run3 names its action ``releasePayload`` where the adapter constant says
    ``actuateRelease``.

    The literal comparison refused to actuate, so the payload never separated and
    the positional/timed checks had no evidence: a harness spelling reported as
    "the model declined". One distinct fired action sharing an actuation term is
    the model's response, and the rename goes on the record.
    """
    model = _MODEL.replace(
        "entry action onReleasing : actuateRelease;",
        "entry action onReleasing : releasePayload;",
    ).replace("action def actuateRelease {}", "action def releasePayload {}")
    mission = ModelDrivenMission(model)

    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert [d.action_definition for d in fired] == ["releasePayload"]
    assert mission.performed(fired, ModelAction.RELEASE_PAYLOAD)
    assert mission.action_resolutions == [("actuateRelease", "releasePayload")]


def test_two_fired_actions_refused():
    """resolve_event's tie discipline, applied to actions: when the event fires two
    different action definitions, picking one would decide which behaviour the run
    exercised.
    """
    model = _MODEL.replace(
        "entry action onReleasing : actuateRelease;",
        "entry action onReleasing : releasePayload;",
    ).replace(
        "action def actuateRelease {}",
        "action def releasePayload {}\n        action def signalRelease {}",
    ).replace(
        "entry action onAborted : lockPayload;",
        "entry action onAborted : signalRelease;",
    ).replace(
        "accept AbortConditionActive",
        "accept DeliveryCoordinateSatisfied",
    )
    mission = ModelDrivenMission(model)

    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert len({d.action_definition for d in fired}) == 2
    assert not mission.performed(fired, ModelAction.RELEASE_PAYLOAD)
    assert mission.action_resolutions == []


def test_unrelated_action_not_release():
    model = _MODEL.replace(
        "entry action onReleasing : actuateRelease;",
        "entry action onReleasing : lockPayload;",
    )
    mission = ModelDrivenMission(model)

    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert [decision.action for decision in fired] == ["onReleasing"]
    assert [decision.action_definition for decision in fired] == ["lockPayload"]
    assert not mission.performed(fired, ModelAction.RELEASE_PAYLOAD)


def test_ignored_event_fires_nothing():
    mission = ModelDrivenMission(_MODEL)
    assert mission.offer("SomeEventTheModelNeverDeclared", time=1.0) == ()
    assert mission.decisions == []
    mission.offer("DeliveryCoordinateSatisfied", time=1.0)
    assert mission.offer("DeliveryCoordinateSatisfied", time=2.0) == ()


def test_accepted_events_from_model():
    mission = ModelDrivenMission(_MODEL)
    assert mission.accepted_events() == (
        "AbortConditionActive", "DeliveryCoordinateSatisfied",
    )
    assert mission.handles("AbortConditionActive")
    assert not mission.handles("CriticalPropulsionFailure")


def test_actions_found_per_machine():
    mission = ModelDrivenMission(_MODEL)

    assert mission.action_definitions_for_machine("DeliveryAbortBehavior") == (
        "lockPayload",
    )


def test_provenance_names_machines():
    mission = ModelDrivenMission(_MODEL)
    prov = mission.provenance()
    assert len(prov["model_sha256"]) == 64
    assert "PayloadMechanism.PayloadReleaseBehavior" in prov["machines"]
    assert prov["decision_owner"] == "generated model state machines"


def test_unguarded_release_not_inhibited():
    """REQ-SAFE-006 shape: locked "whenever a delivery-abort condition is active,
    regardless of proximity".

    The two behaviours are independent machines and the release transition carries
    no guard, so the abort cannot inhibit it. Only driving the model exposes this;
    structural checks see two well-formed state machines.
    """
    mission = ModelDrivenMission(_MODEL)

    mission.offer("AbortConditionActive", time=1.0)
    assert mission.state_of("PayloadMechanism.DeliveryAbortBehavior") == "Aborted"

    fired = mission.offer("DeliveryCoordinateSatisfied", time=2.0)

    assert [d.action for d in fired] == ["onReleasing"], (
        "if this ever stops firing, the generated model gained the arbitration "
        "REQ-SAFE-006 asks for and the inhibition scenario should be re-judged"
    )


_AUTHORITATIVE = Path("examples/output/latest/final_model.sysml")


@pytest.mark.skipif(not _AUTHORITATIVE.exists(), reason="no authoritative run on disk")
def test_authoritative_model_drivable():
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


def test_guard_only_machines_drivable():
    """A machine whose transitions are all guards is still driven: offer() steps every
    machine and a guard fires on the variables it is given.

    Filtering them out hid SafetyArbiter, which expresses REQ-SAFE-005's precedence
    entirely in guards, so the precedence check had nothing to take precedence over
    and returned inconclusive.
    """
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


def test_unbound_guard_flag_listed():
    """With no variables bound, `not deliveryAbortActive` still fired the release.

    An unset flag reads false, so adding the guard changes nothing the harness can
    see: the fix would look applied while the requirement kept failing, blaming the
    model.
    """
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    assert mission.boolean_guard_attributes() == ("deliveryAbortActive",)
    assert mission.unlatched_boolean_attributes() == ("deliveryAbortActive",)


def test_abort_event_raises_model_flag():
    """The flag is found by matching the offered event's words against names the model
    declared, so a model that names it differently is still checkable and the
    harness holds no requirement's spellings.
    """
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)

    assert mission.conditions == {"deliveryAbortActive": True}
    assert mission.latched == [(0.0, "AbortConditionActive", "deliveryAbortActive")]
    assert mission.unlatched_boolean_attributes() == ()


def test_active_abort_inhibits_release():
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)
    fired = mission.offer("DeliveryCoordinateSatisfied", time=1.0)

    assert fired == ()


def test_ordinary_delivery_releases():
    """The guard does not inhibit the ordinary case.

    It did briefly: matching the event against the flag on any shared word raised
    the abort flag from the delivery event itself, because both names carry
    "delivery".
    """
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    fired = mission.offer("DeliveryCoordinateSatisfied", time=0.0)

    assert [d.action for d in fired] == ["onReleasing"]
    assert mission.conditions == {}


def test_condition_stays_raised():
    mission = ModelDrivenMission(model_text=_GUARDED_RELEASE)
    mission.offer("AbortConditionActive", time=0.0)
    mission.offer("SomethingElse", time=1.0)

    assert mission.conditions == {"deliveryAbortActive": True}
    assert mission.offer("DeliveryCoordinateSatisfied", time=2.0) == ()


def test_caller_value_overrides_latched():
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


def test_renamed_event_resolves_and_fires():
    """run3's model accepts `DeliveryCoordinateConditionSatisfied` where the harness
    offered `DeliveryCoordinateSatisfied`. Nothing fired, and the evidence would
    have read "the generated logic declined to release", a harness spelling
    reported as a model defect.
    """
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)

    resolved, how = mission.resolve_event("DeliveryCoordinateSatisfied")
    assert resolved == "DeliveryCoordinateConditionSatisfied"
    assert "semantic match" in how

    fired = mission.offer("DeliveryCoordinateSatisfied", time=0.0)
    assert [d.action for d in fired] == ["onReleasing"]
    assert mission.resolutions[0][1:3] == (
        "DeliveryCoordinateSatisfied", "DeliveryCoordinateConditionSatisfied")


def test_declared_name_used_verbatim():
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    resolved, how = mission.resolve_event("DeliveryCoordinateConditionSatisfied")
    assert resolved == "DeliveryCoordinateConditionSatisfied"
    assert how == "declared verbatim"
    mission.offer("DeliveryCoordinateConditionSatisfied", time=0.0)
    assert mission.resolutions == []


def test_undeclared_event_recorded():
    """"Never asked" and "asked and declined" are different findings, so a scenario
    resting on an unresolved event is recorded as unresolved, not as a refusal.
    """
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    resolved, why = mission.resolve_event("AbortConditionActive")

    assert resolved is None
    assert "no declared event covers" in why


def test_flag_only_event_not_unresolved():
    """run3 has no abort event; the delivery abort is a standing boolean its guards
    read.

    "No event matched" would read as "the model was never told", when the latched
    flag told it. Only an offer that resolves to nothing and raises nothing
    established nothing.
    """
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    mission.offer("AbortConditionActive", time=0.0)

    assert mission.unresolved == []
    assert mission.condition_only == [
        (0.0, "AbortConditionActive", ("deliveryAbortConditionActive",))]
    assert mission.conditions == {"deliveryAbortConditionActive": True}
    assert mission.offer("DeliveryCoordinateSatisfied", time=1.0) == ()


def test_empty_offer_flagged_unresolved():
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    mission.offer("SomethingNobodyModelled", time=0.0)

    assert mission.condition_only == []
    assert mission.unresolved and mission.unresolved[0][1] == "SomethingNobodyModelled"
    assert mission.conditions == {}


def test_different_event_not_matched():
    mission = ModelDrivenMission(model_text=_RENAMED_EVENTS)
    assert mission.resolve_event("CriticalPropulsionFailure")[0] is None
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


def test_guards_for_event_by_role():
    """The inhibition verdict needs no action names: the guards on the transitions
    answering the resolved delivery event are the release guards. run3's model was
    reported as 'carries no guard' because the harness asked for its own spelling
    'actuateRelease'.
    """
    from pathlib import Path
    text = (
        Path(__file__).parent.parent / "examples" / "output"
        / "run3_authoritative_20260831" / "final_model.sysml"
    ).read_text()
    mission = ModelDrivenMission(model_text=text)

    guards = mission.guards_for_event("DeliveryCoordinateSatisfied")
    assert guards is not None
    assert any("deliveryAbortConditionActive" in g for g in guards)
    # The old spelling-bound lookup still sees nothing; that produced the false
    # 'no inhibition logic' verdict.
    assert mission.guards_reaching_action("actuateRelease") == ()


def test_guards_for_event_none_unresolved():
    from pathlib import Path
    text = (
        Path(__file__).parent.parent / "examples" / "output"
        / "run3_authoritative_20260831" / "final_model.sysml"
    ).read_text()
    mission = ModelDrivenMission(model_text=text)
    assert mission.guards_for_event("WarpDriveEngaged") is None
