"""Emergency-branch classification matches identifier words, not substrings.

Regression from the 3-seed ablation (FULL seed 1, 2026-09-02): an entry action
named ``defaultToMechanicallyLockedState`` contains "fault", so the power-on
transition into that state was filed as an emergency branch, the nominal chain
had no first step ("No nominal transitions found"), and a correct model's
REQ_FUNC_005 scenario failed with the run.
"""
from src.simulation.behavioral_sim import (
    _build_nominal_multigraph,
    _classify_accept_transitions,
    _identifier_words,
    _longest_nominal_path,
)
from src.simulation.state_extractor import extract_state_machines


PAYLOAD_MODEL = """
package Drone {
    item def PowerOn;
    item def ReleaseCommand;
    part def PayloadSystem {
        attribute deliveryAbortConditionActive : Boolean = false;
        action def actuateRelease {}
        action def defaultToMechanicallyLockedState {}
        state def PayloadBehavior {
            entry; then Off;
            state Off;
            state Locked {
                entry action onLocked : defaultToMechanicallyLockedState;
            }
            state Released {
                entry action onReleased : actuateRelease;
            }
            transition powerOn
                first Off
                accept PowerOn
                then Locked;
            transition releasePayload
                first Locked
                accept ReleaseCommand
                if not deliveryAbortConditionActive
                then Released;
        }
    }
}
"""


def _payload_machine(text):
    machines = [m for m in extract_state_machines(text) if m.name == "PayloadBehavior"]
    assert len(machines) == 1
    return machines[0]


def test_identifier_words_split_case():
    assert _identifier_words("defaultToMechanicallyLockedState") == [
        "default", "to", "mechanically", "locked", "state",
    ]
    assert _identifier_words("initiate_fault_response RtbDetected") == [
        "initiate", "fault", "response", "rtb", "detected",
    ]


def test_default_not_emergency():
    sm = _payload_machine(PAYLOAD_MODEL)
    nominal, emergency = _classify_accept_transitions(sm)
    assert [t.name for t in nominal] == ["powerOn", "releasePayload"]
    assert emergency == []
    path = _longest_nominal_path(_build_nominal_multigraph(nominal), sm.initial_state)
    assert path == [("PowerOn", "Locked"), ("ReleaseCommand", "Released")]


def test_fault_words_emergency():
    text = PAYLOAD_MODEL.replace(
        "defaultToMechanicallyLockedState", "handleFaultsAndLock"
    )
    sm = _payload_machine(text)
    nominal, emergency = _classify_accept_transitions(sm)
    assert [t.name for t in emergency] == ["powerOn"]
    assert [t.name for t in nominal] == ["releasePayload"]
