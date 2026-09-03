"""Behavioural-tier routing in the verification matrix.

An inhibition requirement ("shall not transition ... during the power-on
self-test") is anchored by the transition not taken, not by initial-state
semantics; one run routed it into the initialization branch on the word
"power-on" and failed a model whose anchor existed and whose scenarios passed.
A default-state requirement ("shall default to the locked state upon power-on")
stays on the initialization branch, where the power-on shape is driven onto its
default state.
"""
from __future__ import annotations

from src.prototyping.verification_matrix import build_matrix
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model


_MODEL = """package DeliveryUAV {
    item def SelfTestPassed;
    item def SensorFailure;
    item def PowerOnEvent;

    requirement def REQ_SAFE_004 {
        doc /* The system shall not transition to the armed or airborne state
               if any onboard sensor reports a failure during the power-on
               self-test sequence. */
    }
    requirement def REQ_SAFE_008 {
        doc /* The payload-release actuator shall default to the mechanically
               locked state upon power-on, before any arming. */
    }

    part def SafetyMonitor {
        satisfy requirement REQ_SAFE_004;
        action def performSelfTest {}
        action def preventArming {}
        state def SafetyMonitorBehavior {
            entry; then PowerOnSelfTest;
            state PowerOnSelfTest {
                do action performSelfTest;
            }
            state SafeState {
                entry action preventArming;
            }
            state Airborne;
            transition selfTestPassed
                first PowerOnSelfTest
                accept SelfTestPassed
                then Airborne;
            transition sensorFailureDetected
                first PowerOnSelfTest
                accept SensorFailure
                then SafeState;
        }
    }

    part def PayloadMechanism {
        satisfy requirement REQ_SAFE_008;
        action def initialize {}
        action def lockMechanism {}
        state def PayloadBehavior {
            entry; then PowerOn;
            state PowerOn {
                do action initialize;
            }
            state Locked {
                entry action lockMechanism;
            }
            transition powerOnComplete
                first PowerOn
                accept PowerOnEvent
                then Locked;
        }
    }

    part safetyMonitor : SafetyMonitor;
    part payloadMechanism : PayloadMechanism;
}"""


def _rows():
    lite = build_lite_model(_MODEL, model_name="DeliveryUAV")
    rows = build_matrix(lite, None, RequirementLinker(lite).compile_evidence())
    return {r.req_id: r for r in rows}


def test_inhibition_not_initialization():
    row = _rows()["REQ_SAFE_004"]
    assert "behavioral_sim_failed" not in row.tiers, row.evidence
    assert "behavioral_sim" in row.tiers


def test_default_state_driven_by_power():
    """Still routed to the initialization branch, with the power-on shape driven
    rather than failed: PowerOn --accept PowerOnEvent--> Locked{entry lockMechanism}
    models "default to locked upon power-on".

    The shapes that keep failing (escape edge from the initial state, guard-only
    exits, a bare landed state) are pinned in test_payload_lock_semantics.py.
    """
    row = _rows()["REQ_SAFE_008"]
    assert any(
        "initial/default-state invariant exercised" in e and "(PASS)" in e
        for e in row.evidence
    ), row.evidence
    assert "behavioral_sim_failed" not in row.tiers


def test_entry_lock_passes():
    model = _MODEL.replace(
        "state PowerOn {\n                do action initialize;\n            }",
        "state PowerOn {\n                entry action lockMechanism;\n"
        "            }",
    )
    lite = build_lite_model(model, model_name="DeliveryUAV")
    rows = build_matrix(lite, None, RequirementLinker(lite).compile_evidence())
    row = {r.req_id: r for r in rows}["REQ_SAFE_008"]
    assert "behavioral_sim" in row.tiers, row.evidence


def test_shared_inhibition_predicate():
    """One shared inhibition predicate serves the matrix and the verification audit.

    Their keyword lists once diverged, so a phrasing in the difference met
    different evidence standards.
    """
    from src.prototyping.verification_obligations import (
        is_inhibition_requirement,
    )
    for text in (
        "The system shall not transition to the armed state during self-test.",
        "The interlock shall block arming while the hatch is open.",
        "The controller shall prevent motor start below minimum voltage.",
        "The actuator shall inhibit release during transport.",
        "A lockout shall apply until the self-test completes.",
    ):
        assert is_inhibition_requirement(text), text
    for text in (
        "The actuator shall default to the locked state upon power-on.",
        "The system shall report status every second.",
    ):
        assert not is_inhibition_requirement(text), text


def test_block_phrasing_not_routed():
    model = _MODEL.replace(
        "doc /* The system shall not transition to the armed or airborne state\n"
        "               if any onboard sensor reports a failure during the power-on\n"
        "               self-test sequence. */",
        "doc /* The system shall block the transition to the armed state if a\n"
        "               sensor fails during the power-on self-test sequence. */",
    )
    lite = build_lite_model(model, model_name="DeliveryUAV")
    rows = build_matrix(lite, None, RequirementLinker(lite).compile_evidence())
    row = {r.req_id: r for r in rows}["REQ_SAFE_004"]
    assert "behavioral_sim_failed" not in row.tiers, row.evidence


def test_do_action_holds_default():
    """run3's shape (A3, SAFE_008): the initial state is the required default and a
    do-action sustains it - `Locked { do lockMechanism; }`.

    The 'behavioral FAIL vs L2 servo PASS' contradiction came from crediting entry
    actions but not sustained holding.
    """
    model = _MODEL.replace(
        """state PowerOn {
                do action initialize;
            }""",
        """state Locked {
                do action lockMechanism;
            }""",
    ).replace("entry; then PowerOn;", "entry; then Locked;").replace(
        """state Locked {
                entry action lockMechanism;
            }
            transition powerOnComplete
                first PowerOn
                accept PowerOnEvent
                then Locked;""",
        """state Released;
            transition release
                first Locked
                accept PowerOnEvent
                then Released;""",
    )
    lite = build_lite_model(model, model_name="DeliveryUAV")
    rows = {r.req_id: r for r in build_matrix(
        lite, None, RequirementLinker(lite).compile_evidence()
    )}
    row = rows["REQ_SAFE_008"]
    assert "behavioral_sim" in row.tiers, row.evidence
    assert "behavioral_sim_failed" not in row.tiers
