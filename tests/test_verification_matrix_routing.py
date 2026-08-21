"""The behavioural-tier routing in the verification matrix.

An inhibition requirement ("shall not transition ... during the power-on
self-test") is anchored by the transition that is NOT taken, never by
initial-state semantics. One measured run routed such a requirement into the
initialization branch because its text contains "power-on", then failed a
model whose inhibition anchor existed and whose scenarios all passed. A
default-state requirement ("shall default to the locked state upon
power-on") stays on the initialization branch, and a machine that only
reaches the locked state after an event keeps failing it."""
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


def test_inhibition_requirement_is_not_routed_to_the_initialization_branch():
    row = _rows()["REQ_SAFE_004"]
    assert "behavioral_sim_failed" not in row.tiers, row.evidence
    assert "behavioral_sim" in row.tiers


def test_default_state_requirement_still_fails_without_initial_semantics():
    row = _rows()["REQ_SAFE_008"]
    assert "behavioral_sim_failed" in row.tiers, row.evidence


def test_default_state_requirement_passes_with_entry_lock_on_initial_state():
    model = _MODEL.replace(
        "state PowerOn {\n                do action initialize;\n            }",
        "state PowerOn {\n                entry action lockMechanism;\n"
        "            }",
    )
    lite = build_lite_model(model, model_name="DeliveryUAV")
    rows = build_matrix(lite, None, RequirementLinker(lite).compile_evidence())
    row = {r.req_id: r for r in rows}["REQ_SAFE_008"]
    assert "behavioral_sim" in row.tiers, row.evidence
