from __future__ import annotations

from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.safety_patterns import select_patterns
from src.prototyping.semantic_trace import FAIL, PASS, build_semantic_trace
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model


def _model(command: str = "CMD_PARACHUTE", with_timing: bool = True) -> str:
    timing = """
        attribute maxParachuteLatency : Real = 0.5 [s];
        attribute currentParachuteLatency : Real = 0.0 [s];
        assert constraint parachuteLatencyBound {
            currentParachuteLatency <= maxParachuteLatency
        }
    """ if with_timing else ""
    return f"""package D {{
        action def {command} {{ }}
        requirement def REQ_SAFE_005 {{
            doc /* Critical propulsion failure shall deploy the parachute within 0.5 seconds. */
        }}
        part def SafetyMonitor {{
            out port chuteCmd : DataPort;
            attribute propulsionCriticalFailure : Boolean = false;
            {timing}
            satisfy requirement REQ_SAFE_005;
            action def deployParachute {{ send {command}() to chuteCmd; }}
            state def PropulsionMonitor {{
                state nominal;
                state chute {{ entry action deploy : deployParachute; }}
                transition initial then nominal;
                transition failed first nominal if propulsionCriticalFailure then chute;
            }}
        }}
    }}"""


def _trace(model_text: str):
    bundle = build_contract_bundle([
        "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
        "within 0.5 seconds."
    ])
    return build_semantic_trace(
        model_text, bundle, pattern_bindings=select_patterns(bundle), model_name="D"
    ).traces[0]


def test_complete_parachute_trace_passes():
    trace = _trace(_model())
    assert trace.conformance == PASS
    assert not trace.findings


def test_official_mavlink_parachute_command_is_accepted():
    trace = _trace(_model(command="MAV_CMD_DO_PARACHUTE"))

    assert trace.conformance == PASS
    assert not trace.findings


def test_wrong_platform_command_is_localized():
    trace = _trace(_model(command="CMD_LAND"))
    assert trace.conformance == FAIL
    assert "ACTION_PLATFORM_BINDING_MISMATCH" in {
        item.finding_code for item in trace.findings
    }


def test_missing_timing_constraint_is_not_inferred_from_requirement_text():
    trace = _trace(_model(with_timing=False))
    assert trace.conformance == FAIL
    assert "MISSING_TIMING_CONSTRAINT" in {
        item.finding_code for item in trace.findings
    }


def test_timing_constraint_for_another_response_is_not_reused():
    unrelated = """
        attribute maxWaypointUpdateLatency : Real = 0.5 [s];
        attribute currentWaypointUpdateLatency : Real = 0.0 [s];
        assert constraint waypointUpdateLatencyBound {
            currentWaypointUpdateLatency <= maxWaypointUpdateLatency
        }
    """
    model = _model(with_timing=False).replace(
        "attribute propulsionCriticalFailure : Boolean = false;",
        "attribute propulsionCriticalFailure : Boolean = false;" + unrelated,
    )

    trace = _trace(model)

    assert trace.conformance == FAIL
    assert "MISSING_TIMING_CONSTRAINT" in {
        item.finding_code for item in trace.findings
    }


def test_contract_first_linker_blocks_model_derived_test_when_trace_is_incomplete():
    bundle = build_contract_bundle([
        "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
        "within 0.5 seconds."
    ])
    model_text = _model(with_timing=False)
    linker = RequirementLinker(
        build_lite_model(model_text, model_name="D"), contract_bundle=bundle
    )
    spec = {item.req_id: item for item in linker.generate_test_specs()}["REQ_SAFE_005"]

    assert spec.tier == "TRACE"
    assert spec.inject.kind == "skip"
    assert "MISSING_TIMING_CONSTRAINT" in spec.notes


def test_single_motor_design_invariant_uses_oracle_trace_not_state_machine():
    requirement = (
        "REQ-SAFE-007: The system shall maintain controlled flight following "
        "the failure of any single propulsion motor."
    )
    bundle = build_contract_bundle([requirement])
    model = """package D {
        requirement def REQ_SAFE_007 {
            doc /* maintain controlled flight after one motor fails */
        }
        part def PropulsionSystem {
            satisfy requirement REQ_SAFE_007;
        }
    }"""

    trace = build_semantic_trace(model, bundle, model_name="D").traces[0]

    assert trace.conformance == PASS
    assert {link.link_kind for link in trace.links} == {
        "requirement_to_owner", "contract_to_oracle",
    }


def test_power_on_locked_default_is_traced_through_initial_state():
    requirement = (
        "REQ-SAFE-008: The payload-release actuator shall default to the "
        "mechanically locked state upon power-on."
    )
    bundle = build_contract_bundle([requirement])
    model = """package D {
        requirement def REQ_SAFE_008 { doc /* locked by default */ }
        part def PayloadMechanism {
            satisfy requirement REQ_SAFE_008;
            state def PayloadState {
                state mechanicallyLocked;
                state released;
                transition initial then mechanicallyLocked;
            }
        }
    }"""

    trace = build_semantic_trace(
        model, bundle, pattern_bindings=select_patterns(bundle), model_name="D"
    ).traces[0]

    assert trace.conformance == PASS
    assert "owner_to_initial_safe_state" in {link.link_kind for link in trace.links}


def test_startup_failure_path_to_armed_state_violates_pattern_invariant():
    requirement = (
        "REQ-SAFE-004: The system shall not transition to the armed state if an "
        "onboard sensor reports a failure during the power-on self-test."
    )
    bundle = build_contract_bundle([requirement])
    model = """package D {
        requirement def REQ_SAFE_004 { doc /* failed self-test inhibits arming */ }
        part def FlightController {
            attribute sensorSelfTestFailed : Boolean = false;
            satisfy requirement REQ_SAFE_004;
            action def inhibitArming { }
            state def StartupMachine {
                state checking;
                state inhibited { entry action inhibit : inhibitArming; }
                state armed;
                transition initial then checking;
                transition fail first checking if sensorSelfTestFailed then inhibited;
                transition unsafe first inhibited then armed;
            }
        }
    }"""

    trace = build_semantic_trace(
        model, bundle, pattern_bindings=select_patterns(bundle), model_name="D"
    ).traces[0]

    assert trace.conformance == FAIL
    assert "INVARIANT_NOT_REPRESENTED" in {
        item.finding_code for item in trace.findings
    }


def test_timed_payload_actuation_has_contract_first_trace_without_sitl_invention():
    requirement = (
        "REQ-PERF-005: The mechanical payload release actuation shall complete "
        "within 2.0 seconds from the moment the delivery coordinate condition "
        "is satisfied."
    )
    bundle = build_contract_bundle([requirement])
    model = """package D {
        action def CMD_GRIPPER_RELEASE { }
        requirement def REQ_PERF_005 { doc /* timed release actuation */ }
        part def PayloadMechanism {
            out port releaseCmd : DataPort;
            attribute maxPayloadReleaseLatency : Real = 2.0 [s];
            attribute currentPayloadReleaseLatency : Real = 0.0 [s];
            satisfy requirement REQ_PERF_005;
            action def releasePayload {
                send CMD_GRIPPER_RELEASE() to releaseCmd;
            }
            assert constraint payloadReleaseLatencyBound {
                currentPayloadReleaseLatency <= maxPayloadReleaseLatency
            }
            state def ReleaseMachine {
                state waiting;
                state released { entry action release : releasePayload; }
                transition initial then waiting;
                transition releaseOnCoordinate first waiting
                    accept DeliveryCoordinateConditionSatisfied
                    then released;
            }
        }
    }"""

    trace = build_semantic_trace(model, bundle, model_name="D").traces[0]

    assert trace.conformance == PASS
    assert not trace.findings


def _abort_lock_trace(release_guard: str, bundle=None):
    requirement = (
        "REQ-SAFE-006: The system shall maintain the payload in the "
        "mechanically locked state whenever a delivery-abort condition is "
        "active, regardless of geographic proximity to the delivery waypoint."
    )
    if bundle is None:
        bundle = build_contract_bundle([requirement])
    model = f"""package D {{
        requirement def REQ_SAFE_006 {{ doc /* abort retains payload */ }}
        part def PayloadSystem {{
            attribute deliveryAbortConditionActive : Boolean = false;
            attribute releaseAuthorized : Boolean = false;
            satisfy requirement REQ_SAFE_006;
            action def lockPayload {{ }}
            action def releasePayload {{ }}
            state def PayloadReleaseBehavior {{
                state PayloadLocked {{ entry action lock : lockPayload; }}
                state PayloadReleased {{ entry action release : releasePayload; }}
                transition initial then PayloadLocked;
                transition release first PayloadLocked
                    {release_guard}
                    then PayloadReleased;
            }}
        }}
    }}"""
    return build_semantic_trace(
        model, bundle, pattern_bindings=select_patterns(bundle), model_name="D"
    ).traces[0]


def test_abort_lock_invariant_accepts_release_inhibition_without_fake_lock_edge():
    trace = _abort_lock_trace("if not deliveryAbortConditionActive")

    assert trace.conformance == PASS
    assert not trace.findings
    assert "release_gate_invariant" in {link.link_kind for link in trace.links}


def test_abort_lock_invariant_rejects_release_edge_without_abort_inhibition():
    trace = _abort_lock_trace("accept DeliveryWaypointReached")

    assert trace.conformance == FAIL
    codes = {item.finding_code for item in trace.findings}
    assert "GUARD_SEMANTIC_MISMATCH" in codes
    # The oracle link fails as a consequence of the invariant failure; that
    # must not be double-reported as a second, independent oracle finding.
    assert "ORACLE_SEMANTIC_MISMATCH" not in codes


def test_abort_lock_invariant_rejects_or_guard_that_can_bypass_abort_inhibition():
    trace = _abort_lock_trace(
        "if not deliveryAbortConditionActive or releaseAuthorized"
    )

    assert trace.conformance == FAIL
    assert "GUARD_SEMANTIC_MISMATCH" in {
        item.finding_code for item in trace.findings
    }


def test_abort_lock_invariant_accepts_conjunctive_release_authorisation():
    trace = _abort_lock_trace(
        "if releaseAuthorized and not deliveryAbortConditionActive"
    )

    assert trace.conformance == PASS
    assert not trace.findings


def _with_observation(bundle, observation_concept: str):
    """Return a copy of a one-contract bundle with a divergent oracle concept."""
    from dataclasses import replace

    contract = bundle.contracts[0]
    obligation = contract.obligations[0]
    patched = replace(
        obligation,
        verification_intent=replace(
            obligation.verification_intent,
            observation_concept=observation_concept,
        ),
    )
    return replace(
        bundle, contracts=(replace(contract, obligations=(patched,)),)
    )


def test_oracle_observation_mismatch_is_a_semantic_finding():
    bundle = _with_observation(
        build_contract_bundle([
            "REQ-SAFE-005: Critical propulsion failure shall deploy the "
            "parachute within 0.5 seconds."
        ]),
        "chute_light_observed",
    )

    trace = build_semantic_trace(_model(), bundle, model_name="D").traces[0]

    assert trace.conformance == FAIL
    assert "ORACLE_SEMANTIC_MISMATCH" in {
        item.finding_code for item in trace.findings
    }
    oracle = next(
        link for link in trace.links if link.link_kind == "action_to_oracle"
    )
    assert oracle.status == FAIL


def test_locked_default_oracle_mismatch_is_a_semantic_finding():
    bundle = _with_observation(
        build_contract_bundle([
            "REQ-SAFE-008: The payload-release actuator shall default to the "
            "mechanically locked state upon power-on."
        ]),
        "gripper_led_observed",
    )
    model = """package D {
        requirement def REQ_SAFE_008 { doc /* locked by default */ }
        part def PayloadMechanism {
            satisfy requirement REQ_SAFE_008;
            state def PayloadState {
                state mechanicallyLocked;
                state released;
                transition initial then mechanicallyLocked;
            }
        }
    }"""

    trace = build_semantic_trace(model, bundle, model_name="D").traces[0]

    assert trace.conformance == FAIL
    codes = {item.finding_code for item in trace.findings}
    assert "ORACLE_SEMANTIC_MISMATCH" in codes
    assert "INVARIANT_NOT_REPRESENTED" not in codes
    initial = next(
        link for link in trace.links
        if link.link_kind == "owner_to_initial_safe_state"
    )
    assert initial.status == PASS


def test_abort_lock_oracle_mismatch_is_reported_without_invariant_findings():
    requirement = (
        "REQ-SAFE-006: The system shall maintain the payload in the "
        "mechanically locked state whenever a delivery-abort condition is "
        "active, regardless of geographic proximity to the delivery waypoint."
    )
    bundle = _with_observation(
        build_contract_bundle([requirement]), "lock_led_observed"
    )

    trace = _abort_lock_trace("if not deliveryAbortConditionActive", bundle=bundle)

    assert trace.conformance == FAIL
    codes = {item.finding_code for item in trace.findings}
    assert codes == {"ORACLE_SEMANTIC_MISMATCH"}
    oracle = next(
        link for link in trace.links if link.link_kind == "invariant_to_oracle"
    )
    assert oracle.status == FAIL


def test_unnamed_transition_with_structured_abort_guard_passes_qualifiers():
    bundle = build_contract_bundle([
        "REQ-FUNC-005: The system shall release the payload when the current "
        "geographic position is within 1.0 metre of the designated delivery "
        "waypoint and no delivery-abort condition is active."
    ])
    model = """package D {
        action def CMD_GRIPPER_RELEASE { }
        requirement def REQ_FUNC_005 { doc /* qualified release */ }
        part def PayloadMechanism {
            out port releaseCmd : DataPort;
            attribute waypointDistance : Real = 5.0 [m];
            attribute deliveryAbortConditionActive : Boolean = false;
            satisfy requirement REQ_FUNC_005;
            action def releasePayload { send CMD_GRIPPER_RELEASE() to releaseCmd; }
            state def ReleaseBehavior {
                state holding;
                state released { entry action release : releasePayload; }
                transition initial then holding;
                transition first holding
                    if waypointDistance <= 1.0 and not deliveryAbortConditionActive
                    then released;
            }
        }
    }"""

    trace = build_semantic_trace(model, bundle, model_name="D").traces[0]

    assert trace.conformance == PASS
    assert not trace.findings
