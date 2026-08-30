"""Coverage-extension regression anchors.

Accept-event L2 mapping (PAYLOAD_ABORT_LOCK / PAYLOAD_POWERON_LOCK), port-surface
static L1 mapping (RTCM_GPS / MAVLINK_PROTOCOL), the secondary fence L2 spec, and
the text gates that keep release-type requirements from claiming lock guards.

The fixture mirrors the authoritative bundle 5af6c666's relevant structure
(accept-driven PayloadMechanism machines; port-only interface parts) so the
matching behaviour proven here is the one offline re-verification relies on —
without depending on the untracked archive itself.
"""
from __future__ import annotations

from src.prototyping.verification_matrix import build_matrix
from src.sitl.requirement_linker import RequirementLinker
from src.sitl.sitl_catalogue import _CONTENT_CATALOGUE, AttrMatcher, ContentEntry
from src.sysml.lite_model import build_lite_model

_FIXTURE = """package DroneSystem {
    port def GnssCorrectionsPort;
    port def MavlinkTelemetryPort;

    requirement def REQ_FUNC_005 { doc /* The system shall release the payload when the current geographic position is within 1.0 metre of the designated delivery waypoint and no delivery-abort condition is active. */ }
    requirement def REQ_PERF_005 { doc /* The mechanical payload release actuation shall complete within 2.0 seconds from the moment the delivery coordinate condition is satisfied. */ }
    requirement def REQ_SAFE_006 { doc /* The system shall maintain the payload in the mechanically locked state whenever a delivery-abort condition is active, regardless of geographic proximity to the delivery waypoint. */ }
    requirement def REQ_SAFE_008 { doc /* The payload-release actuator shall default to the mechanically locked state upon power-on, before any arming or flight authorisation. */ }
    requirement def REQ_INTF_001 { doc /* The system shall exchange telemetry and mission commands with the GCS using the MAVLink v2.0 protocol over an AES-256 encrypted RF channel. */ }
    requirement def REQ_INTF_002 { doc /* The system shall receive and apply differential GNSS corrections formatted according to the RTCM 10403.3 standard to achieve sub-metre positioning accuracy. [V: hardware-in-the-loop / RTK bench] */ }
    requirement def REQ_CONS_001 { doc /* The system shall not exceed a flight altitude of 120 metres above ground level at any point during normal operations. */ }

    part def PerceptionSystem {
        in port gnssCorrections : GnssCorrectionsPort;
        satisfy requirement REQ_INTF_002;
    }

    part def CommunicationSystem {
        in port mavlinkTelemetry : MavlinkTelemetryPort;
        satisfy requirement REQ_INTF_001;
    }

    part def FlightController {
        attribute maxFlightAltitude : Real = 120.0;
        satisfy requirement REQ_CONS_001;
    }

    part def PayloadMechanism {
        attribute isLocked : Boolean = true;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_PERF_005;
        satisfy requirement REQ_SAFE_006;
        satisfy requirement REQ_SAFE_008;
        action def actuateRelease {}
        action def defaultToLocked {}
        action def lockPayload {}
        state def PayloadReleaseBehavior {
            entry; then Locked;
            state Locked;
            state Releasing { entry action actuateRelease; }
            transition toReleasing
                first Locked
                accept DeliveryCoordinateSatisfied
                then Releasing;
        }
        state def DeliveryAbortBehavior {
            entry; then Monitoring;
            state Monitoring;
            state Aborted { entry action lockPayload; }
            transition toAborted
                first Monitoring
                accept AbortConditionActive
                then Aborted;
        }
        state def PowerOnBehavior {
            entry; then Off;
            state Off;
            state PoweredOnLocked { entry action defaultToLocked; }
            transition toPoweredOnLocked
                first Off
                accept PowerOnEvent
                then PoweredOnLocked;
        }
    }
}"""


def _model():
    return build_lite_model(_FIXTURE, model_name="DroneSystem")


def _evidence():
    return RequirementLinker(_model(), llm=None).compile_evidence()


def test_accept_event_machines_map_abort_and_poweron_locks():
    ev = _evidence()
    specs = {(s.req_id, s.tier): s for s in ev.test_specs}

    abort = specs[("REQ_SAFE_006", "L2")]
    assert abort.inject.kind == "mavlink_command"
    assert abort.verify.kind == "assert_servo_pwm"
    assert abort.verify.args["channel"] == 7
    assert abort.verify.args["target_pwm"] == 1000
    ga = ev.guard_assignments["REQ_SAFE_006"]
    assert ga.kind == "accept_event"
    assert ga.attribute == "AbortConditionActive"

    poweron = specs[("REQ_SAFE_008", "L2")]
    assert poweron.inject.kind == "set_param"
    assert poweron.inject.pre_takeoff_m == 0.0  # power-on, before any arming
    assert poweron.verify.kind == "assert_servo_pwm"
    assert poweron.verify.args["channel"] == 7
    assert poweron.verify.args["target_pwm"] == 1000
    ga8 = ev.guard_assignments["REQ_SAFE_008"]
    assert ga8.kind == "accept_event"
    assert ga8.attribute == "PowerOnEvent"


def test_release_requirements_do_not_claim_lock_guards():
    """FUNC_005/PERF_005 share the PAYLOAD family and the same satisfying part;
    the guard-side text gates must keep them off the lock-semantics guards."""
    ev = _evidence()
    ids = {s.req_id for s in ev.test_specs}
    assert "REQ_FUNC_005" not in ids
    assert "REQ_PERF_005" not in ids
    assert "REQ_FUNC_005" not in ev.guard_assignments
    assert "REQ_PERF_005" not in ev.guard_assignments


def test_poweron_lock_params_reach_the_boot_parm():
    ev = _evidence()
    parm = ev.parm_file
    # AP_Gripper reads GRIP_ENABLE at init: boot defaults are the only reliable
    # path, so the linker parm must carry the gripper config.
    for token in ("GRIP_ENABLE", "GRIP_NEUTRAL", "SERVO7_FUNCTION"):
        assert token in parm, parm


def test_port_only_interface_parts_get_static_config_l1():
    ev = _evidence()
    specs = {(s.req_id, s.tier): s for s in ev.test_specs}

    rtcm = specs[("REQ_INTF_002", "L1")]
    assert [p.param_name for p in rtcm.params] == ["GPS_INJECT_TO"]
    assert all(p.source == "static" for p in rtcm.params)

    mav = specs[("REQ_INTF_001", "L1")]
    assert [p.param_name for p in mav.params] == ["SERIAL0_PROTOCOL"]
    assert all(p.source == "static" for p in mav.params)


def test_port_match_refuses_entries_with_dynamic_params():
    """A valueless port name can never resolve @attr_match: an entry that needs
    a model-derived number must not fire from the port surface even when port
    matching is enabled for it."""
    probe = ContentEntry(
        semantic_tag="PORT_DYNAMIC_PROBE",
        attr_matcher=AttrMatcher(
            attr_keywords=["gnss"],
            part_keywords=["perception"],
            allow_port_match=True,
        ),
        ardu_params={"GPS_INJECT_TO": "@attr_match"},
    )
    assert RequirementLinker._has_dynamic_params(probe) is True
    _CONTENT_CATALOGUE.insert(0, probe)
    try:
        ev = RequirementLinker(_model(), llm=None).compile_evidence()
        rtcm = next(
            s for s in ev.test_specs
            if s.req_id == "REQ_INTF_002" and s.tier == "L1"
        )
        # the dynamic probe was refused; the static RTCM_GPS entry still fires
        assert [p.param_name for p in rtcm.params] == ["GPS_INJECT_TO"]
        assert all(p.value == 127 for p in rtcm.params)
        assert "<unresolved" not in ev.parm_file
    finally:
        _CONTENT_CATALOGUE.remove(probe)


def test_fence_requirement_gets_l1_and_secondary_l2_specs():
    ev = _evidence()
    cons = [s for s in ev.test_specs if s.req_id == "REQ_CONS_001"]
    assert sorted(s.tier for s in cons) == ["L1", "L2"]

    l1 = next(s for s in cons if s.tier == "L1")
    assert {p.param_name for p in l1.params} == {"FENCE_ENABLE", "FENCE_ALT_MAX"}

    l2 = next(s for s in cons if s.tier == "L2")
    assert l2.inject.kind == "set_param"
    assert l2.inject.pre_takeoff_m > 0
    # the scaled fence must sit BELOW the pre-takeoff altitude to breach
    assert float(l2.inject.params["FENCE_ALT_MAX"]) < l2.inject.pre_takeoff_m
    assert l2.verify.kind == "wait_mode"
    assert l2.verify.args["mode"] == "RTL"


def test_bool_guard_spelling_still_maps_abort_lock():
    """The guard-driven spelling (pre-existing behaviour) must survive the
    accept-event extension and the new text gates."""
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_006 { doc /* Delivery abort shall keep the payload mechanically locked. */ }
            part def PayloadMechanism {
                attribute deliveryAbortConditionActive : Boolean = false;
                satisfy requirement REQ_SAFE_006;
                action def lockPayload {}
                state def Monitor {
                    state nominal;
                    state locked { entry action l : lockPayload; }
                    transition initial then nominal;
                    transition abort first nominal if deliveryAbortConditionActive then locked;
                }
            }
        }""",
        model_name="D",
    )
    ev = RequirementLinker(model, llm=None).compile_evidence()
    spec = next(s for s in ev.test_specs if s.req_id == "REQ_SAFE_006")
    assert spec.tier == "L2"
    assert spec.verify.kind == "assert_servo_pwm"
    assert ev.guard_assignments["REQ_SAFE_006"].kind == "bool_true"


def test_static_only_l1_rows_stay_partial_and_derived_rows_verify():
    """Config-level L1 (port-matched static params) provides tier evidence but
    must not close a requirement's obligations; model-derived L1 keeps the full
    closure claim."""
    model = _model()
    ev = RequirementLinker(model, llm=None).compile_evidence()
    rows = {
        r.req_id: r
        for r in build_matrix(
            model, None, ev,
            l1_results=[
                {"req_id": "REQ_INTF_001", "passed": True},
                {"req_id": "REQ_INTF_002", "passed": True},
                {"req_id": "REQ_CONS_001", "passed": True},
            ],
            l2_results=[
                {"req_id": "REQ_SAFE_006", "passed": True},
                {"req_id": "REQ_SAFE_008", "passed": True},
                {"req_id": "REQ_CONS_001", "passed": True},
            ],
        )
    }

    intf2 = rows["REQ_INTF_002"]
    assert "l1_param" in intf2.tiers
    assert intf2.status == "partial"
    assert any("config-level" in line for line in intf2.evidence)

    intf1 = rows["REQ_INTF_001"]
    assert "l1_param" in intf1.tiers
    assert intf1.status == "partial"

    cons = rows["REQ_CONS_001"]
    assert "l1_param" in cons.tiers
    assert "l2_sitl" in cons.tiers
    assert cons.status == "verified"


def test_accept_pseudo_guard_does_not_hijack_behavioral_routing():
    """The accept pseudo-guard routes the SITL L2 spec only. Behavioural-sim
    anchoring must keep its own judgement — SAFE_006/008 rows must not lose or
    gain behavioural tiers because a SITL mapping now exists."""
    model = _model()
    ev = RequirementLinker(model, llm=None).compile_evidence()
    rows = {r.req_id: r for r in build_matrix(model, None, ev)}
    row = rows["REQ_SAFE_008"]
    # tiers must come from initialization/functional routing (or none), never
    # from the accept-claimed guard branch, whose signature can match nothing.
    assert "l2_sitl_planned" in row.tiers


_PARACHUTE_ACCEPT_MODEL = """package D {
    item def CriticalPropulsionFailure;
    requirement def REQ_SAFE_005 { doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight. */ }
    part def SafetyMonitor {
        attribute parachuteDeployTime : Real = 0.5 [s];
        satisfy requirement REQ_SAFE_005;
        action def deployParachute {}
        state def ParachuteDeploymentBehavior {
            entry; then Monitoring;
            state Monitoring;
            state DeployingParachute {
                entry action onDeploy : deployParachute;
            }
            transition toDeploy
                first Monitoring
                accept CriticalPropulsionFailure
                then DeployingParachute;
        }
    }
}"""


def test_parachute_accept_machine_maps_to_the_executable_l2():
    """Run 44642597 wrote the parachute as an accept-event machine with an
    abstract (empty) response action; the bool-only matcher left it silently
    unmapped. The accept surface must reach the same servo8 check the
    guard-driven spelling gets."""
    model = build_lite_model(_PARACHUTE_ACCEPT_MODEL, model_name="D")
    ev = RequirementLinker(model, llm=None).compile_evidence()

    spec = next(s for s in ev.test_specs if s.req_id == "REQ_SAFE_005")
    assert spec.tier == "L2"
    assert spec.inject.kind == "mavlink_command"
    assert spec.verify.kind == "assert_servo_pwm"
    assert spec.verify.args["channel"] == 8
    assert spec.verify.args["target_pwm"] == 2000

    ga = ev.guard_assignments["REQ_SAFE_005"]
    assert ga.kind == "accept_event"
    assert ga.attribute == "CriticalPropulsionFailure"
    assert not any(
        m.get("req_id") == "REQ_SAFE_005" for m in ev.traceability_mismatches
    )


def test_parachute_accept_machine_with_wrong_send_stays_blocked():
    """The 5af6c666 defect shape transplanted onto the accept spelling: the
    response state re-sends the detected-failure event instead of a parachute
    command. Opening the accept surface must not open a bypass around the
    response traceability gate."""
    wrong = _PARACHUTE_ACCEPT_MODEL.replace(
        "        action def deployParachute {}\n",
        "        out port parachuteCmd : ParachuteCmdPort;\n"
        "        action def deployParachute {\n"
        "            send CriticalPropulsionFailure() to parachuteCmd;\n"
        "        }\n",
    )
    model = build_lite_model(wrong, model_name="D")
    ev = RequirementLinker(model, llm=None).compile_evidence()

    spec = next(s for s in ev.test_specs if s.req_id == "REQ_SAFE_005")
    assert spec.tier == "TRACE"
    mismatches = [
        m for m in ev.traceability_mismatches
        if m.get("req_id") == "REQ_SAFE_005"
    ]
    assert mismatches
    assert "CRITICALPROPULSIONFAILURE" in mismatches[0]["message"]


def test_a_verify_threshold_comes_from_the_model_or_the_check_does_not_run():
    """A check that invents its own limit reports a verdict about nothing."""
    from src.sitl.requirement_linker import RequirementLinker
    from src.sitl.sitl_specs import VerifySpec

    spec = VerifySpec(kind="assert_waypoint_update_latency",
                      args={"max_latency_s": "@attr_match"})

    bound = RequirementLinker._bind_verify_args(spec, 1.0)
    assert bound.args["max_latency_s"] == 1.0

    # unresolved, or a nonsense zero threshold → drop the spec entirely
    assert RequirementLinker._bind_verify_args(spec, None) is None
    assert RequirementLinker._bind_verify_args(spec, 0.0) is None

    # a spec with no token is passed through untouched
    plain = VerifySpec(kind="wait_mode", args={"mode": "RTL"})
    assert RequirementLinker._bind_verify_args(plain, None) is plain


def test_mavlink_v2_check_states_what_it_does_not_cover():
    """The evidence must say encryption is out of its reach, since the same
    requirement asks for both and only one is testable."""
    from src.sitl.sitl_specs import VerifySpec, render_verify

    body = render_verify(VerifySpec(kind="assert_mavlink_v2_link"))
    assert "encryption is NOT covered" in body
    assert "0xFD" in body          # reads the real v2 start-of-frame byte
