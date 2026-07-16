from types import SimpleNamespace

from src.sitl.requirement_linker import _TAG_TO_ENTRY
from src.sitl.requirement_linker import RequirementLinker
from src.sitl.sitl_bridge import BridgeReport, SITLBridge, TestResult as SITLTestResult
from src.sitl.sitl_specs import (
    RENDER_VERIFY,
    VERIFY_HANDLERS,
    TestContext as SitlTestContext,
    VerifySpec,
    render_inject,
    render_verify,
    run_verify,
)
from src.sysml.lite_model import build_lite_model


class _FakeMav:
    def __init__(self, messages):
        self._messages = list(messages)

    def recv_match(self, type=None, blocking=True, timeout=1):  # noqa: A002, ARG002
        if not self._messages:
            return None
        return self._messages.pop(0)


def _ctx(messages):
    mavlink = SimpleNamespace(MAV_SYS_STATUS_SENSOR_GPS=32)
    return SitlTestContext(mav=_FakeMav(messages), mavutil=SimpleNamespace(mavlink=mavlink))


def test_assert_servo_pwm_accepts_target_with_tolerance():
    ok, msg = run_verify(
        _ctx([SimpleNamespace(servo8_raw=1990)]),
        VerifySpec(kind="assert_servo_pwm", args={"channel": 8, "target_pwm": 2000, "tol": 50}, timeout=0.1),
    )
    assert ok
    assert "servo8_raw=1990" in msg


def test_assert_servo_pwm_rejects_wrong_channel_value():
    ok, msg = run_verify(
        _ctx([SimpleNamespace(servo7_raw=1500)]),
        VerifySpec(kind="assert_servo_pwm", args={"channel": 7, "target_pwm": 2000, "tol": 50}, timeout=0.1),
    )
    assert not ok
    assert "servo7_raw" in msg


def test_assert_sensor_unhealthy_uses_sys_status_health_bit():
    gps_bit = 32
    ok, msg = run_verify(
        _ctx([SimpleNamespace(onboard_control_sensors_health=0xFFFF & ~gps_bit)]),
        VerifySpec(kind="assert_sensor_unhealthy", args={"sensor": "gps"}, timeout=0.1),
    )
    assert ok
    assert "GPS health bit cleared" in msg


def test_assert_sensor_unhealthy_accepts_gps_raw_no_fix():
    ok, msg = run_verify(
        _ctx([SimpleNamespace(get_type=lambda: "GPS_RAW_INT", fix_type=1)]),
        VerifySpec(kind="assert_sensor_unhealthy", args={"sensor": "gps"}, timeout=0.1),
    )
    assert ok
    assert "fix_type=1" in msg


def test_s4_catalogue_cases_use_standard_mavlink_state_not_statustext():
    assert _TAG_TO_ENTRY["SENSOR_GROUND_ALERT"].verify.kind == "assert_sensor_unhealthy"
    assert _TAG_TO_ENTRY["SENSOR_GROUND_ALERT"].verify.args == {"sensor": "gps"}
    assert _TAG_TO_ENTRY["SENSOR_ARMING_INHIBIT"].inject.params["SIM_GPS1_ENABLE"] == 0.0

    parachute = _TAG_TO_ENTRY["PARACHUTE_DEPLOY"]
    assert parachute.ardu_params["SERVO8_FUNCTION"] == 27
    assert parachute.verify.kind == "assert_servo_pwm"
    assert parachute.verify.args == {"channel": 8, "target_pwm": 2000, "tol": 50}

    gripper = _TAG_TO_ENTRY["PAYLOAD_ABORT_LOCK"]
    assert gripper.ardu_params["SERVO7_FUNCTION"] == 28
    assert gripper.inject.params["SERVO7_FUNCTION"] == 28
    assert gripper.inject.params["GRIP_GRAB"] == 1000
    assert gripper.verify.kind == "assert_servo_pwm"
    # abort → LOCK = GRAB → GRIP_GRAB=1000 (not release 2000); see catalogue note.
    assert gripper.verify.args == {"channel": 7, "target_pwm": 1000, "tol": 50}


def test_new_verify_handlers_are_registered_and_renderable():
    assert "assert_servo_pwm" in VERIFY_HANDLERS
    assert "assert_sensor_unhealthy" in VERIFY_HANDLERS
    assert "assert_servo_pwm" in RENDER_VERIFY
    assert "assert_sensor_unhealthy" in RENDER_VERIFY

    servo_code = render_verify(
        VerifySpec(kind="assert_servo_pwm", args={"channel": 8, "target_pwm": 2000, "tol": 50})
    )
    sensor_code = render_verify(VerifySpec(kind="assert_sensor_unhealthy", args={"sensor": "gps"}))
    assert "SERVO_OUTPUT_RAW" in servo_code
    assert "servo8_raw" in servo_code
    assert "SYS_STATUS" in sensor_code
    assert "MAV_SYS_STATUS_SENSOR_GPS" in sensor_code


def test_ast_fallback_does_not_reintroduce_statustext_for_payload_lock():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_005 { doc /* lock payload */ }
            requirement def REQ_SAFE_006 { doc /* lock payload */ }
            requirement def REQ_SAFE_008 { doc /* lock payload */ }
            part def SafetyMonitor {
                action def lockPayload { }
                attribute deliveryAbortConditionActive : Boolean;
                state def Monitor {
                    state nominal;
                    state locked { entry action l : lockPayload; }
                    transition initial then nominal;
                    transition abort first nominal if deliveryAbortConditionActive then locked;
                }
                satisfy requirement REQ_SAFE_005;
                satisfy requirement REQ_SAFE_006;
                satisfy requirement REQ_SAFE_008;
            }
        }""",
        model_name="D",
    )
    specs = {
        s.req_id: s
        for s in RequirementLinker(model).generate_test_specs()
        if s.req_id in {"REQ_SAFE_005", "REQ_SAFE_006", "REQ_SAFE_008"}
    }
    assert specs
    assert {s.verify.kind for s in specs.values()} == {"assert_servo_pwm"}
    assert all(s.verify.args["channel"] == 7 for s in specs.values())


def test_safe_requirement_text_tag_mismatch_is_reported_not_flown():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_003 {
                doc /* The system shall enter RTL or LAND when the GCS link is lost for more than 10 seconds. */
            }
            part def SafetyMonitor {
                action def deployParachute { }
                attribute propulsionCriticalFailure : Boolean;
                attribute parachuteDeployTime : Real = 0.5;
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
                satisfy requirement REQ_SAFE_003;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    specs = {s.req_id: s for s in linker.generate_test_specs()}
    spec = specs["REQ_SAFE_003"]

    assert spec.tier == "TRACE"
    assert spec.inject.kind == "skip"
    assert spec.verify.kind == "skip"
    report = linker.coverage_report()
    assert "Traceability mismatches" in report
    assert "matched=PARACHUTE_DEPLOY" in report
    mismatch = linker.traceability_mismatches()[0]
    assert mismatch["expected_family"] == "GCS"
    assert mismatch["matched_tag"] == "PARACHUTE_DEPLOY"
    assert "SERVO8_FUNCTION" not in linker.generate_parm_file()


def test_geofence_requirement_is_not_verified_as_parachute():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_009 {
                doc /* The system shall automatically initiate a controlled descent if the vehicle deviates more than 50 metres outside the designated segregated airspace boundaries. */
            }
            part def SafetyMonitor {
                action def deployParachute { }
                attribute propulsionCriticalFailure : Boolean;
                attribute parachuteDeployTime : Real = 0.5;
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
                satisfy requirement REQ_SAFE_009;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    specs = {s.req_id: s for s in linker.generate_test_specs()}
    spec = specs["REQ_SAFE_009"]

    assert spec.tier == "TRACE"
    assert spec.inject.kind == "skip"
    assert spec.verify.kind == "skip"
    mismatch = linker.traceability_mismatches()[0]
    assert mismatch["expected_family"] == "GEOFENCE"
    assert mismatch["matched_tag"] == "PARACHUTE_DEPLOY"
    assert "SERVO8_FUNCTION" not in linker.generate_parm_file()


def test_safe_requirement_text_tag_match_still_generates_l2():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_003 {
                doc /* The system shall enter RTL or LAND when the GCS link is lost for more than 10 seconds. */
            }
            part def SafetyMonitor {
                action def enterRtl { }
                attribute commLossTime : Real = 0.0;
                state def Monitor {
                    state nominal;
                    state rtl { entry action r : enterRtl; }
                    transition initial then nominal;
                    transition lost first nominal if commLossTime > 10.0 then rtl;
                }
                satisfy requirement REQ_SAFE_003;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    specs = {s.req_id: s for s in linker.generate_test_specs()}
    spec = specs["REQ_SAFE_003"]

    assert spec.tier == "L2"
    assert spec.inject.kind == "disconnect_gcs"
    assert spec.verify.kind == "wait_mode"
    assert linker.traceability_mismatches() == []


def test_parachute_requirement_text_tag_match_stays_executable_l2():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_005 {
                doc /* The system shall deploy the ballistic recovery parachute within 0.5 seconds of detecting a critical propulsion subsystem failure during flight. */
            }
            part def SafetyMonitor {
                action def deployParachute { }
                attribute propulsionCriticalFailure : Boolean;
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
                satisfy requirement REQ_SAFE_005;
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    specs = {s.req_id: s for s in linker.generate_test_specs()}
    spec = specs["REQ_SAFE_005"]

    assert spec.tier == "L2"
    assert spec.inject.kind == "mavlink_command"
    assert spec.inject.params["CHUTE_ALT_MIN"] == 0
    assert spec.inject.params["SERVO8_FUNCTION"] == 27
    assert spec.verify.kind == "assert_servo_pwm"
    assert spec.verify.args == {"channel": 8, "target_pwm": 2000, "tol": 50}
    params = [(p.param_name, p.value) for p in spec.params]
    assert ("SERVO8_FUNCTION", 27) in params
    assert ("CHUTE_ALT_MIN", 0) in params
    rendered = render_inject(spec.inject)
    assert 'set_param(mav, "CHUTE_ALT_MIN", 0.0)' in rendered
    assert rendered.index('set_param(mav, "CHUTE_ALT_MIN", 0.0)') < rendered.index("command_long_send")
    assert linker.traceability_mismatches() == []


def test_parachute_guard_with_land_command_is_traceability_blocked():
    model = build_lite_model(
        """package D {
            action def CMD_LAND { }
            requirement def REQ_SAFE_005 {
                doc /* Critical propulsion failure shall deploy the parachute. */
            }
            part def SafetyMonitor {
                attribute propulsionCriticalFailure : Boolean = false;
                satisfy requirement REQ_SAFE_005;
                action def deployParachute { send CMD_LAND() to parachuteCmd; }
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    spec = {s.req_id: s for s in linker.generate_test_specs()}["REQ_SAFE_005"]

    assert spec.tier == "TRACE"
    assert spec.inject.kind == "skip"
    assert "CMD_PARACHUTE" in spec.notes
    assert "SERVO8_FUNCTION" not in linker.generate_parm_file()


def test_bridge_report_surfaces_traceability_mismatch(tmp_path):
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_003 {
                doc /* The system shall enter RTL or LAND when the GCS link is lost for more than 10 seconds. */
            }
            part def SafetyMonitor {
                action def deployParachute { }
                attribute propulsionCriticalFailure : Boolean;
                state def Monitor {
                    state nominal;
                    state chute { entry action p : deployParachute; }
                    transition initial then nominal;
                    transition failure first nominal if propulsionCriticalFailure then chute;
                }
                satisfy requirement REQ_SAFE_003;
            }
        }""",
        model_name="D",
    )
    bridge = SITLBridge(model, output_dir=str(tmp_path))

    report = bridge.generate_full_report(run_l2=False)

    assert len(report.trace_results) == 1
    assert report.trace_results[0].tier == "TRACE"
    assert report.trace_results[0].req_id == "REQ_SAFE_003"
    assert "Traceability blocked: 1" in report.summary()
    assert "L1/L2 passed:" in report.summary()


def test_bridge_report_safety_status_distinguishes_blocked_partial_and_fail():
    blocked = SITLTestResult("REQ_SAFE_003", "TRACE", False, "traceability mismatch")
    failed_l2 = SITLTestResult("REQ_SAFE_005", "L2", False, "servo8_raw last=1000")
    passed_l2 = SITLTestResult("REQ_SAFE_005", "L2", True, "servo8_raw=2000")

    assert BridgeReport("D", "D.parm", trace_results=[blocked]).safety_status() == "BLOCKED"
    assert BridgeReport("D", "D.parm", l2_results=[failed_l2], trace_results=[blocked]).safety_status() == "PARTIAL"
    assert BridgeReport("D", "D.parm", l2_results=[failed_l2]).safety_status() == "FAIL"
    assert BridgeReport("D", "D.parm", l2_results=[passed_l2]).safety_status() == "PASS"
    assert BridgeReport("D", "D.parm").safety_status() == "NOT_RUN"


def test_family_aware_guard_assignment_keeps_shared_safety_monitor_traceable():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_002 { doc /* Battery below 15% shall perform controlled landing. */ }
            requirement def REQ_SAFE_003 { doc /* GCS link loss for more than 10 seconds shall trigger safe landing. */ }
            requirement def REQ_SAFE_004 { doc /* The system shall not transition to the armed or airborne state if any onboard sensor reports a failure during self-test. */ }
            requirement def REQ_SAFE_005 { doc /* Critical propulsion failure shall deploy the parachute. */ }
            requirement def REQ_SAFE_006 { doc /* Delivery abort shall keep the payload mechanically locked. */ }
            requirement def REQ_SAFE_008 { doc /* The system shall issue a failure alert to the GCS if self-test fails. */ }
            part def SafetyMonitor {
                attribute batterySoc : Real = 100.0;
                attribute commLossTime : Real = 0.0;
                attribute sensorSelfTestFailed : Boolean = false;
                attribute propulsionCriticalFailure : Boolean = false;
                attribute deliveryAbortConditionActive : Boolean = false;
                attribute parachuteDeployTime : Real = 0.5;

                satisfy requirement REQ_SAFE_002;
                satisfy requirement REQ_SAFE_003;
                satisfy requirement REQ_SAFE_004;
                satisfy requirement REQ_SAFE_005;
                satisfy requirement REQ_SAFE_006;
                satisfy requirement REQ_SAFE_008;

                action def lockPayload { }
                action def issueFailureAlert { }

                state def Monitor {
                    state nominal;
                    state batteryLand;
                    state gcsLoss;
                    state sensorInhibit;
                    state sensorAlert { entry action a : issueFailureAlert; }
                    state chute;
                    state payloadLock { entry action p : lockPayload; }
                    transition initial then nominal;
                    transition battery first nominal if batterySoc < 15.0 then batteryLand;
                    transition gcs first nominal if commLossTime > 10.0 then gcsLoss;
                    transition sensor first nominal if sensorSelfTestFailed then sensorInhibit;
                    transition sensorAlert first nominal if sensorSelfTestFailed then sensorAlert;
                    transition prop first nominal if propulsionCriticalFailure then chute;
                    transition payload first nominal if deliveryAbortConditionActive then payloadLock;
                }
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    specs = {s.req_id: s for s in linker.generate_test_specs()}

    assert linker.traceability_mismatches() == []
    assert specs["REQ_SAFE_002"].inject.kind == "noop"
    assert specs["REQ_SAFE_003"].inject.kind == "disconnect_gcs"
    assert specs["REQ_SAFE_004"].verify.kind == "assert_arm_rejected"
    assert specs["REQ_SAFE_005"].verify.kind == "assert_servo_pwm"
    assert specs["REQ_SAFE_006"].verify.kind == "assert_servo_pwm"
    assert specs["REQ_SAFE_008"].verify.kind == "assert_sensor_unhealthy"


def _model_with_req(req_id: str, doc: str, part_body: str, part_name: str = "FlightController"):
    return build_lite_model(
        f"""package D {{
            requirement def {req_id} {{ doc /* {doc} */ }}
            part def {part_name} {{
                {part_body}
                satisfy requirement {req_id};
            }}
        }}""",
        model_name="D",
    )


def test_attr_match_requires_requirement_text_relevance():
    # MTOW requirement satisfied by a part that happens to declare maxAirspeed:
    # the attr exists, but the requirement is not about speed → no mapping
    # (honest unmapped beats a wrong "L1 PASS WPNAV_SPEED").
    model = _model_with_req(
        "REQ_CONS_003",
        "The system maximum takeoff weight, including payload and battery, shall not exceed 25.0 kg.",
        "attribute maxAirspeed : Real = 15.0;",
        part_name="Airframe",
    )
    linker = RequirementLinker(model)
    assert linker._lookup_catalogue("REQ_CONS_003") is None  # noqa: SLF001
    assert "WPNAV_SPEED" not in linker.generate_parm_file()


def test_verification_method_annotation_does_not_trigger_control_loop_mapping():
    model = _model_with_req(
        "REQ_FUNC_001",
        "The system shall navigate to GPS waypoints with CEP below 1.0 metre. "
        "[V: hardware-in-the-loop / field survey]",
        "attribute controlFrequency : Real = 100.0;",
    )
    linker = RequirementLinker(model)
    assert linker._lookup_catalogue("REQ_FUNC_001") is None  # noqa: SLF001
    assert "SCHED_LOOP_RATE" not in linker.generate_parm_file()


def test_headwind_groundspeed_is_not_verified_by_navigation_speed_setpoint():
    model = _model_with_req(
        "REQ_PERF_004",
        "The system shall maintain a minimum forward ground speed of 2 m/s "
        "in sustained headwinds of 15 m/s.",
        "attribute minCruiseAirspeed : Real = 18.0;",
        part_name="PropulsionSystem",
    )
    linker = RequirementLinker(model)
    assert linker._lookup_catalogue("REQ_PERF_004") is None  # noqa: SLF001
    assert "WPNAV_SPEED" not in linker.generate_parm_file()


def test_postflight_report_to_gcs_does_not_claim_gcs_loss_guard():
    model = build_lite_model(
        """package D {
            requirement def REQ_FUNC_008 {
                doc /* The system shall transmit a post-flight health report to the GCS after landing. */
            }
            part def SafetyMonitor {
                attribute commLossTime : Real = 0.0;
                satisfy requirement REQ_FUNC_008;
                state def Monitor {
                    state nominal;
                    state land;
                    transition initial then nominal;
                    transition lost first nominal if commLossTime > 10.0 then land;
                }
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)

    assert linker._lookup_catalogue("REQ_FUNC_008") is None  # noqa: SLF001
    assert all(s.req_id != "REQ_FUNC_008" for s in linker.generate_test_specs())


def test_compound_contingency_is_not_proven_by_one_fault_guard():
    model = build_lite_model(
        """package D {
            requirement def REQ_FUNC_007 {
                doc /* The system shall return to base for GCS link loss, geofence breach, or battery state-of-charge at the return threshold. */
            }
            part def SafetyMonitor {
                attribute batterySoc : Real = 100.0;
                satisfy requirement REQ_FUNC_007;
                state def Monitor {
                    state nominal;
                    state rtb;
                    transition initial then nominal;
                    transition low first nominal if batterySoc <= 25.0 then rtb;
                }
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)

    assert linker._requirement_families("REQ_FUNC_007") == {  # noqa: SLF001
        "BATTERY", "GCS", "GEOFENCE"
    }
    assert linker._lookup_catalogue("REQ_FUNC_007") is None  # noqa: SLF001


def test_gcs_no_response_boundary_is_not_proven_by_positive_failsafe():
    model = build_lite_model(
        """package D {
            requirement def REQ_SAFE_009 {
                doc /* The system shall maintain its active flight plan without initiating a communication-loss emergency landing when the GCS uplink is interrupted for 9.9 seconds or less. */
            }
            part def SafetyMonitor {
                attribute commLossTime : Real = 0.0;
                satisfy requirement REQ_SAFE_009;
                state def Monitor {
                    state nominal;
                    state land;
                    transition initial then nominal;
                    transition lost first nominal if commLossTime > 10.0 then land;
                }
            }
        }""",
        model_name="D",
    )
    linker = RequirementLinker(model)
    spec = {s.req_id: s for s in linker.generate_test_specs()}["REQ_SAFE_009"]

    assert spec.tier == "TRACE"
    assert spec.inject.kind == "skip"
    assert "positive response" in spec.notes
    mismatch = linker.traceability_mismatches()[0]
    assert mismatch["expected_family"] == "GCS_NO_RESPONSE_BOUNDARY"


def test_nil_wind_airspeed_still_maps_to_navigation_speed_setpoint():
    model = _model_with_req(
        "REQ_PERF_003",
        "The system shall achieve a maximum airspeed of at least 18 m/s "
        "in nil-wind conditions.",
        "attribute maxAirspeed : Real = 18.0;",
        part_name="PropulsionSystem",
    )
    linker = RequirementLinker(model)
    match = linker._lookup_catalogue("REQ_PERF_003")  # noqa: SLF001

    assert match is not None
    assert match["semantic_tag"] == "MAX_SPEED"
    wpnav_line = next(
        line for line in linker.generate_parm_file().splitlines()
        if line.startswith("WPNAV_SPEED")
    )
    assert wpnav_line.split()[1] == "1800"


def test_attr_match_text_gate_unlocks_the_right_entry():
    # A loop-rate requirement on a part with BOTH maxAltitude and controlFrequency
    # previously matched ALTITUDE_FENCE (first catalogue hit). With the text gate
    # the altitude entry is skipped and the loop-rate entry resolves.
    model = _model_with_req(
        "REQ_PERF_006",
        "The AutonomousDrone shall execute the primary flight control loop at a frequency of no less than 100 Hz.",
        "attribute maxAltitude : Real = 120.0; attribute controlFrequency : Real = 100.0;",
    )
    linker = RequirementLinker(model)
    cat = linker._lookup_catalogue("REQ_PERF_006")  # noqa: SLF001
    assert cat is not None
    assert cat["semantic_tag"] == "CONTROL_LOOP_RATE"
    assert "FENCE_ALT_MAX" not in linker.generate_parm_file()


def test_attr_match_altitude_requirement_still_maps_to_fence():
    model = _model_with_req(
        "REQ_CONS_001",
        "The system shall not exceed a flight altitude of 120 metres above ground level.",
        "attribute maxAltitude : Real = 120.0;",
    )
    linker = RequirementLinker(model)
    cat = linker._lookup_catalogue("REQ_CONS_001")  # noqa: SLF001
    assert cat is not None
    assert cat["semantic_tag"] == "ALTITUDE_FENCE"
    assert "FENCE_ALT_MAX" in linker.generate_parm_file()


def _gcs_model(doc: str):
    return build_lite_model(
        f"""package D {{
            requirement def REQ_SAFE_003 {{ doc /* {doc} */ }}
            part def SafetyMonitor {{
                attribute commLossTime : Real = 0.0;
                satisfy requirement REQ_SAFE_003;
                state def Monitor {{
                    state nominal;
                    state lost;
                    transition initial then nominal;
                    transition gcs first nominal if commLossTime > 10.0 then lost;
                }}
            }}
        }}""",
        model_name="D",
    )


def test_gcs_loss_land_only_text_requires_land_action():
    linker = RequirementLinker(_gcs_model(
        "The system shall perform an autonomous safe landing at the current "
        "position when the GCS uplink has been absent for more than 10 seconds."
    ))
    spec = {s.req_id: s for s in linker.generate_test_specs()}["REQ_SAFE_003"]

    assert spec.verify.kind == "wait_mode"
    assert spec.verify.args == {"mode": "LAND"}  # no RTL fallback: wrong action ≠ pass
    assert spec.inject.params.get("FS_GCS_ENABLE") == 5
    params = {p.param_name: p.value for p in spec.params}
    assert params.get("FS_GCS_ENABLE") == 5
    assert linker.traceability_mismatches() == []


def test_gcs_loss_return_only_text_requires_rtl_action():
    linker = RequirementLinker(_gcs_model(
        "The system shall autonomously return to base when the GCS uplink has "
        "been absent for more than 10 seconds."
    ))
    spec = {s.req_id: s for s in linker.generate_test_specs()}["REQ_SAFE_003"]

    assert spec.verify.args == {"mode": "RTL"}
    assert spec.inject.params.get("FS_GCS_ENABLE") == 1
    assert linker.traceability_mismatches() == []


def test_gcs_loss_ambiguous_text_keeps_lenient_entry():
    linker = RequirementLinker(_gcs_model(
        "The system shall enter RTL or LAND when the GCS link is lost for more "
        "than 10 seconds."
    ))
    spec = {s.req_id: s for s in linker.generate_test_specs()}["REQ_SAFE_003"]

    # either failsafe reaction satisfies this requirement → fallback stays
    assert spec.verify.args.get("mode") == "LAND"
    assert spec.verify.args.get("fallback") == "RTL"


def test_disconnect_gcs_inject_and_render_honor_action_param():
    from src.sitl.sitl_specs import InjectSpec, render_inject

    rendered = render_inject(InjectSpec(kind="disconnect_gcs",
                                        params={"FS_GCS_ENABLE": 5}))
    assert 'set_param(mav, "FS_GCS_ENABLE", 5.0)' in rendered
    default = render_inject(InjectSpec(kind="disconnect_gcs"))
    assert 'set_param(mav, "FS_GCS_ENABLE", 1.0)' in default
