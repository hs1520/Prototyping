from types import SimpleNamespace

from src.sitl.requirement_linker import _TAG_TO_ENTRY
from src.sitl.requirement_linker import RequirementLinker
from src.sitl.sitl_specs import (
    RENDER_VERIFY,
    VERIFY_HANDLERS,
    TestContext as SitlTestContext,
    VerifySpec,
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


def test_s4_catalogue_cases_use_standard_mavlink_state_not_statustext():
    assert _TAG_TO_ENTRY["SENSOR_GROUND_ALERT"].verify.kind == "assert_sensor_unhealthy"
    assert _TAG_TO_ENTRY["SENSOR_GROUND_ALERT"].verify.args == {"sensor": "gps"}

    parachute = _TAG_TO_ENTRY["PARACHUTE_DEPLOY"]
    assert parachute.ardu_params["SERVO8_FUNCTION"] == 27
    assert parachute.verify.kind == "assert_servo_pwm"
    assert parachute.verify.args == {"channel": 8, "target_pwm": 2000, "tol": 50}

    gripper = _TAG_TO_ENTRY["PAYLOAD_ABORT_LOCK"]
    assert gripper.ardu_params["SERVO7_FUNCTION"] == 28
    assert gripper.verify.kind == "assert_servo_pwm"
    assert gripper.verify.args == {"channel": 7, "target_pwm": 2000, "tol": 50}


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
