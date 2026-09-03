from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest

from src.sitl.sitl_bridge import (
    ARDUPILOT_COPTER_PROFILE,
    SITLBridge,
    _DEFAULT_ARDUCOPTER,
)
from src.sitl.sitl_specs import TestContext as SitlTestContext
from src.sysml.lite_model import build_lite_model

ROOT = Path(__file__).resolve().parents[1]
SYSML_PATH = ROOT / "examples" / "output" / "final_model.sysml"
MODEL_NAME = "AutonomousDrone"


@pytest.fixture
def sitl_bridge(tmp_path):
    if not os.path.exists(_DEFAULT_ARDUCOPTER):
        pytest.skip(f"arducopter SITL binary not found: {_DEFAULT_ARDUCOPTER}")
    if not SYSML_PATH.exists():
        pytest.skip(f"recommended SysML model not found: {SYSML_PATH}")
    text = SYSML_PATH.read_text(encoding="utf-8")
    model = build_lite_model(text, model_name=MODEL_NAME)
    bridge = SITLBridge(
        model,
        output_dir=str(tmp_path / "sitl"),
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        fdm_backend="native",
        verbose=False,
    )
    try:
        yield bridge
    finally:
        bridge.stop_sitl()


def _connect():
    from pymavlink import mavutil

    mav = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
    mav.wait_heartbeat(timeout=15)
    mav.mav.request_data_stream_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10,
        1,
    )
    return mav, mavutil


def _force_arm(ctx: SitlTestContext, timeout_s: float = 45.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ctx.mav.mav.command_long_send(
            ctx.mav.target_system,
            ctx.mav.target_component,
            ctx.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            21196,
            0,
            0,
            0,
            0,
            0,
        )
        ack = ctx.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=2)
        if (
            ack
            and ack.command == ctx.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
            and ack.result == ctx.mavutil.mavlink.MAV_RESULT_ACCEPTED
        ):
            return True
        time.sleep(2)
    return False


def _takeoff_and_wait(
    ctx: SitlTestContext,
    altitude_m: float = 5.0,
    threshold_ratio: float = 0.6,
    timeout_s: float = 35.0,
) -> tuple[bool, float]:
    ctx.mav.mav.command_long_send(
        ctx.mav.target_system,
        ctx.mav.target_component,
        ctx.mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        altitude_m,
    )
    target_mm = int(altitude_m * threshold_ratio * 1000)
    plausible_max_mm = int(altitude_m * 3 * 1000) + 5000
    drain_until = time.time() + 0.5
    while time.time() < drain_until:
        ctx.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
    baseline_mm = None
    max_climb_m = 0.0
    hits = 0
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = ctx.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is not None:
            if baseline_mm is None:
                if -1000 <= msg.relative_alt <= plausible_max_mm:
                    baseline_mm = msg.relative_alt
                else:
                    continue
            elif msg.relative_alt < baseline_mm and -1000 <= msg.relative_alt <= plausible_max_mm:
                baseline_mm = msg.relative_alt
            climb_mm = msg.relative_alt - (baseline_mm or 0)
            if 0 <= climb_mm <= plausible_max_mm:
                max_climb_m = max(max_climb_m, climb_mm / 1000.0)
            if 0 < climb_mm <= plausible_max_mm and climb_mm >= target_mm:
                hits += 1
                if hits >= 2:
                    return True, max_climb_m
            else:
                hits = 0
        time.sleep(0.3)
    return False, max_climb_m


def _prepare_guided_airborne(ctx: SitlTestContext, altitude_m: float) -> float:
    ctx.reset_drone_state()
    ctx.set_param("ARMING_CHECK", 0)
    ctx.set_param("FENCE_ENABLE", 0)
    ctx.set_param("SIM_GPS1_ENABLE", 1)
    ctx.set_mode("GUIDED")
    assert _force_arm(ctx), "force-arm failed"
    takeoff_ok, climb_m = _takeoff_and_wait(ctx, altitude_m=altitude_m)
    assert takeoff_ok, f"takeoff/climb threshold not reached; max_climb_m={climb_m:.2f}"
    return climb_m


def _hover_observe(ctx: SitlTestContext, duration_s: float = 15.0) -> dict:
    altitudes: list[float] = []
    throttles: list[float] = []
    tilts: list[float] = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        msg = ctx.mav.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype == "GLOBAL_POSITION_INT":
            altitude_m = msg.relative_alt / 1000.0
            if -5.0 <= altitude_m <= 30.0:
                altitudes.append(altitude_m)
        elif mtype == "VFR_HUD":
            throttles.append(float(getattr(msg, "throttle", 0.0)))
        elif mtype == "ATTITUDE":
            tilts.append(max(abs(float(msg.roll)), abs(float(msg.pitch))))
    alt_min = min(altitudes) if altitudes else 0.0
    alt_max = max(altitudes) if altitudes else 0.0
    throttle_mean = statistics.fmean(throttles) if throttles else None
    tilt_max_rad = max(tilts) if tilts else None
    stable = bool(altitudes) and alt_min >= 2.5 and alt_max <= 8.0 and (alt_max - alt_min) <= 3.0
    tilt_ok = tilt_max_rad is None or tilt_max_rad < 0.8
    return {
        "stable_hover": stable and tilt_ok,
        "altitude_min_m": round(alt_min, 3),
        "altitude_max_m": round(alt_max, 3),
        "altitude_span_m": round(alt_max - alt_min, 3),
        "hover_throttle_pct_mean": round(throttle_mean, 1) if throttle_mean is not None else None,
        "max_tilt_rad": round(tilt_max_rad, 3) if tilt_max_rad is not None else None,
    }


def _wait_mode(ctx: SitlTestContext, mode: str, timeout_s: float = 35.0) -> tuple[bool, str]:
    target = mode.upper()
    last = "UNKNOWN"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = ctx.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg:
            last = ctx.mavutil.mode_string_v10(msg)
            if target in last.upper():
                return True, last
        time.sleep(0.5)
    return False, last


@pytest.mark.sitl
def test_recommended_design_flies(sitl_bridge):
    assert sitl_bridge.launch_sitl(wait_s=8.0)
    mav = None
    try:
        mav, mavutil = _connect()
        ctx = SitlTestContext(mav=mav, mavutil=mavutil)
        climb_m = _prepare_guided_airborne(ctx, altitude_m=5.0)
        hover = _hover_observe(ctx, duration_s=15.0)
        print(f"SITL flyability: max_climb_m={climb_m:.2f}, hover={hover}")
        assert climb_m >= 2.5
        assert hover["stable_hover"], hover
    finally:
        if mav is not None:
            mav.close()


@pytest.mark.sitl
def test_parachute_actuates(sitl_bridge):
    assert sitl_bridge.launch_sitl(wait_s=8.0)
    mav = None
    try:
        mav, mavutil = _connect()
        ctx = SitlTestContext(mav=mav, mavutil=mavutil)
        climb_m = _prepare_guided_airborne(ctx, altitude_m=10.0)
        ctx.set_param("CHUTE_ENABLED", 1)
        ctx.set_param("CHUTE_TYPE", 10)
        ctx.set_param("CHUTE_ALT_MIN", 0)
        ctx.set_param("CHUTE_SERVO_ON", 2000)
        ctx.set_param("CHUTE_SERVO_OFF", 1000)
        ctx.set_param("SERVO8_FUNCTION", 27)
        time.sleep(1.0)
        ctx.mav.mav.command_long_send(
            ctx.mav.target_system,
            ctx.mav.target_component,
            ctx.mavutil.mavlink.MAV_CMD_DO_PARACHUTE,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        peak = 0
        deadline = time.time() + 12
        while time.time() < deadline:
            msg = ctx.mav.recv_match(type="SERVO_OUTPUT_RAW", blocking=True, timeout=1)
            if msg is None:
                continue
            pwm = int(getattr(msg, "servo8_raw", 0) or 0)
            peak = max(peak, pwm)
            if abs(pwm - 2000) <= 50:
                break
        print(f"SITL parachute independent probe: max_climb_m={climb_m:.2f}, servo8_peak={peak}")
        assert peak >= 1950
    finally:
        if mav is not None:
            mav.close()


@pytest.mark.sitl
def test_battery_failsafe_enters_rtl(sitl_bridge):
    assert sitl_bridge.launch_sitl(wait_s=8.0)
    mav = None
    try:
        mav, mavutil = _connect()
        ctx = SitlTestContext(mav=mav, mavutil=mavutil)
        climb_m = _prepare_guided_airborne(ctx, altitude_m=5.0)
        ctx.set_param("BATT_MONITOR", 4)
        ctx.set_param("BATT_FS_LOW_ACT", 2)
        ctx.set_param("BATT_LOW_TIMER", 1)
        ctx.set_param("BATT_LOW_VOLT", 100)
        ok, mode = _wait_mode(ctx, "RTL", timeout_s=35.0)
        print(f"SITL battery failsafe: max_climb_m={climb_m:.2f}, mode={mode}, rtl={ok}")
        assert ok, f"battery failsafe did not enter RTL; last mode={mode}"
    finally:
        if mav is not None:
            mav.close()
