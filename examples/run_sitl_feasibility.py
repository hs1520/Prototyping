"""Native-SITL feasibility check for the recommended design.

This verifies flight feasibility and L2 safety behavior only. It does NOT verify
datasheet endurance; native SITL's hover current model is architecture-
nondiscriminating, so any current/endurance observations are explicitly
non-authoritative.

Run:
  PYTHONPATH=. .venv/bin/python examples/run_sitl_feasibility.py
"""
from __future__ import annotations

import json
import signal
import shutil
import statistics
import sys
import time
from pathlib import Path

from pymavlink import mavutil

from src.dse.physics_estimator import DesignInputs
from src.sitl.dse_sitl_params import design_to_sitl_parm
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE, SITLBridge
from src.sitl.sitl_specs import TestContext
from src.sysml.lite_model import build_lite_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "output"
SYSML_PATH = OUT / "final_model.sysml"
FALLBACK_SYSML_PATH = ROOT / "examples" / "drone_system_v2.sysml"
PARM_PATH = OUT / "recommended.parm"
RUN_JSON = OUT / "realization_run.json"
WORK = OUT / "sitl_feasibility"
REPORT_JSON = OUT / "sitl_feasibility_report.json"
REPORT_MD = OUT / "sitl_feasibility_report.md"
MODEL_NAME = "AutonomousDrone"


def _read_model():
    if SYSML_PATH.exists():
        text = SYSML_PATH.read_text(encoding="utf-8")
        note = "current" if PARM_PATH.exists() else "existing_model_not_from_this_run"
        return build_lite_model(text, model_name=MODEL_NAME), str(SYSML_PATH), note
    if FALLBACK_SYSML_PATH.exists():
        text = FALLBACK_SYSML_PATH.read_text(encoding="utf-8")
        return build_lite_model(text, model_name=MODEL_NAME), str(FALLBACK_SYSML_PATH), "fallback_existing_model"
    raise FileNotFoundError(f"{SYSML_PATH} not found and fallback {FALLBACK_SYSML_PATH} is unavailable")


def _parm_lines_from_json() -> list[str]:
    data = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    d = data.get("recommended_design_inputs") or {}
    if not d:
        raise FileNotFoundError(f"{PARM_PATH} missing and {RUN_JSON} lacks recommended_design_inputs")
    di = DesignInputs(**d)
    lines = list(design_to_sitl_parm(di))
    existing = {line.split()[0] for line in lines if line.strip() and not line.lstrip().startswith("#")}
    for key, value in (ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {}).items():
        if key not in existing:
            lines.append(f"{key:<20} {value}")
    return lines


def _prepare_bridge_inputs(model) -> tuple[SITLBridge, str]:
    WORK.mkdir(parents=True, exist_ok=True)
    bridge_parm = WORK / f"{MODEL_NAME}.parm"
    source = "examples/output/recommended.parm"
    if PARM_PATH.exists():
        shutil.copyfile(PARM_PATH, bridge_parm)
    else:
        source = "reconstructed_from_existing_realization_run.json"
        lines = _parm_lines_from_json()
        bridge_parm.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bridge = SITLBridge(
        model=model,
        output_dir=str(WORK),
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        fdm_backend="native",
        verbose=False,
    )
    return bridge, source


def _connect():
    mav = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
    mav.wait_heartbeat(timeout=15)
    mav.mav.request_data_stream_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10,
        1,
    )
    return mav


def _force_arm(ctx: TestContext, timeout_s: float = 45.0) -> bool:
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


def _takeoff_and_wait(ctx: TestContext, altitude_m: float = 5.0, timeout_s: float = 35.0) -> tuple[bool, float]:
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
    target_mm = int(altitude_m * 0.6 * 1000)
    plausible_max_mm = int(altitude_m * 3 * 1000) + 5000
    baseline_mm = None
    max_climb = 0.0
    hits = 0
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = ctx.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is not None:
            if baseline_mm is None or msg.relative_alt < baseline_mm:
                baseline_mm = msg.relative_alt
            climb_mm = msg.relative_alt - (baseline_mm or 0)
            if 0 <= climb_mm <= plausible_max_mm:
                max_climb = max(max_climb, climb_mm / 1000.0)
            if 0 < climb_mm <= plausible_max_mm and climb_mm >= target_mm:
                hits += 1
                if hits >= 2:
                    return True, max_climb
            else:
                hits = 0
        time.sleep(0.3)
    return False, max_climb


def _hover_observe(ctx: TestContext, duration_s: float = 15.0) -> dict:
    altitudes = []
    throttles = []
    tilts = []
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
        "twr_sanity": "climb achieved and stable hover observed" if stable and tilt_ok else "not established",
    }


def run_flight_feasibility(bridge: SITLBridge) -> dict:
    result = {"passed": False, "arm": False, "takeoff": False, "max_climb_m": 0.0, "hover": {}}
    if not bridge.launch_sitl(wait_s=8.0):
        result["message"] = "SITL launch failed"
        return result
    mav = None
    try:
        mav = _connect()
        ctx = TestContext(mav=mav, mavutil=mavutil)
        ctx.reset_drone_state()
        ctx.set_param("ARMING_CHECK", 0)
        ctx.set_param("FENCE_ENABLE", 0)
        ctx.set_param("SIM_GPS1_ENABLE", 1)
        ctx.set_mode("GUIDED")
        result["arm"] = _force_arm(ctx)
        if not result["arm"]:
            result["message"] = "force-arm failed"
            return result
        takeoff_ok, climb = _takeoff_and_wait(ctx, altitude_m=5.0)
        result["takeoff"] = takeoff_ok
        result["max_climb_m"] = round(climb, 3)
        if not takeoff_ok:
            result["message"] = "takeoff/climb threshold not reached"
            return result
        hover = _hover_observe(ctx, duration_s=15.0)
        result["hover"] = hover
        result["passed"] = bool(hover.get("stable_hover"))
        result["message"] = "stable hover observed" if result["passed"] else "hover stability not established"
        return result
    finally:
        try:
            if mav is not None:
                mav.close()
        except Exception:
            pass
        bridge.stop_sitl()


class _CaseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001 - signal handler signature
    raise _CaseTimeout("L2 case timeout")


def _l2_note(req_id: str, passed: bool, message: str) -> str:
    if req_id == "REQ_SAFE_003" and not passed and "STATUSTEXT" in message.upper():
        return "suspected STATUSTEXT keyword fragility, not confirmed safety defect"
    if not passed and "L2 case timeout" in message:
        return "bounded SITL run timed out; not counted as a safety pass"
    return ""


def _run_single_l2_with_timeout(bridge: SITLBridge, spec, timeout_s: int = 240) -> dict:
    print(f"\n[L2] 运行测试 {spec.req_id} ...", flush=True)
    t0 = time.time()
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    launched = False
    try:
        eeprom = Path.cwd() / "eeprom.bin"
        if eeprom.exists():
            try:
                eeprom.unlink()
            except OSError:
                pass
        launched = bridge.launch_sitl(wait_s=45.0)
        if not launched:
            return {
                "req_id": spec.req_id,
                "tier": "L2",
                "passed": False,
                "message": "独立 SITL 启动失败",
                "duration_s": round(time.time() - t0, 2),
                "note": "",
            }
        passed, message = bridge._run_single_test(  # noqa: SLF001 - script-level wrapper around existing L2 runner
            spec,
            bridge._connection_string,  # noqa: SLF001
            mavutil,
            fresh_sitl=True,
        )
        return {
            "req_id": spec.req_id,
            "tier": "L2",
            "passed": bool(passed),
            "message": message,
            "duration_s": round(time.time() - t0, 2),
            "note": _l2_note(spec.req_id, bool(passed), message),
        }
    except _CaseTimeout:
        return {
            "req_id": spec.req_id,
            "tier": "L2",
            "passed": False,
            "message": f"case timed out after {timeout_s}s",
            "duration_s": round(time.time() - t0, 2),
            "note": "outer timeout; SITL process stopped and result is not a pass",
        }
    except Exception as exc:
        message = f"异常: {exc}"
        return {
            "req_id": spec.req_id,
            "tier": "L2",
            "passed": False,
            "message": message,
            "duration_s": round(time.time() - t0, 2),
            "note": _l2_note(spec.req_id, False, message),
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if launched:
            bridge.stop_sitl()


def run_safety_l2(model, parm_source: str) -> list[dict]:  # noqa: ARG001 - parm source is reported by caller
    # Delegate to the bridge's own tested per-test L2 flow (arm/takeoff/inject/verify).
    # A previous custom signal.alarm harness here reinvented the per-test flow and
    # under-reported passes (1/6 vs the bridge-native 5/6) — the bridge path is
    # authoritative, so we call it directly and map its TestResults to our dict.
    bridge, _ = _prepare_bridge_inputs(model)
    results = bridge.run_l2(per_test_sitl=True)
    return [
        {
            "req_id": r.req_id,
            "tier": "L2",
            "passed": bool(r.passed),
            "message": getattr(r, "message", ""),
            "duration_s": getattr(r, "duration_s", None),
        }
        for r in results
    ]


def _load_realization_summary() -> dict:
    if not RUN_JSON.exists():
        return {}
    data = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    realization = data.get("realization") or {}
    chosen = realization.get("chosen") or {}
    per_req = realization.get("per_requirement") or []
    speed_values = [r.get("realized_value") for r in per_req if r.get("scope") == "forward_flight" and r.get("family") == "speed"]
    range_values = [r.get("realized_value") for r in per_req if r.get("scope") == "forward_flight" and r.get("family") == "range"]
    return {
        "recommended_by": data.get("recommended_by"),
        "verdict": realization.get("verdict"),
        "combo": chosen.get("combo"),
        "pack": chosen.get("pack"),
        "frame": chosen.get("frame"),
        "endurance_min": chosen.get("endurance_min"),
        "total_mass_kg": chosen.get("total_mass_kg"),
        "forward_flight_ok": realization.get("forward_flight_ok"),
        "max_speed_mps": max(speed_values) if speed_values else None,
        "range_km": max(range_values) / 1000.0 if range_values else None,
    }


def _write_reports(report: dict) -> None:
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    l2 = report.get("safety_l2", [])
    l2_ok = sum(1 for r in l2 if r.get("passed"))
    realization = report.get("realization_summary") or {}
    endurance = realization.get("endurance_min")
    mass = realization.get("total_mass_kg")
    speed = realization.get("max_speed_mps")
    range_km = realization.get("range_km")
    lines = [
        "# SITL Feasibility Report",
        "",
        f"- Model source: {report.get('model_source')} ({report.get('model_source_note')})",
        f"- Parameter source: {report.get('parm_source')}",
        "- Honesty redline: native SITL does not validate datasheet endurance; hover current is architecture-nondiscriminating.",
        "",
        "## SITL-verified",
        f"- Flight feasibility: {'PASS' if report['flight']['passed'] else 'FAIL'} — {report['flight'].get('message', '')}",
        f"- Arm: {report['flight'].get('arm')}  Takeoff: {report['flight'].get('takeoff')}  Max climb: {report['flight'].get('max_climb_m')} m",
        f"- Hover: {report['flight'].get('hover')}",
        f"- L2 safety: {l2_ok}/{len(l2)} passed",
        "",
        "## Datasheet-verified (not true flight)",
        f"- Realization verdict: {realization.get('verdict', 'unknown')}",
        f"- Components: {realization.get('combo', 'unknown')} / {realization.get('pack', 'unknown')} / {realization.get('frame', 'unknown')}",
        f"- Endurance: {endurance:.2f} min" if endurance is not None else "- Endurance: unavailable",
        f"- Mass: {mass:.3f} kg" if mass is not None else "- Mass: unavailable",
        "- native SITL hover current is architecture-nondiscriminating; do not treat it as datasheet endurance validation.",
        "",
        "## Forward-flight lumped (not true flight)",
        f"- forward_flight_ok: {realization.get('forward_flight_ok')}",
        f"- Speed estimate: {speed:.1f} m/s" if speed is not None else "- Speed estimate: unavailable",
        f"- Range estimate: {range_km:.3f} km" if range_km is not None else "- Range estimate: unavailable",
        "- speed/range come from the lumped momentum-theory forward_flight tier; Gazebo calibration remains future work.",
        "",
        "## L2 Details",
    ]
    for r in l2:
        suffix = f" ({r['note']})" if r.get("note") else ""
        lines.append(f"- {r['req_id']}: {'PASS' if r['passed'] else 'FAIL'} — {r['message']}{suffix}")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    model, model_source, model_source_note = _read_model()
    bridge, parm_source = _prepare_bridge_inputs(model)
    report = {
        "model_source": model_source,
        "model_source_note": model_source_note,
        "parm_source": parm_source,
        "honesty_redline": (
            "Native SITL verifies arm/takeoff/stable hover and L2 safety behavior only; "
            "it does not validate datasheet endurance because native SITL hover current is architecture-nondiscriminating."
        ),
        "realization_summary": _load_realization_summary(),
    }
    print("=== SITL flight feasibility ===", flush=True)
    report["flight"] = run_flight_feasibility(bridge)
    print(json.dumps(report["flight"], indent=2), flush=True)
    print("\n=== SITL L2 safety ===", flush=True)
    report["safety_l2"] = run_safety_l2(model, parm_source)
    for item in report["safety_l2"]:
        print(f"{item['req_id']}: {'PASS' if item['passed'] else 'FAIL'} — {item['message']}")
    _write_reports(report)
    print(f"\nWrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    return 0 if report["flight"].get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
