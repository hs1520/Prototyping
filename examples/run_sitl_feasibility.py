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
import argparse
import re
import signal
import statistics
import time
from pathlib import Path

from pymavlink import mavutil

from src.dse.physics_estimator import DesignInputs
from src.realization.closure import close_the_loop
from src.prototyping.artifact_provenance import (
    evidence_reuse_allowed, validate_derived_provenance, validate_run_provenance,
)
from src.prototyping.artifact_store import (
    atomic_write_json,
    atomic_write_text,
    ensure_open_bundle,
    input_dir,
    latest_output_dir,
    output_dir,
)
from src.sitl.parameter_projection import (
    FRAME_CLASS_BY_ROTOR_COUNT,
    design_parm_lines,
)
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE, SITLBridge
from src.sitl.sitl_specs import TestContext
from src.sysml.lite_model import build_lite_model

ROOT = Path(__file__).resolve().parents[1]
OUT = output_dir()
INPUT = input_dir()
SYSML_PATH = INPUT / "final_model.sysml"
FALLBACK_SYSML_PATH = ROOT / "examples" / "drone_system_v2.sysml"
PARM_PATH = INPUT / "recommended.parm"
RUN_JSON = INPUT / "realization_run.json"
GAZEBO_REPORT_JSON = INPUT / "gazebo_feasibility_report.json"
WORK = OUT / "sitl_feasibility"
REPORT_JSON = OUT / "sitl_feasibility_report.json"
REPORT_MD = OUT / "sitl_feasibility_report.md"
STATIC_REPORT_JSON = OUT / "sitl_feasibility_static_report.json"
STATIC_REPORT_MD = OUT / "sitl_feasibility_static_report.md"
MATRIX_JSON = OUT / "verification_matrix.json"
MATRIX_MD = OUT / "verification_matrix.md"
MODEL_NAME = "AutonomousDrone"


def _read_model():
    if SYSML_PATH.exists():
        text = SYSML_PATH.read_text(encoding="utf-8")
        note = "current" if PARM_PATH.exists() else "existing_model_not_from_this_run"
        return build_lite_model(text, model_name=MODEL_NAME), str(SYSML_PATH), note
    raise FileNotFoundError(
        f"{SYSML_PATH} not found; provenance-safe SITL requires the exact model "
        "persisted by examples/run_realization_report.py"
    )


def _parm_lines_from_json() -> list[str]:
    data = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    d = data.get("recommended_design_inputs") or {}
    if not d:
        raise FileNotFoundError(f"{PARM_PATH} missing and {RUN_JSON} lacks recommended_design_inputs")
    di = DesignInputs(**d)
    return design_parm_lines(
        di,
        base_params=ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {},
    )


def _parm_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    return stripped.split()[0]


def merge_parm_lines(primary: list[str], supplemental: list[str]) -> list[str]:
    """Merge .parm defaults without overriding the recommended design params."""
    merged = list(primary)
    seen = {key for line in merged if (key := _parm_key(line))}
    additions = []
    for line in supplemental:
        key = _parm_key(line)
        if key is None or key in seen:
            continue
        additions.append(line)
        seen.add(key)
    if additions:
        merged.extend(["", "# Safety/L2 actuator params from requirement linker"])
        merged.extend(additions)
    return merged


def parm_freshness(primary_lines: list[str], run_json: dict | None,
                   model_sysml: str | None = None) -> tuple[bool, str]:
    """Cross-check recommended.parm against the latest realization run.

    Guards against silently flying a previous run's design: a pipeline run that
    falls back before Phase 8 leaves no recommendation, and an older
    recommended.parm on disk would otherwise be picked up as if it were current.
    Returns (fresh, reason).
    """
    if run_json is None:
        return False, "realization_run.json is required; standalone parm is untrusted"
    d = run_json.get("recommended_design_inputs")
    if not d:
        return False, ("latest realization_run.json has no recommended design "
                       "(DSE fell back before Phase 8); recommended.parm is from an earlier run")
    values: dict[str, float] = {}
    for line in primary_lines:
        key = _parm_key(line)
        if key is None:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                values[key] = float(parts[1])
            except ValueError:
                continue
    cap = float(d.get("battery_capacity_mah", 0.0))
    if "BATT_CAPACITY" in values and abs(values["BATT_CAPACITY"] - round(cap)) > 0.5:
        return False, (f"BATT_CAPACITY {values['BATT_CAPACITY']:.0f} does not match the "
                       f"latest recommended design ({cap:.0f} mAh)")
    fc = FRAME_CLASS_BY_ROTOR_COUNT.get(int(d.get("rotor_count", 0)))
    if fc is not None and "FRAME_CLASS" in values and int(values["FRAME_CLASS"]) != fc:
        return False, (f"FRAME_CLASS {int(values['FRAME_CLASS'])} does not match the latest "
                       f"recommended rotor count ({int(d.get('rotor_count', 0))} → {fc})")
    if model_sysml is None and SYSML_PATH.exists():
        model_sysml = SYSML_PATH.read_text(encoding="utf-8")
    parm_text = "\n".join(primary_lines) + "\n"
    fresh, reason = validate_run_provenance(
        run_json, model_sysml=model_sysml, parm_text=parm_text,
    )
    if not fresh:
        return False, reason
    return True, "consistent with the latest realization_run.json recommended design"


def _prepare_bridge_inputs(model, allow_stale: bool = False) -> tuple[SITLBridge, str]:
    WORK.mkdir(parents=True, exist_ok=True)
    bridge_parm = WORK / f"{MODEL_NAME}.parm"
    run_json = json.loads(RUN_JSON.read_text(encoding="utf-8")) if RUN_JSON.exists() else None
    bridge = SITLBridge(
        model=model,
        output_dir=str(WORK),
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        fdm_backend="native",
        verbose=False,
    )
    source = f"{PARM_PATH} + requirement_linker safety params"
    if PARM_PATH.exists():
        primary = PARM_PATH.read_text(encoding="utf-8").splitlines()
        fresh, reason = parm_freshness(primary, run_json)
        if not fresh and not allow_stale:
            raise SystemExit(
                f"recommended.parm is STALE: {reason}\n"
                "Re-run examples/run_realization_report.py to regenerate it, "
                "or pass --allow-stale to proceed (the report will be marked STALE)."
            )
        if not fresh:
            source += f" [STALE: {reason}]"
    else:
        source = "reconstructed_from_existing_realization_run.json + requirement_linker safety params"
        if not RUN_JSON.exists():
            raise SystemExit("realization_run.json is required to reconstruct recommended.parm")
        run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
        model_sysml = SYSML_PATH.read_text(encoding="utf-8") if SYSML_PATH.exists() else None
        fresh, reason = validate_run_provenance(run_json, model_sysml=model_sysml)
        if not fresh:
            raise SystemExit(f"STALE artifact set: {reason}")
        primary = _parm_lines_from_json()
    supplemental = bridge.requirement_evidence.parm_file.splitlines()
    lines = merge_parm_lines(primary, supplemental)
    atomic_write_text(bridge_parm, "\n".join(lines) + "\n")
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


def _l2_note(req_id: str, passed: bool, message: str, inject_kind: str = "") -> str:  # noqa: ARG001
    if not passed and "L2 case timeout" in message:
        return "bounded SITL run timed out; not counted as a safety pass"
    if passed and inject_kind == "mavlink_command":
        # Parachute/gripper cases command the actuator directly instead of
        # injecting the fault condition: they verify actuation exists, not the
        # detect→react chain (that chain is the behavioral-sim tier's job).
        return "actuator-existence check; fault-trigger chain covered by behavioral sim tier"
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
        for attempt in range(2):
            launched = bridge.launch_sitl(wait_s=45.0)
            if launched:
                break
            bridge.stop_sitl()
            if attempt == 0:
                print("  ↻ SITL 启动/EKF 就绪失败，重试一次 ...", flush=True)
                time.sleep(2.0)
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
            "note": _l2_note(spec.req_id, bool(passed), message, spec.inject.kind),
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
            "note": _l2_note(spec.req_id, False, message, spec.inject.kind),
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if launched:
            bridge.stop_sitl()


def run_safety_l2(model, parm_source: str, allow_stale: bool = False) -> list[dict]:  # noqa: ARG001 - parm source is reported by caller
    bridge, _ = _prepare_bridge_inputs(model, allow_stale=allow_stale)
    specs = [s for s in bridge.requirement_evidence.test_specs if s.tier == "L2"]
    results = []
    for spec in specs:
        result = _run_single_l2_with_timeout(bridge, spec)
        result["model_guard"] = _model_guard(bridge, spec.req_id)
        results.append(result)
    return results


def _model_guard(bridge: SITLBridge, req_id: str) -> str | None:
    """The model guard a spec traces to — grounds 'behavioral sim covers the trigger'."""
    assigned = bridge.requirement_evidence.guard_assignments.get(req_id)
    if not assigned:
        return None
    if assigned.kind == "bool_true" or not assigned.operator or assigned.threshold is None:
        return f"{assigned.attribute} (bool)"
    return f"{assigned.attribute} {assigned.operator} {assigned.threshold}"


def coverage_summary(bridge: SITLBridge) -> dict:
    return bridge.requirement_evidence.coverage_payload()


def matrix_summary(model, bridge: SITLBridge,
                   l1_results=None, l2_results=None) -> dict | None:
    """Verification-matrix counts (best-effort): tiers make 'unmapped' interpretable."""
    try:
        from src.prototyping.verification_matrix import (
            build_matrix, summarize, to_json, to_markdown,
        )

        run_json = None
        realization = None
        if RUN_JSON.exists():
            run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
            realization = run_json.get("realization")
        gazebo = _fresh_gazebo_report(run_json)
        rows = build_matrix(
            model, realization, bridge.requirement_evidence, gazebo=gazebo,
            l1_results=l1_results, l2_results=l2_results,
        )
        # The live SITL run is the last evidence-producing tier, so it owns the
        # final matrix artifact.  Persist rows with executed L1/L2 results instead
        # of leaving the earlier strategy-only matrix (all SITL checks "planned").
        matrix_payload = to_json(rows)
        if RUN_JSON.exists():
            run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
            matrix_payload["source_provenance"] = run_json.get("artifact_provenance")
        atomic_write_json(MATRIX_JSON, matrix_payload)
        atomic_write_text(MATRIX_MD, to_markdown(rows))
        return summarize(rows)
    except Exception:
        return None


def _fresh_gazebo_report(run_json: dict | None) -> dict | None:
    if not GAZEBO_REPORT_JSON.exists():
        return None
    report = json.loads(GAZEBO_REPORT_JSON.read_text(encoding="utf-8"))
    if not run_json or not SYSML_PATH.exists():
        return None
    fresh, _ = validate_derived_provenance(
        report, run_json, SYSML_PATH.read_text(encoding="utf-8")
    )
    return report if fresh else None


def _matrix_lines(matrix: dict | None) -> list[str]:
    if not matrix:
        return []
    st = matrix.get("by_status", {})
    return [
        "- Verification strategy matrix: "
        f"{st.get('verified', 0)}/{matrix.get('total', 0)} verified across tiers, "
        f"{st.get('partial', 0)} partial, "
        f"{st.get('planned', 0)} planned (Gazebo), "
        f"{st.get('failed', 0)} failed (Gazebo), "
        f"{st.get('out-of-sim-scope', 0)} inspection/analysis, "
        f"{st.get('blocked', 0)} blocked, "
        f"{st.get('unassigned', 0)} unassigned — see verification_matrix.md",
    ]


def traceability_results(bridge: SITLBridge) -> list[dict]:
    return [
        {
            "req_id": r.req_id,
            "tier": r.tier,
            "passed": bool(r.passed),
            "message": r.message,
            "duration_s": getattr(r, "duration_s", None),
        }
        for r in bridge.validate_traceability()
    ]


def planned_l2_specs(bridge: SITLBridge) -> list[dict]:
    return [
        {
            "req_id": s.req_id,
            "tier": s.tier,
            "inject": s.inject.kind,
            "verify": s.verify.kind,
            "pre_takeoff_m": s.inject.pre_takeoff_m,
            "notes": s.notes,
            "model_guard": _model_guard(bridge, s.req_id),
        }
        for s in bridge.requirement_evidence.test_specs
        if s.tier == "L2"
    ]


def build_static_report(model, model_source: str, model_source_note: str,
                        allow_stale: bool = False) -> dict:
    bridge, parm_source = _prepare_bridge_inputs(model, allow_stale=allow_stale)
    trace = traceability_results(bridge)
    planned_l2 = planned_l2_specs(bridge)
    return {
        "model_source": model_source,
        "model_source_note": model_source_note,
        "parm_source": parm_source,
        "static_only": True,
        "honesty_redline": (
            "Dry-run only: no arducopter process was launched, no flight occurred, "
            "and no L2 behavior was verified."
        ),
        "traceability": trace,
        "planned_l2": planned_l2,
        "planned_l2_total": len(planned_l2),
        "traceability_blocked": sum(1 for r in trace if not r.get("passed")),
        "coverage": coverage_summary(bridge),
        "verification_matrix": matrix_summary(
            model, bridge, l1_results=bridge.validate_l1(), l2_results=[]
        ),
        "realization_summary": _load_realization_summary(),
    }


def _coverage_lines(coverage: dict) -> list[str]:
    """MD lines for the honest-gap bucket, so 'blocked = 0' is not over-read."""
    n = coverage.get("unmapped", 0)
    ids = ", ".join(coverage.get("unmapped_req_ids", [])) or "none"
    return [
        f"- Requirements with no SITL mapping: {n}/{coverage.get('satisfied', 0)} "
        f"satisfied (honest gap — not verified at any SITL tier): {ids}",
    ]


def _write_static_report(report: dict) -> None:
    atomic_write_json(STATIC_REPORT_JSON, report)
    lines = [
        "# SITL Feasibility Static Plan",
        "",
        f"- Model source: {report.get('model_source')} ({report.get('model_source_note')})",
        f"- Parameter source: {report.get('parm_source')}",
        "- Dry-run only: no arducopter process was launched and no flight/L2 behavior was verified.",
        f"- Traceability blocked: {report.get('traceability_blocked')}",
        f"- Planned executable L2 checks: {report.get('planned_l2_total')}",
        *_coverage_lines(report.get("coverage", {})),
        *_matrix_lines(report.get("verification_matrix")),
        "",
        "## Traceability blocked",
    ]
    trace = report.get("traceability", [])
    if trace:
        for r in trace:
            lines.append(f"- {r['req_id']}: {r['message']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Planned L2 Checks"])
    for r in report.get("planned_l2", []):
        guard = f", guard={r['model_guard']}" if r.get("model_guard") else ""
        lines.append(
            f"- {r['req_id']}: inject={r['inject']}, verify={r['verify']}, "
            f"pre_takeoff_m={r['pre_takeoff_m']}{guard}"
        )
    atomic_write_text(STATIC_REPORT_MD, "\n".join(lines) + "\n")


def safety_verification_status(l2: list[dict], trace: list[dict]) -> dict:
    blocked = sum(1 for r in trace if not r.get("passed"))
    l2_total = len(l2)
    l2_passed = sum(1 for r in l2 if r.get("passed"))
    if blocked and l2_total:
        status = "PARTIAL"
        message = (
            f"{blocked} requirement(s) blocked by traceability mismatch; "
            f"{l2_passed}/{l2_total} executable L2 checks passed."
        )
    elif blocked:
        status = "BLOCKED"
        message = f"{blocked} requirement(s) blocked by traceability mismatch; no trustworthy L2 checks executed."
    elif l2_total == 0:
        status = "NOT_RUN"
        message = "No traceability blocks and no executable L2 checks were run."
    elif l2_passed == l2_total:
        status = "PASS"
        message = f"All executable L2 checks passed ({l2_passed}/{l2_total})."
    else:
        status = "FAIL"
        message = f"Executable L2 checks failed ({l2_passed}/{l2_total} passed)."
    return {
        "status": status,
        "l2_passed": l2_passed,
        "l2_total": l2_total,
        "traceability_blocked": blocked,
        "message": message,
    }


def _load_realization_summary() -> dict:
    if not RUN_JSON.exists():
        return {}
    data = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    recomputed = _recompute_realization_summary(data)
    if recomputed:
        return recomputed
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


def _requirements_from_sysml(text: str) -> list[str]:
    out = []
    for m in re.finditer(
        r"requirement\s+def\s+([A-Za-z_][\w]*)\s*\{(?P<body>.*?)\}",
        text,
        re.S,
    ):
        doc = re.search(r"doc\s*/\*(.*?)\*/", m.group("body"), re.S)
        if doc:
            out.append(f"{m.group(1).replace('_', '-')}: {doc.group(1).strip()}")
    return out


def _recompute_realization_summary(data: dict) -> dict:
    """Best-effort Phase-8 recompute so reports use the current realization code."""
    try:
        design_inputs = data.get("recommended_design_inputs") or {}
        if not design_inputs or not SYSML_PATH.exists():
            return {}
        requirements = _requirements_from_sysml(SYSML_PATH.read_text(encoding="utf-8"))
        rep = close_the_loop(DesignInputs(**design_inputs), [], requirements)
        chosen = rep.chosen
        if chosen is None:
            return {
                "recommended_by": data.get("recommended_by"),
                "verdict": rep.verdict,
                "forward_flight_ok": rep.forward_flight_ok,
            }
        speed_values = [
            v.realized_value for v in rep.per_requirement
            if v.scope == "forward_flight" and v.family == "speed" and v.realized_value is not None
        ]
        range_values = [
            v.realized_value for v in rep.per_requirement
            if v.scope == "forward_flight" and v.family == "range" and v.realized_value is not None
        ]
        return {
            "recommended_by": data.get("recommended_by"),
            "verdict": rep.verdict,
            "combo": chosen.rd.combo.name,
            "pack": chosen.rd.pack.name,
            "frame": chosen.rd.frame.name,
            "endurance_min": chosen.metrics.endurance_min,
            "total_mass_kg": chosen.metrics.total_mass_kg,
            "forward_flight_ok": rep.forward_flight_ok,
            "max_speed_mps": max(speed_values) if speed_values else None,
            "range_km": max(range_values) / 1000.0 if range_values else None,
        }
    except Exception:
        return {}


def _write_reports(report: dict) -> None:
    atomic_write_json(REPORT_JSON, report)
    l2 = report.get("safety_l2", [])
    l2_ok = sum(1 for r in l2 if r.get("passed"))
    safety = report.get("safety_verification") or safety_verification_status(
        l2, report.get("traceability", [])
    )
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
        f"- L2 safety status: {safety.get('status')} — {safety.get('message')}",
        f"- Executable L2 safety: {l2_ok}/{len(l2)} passed",
        *_coverage_lines(report.get("coverage", {})),
        *_matrix_lines(report.get("verification_matrix")),
        "",
        "## Traceability blocked",
    ]
    trace = report.get("traceability", [])
    if trace:
        for r in trace:
            lines.append(f"- {r['req_id']}: {r['message']}")
    else:
        lines.append("- none")
    lines.extend([
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
        "- note: actuator-existence cases verify actuation only; each case's "
        "fault-trigger guard (shown below) exists in the model and is exercised "
        "at the behavioral-sim tier.",
    ])
    for r in l2:
        suffix = f" ({r['note']})" if r.get("note") else ""
        guard = f" [guard: {r['model_guard']}]" if r.get("model_guard") else ""
        lines.append(f"- {r['req_id']}: {'PASS' if r['passed'] else 'FAIL'} — {r['message']}{suffix}{guard}")
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the traceability/L2 execution plan; do not launch native SITL.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Proceed even if recommended.parm does not match the latest realization "
             "run (parm_source is then marked STALE in the reports).",
    )
    args = parser.parse_args(argv)
    ensure_open_bundle(OUT)
    model, model_source, model_source_note = _read_model()
    if args.dry_run:
        report = build_static_report(model, model_source, model_source_note,
                                     allow_stale=args.allow_stale)
        _write_static_report(report)
        print(json.dumps({
            "traceability_blocked": report["traceability_blocked"],
            "planned_l2_total": report["planned_l2_total"],
            "static_report": str(STATIC_REPORT_JSON),
        }, indent=2), flush=True)
        return 0
    bridge, parm_source = _prepare_bridge_inputs(model, allow_stale=args.allow_stale)
    run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    current_model = SYSML_PATH.read_text(encoding="utf-8")
    try:
        previous_dir = latest_output_dir()
        previous_run = json.loads(
            (previous_dir / "realization_run.json").read_text(encoding="utf-8")
        )
        previous_model = (previous_dir / "final_model.sysml").read_text(
            encoding="utf-8"
        )
        previous_report = json.loads(
            (previous_dir / "sitl_feasibility_report.json").read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError:
        previous_dir = INPUT.resolve()
        previous_run = previous_model = previous_report = None
    if previous_dir != INPUT.resolve():
        current_ids = {
            str(spec.req_id).upper().replace("-", "_")
            for spec in bridge.requirement_evidence.test_specs
            if spec.tier == "L2"
        }
        previous_ids = {
            str(item.get("req_id") or "").upper().replace("-", "_")
            for item in previous_report.get("safety_l2", ())
            if item.get("passed") is True
        }
        if (
            (previous_report.get("flight") or {}).get("passed") is True
            and previous_ids == current_ids
            and previous_report.get("source_provenance")
            == previous_run.get("artifact_provenance")
            and evidence_reuse_allowed(
                previous_run,
                run_json,
                previous_model,
                current_model,
                current_ids,
                require_parm_match=True,
            )
        ):
            report = dict(previous_report)
            report.update({
                "model_source": model_source,
                "model_source_note": model_source_note,
                "parm_source": parm_source,
                "source_provenance": run_json.get("artifact_provenance"),
                "evidence_origin_provenance": previous_run.get(
                    "artifact_provenance"
                ),
                "reused_from_run_id": (
                    previous_run.get("artifact_provenance") or {}
                ).get("run_id"),
                "coverage": coverage_summary(bridge),
                "traceability": traceability_results(bridge),
                "safety_l2": [
                    {**item, "evidence_reused": True}
                    for item in previous_report.get("safety_l2", ())
                ],
                "flight": {
                    **(previous_report.get("flight") or {}),
                    "evidence_reused": True,
                },
            })
            report["verification_matrix"] = matrix_summary(
                model,
                bridge,
                l1_results=bridge.validate_l1(),
                l2_results=report["safety_l2"],
            )
            report["safety_verification"] = safety_verification_status(
                report["safety_l2"], report["traceability"]
            )
            _write_reports(report)
            print(
                "Reused unchanged PASS flight and L2 evidence from "
                f"{report['reused_from_run_id']}",
                flush=True,
            )
            return 0
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
    if RUN_JSON.exists():
        try:
            run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
            report["source_provenance"] = run_json.get("artifact_provenance")
        except (OSError, ValueError):
            report["source_provenance"] = None
    report["coverage"] = coverage_summary(bridge)
    report["traceability"] = traceability_results(bridge)
    if report["traceability"]:
        print("=== Traceability blocked ===", flush=True)
        for item in report["traceability"]:
            print(f"{item['req_id']}: {item['message']}", flush=True)
    print("=== SITL flight feasibility ===", flush=True)
    report["flight"] = run_flight_feasibility(bridge)
    print(json.dumps(report["flight"], indent=2), flush=True)
    print("\n=== SITL L2 safety ===", flush=True)
    report["safety_l2"] = run_safety_l2(model, parm_source, allow_stale=args.allow_stale)
    report["verification_matrix"] = matrix_summary(
        model,
        bridge,
        l1_results=bridge.validate_l1(),
        l2_results=report["safety_l2"],
    )
    report["safety_verification"] = safety_verification_status(
        report["safety_l2"], report["traceability"]
    )
    for item in report["safety_l2"]:
        print(f"{item['req_id']}: {'PASS' if item['passed'] else 'FAIL'} — {item['message']}")
    print(
        f"Safety verification: {report['safety_verification']['status']} — "
        f"{report['safety_verification']['message']}",
        flush=True,
    )
    _write_reports(report)
    print(f"\nWrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    return 0 if report["flight"].get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
