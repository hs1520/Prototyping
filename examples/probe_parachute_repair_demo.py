"""Route-2 repair demonstration for the SAFE-005 parachute traceability block.

Sequence: archived model (gate withholds the check) -> one-payload repair on a
copy -> gate reopens -> the withheld L2 executes live in native SITL. The
archived bundle is not modified; the repair lands in labelled derived artefacts
(``final_model.repaired.sysml`` + ``repair.diff``). Gate decisions come from
``RequirementLinker`` on both models, the executable check runs through
``SITLBridge._run_single_test`` (the path examples/run_sitl_feasibility.py
uses), and the recorder observes on SITL's second MAVLink port (tcp:5762), so
the check under test is uninstrumented. This shows the gate and the actuator
chain, not pipeline self-repair.

Run:
  PYTHONPATH=. .venv/bin/python examples/probe_parachute_repair_demo.py
"""
from __future__ import annotations

import difflib
import json
import threading
import time
from pathlib import Path

from pymavlink import mavutil

from src.sitl.requirement_linker import RequirementLinker
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE, SITLBridge
from src.sysml.lite_model import build_lite_model
from examples.run_sitl_feasibility import merge_parm_lines

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples/output/runs/5af6c666-7654-4ebe-b79e-75771895e7e5"
DEMO = ROOT / "examples/output/parachute_repair_demo"
SITL_WORK = DEMO / "sitl"
MODEL_NAME = "AutonomousDrone"

DEFECT = "send CriticalPropulsionFailure() to parachuteCmd;"
REPAIR = "send ParachuteCmdData() to parachuteCmd;"

_LOG: list[str] = []
_T0 = time.monotonic()


def log(line: str) -> None:
    stamp = time.monotonic() - _T0
    entry = f"[{stamp:7.2f}s] {line}"
    _LOG.append(entry)
    print(entry, flush=True)


def gate_snapshot(tag: str, text: str) -> dict:
    """Linker run: what the traceability gate decides for SAFE-005."""
    model = build_lite_model(text, model_name=MODEL_NAME)
    ev = RequirementLinker(model, llm=None).compile_evidence()
    mismatches = [dict(m) for m in ev.traceability_mismatches]
    safe5 = [
        {
            "tier": s.tier,
            "inject": s.inject.kind,
            "verify": s.verify.kind,
            "verify_args": dict(s.verify.args),
            "params": [
                {"name": p.param_name, "value": p.value, "source": p.source}
                for p in s.params
            ],
        }
        for s in ev.test_specs
        if s.req_id == "REQ_SAFE_005"
    ]
    l2_ids = sorted(s.req_id for s in ev.test_specs if s.tier == "L2")
    log(f"[{tag}] traceability mismatches: {len(mismatches)}")
    for m in mismatches:
        log(f"[{tag}]   {m.get('req_id')}: {m.get('message')}")
    log(f"[{tag}] executable L2 requirements: {l2_ids}")
    for s in safe5:
        log(f"[{tag}] REQ_SAFE_005 spec: tier={s['tier']} "
            f"inject={s['inject']} verify={s['verify']} args={s['verify_args']}")
    return {"mismatches": mismatches, "safe5_specs": safe5, "l2_ids": l2_ids}


class Recorder:
    """Side-channel observer on SITL's second MAVLink port (tcp:5762)."""

    def __init__(self):
        self.samples: list[dict] = []
        self.connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            mav = mavutil.mavlink_connection("tcp:127.0.0.1:5762")
            mav.wait_heartbeat(timeout=15)
            mav.mav.request_data_stream_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
            )
            self.connected = True
            log("[recorder] connected on tcp:5762 (side channel; test untouched)")
        except Exception as exc:  # noqa: BLE001 - recorder is best-effort
            log(f"[recorder] side channel unavailable: {exc}")
            return
        last_mode = None
        while not self._stop.is_set():
            msg = mav.recv_match(
                type=["SERVO_OUTPUT_RAW", "GLOBAL_POSITION_INT", "HEARTBEAT"],
                blocking=True, timeout=1,
            )
            if msg is None:
                continue
            t = time.monotonic() - _T0
            kind = msg.get_type()
            if kind == "SERVO_OUTPUT_RAW":
                self.samples.append(
                    {"t": round(t, 2), "servo8": int(msg.servo8_raw)})
            elif kind == "GLOBAL_POSITION_INT":
                self.samples.append(
                    {"t": round(t, 2), "alt": round(msg.relative_alt / 1000.0, 2)})
            elif kind == "HEARTBEAT":
                mode = mavutil.mode_string_v10(msg)
                if mode != last_mode:
                    self.samples.append({"t": round(t, 2), "mode": mode})
                    log(f"[recorder] mode -> {mode}")
                    last_mode = mode
        try:
            mav.close()
        except Exception:  # noqa: BLE001
            pass

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)
    SITL_WORK.mkdir(parents=True, exist_ok=True)

    archived_text = (BUNDLE / "final_model.sysml").read_text(encoding="utf-8")
    log(f"archived model: {BUNDLE / 'final_model.sysml'}")
    log(f"defect line   : {DEFECT}")

    log("=== PHASE A: gate on the archived model ===")
    before = gate_snapshot("before", archived_text)

    log("=== PHASE B: one-payload repair (new derived artefact) ===")
    occurrences = archived_text.count(DEFECT)
    if occurrences != 1:
        log(f"ABORT: expected exactly 1 defect occurrence, found {occurrences}")
        return 2
    repaired_text = archived_text.replace(DEFECT, REPAIR)
    (DEMO / "final_model.repaired.sysml").write_text(
        repaired_text, encoding="utf-8")
    diff = "".join(difflib.unified_diff(
        archived_text.splitlines(keepends=True),
        repaired_text.splitlines(keepends=True),
        fromfile="archived/final_model.sysml",
        tofile="demo/final_model.repaired.sysml",
    ))
    (DEMO / "repair.diff").write_text(diff, encoding="utf-8")
    log("repair applied on a COPY; archived bundle untouched")
    for line in diff.splitlines():
        if line.startswith(("-", "+")) and "send" in line:
            log(f"  {line.rstrip()}")

    log("=== PHASE C: gate on the repaired model ===")
    after = gate_snapshot("after", repaired_text)
    safe5_l2 = [s for s in after["safe5_specs"] if s["tier"] == "L2"]
    if not safe5_l2:
        log("ABORT: repaired model did not produce an executable SAFE_005 spec")
        return 2

    log("=== PHASE D: execute the previously withheld L2 in native SITL ===")
    repaired_model = build_lite_model(repaired_text, model_name=MODEL_NAME)
    bridge = SITLBridge(
        model=repaired_model,
        output_dir=str(SITL_WORK),
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        fdm_backend="native",
        verbose=False,
    )
    primary = (BUNDLE / "recommended.parm").read_text(encoding="utf-8").splitlines()
    supplemental = bridge.requirement_evidence.parm_file.splitlines()
    parm_lines = merge_parm_lines(primary, supplemental)
    (SITL_WORK / f"{MODEL_NAME}.parm").write_text(
        "\n".join(parm_lines) + "\n", encoding="utf-8")
    log("boot parm: archived recommended.parm + repaired-model linker params")

    spec = next(
        s for s in bridge.requirement_evidence.test_specs
        if s.req_id == "REQ_SAFE_005" and s.tier == "L2"
    )
    log(f"spec: inject={spec.inject.kind} (MAV_CMD_DO_PARACHUTE, "
        f"pre_takeoff={spec.inject.pre_takeoff_m} m) "
        f"verify={spec.verify.kind} args={dict(spec.verify.args)}")

    eeprom = Path.cwd() / "eeprom.bin"
    if eeprom.exists():
        eeprom.unlink()

    launched = False
    for attempt in range(2):
        launched = bridge.launch_sitl(wait_s=45.0)
        if launched:
            break
        bridge.stop_sitl()
        if attempt == 0:
            log("SITL launch retry ...")
            time.sleep(2.0)
    if not launched:
        log("ABORT: SITL failed to launch")
        return 2
    log("SITL up, EKF ready")

    recorder = Recorder()
    recorder.start()
    time.sleep(2.0)

    t_test = time.monotonic() - _T0
    log("running bridge._run_single_test (the real L2 runner) ...")
    passed, message = bridge._run_single_test(  # noqa: SLF001 - demo drives the real runner
        spec, "tcp:127.0.0.1:5760", mavutil, fresh_sitl=True,
    )
    t_done = time.monotonic() - _T0
    log(f"RESULT: {'PASS' if passed else 'FAIL'} — {message} "
        f"({t_done - t_test:.1f}s)")

    time.sleep(2.0)
    recorder.stop()
    bridge.stop_sitl()

    result = {
        "defect": DEFECT,
        "repair": REPAIR,
        "gate_before": before,
        "gate_after": {k: v for k, v in after.items()},
        "l2_result": {"passed": bool(passed), "message": message,
                      "window_s": [round(t_test, 2), round(t_done, 2)]},
        "recorder_connected": recorder.connected,
        "claim": (
            "Instrument demonstration: the traceability gate reopens after a "
            "one-payload model revision and the previously withheld check "
            "executes and passes. Hand-applied repair on a labelled copy; not "
            "pipeline self-repair; archived bundle untouched."
        ),
    }
    (DEMO / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEMO / "servo8_series.json").write_text(
        json.dumps(recorder.samples, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEMO / "demo.log").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    log(f"artifacts written to {DEMO}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
