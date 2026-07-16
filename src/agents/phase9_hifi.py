"""Phase 9 seam: connect the Phase 8 recommendation to the high-fidelity runners.

This bridge connects an explicitly requested recommendation to native SITL
feasibility and/or Gazebo dynamics.  It is default-OFF: authoritative runs are
owned by ``examples/run_realization_report.py``, which first freezes one base
bundle and then collects every evidence layer into that same bundle.  This module
is deliberately kept out of
``orchestrator`` so the launch mechanics (artifact persistence + subprocess to the
mature standalone runners) are isolated and easy to monkeypatch in tests.

Honesty invariants preserved here:
  * native SITL verifies arm/takeoff/hover + L2 safety, NOT endurance;
  * Gazebo verifies high-fidelity dynamics, NOT endurance;
  * datasheet CLOSED (Phase 8) is never upgraded by these runs;
  * environment absence (no Docker / no arducopter) is reported as an honest
    "skipped", never faked into a pass.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..prototyping.artifact_provenance import sha256_json, validate_run_provenance
from ..prototyping.artifact_store import (
    OUTPUT_DIR_ENV, atomic_write_json, atomic_write_text, output_dir,
)

# Runners are invoked as subprocesses (src must not import examples/), so the
# module dependency direction stays clean.
_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = output_dir()
_SITL_RUNNER = _ROOT / "examples" / "run_sitl_feasibility.py"
_GAZEBO_RUNNER = _ROOT / "examples" / "run_gazebo_feasibility.py"

VALID_MODES = ("sitl", "gazebo", "both")


def _layers_for(mode: str) -> List[str]:
    mode = (mode or "").strip().lower()
    if mode == "both":
        # Gazebo must precede SITL: the SITL runner is the sole writer of the
        # executed verification matrix and can therefore incorporate both L3
        # dynamics evidence and its own L1/L2 evidence in one final matrix.
        return ["gazebo", "sitl"]
    if mode in ("sitl", "gazebo"):
        return [mode]
    return []


def _persist_recommendation(design, model_text: str, output_dir: Path) -> None:
    """Write the artifacts the standalone runners read, CONSISTENT with the design
    being flown: final model, recommended design inputs, and recommended.parm.

    Writing recommended.parm here (not just the design inputs) is required: the
    SITL runner's stale-guard refuses to fly a .parm that does not match the
    latest recommended design, so a fresh run whose recommendation differs from a
    previous run's leftover .parm would otherwise be rejected."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json = output_dir / "realization_run.json"
    existing: Dict[str, Any] = {}
    if run_json.exists():
        try:
            existing = json.loads(run_json.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
    if existing.get("artifact_provenance"):
        ok, reason = validate_run_provenance(existing, model_sysml=model_text or "")
        if not ok:
            raise RuntimeError(f"refusing to mutate authoritative base bundle: {reason}")
        expected = existing.get("recommended_design_inputs")
        actual = dict(vars(design)) if design is not None else None
        if sha256_json(expected) != sha256_json(actual):
            raise RuntimeError("refusing Phase 9 design that differs from authoritative run")
        return  # authoritative base artifacts are immutable during evidence collection
    atomic_write_text(output_dir / "final_model.sysml", model_text or "")
    existing["recommended_design_inputs"] = dict(vars(design)) if design is not None else None
    atomic_write_json(run_json, existing)
    if design is not None:
        try:
            from ..sitl.dse_sitl_params import design_to_sitl_parm
            from ..sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE
            lines = list(design_to_sitl_parm(design))
            present = {ln.split()[0] for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}
            for key, value in (ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {}).items():
                if key not in present:
                    lines.append(f"{key:<20} {value}")
            atomic_write_text(
                output_dir / "recommended.parm",
                "# Phase 9: recommended design SITL params; native SITL is "
                "architecture-nondiscriminating for endurance.\n" + "\n".join(lines) + "\n",
            )
        except Exception:
            # If .parm generation fails, remove any stale one so the runner
            # reconstructs from realization_run.json rather than refusing.
            stale = output_dir / "recommended.parm"
            if stale.exists():
                stale.unlink()


def _env_available(layer: str) -> Optional[str]:
    """Return a human reason when the layer's environment is unavailable, else None."""
    if layer == "sitl":
        server = os.path.expanduser("~/ardupilot/build/sitl/bin/arducopter")
        local = os.path.expanduser("~/PycharmProjects/ardupilot/build/sitl/bin/arducopter")
        if not (os.path.exists(server) or os.path.exists(local)):
            return "arducopter SITL binary not found"
        return None
    if layer == "gazebo":
        if shutil.which("docker") is None:
            return "docker not available"
        return None
    return f"unknown layer {layer!r}"


def _run_subprocess(script: Path, extra_args: List[str], timeout_s: int,
                    output_dir: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(script), *extra_args],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONPATH": str(_ROOT), OUTPUT_DIR_ENV: str(output_dir)},
    )
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]}


def _read_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def run_layer(layer: str, design, model_text: str,
              output_dir: Path = _OUTPUT_DIR, timeout_s: int = 1800) -> Dict[str, Any]:
    """Run one high-fidelity layer. Honest about environment absence; never fakes."""
    reason = _env_available(layer)
    if reason is not None:
        return {"layer": layer, "status": "skipped", "reason": reason}

    _persist_recommendation(design, model_text, output_dir)
    if layer == "sitl":
        run = _run_subprocess(_SITL_RUNNER, [], timeout_s, output_dir)
        report = _read_report(output_dir / "sitl_feasibility_report.json")
        flight = (report or {}).get("flight") or {}
        safety = (report or {}).get("safety_verification") or {}
        # "ran" iff the runner produced a parseable report (a nonzero exit code
        # just signals flight-not-passed, NOT a crash); "error" only when no
        # report came back (crash / environment failure) — never faked.
        return {
            "layer": "sitl",
            "status": "ran" if report is not None else "error",
            "returncode": run["returncode"],
            "flight_passed": flight.get("passed"),
            "safety_status": safety.get("status"),
            "report": str(output_dir / "sitl_feasibility_report.md"),
            "redline": "native SITL verifies flight feasibility + L2 safety, not endurance",
        }
    # gazebo: exit 2 = a requirement FAILED (ran fine); "error" only when no report.
    run = _run_subprocess(_GAZEBO_RUNNER, [], timeout_s, output_dir)
    report = _read_report(output_dir / "gazebo_feasibility_report.json")
    return {
        "layer": "gazebo",
        "status": "ran" if report is not None else "error",
        "returncode": run["returncode"],
        "gazebo_status": (report or {}).get("status"),
        "report": str(output_dir / "gazebo_feasibility_report.md"),
        "redline": "Gazebo verifies high-fidelity dynamics, not endurance",
    }


def run_hifi_closure(mode: str, design, model_text: str,
                     requirements: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the requested high-fidelity layer(s) for the Phase 8 recommendation."""
    layers = _layers_for(mode)
    results = [run_layer(layer, design, model_text) for layer in layers]
    return {"mode": mode, "layers": results}
