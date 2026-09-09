"""Rebuild recommended.parm from the current realization run."""
from __future__ import annotations

import json
from pathlib import Path

from src.dse.physics_estimator import DesignInputs
from src.prototyping.artifact_store import (
    atomic_write_text,
    ensure_open_bundle,
    input_dir,
    output_dir,
)
from src.sitl.parameter_projection import design_parm_lines
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE


ROOT = Path(__file__).resolve().parents[1]
OUT = output_dir()
INPUT = input_dir()


def main() -> int:
    ensure_open_bundle(OUT)
    run = json.loads(
        (INPUT / "realization_run.json").read_text(encoding="utf-8")
    )
    raw = run.get("recommended_design_inputs")
    if not raw:
        raise SystemExit("current run has no recommended design")
    design = DesignInputs(**raw)
    lines = design_parm_lines(
        design,
        base_params=ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {},
    )
    text = (
        "# Recommended design SITL params; native SITL is architecture-"
        "nondiscriminating for endurance.\n" + "\n".join(lines) + "\n"
    )
    atomic_write_text(OUT / "recommended.parm", text)
    print(f"rebuilt recommended.parm for rotor_count={design.rotor_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
