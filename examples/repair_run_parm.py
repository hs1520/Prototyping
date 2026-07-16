"""Rebuild recommended.parm from the current realization run, hash-gated."""
from __future__ import annotations

import json
from pathlib import Path

from src.dse.physics_estimator import DesignInputs
from src.prototyping.artifact_provenance import sha256_text
from src.prototyping.artifact_store import atomic_write_text, ensure_open_bundle, output_dir
from src.sitl.dse_sitl_params import design_to_sitl_parm
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE


ROOT = Path(__file__).resolve().parents[1]
OUT = output_dir()


def main() -> int:
    ensure_open_bundle(OUT)
    run = json.loads((OUT / "realization_run.json").read_text(encoding="utf-8"))
    raw = run.get("recommended_design_inputs")
    if not raw:
        raise SystemExit("current run has no recommended design")
    design = DesignInputs(**raw)
    lines = list(design_to_sitl_parm(design))
    present = {
        line.split()[0] for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }
    for key, value in (ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {}).items():
        if key not in present:
            lines.append(f"{key:<20} {value}")
    text = (
        "# Recommended design SITL params; native SITL is architecture-"
        "nondiscriminating for endurance.\n" + "\n".join(lines) + "\n"
    )
    actual = sha256_text(text)
    expected = (run.get("artifact_provenance") or {}).get("parm_sha256")
    if actual != expected:
        raise SystemExit(
            f"refusing repair: reconstructed hash {actual} != recorded hash {expected}"
        )
    atomic_write_text(OUT / "recommended.parm", text)
    print(f"repaired recommended.parm for rotor_count={design.rotor_count}; hash={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
