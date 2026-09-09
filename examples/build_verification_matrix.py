"""Build the per-requirement verification strategy matrix from existing artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from src.prototyping.verification_matrix import build_matrix, summarize, to_json, to_markdown
from src.prototyping.artifact_store import (
    LATEST_NAME,
    output_root,
    atomic_write_json, atomic_write_text, ensure_open_bundle, input_dir, output_dir,
)
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

ROOT = Path(__file__).resolve().parents[1]
OUT = output_dir()

def _input_dir_or_placeholder() -> Path:
    # Importing this module must not require a published bundle; main() re-validates.
    try:
        return input_dir()
    except FileNotFoundError:
        return output_root() / LATEST_NAME


INPUT = _input_dir_or_placeholder()
SYSML_PATH = INPUT / "final_model.sysml"
RUN_JSON = INPUT / "realization_run.json"
GAZEBO_JSON = INPUT / "gazebo_feasibility_report.json"
MATRIX_MD = OUT / "verification_matrix.md"
MATRIX_JSON = OUT / "verification_matrix.json"


def _fresh_gazebo_report(run_json: dict | None, model_sysml: str) -> dict | None:
    if not GAZEBO_JSON.exists():
        return None
    report = json.loads(GAZEBO_JSON.read_text(encoding="utf-8"))
    if not run_json:
        return None
    return report if report.get("source_run_id") == run_json.get("run_id") else None


def main() -> int:
    input_dir()  # fail closed before any work when no bundle is published
    ensure_open_bundle(OUT)
    model_sysml = SYSML_PATH.read_text(encoding="utf-8")
    model = build_lite_model(model_sysml,
                             model_name="AutonomousDrone")
    run_json = None
    realization = None
    if RUN_JSON.exists():
        run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
        if not run_json.get("run_id"):
            raise SystemExit("realization_run.json has no run_id; regenerate the run")
        realization = run_json.get("realization")
    gazebo = _fresh_gazebo_report(run_json, model_sysml)
    linker = RequirementLinker(model)
    rows = build_matrix(
        model, realization, linker.compile_evidence(), gazebo=gazebo
    )
    atomic_write_text(MATRIX_MD, to_markdown(rows))
    payload = to_json(rows)
    if run_json:
        payload["source_run_id"] = run_json.get("run_id")
    atomic_write_json(MATRIX_JSON, payload)
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"Wrote {MATRIX_MD}")
    print(f"Wrote {MATRIX_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
