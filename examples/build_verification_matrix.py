"""Build the per-requirement verification strategy matrix from existing artifacts.

Reads examples/output/final_model.sysml + realization_run.json (no LLM, no SITL),
writes examples/output/verification_matrix.{md,json}.

Run:
  PYTHONPATH=. .venv/bin/python examples/build_verification_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.prototyping.verification_matrix import build_matrix, summarize, to_json, to_markdown
from src.prototyping.artifact_provenance import (
    validate_derived_provenance, validate_run_provenance,
)
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "output"
SYSML_PATH = OUT / "final_model.sysml"
RUN_JSON = OUT / "realization_run.json"
GAZEBO_JSON = OUT / "gazebo_feasibility_report.json"
MATRIX_MD = OUT / "verification_matrix.md"
MATRIX_JSON = OUT / "verification_matrix.json"


def _fresh_gazebo_report(run_json: dict | None, model_sysml: str) -> dict | None:
    if not GAZEBO_JSON.exists():
        return None
    report = json.loads(GAZEBO_JSON.read_text(encoding="utf-8"))
    if not run_json:
        return None
    fresh, _ = validate_derived_provenance(report, run_json, model_sysml)
    return report if fresh else None


def main() -> int:
    model_sysml = SYSML_PATH.read_text(encoding="utf-8")
    model = build_lite_model(model_sysml,
                             model_name="AutonomousDrone")
    run_json = None
    realization = None
    if RUN_JSON.exists():
        run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
        fresh, reason = validate_run_provenance(run_json, model_sysml=model_sysml)
        if not fresh:
            raise SystemExit(f"STALE artifact set: {reason}")
        realization = run_json.get("realization")
    gazebo = _fresh_gazebo_report(run_json, model_sysml)
    linker = RequirementLinker(model)
    rows = build_matrix(model, realization, linker, gazebo=gazebo)
    MATRIX_MD.write_text(to_markdown(rows), encoding="utf-8")
    MATRIX_JSON.write_text(json.dumps(to_json(rows), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"Wrote {MATRIX_MD}")
    print(f"Wrote {MATRIX_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
