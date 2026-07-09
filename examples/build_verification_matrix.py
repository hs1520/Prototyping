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
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "output"
SYSML_PATH = OUT / "final_model.sysml"
RUN_JSON = OUT / "realization_run.json"
MATRIX_MD = OUT / "verification_matrix.md"
MATRIX_JSON = OUT / "verification_matrix.json"


def main() -> int:
    model = build_lite_model(SYSML_PATH.read_text(encoding="utf-8"),
                             model_name="AutonomousDrone")
    realization = None
    if RUN_JSON.exists():
        realization = json.loads(RUN_JSON.read_text(encoding="utf-8")).get("realization")
    linker = RequirementLinker(model)
    rows = build_matrix(model, realization, linker)
    MATRIX_MD.write_text(to_markdown(rows), encoding="utf-8")
    MATRIX_JSON.write_text(json.dumps(to_json(rows), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"Wrote {MATRIX_MD}")
    print(f"Wrote {MATRIX_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
