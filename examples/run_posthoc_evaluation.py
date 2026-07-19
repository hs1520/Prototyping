"""Apply the uniform read-only Option 2 evaluator to an archived run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototyping.artifact_store import atomic_write_json
from src.prototyping.posthoc_evaluation import build_uniform_posthoc_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one completed model without invoking its generation "
            "pipeline, an LLM, a simulator, or a repair path."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model-name", default="AutonomousDrone")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    report = json.loads(
        (run_dir / "realization_run.json").read_text(encoding="utf-8")
    )
    frozen_path = run_dir / "frozen_requirements.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    else:
        frozen = report.get("requirement_input")
    if not frozen:
        parser.error("run has no frozen requirement artifact")
    model = (run_dir / "final_model.sysml").read_text(encoding="utf-8")
    posthoc = build_uniform_posthoc_evaluation(
        model_text=model,
        model_name=args.model_name,
        frozen_requirements=frozen,
    )
    output = run_dir / "posthoc_evaluation.json"
    atomic_write_json(output, posthoc)
    metrics = posthoc["metrics"]
    print(
        f"{run_dir.name}: {metrics['semantic_trace_pass_count']}/"
        f"{metrics['supported_trace_count']} supported traces PASS; "
        f"{metrics['diagnostic_count']} diagnostic(s); wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
