"""Prepare source-first Option 2 gold without exposing pipeline predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototyping.robustness_gold import (
    OPTION2_GOLD_REQUIREMENT_IDS,
    prepare_gold_template,
)
from src.prototyping.run_artifacts import load_archived_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a source-first gold-review worksheet. Pipeline candidates "
            "are hidden by default to prevent anchoring."
        )
    )
    parser.add_argument("run", type=Path, help="run directory or realization_run.json")
    parser.add_argument("output", type=Path, help="review worksheet JSON")
    parser.add_argument(
        "--include-pipeline-candidates",
        action="store_true",
        help=(
            "include non-authoritative candidates for a documented second pass; "
            "never use this option for the first source review"
        ),
    )
    parser.add_argument(
        "--all-requirements",
        action="store_true",
        help="review the full source set instead of the frozen Option 2 10+5 set",
    )
    args = parser.parse_args(argv)
    worksheet = prepare_gold_template(
        load_archived_run(args.run),
        include_pipeline_candidates=args.include_pipeline_candidates,
        selected_req_ids=(
            None if args.all_requirements else OPTION2_GOLD_REQUIREMENT_IDS
        ),
    )
    args.output.expanduser().resolve().write_text(
        json.dumps(worksheet, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mode = "candidate-assisted second-pass" if args.include_pipeline_candidates else "source-only"
    print(f"Wrote PENDING {mode} gold-review worksheet: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
