"""Regenerate Option 2 B0/B1/B2 aggregate metrics from archived run bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototyping.robustness_evaluation import evaluate_configurations
from src.prototyping.run_artifacts import load_archived_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate archived Option 2 runs without launching the pipeline."
    )
    for name in ("b0", "b1", "b2"):
        parser.add_argument(
            f"--{name}", action="append", default=[], metavar="RUN",
            help="run directory or realization_run.json (repeat for each seed)",
        )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument(
        "--gold", type=Path,
        help="completed independently reviewed gold JSON for accuracy/F1 metrics",
    )
    parser.add_argument(
        "--run-review", type=Path, action="append", default=[],
        help=(
            "completed blind per-run trace/routing review JSON; repeat once for "
            "every run when reporting trace F1 or routing accuracy"
        ),
    )
    parser.add_argument(
        "--allow-unfrozen-inputs",
        action="store_true",
        help=(
            "permit descriptive analysis of legacy/variable-input runs; these "
            "runs are not valid controlled B0/B1/B2 evidence"
        ),
    )
    parser.add_argument(
        "--allow-incomplete-design",
        action="store_true",
        help=(
            "allow descriptive aggregation without complete balanced B0/B1/B2 "
            "repetitions and paired MCTS seeds"
        ),
    )
    parser.add_argument(
        "--allow-legacy-measurement",
        action="store_true",
        help=(
            "allow descriptive aggregation without one verified uniform "
            "versioned post-hoc measurement stack on every run"
        ),
    )
    args = parser.parse_args(argv)
    groups = {
        name.upper(): [load_archived_run(path) for path in getattr(args, name)]
        for name in ("b0", "b1", "b2") if getattr(args, name)
    }
    if not groups:
        parser.error("provide at least one --b0, --b1, or --b2 run")
    gold = None
    if args.gold:
        gold = json.loads(
            args.gold.expanduser().resolve().read_text(encoding="utf-8")
        )
    run_reviews = None
    if args.run_review:
        loaded = [
            json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
            for path in args.run_review
        ]
        run_reviews = {str(item.get("run_id", "")): item for item in loaded}
        if "" in run_reviews or len(run_reviews) != len(loaded):
            parser.error("every --run-review must have a unique non-empty run_id")
    report = evaluate_configurations(
        groups,
        gold=gold,
        run_reviews_by_id=run_reviews,
        require_frozen_inputs=not args.allow_unfrozen_inputs,
        require_complete_design=not args.allow_incomplete_design,
        require_uniform_posthoc=not args.allow_legacy_measurement,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
