"""Prepare blind per-run trace/routing labels from frozen Option 2 source gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototyping.robustness_gold import prepare_run_review_template
from src.prototyping.run_artifacts import load_archived_run


def _load_json(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a per-run blind worksheet. It contains frozen expected link "
            "structure but no pipeline trace status, diagnostics, or route verdict."
        )
    )
    parser.add_argument("run", type=Path, help="run directory or realization_run.json")
    parser.add_argument("gold", type=Path, help="completed source-first gold JSON")
    parser.add_argument("output", type=Path, help="per-run review worksheet JSON")
    args = parser.parse_args(argv)
    run = load_archived_run(args.run)
    gold = _load_json(args.gold)
    worksheet = prepare_run_review_template(run, gold)
    args.output.expanduser().resolve().write_text(
        json.dumps(worksheet, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Wrote PENDING blind per-run review worksheet "
        f"for {worksheet['run_id']}: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
