"""Measure what the coverage-denominator fix changes, per archived model.

Not a re-run: the archived models are read from disk and no provider is called.

It deliberately does NOT compare against the archived `final_score`. That number
was computed with the run's simulation result, which is not fully archived, so a
fresh evaluation would score `behavioral_verification` as N/A=1.0 and the
difference would mix the fix with a missing input. One R1 run moved +0.0223 that
way, with no A/G contract anywhere in it.

Instead both denominators are evaluated on the same model in the same process,
so the delta is attributable to the change and nothing else.

R0-CURRENT is absent: the archiving gap that left the baseline without a
committed model was fixed after every pilot here had run. The change is a no-op
for a model carrying no contracts, which the R1-BBCTX rows show empirically.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dse.design_space import DesignConfiguration  # noqa: E402
from src.dse.evaluator import DesignEvaluator  # noqa: E402
from src.sysml.lite_model import build_lite_model  # noqa: E402

ARMS = ("R0-CURRENT", "R1-BBCTX", "R2-BBAG")


def _score(model, requirements, *, stakeholder_only: bool):
    """Evaluate once under one denominator rule."""
    import src.dse.evaluator as evaluator

    original_req = evaluator._STAKEHOLDER_REQ
    original_tx = evaluator._GUARDED_TRANSITION
    if not stakeholder_only:
        # the archived rules: every requirement def counted, contracts
        # included; and a guarded transition had to put `if` immediately
        # after the source state, so an `accept` clause hid it
        evaluator._STAKEHOLDER_REQ = re.compile(r"(?!)")
        evaluator._GUARDED_TRANSITION = re.compile(
            r"\btransition\s+\w+\s+first\s+\w+\s+if\s+[^;]+?"
            r"\s+then\s+\w+\s*;",
            re.IGNORECASE | re.DOTALL,
        )
    try:
        return DesignEvaluator().evaluate(
            DesignConfiguration({}), model, requirements=requirements
        )
    finally:
        evaluator._STAKEHOLDER_REQ = original_req
        evaluator._GUARDED_TRANSITION = original_tx


def rescore_run(run_dir: Path) -> dict | None:
    model_path = run_dir / "shared_model_final.sysml"
    report_path = run_dir / "run_report.json"
    if not model_path.exists() or not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = build_lite_model(
        model_path.read_text(encoding="utf-8"), model_name="DeliveryUAV"
    )
    requirements = report.get("requirements") or []
    before = _score(model, requirements, stakeholder_only=False)
    after = _score(model, requirements, stakeholder_only=True)
    changed = sorted(
        dim for dim, value in after.criteria_scores.items()
        if before.criteria_scores.get(dim) != value
    )
    return {
        "archived_final_score": report.get("final_score"),
        "old_denominator_total": before.weighted_total,
        "new_denominator_total": after.weighted_total,
        "delta": round(after.weighted_total - before.weighted_total, 4),
        "dimensions_changed": changed,
        "criteria_scores": dict(after.criteria_scores),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "unknown"

    for pilot in args.pilot_dirs:
        rows: dict[str, dict] = {}
        for seed_dir in sorted(pilot.glob("seed-*")):
            for arm in ARMS:
                out = rescore_run(seed_dir / arm)
                if out is not None:
                    rows[f"{seed_dir.name}/{arm}"] = out
        payload = {
            "schema_version": "1.0",
            "artifact_role": "PILOT_RESCORE",
            "measurement_boundary": "POST_HOC_RESCORE_NO_PROVIDER_CALL",
            "evaluator_code_revision": revision,
            "note": (
                "archived models, current evaluator. R0-CURRENT is absent "
                "because those runs predate the baseline-archiving fix; the "
                "change is a no-op for a model with no A/G contracts, which the "
                "R1-BBCTX deltas show."
            ),
            "runs": rows,
        }
        out_path = pilot / "rescored.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n=== {pilot.name}")
        for key, row in rows.items():
            print(f"  {key:22} {row['old_denominator_total']:.4f} -> "
                  f"{row['new_denominator_total']:.4f}  "
                  f"({row['delta']:+.4f})  "
                  f"changed={row['dimensions_changed'] or '—'}")
        print(f"  written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
