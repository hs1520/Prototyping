"""Re-derive archived behavioural verdicts under the repaired intent lookup.

Through the v21 era, functional_behavior_status looked recorded response
intents up by the trace's hyphenated requirement id while producers key them
REQ_XXX_NNN, so keyword inference governed at that check (the matrix's
planned_no_response bucket normalised its own keys and was unaffected). The
lookup was repaired with v22. This probe recomputes per-requirement behavioural
status both ways - keyword fallback and recorded intents honoured - for every
archived run that recorded intents, and re-runs the terminal functional-closure
audit on the archived extraction-path run. Deterministic; no LLM calls.

Archived result (2026-08-28) in
examples/output/probe_intent_key_regression_20260828/report.json:
  - reference pilot batches (pilot_n6_0c26731_*, 32 runs): the only verdict
    difference is REQ-FUNC-002 (obstacle avoidance), which gains
    behaviorally-verified once its recorded 'navigate' intent is honoured; the
    keyword table has no avoidance vocabulary. Archived reports under-credit,
    none over-credit.
  - Study-A anchor (runs/f99ac140): REQ-FUNC-002/005/007 gain
    behaviorally-verified at this check, but 005 and 007 were already anchored
    behaviourally through the guard-assignment path, so only FUNC-002 changes
    in the published rows. Same direction.
  - extraction-path run (runs/fc9f76ee, 31 requirements): the adverse case.
    Keyword inference read 'receive a flight waypoint sequence' as a navigate
    obligation; the plan had recorded none, so honouring the record withdraws
    that credit. Terminal functional closure still passes (0 gaps) under the
    none + extractor-unmeasurable two-signal rule, so the claim this run
    supports - that the extraction path closes - holds.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import pathlib
import sys

from src.agents.verification_audit import functional_verification_gap_issues
from src.dse.functional_behavior import functional_behavior_status


def _intents(plan: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in plan.get("requirement_realizations") or []:
        rid = str(item.get("requirement_id") or "").strip().upper().replace("-", "_")
        value = str(item.get("response_intent") or "").strip().lower()
        if rid and value:
            out[rid] = value
    return out


def _both_ways(model: str, reqs: list[str], intents: dict[str, str]) -> dict:
    bug_era = functional_behavior_status(model, reqs)
    repaired = functional_behavior_status(model, reqs, planned_intents=intents)
    return {
        rid: {"keyword": bug_era.get(rid), "recorded": repaired.get(rid)}
        for rid in sorted(set(bug_era) | set(repaired))
        if bug_era.get(rid) != repaired.get(rid)
    }


def main() -> int:
    report: dict = {"pilot_batches": {}, "canonical_runs": {}}

    for rp in sorted(glob.glob(
        "examples/output/pilot_n6_0c26731_*/seed-*/R*-*/run_report.json"
    )):
        run_dir = os.path.dirname(rp)
        model_path = os.path.join(run_dir, "shared_model_final.sysml")
        if not os.path.exists(model_path):
            continue
        doc = json.load(open(rp))
        diffs = _both_ways(
            open(model_path).read(),
            [str(r) for r in doc.get("requirements") or []],
            _intents(doc.get("whole_model_generation_plan") or {}),
        )
        if diffs:
            report["pilot_batches"][run_dir.split("output/")[1]] = diffs

    for cj in sorted(glob.glob("examples/output/runs/*/canonical_run.json")):
        root = os.path.dirname(cj)
        doc = json.load(open(cj))
        pipeline = doc.get("pipeline_report") or {}
        model_path = os.path.join(
            root, str(doc.get("model_artifact") or "final_model.sysml")
        )
        reqs = [str(r) for r in (
            pipeline.get("requirements") or doc.get("requirements") or []
        )]
        intents = _intents(pipeline.get("whole_model_generation_plan") or {})
        if not (os.path.exists(model_path) and reqs and intents):
            continue
        entry: dict = {
            "requirement_count": len(reqs),
            "verdict_differences": _both_ways(
                open(model_path).read(), reqs, intents
            ),
        }
        unmeasurable = (
            (pipeline.get("requirement_input") or {}).get("unmeasurable_req_ids")
            or []
        )
        if unmeasurable:
            model = open(model_path).read()
            entry["closure_gaps_keyword"] = functional_verification_gap_issues(
                model, "DroneSystem", strict=True,
                unmeasurable_req_ids=unmeasurable,
            )
            entry["closure_gaps_recorded"] = functional_verification_gap_issues(
                model, "DroneSystem", strict=True,
                unmeasurable_req_ids=unmeasurable,
                planned_intents=intents,
            )
        report["canonical_runs"][os.path.basename(root)] = entry

    stamp = datetime.date.today().strftime("%Y%m%d")
    out_dir = pathlib.Path(f"examples/output/probe_intent_key_regression_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    pilot_diffs = sum(len(v) for v in report["pilot_batches"].values())
    print(f"pilot runs with differences: {len(report['pilot_batches'])} "
          f"({pilot_diffs} rows)")
    for name, entry in report["canonical_runs"].items():
        print(name[:12], entry["verdict_differences"] or "identical",
              "| closure:",
              len(entry.get("closure_gaps_recorded", [])) if "closure_gaps_recorded" in entry else "n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
