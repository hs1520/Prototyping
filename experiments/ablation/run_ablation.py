"""Component-ablation campaign driver (drone_v2 frozen requirements).

Runs the paid ablation arms (see arms.py) over a shared seed list, one fresh
seeded provider instance per run, and archives every artifact needed to make a
claim traceable: the effective pipeline configuration, the canonical run
report, the final model text, per-run stdout, and — on failure — the rejected
model plus the gate evidence that rejected it.

Usage
-----
    # plumbing smoke (MockLLM cannot produce a real design; the run is
    # recorded as failed, which exercises capture/archive/aggregate end to end):
    /Users/huangsongyi/miniforge3/envs/AI-Prototyping/bin/python \
        experiments/ablation/run_ablation.py --provider mock --arms FULL --seeds 1

    # single real pilot run to calibrate cost before committing to a campaign:
    ... run_ablation.py --provider vertex --arms FULL --seeds 1

    # the full campaign (7 paid arms × 3 seeds):
    ... run_ablation.py --provider vertex --seeds 3

Outputs one campaign directory under experiments/ablation/results/:
    campaign.json     frozen manifest (git commit, arm digests, requirement digest)
    records.jsonl     one flat record per run, appended crash-safe
    summary.json/.md  aggregates + paired deltas vs FULL (via analyze.py)
    runs/             per-run report JSON, final .sysml, stdout log
    failed_runs/      rejected models + gate evidence for failed runs

W-UNIFORM is post-hoc: after the campaign, run posthoc_weights.py on this
campaign directory (no LLM cost).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

import src.config  # noqa: F401,E402  (loads .env)
from src.utils.digest import sha256_text  # noqa: E402

from arms import ARMS, BASELINE_ARM, PAID_ARMS, registry_manifest  # noqa: E402
from analyze import aggregate, paired_deltas, render_markdown  # noqa: E402

SYSTEM_NAME = "AutonomousDrone"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def _archive_failure(
    error: BaseException,
    campaign_dir: Path,
    arm_name: str,
    seed: int,
    record: Dict[str, Any],
) -> None:
    """Persist the rejected model and the evidence that rejected it.

    Mirrors scripts/benchmark.py: fail-closed gates attach their evidence to
    the raised error; without this the exact revision that failed is lost.
    """
    model_text = getattr(error, "terminal_model_text", None)
    if model_text is None:
        model_text = getattr(error, "candidate_model_text", None)
    closure = getattr(error, "functional_closure", None)
    rejections = getattr(error, "plan_conformance_rejections", None)
    provenance = getattr(error, "generation_plan_provenance", None)
    diagnostics = getattr(error, "diagnostics", None)

    failed_dir = campaign_dir / "failed_runs"
    failed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{arm_name}_seed{seed}"
    if isinstance(model_text, str) and model_text:
        model_path = failed_dir / f"{stem}.sysml"
        model_path.write_text(model_text, encoding="utf-8")
        record["failed_model_path"] = str(model_path)
        record["failed_model_digest"] = sha256_text(model_text)
    evidence = {
        "arm": arm_name,
        "seed": seed,
        "error": f"{type(error).__name__}: {error}",
        "functional_closure": closure,
        "plan_conformance_rejections": rejections,
        "generation_plan_provenance": provenance,
        "diagnostics": diagnostics,
        "traceback": traceback.format_exc(),
    }
    evidence_path = failed_dir / f"{stem}.evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8"
    )
    record["failed_evidence_path"] = str(evidence_path)
    if isinstance(closure, dict):
        record["functional_closure_status"] = closure.get("status")
    if isinstance(rejections, list):
        record["plan_conformance_rejections"] = len(rejections)
    if isinstance(provenance, dict):
        record["plan_status"] = provenance.get("plan_status")
        record["step1_plan_retries"] = provenance.get("step1_plan_retries")


def run_one(
    provider: str,
    arm_name: str,
    seed: int,
    base_kwargs: Dict[str, Any],
    mcts_iterations: int,
    campaign_dir: Path,
) -> Dict[str, Any]:
    """One ablation run → flat record (never raises; failures are recorded)."""
    from src.app.pipeline import PrototypingPipeline
    from src.prototyping.provider_factory import create_llm
    from src.simulation.controlled_scenarios import evaluate_controlled_scenarios
    from drone_system_v2 import DRONE_DESCRIPTION, DRONE_FROZEN_REQUIREMENTS

    arm = ARMS[arm_name]
    effective_kwargs = {**base_kwargs, **dict(arm.pipeline_kwargs)}
    ablation_stamp = {
        "arm": arm.name,
        "arm_digest": arm.digest(),
        "ablated_component": arm.ablated_component,
        "seed": seed,
        "run_dse": arm.run_dse,
        "effective_pipeline_kwargs": effective_kwargs,
        "mcts_iterations": mcts_iterations if arm.run_dse else None,
    }
    record: Dict[str, Any] = {
        "arm": arm.name,
        "seed": seed,
        "provider": provider,
        "effective_pipeline_kwargs": effective_kwargs,
    }
    runs_dir = campaign_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{arm.name}_seed{seed}"
    log_path = runs_dir / f"{stem}.log"
    started = time.time()
    llm = None

    try:
        provider_kwargs = (
            {"seed": seed}
            if provider.strip().lower() in {"vertex", "gemini"}
            else None
        )
        llm = create_llm(provider=provider, provider_kwargs=provider_kwargs)
        record.update({
            "llm_model": getattr(llm, "model", None),
            "provider_seed": getattr(llm, "seed", None),
        })
        pipeline = PrototypingPipeline(llm=llm, **effective_kwargs)
        # The pipeline narrates heavily; keep the campaign console readable and
        # archive the full narration per run instead.
        with open(log_path, "w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file):
                gen = pipeline.orchestrator.generate(
                    system_name=SYSTEM_NAME,
                    system_description=DRONE_DESCRIPTION,
                    frozen_requirements=DRONE_FROZEN_REQUIREMENTS,
                )
                if arm.run_dse:
                    result = pipeline.orchestrator.explore(
                        generate_result=gen,
                        mcts_iterations=mcts_iterations,
                        mcts_seed=seed,
                    )
                else:
                    result = gen
        result["ablation"] = ablation_stamp
        report = pipeline.build_run_report(result)

        model_sysml = result.get("model_sysml") or ""
        controlled = None
        if model_sysml:
            (runs_dir / f"{stem}.final.sysml").write_text(
                model_sysml, encoding="utf-8"
            )
            try:
                controlled = evaluate_controlled_scenarios(
                    model_sysml, model_name=SYSTEM_NAME
                )
                report["controlled_scenarios"] = controlled
            except Exception as scenario_error:
                report["controlled_scenarios_error"] = (
                    f"{type(scenario_error).__name__}: {scenario_error}"
                )
        (runs_dir / f"{stem}.report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

        closure = result.get("functional_closure") or {}
        qualification = result.get("model_qualification") or {}
        usage = report.get("llm_usage") or {}
        record.update({
            "ok": True,
            "final_score": report.get("final_score"),
            "reachability": (report.get("simulation") or {}).get(
                "reachability_score"
            ),
            "controlled_pass_rate": (
                controlled.get("pass_rate") if controlled else None
            ),
            "controlled_pass": (
                (controlled.get("counts") or {}).get("PASS")
                if controlled else None
            ),
            "iterations": report.get("iterations"),
            "pareto_size": len(report.get("pareto_alternatives") or []),
            "recommended_by": report.get("recommended_by"),
            "recommendation_status": report.get("recommendation_status"),
            "llm_calls": usage.get("calls"),
            "llm_total_tokens": usage.get("total_tokens"),
            "llm_elapsed_s": usage.get("elapsed_seconds"),
            "qualification_status": qualification.get("status"),
            "functional_closure_status": closure.get("status"),
            "report_path": str(runs_dir / f"{stem}.report.json"),
            "log_path": str(log_path),
        })
    except Exception as error:
        record.update({
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "log_path": str(log_path),
        })
        # Spend up to the point of failure.  Deliberately NOT the llm_* names
        # the analysis consumes: a failed run stopped at an arbitrary point, so
        # its cost is not comparable to a completed run's and must never reach
        # aggregate()/paired_deltas().  This is campaign accounting — the run
        # report (and the usage line it prints) is never built on this path, so
        # without this the tokens a failed run burned are unrecoverable.
        ledger = getattr(llm, "ledger", None)
        if ledger is not None:
            usage = ledger.as_dict()
            record["llm_usage_at_failure"] = usage
            record["llm_calls_at_failure"] = usage.get("calls")
            record["llm_total_tokens_at_failure"] = usage.get("total_tokens")
        try:
            _archive_failure(error, campaign_dir, arm.name, seed, record)
        except Exception as archive_error:   # never mask the real failure
            record["archive_error"] = (
                f"{type(archive_error).__name__}: {archive_error}"
            )
    record["elapsed_s"] = round(time.time() - started, 1)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider", default="mock",
        help="LLM provider (mock/vertex/gemini/github_copilot). Default mock: "
             "a real campaign must name its provider explicitly.",
    )
    parser.add_argument(
        "--arms", nargs="*", default=list(PAID_ARMS), choices=list(PAID_ARMS),
        help="paid arms to run (default: all; W-UNIFORM is post-hoc only)",
    )
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds per arm (0..N-1, shared across arms)")
    parser.add_argument("--mcts-iterations", type=int, default=20,
                        help="DSE budget per run (flagship run_pipeline.py uses 20)")
    parser.add_argument("--max-iterations", type=int, default=4,
                        help="refinement budget (flagship uses 4; NO-REFINE "
                             "overrides to 1)")
    parser.add_argument("--out-root",
                        default=str(REPO / "experiments/ablation/results"))
    parser.add_argument("--label", default="",
                        help="optional campaign label suffix")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit a real-provider campaign on a dirty "
                             "worktree (recorded either way)")
    args = parser.parse_args()

    commit = _git("rev-parse", "HEAD")
    # Tracked modifications make a campaign non-reproducible; pre-existing
    # untracked scratch dirs do not — they are recorded, not refused.
    dirty = bool(_git("status", "--porcelain", "-uno"))
    untracked = [
        line[3:] for line in _git("status", "--porcelain").splitlines()
        if line.startswith("??")
    ]
    real_provider = args.provider.strip().lower() != "mock"
    if real_provider and dirty and not args.allow_dirty:
        print("✗ refusing a real-provider campaign on a dirty worktree "
              "(commit first, or pass --allow-dirty); mock runs are exempt")
        return 2

    from drone_system_v2 import DRONE_FROZEN_REQUIREMENTS, DRONE_REQUIREMENTS

    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{args.label}" if args.label else stamp
    campaign_dir = Path(args.out_root) / name
    campaign_dir.mkdir(parents=True, exist_ok=False)

    base_kwargs = {
        "max_iterations": args.max_iterations,
        "dse_mode": "variation",
        "verbose": False,
    }
    manifest = {
        "campaign": name,
        "system": SYSTEM_NAME,
        "requirement_source": "examples/drone_system_v2.py "
                              "(DRONE_FROZEN_REQUIREMENTS)",
        "requirements_digest": sha256_text("\n".join(DRONE_REQUIREMENTS)),
        "frozen_set_digest": getattr(DRONE_FROZEN_REQUIREMENTS, "digest", None),
        "provider": args.provider,
        "seeds": list(range(args.seeds)),
        "arms_requested": list(args.arms),
        "base_pipeline_kwargs": base_kwargs,
        "mcts_iterations": args.mcts_iterations,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_untracked": untracked,
        "arm_registry": registry_manifest(),
        "python": sys.version,
        "started": stamp,
    }
    (campaign_dir / "campaign.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"campaign: {campaign_dir}")
    print(f"commit  : {commit[:12]}{' (DIRTY)' if dirty else ''}")

    records: List[Dict[str, Any]] = []
    records_path = campaign_dir / "records.jsonl"
    total = len(args.arms) * args.seeds
    done = 0
    for arm_name in args.arms:
        for seed in range(args.seeds):
            done += 1
            print(f"\n=== [{done}/{total}] {arm_name} seed={seed} "
                  f"({args.provider}) ===", flush=True)
            record = run_one(
                args.provider, arm_name, seed, base_kwargs,
                args.mcts_iterations, campaign_dir,
            )
            records.append(record)
            with open(records_path, "a", encoding="utf-8") as records_file:
                records_file.write(json.dumps(record, default=str) + "\n")
            status = "✓" if record.get("ok") else f"✗ {record.get('error')}"
            print(f"    {status}  score={record.get('final_score')}  "
                  f"tokens={record.get('llm_total_tokens')}  "
                  f"{record.get('elapsed_s')}s", flush=True)

    summary = {
        "aggregate": aggregate(records),
        "paired_vs_baseline": paired_deltas(records, baseline=BASELINE_ARM),
    }
    (campaign_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (campaign_dir / "summary.md").write_text(
        render_markdown(manifest, records, summary), encoding="utf-8"
    )
    print("\n=== aggregate ===")
    print(json.dumps(summary["aggregate"], indent=2, default=str))
    print(f"\nsaved: {campaign_dir}")
    if BASELINE_ARM in args.arms:
        print("next (free): python experiments/ablation/posthoc_weights.py "
              f"{campaign_dir}")
    return 0 if all(record.get("ok") for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
