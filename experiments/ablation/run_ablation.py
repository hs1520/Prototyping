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
import re
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

#: Provider/transport exhaustion is the harness's environment failing, not the
#: ablated configuration failing. A 429-aborted run polluted runs_failed on
#: 2026-08-30 (renamed *_ABORTED_429 by hand); this classifier makes the
#: separation mechanical, mirroring the authoritative path's infrastructure
#: routing (examples/finalize_authoritative_run.py).
_INFRA_MARKERS = (
    # "provider error" is the wrapper every transport/quota failure funnels
    # through (LLMInterface raises "<Provider> provider error (...): ...");
    # gate/validation errors never contain it. Bare words are dangerous:
    # "connection" matched "connections entry" and "unavailable" matched
    # TYPED_MODEL_PLAN_UNAVAILABLE — both measured misclassifications.
    "provider error", "rate limit", "resource_exhausted",
    "resource exhausted", "quota", "service unavailable", "overloaded",
    "timed out", "wall-clock timeout", "deadline exceeded",
    "connection refused",
    "connection reset", "connection aborted", "connectionerror",
    "eof occurred", "eoferror", "broken pipe",
)


def _infrastructure_failure(message: str) -> bool:
    # Bare substrings are dangerous here: the first live failure was
    # misclassified as infrastructure because the marker "connection"
    # matched the word "connections" inside plan-validation issue text.
    # Markers are specific phrases; 429 is matched as a standalone token.
    low = str(message or "").lower()
    if re.search(r"(?<![0-9a-z])429(?![0-9a-z])", low):
        return True
    return any(token in low for token in _INFRA_MARKERS)


def _request_digest(
    message_dicts: List[Dict[str, str]], temperature: float, max_tokens: int,
) -> str:
    """Byte-stable identity of one provider request (provider-agnostic)."""
    return sha256_text(json.dumps(
        {
            "messages": message_dicts,
            "temperature": temperature,
            "max_tokens": int(max_tokens),
        },
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ))


def _capture_observer(calls_path: Path):
    """Append every completed provider call to a crash-safe JSONL.

    This is what makes a dead run resumable: s0v12 hung on one provider
    request 34 minutes in and 187k tokens of identical prefix work had to
    be re-bought. The record is pure observation — prompts, pipeline, and
    arm digests are untouched."""
    def _observer(event: Dict[str, Any]) -> None:
        line = json.dumps({
            "request_digest": _request_digest(
                event["messages"], event["temperature"], event["max_tokens"],
            ),
            "label": event.get("label"),
            "response": event["response"],
        }, ensure_ascii=False, default=str)
        with open(calls_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return _observer


class PrefixReplayLLM:
    """Serve archived responses while requests match the recorded prefix.

    Replay is free and instant; the FIRST divergence (or exhaustion) flips
    permanently to the live provider — which is exactly resume-from-where-
    it-broke semantics, with the caveat that any divergence point starts
    paying from there. The inner ledger counts live calls only, so the
    run's cost accounting stays honest."""

    def __init__(self, inner, recorded: List[Dict[str, Any]]) -> None:
        self._inner = inner
        self._recorded = list(recorded)
        self._cursor = 0
        self._live = False
        self.replayed_calls = 0
        self.replayed_prompt_tokens = 0
        self.replayed_completion_tokens = 0

    @property
    def replayed_tokens(self) -> int:
        return self.replayed_prompt_tokens + self.replayed_completion_tokens

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def complete(self, messages, temperature=None, max_tokens=None, **kwargs):
        from src.llm.interface import (
            DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, LLMResponse,
        )
        resolved_temp = (
            DEFAULT_TEMPERATURE if temperature is None else temperature
        )
        resolved_max = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        if not self._live and self._cursor < len(self._recorded):
            record = self._recorded[self._cursor]
            digest = _request_digest(
                [m.to_dict() for m in messages], resolved_temp, resolved_max,
            )
            if record.get("request_digest") == digest:
                self._cursor += 1
                self.replayed_calls += 1
                payload = record.get("response") or {}
                response = LLMResponse(
                    content=str(payload.get("content") or ""),
                    model=str(payload.get("model") or ""),
                    prompt_tokens=int(payload.get("prompt_tokens") or 0),
                    completion_tokens=int(payload.get("completion_tokens") or 0),
                )
                self.replayed_prompt_tokens += response.prompt_tokens or 0
                self.replayed_completion_tokens += response.completion_tokens or 0
                # Archive the replayed call too, so this run's calls.jsonl is
                # its complete trajectory (prefix + live) and can itself seed
                # a later replay. The inner ledger still counts live calls only.
                notify = getattr(self._inner, "_notify_call_observers", None)
                if callable(notify):
                    try:
                        notify(
                            messages=messages, response=response,
                            temperature=resolved_temp, max_tokens=resolved_max,
                            retries=0, label=record.get("label"),
                        )
                    except Exception:
                        pass
                return response
        if not self._live:
            self._live = True
            print(f"  [resume] replayed {self.replayed_calls} archived "
                  f"call(s) free; live provider from call "
                  f"{self.replayed_calls + 1}", flush=True)
        return self._inner.complete(
            messages, temperature=temperature,
            **({} if max_tokens is None else {"max_tokens": max_tokens}),
            **kwargs,
        )


#: Narration markers of the repair mechanisms, counted from the archived
#: per-run stdout. They exist so the ablation can state whether the
#: component an arm removes was exercised at all on that trajectory — a
#: "no difference" cell is uninformative when FULL never invoked the layer.
_LOG_MARKERS = {
    # Tier 0 deterministic syntax rewrites (syntax gate, before any LLM call)
    "tier0_fixes": ("┌─ [DOC-FIX]", "┌─ [RO-FIX]", "┌─ [KW-FIX]",
                    "┌─ [LEV-FIX]", "┌─ [ATTR-INJ]"),
    # Tier 1 localized LLM syntax repair attempts
    "syntax_gate_llm_attempts": ("[SYNTAX-GATE] attempt",),
    # deterministic connectivity repairs accepted by the simulation guard
    "det_connectivity_fixes": ("direction fix (deterministic)",
                               "connect fix (deterministic)"),
    # block-level surgical refinement candidates accepted in the main loop
    "surgical_refinement_accepted": ("✓ Surgical refinement:",),
    # terminal functional-closure targeted passes
    "closure_passes_narrated": ("│  Pass ",),
}


def _log_counts(log_text: str) -> Dict[str, int]:
    return {
        key: sum(log_text.count(marker) for marker in markers)
        for key, markers in _LOG_MARKERS.items()
    }


def _calls_by_label(calls_path: Path) -> Dict[str, Dict[str, int]]:
    """Per-stage call/token attribution from the archived call log."""
    out: Dict[str, Dict[str, int]] = {}
    if not calls_path.exists():
        return out
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = str(record.get("label") or "unlabelled")
        payload = record.get("response") or {}
        cell = out.setdefault(label, {"calls": 0, "tokens": 0})
        cell["calls"] += 1
        cell["tokens"] += int(payload.get("prompt_tokens") or 0) + int(
            payload.get("completion_tokens") or 0
        )
    return out


def _mechanism_ledger(
    result: Dict[str, Any],
    orchestrator: Any,
    log_path: Path,
    calls_path: Optional[Path],
) -> Dict[str, Any]:
    """Was each ablatable mechanism exercised, and how much did it do?

    Structured sources first (closure record, plan conformance, pipeline
    state lists); narration counts fill the layers that leave no record.
    """
    ledger: Dict[str, Any] = {}
    closure = result.get("functional_closure") or {}
    contexts = closure.get("repair_contexts") or []
    ledger["refine_iterations"] = result.get("iterations")
    ledger["plan_retries"] = result.get("step1_plan_retries")
    ledger["closure_ran"] = bool(closure)
    ledger["closure_initial_gaps"] = len(closure.get("initial_gap_req_ids") or [])
    ledger["closure_attempts"] = int(closure.get("attempts") or 0)
    ledger["closure_accepted"] = int(closure.get("accepted_repairs") or 0)
    ledger["closure_full_rewrite_passes"] = sum(
        1 for c in contexts if isinstance(c, dict)
        and c.get("generator") == "FULL_REWRITE"
    )
    ledger["closure_surgical_passes"] = sum(
        1 for c in contexts if isinstance(c, dict) and "surgical_audit" in c
    )
    exit_lists = (
        "last_namespace_repair_attempts",
        "last_response_conformance_repair_attempts",
        "last_verification_anchor_attempts",
        "last_unplanned_connect_removal_attempts",
    )
    ledger["surgical_exit_passes"] = sum(
        len(getattr(orchestrator, name, None) or []) for name in exit_lists
    )
    conformance = result.get("generation_plan_conformance") or {}
    ledger["plan_det_additions"] = sum(
        len(conformance.get(key) or []) for key in (
            "deterministically_added_connections",
            "deterministically_added_ports",
            "deterministically_retyped_ports",
            "deterministically_added_port_defs",
        )
    )
    ledger["plan_conformance_rejections"] = len(
        result.get("plan_conformance_rejections") or []
    )
    try:
        ledger.update(_log_counts(log_path.read_text(encoding="utf-8")))
    except OSError:
        pass
    if calls_path is not None:
        ledger["calls_by_label"] = _calls_by_label(calls_path)
    return ledger


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
    resume_calls: Optional[List[Dict[str, Any]]] = None,
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
        calls_path = runs_dir / f"{stem}.calls.jsonl"
        if hasattr(llm, "add_call_observer"):
            llm.add_call_observer(_capture_observer(calls_path))
            record["calls_capture_path"] = str(calls_path)
        if resume_calls:
            llm = PrefixReplayLLM(llm, resume_calls)
            record["resumed_from_calls"] = True
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
        # final_score is measured at different pipeline stages per arm: DSE
        # arms re-score the terminal snapshot (A/G layer + DSE closure
        # injections on the text), non-DSE arms keep the generate-phase
        # score. Measured on the seed-0 wave: every arm's generate-phase
        # score was byte-identical (0.9608) while final_score ranged
        # 0.9181-0.9608 purely by measurement point. generate_phase_score
        # is the cross-arm comparable number.
        history_scores = [
            h.get("score") for h in (report.get("evaluation_history") or ())
            if isinstance(h, dict) and h.get("score") is not None
        ]
        record.update({
            "ok": True,
            "final_score": report.get("final_score"),
            "generate_phase_score": (
                history_scores[0] if history_scores else None
            ),
            "terminal_rescore": (
                history_scores[-1] if len(history_scores) > 1 else None
            ),
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
        # Effort accounting under prefix replay: the ledger counts live
        # calls only, so add the replayed prefix back to make the arm's
        # total comparable with an unreplayed FULL. Wall time stays live.
        replayed_calls = int(getattr(llm, "replayed_calls", 0) or 0)
        if replayed_calls:
            replayed_tokens = int(getattr(llm, "replayed_tokens", 0) or 0)
            record.update({
                "llm_live_calls": usage.get("calls"),
                "llm_live_tokens": usage.get("total_tokens"),
                "replayed_calls": replayed_calls,
                "replayed_tokens": replayed_tokens,
                "llm_calls": int(usage.get("calls") or 0) + replayed_calls,
                "llm_total_tokens": (
                    int(usage.get("total_tokens") or 0) + replayed_tokens
                ),
            })
        # Mechanism ledger: observational, must never taint the run record.
        try:
            calls_archive = (
                Path(record["calls_capture_path"])
                if record.get("calls_capture_path") else None
            )
            ledger = _mechanism_ledger(
                result, pipeline.orchestrator, log_path, calls_archive,
            )
            record["mechanism_ledger"] = ledger
            for key, value in ledger.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    record[f"mech_{key}"] = value
        except Exception as ledger_error:
            record["mechanism_ledger_error"] = (
                f"{type(ledger_error).__name__}: {ledger_error}"
            )
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        record.update({
            "ok": False,
            "error": error_text,
            "log_path": str(log_path),
            "infrastructure_failure": _infrastructure_failure(error_text),
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
    parser.add_argument("--seed-ids", type=int, nargs="*", default=None,
                        help="explicit seed ids to run (overrides --seeds); "
                             "used to re-run one seed after an instrument fix")
    parser.add_argument("--mcts-iterations", type=int, default=20,
                        help="DSE budget per run (flagship run_pipeline.py uses 20)")
    parser.add_argument("--max-iterations", type=int, default=4,
                        help="refinement budget (flagship uses 4; NO-REFINE "
                             "overrides to 1)")
    parser.add_argument("--out-root",
                        default=str(REPO / "experiments/ablation/results"))
    parser.add_argument("--label", default="",
                        help="optional campaign label suffix")
    parser.add_argument("--resume-calls", default="",
                        help="path to a previous run's .calls.jsonl: matching "
                             "request prefix replays free, live from the "
                             "first divergence (single arm+seed only)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit a real-provider campaign on a dirty "
                             "worktree (recorded either way)")
    parser.add_argument("--prefix-from-baseline", action="store_true",
                        help="trajectory-matched ablation: every non-FULL "
                             "arm replays FULL's archived call prefix for "
                             "the same seed (free, byte-identical) and goes "
                             "live from the first request the ablated "
                             "component changes; FULL must run first in "
                             "this campaign or be named via "
                             "--baseline-campaign")
    parser.add_argument("--baseline-campaign", default="",
                        help="campaign dir whose runs/FULL_seed<N>.calls.jsonl "
                             "supply the replay prefix (default: this "
                             "campaign)")
    args = parser.parse_args()

    seed_ids = (
        list(args.seed_ids) if args.seed_ids is not None
        else list(range(args.seeds))
    )
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
        "seeds": seed_ids,
        "arms_requested": list(args.arms),
        "base_pipeline_kwargs": base_kwargs,
        "mcts_iterations": args.mcts_iterations,
        "git_commit": commit,
        "git_dirty": dirty,
        "prefix_from_baseline": bool(args.prefix_from_baseline),
        "baseline_campaign": args.baseline_campaign or None,
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

    resume_calls: Optional[List[Dict[str, Any]]] = None
    if args.resume_calls:
        if len(args.arms) != 1 or len(seed_ids) != 1:
            print("✗ --resume-calls applies to exactly one arm × one seed")
            return 2
        resume_calls = [
            json.loads(line)
            for line in Path(args.resume_calls).read_text().splitlines()
            if line.strip()
        ]
        print(f"resume  : {len(resume_calls)} archived call(s) from "
              f"{args.resume_calls}")

    records: List[Dict[str, Any]] = []
    records_path = campaign_dir / "records.jsonl"
    total = len(args.arms) * len(seed_ids)
    done = 0
    baseline_dir = (
        Path(args.baseline_campaign) if args.baseline_campaign else campaign_dir
    )
    for arm_name in args.arms:
        for seed in seed_ids:
            done += 1
            print(f"\n=== [{done}/{total}] {arm_name} seed={seed} "
                  f"({args.provider}) ===", flush=True)
            run_resume = resume_calls
            if (
                args.prefix_from_baseline and arm_name != BASELINE_ARM
                and run_resume is None
            ):
                prefix_path = (
                    baseline_dir / "runs" / f"{BASELINE_ARM}_seed{seed}.calls.jsonl"
                )
                if prefix_path.exists():
                    run_resume = [
                        json.loads(line)
                        for line in prefix_path.read_text().splitlines()
                        if line.strip()
                    ]
                    print(f"    prefix : {len(run_resume)} archived "
                          f"{BASELINE_ARM} call(s) from {prefix_path.name}",
                          flush=True)
                else:
                    print(f"    prefix : none ({prefix_path.name} missing) "
                          "— running live", flush=True)
            record = run_one(
                args.provider, arm_name, seed, base_kwargs,
                args.mcts_iterations, campaign_dir,
                resume_calls=run_resume,
            )
            if run_resume is not None and run_resume is not resume_calls:
                record["prefix_source"] = str(
                    baseline_dir / "runs" / f"{BASELINE_ARM}_seed{seed}.calls.jsonl"
                )
            records.append(record)
            with open(records_path, "a", encoding="utf-8") as records_file:
                records_file.write(json.dumps(record, default=str) + "\n")
            status = "✓" if record.get("ok") else f"✗ {record.get('error')}"
            replay_note = (
                f"  replayed={record.get('replayed_calls')} calls/"
                f"{record.get('replayed_tokens')} tok"
                if record.get("replayed_calls") else ""
            )
            print(f"    {status}  qual={record.get('qualification_status')}  "
                  f"closure={record.get('functional_closure_status')}  "
                  f"tokens={record.get('llm_total_tokens')}{replay_note}  "
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
