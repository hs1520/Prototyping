"""Run one arm, one seed, of the full pipeline - to watch it, not to measure it.

Diagnostic only: no pilot manifest and no protocol claim, since a descriptive
pilot is three paired seeds across three arms and `run_revised_experiment.py`
is the only way to produce evidence. This shows one complete run at a ninth of
the cost: requirements -> design generation -> refinement (syntax gate, repair
rounds) -> board-mediated handoffs -> A/G layer -> verification -> artifacts. It
executes the same path a pilot arm does, including `_validate_run_result`.

    .venv/bin/python examples/probe_single_arm.py --out /tmp/one_arm         --provider vertex --model gemini-3.1-pro-preview --confirm-external-call

`--provider mock` checks config, arm binding and provider construction for
free, but cannot complete a run: it produces no model and the design phase
fails closed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import src.config  # noqa: F401  (loads .env)
from run_revised_experiment import FROZEN_REQUIREMENTS
from src.app.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.app.revised_pilot import (
    R2_INTERVENTION_VERSION_BY_MODE,
    REVISED_PILOT_ARMS,
    RevisedPilotConfig,
    _validate_run_result,
)
from src.prototyping.run_artifacts import (
    write_revised_run_artifacts,
)


# Diagnostic-only extra chains. The pilot's frozen requirement set is
# digest-checked and cited verbatim by the gold drafts, so a new pattern is added
# here rather than in `run_revised_experiment.FROZEN_REQUIREMENTS`. A run using
# them carries a different requirement digest, so it cannot pool with pilot arms.
#
# Text copied verbatim from `examples/drone_system_v2.py`.
PROBE_ONLY_CHAINS = {
    "REQ_SAFE_002": (
        "REQ-SAFE-002: The system shall perform a controlled descent to the "
        "nearest safe landing area when the battery state-of-charge falls below "
        "15%, superseding any lower-priority contingency response."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="a directory that must not exist")
    parser.add_argument("--arm", default="R2-BBAG", choices=REVISED_PILOT_ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--r2-generation-mode", default="LLM_DECIDED_SPEC")
    parser.add_argument(
        "--r2-authored-syntax-max-attempts",
        type=int,
        default=3,
        help="bounded notation-only attempts per authored A/G chain",
    )
    parser.add_argument("--llm-timeout-seconds", type=float, default=420.0)
    parser.add_argument(
        "--include-chain", action="append", default=[],
        choices=sorted(PROBE_ONLY_CHAINS),
        help="add a diagnostic-only A/G chain and its requirement to this run; "
             "the resulting requirement digest differs from the pilot's, so the "
             "run can never be pooled with pilot arms",
    )
    parser.add_argument(
        "--confirm-external-call", action="store_true",
        help="confirm this invocation is authorised to call a paid provider",
    )
    args = parser.parse_args()
    if not args.confirm_external_call:
        parser.error("refusing to call a provider without --confirm-external-call")
    out = Path(args.out)
    if out.exists():
        parser.error(f"output directory already exists: {out}")

    # The config supplies the frozen requirement set, chain selection and version
    # bindings, so the arm runs as it would inside a pilot. The seed tuple is the
    # protocol's three; --seed picks the one executed.
    extra_chains = tuple(dict.fromkeys(args.include_chain))
    requirements = FROZEN_REQUIREMENTS + tuple(
        PROBE_ONLY_CHAINS[chain_id] for chain_id in extra_chains
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip() or "unknown"
    # A probe does not need a clean worktree, but a recorded revision that lacks
    # the code under test is a provenance defect, so an uncommitted tree is marked
    # rather than reported as its HEAD. Fail towards marking: a `git status` that
    # did not run cleanly says nothing about the tree - one run raced a concurrent
    # commit, got empty output, and recorded a revision without the code under test.
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        revision = f"{revision}-dirty"
    config = RevisedPilotConfig(
        provider=args.provider,
        model=args.model,
        seeds=(0, 1, 2),
        max_iterations=args.max_iterations,
        code_revision=revision,
        requirements=requirements,
        selected_ag_chain_ids=(
            "REQ_SAFE_004", "REQ_SAFE_005", "REQ_SAFE_008", *extra_chains
        ),
        r2_generation_mode=args.r2_generation_mode,
        r2_intervention_version=R2_INTERVENTION_VERSION_BY_MODE[
            args.r2_generation_mode
        ],
        r2_authored_syntax_max_attempts=(
            args.r2_authored_syntax_max_attempts
        ),
    )
    frozen = config.frozen_requirements()
    out.mkdir(parents=True)

    print(f"DIAGNOSTIC single-arm run: {args.arm}, seed {args.seed}, "
          f"rev {config.code_revision[:7]}", flush=True)
    # the pilot's provider kwargs, so seeding and timeouts match. The mock
    # provider takes none of them; it only dry-runs the wiring for free.
    provider_kwargs = (
        {}
        if args.provider.strip().lower() == "mock"
        else {
            "seed": args.seed,
            "enable_langsmith": False,
            "timeout_seconds": args.llm_timeout_seconds,
        }
    )
    pipeline = PrototypingPipeline(
        llm=create_llm(
            provider=args.provider,
            model=args.model,
            provider_kwargs=provider_kwargs,
        ),
        max_iterations=args.max_iterations,
        quality_threshold=config.quality_threshold,
        verbose=False,
        revised_experiment_arm=args.arm,
        r2_generation_mode=args.r2_generation_mode,
        r2_authored_syntax_max_attempts=config.r2_authored_syntax_max_attempts,
        task_session_max_turns=config.task_session_max_turns,
        task_session_max_tokens=config.task_session_max_tokens,
    )
    try:
        result = pipeline.orchestrator.generate(
            system_name=config.system_name,
            system_description=config.system_description,
            frozen_requirements=frozen,
        )
    except Exception as exc:
        attempts = list(
            pipeline.orchestrator.last_ag_authoring_attempts
        )
        step1_attempts = list(
            getattr(exc, "plan_attempts", ()) or ()
        )
        (out / "ag_authoring_attempts.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "artifact_role": "R2_AUTHORED_SYNTAX_ATTEMPTS",
                "feedback_scope": "SYNTAX_ONLY",
                "maximum_attempts_per_chain": (
                    config.r2_authored_syntax_max_attempts
                ),
                "attempts": attempts,
            }, indent=2),
            encoding="utf-8",
        )
        (out / "probe_failure.json").write_text(
            json.dumps({
                "artifact_role": "DIAGNOSTIC_PROBE_FAILURE",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "r2_generation_mode": args.r2_generation_mode,
                "r2_authored_syntax_max_attempts": (
                    config.r2_authored_syntax_max_attempts
                ),
                "authoring_attempt_count": len(attempts),
                "step1_plan_attempt_count": len(step1_attempts),
            }, indent=2),
            encoding="utf-8",
        )
        if step1_attempts:
            (out / "step1_plan_attempts.json").write_text(
                json.dumps({
                    "schema_version": "1.0",
                    "artifact_role": "TYPED_MODEL_PLAN_ATTEMPTS",
                    "maximum_attempts": 3,
                    "retry_policy": (
                        "ONE_FORMAT_RECOVERY_PLUS_ONE_SEMANTIC_CORRECTION"
                    ),
                    "attempts": step1_attempts,
                }, indent=2),
                encoding="utf-8",
            )
        print(
            f"  diagnostic failure artifacts written to {out}",
            flush=True,
        )
        raise
    report = pipeline.build_run_report(result)
    # the pilot's own validation, so a defect it would reject is rejected here
    _validate_run_result(
        result,
        report,
        arm=args.arm,
        requirement_set_digest=str(frozen["requirement_set_digest"]),
        ag_checker_version=config.ag_checker_version,
        r2_generation_mode=args.r2_generation_mode,
        r2_intervention_version=config.r2_intervention_version,
    )
    (out / "run_report.json").write_text(json.dumps(report, indent=2))
    written = (
        write_revised_run_artifacts(result, out)
        if result.get("collaboration") else {}
    )

    print(f"\n--- {args.arm} seed {args.seed} ---")
    print(f"  final_score        {report.get('final_score')}")
    print(f"  A/G verdict        "
          f"{(result.get('ag_contract_graph') or {}).get('verdict')}")
    print(f"  pattern conformance"
          f" {(result.get('pattern_conformance_report') or {}).get('verdict')}")
    path = written.get("requirement_traceability")
    if path:
        trace = json.loads(Path(path).read_text()).get("traceability") or {}
        out_of_scope_items = [
            item.get("requirement")
            for item in trace.get("out_of_scope_requirements") or ()
        ]
        print(f"  traceability       "
              f"{trace.get('fully_traced')}/{trace.get('requirements')} in scope, "
              f"out_of_scope={out_of_scope_items}")
    print(f"  artifacts          {len(written)} files in {out}")
    print("DIAGNOSTIC ONLY — not experiment evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
