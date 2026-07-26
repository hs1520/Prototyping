"""Run ONE arm, ONE seed, of the full pipeline — to watch it, not to measure it.

DIAGNOSTIC ONLY. It writes no pilot manifest and makes no protocol claim, because
a descriptive pilot is three paired seeds across three arms by definition and
nothing less can stand in for one: `run_revised_experiment.py` remains the only
way to produce evidence.

What it is for is the other need — seeing a complete run go by once, at a ninth of
the cost: requirements → design generation → refinement (syntax gate, repair
rounds) → the board-mediated handoffs → the A/G layer → verification → artifacts.
Watching that is how a defect nobody predicted gets spotted, and paying for nine
runs to read the first one is waste.

It executes the SAME path the pilot executes, including the pilot's own
`_validate_run_result`, so what you watch is what a pilot arm would do.

    .venv/bin/python examples/probe_single_arm.py --out /tmp/one_arm \
        --provider vertex --model gemini-3.1-pro-preview --confirm-external-call

`--provider mock` checks the wiring for free — config, arm binding, provider
construction — but cannot complete a run, because the mock provider does not
produce a model and the design phase fails closed on that, correctly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config  # noqa: F401,E402  (loads .env)
from run_revised_experiment import FROZEN_REQUIREMENTS  # noqa: E402
from src.prototyping.pipeline import PrototypingPipeline  # noqa: E402
from src.prototyping.provider_factory import create_llm  # noqa: E402
from src.prototyping.revised_pilot import (  # noqa: E402
    R2_INTERVENTION_VERSION_BY_MODE,
    REVISED_PILOT_ARMS,
    RevisedPilotConfig,
    _validate_run_result,
)
from src.prototyping.run_artifacts import (  # noqa: E402
    write_revised_run_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="a directory that must not exist")
    parser.add_argument("--arm", default="R2-BBAG", choices=REVISED_PILOT_ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--r2-generation-mode", default="LLM_DECIDED_SPEC")
    parser.add_argument("--llm-timeout-seconds", type=float, default=420.0)
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

    # The config supplies the frozen requirement set, the chain selection and the
    # version bindings, so the arm runs exactly as it would inside a pilot. Its
    # seed tuple is the protocol's three; the one actually executed is --seed.
    config = RevisedPilotConfig(
        provider=args.provider,
        model=args.model,
        seeds=(0, 1, 2),
        max_iterations=args.max_iterations,
        code_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        ).stdout.strip() or "unknown",
        requirements=FROZEN_REQUIREMENTS,
        selected_ag_chain_ids=("REQ_SAFE_004", "REQ_SAFE_005", "REQ_SAFE_008"),
        r2_generation_mode=args.r2_generation_mode,
        r2_intervention_version=R2_INTERVENTION_VERSION_BY_MODE[
            args.r2_generation_mode
        ],
    )
    frozen = config.frozen_requirements()
    out.mkdir(parents=True)

    print(f"DIAGNOSTIC single-arm run: {args.arm}, seed {args.seed}, "
          f"rev {config.code_revision[:7]}", flush=True)
    # the pilot's own provider kwargs, so seeding and timeouts match. The mock
    # provider takes none of them; it exists here only to dry-run the wiring for
    # free before spending anything.
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
        task_session_max_turns=config.task_session_max_turns,
        task_session_max_tokens=config.task_session_max_tokens,
    )
    result = pipeline.orchestrator.generate(
        system_name=config.system_name,
        system_description=config.system_description,
        frozen_requirements=frozen,
    )
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
        print(f"  traceability       "
              f"{trace.get('fully_traced')}/{trace.get('requirements')} in scope, "
              f"out_of_scope={[
                  item.get('requirement')
                  for item in trace.get('out_of_scope_requirements') or ()
              ]}")
    print(f"  artifacts          {len(written)} files in {out}")
    print("DIAGNOSTIC ONLY — not experiment evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
