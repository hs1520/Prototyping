"""Run isolated, low-cost Option 2 B0/B1/B2 pilot repetitions.

This runner checks that every ablation can complete the same frozen-input
pipeline and persist inspectable artifacts.  It deliberately does not launch
Gazebo/SITL, publish ``examples/output/latest``, or claim authoritative
experimental evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.prototyping.artifact_store import atomic_write_json, atomic_write_text
from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.posthoc_evaluation import build_uniform_posthoc_evaluation
from src.prototyping.provider_factory import create_llm
from src.prototyping.contract_types import INCOMPLETE, READY, UNSUPPORTED
from src.prototyping.requirement_inputs import (
    build_frozen_requirement_set,
    normalise_requirement_id,
    resolve_frozen_requirement_set,
)
from src.prototyping.robustness import RobustnessOptions
from src.prototyping.robustness_gold import validate_gold_dataset

sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import DRONE_DESCRIPTION, DRONE_REQUIREMENTS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "examples" / "output" / "pilots"
SYSTEM = "AutonomousDrone"
CONFIGURATIONS = {
    "B0": RobustnessOptions.b0,
    "B1": RobustnessOptions.b1,
    "B2": RobustnessOptions.b2,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_dirty": bool(status),
        "worktree_status": status,
    }


def _runtime_metadata() -> dict[str, Any]:
    """Verify that the parser needed by every experimental arm is operational."""
    metadata: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "syside_available": False,
        "syside_version": None,
        "syside_probe_part_count": 0,
        "error": None,
    }
    try:
        import syside

        metadata["syside_available"] = True
        metadata["syside_version"] = str(
            getattr(syside, "__version__", "unknown")
        )
        probe, diagnostics = syside.try_load_model(
            sysml_source="package RuntimeProbe { part def ProbePart { } }"
        )
        parts = list(probe.elements(syside.PartDefinition))
        metadata["syside_probe_part_count"] = len(parts)
        parser_errors = list(getattr(diagnostics, "parser", ()) or ())
        if len(parts) != 1 or parser_errors:
            metadata["error"] = (
                "Syside probe did not recover exactly one part definition "
                f"(parts={len(parts)}, parser_errors={len(parser_errors)})"
            )
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    metadata["ready"] = bool(
        metadata["syside_available"]
        and metadata["syside_probe_part_count"] == 1
        and not metadata["error"]
    )
    return metadata


def _pilot_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _read_json_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} artifact must contain a JSON object")
    return value


def _artifact_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_experiment_inputs(
    frozen: dict[str, Any], gold: dict[str, Any]
) -> dict[str, Any]:
    """Bind one reviewed source-first gold file to one frozen 10+5 input.

    Gold is an experiment-control/evaluation artifact.  Its reviewed contracts
    are deliberately not returned to the generation pipeline.
    """
    errors = validate_gold_dataset(gold, require_complete=True)
    if errors:
        raise ValueError("invalid/incomplete gold dataset: " + "; ".join(errors))
    if gold.get("review_mode") != "SOURCE_FIRST":
        raise ValueError("controlled experiment requires SOURCE_FIRST gold")

    requirements, canonical = resolve_frozen_requirement_set(frozen)
    records = {
        normalise_requirement_id(text): text for text in requirements
    }
    selected_ids = [
        normalise_requirement_id(item)
        for item in gold.get("selected_requirement_ids", ())
    ]
    frozen_ids = [normalise_requirement_id(text) for text in requirements]
    if selected_ids != frozen_ids:
        raise ValueError(
            "gold selected_requirement_ids must exactly match frozen "
            "requirements and order"
        )
    selected_payload = json.dumps(
        [
            (req_id, canonical["source_digests"][req_id])
            for req_id in selected_ids
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    selected_fingerprint = hashlib.sha256(
        selected_payload.encode("utf-8")
    ).hexdigest()
    if gold.get("selected_source_fingerprint") != selected_fingerprint:
        raise ValueError(
            "gold selected_source_fingerprint differs from frozen input"
        )

    counts = {READY: 0, INCOMPLETE: 0, UNSUPPORTED: 0}
    rows = list(gold.get("requirements", ()))
    for row in rows:
        req_id = normalise_requirement_id(str(row.get("req_id", "")))
        source_text = records.get(req_id)
        if source_text is None or row.get("source_text") != source_text:
            raise ValueError(f"{req_id}: gold source text differs from frozen input")
        expected_digest = canonical["source_digests"].get(req_id)
        if row.get("source_digest") != expected_digest:
            raise ValueError(f"{req_id}: gold source digest differs from frozen input")
        reviewed_contract = row.get("reviewed_contract") or {}
        if reviewed_contract.get("req_id") != req_id:
            raise ValueError(f"{req_id}: reviewed contract id differs from gold row")
        if reviewed_contract.get("source_text") != source_text:
            raise ValueError(
                f"{req_id}: reviewed contract source differs from frozen input"
            )
        if reviewed_contract.get("source_digest") != expected_digest:
            raise ValueError(
                f"{req_id}: reviewed contract digest differs from frozen input"
            )
        status = reviewed_contract.get("completeness")
        if status in counts:
            counts[status] += 1

    coverage_limit_count = counts[INCOMPLETE] + counts[UNSUPPORTED]
    if not (
        len(rows) == 15
        and 8 <= counts[READY] <= 10
        and 3 <= coverage_limit_count <= 5
    ):
        raise ValueError(
            "controlled experiment requires the reviewed 15-item source set "
            "with 8-10 READY and 3-5 INCOMPLETE/UNSUPPORTED requirements; "
            f"observed {counts}"
        )
    return {
        "gate": "APPROVED_FROZEN_REQUIREMENTS_AND_SOURCE_FIRST_GOLD",
        "frozen_requirement_set_digest": canonical["requirement_set_digest"],
        "frozen_artifact_digest": _artifact_digest(frozen),
        "gold_artifact_digest": _artifact_digest(gold),
        "gold_schema_version": gold.get("schema_version"),
        "gold_reviewer": gold.get("reviewer"),
        "gold_reviewed_at": gold.get("reviewed_at"),
        "selected_requirement_ids": selected_ids,
        "reviewed_contract_counts": counts,
        "gold_used_as_pipeline_input": False,
    }


def _run_one(
    configuration: str,
    batch_dir: Path,
    *,
    batch_id: str,
    run_name: str,
    repetition: int,
    max_iterations: int,
    mcts_iterations: int,
    mcts_seed: int,
    generation_seed: int,
    controlled_experiment: bool,
    git_metadata: dict[str, Any],
    runtime_metadata: dict[str, Any],
    frozen: dict[str, Any],
    experiment_approval: dict[str, Any] | None,
) -> dict[str, Any]:
    run_dir = batch_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"{batch_id}/{run_name}"
    options = CONFIGURATIONS[configuration]()
    started_at = _utc_now()
    metadata: dict[str, Any] = {
        "artifact_type": "OPTION2_PILOT_RUN",
        "pilot_only": not controlled_experiment,
        "controlled_experiment": controlled_experiment,
        "authoritative": False,
        "configuration": configuration,
        "robustness_options": options.as_dict(),
        "provider": "vertex",
        "requested_model": None,
        "generation_seed": generation_seed,
        "generation_seed_control": "PROVIDER_BEST_EFFORT",
        "repetition": repetition,
        "mcts_seed": mcts_seed,
        "run_id": run_id,
        "max_iterations": max_iterations,
        "mcts_iterations": mcts_iterations,
        "high_fidelity_evidence": "DISABLED",
        "started_at": started_at,
        "requirement_set_digest": frozen["requirement_set_digest"],
        "requirement_count": len(frozen["requirements"]),
        "git": git_metadata,
        "runtime": runtime_metadata,
        "experiment_input_approval": experiment_approval,
    }
    atomic_write_json(run_dir / "pilot_metadata.json", metadata)
    atomic_write_json(run_dir / "frozen_requirements.json", frozen)
    t0 = time.time()
    try:
        llm = create_llm(
            provider="vertex", provider_kwargs={"seed": generation_seed}
        )
        metadata["requested_model"] = getattr(llm, "model", None)
        metadata["langsmith_enabled"] = bool(
            getattr(llm, "langsmith_enabled", False)
        )
        print(
            f"[{configuration}] starting with {metadata['requested_model']} "
            f"({len(frozen['requirements'])} frozen requirements)",
            flush=True,
        )
        pipe = PrototypingPipeline(
            llm=llm,
            max_iterations=max_iterations,
            verbose=False,
            dse_mode="variation",
            phase9_hifi=None,
            robustness_options=options,
        )
        generated = pipe.orchestrator.generate(
            system_name=SYSTEM,
            system_description=DRONE_DESCRIPTION,
            frozen_requirements=frozen,
        )
        initial_model = generated.get("model_sysml") or ""
        result = pipe.orchestrator.explore(
            generated,
            mcts_iterations=mcts_iterations,
            mcts_seed=mcts_seed,
        )
        final_model = result.get("model_sysml") or ""
        report = pipe.build_run_report(result)
        posthoc = build_uniform_posthoc_evaluation(
            model_text=final_model,
            model_name=SYSTEM,
            frozen_requirements=frozen,
        )
        elapsed_s = round(time.time() - t0, 3)
        report.update({
            "artifact_type": "OPTION2_PILOT_REPORT",
            "pilot_only": not controlled_experiment,
            "controlled_experiment": controlled_experiment,
            "authoritative": False,
            "configuration": configuration,
            "run_id": run_id,
            "repetition": repetition,
            "mcts_seed": mcts_seed,
            "elapsed_s": elapsed_s,
        })
        provenance = dict(report.get("artifact_provenance") or {})
        provenance.update({
            "run_id": run_id,
            "pilot_batch_id": batch_id,
            "configuration": configuration,
            "repetition": repetition,
            "mcts_seed": mcts_seed,
        })
        report["artifact_provenance"] = provenance
        metadata.update({
            "status": "COMPLETED",
            "completed_at": _utc_now(),
            "elapsed_s": elapsed_s,
            "resolved_model": getattr(llm, "model", None),
        })
        atomic_write_text(run_dir / "initial_model.sysml", initial_model)
        atomic_write_text(run_dir / "final_model.sysml", final_model)
        atomic_write_json(run_dir / "realization_run.json", report)
        atomic_write_json(run_dir / "posthoc_evaluation.json", posthoc)
        atomic_write_json(run_dir / "pilot_metadata.json", metadata)
        print(
            f"[{configuration}] completed in {elapsed_s:.1f}s; "
            f"score={report.get('final_score')}",
            flush=True,
        )
        return {
            "configuration": configuration,
            "run_id": run_id,
            "repetition": repetition,
            "mcts_seed": mcts_seed,
            "status": "COMPLETED",
            "elapsed_s": elapsed_s,
            "run_dir": str(run_dir),
            "final_score": report.get("final_score"),
        }
    except BaseException as exc:
        elapsed_s = round(time.time() - t0, 3)
        metadata.update({
            "status": "FAILED",
            "completed_at": _utc_now(),
            "elapsed_s": elapsed_s,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        })
        failure_artifacts: dict[str, str] = {}
        original_model = getattr(exc, "original_model_text", None)
        candidate_model = getattr(exc, "candidate_model_text", None)
        diagnostics = getattr(exc, "diagnostics", None)
        generation_metadata = getattr(exc, "generation_metadata", None)
        parser_metadata = getattr(exc, "parser_metadata", None)
        if original_model:
            atomic_write_text(run_dir / "failed_original_model.sysml", original_model)
            failure_artifacts["original_model"] = "failed_original_model.sysml"
        if candidate_model:
            atomic_write_text(run_dir / "failed_candidate_model.sysml", candidate_model)
            failure_artifacts["candidate_model"] = "failed_candidate_model.sysml"
        if diagnostics is not None or generation_metadata is not None:
            atomic_write_json(run_dir / "failure_diagnostics.json", {
                "diagnostics": diagnostics or [],
                "generation_metadata": generation_metadata or {},
                "parser_metadata": parser_metadata or {},
            })
            failure_artifacts["diagnostics"] = "failure_diagnostics.json"
        if failure_artifacts:
            metadata["failure_artifacts"] = failure_artifacts
        atomic_write_json(run_dir / "pilot_metadata.json", metadata)
        atomic_write_text(run_dir / "failure_traceback.txt", traceback.format_exc())
        print(
            f"[{configuration}] FAILED in {elapsed_s:.1f}s: "
            f"{exc.__class__.__name__}: {exc}",
            flush=True,
        )
        return {
            "configuration": configuration,
            "run_id": run_id,
            "repetition": repetition,
            "mcts_seed": mcts_seed,
            "status": "FAILED",
            "elapsed_s": elapsed_s,
            "run_dir": str(run_dir),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run non-authoritative B0/B1/B2 pipeline smoke tests."
    )
    parser.add_argument(
        "--config", action="append", choices=tuple(CONFIGURATIONS),
        help="configuration to run; repeat as needed (default: B0, B1, B2)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help=(
            "outer generation/refinement iterations (default: 2; B2 needs "
            "at least 2 to exercise its repair intervention)"
        ),
    )
    parser.add_argument("--mcts-iterations", type=int, default=2)
    parser.add_argument(
        "--repetitions", type=int, default=1,
        help=(
            "independent stochastic repetitions per configuration (use 3 for "
            "the controlled experiment; default: 1 for a low-cost Pilot)"
        ),
    )
    parser.add_argument(
        "--mcts-seed-base", type=int, default=0,
        help="first deterministic MCTS seed; repetition r uses base+r-1",
    )
    parser.add_argument(
        "--llm-seed-base", type=int, default=1000,
        help=(
            "first best-effort Vertex generation seed; repetition r uses "
            "base+r-1"
        ),
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help=(
            "run the controlled ablation rather than a smoke Pilot; requires "
            "all B0/B1/B2 arms, at least three repetitions, and a clean worktree"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--requirements-artifact", type=Path,
        help=(
            "validated FROZEN_REQUIREMENT_SET JSON; mandatory with --experiment"
        ),
    )
    parser.add_argument(
        "--gold", type=Path,
        help=(
            "completed SOURCE_FIRST reviewed-gold JSON used only for experiment "
            "admission and post-hoc evaluation; mandatory with --experiment"
        ),
    )
    args = parser.parse_args(argv)
    if (
        args.max_iterations < 1
        or args.mcts_iterations < 1
        or args.repetitions < 1
    ):
        parser.error("iteration counts must be positive")

    selected = list(dict.fromkeys(args.config or list(CONFIGURATIONS)))
    if "B2" in selected and args.max_iterations < 2:
        parser.error(
            "B2 requires --max-iterations >= 2 so its repair intervention "
            "can actually be exercised"
        )
    if args.experiment and (
        set(selected) != set(CONFIGURATIONS) or args.repetitions < 3
    ):
        parser.error(
            "--experiment requires all B0/B1/B2 configurations and at least "
            "three repetitions"
        )
    git_metadata = _git_metadata()
    if args.experiment and git_metadata["worktree_dirty"]:
        parser.error(
            "--experiment requires a clean committed worktree so every run "
            "has reproducible source provenance"
        )
    runtime_metadata = _runtime_metadata()
    if not runtime_metadata["ready"]:
        parser.error(
            "Syside runtime preflight failed before any LLM calls: "
            f"{runtime_metadata.get('error') or 'parser probe failed'}; "
            f"python={runtime_metadata['python_executable']}"
        )
    if args.experiment and (args.requirements_artifact is None or args.gold is None):
        parser.error(
            "--experiment requires --requirements-artifact and --gold so the "
            "reviewed 10+5 source set is frozen before any LLM calls"
        )
    try:
        if args.requirements_artifact is not None:
            supplied_frozen = _read_json_artifact(
                args.requirements_artifact, label="requirements"
            )
            _requirements, frozen = resolve_frozen_requirement_set(supplied_frozen)
        else:
            frozen = build_frozen_requirement_set(
                DRONE_REQUIREMENTS,
                name="option2-pilot-requirements",
                source="examples.drone_system_v2.DRONE_REQUIREMENTS",
            )
        experiment_approval = None
        if args.experiment:
            gold = _read_json_artifact(args.gold, label="gold")
            experiment_approval = _validate_experiment_inputs(frozen, gold)
    except ValueError as exc:
        parser.error(str(exc))
    batch_id = _pilot_id()
    batch_dir = args.output_root.expanduser().resolve() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(batch_dir / "pilot_batch.json", {
        "artifact_type": "OPTION2_PILOT_BATCH",
        "pilot_only": not args.experiment,
        "controlled_experiment": args.experiment,
        "authoritative": False,
        "created_at": _utc_now(),
        "configurations": selected,
        "repetitions": args.repetitions,
        "planned_run_count": len(selected) * args.repetitions,
        "max_iterations": args.max_iterations,
        "mcts_iterations": args.mcts_iterations,
        "mcts_seed_base": args.mcts_seed_base,
        "llm_seed_base": args.llm_seed_base,
        "git": git_metadata,
        "runtime": runtime_metadata,
        "experiment_input_approval": experiment_approval,
        "status": "RUNNING",
    })
    print(f"Pilot batch: {batch_dir}", flush=True)

    jobs = [
        (
            name,
            repetition,
            (
                name.lower()
                if args.repetitions == 1
                else f"{name.lower()}-r{repetition:02d}"
            ),
            args.mcts_seed_base + repetition - 1,
            args.llm_seed_base + repetition - 1,
        )
        for repetition in range(1, args.repetitions + 1)
        for name in selected
    ]
    results = [
        _run_one(
            name,
            batch_dir,
            batch_id=batch_id,
            run_name=run_name,
            repetition=repetition,
            max_iterations=args.max_iterations,
            mcts_iterations=args.mcts_iterations,
            mcts_seed=mcts_seed,
            generation_seed=generation_seed,
            controlled_experiment=args.experiment,
            git_metadata=git_metadata,
            runtime_metadata=runtime_metadata,
            frozen=frozen,
            experiment_approval=experiment_approval,
        )
        for name, repetition, run_name, mcts_seed, generation_seed in jobs
    ]
    completed = sum(item["status"] == "COMPLETED" for item in results)
    batch = {
        "artifact_type": "OPTION2_PILOT_BATCH",
        "pilot_only": not args.experiment,
        "controlled_experiment": args.experiment,
        "authoritative": False,
        "completed_at": _utc_now(),
        "configurations": selected,
        "repetitions": args.repetitions,
        "planned_run_count": len(jobs),
        "max_iterations": args.max_iterations,
        "mcts_iterations": args.mcts_iterations,
        "mcts_seed_base": args.mcts_seed_base,
        "llm_seed_base": args.llm_seed_base,
        "git": git_metadata,
        "runtime": runtime_metadata,
        "experiment_input_approval": experiment_approval,
        "status": "COMPLETED" if completed == len(results) else "PARTIAL_FAILURE",
        "completed_count": completed,
        "failed_count": len(results) - completed,
        "results": results,
    }
    atomic_write_json(batch_dir / "pilot_batch.json", batch)
    print(json.dumps(batch, indent=2, ensure_ascii=False), flush=True)
    return 0 if completed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
