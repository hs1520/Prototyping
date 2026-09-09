"""Run one isolated, authoritative end-to-end realization experiment.

Single-writer, in order:

  Phase 1-8 -> base bundle -> Gazebo -> SITL/executed matrix
  -> finalizer -> atomic ``examples/output/latest`` publication.

Runs live under ``examples/output/runs/<run_id>``; a failed or interrupted run
is kept for diagnosis and does not replace ``latest``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import src.config  # noqa: F401  (loads .env)
import uuid
from src.prototyping.artifact_store import (
    OUTPUT_DIR_ENV,
    AuthoritativeRunLock,
    assign_run_id,
    atomic_write_json,
    atomic_write_text,
    create_staging_bundle,
    latest_output_dir,
    output_root,
    publish_latest,
    write_state,
)
from src.app.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.prototyping.requirement_inputs import requirement_change_impact
from src.sitl.parameter_projection import design_parm_lines
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE

sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import (  # noqa: E402
    DRONE_DESCRIPTION,
    DRONE_FROZEN_REQUIREMENTS,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "AutonomousDrone"


def require_authoritative_runtime() -> None:
    """Fail before any LLM call when the Syside-backed runtime is unavailable.

    ``conda run -n AI-Prototyping python`` can resolve a pyenv shim ahead of the
    environment interpreter, after which syntax checking and the structured
    LiteModel degrade to permissive/empty fallbacks.
    """
    from src.simulation import syntax_checker
    from src.sysml import lite_model

    unavailable = []
    if not getattr(syntax_checker, "_SYSIDE_OK", False):
        unavailable.append("syntax checker")
    if not getattr(lite_model, "_SYSIDE_OK", False):
        unavailable.append("LiteModel extractor")
    if unavailable:
        raise RuntimeError(
            "authoritative runtime blocked: Syside Python API is unavailable "
            f"for {', '.join(unavailable)} under {sys.executable}. "
            "Invoke the AI-Prototyping environment's Python executable "
            "directly; do not rely on a PATH-resolved `python` shim."
        )


def retire_stale_parm(parm_path: str) -> bool:
    """Legacy scratch-output helper; isolated run bundles do not use this path."""
    path = Path(parm_path)
    if not path.exists():
        return False
    os.replace(path, Path(str(path) + ".stale"))
    return True


def require_authoritative_functional_closure(
    model_sysml: str,
    *,
    unmeasurable_req_ids=None,
    planned_intents=None,
    planned_markers=None,
) -> list[str]:
    """Reject publication while model-fixable FUNC gaps remain.

    Reads the two signals the pipeline's terminal closure gate reads -- the plan's
    recorded response intents and the extractor's no-measurable-criterion flags --
    so a requirement with nothing to anchor to is not re-raised here as a model
    gap. Both are None on the frozen path, which is therefore unchanged.
    """
    from src.agents.verification_audit import functional_verification_gap_issues
    from src.simulation.syntax_checker import check_syntax

    if not (model_sysml or "").strip():
        raise RuntimeError("authoritative run blocked: final model is empty")
    syntax = check_syntax(model_sysml)
    if syntax.has_errors:
        raise RuntimeError(
            "authoritative run blocked: final model fails syntax/semantic validation"
        )
    functional_gaps = functional_verification_gap_issues(
        model_sysml, SYSTEM, strict=True,
        unmeasurable_req_ids=unmeasurable_req_ids,
        planned_intents=planned_intents,
        planned_markers=planned_markers,
    )
    ids = sorted(set(
        match.group(0)
        for issue in functional_gaps
        for match in [re.search(r"REQ[_-]FUNC[_-]\d+", issue)]
        if match
    ))
    if ids:
        raise RuntimeError(
            "authoritative run blocked: model-fixable functional verification "
            "gaps remain after targeted closure: " + ", ".join(ids)
        )
    return ids


def _parm_text(design) -> str | None:
    if design is None:
        return None
    lines = design_parm_lines(
        design,
        base_params=ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {},
    )
    return (
        "# Recommended design SITL params; native SITL is architecture-"
        "nondiscriminating for endurance.\n" + "\n".join(lines) + "\n"
    )


def _run_evidence(script: str, run_dir: Path, timeout_s: int = 1800) -> int:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        OUTPUT_DIR_ENV: str(run_dir),
    }
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script)],
        cwd=ROOT,
        env=env,
        timeout=timeout_s,
    )
    return proc.returncode


def _require_clean_worktree() -> None:
    dirty = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if dirty:
        preview = "\n".join(dirty[:20])
        remainder = len(dirty) - min(len(dirty), 20)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise RuntimeError(
            "authoritative runs require a clean Git worktree so the recorded commit "
            f"fully identifies the executed code. Commit or stash these changes first:\n"
            f"{preview}{suffix}"
        )


def _build_base_artifacts(pipe, res, elapsed_s: float) -> tuple[dict, str, str]:
    orch = pipe.orchestrator
    rec = getattr(orch, "last_recommended_design", None)
    if rec is None:
        status = getattr(orch, "last_recommendation_status", None) or "UNKNOWN"
        raise RuntimeError(
            f"authoritative full-stack run stopped at {status}: no design passed "
            "estimator feasibility + catalog mapping + Phase 8 closure; the staging "
            "bundle will not be published as latest"
        )
    realization = res.get("realization")
    if realization is not None:
        realization = dict(realization)
        realization["recommended_realizable"] = getattr(
            orch, "last_recommended_realizable", None
        )
        realization["realizable_front_count"] = getattr(
            orch, "last_realizable_front_count", None
        )
        realization["recommended_by"] = getattr(orch, "last_recommended_by", None)
        realization["recommended_estimator_feasible"] = getattr(
            orch, "last_recommended_estimator_feasible", None
        )
    final_sysml = res.get("model_sysml") or ""
    # Independent final publication gate: not run metadata alone, but the
    # same plan/extractor context the pipeline's gate read.
    _ri = getattr(orch, "last_requirement_input", None) or {}
    _plan = getattr(orch, "_active_model_generation_plan", None) or {}
    _intents = {}
    _markers = {}
    for _item in _plan.get("requirement_realizations") or ():
        if isinstance(_item, dict):
            _rid = str(_item.get("requirement_id") or "").strip().upper().replace("-", "_")
            _iv = str(_item.get("response_intent") or "").strip().lower()
            if _rid and _iv:
                _intents[_rid] = _iv
            _mv = _item.get("response_markers")
            if _rid and isinstance(_mv, (list, tuple)):
                _ms = frozenset(
                    str(v or "").strip().lower() for v in _mv if str(v or "").strip()
                )
                if _ms:
                    _markers[_rid] = _ms
    require_authoritative_functional_closure(
        final_sysml,
        unmeasurable_req_ids=_ri.get("unmeasurable_req_ids") if isinstance(_ri, dict) else None,
        planned_intents=_intents or None,
        planned_markers=_markers or None,
    )
    requirements = list(res.get("requirements") or [])
    parm_text = _parm_text(rec)
    out = {
        "elapsed_s": round(elapsed_s, 1),
        "system_name": SYSTEM,
        "requirements": requirements,
        "requirement_input": dict(res.get("requirement_input") or {}),
        "final_score": res.get("final_score"),
        "best_config": res.get("best_config"),
        "pareto_alternatives": res.get("pareto_alternatives"),
        "exploratory_pareto_alternatives": res.get(
            "exploratory_pareto_alternatives"
        ),
        "dse_constraint_counts": res.get("dse_constraint_counts"),
        "dse_search_coverage": res.get("dse_search_coverage"),
        "recommended_by": res.get("recommended_by"),
        "weight_sensitivity": res.get("weight_sensitivity"),
        "recommended_estimator_feasible": res.get("recommended_estimator_feasible"),
        "recommendation_status": res.get("recommendation_status"),
        "recommendable_front_count": res.get("recommendable_front_count"),
        "variation_proposal_source": getattr(
            orch, "last_variation_proposal_source", None
        ),
        "estimator_calibration": getattr(orch, "last_estimator_calibration", None),
        "recommended_design_inputs": vars(rec),
        "dse_verification_summary": (res.get("dse_verification") or {}).get("summary"),
        "functional_closure": dict(res.get("functional_closure") or {}),
        "realization": realization,
    }
    out["run_id"] = str(uuid.uuid4())
    previous_input = None
    try:
        previous_run = json.loads(
            (latest_output_dir() / "realization_run.json").read_text(
                encoding="utf-8"
            )
        )
        previous_input = previous_run.get("requirement_input")
    except FileNotFoundError:
        pass
    out["requirement_impact"] = requirement_change_impact(
        previous_input, out["requirement_input"]
    )
    return out, final_sysml, parm_text or ""


def main() -> int:
    t0 = time.time()
    staging: Path | None = None
    run_dir: Path | None = None
    with AuthoritativeRunLock(output_root()):
        try:
            # Fail-fast marker for in-process gates (e.g. the variation-DSE catalog
            # seed): an authoritative run stops at the first unpublishable condition
            # instead of spending the rest of the budget before the finalizer refuses.
            os.environ["PROTOTYPING_AUTHORITATIVE"] = "1"
            require_authoritative_runtime()
            _require_clean_worktree()
            staging = create_staging_bundle(output_root())
            llm = create_llm(provider="vertex")
            print("LLM:", llm.__class__.__name__, flush=True)
            # Phase 9 is disabled inside the orchestrator: external evidence starts
            # only after the base bundle is written.
            pipe = PrototypingPipeline(
                llm=llm,
                max_iterations=4,
                verbose=False,
                dse_mode="variation",
                phase9_hifi=None,
            )
            # REQUIREMENT_INPUT=extract runs Phase 1 (LLM extraction from the system
            # description) instead of loading the frozen set; the frozen set stays the
            # default. The artefact records requirement_input.mode ("frozen" |
            # "llm_extracted"), so the two paths stay distinguishable.
            requirement_input_mode = os.environ.get("REQUIREMENT_INPUT", "frozen")
            if requirement_input_mode not in ("frozen", "extract"):
                raise SystemExit(
                    f"REQUIREMENT_INPUT must be 'frozen' or 'extract', got "
                    f"{requirement_input_mode!r}"
                )
            print(f"=== requirement input: {requirement_input_mode} ===", flush=True)
            gen = pipe.orchestrator.generate(
                system_name=SYSTEM,
                system_description=DRONE_DESCRIPTION,
                frozen_requirements=(
                    None if requirement_input_mode == "extract"
                    else DRONE_FROZEN_REQUIREMENTS
                ),
            )
            # Persist the generation-stage inputs and output before explore runs.
            # Explore can fail (no admissible design space, no realisable design) after
            # generation succeeded, and nothing is written to the staging bundle until
            # then, so a failed run left only a one-line error. On the extraction path
            # this is the only record of the extracted set.
            try:
                atomic_write_json(
                    staging / "generation_stage.json",
                    {
                        "artifact_role": "GENERATION_STAGE_SNAPSHOT",
                        "requirement_input": pipe.orchestrator.last_requirement_input,
                        "requirements": list(
                            getattr(pipe.orchestrator.state, "requirements", []) or []
                        ),
                        "final_score": gen.get("final_score") if isinstance(gen, dict) else None,
                        "model_qualification": gen.get("model_qualification") if isinstance(gen, dict) else None,
                    },
                )
                if isinstance(gen, dict) and isinstance(gen.get("model_sysml"), str):
                    atomic_write_text(staging / "committed_model.sysml", gen["model_sysml"])
            except Exception as snap_exc:  # diagnostics do not fail the run
                print(f"  (generation-stage snapshot not written: {snap_exc})", flush=True)
            res = pipe.orchestrator.explore(gen, mcts_iterations=20)
            base, final_sysml, parm_text = _build_base_artifacts(
                pipe, res, time.time() - t0
            )
            run_id = base["run_id"]
            run_dir = assign_run_id(staging, run_id, output_root())
            staging = None

            atomic_write_json(run_dir / "realization_run.json", base)
            atomic_write_json(
                run_dir / "requirement_dependency_graph.json",
                base["requirement_input"]["dependency_graph"],
            )
            atomic_write_json(
                run_dir / "requirement_impact.json",
                base["requirement_impact"],
            )
            atomic_write_text(run_dir / "final_model.sysml", final_sysml)
            atomic_write_text(run_dir / "recommended.parm", parm_text)
            atomic_write_json(run_dir / "canonical_run.json", {
                "run_id": run_id,
                "pipeline_report": pipe.build_run_report(res),
                "requirements": base["requirements"],
                "model_artifact": "final_model.sysml",
                "parm_artifact": "recommended.parm",
                "execution": {
                    "llm_provider": "vertex",
                    "llm_model": getattr(llm, "model", None),
                    "langsmith_tracing": os.getenv("LANGSMITH_TRACING", "").lower()
                    in {"1", "true", "yes"},
                    "langsmith_project": os.getenv("LANGSMITH_PROJECT"),
                },
            })
            write_state(run_dir, "EVIDENCE", run_id=run_id)
            print(f"\n=== base bundle fixed: {run_dir} ===", flush=True)

            # Gazebo first; SITL then writes the execution-aware matrix from the fresh
            # Gazebo report. A non-zero exit can be a scientific FAIL, so the finalizer
            # decides completeness.
            for layer, script in (
                ("gazebo", "run_gazebo_feasibility.py"),
                ("sitl", "run_sitl_feasibility.py"),
            ):
                print(f"\n=== authoritative evidence: {layer} ===", flush=True)
                rc = _run_evidence(script, run_dir)
                print(f"  {layer} runner returncode={rc}", flush=True)

            print("\n=== provenance finalization ===", flush=True)
            rc = _run_evidence("finalize_authoritative_run.py", run_dir, timeout_s=120)
            if rc != 0:
                raise RuntimeError(f"authoritative finalizer failed with returncode={rc}")
            authority = json.loads(
                (run_dir / "authoritative_run.json").read_text(encoding="utf-8")
            )
            if authority.get("run_id") != run_id:
                raise RuntimeError("finalizer returned a different run_id")
            latest = publish_latest(run_dir, output_root())
            print(f"\n=== authoritative run published: {run_id} ===", flush=True)
            print(f"bundle: {run_dir}", flush=True)
            print(f"latest: {latest}", flush=True)
            return 0
        except BaseException as exc:
            failed = run_dir or staging
            if failed is not None and failed.exists():
                try:
                    write_state(
                        failed, "FAILED",
                        error=str(exc) or exc.__class__.__name__,
                    )
                except Exception:
                    pass
                # Without this the bounded Step 1 attempts - response excerpts,
                # per-attempt parse status - die with the process, leaving a one-line error.
                step1_attempts = list(getattr(exc, "plan_attempts", ()) or ())
                if step1_attempts:
                    try:
                        atomic_write_json(
                            failed / "step1_plan_attempts.json",
                            {
                                "schema_version": "1.0",
                                "artifact_role": "TYPED_MODEL_PLAN_ATTEMPTS",
                                "error_type": type(exc).__name__,
                                "attempts": step1_attempts,
                            },
                        )
                    except Exception:
                        pass
                closure = getattr(exc, "functional_closure", None)
                if closure:
                    try:
                        atomic_write_json(
                            failed / "functional_closure_failure.json",
                            {
                                "schema_version": "1.0",
                                "artifact_role": "FUNCTIONAL_CLOSURE_FAILURE",
                                "error_type": type(exc).__name__,
                                "functional_closure": closure,
                            },
                        )
                        # The audit's verdict is only checkable against the
                        # exact revision it judged.
                        atomic_write_text(
                            failed / "terminal_model.sysml",
                            getattr(exc, "terminal_model_text", "") or "",
                        )
                    except Exception:
                        pass
            raise


if __name__ == "__main__":
    raise SystemExit(main())
