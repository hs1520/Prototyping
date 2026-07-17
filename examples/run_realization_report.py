"""Run one isolated, authoritative end-to-end realization experiment.

The run is scientifically ordered and single-writer:

  Phase 1-8 -> immutable base fingerprints -> Gazebo -> SITL/executed matrix
  -> provenance finalizer -> atomic ``examples/output/latest`` publication.

Every run lives under ``examples/output/runs/<run_id>``.  A failed/interrupted
run is retained for diagnosis but can never replace ``latest``.
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
from src.prototyping.artifact_provenance import build_run_provenance
from src.prototyping.artifact_store import (
    OUTPUT_DIR_ENV,
    AuthoritativeRunLock,
    assign_run_id,
    atomic_write_json,
    atomic_write_text,
    create_staging_bundle,
    output_root,
    publish_latest,
    write_state,
)
from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.sitl.dse_sitl_params import design_to_sitl_parm
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE

sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import DRONE_DESCRIPTION, DRONE_REQUIREMENTS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "AutonomousDrone"


def require_authoritative_runtime() -> None:
    """Fail before any LLM call when the Syside-backed runtime is unavailable.

    ``conda run -n AI-Prototyping python`` can resolve a pyenv shim ahead of
    the environment interpreter on some shells. Both syntax checking and the
    structured LiteModel then degrade to permissive/empty fallbacks, which is
    unacceptable for an authoritative run.
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
    """Legacy scratch-output helper; isolated run bundles never need this path."""
    path = Path(parm_path)
    if not path.exists():
        return False
    os.replace(path, Path(str(path) + ".stale"))
    return True


def require_authoritative_functional_closure(model_sysml: str) -> list[str]:
    """Reject publication while model-fixable FUNC gaps remain."""
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
        model_sysml, SYSTEM, strict=True
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
    lines = list(design_to_sitl_parm(design))
    present = {
        line.split()[0] for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }
    for key, value in (ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {}).items():
        if key not in present:
            lines.append(f"{key:<20} {value}")
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
    """Fail before the expensive LLM run unless the executable code is committed."""
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
    # Independent final publication gate (do not trust only run metadata).
    require_authoritative_functional_closure(final_sysml)
    parm_text = _parm_text(rec)
    out = {
        "elapsed_s": round(elapsed_s, 1),
        "system_name": SYSTEM,
        # Persist the exact extracted requirement set, not merely its count.
        "requirements": list(res.get("requirements") or []),
        "final_score": res.get("final_score"),
        "best_config": res.get("best_config"),
        "pareto_alternatives": res.get("pareto_alternatives"),
        "exploratory_pareto_alternatives": res.get(
            "exploratory_pareto_alternatives"
        ),
        "dse_constraint_counts": res.get("dse_constraint_counts"),
        "dse_search_coverage": res.get("dse_search_coverage"),
        "recommended_by": res.get("recommended_by"),
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
    out["artifact_provenance"] = build_run_provenance(
        model_sysml=final_sysml,
        recommended_design=out["recommended_design_inputs"],
        realization=realization,
        requirements=out["requirements"],
        parm_text=parm_text,
    )
    return out, final_sysml, parm_text or ""


def main() -> int:
    t0 = time.time()
    staging: Path | None = None
    run_dir: Path | None = None
    with AuthoritativeRunLock(output_root()):
        try:
            require_authoritative_runtime()
            _require_clean_worktree()
            staging = create_staging_bundle(output_root())
            llm = create_llm(provider="vertex")
            print("LLM:", llm.__class__.__name__, flush=True)
            # Phase 9 is deliberately disabled inside the orchestrator.  External
            # evidence may start only after the base bundle is fingerprinted.
            pipe = PrototypingPipeline(
                llm=llm,
                max_iterations=4,
                verbose=False,
                dse_mode="variation",
                phase9_hifi=None,
            )
            gen = pipe.orchestrator.generate(
                system_name=SYSTEM,
                system_description=DRONE_DESCRIPTION,
                additional_requirements=DRONE_REQUIREMENTS,
            )
            res = pipe.orchestrator.explore(gen, mcts_iterations=20)
            base, final_sysml, parm_text = _build_base_artifacts(
                pipe, res, time.time() - t0
            )
            run_id = base["artifact_provenance"]["run_id"]
            run_dir = assign_run_id(staging, run_id, output_root())
            staging = None

            atomic_write_json(run_dir / "realization_run.json", base)
            atomic_write_text(run_dir / "final_model.sysml", final_sysml)
            atomic_write_text(run_dir / "recommended.parm", parm_text)
            # Canonical snapshot contains the report plus exact model/requirement
            # references.  It is sufficient to recover the run without another LLM call.
            atomic_write_json(run_dir / "canonical_run.json", {
                "run_id": run_id,
                "pipeline_report": pipe.build_run_report(res),
                "requirements": base["requirements"],
                "model_artifact": "final_model.sysml",
                "parm_artifact": "recommended.parm",
                "artifact_provenance": base["artifact_provenance"],
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

            # Gazebo first; SITL writes the final execution-aware matrix using
            # the already-fresh Gazebo report.  Non-zero can be a scientific
            # FAIL, so the finalizer—not the return code—decides completeness.
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
            if not authority.get("provenance_validated"):
                raise RuntimeError("finalizer did not validate provenance")
            if authority.get("artifact_provenance", {}).get("run_id") != run_id:
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
            raise


if __name__ == "__main__":
    raise SystemExit(main())
