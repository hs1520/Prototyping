"""真 LLM 端到端(Vertex)过 Phase 8,产出 meet-in-the-middle closure 报告数据。

在 run_pipeline.py 的标准流程上,额外把 result["realization"](Phase 8)与关键
DSE 字段显式 dump 到 examples/output/realization_run.json,供导师报告引用。

运行:
  PYTHONPATH=. /Users/huangsongyi/miniforge3/envs/AI-Prototyping/bin/python \
      examples/run_realization_report.py
"""
import json
import os
import sys
import time

import src.config  # noqa: F401  (loads .env)
from src.prototyping.provider_factory import create_llm
from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.artifact_provenance import build_run_provenance
from src.sitl.dse_sitl_params import design_to_sitl_parm
from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE

sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import DRONE_DESCRIPTION, DRONE_REQUIREMENTS  # noqa: E402

SYSTEM = "AutonomousDrone"


def retire_stale_parm(parm_path: str) -> bool:
    """No recommendation this run → retire any leftover recommended.parm.

    A run that falls back before Phase 8 (no recommended design) must not leave a
    previous run's recommended.parm in place: downstream SITL runs would silently
    fly a design unrelated to the current model. Returns True if a file was retired
    (renamed to ``<parm_path>.stale``).
    """
    if not os.path.exists(parm_path):
        return False
    os.replace(parm_path, parm_path + ".stale")
    return True


if __name__ == "__main__":
    t0 = time.time()
    llm = create_llm(provider="vertex")
    print("LLM:", llm.__class__.__name__, flush=True)

    pipe = PrototypingPipeline(llm=llm, max_iterations=4, verbose=False, dse_mode="variation")
    gen = pipe.orchestrator.generate(system_name=SYSTEM, system_description=DRONE_DESCRIPTION,
                                     additional_requirements=DRONE_REQUIREMENTS)
    res = pipe.orchestrator.explore(gen, mcts_iterations=20)
    pipe.save_run_report(res)

    orch = pipe.orchestrator
    rec = getattr(orch, "last_recommended_design", None)
    realization = res.get("realization")
    if realization is not None:
        realization = dict(realization)
        realization["recommended_realizable"] = getattr(orch, "last_recommended_realizable", None)
        realization["realizable_front_count"] = getattr(orch, "last_realizable_front_count", None)
        realization["recommended_by"] = getattr(orch, "last_recommended_by", None)
        realization["recommended_estimator_feasible"] = getattr(
            orch, "last_recommended_estimator_feasible", None)
    final_sysml = res.get("model_sysml") or ""
    parm_text = None
    if rec is not None:
        lines = list(design_to_sitl_parm(rec))
        existing = {
            line.split()[0] for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        }
        for key, value in (ARDUPILOT_COPTER_PROFILE.get("base_sitl_params") or {}).items():
            if key not in existing:
                lines.append(f"{key:<20} {value}")
        parm_text = (
            "# Recommended design SITL params; native SITL is architecture-"
            "nondiscriminating for endurance.\n" + "\n".join(lines) + "\n"
        )
    out = {
        "elapsed_s": round(time.time() - t0, 1),
        "final_score": res.get("final_score"),
        "best_config": res.get("best_config"),
        "pareto_alternatives": res.get("pareto_alternatives"),
        "recommended_by": res.get("recommended_by"),
        "recommended_estimator_feasible": res.get("recommended_estimator_feasible"),
        "variation_proposal_source": getattr(orch, "last_variation_proposal_source", None),
        "estimator_calibration": getattr(orch, "last_estimator_calibration", None),
        "recommended_design_inputs": (vars(rec) if rec is not None else None),
        "dse_verification_summary": (res.get("dse_verification") or {}).get("summary"),
        "realization": realization,
    }
    out["artifact_provenance"] = build_run_provenance(
        model_sysml=final_sysml,
        recommended_design=out["recommended_design_inputs"],
        realization=realization,
        parm_text=parm_text,
    )
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "realization_run.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    # S2 traceability: persist the exact final SysML and the recommended design's
    # native-SITL parameterization. Best-effort: never break the JSON artifact.
    try:
        final_sysml_path = os.path.join(outdir, "final_model.sysml")
        with open(final_sysml_path, "w", encoding="utf-8") as f:
            f.write(final_sysml)
        parm_path = os.path.join(outdir, "recommended.parm")
        if parm_text is not None:
            with open(parm_path, "w", encoding="utf-8") as f:
                f.write(parm_text)
        elif retire_stale_parm(parm_path):
            print("  ⚠ no recommended design this run — retired stale recommended.parm "
                  "→ recommended.parm.stale", flush=True)
    except Exception as e:
        print(f"  ⚠ final model / recommended.parm persistence skipped ({e})", flush=True)
    print(f"\n=== realization run dumped → {path} ({out['elapsed_s']}s) ===", flush=True)
    r = out["realization"] or {}
    print("verdict:", r.get("verdict"), "|", r.get("summary"), flush=True)
