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

sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import DRONE_DESCRIPTION, DRONE_REQUIREMENTS  # noqa: E402

SYSTEM = "AutonomousDrone"

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
    out = {
        "elapsed_s": round(time.time() - t0, 1),
        "final_score": res.get("final_score"),
        "best_config": res.get("best_config"),
        "pareto_alternatives": res.get("pareto_alternatives"),
        "recommended_by": res.get("recommended_by"),
        "recommended_design_inputs": (vars(rec) if rec is not None else None),
        "dse_verification_summary": (res.get("dse_verification") or {}).get("summary"),
        "realization": realization,
    }
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "realization_run.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n=== realization run dumped → {path} ({out['elapsed_s']}s) ===", flush=True)
    r = out["realization"] or {}
    print("verdict:", r.get("verdict"), "|", r.get("summary"), flush=True)
