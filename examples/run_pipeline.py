"""手动运行完整 DSE 流程（generate → variation 双层 DSE → 精修/仿真 → Phase 7 验证）。

前置：
  - conda 环境 AI-Prototyping（含 langsmith / numpy / scipy / syside）
  - .env 提供 Vertex 凭据（src.config 自动加载）

运行：
  cd <repo>
  PYTHONPATH=. /Users/huangsongyi/miniforge3/envs/AI-Prototyping/bin/python examples/run_pipeline.py

改 system / 需求：编辑下面的 SYSTEM 和 DESC。
切 DSE 路径：改 PrototypingPipeline(dse_mode=...) — "variation"(默认,双层) / "bilevel"(算子) / "off"(标量)。
"""
import os
import sys
import src.config  # noqa: F401  (loads .env)
from src.prototyping.provider_factory import create_llm
from src.prototyping.pipeline import PrototypingPipeline

# Reuse the industry-grade v2 system description + INCOSE-style requirements.
sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import DRONE_DESCRIPTION, DRONE_REQUIREMENTS  # noqa: E402

SYSTEM = "AutonomousDrone"
DESC = DRONE_DESCRIPTION

if __name__ == "__main__":
    llm = create_llm(provider="vertex")
    print("LLM:", llm.__class__.__name__,
          "langsmith:", getattr(llm, "langsmith_enabled", "?"), flush=True)

    pipe = PrototypingPipeline(llm=llm, max_iterations=4, verbose=False, dse_mode="variation")
    print("dse_mode → use_variation_dse:", pipe.orchestrator.use_variation_dse, flush=True)

    # Split generate / explore so we capture BOTH the pre-DSE model and the post-DSE model.
    gen = pipe.orchestrator.generate(system_name=SYSTEM, system_description=DESC,
                                     additional_requirements=DRONE_REQUIREMENTS)
    initial_sysml = gen["model_sysml"]                          # after generate, BEFORE DSE
    res = pipe.orchestrator.explore(gen, mcts_iterations=20)    # DSE + refinement + verification
    final_sysml = res["model_sysml"]                            # AFTER DSE

    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    initial_path = os.path.join(outdir, "initial_model.sysml")
    final_path = os.path.join(outdir, "final_model.sysml")
    with open(initial_path, "w") as f:
        f.write(initial_sysml)
    with open(final_path, "w") as f:
        f.write(final_sysml)

    print("\n=== RESULT ===", flush=True)
    print("final score      :", res.get("final_score"))
    sim = res.get("simulation_result")
    print("reachability     :", getattr(sim, "reachability_score", "?"))
    print("best_config      :", res.get("best_config"))
    print("pareto front     :", len(res.get("pareto_alternatives", [])), "non-dominated point(s)")
    dv = res.get("dse_verification")
    print("DSE→SITL verify  :", dv["summary"] if dv else "(no quantified requirements)")

    # Honest verification-coverage of the satisfy claims: satisfy = allocation/intent, not
    # proof. Report how many requirements actually have evidence vs are allocated-only.
    from src.dse.requirement_coverage import classify_requirement_coverage, coverage_summary
    cov = classify_requirement_coverage(final_sysml, DRONE_REQUIREMENTS, dynamic=True)
    print("req evidence     :", coverage_summary(cov))
    allocated = sorted(r for r, l in cov.items() if l == "allocated-only")
    if allocated:
        print("  allocated-only (satisfy=intent, NOT verified — behaviour/protocol, out of scope):")
        print("   ", ", ".join(allocated))

    print("\n初始模型 (DSE 前) →", initial_path)
    print("最终模型 (DSE 后) →", final_path)
