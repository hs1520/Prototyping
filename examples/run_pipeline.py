"""手动运行完整 DSE 流程（generate -> variation 双层 DSE -> 精修/仿真 -> Phase 7 验证）。"""
import os
import sys
import src.config  # noqa: F401  (loads .env)
from src.prototyping.provider_factory import create_llm
from src.app.pipeline import PrototypingPipeline

sys.path.insert(0, os.path.dirname(__file__))
from drone_system_v2 import (  # noqa: E402
    DRONE_DESCRIPTION,
    DRONE_FROZEN_REQUIREMENTS,
    DRONE_REQUIREMENTS,
)

SYSTEM = "AutonomousDrone"
DESC = DRONE_DESCRIPTION

if __name__ == "__main__":
    llm = create_llm(provider="vertex")
    print("LLM:", llm.__class__.__name__,
          "langsmith:", getattr(llm, "langsmith_enabled", "?"), flush=True)

    pipe = PrototypingPipeline(llm=llm, max_iterations=4, verbose=False, dse_mode="variation")
    print("dse_mode → use_variation_dse:", pipe.orchestrator.use_variation_dse, flush=True)

    # Split generate / explore to capture the pre-DSE and post-DSE models.
    gen = pipe.orchestrator.generate(system_name=SYSTEM, system_description=DESC,
                                     frozen_requirements=DRONE_FROZEN_REQUIREMENTS)
    initial_sysml = gen["model_sysml"]
    res = pipe.orchestrator.explore(gen, mcts_iterations=20)
    final_sysml = res["model_sysml"]
    # Drives the orchestrator directly to capture the pre-DSE model, bypassing the
    # auto-saving pipeline entry points, so save explicitly.
    pipe.save_run_report(res)

    # Restore verbatim requirement doc text; generation paraphrases it (SAFE-005
    # "all other safety responses" -> "all calculations"). Deterministic pass.
    from src.dse.requirement_spec import bind_current_payload, enforce_requirement_text
    for fn in (enforce_requirement_text, bind_current_payload):
        initial_sysml = fn(initial_sysml, DRONE_REQUIREMENTS)
        final_sysml = fn(final_sysml, DRONE_REQUIREMENTS)

    # artifact_store.output_dir() treats the examples/output root as not a
    # writable artifact bundle; honour PROTOTYPING_OUTPUT_DIR when set (default
    # unchanged for manual runs).
    outdir = os.environ.get("PROTOTYPING_OUTPUT_DIR") or os.path.join(
        os.path.dirname(__file__), "output")
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

    # Opt-in Gazebo flight of the DSE-recommended design (~5 min, Docker), gated
    # by RUN_GAZEBO=1 and off by default. Runs before coverage so its verdict can
    # upgrade the endurance requirement to flight-verified.
    gv = None
    if os.environ.get("RUN_GAZEBO") == "1":
        design = getattr(pipe.orchestrator, "last_recommended_design", None)
        from gazebo_poc.gazebo_verify import summary_line, verify_recommended_design
        print("\n[RUN_GAZEBO] flying recommended design in Gazebo ...", flush=True)
        gv = verify_recommended_design(design, DRONE_REQUIREMENTS)
        print("gazebo verify  :", summary_line(gv))

    # Opt-in architecture-axis calibration: sweep quad/hexa/octa variants through
    # Gazebo and rank-correlate estimated against measured hover power. SITL-default
    # fixes frame mass (octa==quad), so it gives no such anchor. Three flights
    # ~15 min, hence a gate on top of RUN_GAZEBO.
    if (os.environ.get("RUN_GAZEBO") == "1"
            and os.environ.get("RUN_GAZEBO_CALIB") == "1"):
        design = getattr(pipe.orchestrator, "last_recommended_design", None)
        if design is not None:
            from src.dse.gazebo_oracle import calibrate_architecture_axis
            print("\n[RUN_GAZEBO_CALIB] architecture sweep (quad/hexa/octa) ...",
                  flush=True)
            arch = calibrate_architecture_axis(design)
            print("gazebo arch calib:", arch.summary())
            for p in arch.points:
                measured = (f"{p.measured_power_w:.0f} W"
                            if p.measured_power_w is not None else f"— ({p.note})")
                print(f"  {p.label}: predicted {p.predicted_power_w:.0f} W, "
                      f"measured {measured}, stable={p.hover_stable}")

    # satisfy is allocation/intent, not proof: report how many requirements have
    # evidence versus are allocated-only.
    from src.dse.requirement_coverage import classify_requirement_coverage, coverage_summary
    cov = classify_requirement_coverage(final_sysml, DRONE_REQUIREMENTS, dynamic=True, gazebo=gv)
    print("req evidence     :", coverage_summary(cov))
    allocated = sorted(r for r, l in cov.items() if l == "allocated-only")
    if allocated:
        print("  allocated-only (satisfy=intent, NOT verified — behaviour/protocol, out of scope):")
        print("   ", ", ".join(allocated))

    print("\n初始模型 (DSE 前) →", initial_path)
    print("最终模型 (DSE 后) →", final_path)
