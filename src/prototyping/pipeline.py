"""
Main orchestration module.

Provides the high-level API for the AI-assisted MBSE rapid prototyping
framework, combining all components into a unified workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..agents.orchestrator import Orchestrator
from ..llm.interface import LLMInterface
from ..rag.pinecone_wrapper import PineconeWrapper
from ..rag.retriever import RAGRetriever
from ..sysml.model import SysMLModel
from ..sysml.lite_model import SysMLLiteModel
from ..utils.suppressed import suppressed_summary

_SysMLModelTypes = (SysMLModel, SysMLLiteModel)


class PrototypingPipeline:
    """
    High-level API for AI-assisted MBSE rapid prototyping.

    This is the main entry point for using the framework. It combines:
    - Chain of Thought prompting for design reasoning
    - Retrieval Augmented Generation for domain knowledge
    - Multi-agent coordination for specialized tasks
    - Monte Carlo Tree Search for design space exploration

    Example usage:
        pipeline = PrototypingPipeline(llm=llm)
        result = pipeline.prototype_system(
            system_name="AutonomousDrone",
            description="A drone that autonomously delivers packages...",
        )
        print(result["model_sysml"])
    """

    def __init__(
        self,
        llm: LLMInterface,
        pinecone_wrapper: Optional[PineconeWrapper] = None,
        rag_index_name: str = "ai-prototyping-sysml-v2",
        rag_namespace: str = "SysML-V2-Release",
        quality_threshold: float = 0.75,
        max_iterations: int = 3,
        parse_strict: bool = False,
        verbose: bool = False,
        dse_mode: str = "variation",
        phase9_hifi: Optional[str] = None,
        revised_experiment_arm: Optional[Any] = None,
        r2_generation_mode: Optional[str] = None,
        task_session_max_turns: int = 12,
        task_session_max_tokens: int = 600000,
        r2_authored_syntax_max_attempts: int = 3,
        maximum_ag_repair_attempts: int = 3,
        maximum_plan_attempts: Optional[int] = None,
    ):
        self.llm = llm
        self.parse_strict = parse_strict
        # RAG is an enhancement, not a hard dependency: without a Pinecone key
        # the pipeline runs RAG-free (agents accept rag_retriever=None) instead
        # of crashing — enabling offline runs and clean RAG on/off ablations.
        self.pinecone: Optional[PineconeWrapper] = None
        self.rag: Optional[RAGRetriever] = None
        try:
            self.pinecone = pinecone_wrapper or PineconeWrapper(
                default_namespace=rag_namespace
            )
            self.rag = RAGRetriever(
                llm=self.llm,
                pinecone_wrapper=self.pinecone,
                index_name=rag_index_name,
                namespace=rag_namespace,
            )
        except Exception as e:
            print(f"  ⚠ RAG unavailable ({e}) — continuing without retrieval")
        self.orchestrator = Orchestrator(
            llm=self.llm,
            rag_retriever=self.rag,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
            verbose=verbose,
            revised_experiment_arm=revised_experiment_arm,
            r2_generation_mode=r2_generation_mode,
            r2_authored_syntax_max_attempts=r2_authored_syntax_max_attempts,
            maximum_ag_repair_attempts=maximum_ag_repair_attempts,
            **(
                {"maximum_plan_attempts": maximum_plan_attempts}
                if maximum_plan_attempts is not None else {}
            ),
            task_session_max_turns=task_session_max_turns,
            task_session_max_tokens=task_session_max_tokens,
        )
        # DSE mode at the user entry point. Orchestrator's own flag stays default
        # OFF (direct-construction contract); the pipeline opts the chosen path in.
        self.orchestrator.use_variation_dse = self._dse_flags(dse_mode)
        # Phase 9 is default-OFF here.  The authoritative driver freezes a single
        # base bundle before running Gazebo then SITL and is the only path allowed
        # to publish ``examples/output/latest``.  This switch remains an explicit,
        # best-effort developer seam; it never creates a published authority.
        self.orchestrator.phase9_hifi = phase9_hifi

    @staticmethod
    def _dse_flags(mode: str) -> bool:
        """Map a dse_mode string to use_variation_dse.

        "variation" (default) — LLM-declared variation points + domain objective.
        "bilevel"             — catalog-operator bilevel DSE (MO-MCTS + inner BO).

        The legacy scalar DSE ("off") was removed; requesting it is an error.
        """
        m = (mode or "").strip().lower()
        if m == "off":
            raise ValueError(
                "dse_mode='off' (legacy scalar DSE) was removed; "
                "use 'variation' or 'bilevel'."
            )
        if m == "bilevel":
            return False
        return True  # default: variation

    def generate_system(
        self,
        system_name: str,
        description: str,
        additional_requirements: Optional[List[str]] = None,
        parse_strict: Optional[bool] = None,
        # ── Platform Profile（影响 mode machine accept 命令命名）────────
        platform_profile: Optional[Dict[str, Any]] = None,
        # ── Phase 5: ArduPilot SITL 验证 ──────────────────────────────
        sitl: bool = False,
        sitl_output_dir: str = "sitl_output",
        sitl_run_l2: bool = False,
        sitl_auto_launch: bool = False,
        sitl_host: str = "127.0.0.1",
        sitl_port: int = 5760,
        sitl_fdm_backend: str = "native",
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate a validated SysML v2 model without Design Space Exploration.

        Run Phases 1-4 (requirements → design → syntax gate → refinement
        → simulation).  Optionally runs Phase 5 (ArduPilot SITL validation).

        Parameters
        ----------
        sitl : bool
            Enable Phase 5 — generate .parm file + L2 test scripts.
        sitl_output_dir : str
            Directory to write SITL artifacts (.parm, test_*.py).
        sitl_run_l2 : bool
            Also execute L2 tests against a running SITL instance.
        sitl_fdm_backend : str
            "native" (default) uses ArduPilot's built-in simplified physics.
            "gazebo" switches to external FDM and auto-launches the
            headless_gazebo Docker container (needed for requirements that
            depend on real flight dynamics, e.g. gripper/parachute checks).
        sitl_auto_launch : bool
            Auto-launch/stop arducopter when sitl_run_l2=True.
        sitl_host / sitl_port
            SITL connection endpoint (used when sitl_run_l2=True).

        Returns
        -------
        {system_name, requirements, model, model_sysml, model_summary,
         final_score, iterations, evaluation_history, simulation_result,
         sitl_report (if sitl=True)}
        """
        result = self.orchestrator.generate(
            system_name=system_name,
            system_description=description,
            additional_requirements=additional_requirements,
            parse_strict=(parse_strict if parse_strict is not None else self.parse_strict),
            platform_profile=platform_profile,
            frozen_requirements=frozen_requirements,
        )

        if sitl:
            result = self._run_sitl_phase(
                result=result,
                output_dir=sitl_output_dir,
                run_l2=sitl_run_l2,
                auto_launch=sitl_auto_launch,
                host=sitl_host,
                port=sitl_port,
                platform_profile=platform_profile,
                fdm_backend=sitl_fdm_backend,
            )

        self.save_run_report(result)
        return result

    # ------------------------------------------------------------------
    # Phase 5 — ArduPilot SITL
    # ------------------------------------------------------------------

    def _run_sitl_phase(
        self,
        result: Dict[str, Any],
        output_dir: str,
        run_l2: bool,
        auto_launch: bool,
        host: str,
        port: int,
        platform_profile: Optional[Dict[str, Any]] = None,
        fdm_backend: str = "native",
    ) -> Dict[str, Any]:
        """内部方法：运行 SITL 阶段并打印进度，将报告写入 result。"""
        from ..sitl.sitl_bridge import SITLBridge

        W = 62
        print()
        print("=" * W)
        print("Phase 5: ArduPilot SITL 验证")
        if platform_profile:
            print(f"  Platform: {platform_profile.get('platform', 'unknown')}")
        else:
            print("  Platform: none (L2 accept tests skipped — no platform_profile)")
        print(f"  FDM backend: {fdm_backend}"
              + ("  (will auto-launch headless_gazebo)" if fdm_backend == "gazebo" else ""))
        print("-" * W)

        model = result.get("model")
        if model is None:
            print("  ✗ 模型不存在，跳过 SITL 阶段")
            return result

        conn = f"tcp:{host}:{port}"
        bridge = SITLBridge(
            model=model,
            output_dir=output_dir,
            connection_string=conn,
            llm=self.llm,
            platform_profile=platform_profile,
            verbose=True,
            fdm_backend=fdm_backend,
        )

        # ── L1：生成 .parm + 静态验证 ────────────────────────────────
        print()
        print("  [L1] 参数文件生成")
        parm_path = bridge.generate_l1()
        l1_results = bridge.validate_l1()
        trace_results = bridge.validate_traceability()

        ok_l1 = sum(1 for r in l1_results if r.passed)
        print(f"  参数验证: {ok_l1}/{len(l1_results)} 通过")
        for r in l1_results:
            icon = "✓" if r.passed else "✗"
            print(f"    {icon} {r.req_id:<20} {r.message}")
        if trace_results:
            print()
            print("  [TRACE] 需求-验证一致性")
            for r in trace_results:
                print(f"    ✗ {r.req_id:<20} {r.message}")

        # ── L2：生成测试脚本 ─────────────────────────────────────────
        print()
        print("  [L2] 测试脚本生成")
        scripts = bridge.generate_l2_scripts()
        for s in scripts:
            print(f"    → {s.name}")

        # ── L2：执行测试（可选）──────────────────────────────────────
        l2_results = []
        if run_l2:
            print()
            print("  [L2] 执行测试" + (" (自动启动 SITL)" if auto_launch else ""))
            # per_test_sitl=True 时每个测试自己负责启停 SITL，无需外层 launch
            l2_results = bridge.run_l2(
                per_test_sitl=auto_launch,
            )
            ok_l2 = sum(1 for r in l2_results if r.passed)
            print(f"\n  L2 测试结果: {ok_l2}/{len(l2_results)} 通过")
            for r in l2_results:
                icon = "✓" if r.passed else "✗"
                dur = f"  [{r.duration_s:.1f}s]" if r.duration_s else ""
                print(f"    {icon} {r.req_id:<20} {r.message}{dur}")

        # ── 汇总 ──────────────────────────────────────────────────────
        from ..sitl.sitl_bridge import BridgeReport
        report = BridgeReport(
            model_name=model.name,
            parm_file=str(parm_path),
            l1_results=l1_results,
            l2_results=l2_results,
            trace_results=trace_results,
        )

        all_n = len(l1_results) + len(l2_results)
        all_ok = sum(1 for r in l1_results + l2_results if r.passed)
        print()
        blocked = len([r for r in trace_results if not r.passed])
        suffix = f", TRACE blocked {blocked}" if blocked else ""
        print(f"  SITL 阶段完成  ({all_ok}/{all_n} 通过{suffix})")
        print(f"  .parm 文件: {parm_path}")
        print("=" * W)

        result["sitl_report"] = report
        result["sitl_parm_file"] = str(parm_path)
        result["sitl_l2_scripts"] = [str(s) for s in scripts]
        result["sitl_traceability"] = [vars(r) for r in trace_results]
        return result

    def explore_design_space(
        self,
        generate_result: Dict[str, Any],
        mcts_iterations: int = 50,
        mcts_seed: Optional[int] = None,
        mcts_patience: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """
        Run multi-objective Design Space Exploration on a previously validated model.

        Takes the dict returned by generate_system() and explores the variation /
        catalog operator space (per ``dse_mode``) to find the recommended
        configuration.  The mcts_* parameters are retained for API compatibility;
        the bilevel search manages its own budget.

        Returns
        -------
        Full result dict — superset of generate_result — with updated
        model/score/sim fields plus DSE fields:
        {design_space_summary, design_space_parameters,
         best_config, pareto_alternatives}
        """
        result = self.orchestrator.explore(
            generate_result=generate_result,
            mcts_iterations=mcts_iterations,
            mcts_seed=mcts_seed,
            mcts_patience=mcts_patience,
        )
        self.save_run_report(result)
        return result

    def prototype_system(
        self,
        system_name: str,
        description: str,
        additional_requirements: Optional[List[str]] = None,
        mcts_iterations: int = 50,
        parse_strict: Optional[bool] = None,
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete AI-assisted prototyping pipeline.

        Args:
            system_name: Name of the system to design
            description: Natural language description of the system
            additional_requirements: Additional manually-specified requirements
            mcts_iterations: Number of MCTS iterations for design space exploration

        Returns:
            Dictionary containing:
            - model: SysMLModel object
            - model_sysml: SysML v2 text representation
            - requirements: List of extracted requirements
            - design_space_summary: Summary of design space exploration
            - final_score: Quality score of the final design (0-1)
            - evaluation_history: Per-iteration scores
        """
        result = self.orchestrator.prototype(
            system_name=system_name,
            system_description=description,
            additional_requirements=additional_requirements,
            mcts_iterations=mcts_iterations,
            parse_strict=(parse_strict if parse_strict is not None else self.parse_strict),
            frozen_requirements=frozen_requirements,
        )
        self.save_run_report(result)
        return result

    @staticmethod
    def build_run_report(result: Dict[str, Any]) -> Dict[str, Any]:
        """JSON-serialisable snapshot of a pipeline run (no model objects).

        Captures what a benchmark/paper needs from a run: scores, per-iteration
        history, simulation outcome, DSE decision + front, verification summary,
        and the LLM token/call ledger.
        """
        sim = result.get("simulation_result")
        report: Dict[str, Any] = {
            "system_name": result.get("system_name"),
            "final_score": result.get("final_score"),
            "iterations": result.get("iterations"),
            "requirements_count": len(result.get("requirements") or []),
            "requirements": list(result.get("requirements") or []),
            "requirement_input": result.get("requirement_input"),
            "evaluation_history": result.get("evaluation_history"),
            "best_config": result.get("best_config"),
            "pareto_alternatives": result.get("pareto_alternatives"),
            "exploratory_pareto_alternatives": result.get(
                "exploratory_pareto_alternatives"
            ),
            "dse_constraint_counts": result.get("dse_constraint_counts"),
            "dse_search_coverage": result.get("dse_search_coverage"),
            "design_space_summary": result.get("design_space_summary"),
            "llm_usage": result.get("llm_usage"),
            "terminal_consistency": result.get("terminal_consistency"),
            "model_qualification": result.get("model_qualification"),
            "model_acceptance_status": result.get(
                "model_acceptance_status"
            ),
            "whole_model_generation_plan": result.get(
                "whole_model_generation_plan"
            ),
            "generation_plan_conformance": result.get(
                "generation_plan_conformance"
            ),
            "structural_obligation_report": result.get(
                "structural_obligation_report"
            ),
            "semantic_fidelity_report": result.get(
                "semantic_fidelity_report"
            ),
            "ag_binding_report": result.get("ag_binding_report"),
            "ag_non_degradation": result.get("ag_non_degradation"),
        }
        revised = result.get("revised_experiment")
        if revised:
            report["experiment_namespace"] = revised.get(
                "experiment_namespace"
            )
            report["configuration"] = revised.get("configuration")
            report["revised_experiment"] = revised
            collaboration = result.get("collaboration")
            report["collaboration"] = collaboration
            if collaboration:
                from .run_metrics import compute_coordination_metrics
                report["coordination_metrics"] = compute_coordination_metrics(
                    collaboration, llm_usage=result.get("llm_usage")
                )
            if result.get("ag_contract_graph") is not None:
                report["ag_contract_graph"] = result.get("ag_contract_graph")
            for artifact in (
                "pattern_conformance_report",
                "failure_diagnostics",
                "repair_decisions",
                "ag_authoring_attempts",
                "verification_plan",
                "control_agenda",
            ):
                if result.get(artifact) is not None:
                    report[artifact] = result.get(artifact)
        suppressed = suppressed_summary()
        if suppressed:
            report["suppressed"] = suppressed
        if sim is not None:
            consistency = result.get("terminal_consistency") or {}
            report["simulation"] = {
                "reachability_score": getattr(sim, "reachability_score", None),
                "scenarios_passed": len(sim.passed_scenarios()),
                "scenarios_total": len(sim.scenario_results),
                "source_model_digest": consistency.get(
                    "simulation_source_model_digest"
                ),
            }
        ver = result.get("dse_verification")
        if ver:
            report["dse_verification_summary"] = ver.get("summary")
        # Phase 8 outcome belongs in the canonical run report: without it the
        # meet-in-the-middle verdict only existed in the example script's dump.
        realization = result.get("realization")
        if realization:
            report["realization"] = {
                "verdict": realization.get("verdict"),
                "summary": realization.get("summary"),
                "chosen": realization.get("chosen"),
                "forward_flight_ok": realization.get("forward_flight_ok"),
                "rank_preservation": realization.get("rank_preservation"),
                "resize_note": realization.get("resize_note"),
            }
        for key in ("recommended_by", "recommended_estimator_feasible",
                    "recommendation_status", "recommendable_front_count",
                    "variation_proposal_source", "estimator_calibration"):
            if result.get(key) is not None:
                report[key] = result.get(key)
        hifi = result.get("phase9_hifi")
        if hifi:
            report["phase9_hifi"] = {
                "mode": hifi.get("mode"),
                "summary": hifi.get("summary"),
                "layers": hifi.get("layers"),
            }
        sitl = result.get("sitl_report")
        if sitl is not None:
            l1 = list(getattr(sitl, "l1_results", []) or [])
            l2 = list(getattr(sitl, "l2_results", []) or [])
            trace = list(getattr(sitl, "trace_results", []) or [])
            report["sitl"] = {
                "safety_status": (
                    sitl.safety_status() if hasattr(sitl, "safety_status") else None
                ),
                "l1_passed": sum(1 for r in l1 if getattr(r, "passed", False)),
                "l1_total": len(l1),
                "l2_passed": sum(1 for r in l2 if getattr(r, "passed", False)),
                "l2_total": len(l2),
                "traceability_blocked": sum(1 for r in trace if not getattr(r, "passed", False)),
                "traceability": [vars(r) for r in trace],
            }
        elif result.get("sitl_traceability"):
            trace = list(result.get("sitl_traceability") or [])
            report["sitl"] = {
                "traceability_blocked": sum(1 for r in trace if not r.get("passed")),
                "traceability": trace,
            }
        return report

    def save_run_report(
        self, result: Dict[str, Any], directory: str = "logs"
    ) -> Optional[str]:
        """Write the run report to ``logs/run_<system>_<timestamp>.json``.

        Best-effort: any failure is reported but never breaks the pipeline.
        Returns the path written, or None.
        """
        import json
        import time
        from pathlib import Path

        try:
            report = self.build_run_report(result)
            out_dir = Path(directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            name = str(result.get("system_name") or "system").replace(" ", "_")
            path = out_dir / f"run_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(report, indent=2, default=str))
            print(f"  ✓ Run report saved: {path}")
            return str(path)
        except Exception as e:
            print(f"  ⚠ run report not saved ({e})")
            return None
