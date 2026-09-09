"""Main orchestration module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.orchestrator import Orchestrator
from ..llm.interface import LLMInterface
from ..rag.pinecone_wrapper import PineconeWrapper
from ..rag.retriever import RAGRetriever
from ..sysml.model import SysMLModel
from ..sysml.lite_model import SysMLLiteModel
from ..dse.evaluator import EVALUATOR_VERSION as _EVALUATOR_VERSION
from ..utils.suppressed import suppressed_summary

_SysMLModelTypes = (SysMLModel, SysMLLiteModel)


class PrototypingPipeline:
    """High-level API for AI-assisted MBSE rapid prototyping."""

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
        use_surgical_refinement: bool = True,
        use_deterministic_fixers: bool = True,
        design_generation_mode: str = "multistep",
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
        # RAG is optional: without a Pinecone key the pipeline runs RAG-free (agents
        # accept rag_retriever=None) instead of crashing, which also gives offline runs
        # and RAG on/off ablations.
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
            use_surgical_refinement=use_surgical_refinement,
            use_deterministic_fixers=use_deterministic_fixers,
            design_generation_mode=design_generation_mode,
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
        # DSE mode at the user entry point. The orchestrator flag defaults off
        # (direct-construction contract); the pipeline opts the chosen path in.
        self.orchestrator.use_variation_dse = self._dse_flags(dse_mode)
        # Phase 9 is off by default here. The authoritative driver freezes one base
        # bundle before running Gazebo then SITL and is the only path that publishes
        # ``examples/output/latest``; this switch is a best-effort developer seam.
        self.orchestrator.phase9_hifi = phase9_hifi

    @staticmethod
    def _dse_flags(mode: str) -> bool:
        m = (mode or "").strip().lower()
        if m == "off":
            raise ValueError(
                "dse_mode='off' (legacy scalar DSE) was removed; "
                "use 'variation' or 'bilevel'."
            )
        if m == "bilevel":
            return False
        return True

    def generate_system(
        self,
        system_name: str,
        description: str,
        additional_requirements: Optional[List[str]] = None,
        parse_strict: Optional[bool] = None,
        platform_profile: Optional[Dict[str, Any]] = None,
        sitl: bool = False,
        sitl_output_dir: str = "sitl_output",
        sitl_run_l2: bool = False,
        sitl_auto_launch: bool = False,
        sitl_host: str = "127.0.0.1",
        sitl_port: int = 5760,
        sitl_fdm_backend: str = "native",
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate a validated SysML v2 model without Design Space Exploration."""
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

        print()
        print("  [L2] 测试脚本生成")
        scripts = bridge.generate_l2_scripts()
        for s in scripts:
            print(f"    → {s.name}")

        l2_results = []
        if run_l2:
            print()
            print("  [L2] 执行测试" + (" (自动启动 SITL)" if auto_launch else ""))
            l2_results = bridge.run_l2(
                per_test_sitl=auto_launch,
            )
            ok_l2 = sum(1 for r in l2_results if r.passed)
            print(f"\n  L2 测试结果: {ok_l2}/{len(l2_results)} 通过")
            for r in l2_results:
                icon = "✓" if r.passed else "✗"
                dur = f"  [{r.duration_s:.1f}s]" if r.duration_s else ""
                print(f"    {icon} {r.req_id:<20} {r.message}{dur}")

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
        """Run multi-objective Design Space Exploration on a previously validated model."""
        result = self.orchestrator.explore(
            generate_result=generate_result,
            mcts_iterations=mcts_iterations,
            mcts_seed=mcts_seed,
            mcts_patience=mcts_patience,
        )
        self.save_run_report(result)
        return result

    @staticmethod
    def build_run_report(result: Dict[str, Any]) -> Dict[str, Any]:
        """JSON-serialisable snapshot of a pipeline run (no model objects)."""
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
            "evaluator_version": _EVALUATOR_VERSION,
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
            "action_semantics_audit": result.get("action_semantics_audit"),
        }
        if result.get("ablation") is not None:
            report["ablation"] = result.get("ablation")
        revised = result.get("revised_experiment")
        if revised:
            report["experiment_namespace"] = revised.get(
                "experiment_namespace"
            )
            report["configuration"] = revised.get("configuration")
            report["revised_experiment"] = revised
            collaboration = result.get("collaboration")
            report["collaboration"] = collaboration
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
            report["simulation"] = {
                "reachability_score": getattr(sim, "reachability_score", None),
                "scenarios_passed": len(sim.passed_scenarios()),
                "scenarios_total": len(sim.scenario_results),
            }
        ver = result.get("dse_verification")
        if ver:
            report["dse_verification_summary"] = ver.get("summary")
        # Phase 8 outcome goes in the canonical run report; otherwise the
        # meet-in-the-middle verdict lives only in the example script's dump.
        realization = result.get("realization")
        if realization:
            report["realization"] = {
                "verdict": realization.get("verdict"),
                "summary": realization.get("summary"),
                "chosen": realization.get("chosen"),
                "forward_flight_ok": realization.get("forward_flight_ok"),
                "rank_preservation": realization.get("rank_preservation"),
                "resize_note": realization.get("resize_note"),
                # The datasheet/forward-flight tier input. Dropping it in this projection made
                # the matrix report three "unassigned" requirements that were a runner
                # artefact, not a model defect (run3, A2).
                "per_requirement": realization.get("per_requirement"),
                "failed_checks": realization.get("failed_checks"),
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
        """Write the run report to ``logs/run_<system>_<timestamp>.json``."""
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
