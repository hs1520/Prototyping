"""
Main orchestration module.

Provides the high-level API for the AI-assisted MBSE rapid prototyping
framework, combining all components into a unified workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..agents.orchestrator import Orchestrator
from ..llm.interface import LLMInterface
from ..rag.pinecone_wrapper import PineconeWrapper
from ..rag.retriever import RAGRetriever
from ..sysml.model import SysMLModel
from ..sysml.lite_model import SysMLLiteModel, build_lite_model

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
    ):
        self.llm = llm
        self.pinecone = pinecone_wrapper or PineconeWrapper(default_namespace=rag_namespace)
        self.parse_strict = parse_strict
        self.rag = RAGRetriever(
            llm=self.llm,
            pinecone_wrapper=self.pinecone,
            index_name=rag_index_name,
            namespace=rag_namespace,
        )
        self.orchestrator = Orchestrator(
            llm=self.llm,
            rag_retriever=self.rag,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        # DSE mode at the user entry point. Orchestrator's own flags stay default
        # OFF (direct-construction contract); the pipeline opts the chosen path in.
        self.orchestrator.use_variation_dse, self.orchestrator.use_bilevel_dse = (
            self._dse_flags(dse_mode)
        )

    @staticmethod
    def _dse_flags(mode: str) -> tuple[bool, bool]:
        """Map a dse_mode string to (use_variation_dse, use_bilevel_dse).

        "variation" (default) — LLM-declared variation points + domain objective.
        "bilevel"             — catalog-operator bilevel DSE (MO-MCTS + inner BO).
        "off"                 — legacy scalar DSE (both flags off).
        """
        m = (mode or "").strip().lower()
        if m == "bilevel":
            return (False, True)
        if m == "off":
            return (False, False)
        return (True, False)  # default: variation

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

        ok_l1 = sum(1 for r in l1_results if r.passed)
        print(f"  参数验证: {ok_l1}/{len(l1_results)} 通过")
        for r in l1_results:
            icon = "✓" if r.passed else "✗"
            print(f"    {icon} {r.req_id:<20} {r.message}")

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
        )

        all_n = len(l1_results) + len(l2_results)
        all_ok = sum(1 for r in l1_results + l2_results if r.passed)
        print()
        print(f"  SITL 阶段完成  ({all_ok}/{all_n} 通过)")
        print(f"  .parm 文件: {parm_path}")
        print("=" * W)

        result["sitl_report"] = report
        result["sitl_parm_file"] = str(parm_path)
        result["sitl_l2_scripts"] = [str(s) for s in scripts]
        return result

    def explore_design_space(
        self,
        generate_result: Dict[str, Any],
        mcts_iterations: int = 50,
        mcts_seed: Optional[int] = None,
        mcts_patience: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """
        Run MCTS Design Space Exploration on a previously validated model.

        Takes the dict returned by generate_system() and explores the
        parameter space to find the optimal configuration.

        Returns
        -------
        Full result dict — superset of generate_result — with updated
        model/score/sim fields plus DSE fields:
        {design_space_summary, design_space_parameters,
         best_config, pareto_alternatives}
        """
        return self.orchestrator.explore(
            generate_result=generate_result,
            mcts_iterations=mcts_iterations,
            mcts_seed=mcts_seed,
            mcts_patience=mcts_patience,
        )

    def prototype_system(
        self,
        system_name: str,
        description: str,
        additional_requirements: Optional[List[str]] = None,
        mcts_iterations: int = 50,
        parse_strict: Optional[bool] = None,
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
        return self.orchestrator.prototype(
            system_name=system_name,
            system_description=description,
            additional_requirements=additional_requirements,
            mcts_iterations=mcts_iterations,
            parse_strict=(parse_strict if parse_strict is not None else self.parse_strict),
        )

    def quick_design(
        self,
        system_name: str,
        requirements: List[str],
        parse_strict: Optional[bool] = None,
    ) -> SysMLModel:
        """
        Quickly generate a SysML v2 model from a list of requirements.

        Skips design space exploration for faster results.

        Args:
            system_name: Name of the system
            requirements: List of requirement strings

        Returns:
            A SysMLModel representing the design
        """
        from ..agents.design_agent import DesignAgent
        from ..agents.requirements_agent import RequirementsAgent

        # Create DesignAgent without the old ast_client parameter
        design_agent = DesignAgent(self.llm, self.rag)
        req_agent = RequirementsAgent(self.llm, self.rag)

        task = {
            "system_name": system_name,
            "requirements": requirements,
            "parse_strict": (parse_strict if parse_strict is not None else self.parse_strict),
        }
        result = design_agent.run(task)

        if result.success and isinstance(result.output, _SysMLModelTypes):
            model = result.output
        else:
            model = build_lite_model("", model_name=system_name)

        req_agent.create_sysml_requirements(requirements, model)
        return model

    def explore_alternatives(
        self,
        system_name: str,
        description: str,
        num_alternatives: int = 3,
        requirements: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple alternative design candidates for comparison.

        Uses self-consistency CoT to produce diverse designs at varying
        temperatures, giving a spread of structural alternatives.

        Args:
            system_name:      Name of the system.
            description:      System description (used only when
                              ``requirements`` is not supplied).
            num_alternatives: Number of design alternatives to generate.
            requirements:     Pre-extracted requirement list.  When provided
                              (e.g. taken from a prior ``prototype_system``
                              result) the method skips the Phase-1 LLM call
                              that would otherwise re-extract them from
                              ``description``.

        Returns:
            List of dicts with keys: name, sysml, reasoning, thought_steps,
            requirements_source ("provided" | "extracted").
        """
        from ..agents.requirements_agent import RequirementsAgent
        from ..llm.chain_of_thought import ChainOfThoughtPrompter

        cot = ChainOfThoughtPrompter(self.llm)

        if requirements:
            # Reuse caller-supplied requirements — no extra LLM call needed.
            reqs = requirements
            req_source = "provided"
        else:
            # Fall back to extracting from description when no list is given.
            req_agent = RequirementsAgent(self.llm, self.rag)
            req_result = req_agent.run({"system_description": description})
            reqs = req_result.output if req_result.success else []
            req_source = "extracted"

        alternatives: List[Dict[str, Any]] = []
        for i in range(num_alternatives):
            cot_result = cot.generate_design(
                system_name=f"{system_name}_v{i + 1}",
                requirements=reqs,
                temperature=0.6 + i * 0.1,
            )
            alternatives.append({
                "name": f"{system_name}_v{i + 1}",
                "sysml": cot_result.extracted_sysml or "",
                "reasoning": cot_result.final_answer,
                "thought_steps": len(cot_result.thought_steps),
                "requirements_source": req_source,
            })

        return alternatives


