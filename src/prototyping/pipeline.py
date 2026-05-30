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

    def generate_system(
        self,
        system_name: str,
        description: str,
        additional_requirements: Optional[List[str]] = None,
        parse_strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Generate a validated SysML v2 model without Design Space Exploration.

        Run Phases 1-4 only (requirements → design → syntax gate → refinement
        → simulation).  No MCTS, no parameter injection.  Use this to get a
        correct, working model first; call explore_design_space() afterwards
        when DSE is needed.

        Returns
        -------
        {system_name, requirements, model, model_sysml, model_summary,
         final_score, iterations, evaluation_history, simulation_result}
        """
        return self.orchestrator.generate(
            system_name=system_name,
            system_description=description,
            additional_requirements=additional_requirements,
            parse_strict=(parse_strict if parse_strict is not None else self.parse_strict),
        )

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


