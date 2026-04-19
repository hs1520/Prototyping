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
        quality_threshold: float = 0.70,
        max_iterations: int = 3,
    ):
        self.llm = llm
        self.pinecone = pinecone_wrapper or PineconeWrapper(default_namespace=rag_namespace)
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
        )

    def prototype_system(
        self,
        system_name: str,
        description: str,
        additional_requirements: Optional[List[str]] = None,
        mcts_iterations: int = 50,
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
        )

    def quick_design(
        self,
        system_name: str,
        requirements: List[str],
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

        design_agent = DesignAgent(self.llm, self.rag)
        req_agent = RequirementsAgent(self.llm, self.rag)

        result = design_agent.run({
            "system_name": system_name,
            "requirements": requirements,
        })

        if result.success and isinstance(result.output, SysMLModel):
            model = result.output
        else:
            model = SysMLModel(name=system_name)

        req_agent.create_sysml_requirements(requirements, model)
        return model

    def explore_alternatives(
        self,
        system_name: str,
        description: str,
        num_alternatives: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple alternative design candidates for comparison.

        Uses self-consistency CoT to produce diverse designs.

        Args:
            system_name: Name of the system
            description: System description
            num_alternatives: Number of design alternatives to generate

        Returns:
            List of design dictionaries with model and metadata
        """
        from ..agents.requirements_agent import RequirementsAgent
        from ..llm.chain_of_thought import ChainOfThoughtPrompter

        req_agent = RequirementsAgent(self.llm, self.rag)
        cot = ChainOfThoughtPrompter(self.llm)

        # Extract requirements
        req_result = req_agent.run({"system_description": description})
        requirements = req_result.output if req_result.success else []

        # Generate alternatives using self-consistency
        alternatives: List[Dict[str, Any]] = []
        for i in range(num_alternatives):
            cot_result = cot.generate_design(
                system_name=f"{system_name}_v{i+1}",
                requirements=requirements,
                temperature=0.6 + i * 0.1,  # Slightly different temperature each time
            )
            alternatives.append({
                "name": f"{system_name}_v{i+1}",
                "sysml": cot_result.extracted_sysml or "",
                "reasoning": cot_result.final_answer,
                "thought_steps": len(cot_result.thought_steps),
            })

        return alternatives


