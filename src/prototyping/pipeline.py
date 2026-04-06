"""
Main prototyping pipeline module.

Provides the high-level API for the AI-assisted MBSE rapid prototyping
framework, combining all components into a unified workflow.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from ..agents.orchestrator import Orchestrator
from ..llm.interface import LLMInterface, MockLLM, OpenAILLM, GeminiLLM
from ..rag.knowledge_base import KnowledgeBase
from ..rag.retriever import RAGRetriever
from ..sysml.model import SysMLModel


LLM_PROVIDER_FACTORIES: Dict[str, Any] = {
    "mock": MockLLM,
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
}

LLM_PROVIDER_ALIASES: Dict[str, str] = {
    "default": "mock",
    "test": "mock",
    "open_ai": "openai",
    "gpt": "openai",
    "google": "gemini",
}


def _normalize_provider_name(provider: Optional[str]) -> str:
    """Normalize provider names and common aliases to a canonical key."""
    normalized = (provider or "mock").strip().lower().replace("-", "_")
    return LLM_PROVIDER_ALIASES.get(normalized, normalized)


def _build_constructor_kwargs(
    factory: Any,
    model: Optional[str],
    api_key: Optional[str],
    provider_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Only pass kwargs accepted by a provider constructor."""
    kwargs = dict(provider_kwargs or {})
    signature = inspect.signature(factory.__init__)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )

    if model is not None and ("model" in parameters or accepts_var_kwargs):
        kwargs.setdefault("model", model)
    if api_key is not None and ("api_key" in parameters or accepts_var_kwargs):
        kwargs.setdefault("api_key", api_key)

    return kwargs


def register_llm_provider(name: str, provider_factory: Any) -> None:
    """Register a custom LLM provider for create_llm()."""
    LLM_PROVIDER_FACTORIES[_normalize_provider_name(name)] = provider_factory


def available_llm_providers() -> List[str]:
    """Return all currently registered canonical provider names."""
    return sorted(LLM_PROVIDER_FACTORIES.keys())


def create_llm(
    provider: Optional[str] = None,
    use_openai: bool = False,
    use_llm: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
) -> LLMInterface:
    """
    Create an LLM instance.

    Args:
        provider: Provider name, e.g. "mock", "gemini", "openai"
        use_openai: Legacy flag for selecting OpenAI when provider is not set
        use_llm: Legacy flag for selecting Gemini when provider is not set
        model: Model name for providers that support it
        api_key: API key for providers that support it
        provider_kwargs: Extra provider-specific constructor args

    Returns:
        An LLM interface instance
    """
    if provider is None:
        if use_openai:
            provider = "openai"
        elif use_llm:
            provider = "gemini"
        else:
            provider = "mock"

    provider_name = _normalize_provider_name(provider)
    factory = LLM_PROVIDER_FACTORIES.get(provider_name)
    if factory is None:
        supported = ", ".join(available_llm_providers())
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Supported providers: {supported}"
        )

    if model is None:
        if provider_name == "openai":
            model = "gpt-4o"
        elif provider_name == "gemini":
            model = "gemini-3-flash-preview"

    kwargs = _build_constructor_kwargs(factory, model, api_key, provider_kwargs)
    # noinspection PyArgumentList
    return factory(**kwargs)


class PrototypingPipeline:
    """
    High-level API for AI-assisted MBSE rapid prototyping.

    This is the main entry point for using the framework. It combines:
    - Chain of Thought prompting for design reasoning
    - Retrieval Augmented Generation for domain knowledge
    - Multi-agent coordination for specialized tasks
    - Monte Carlo Tree Search for design space exploration

    Example usage:
        pipeline = PrototypingPipeline()
        result = pipeline.prototype_system(
            system_name="AutonomousDrone",
            description="A drone that autonomously delivers packages...",
        )
        print(result["model_sysml"])
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_options: Optional[Dict[str, Any]] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        quality_threshold: float = 0.70,
        max_iterations: int = 3,
    ):
        self.llm = llm or create_llm(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            provider_kwargs=llm_options,
        )
        self.kb = knowledge_base or KnowledgeBase()
        self.rag = RAGRetriever(self.llm, self.kb)
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
