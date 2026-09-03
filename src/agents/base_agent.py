"""Base agent class for the MBSE multi-agent system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever


@dataclass
class AgentResult:
    """Result produced by an agent."""
    agent_name: str
    success: bool
    output: Any
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all MBSE agents."""

    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        self.name = name
        self.llm = llm
        self.rag = rag_retriever
        self._results_history: List[AgentResult] = []

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Execute the agent's primary task."""

    def get_augmented_context(
        self,
        query: str,
        include_official_sysml: bool = False,
        total_token_budget: int = 4096,
        allowed_extensions: Optional[Sequence[str]] = None,
    ) -> str:
        """Retrieve relevant MBSE knowledge to augment the agent's reasoning."""
        if self.rag is None:
            return ""
        context = self.rag.retrieve(
            query,
            top_k=3,
            include_official_sysml=include_official_sysml,
            total_token_budget=total_token_budget,
            allowed_extensions=allowed_extensions,
        )
        return context.format_for_prompt()

    def record_result(self, result: AgentResult) -> None:
        """Record a result produced by this agent."""
        self._results_history.append(result)
