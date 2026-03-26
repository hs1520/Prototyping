"""
Base agent class for the MBSE multi-agent system.

Provides common functionality for all specialized agents in the
rapid prototyping framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever


@dataclass
class AgentMessage:
    """A message passed between agents."""
    sender: str
    recipient: str
    message_type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result produced by an agent."""
    agent_name: str
    success: bool
    output: Any
    reasoning: str = ""
    messages_sent: List[AgentMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all MBSE agents.

    Each agent has a specific role in the design process and can
    communicate with other agents through the orchestrator.
    """

    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        self.name = name
        self.llm = llm
        self.rag = rag_retriever
        self._message_history: List[AgentMessage] = []
        self._results_history: List[AgentResult] = []

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Execute the agent's primary task."""

    def get_augmented_context(self, query: str) -> str:
        """Retrieve relevant MBSE knowledge to augment the agent's reasoning."""
        if self.rag is None:
            return ""
        context = self.rag.retrieve(query, top_k=3)
        return context.format_for_prompt()

    def send_message(
        self,
        recipient: str,
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Create and record a message to another agent."""
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        self._message_history.append(msg)
        return msg

    def record_result(self, result: AgentResult) -> None:
        """Record a result produced by this agent."""
        self._results_history.append(result)

    @property
    def last_result(self) -> Optional[AgentResult]:
        """Return the most recent result produced by this agent."""
        return self._results_history[-1] if self._results_history else None
