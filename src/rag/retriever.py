"""
RAG retriever module.

Combines knowledge base retrieval with LLM prompting to provide
context-aware responses for MBSE design tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_base import KnowledgeBase, KnowledgeEntry
from ..llm.interface import LLMInterface, Message


@dataclass
class RetrievedContext:
    """Context retrieved from the knowledge base for augmenting LLM prompts."""
    entries: List[Tuple[KnowledgeEntry, float]]
    query: str

    def format_for_prompt(self, max_entries: int = 3) -> str:
        """Format retrieved entries as context for an LLM prompt."""
        top_entries = self.entries[:max_entries]
        if not top_entries:
            return "No relevant context found."

        lines = ["Relevant MBSE knowledge:"]
        for i, (entry, score) in enumerate(top_entries, 1):
            lines.append(f"\n[{i}] {entry.title} (relevance: {score:.2f})")
            lines.append(entry.content)
        return "\n".join(lines)


class RAGRetriever:
    """
    Retrieval Augmented Generation for MBSE design assistance.

    Combines knowledge base retrieval with LLM generation to provide
    informed responses that leverage domain-specific MBSE knowledge.
    """

    def __init__(self, llm: LLMInterface, knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.kb = knowledge_base or KnowledgeBase()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> RetrievedContext:
        """Retrieve relevant knowledge entries for a query."""
        results = self.kb.search(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
            tag_filter=tag_filter,
        )
        return RetrievedContext(entries=results, query=query)

    def generate_with_context(
        self,
        query: str,
        system_prompt: str = "",
        top_k: int = 3,
        category_filter: Optional[str] = None,
        temperature: float = 0.5,
    ) -> str:
        """
        Generate an LLM response augmented with retrieved knowledge.
        """
        context = self.retrieve(query, top_k=top_k, category_filter=category_filter)
        context_text = context.format_for_prompt(max_entries=top_k)

        augmented_prompt = f"{context_text}\n\n---\n\nTask: {query}"

        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=augmented_prompt))

        response = self.llm.complete(messages, temperature=temperature)
        return response.content

    def retrieve_design_patterns(
        self,
        system_description: str,
        top_k: int = 3,
    ) -> RetrievedContext:
        """Retrieve relevant SysML v2 design patterns for a system description."""
        return self.retrieve(
            query=system_description,
            top_k=top_k,
            category_filter="sysml_pattern",
        )

    def retrieve_examples(
        self,
        domain: str,
        top_k: int = 2,
    ) -> RetrievedContext:
        """Retrieve relevant design examples for a domain."""
        return self.retrieve(
            query=domain,
            top_k=top_k,
            category_filter="example",
        )

    def retrieve_principles(
        self,
        concern: str,
        top_k: int = 3,
    ) -> RetrievedContext:
        """Retrieve relevant design principles for a specific concern."""
        return self.retrieve(
            query=concern,
            top_k=top_k,
            category_filter="design_principle",
        )
