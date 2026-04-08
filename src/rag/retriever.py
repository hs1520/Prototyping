"""RAG retriever module backed by Pinecone search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .pinecone_wrapper import PineconeWrapper
from ..llm.interface import LLMInterface, Message


DEFAULT_SEARCH_FIELDS: List[str] = [
    "chunk_text",
    "title",
    "category",
    "tags",
    "has_description",
    "file_path",
    "source_csv",
    "split",
]


@dataclass
class RetrievedContext:
    """Context retrieved from Pinecone for augmenting LLM prompts."""
    entries: List[Tuple[Dict[str, Any], float]]
    query: str

    def format_for_prompt(self, max_entries: int = 3) -> str:
        """Format retrieved entries as context for an LLM prompt."""
        top_entries = self.entries[:max_entries]
        if not top_entries:
            return "No relevant context found."

        lines = ["Relevant MBSE knowledge:"]
        for i, (entry, score) in enumerate(top_entries, 1):
            title = str(entry.get("title") or entry.get("id") or "Untitled")
            content = str(entry.get("content") or "")
            lines.append(f"\n[{i}] {title} (relevance: {score:.2f})")
            lines.append(content)
        return "\n".join(lines)


class RAGRetriever:
    """
    Retrieval Augmented Generation for MBSE design assistance.

    Uses Pinecone retrieval with optional metadata filtering, then augments
    LLM prompts with the retrieved context.
    """

    def __init__(
        self,
        llm: LLMInterface,
        pinecone_wrapper: Optional[Any] = None,
        index_name: str = "ai-prototyping-sysml-v2",
        namespace: Optional[str] = None,
    ):
        self.llm = llm
        self.index_name = index_name
        self.namespace = namespace
        self.pinecone = pinecone_wrapper or PineconeWrapper(
            default_namespace=namespace or "SysML-V2-Release"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> RetrievedContext:
        """Retrieve relevant Pinecone records for a query."""
        raw_results = self.pinecone.search(
            index_name=self.index_name,
            query_text=query,
            top_k=top_k,
            namespace=self.namespace,
            filter_dict=self._build_filter(category_filter, tag_filter),
            fields=DEFAULT_SEARCH_FIELDS,
        )
        return RetrievedContext(entries=self._parse_search_results(raw_results)[:top_k], query=query)

    @staticmethod
    def _build_filter(
        category_filter: Optional[str],
        tag_filter: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        clauses = [
            *([{"category": {"$eq": category_filter}}] if category_filter else []),
            *([{"tags": {"$in": tag_filter}}] if tag_filter else []),
        ]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses} if clauses else None

    def _parse_search_results(
        self,
        raw_results: Any,
    ) -> List[Tuple[Dict[str, Any], float]]:
        parsed = [self._parse_hit(hit) for hit in self._extract_hits(raw_results)]
        parsed.sort(key=lambda x: x[1], reverse=True)
        return parsed

    def _parse_hit(self, hit: Any) -> Tuple[Dict[str, Any], float]:
        item = self._to_dict(hit)
        fields = self._to_dict(item.get("fields"))
        metadata = self._to_dict(item.get("metadata"))

        entry = {
            "id": self._coalesce(item.get("_id"), item.get("id"), metadata.get("id"), default=""),
            "title": self._coalesce(
                fields.get("title"),
                metadata.get("title"),
                item.get("id"),
                item.get("_id"),
                default="Untitled",
            ),
            "content": str(
                self._coalesce(
                    fields.get("chunk_text"),
                    fields.get("text"),
                    fields.get("content"),
                    item.get("chunk_text"),
                    item.get("text"),
                    item.get("content"),
                    default="",
                )
            ),
            "category": self._coalesce(fields.get("category"), metadata.get("category"), default=""),
            "tags": self._normalize_tags(fields.get("tags", metadata.get("tags", []))),
            "metadata": metadata,
        }
        score = self._to_float(item.get("_score", item.get("score", 0.0)))
        return entry, score

    @staticmethod
    def _extract_hits(raw_results: Any) -> List[Any]:
        payload = RAGRetriever._to_dict(raw_results)
        result_block = RAGRetriever._to_dict(payload.get("result"))
        for container in (payload, result_block):
            for key in ("hits", "matches"):
                value = container.get(key)
                if isinstance(value, list):
                    return value
        return []

    @staticmethod
    def _to_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, dict):
                return converted
        if hasattr(item, "__dict__"):
            return {str(k): v for k, v in item.__dict__.items()}
        return {}

    @staticmethod
    def _coalesce(*values: Any, default: str = "") -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return str(value)
        return default

    @staticmethod
    def _normalize_tags(tags_value: Any) -> List[str]:
        if isinstance(tags_value, str):
            return [t.strip() for t in tags_value.split(",") if t.strip()]
        if isinstance(tags_value, list):
            return [str(t) for t in tags_value]
        return []

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

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
