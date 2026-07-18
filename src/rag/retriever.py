"""RAG retriever module backed by Pinecone search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

DEFAULT_ALLOWED_SYSML_EXTENSIONS: Tuple[str, ...] = (".sysml",)
DEFAULT_TOTAL_TOKEN_BUDGET = 4096
DEFAULT_CHARS_PER_TOKEN = 4.0


@dataclass
class RetrievedContext:
    """Context retrieved from Pinecone for augmenting LLM prompts."""
    entries: List[Tuple[Dict[str, Any], float]]
    query: str

    def format_for_prompt(self, max_entries: int = 3) -> str:
        """Format retrieved entries as context for an LLM prompt.

        Token-efficiency rules applied here:
        - Relevance scores are omitted (the LLM gains nothing from them).
        - Absolute ``Source:`` paths are omitted (machine-local, meaningless
          to the model and expensive in tokens).
        - Auto-generated vector IDs (e.g. ``uc3m#…#row-000021`` or full
          ``/Users/…`` paths) are suppressed; only human-readable titles are
          shown.
        """
        top_entries = self.entries[:max_entries]
        if not top_entries:
            return "No relevant context found."

        def _clean_title(raw: str) -> str:
            """Return a human-readable title or empty string to suppress it."""
            if not raw or raw == "Untitled":
                return ""
            # Auto-generated Pinecone / CSV row IDs contain '#'
            if "#" in raw:
                return ""
            # Absolute file paths — show only the stem filename
            if raw.startswith("/") or (len(raw) > 2 and raw[1] == ":"):
                from pathlib import Path
                stem = Path(raw).name
                # Still looks like an ID? suppress it.
                return stem if stem and "#" not in stem else ""
            return raw

        lines = ["Relevant MBSE knowledge:"]
        for i, (entry, _score) in enumerate(top_entries, 1):
            raw_title = str(entry.get("title") or entry.get("id") or "")
            title = _clean_title(raw_title)
            content = str(entry.get("content") or "")

            header = f"\n[{i}]" + (f" {title}" if title else "")
            lines.append(header)
            if content:
                lines.append(content)

            # Official SysML reference snippet (no path — content only)
            official_snippet = str(entry.get("official_sysml_reference") or "")
            if official_snippet:
                lines.append("Official SysML v2 reference snippet:")
                lines.append("```sysml")
                lines.append(official_snippet)
                lines.append("```")

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
        official_release_root: Optional[str] = None,
    ):
        self.llm = llm
        self.index_name = index_name
        self.namespace = namespace
        self.pinecone = pinecone_wrapper or PineconeWrapper(
            default_namespace=namespace or "SysML-V2-Release"
        )
        # Corpus lives outside the package tree: <repo>/data/SysML-v2-release-src
        default_release_root = (
            Path(__file__).resolve().parents[2] / "data" / "SysML-v2-release-src"
        )
        self.official_release_root = (
            Path(official_release_root).expanduser().resolve()
            if official_release_root
            else default_release_root.resolve()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        include_official_sysml: bool = False,
        total_token_budget: int = DEFAULT_TOTAL_TOKEN_BUDGET,
        allowed_extensions: Optional[Sequence[str]] = None,
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
        entries = self._parse_search_results(raw_results)[:top_k]
        if include_official_sysml:
            self._attach_official_sysml_references(
                entries=entries,
                query=query,
                total_token_budget=total_token_budget,
                allowed_extensions=allowed_extensions,
            )
        return RetrievedContext(entries=entries, query=query)

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
            "file_path": self._coalesce(
                fields.get("file_path"),
                metadata.get("file_path"),
                item.get("file_path"),
                default="",
            ),
            "source_csv": self._coalesce(
                fields.get("source_csv"),
                metadata.get("source_csv"),
                item.get("source_csv"),
                default="",
            ),
            "split": self._coalesce(
                fields.get("split"),
                metadata.get("split"),
                item.get("split"),
                default="",
            ),
            "metadata": metadata,
        }
        score = self._to_float(item.get("_score", item.get("score", 0.0)))
        return entry, score

    def _attach_official_sysml_references(
        self,
        entries: List[Tuple[Dict[str, Any], float]],
        query: str,
        total_token_budget: int,
        allowed_extensions: Optional[Sequence[str]] = None,
    ) -> None:
        if not entries or total_token_budget <= 0:
            return

        allowed_exts = {
            ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
            for ext in (allowed_extensions or DEFAULT_ALLOWED_SYSML_EXTENSIONS)
            if str(ext).strip()
        }
        if not allowed_exts:
            return

        budgets = self._allocate_budget_first_hit_plus_score(entries, total_token_budget)
        for (entry, _), entry_budget in zip(entries, budgets):
            file_path = str(entry.get("file_path") or "")
            resolved = self._resolve_official_file_path(file_path, allowed_exts)
            if resolved is None:
                continue

            source_text = self._read_text_file(resolved)
            if not source_text:
                continue

            snippet = self._build_sysml_snippet_full_or_trimmed(
                source_text=source_text,
                budget_tokens=entry_budget,
                query=query,
            )
            if snippet:
                entry["official_source_path"] = str(resolved)
                entry["official_sysml_reference"] = snippet

    def _allocate_budget_first_hit_plus_score(
        self,
        entries: List[Tuple[Dict[str, Any], float]],
        total_token_budget: int,
    ) -> List[int]:
        if not entries:
            return []
        if total_token_budget <= 0:
            return [0 for _ in entries]

        budgets = [0 for _ in entries]
        first_floor = min(max(180, total_token_budget // 4), total_token_budget)
        budgets[0] = first_floor
        remaining = total_token_budget - first_floor
        if remaining <= 0:
            return budgets

        weights = [max(score, 0.0) for _, score in entries]
        total_weight = sum(weights)
        if total_weight <= 0:
            per_entry = remaining // len(entries)
            for i in range(len(entries)):
                budgets[i] += per_entry
            budgets[0] += remaining - per_entry * len(entries)
            return budgets

        distributed = 0
        for i, weight in enumerate(weights):
            alloc = int(remaining * (weight / total_weight))
            budgets[i] += alloc
            distributed += alloc

        remainder = remaining - distributed
        if remainder > 0:
            rank_order = sorted(range(len(entries)), key=lambda idx: weights[idx], reverse=True)
            for i in range(remainder):
                budgets[rank_order[i % len(rank_order)]] += 1
        return budgets

    def _resolve_official_file_path(
        self,
        raw_file_path: str,
        allowed_extensions: set[str],
    ) -> Optional[Path]:
        if not raw_file_path:
            return None

        marker = "SysML-v2-release-src/"
        if marker not in raw_file_path:
            return None

        relative = raw_file_path.split(marker, 1)[1].replace("\\", "/").lstrip("/")
        if not relative:
            return None

        candidate = (self.official_release_root / relative).resolve()
        try:
            candidate.relative_to(self.official_release_root)
        except ValueError:
            return None

        if candidate.suffix.lower() not in allowed_extensions:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    @staticmethod
    def _read_text_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def _build_sysml_snippet_full_or_trimmed(
        self,
        source_text: str,
        budget_tokens: int,
        query: str,
    ) -> str:
        if budget_tokens <= 0:
            return ""
        if self._estimate_tokens_chars(source_text) <= budget_tokens:
            return source_text.strip()

        lines = source_text.splitlines()
        if not lines:
            return ""

        head_count = min(40, len(lines))
        head_lines = lines[:head_count]

        token_terms = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+", query)
            if len(token) >= 4
        ]

        relevant_indices: List[int] = []
        if token_terms:
            for idx, line in enumerate(lines):
                lowered = line.lower()
                if any(term in lowered for term in token_terms):
                    relevant_indices.extend(range(max(0, idx - 3), min(len(lines), idx + 4)))

        trimmed_body: List[str] = []
        seen: set[int] = set()
        for idx in relevant_indices:
            if idx in seen:
                continue
            seen.add(idx)
            trimmed_body.append(lines[idx])

        if not trimmed_body:
            start = head_count
            end = min(len(lines), start + 50)
            trimmed_body = lines[start:end]

        if trimmed_body:
            combined = "\n".join(head_lines + ["...", *trimmed_body]).strip()
        else:
            combined = "\n".join(head_lines).strip()
        return self._clip_to_token_budget(combined, budget_tokens)

    def _clip_to_token_budget(self, text: str, budget_tokens: int) -> str:
        if budget_tokens <= 0:
            return ""
        max_chars = int(budget_tokens * DEFAULT_CHARS_PER_TOKEN)
        if len(text) <= max_chars:
            return text.strip()
        clipped = text[:max_chars].rstrip()
        # Preserve clean line breaks while making truncation explicit.
        return clipped.rsplit("\n", 1)[0].rstrip() + "\n..."

    @staticmethod
    def _estimate_tokens_chars(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / DEFAULT_CHARS_PER_TOKEN))

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
        context = self.retrieve(
            query,
            top_k=top_k,
            category_filter=category_filter,
            include_official_sysml=False,
        )
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
