"""Tests for the RAG knowledge base and retriever."""

import pytest
from src.rag.knowledge_base import KnowledgeBase, KnowledgeEntry
from src.rag.retriever import RAGRetriever, RetrievedContext
from src.llm.interface import MockLLM


class TestKnowledgeBase:
    @pytest.fixture
    def kb(self):
        return KnowledgeBase()

    def test_builtin_knowledge_loaded(self, kb):
        assert len(kb.entries) > 0

    def test_has_sysml_patterns(self, kb):
        patterns = kb.get_by_category("sysml_pattern")
        assert len(patterns) > 0

    def test_has_design_principles(self, kb):
        principles = kb.get_by_category("design_principle")
        assert len(principles) > 0

    def test_has_examples(self, kb):
        examples = kb.get_by_category("example")
        assert len(examples) > 0

    def test_search_returns_results(self, kb):
        results = kb.search("SysML port connection interface")
        assert len(results) > 0
        assert all(isinstance(entry, KnowledgeEntry) for entry, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_search_top_k(self, kb):
        results = kb.search("design architecture", top_k=3)
        assert len(results) <= 3

    def test_search_with_category_filter(self, kb):
        results = kb.search("part definition block", category_filter="sysml_pattern")
        for entry, _ in results:
            assert entry.category == "sysml_pattern"

    def test_search_relevance_ordering(self, kb):
        results = kb.search("SysML port")
        scores = [score for _, score in results]
        # Scores should be in descending order
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_add_custom_entry(self, kb):
        initial_count = len(kb.entries)
        custom = KnowledgeEntry(
            id="custom_001",
            title="Custom Pattern",
            content="A custom MBSE pattern",
            category="custom",
            tags=["test"],
        )
        kb.add_entry(custom)
        assert len(kb.entries) == initial_count + 1

    def test_search_finds_custom_entry(self, kb):
        custom = KnowledgeEntry(
            id="unique_pattern_xyz",
            title="Unique XYZ Pattern",
            content="This is a very unique xyz design pattern for testing",
            category="custom",
            tags=["xyz", "unique"],
        )
        kb.add_entry(custom)
        results = kb.search("unique xyz design pattern", top_k=5)
        result_ids = [entry.id for entry, _ in results]
        assert "unique_pattern_xyz" in result_ids

    def test_get_by_tags(self, kb):
        entries = kb.get_by_tags(["port", "connection"])
        assert len(entries) > 0

    def test_search_drone_domain(self, kb):
        results = kb.search("autonomous drone UAV flight", top_k=3)
        assert len(results) > 0
        # Should find drone-related example
        found_drone = any("drone" in entry.tags for entry, _ in results)
        assert found_drone


class TestRAGRetriever:
    @pytest.fixture
    def retriever(self):
        llm = MockLLM()
        return RAGRetriever(llm)

    def test_retrieve_returns_context(self, retriever):
        context = retriever.retrieve("SysML v2 port definition")
        assert isinstance(context, RetrievedContext)
        assert len(context.entries) > 0

    def test_format_for_prompt(self, retriever):
        context = retriever.retrieve("sensor controller design")
        formatted = context.format_for_prompt()
        assert "Relevant MBSE knowledge" in formatted
        assert len(formatted) > 50

    def test_generate_with_context(self, retriever):
        response = retriever.generate_with_context(
            "How do I define a SysML v2 port?"
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_retrieve_design_patterns(self, retriever):
        context = retriever.retrieve_design_patterns(
            "cyber-physical control system"
        )
        for entry, _ in context.entries:
            assert entry.category == "sysml_pattern"

    def test_retrieve_examples(self, retriever):
        context = retriever.retrieve_examples("drone system")
        for entry, _ in context.entries:
            assert entry.category == "example"

    def test_retrieve_principles(self, retriever):
        context = retriever.retrieve_principles("fault tolerance redundancy")
        for entry, _ in context.entries:
            assert entry.category == "design_principle"

    def test_no_results_formats_gracefully(self, retriever):
        context = retriever.retrieve("xyzzy_nonexistent_term_12345")
        formatted = context.format_for_prompt()
        assert isinstance(formatted, str)


class TestRetrievedContext:
    def test_empty_context(self):
        context = RetrievedContext(entries=[], query="test")
        formatted = context.format_for_prompt()
        assert "No relevant context found" in formatted

    def test_max_entries_limit(self):
        kb = KnowledgeBase()
        entries = kb.search("design", top_k=5)
        context = RetrievedContext(entries=entries, query="design")
        formatted = context.format_for_prompt(max_entries=2)
        # Should contain content from at most 2 entries
        # Count occurrences of the entry marker "[1]", "[2]", "[3]"
        assert "[3]" not in formatted
