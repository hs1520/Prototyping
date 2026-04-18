"""Tests for Pinecone-backed RAG retriever behavior."""

from pathlib import Path

import pytest
from src.llm.interface import MockLLM
from src.rag.retriever import RAGRetriever, RetrievedContext


class FakePineconeWrapper:
    def __init__(self):
        self.last_filter = None
        self.last_query = None

    def search(
        self,
        index_name,
        query_text,
        top_k=3,
        namespace=None,
        filter_dict=None,
        fields=None,
    ):
        self.last_query = query_text
        self.last_filter = filter_dict

        if "xyzzy_nonexistent_term_12345" in query_text:
            return {"result": {"hits": []}}

        hits = [
            {
                "_id": "sysml_port_001",
                "_score": 0.93,
                "fields": {
                    "title": "SysML v2 Port Definitions",
                    "chunk_text": "Ports define connection points between parts.",
                    "category": "sysml_pattern",
                    "tags": ["port", "interface", "connection"],
                    "file_path": "../SysML-v2-release-src/training/ports/PortsExample.sysml",
                },
            },
            {
                "_id": "cps_example_001",
                "_score": 0.75,
                "fields": {
                    "title": "Drone System Architecture",
                    "chunk_text": "Example architecture for autonomous drone systems.",
                    "category": "example",
                    "tags": ["drone", "uav"],
                    "file_path": "../SysML-v2-release-src/training/examples/DroneExample.sysml",
                },
            },
            {
                "_id": "design_003",
                "_score": 0.62,
                "fields": {
                    "title": "Redundancy and Fault Tolerance",
                    "chunk_text": "Redundancy patterns improve reliability.",
                    "category": "design_principle",
                    "tags": ["safety", "fault-tolerance", "redundancy"],
                    "file_path": "../SysML-v2-release-src/training/principles/Reliability.sysml",
                },
            },
        ]

        if isinstance(filter_dict, dict):
            category = None
            tags = None
            if "$and" in filter_dict:
                for clause in filter_dict["$and"]:
                    if "category" in clause:
                        category = clause["category"]["$eq"]
                    if "tags" in clause:
                        tags = clause["tags"]["$in"]
            else:
                if "category" in filter_dict:
                    category = filter_dict["category"]["$eq"]
                if "tags" in filter_dict:
                    tags = filter_dict["tags"]["$in"]

            if category:
                hits = [h for h in hits if h["fields"].get("category") == category]
            if tags:
                hits = [
                    h for h in hits
                    if any(tag in h["fields"].get("tags", []) for tag in tags)
                ]

        return {"result": {"hits": hits[:top_k]}}


class TestRAGRetriever:
    @pytest.fixture
    def retriever(self):
        llm = MockLLM()
        fake_pinecone = FakePineconeWrapper()
        return RAGRetriever(
            llm=llm,
            pinecone_wrapper=fake_pinecone,
            index_name="test-index",
            namespace="test-ns",
        )

    def test_retrieve_returns_context(self, retriever):
        context = retriever.retrieve("SysML v2 port definition")
        assert isinstance(context, RetrievedContext)
        assert len(context.entries) > 0
        assert isinstance(context.entries[0][0], dict)
        assert isinstance(context.entries[0][1], float)

    def test_format_for_prompt(self, retriever):
        context = retriever.retrieve("sensor controller design")
        formatted = context.format_for_prompt()
        assert "Relevant MBSE knowledge" in formatted
        assert len(formatted) > 50

    def test_generate_with_context(self, retriever):
        response = retriever.generate_with_context("How do I define a SysML v2 port?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_retrieve_design_patterns(self, retriever):
        context = retriever.retrieve_design_patterns("cyber-physical control system")
        for entry, _ in context.entries:
            assert entry["category"] == "sysml_pattern"

    def test_retrieve_examples(self, retriever):
        context = retriever.retrieve_examples("drone system")
        for entry, _ in context.entries:
            assert entry["category"] == "example"

    def test_retrieve_principles(self, retriever):
        context = retriever.retrieve_principles("fault tolerance redundancy")
        for entry, _ in context.entries:
            assert entry["category"] == "design_principle"

    def test_no_results_formats_gracefully(self, retriever):
        context = retriever.retrieve("xyzzy_nonexistent_term_12345")
        formatted = context.format_for_prompt()
        assert isinstance(formatted, str)
        assert "No relevant context found" in formatted

    def test_retrieve_includes_official_sysml_reference_when_enabled(self, tmp_path: Path):
        release_root = tmp_path / "SysML-v2-release-src"
        source_file = release_root / "training" / "ports" / "PortsExample.sysml"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(
            "package P {\n"
            "    part def Controller {\n"
            "        port sensorIn : SensorPort;\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )

        retriever = RAGRetriever(
            llm=MockLLM(),
            pinecone_wrapper=FakePineconeWrapper(),
            index_name="test-index",
            namespace="test-ns",
            official_release_root=str(release_root),
        )

        context = retriever.retrieve(
            "SysML port controller",
            include_official_sysml=True,
            total_token_budget=300,
        )
        first_entry = context.entries[0][0]
        assert "official_sysml_reference" in first_entry
        assert "package P" in first_entry["official_sysml_reference"]
        formatted = context.format_for_prompt()
        assert "Official SysML v2 reference snippet" in formatted

    def test_retrieve_ignores_disallowed_extension(self, tmp_path: Path):
        release_root = tmp_path / "SysML-v2-release-src"
        text_file = release_root / "training" / "ports" / "PortsExample.txt"
        text_file.parent.mkdir(parents=True, exist_ok=True)
        text_file.write_text("not sysml", encoding="utf-8")

        retriever = RAGRetriever(
            llm=MockLLM(),
            pinecone_wrapper=FakePineconeWrapper(),
            index_name="test-index",
            namespace="test-ns",
            official_release_root=str(release_root),
        )

        context = retriever.retrieve(
            "SysML port controller",
            include_official_sysml=True,
            allowed_extensions=(".txt",),
        )
        first_entry = context.entries[0][0]
        assert "official_sysml_reference" not in first_entry


class TestRetrievedContext:
    def test_empty_context(self):
        context = RetrievedContext(entries=[], query="test")
        formatted = context.format_for_prompt()
        assert "No relevant context found" in formatted

    def test_max_entries_limit(self):
        entries = [
            ({"title": "A", "content": "alpha"}, 0.9),
            ({"title": "B", "content": "beta"}, 0.8),
            ({"title": "C", "content": "gamma"}, 0.7),
        ]
        context = RetrievedContext(entries=entries, query="design")
        formatted = context.format_for_prompt(max_entries=2)
        assert "[3]" not in formatted
