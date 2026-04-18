"""Tests for the multi-agent framework."""

import pytest
from typing import cast

from src.agents.base_agent import AgentMessage, AgentResult
from src.agents.requirements_agent import RequirementsAgent
from src.agents.design_agent import DesignAgent
from src.agents.orchestrator import Orchestrator
from src.llm.interface import MockLLM
from src.rag.retriever import RAGRetriever, RetrievedContext
from src.sysml.model import SysMLModel


class FakePineconeWrapper:
    def search(
        self,
        index_name,
        query_text,
        top_k=3,
        namespace=None,
        filter_dict=None,
        fields=None,
    ):
        return {
            "result": {
                "hits": [
                    {
                        "_id": "test-hit",
                        "_score": 0.9,
                        "fields": {
                            "title": "Test Context",
                            "chunk_text": "This is test retrieval context.",
                            "category": "sysml_pattern",
                            "tags": ["test"],
                        },
                    }
                ]
            }
        }


class TestAgentMessage:
    def test_creation(self):
        msg = AgentMessage(
            sender="AgentA",
            recipient="AgentB",
            message_type="design_request",
            content={"system": "drone"},
        )
        assert msg.sender == "AgentA"
        assert msg.message_type == "design_request"


class TestAgentResult:
    def test_creation(self):
        result = AgentResult(
            agent_name="TestAgent",
            success=True,
            output="some output",
        )
        assert result.success
        assert result.output == "some output"


class TestRequirementsAgent:
    @pytest.fixture
    def agent(self):
        llm = MockLLM()
        return RequirementsAgent(llm)

    def test_run_with_description(self, agent):
        task = {
            "system_description": "An autonomous drone that delivers packages",
        }
        result = agent.run(task)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "RequirementsAgent"

    def test_run_empty_description_fails(self, agent):
        result = agent.run({"system_description": ""})
        assert not result.success

    def test_run_extracts_requirements(self, agent):
        # Inject a response with requirements
        cast(MockLLM, agent.llm).inject_response(
            "Here are the requirements:\n"
            "REQ-FUNC-001: The system shall fly autonomously\n"
            "REQ-PERF-001: The system shall maintain stability within 0.1 degrees\n"
            "REQ-SAFE-001: The system shall auto-land on low battery\n"
        )
        result = agent.run({"system_description": "Autonomous drone system"})
        assert result.success
        requirements = result.output
        assert isinstance(requirements, list)

    def test_validate_requirements(self, agent):
        requirements = [
            "REQ-001: The system shall respond within 100ms",
            "REQ-002: The system shall be user-friendly",
        ]
        validation = agent.validate_requirements(requirements)
        assert "valid" in validation
        assert "issues" in validation
        assert "warnings" in validation
        assert validation["total_requirements"] == 2

    def test_validate_missing_shall(self, agent):
        requirements = ["The system must respond quickly"]
        validation = agent.validate_requirements(requirements)
        assert len(validation["issues"]) > 0

    def test_create_sysml_requirements(self, agent):
        model = SysMLModel(name="TestSystem")
        requirements = [
            "REQ-FUNC-001: The system shall operate at 100Hz",
            "REQ-SAFE-001: The system shall fail safely",
        ]
        agent.create_sysml_requirements(requirements, model)
        assert len(model.requirements) == 2

    def test_last_result_tracking(self, agent):
        agent.run({"system_description": "A test system"})
        assert agent.last_result is not None
        assert agent.last_result.agent_name == "RequirementsAgent"

    def test_run_does_not_call_rag(self):
        class _FailIfCalledRAG:
            def retrieve(self, *args, **kwargs):
                raise AssertionError("RequirementsAgent should not call RAG retrieve")

        agent = RequirementsAgent(MockLLM(), rag_retriever=_FailIfCalledRAG())
        result = agent.run({"system_description": "A simple monitoring system"})
        assert result.success


class TestDesignAgent:
    @pytest.fixture
    def agent(self):
        llm = MockLLM()
        return DesignAgent(llm)

    def test_run_generates_model(self, agent):
        task = {
            "system_name": "DroneSystem",
            "requirements": ["REQ-001: The drone shall fly autonomously"],
        }
        result = agent.run(task)
        assert isinstance(result, AgentResult)
        assert result.success
        assert isinstance(result.output, SysMLModel)

    def test_run_with_sysml_response(self, agent):
        cast(MockLLM, agent.llm).inject_response(
            "Here is the design:\n"
            "```sysml\n"
            "package DroneSystem {\n"
            "    part def FlightController {\n"
            "        port sensorIn : SensorPort;\n"
            "        port cmdOut : CommandPort;\n"
            "        attribute loopRate : Real = 100.0;\n"
            "    }\n"
            "    part def IMU {\n"
            "        port dataOut : SensorPort;\n"
            "    }\n"
            "}\n"
            "```\n"
        )
        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": ["REQ-001: The drone shall fly"],
        })
        assert result.success
        model = result.output
        assert len(model.blocks) > 0
        block_names = [b.name for b in model.blocks]
        assert "FlightController" in block_names

    def test_run_no_system_name_uses_default(self, agent):
        task = {"requirements": ["REQ-001: system shall work"]}
        result = agent.run(task)
        assert result.success

    def test_refinement_mode(self, agent):
        existing_model = SysMLModel(name="TestSystem")
        task = {
            "system_name": "TestSystem",
            "requirements": ["REQ-001: system shall work"],
            "existing_model": existing_model,
            "refinement_feedback": "Missing sensor components",
        }
        result = agent.run(task)
        assert isinstance(result, AgentResult)
        assert isinstance(result.output, SysMLModel)
        assert any(block.satisfies for block in result.output.blocks)

    def test_run_requests_official_sysml_context(self):
        class _CapturingRAG:
            def __init__(self):
                self.last_kwargs = None

            def retrieve(self, query, top_k=3, **kwargs):
                self.last_kwargs = kwargs
                return RetrievedContext(entries=[], query=query)

        rag = _CapturingRAG()
        agent = DesignAgent(MockLLM(), rag_retriever=rag)
        result = agent.run({
            "system_name": "CaptureTest",
            "requirements": ["REQ-001: The system shall operate safely"],
        })
        assert result.success
        assert rag.last_kwargs is not None
        assert rag.last_kwargs.get("include_official_sysml") is True


class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        llm = MockLLM()
        rag = RAGRetriever(
            llm=llm,
            pinecone_wrapper=FakePineconeWrapper(),
            index_name="test-index",
            namespace="test-ns",
        )
        return Orchestrator(
            llm=llm,
            rag_retriever=rag,
            quality_threshold=0.5,  # Low threshold for testing
            max_iterations=2,
        )

    def test_prototype_returns_result(self, orchestrator):
        result = orchestrator.prototype(
            system_name="TestDrone",
            system_description="A drone that delivers packages autonomously",
            mcts_iterations=5,  # Minimal for testing
        )
        assert isinstance(result, dict)
        assert "model" in result
        assert "requirements" in result
        assert "model_sysml" in result
        assert "final_score" in result

    def test_prototype_model_is_sysml(self, orchestrator):
        result = orchestrator.prototype(
            system_name="SimpleSystem",
            system_description="A simple control system",
            mcts_iterations=5,
        )
        model = result["model"]
        assert isinstance(model, SysMLModel)

    def test_prototype_sysml_text_is_valid(self, orchestrator):
        result = orchestrator.prototype(
            system_name="ControlSystem",
            system_description="An automated control system",
            mcts_iterations=5,
        )
        sysml_text = result["model_sysml"]
        assert "package ControlSystem" in sysml_text

    def test_additional_requirements_included(self, orchestrator):
        additional = [
            "REQ-MANUAL-001: The system shall operate in extreme temperatures",
        ]
        result = orchestrator.prototype(
            system_name="RobustSystem",
            system_description="A robust control system",
            additional_requirements=additional,
            mcts_iterations=5,
        )
        assert len(result["requirements"]) >= len(additional)

    def test_design_space_explored(self, orchestrator):
        result = orchestrator.prototype(
            system_name="ExploredSystem",
            system_description="A system to explore",
            mcts_iterations=10,
        )
        summary = result["design_space_summary"]
        assert summary["configurations_evaluated"] > 0

    def test_prototyping_state_updated(self, orchestrator):
        orchestrator.prototype(
            system_name="StateTest",
            system_description="A test system for state tracking",
            mcts_iterations=5,
        )
        assert orchestrator.state is not None
        assert orchestrator.state.system_name == "StateTest"
        assert orchestrator.state.iteration > 0
