"""Tests for the multi-step DesignAgent generation pipeline (Phase 2-a)."""

from __future__ import annotations

import sys
import json
from typing import List
from collections import deque

import pytest

# Stub out heavy optional dependencies that are not installed in the test env
from types import ModuleType

def _stub(name: str) -> ModuleType:
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod

if "dotenv" not in sys.modules:
    dotenv_stub = _stub("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]

if "pinecone" not in sys.modules:
    pinecone_stub = _stub("pinecone")
    pinecone_stub.Pinecone = type("Pinecone", (), {"__init__": lambda self, **kw: None})  # type: ignore[attr-defined]

if "syside" not in sys.modules:
    try:
        __import__("syside")
    except ImportError:
        _stub("syside")

from src.llm.interface import LLMResponse, Message
from src.llm.chain_of_thought import ChainOfThoughtPrompter, CoTResult
from src.agents.typed_plan_generation import (
    TypedModelPlanError,
    TypedPlanGeneration,
    TypedPlanRequest,
)
from src.agents.assembly_finalization import (
    AssemblyFinalizer,
    AssemblyRequest,
)
from src.agents.model_authoring import AuthoringRequest, ModelAuthoring
from src.agents.generated_model_admission import (
    GeneratedModelAdmission,
    ModelAdmissionRequest,
)
from src.agents.refinement_authoring import (
    RefinementAuthoring,
    RefinementRequest,
)
from src.sysml.lite_model import build_lite_model
from src.sysml.text_normalization import (
    fix_capability_semantics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYSML_FRAGMENT = (
    "```sysml\n"
    "part def FlightController {\n"
    "    port commandIn : in;\n"
    "    attribute maxAltitude : Real = 400.0 [m];\n"
    "}\n"
    "part def PayloadManager {\n"
    "    port releaseOut : out;\n"
    "    attribute maxPayload : Real = 2.5 [kg];\n"
    "}\n"
    "```\n"
)

_BEHAVIOR_FRAGMENT = (
    "```sysml\n"
    "action def navigateToWaypoint { }\n"
    "state def SafetyMonitor {\n"
    "    state nominal;\n"
    "    state fault { entry; action def emergencyLand { } }\n"
    "    transition nominal -> fault when sensorFailureDetected;\n"
    "}\n"
    "```\n"
)

_INTERFACE_FRAGMENT = (
    "```sysml\n"
    "item def TelemetryData { attribute rate : Real; }\n"
    "port def TelemetryPort { out item signal : TelemetryData; }\n"
    "```\n"
)

_ASSEMBLED_MODEL = (
    "```sysml\n"
    "package DroneSystem {\n"
    "    part def FlightController {\n"
    "        port commandIn : in;\n"
    "        attribute maxAltitude : Real = 400.0 [m];\n"
    "        action def navigateToWaypoint { }\n"
    "        state def SafetyMonitor {\n"
    "            state nominal;\n"
    "            state fault { entry; action def emergencyLand { } }\n"
    "            transition nominal -> fault when sensorFailureDetected;\n"
    "        }\n"
    "    }\n"
    "    part def PayloadManager {\n"
    "        port releaseOut : out;\n"
    "        attribute maxPayload : Real = 2.5 [kg];\n"
    "    }\n"
    "    connect FlightController::commandIn to PayloadManager::releaseOut;\n"
    "    satisfy REQ_FUNC_001 by FlightController;\n"
    "    satisfy REQ_SAFE_001 by FlightController;\n"
    "    satisfy REQ_PERF_001 by FlightController;\n"
    "    satisfy REQ_INTF_001 by PayloadManager;\n"
    "}\n"
    "```\n"
)


def _finalize_assembly(
    assembled: str,
    *,
    parts: str = "",
    interfaces: str = "",
    behavior: str = "",
    generation_plan=None,
):
    return AssemblyFinalizer().finalize(AssemblyRequest(
        response=CoTResult(final_answer="", extracted_sysml=assembled),
        parts_fragment=parts,
        interfaces_fragment=interfaces,
        behavior_fragment=behavior,
        generation_plan=generation_plan,
    ))


def _admit_model(text: str, requirements):
    return GeneratedModelAdmission(build_lite_model, None).accept(
        ModelAdmissionRequest(
            response=CoTResult(final_answer=text, extracted_sysml=text),
            system_name="D",
            requirements=requirements,
            generation_metadata={},
            is_refinement=True,
        )
    )


class QueuedMockLLM:
    """Mock LLM that returns responses from a queue in order."""

    def __init__(self, responses: List[str]):
        self._queue: deque = deque(responses)
        self.call_count = 0
        self.messages: List[List[Message]] = []
        self.temperatures: List[float] = []

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 20480,
        **kwargs,
    ) -> LLMResponse:
        # **kwargs absorbs the multi-turn transcript bookkeeping
        # (conversation_id / new_message_offset), which does not change what a
        # provider is asked to produce.
        self.call_count += 1
        self.messages.append(list(messages))
        self.temperatures.append(temperature)
        content = self._queue.popleft() if self._queue else "fallback response"
        return LLMResponse(content=content, model="mock-queued")


# ---------------------------------------------------------------------------
# ChainOfThoughtPrompter multi-step method tests
# ---------------------------------------------------------------------------

class TestDecomposeArchitecture:
    def test_returns_cot_result(self):
        llm = QueuedMockLLM(["1. FlightController — controls flight\n   Addresses: REQ-FUNC-001"])
        cot = ChainOfThoughtPrompter(llm)
        result = cot.decompose_architecture(
            system_name="DroneSystem",
            requirements=["REQ-FUNC-001: The drone shall navigate autonomously."],
        )
        assert isinstance(result, CoTResult)
        assert "FlightController" in result.final_answer
        assert llm.call_count == 1
        assert "typed model-planning compiler" in (
            llm.messages[0][0].content
        )
        assert "expert in Model Based Systems Engineering" not in (
            llm.messages[0][0].content
        )

    def test_typed_plan_uses_stable_temperature_and_full_output_budget(self):
        captured = {}

        class CaptureLLM:
            ARCHITECTURE_MAX_TOKENS = 65536

            def complete(self, _messages, **kwargs):
                captured.update(kwargs)
                return LLMResponse(content="{}", model="capture")

        ChainOfThoughtPrompter(CaptureLLM()).decompose_architecture(
            system_name="DroneSystem",
            requirements=["REQ-FUNC-001: The drone shall navigate."],
        )

        assert captured["temperature"] == 0.2
        assert captured["max_tokens"] == 65536

    def test_planning_prompt_commits_state_identity_and_local_power_on(self):
        llm = QueuedMockLLM(["{}"])
        cot = ChainOfThoughtPrompter(llm)

        cot.decompose_architecture(
            system_name="DeliveryUAV",
            requirements=[
                "REQ-SAFE-008: The payload lock shall default locked "
                "upon power-on."
            ],
        )

        prompt = llm.messages[0][-1].content
        assert "BehaviorId::StateId" in prompt
        assert "FlightControllerBehavior::AvoidingObstacle" in prompt
        assert "power-on initialization" in prompt
        assert "LOCAL_BEHAVIOR" in prompt

    def test_json_parser_accepts_uppercase_fence_crlf_and_spaces(self):
        cot = ChainOfThoughtPrompter(QueuedMockLLM([]))

        result = cot._parse_cot_response(
            "``` JSON  \r\n{\"components\": [], \"connections\": []}\r\n```"
        )

        assert result.extracted_json == {
            "components": [],
            "connections": [],
        }
        assert result.metadata["json_parse"]["status"] == "PASS"
        assert result.metadata["json_parse"]["source"] == "FENCED_JSON"

    def test_json_parser_accepts_only_a_complete_raw_json_value(self):
        cot = ChainOfThoughtPrompter(QueuedMockLLM([]))

        result = cot._parse_cot_response(
            '{"components": [], "connections": []}'
        )

        assert result.extracted_json is not None
        assert result.metadata["json_parse"]["status"] == "PASS"
        assert result.metadata["json_parse"]["source"] == "RAW_JSON"

    def test_json_parser_reports_decode_error_and_non_object_root(self):
        cot = ChainOfThoughtPrompter(QueuedMockLLM([]))

        malformed = cot._parse_cot_response(
            "```json\n{\"components\": [}\n```"
        )
        non_object = cot._parse_cot_response("[1, 2, 3]")

        assert malformed.extracted_json is None
        assert malformed.metadata["json_parse"]["status"] == (
            "JSON_DECODE_FAILED"
        )
        assert malformed.metadata["json_parse"]["error_line"] == 1
        assert non_object.metadata["json_parse"]["status"] == (
            "JSON_ROOT_NOT_OBJECT"
        )

    def test_json_parser_separates_truncation_from_a_missing_block(self):
        """A response cut off mid-plan must not look like one that had no JSON.

        HIGH thinking shares the output budget with the answer, so an
        over-long plan comes back as an opened-but-never-closed ```json
        fence.  Reporting that as JSON_BLOCK_ABSENT hid a length failure
        behind a format failure.
        """
        cot = ChainOfThoughtPrompter(QueuedMockLLM([]))

        truncated = cot._parse_cot_response(
            '```json\n{"components": [{"name": "FlightCont'
        )
        prose = cot._parse_cot_response(
            "I propose Producer and Consumer components."
        )

        assert truncated.extracted_json is None
        assert truncated.metadata["json_parse"]["status"] == (
            "JSON_FENCE_UNCLOSED"
        )
        assert prose.metadata["json_parse"]["status"] == "JSON_BLOCK_ABSENT"

    def test_context_block_injected_when_provided(self):
        captured = []

        class CaptureLLM:
            def complete(self, messages, **_):
                captured.extend(messages)
                return LLMResponse(content="plan", model="cap")

        cot = ChainOfThoughtPrompter(CaptureLLM())
        cot.decompose_architecture(
            system_name="Sys",
            requirements=["REQ-FUNC-001: The sys shall do X with value 1."],
            context="Domain reference material",
        )
        prompt_text = " ".join(m.content for m in captured)
        assert "Domain reference material" in prompt_text

    def test_no_context_block_when_empty(self):
        captured = []

        class CaptureLLM:
            def complete(self, messages, **_):
                captured.extend(messages)
                return LLMResponse(content="plan", model="cap")

        cot = ChainOfThoughtPrompter(CaptureLLM())
        cot.decompose_architecture(
            system_name="Sys",
            requirements=["REQ-FUNC-001: The sys shall do X with value 1."],
            context="",
        )
        prompt_text = " ".join(m.content for m in captured)
        assert "Relevant domain context" not in prompt_text


class TestGeneratePartDefinitions:
    def test_extracts_sysml_fragment(self):
        llm = QueuedMockLLM([_SYSML_FRAGMENT])
        cot = ChainOfThoughtPrompter(llm)
        result = cot.generate_part_definitions(
            system_name="DroneSystem",
            architecture="1. FlightController\n2. PayloadManager",
            requirements=["REQ-PERF-001: The drone shall maintain altitude ≤ 400 m."],
        )
        assert result.extracted_sysml is not None
        assert "FlightController" in result.extracted_sysml
        assert "maxAltitude" in result.extracted_sysml


class TestGenerateBehavior:
    def test_extracts_behavioral_fragment(self):
        llm = QueuedMockLLM([_BEHAVIOR_FRAGMENT])
        cot = ChainOfThoughtPrompter(llm)
        result = cot.generate_behavior(
            system_name="DroneSystem",
            architecture="1. FlightController",
            behavioral_requirements=["REQ-FUNC-001: The drone shall navigate autonomously."],
            parts_fragment="part def FlightController { port commandIn : in; }",
        )
        assert result.extracted_sysml is not None
        assert "navigateToWaypoint" in result.extracted_sysml
        assert "SafetyMonitor" in result.extracted_sysml

    def test_behavior_prompt_contains_guard_rules(self):
        """The behavior prompt must carry guard-authoring rules so the LLM does
        not emit dead guards (`== <number>`, `== false`, threshold-vs-self)."""
        captured = []

        class CaptureLLM:
            def complete(self, messages, **_):
                captured.extend(messages)
                return LLMResponse(content=_BEHAVIOR_FRAGMENT, model="cap")

        cot = ChainOfThoughtPrompter(CaptureLLM())
        cot.generate_behavior(
            system_name="DroneSystem",
            architecture="1. SafetyMonitor",
            behavioral_requirements=["REQ-SAFE-001: emergency-land when battery low."],
            parts_fragment="part def SafetyMonitor { }",
        )
        prompt = " ".join(m.content for m in captured)

        # Section header present
        assert "GUARD CONDITION RULES" in prompt
        # Forbids == / != for numeric guards
        assert "NEVER use `==` or `!=`" in prompt
        # Left operand must be a dynamic variable; warns against threshold-vs-self
        assert "DYNAMIC measured" in prompt
        assert "comparing a threshold to itself" in prompt
        # Boolean flag guidance (affirmative naming, no == false)
        assert "affirmative flag" in prompt
        # Concrete correct example + wrong anti-patterns
        assert "if batteryCharge < 15.0" in prompt
        assert "if someFlag == false" in prompt
        # Encourages named dynamic thresholds and preserves unit consistency.
        assert "PREFER a dynamic threshold" in prompt
        assert "if batteryCharge <= returnEnergyRequired" in prompt
        assert "UNIT CONSISTENCY RULE" in prompt
        assert "minimumSeparation + 5.0" in prompt
        assert "avoidanceActivationDistance : Real = 10 [m]" in prompt
        assert "inside the exact state" in prompt
        assert "state def FlightControllerBehavior" in prompt
        assert "assert constraint enforceMinSeparation" in prompt
        assert "MUST NOT be serialized as an always-true assert" not in prompt


class TestAssembleModel:
    def test_extracts_complete_package(self):
        llm = QueuedMockLLM([_ASSEMBLED_MODEL])
        cot = ChainOfThoughtPrompter(llm)
        result = cot.assemble_model(
            system_name="DroneSystem",
            parts_fragment="part def FlightController { ... }",
            interfaces_fragment="",
            behavior_fragment="action def navigateToWaypoint { }",
            requirements=[
                "REQ-FUNC-001: The drone shall navigate autonomously.",
                "REQ-SAFE-001: The drone shall emergency-land when sensor failure is detected.",
            ],
        )
        assert result.extracted_sysml is not None
        assert "package DroneSystem" in result.extracted_sysml
        assert "satisfy REQ_FUNC_001" in result.extracted_sysml
        assert result.metadata["assembly_retries"] == 0
        assert llm.call_count == 1
        assert "deterministic SysML v2 assembly compiler" in (
            llm.messages[0][0].content
        )
        assert "Always think step-by-step" not in llm.messages[0][0].content

    def test_incomplete_code_fence_gets_one_compact_retry(self):
        llm = QueuedMockLLM([
            "```sysml\npackage DroneSystem {",
            _ASSEMBLED_MODEL,
        ])
        cot = ChainOfThoughtPrompter(llm)

        result = cot.assemble_model(
            system_name="DroneSystem",
            parts_fragment="part def FlightController {}",
            interfaces_fragment="",
            behavior_fragment="",
            requirements=["REQ-FUNC-001: The system shall navigate."],
        )

        assert result.extracted_sysml is not None
        assert result.metadata["assembly_retries"] == 1
        assert result.metadata["assembly_initial_extraction"] == (
            "UNCLOSED_SYSML_FENCE"
        )
        assert llm.call_count == 2
        assert "ASSEMBLY OUTPUT CORRECTION" in llm.messages[1][1].content

    def test_incomplete_code_fence_remains_fail_closed_after_retry(self):
        llm = QueuedMockLLM([
            "```sysml\npackage DroneSystem {",
            "```sysml\npackage DroneSystem {",
        ])
        cot = ChainOfThoughtPrompter(llm)

        with pytest.raises(RuntimeError, match="after one targeted retry"):
            cot.assemble_model(
                system_name="DroneSystem",
                parts_fragment="part def FlightController {}",
                interfaces_fragment="",
                behavior_fragment="",
                requirements=["REQ-FUNC-001: The system shall navigate."],
            )

        assert llm.call_count == 2

    def test_package_name_strips_special_chars(self):
        captured_prompts = []

        class CaptureLLM:
            def complete(self, messages, **_):
                captured_prompts.extend(messages)
                return LLMResponse(content=_ASSEMBLED_MODEL, model="cap")

        cot = ChainOfThoughtPrompter(CaptureLLM())
        cot.assemble_model(
            system_name="My-Drone System 2",
            parts_fragment="",
            interfaces_fragment="",
            behavior_fragment="",
            requirements=["REQ-FUNC-001: The system shall fly with speed 10 m/s."],
        )
        prompt_text = " ".join(m.content for m in captured_prompts)
        # Special chars stripped → "MyDroneSystem2"
        assert "MyDroneSystem2" in prompt_text


# ---------------------------------------------------------------------------
# _multistep_generate integration test
# ---------------------------------------------------------------------------

class TestMultistepGeneratePipeline:
    """
    Test _multistep_generate via DesignAgent using stubs.
    The Syside parser is bypassed by monkeypatching build_lite_model.
    """

    _REQUIREMENTS = [
        "REQ-FUNC-001: The drone shall navigate to waypoints autonomously.",
        "REQ-PERF-001: The drone shall maintain altitude below 400 m.",
        "REQ-SAFE-001: The drone shall execute emergency landing when battery < 10% is detected.",
        "REQ-INTF-001: The drone shall exchange telemetry with ground station via MAVLink.",
    ]

    @staticmethod
    def _minimal_typed_payload(target_name: str = "Consumer"):
        return {
            "components": [
                {
                    "name": "Producer",
                    "responsibility": "Produces data.",
                    "requirements": [],
                    "ports": [{
                        "name": "data",
                        "direction": "out",
                        "type": "DataPort",
                    }],
                },
                {
                    "name": target_name,
                    "responsibility": "Consumes data.",
                    "requirements": [],
                    "ports": [{
                        "name": "data",
                        "direction": "in",
                        "type": "DataPort",
                    }],
                },
            ],
            "connections": [{
                "source": {
                    "component": "Producer",
                    "port": "data",
                },
                "target": {
                    "component": target_name,
                    "port": "data",
                },
                "item_type": "DataPort",
            }],
            "requirement_realizations": [],
            "semantic_bindings": [],
        }

    def _make_agent(self, responses: List[str], monkeypatch):
        """Build a DesignAgent with a queued mock LLM and a no-op parser."""
        from src.agents.design_agent import DesignAgent

        llm = QueuedMockLLM(responses)
        agent = DesignAgent(
            llm,
            allow_legacy_architecture_plan=True,
        )

        # Stub out RAG so it returns nothing
        monkeypatch.setattr(agent, "get_augmented_context", lambda *a, **kw: "")

        return agent

    def _generate_typed_plan(
        self,
        agent,
        system_name,
        requirements,
        *,
        context="",
        semantic_guidance="",
        behavior_plan=None,
        allow_legacy_plan=None,
    ):
        """Exercise the public Step 1 protocol without DesignAgent internals."""
        if allow_legacy_plan is None:
            allow_legacy_plan = agent.allow_legacy_architecture_plan
        return TypedPlanGeneration(
            agent.cot,
            maximum_attempts=agent.maximum_plan_attempts,
        ).generate(TypedPlanRequest(
            system_name=system_name,
            requirements=requirements,
            context=context,
            semantic_guidance=semantic_guidance,
            behavior_plan=behavior_plan,
            allow_legacy_plan=allow_legacy_plan,
        ))

    def _author_model(
        self,
        agent,
        system_name,
        requirements,
        architecture_text,
        generation_plan,
        *,
        semantic_guidance=None,
        behavior_plan=None,
        platform_profile=None,
    ):
        """Exercise the public Steps 2–5 authoring interface."""
        return ModelAuthoring(agent.cot, lambda _query: "").generate(
            AuthoringRequest(
                system_name=system_name,
                architecture_text=architecture_text,
                requirements=requirements,
                generation_plan=generation_plan,
                semantic_guidance=semantic_guidance,
                behavior_plan=behavior_plan,
                platform_profile=platform_profile,
            )
        )

    def test_five_llm_calls_made(self, monkeypatch):
        """Pipeline now has 5 steps: arch, parts, interfaces, behavior, assembly."""
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        responses = [
            "Architecture plan text with component list",  # step 1
            _SYSML_FRAGMENT,                               # step 2
            _INTERFACE_FRAGMENT,                           # step 3 (interfaces)
            _BEHAVIOR_FRAGMENT,                            # step 4 (behavior)
            _ASSEMBLED_MODEL,                              # step 5 (assembly)
        ]
        agent = self._make_agent(responses, monkeypatch)

        # Stub build_lite_model to return a minimal model
        dummy_model = SysMLModel(name="DroneSystem", description="test")
        dummy_model.part_definitions.append(
            PartDefinition(name="FlightController", short_description="controls flight")
        )
        monkeypatch.setattr(da_module, "build_lite_model", lambda *a, **kw: dummy_model)

        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": self._REQUIREMENTS,
        })

        assert result.success
        assert agent.llm.call_count == 5
        assert any(
            "explicit legacy" in item
            for item in result.metadata["degraded_steps"]
        )

    def test_generation_steps_are_independent_calls_not_one_conversation(
        self, monkeypatch
    ):
        """Each step is its own single-turn call. This was measured, not assumed.

        Sharing one conversation across the steps took paired-seed qualification
        from 3/3 to 0/3 at 2.1x the prompt cost, failing a different check each
        seed. Each step is already handed what it needs in curated form, so the
        history added a second uncurated copy plus stale earlier instructions.
        A future change that reintroduces a shared conversation here should have
        to argue with that measurement first.
        """
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        step1 = "Architecture plan text with component list"
        responses = [
            step1,
            _SYSML_FRAGMENT,
            _INTERFACE_FRAGMENT,
            _BEHAVIOR_FRAGMENT,
            _ASSEMBLED_MODEL,
        ]
        agent = self._make_agent(responses, monkeypatch)
        dummy_model = SysMLModel(name="DroneSystem", description="test")
        dummy_model.part_definitions.append(
            PartDefinition(name="FlightController", short_description="x")
        )
        monkeypatch.setattr(
            da_module, "build_lite_model", lambda *a, **kw: dummy_model
        )

        agent.run({
            "system_name": "DroneSystem",
            "requirements": self._REQUIREMENTS,
        })

        planning, *authoring = agent.llm.messages
        # every call is [system, user] — no assistant turn is ever resent
        assert all(len(turns) == 2 for turns in agent.llm.messages)
        assert not any(
            message.role == "assistant"
            for turns in agent.llm.messages for message in turns
        )
        # step 1 keeps its own compiler system prompt
        assert "typed model-planning compiler" in planning[0].content
        assert "expert in Model Based Systems Engineering" not in (
            planning[0].content
        )
        # design authoring keeps the expert role; mechanical assembly gets a
        # compact compiler role that does not demand displayed reasoning.
        *design_authoring, assembly = authoring
        assert len({turns[0].content for turns in design_authoring}) == 1
        assert "SysML v2" in design_authoring[0][0].content
        assert "typed model-planning compiler" not in (
            design_authoring[0][0].content
        )
        assert "deterministic SysML v2 assembly compiler" in (
            assembly[0].content
        )
        assert "Always think step-by-step" not in assembly[0].content
        # what each step needs reaches it through the curated prompt, not history
        assert "FlightController" in authoring[1][1].content

    def test_refinement_interface_owns_role_and_source_selection(
        self, monkeypatch
    ):
        from src.sysml.model import SysMLModel

        agent = self._make_agent([_ASSEMBLED_MODEL], monkeypatch)
        existing = SysMLModel(name="D", description="fallback")
        existing.metadata["last_sysml_text"] = "package D { part def A {} }"

        outcome = RefinementAuthoring(
            agent.cot,
            lambda query: f"retrieved for {query}",
        ).refine(RefinementRequest(
            existing_model=existing,
            feedback="Fix the undirected port",
            issues=("port direction is undirected",),
        ))

        assert outcome.response.extracted_sysml
        assert outcome.metadata["source"] == "original LLM text"
        assert "port def direction" in outcome.metadata["rag_query"]
        assert "FIX specific reported issues" in agent.llm.messages[0][0].content
        assert "package D { part def A {} }" in agent.llm.messages[0][1].content

    def test_authoring_interface_owns_role_prompt(self, monkeypatch):
        from src.prototyping.generation_plan import ModelGenerationPlan

        plan = ModelGenerationPlan.from_payload(
            self._minimal_typed_payload()
        )
        agent = self._make_agent([
            "```sysml\npart def Producer {}\npart def Consumer {}\n```",
            _INTERFACE_FRAGMENT,
            _ASSEMBLED_MODEL,
        ], monkeypatch)
        self._author_model(
            agent,
            "P",
            [],
            plan.render_for_prompt(),
            plan,
        )

        authoring_turns = agent.llm.messages[:2]
        assert all(
            "expert MBSE architect generating SysML v2" in turns[0].content
            for turns in authoring_turns
        )

    def test_invalid_typed_plan_gets_one_bounded_retry(self, monkeypatch):
        invalid = {
            "components": [{
                "name": "Controller",
                "responsibility": "Controls the system.",
                "requirements": [
                    "REQ_FUNC_001", "REQ_PERF_001",
                    "REQ_SAFE_001", "REQ_INTF_001",
                ],
                "ports": [{
                    "name": "status",
                    "direction": "out",
                    "type": "StatusPort",
                    "external": False,
                }],
            }],
            "connections": [],
        }
        valid = {
            "components": [
                {
                    "name": "Controller",
                    "responsibility": (
                        "Navigates the drone and detects critical battery."
                    ),
                    "requirements": ["REQ_FUNC_001", "REQ_PERF_001"],
                    "ports": [{
                        "name": "status",
                        "direction": "out",
                        "type": "StatusPort",
                        "external": False,
                    }],
                },
                {
                    "name": "SafetyMonitor",
                    "responsibility": (
                        "Observes waypoint navigation, executes emergency "
                        "landing, and exchanges telemetry with the ground "
                        "station via MAVLink."
                    ),
                    "requirements": ["REQ_SAFE_001", "REQ_INTF_001"],
                    "ports": [{
                        "name": "status",
                        "direction": "in",
                        "type": "StatusPort",
                        "external": False,
                    }],
                },
            ],
            "connections": [{
                "source": {"component": "Controller", "port": "status"},
                "target": {"component": "SafetyMonitor", "port": "status"},
                "item_type": "StatusPort",
                "requirements": [
                    "REQ_FUNC_001",
                    "REQ_SAFE_001",
                    "REQ_INTF_001",
                ],
            }],
            "requirement_realizations": [
                {
                    "requirement_id": "REQ_FUNC_001",
                    "realization_kind": "CAUSAL_PATH",
                    "trigger_concept": "The drone",
                    "effect_concept": "navigate to waypoints autonomously",
                    "connection_path": [{
                        "source_component": "Controller",
                        "source_port": "status",
                        "target_component": "SafetyMonitor",
                        "target_port": "status",
                        "item_type": "StatusPort",
                    }],
                },
                {
                    "requirement_id": "REQ_SAFE_001",
                    "realization_kind": "CAUSAL_PATH",
                    "trigger_concept": "battery < 10% is detected",
                    "effect_concept": "execute emergency landing",
                    "connection_path": [{
                        "source_component": "Controller",
                        "source_port": "status",
                        "target_component": "SafetyMonitor",
                        "target_port": "status",
                        "item_type": "StatusPort",
                    }],
                },
                {
                    "requirement_id": "REQ_INTF_001",
                    "realization_kind": "CAUSAL_PATH",
                    "trigger_concept": "The drone",
                    "effect_concept": (
                        "exchange telemetry with ground station via MAVLink"
                    ),
                    "connection_path": [{
                        "source_component": "Controller",
                        "source_port": "status",
                        "target_component": "SafetyMonitor",
                        "target_port": "status",
                        "item_type": "StatusPort",
                    }],
                },
            ],
            # REQ-FUNC-001 asks the drone to navigate, so the plan owes the
            # navigate response the closure gate will look for.
            "behaviors": [{
                "owner": "Controller",
                "behavior_id": "WaypointNavigationBehavior",
                "initial_state": "Idle",
                "states": [
                    {"state_id": "Idle", "role": "INITIAL"},
                    {
                        "state_id": "NavigatingToWaypoint",
                        "role": "RESPONSE",
                        "entry_action": "navigateToWaypoint",
                    },
                ],
                "transitions": [{
                    "transition_id": "startNavigation",
                    "source": "Idle",
                    "target": "NavigatingToWaypoint",
                    "trigger_kind": "ACCEPT",
                    "trigger": "WaypointMissionAcceptedSignal",
                }],
                "provenance": {
                    "kind": "FROZEN_REQUIREMENT",
                    "requirement_id": "REQ_FUNC_001",
                },
            }],
        }
        responses = [
            f"```json\n{json.dumps(invalid)}\n```",
            f"```json\n{json.dumps(valid)}\n```",
        ]
        agent = self._make_agent(responses, monkeypatch)
        outcome = self._generate_typed_plan(
            agent,
            "DroneSystem",
            self._REQUIREMENTS,
        )
        metadata = outcome.metadata
        rendered = outcome.architecture_text

        assert agent.llm.call_count == 2
        assert metadata["step1_plan_retries"] == 1
        assert metadata["whole_model_generation_plan"]["status"] == "PASS"
        assert "Controller.status -> SafetyMonitor.status" in rendered

    def test_missing_typed_json_gets_one_bounded_retry(self, monkeypatch):
        payload = self._minimal_typed_payload()
        agent = self._make_agent([
            "I propose Producer and Consumer components.",
            f"```JSON\r\n{json.dumps(payload)}\r\n```",
        ], monkeypatch)
        agent.allow_legacy_architecture_plan = False
        outcome = self._generate_typed_plan(agent, "P", [])
        metadata = outcome.metadata
        rendered = outcome.architecture_text

        assert agent.llm.call_count == 2
        assert metadata["step1_plan_retries"] == 1
        assert [
            item["json_parse"]["status"]
            for item in metadata["step1_plan_attempts"]
        ] == ["JSON_BLOCK_ABSENT", "PASS"]
        assert metadata["degraded_steps"] == []
        assert "Producer.data -> Consumer.data" in rendered
        assert "TYPED MODEL PLAN FORMAT CORRECTION" in (
            agent.llm.messages[1][-1].content
        )

    def test_format_and_semantic_failures_receive_separate_budgets(
        self, monkeypatch
    ):
        valid = self._minimal_typed_payload()
        invalid = json.loads(json.dumps(valid))
        invalid["connections"] = []
        agent = self._make_agent([
            "```sysml\npackage WrongFormat {}\n```",
            f"```json\n{json.dumps(invalid)}\n```",
            f"```json\n{json.dumps(valid)}\n```",
        ], monkeypatch)
        agent.allow_legacy_architecture_plan = False
        metadata = self._generate_typed_plan(agent, "P", []).metadata

        assert agent.llm.call_count == 3
        assert [
            item["failure_kind"]
            for item in metadata["step1_plan_attempts"]
        ] == [
            "FORMAT_UNAVAILABLE",
            "SEMANTIC_PLAN_INVALID",
            "NONE",
        ]
        assert metadata["step1_format_retries"] == 1
        assert metadata["step1_semantic_retries"] == 1
        assert metadata["step1_plan_retries"] == 2
        assert "TYPED MODEL PLAN SEMANTIC CORRECTION" in (
            agent.llm.messages[2][-1].content
        )

    def test_two_semantic_corrections_can_use_all_three_attempts(
        self, monkeypatch
    ):
        base = self._minimal_typed_payload()
        consumer = base["components"][1]
        consumer["attributes"] = [
            {
                "name": "measured",
                "value_type": "Real",
                "role": "LOCAL_STATE",
                "initial_value": "0",
                "provenance": "DESIGN_DECISION",
            },
            {
                "name": "minimum",
                "value_type": "Real",
                "role": "DESIGN_PARAMETER",
                "initial_value": "1",
                "provenance": "DESIGN_DECISION",
            },
        ]
        unqualified = json.loads(json.dumps(base))
        unqualified["constraints"] = [{
            "constraint_id": "minimumBound",
            "owner": "Consumer",
            "expression": {
                "lhs": "measured",
                "operator": ">=",
                "rhs": "minimum",
            },
            "activation": {
                "kind": "STATE_ACTIVE",
                "state": "Active",
            },
            "provenance": {"kind": "DESIGN_DECISION"},
            "verification_tier": "INSPECTION",
        }]
        wrong_tier = json.loads(json.dumps(unqualified))
        wrong_tier["constraints"][0]["activation"]["state"] = (
            "ConsumerBehavior::Active"
        )
        wrong_tier["constraints"][0]["verification_tier"] = (
            "PARAMETRIC_SWEEP"
        )
        wrong_tier["behaviors"] = [{
            "owner": "Consumer",
            "behavior_id": "ConsumerBehavior",
            "initial_state": "Idle",
            "states": [
                {"state_id": "Idle", "role": "INITIAL"},
                {
                    "state_id": "Active",
                    "role": "RESPONSE",
                    "entry_action": "activateConsumer",
                },
            ],
            "transitions": [{
                "transition_id": "activate",
                "source": "Idle",
                "target": "Active",
                "trigger_kind": "ACCEPT",
                "trigger": "ActivateSignal",
            }],
            "provenance": {"kind": "DESIGN_DECISION"},
        }]
        valid = json.loads(json.dumps(wrong_tier))
        # INSPECTION, not STATE_EXECUTION: the subject carries no input binding,
        # so the state executor cannot sweep it and the validator now requires
        # the plan to say so. The test still exercises two semantic corrections
        # across three attempts.
        valid["constraints"][0]["verification_tier"] = "INSPECTION"
        agent = self._make_agent([
            f"```json\n{json.dumps(unqualified)}\n```",
            f"```json\n{json.dumps(wrong_tier)}\n```",
            f"```json\n{json.dumps(valid)}\n```",
        ], monkeypatch)
        agent.allow_legacy_architecture_plan = False
        metadata = self._generate_typed_plan(agent, "P", []).metadata

        attempts = metadata["step1_plan_attempts"]
        assert agent.llm.call_count == 3
        assert metadata["step1_semantic_retries"] == 2
        assert [item["correction_outcome"] for item in attempts] == [
            "INITIAL_ATTEMPT",
            "REGRESSED_NEW_ISSUE",
            "RESOLVED",
        ]
        assert "PREVIOUS PARSEABLE PLAN" in (
            agent.llm.messages[1][-1].content
        )
        assert '"state": "Active"' in (
            agent.llm.messages[1][-1].content
        )
        assert '"state": "ConsumerBehavior::Active"' in (
            agent.llm.messages[2][-1].content
        )
        assert all(
            "typed model-planning compiler" in messages[0].content
            for messages in agent.llm.messages
        )
        assert metadata["whole_model_generation_plan"]["status"] == "PASS"

    def test_format_regression_keeps_last_parseable_repair_base(
        self, monkeypatch
    ):
        valid = self._minimal_typed_payload()
        invalid = json.loads(json.dumps(valid))
        invalid["connections"] = []
        agent = self._make_agent([
            f"```json\n{json.dumps(invalid)}\n```",
            "```sysml\npackage WrongKnowledgeType {}\n```",
            f"```json\n{json.dumps(valid)}\n```",
        ], monkeypatch)
        agent.allow_legacy_architecture_plan = False
        metadata = self._generate_typed_plan(agent, "P", []).metadata

        assert [
            item["failure_kind"]
            for item in metadata["step1_plan_attempts"]
        ] == [
            "SEMANTIC_PLAN_INVALID",
            "FORMAT_UNAVAILABLE",
            "NONE",
        ]
        third_prompt = agent.llm.messages[2][-1].content
        assert "PREVIOUS PARSEABLE PLAN" in third_prompt
        assert '"components"' in third_prompt
        assert '"connections": []' in third_prompt
        assert "Do not emit SysML" in third_prompt

    def test_truncated_plan_escalates_then_stops_replaying_one_request(
        self, monkeypatch
    ):
        """A truncated Step 1 must not burn the whole attempt budget.

        With no payload ever parsed the repair prompt is rebuilt from
        constants, so every attempt after the first was byte-identical and —
        against a fixed provider seed — could only reproduce the same
        truncation, at several minutes per call.
        """
        truncated = '```json\n{"components": [{"name": "FlightCont'
        agent = self._make_agent([truncated] * 6, monkeypatch)
        agent.allow_legacy_architecture_plan = False

        with pytest.raises(TypedModelPlanError) as captured:
            self._generate_typed_plan(agent, "P", [])

        prompts = [m[-1].content for m in agent.llm.messages]
        temperatures = agent.llm.temperatures
        # three genuinely different requests, then an honest stop instead of
        # three more replays of the third one
        assert agent.llm.call_count == 3
        assert agent.maximum_plan_attempts == 6
        assert temperatures == [0.2, 0.6, 1.0]
        assert len(set(zip(prompts, temperatures))) == 3
        assert captured.value.metadata["step1_plan_exhausted_reason"] == (
            "CORRECTION_CANNOT_VARY_REQUEST"
        )
        assert "CORRECTION_CANNOT_VARY_REQUEST" in str(captured.value)
        # the length failure must be named as such, and be repairable-looking
        assert all(
            item["json_parse"]["status"] == "JSON_FENCE_UNCLOSED"
            for item in captured.value.plan_attempts
        )
        assert "TYPED MODEL PLAN LENGTH CORRECTION" in prompts[1]

    def test_production_mode_never_silently_uses_legacy_plan(
        self, monkeypatch
    ):
        agent = self._make_agent([
            "1. Producer — produces data",
            "1. Producer — still produces data",
        ], monkeypatch)
        agent.allow_legacy_architecture_plan = False

        with pytest.raises(
            TypedModelPlanError,
            match="TYPED_MODEL_PLAN_UNAVAILABLE",
        ) as captured:
            self._generate_typed_plan(agent, "P", [])

        # The attempt budget is the ceiling, not a quota to spend: once the
        # correction can no longer vary the request, further attempts are
        # replays and the loop stops short of the bound.
        assert agent.llm.call_count <= agent.maximum_plan_attempts
        assert len(captured.value.plan_attempts) == agent.llm.call_count
        assert all(
            item["json_parse"]["status"] == "JSON_BLOCK_ABSENT"
            for item in captured.value.plan_attempts
        )
        assert not any(
            "legacy" in item.lower()
            for item in captured.value.metadata["degraded_steps"]
        )

    def test_explicit_legacy_compatibility_remains_available(
        self, monkeypatch
    ):
        agent = self._make_agent([
            "1. Producer — produces data\n"
            "   Addresses: REQ-FUNC-001\n"
            "   Ports needed: out data",
        ], monkeypatch)
        metadata = self._generate_typed_plan(
            agent,
            "P",
            ["REQ-FUNC-001: produce data"],
            allow_legacy_plan=True,
        ).metadata

        assert agent.llm.call_count == 1
        assert metadata["step1_plan_attempts"][0][
            "legacy_compatibility_used"
        ] is True
        assert any(
            "explicit legacy" in item
            for item in metadata["degraded_steps"]
        )

    def test_ag_owner_cross_validation_participates_in_step1_retry(
        self, monkeypatch
    ):
        from src.prototyping.ag_behavior_plan import (
            BehaviorObligation,
            BehaviorObligationPlan,
            INVARIANT,
        )

        requirement = (
            "REQ-SAFE-001: When a fault is detected, the system shall "
            "activate recovery."
        )

        def payload(target_name):
            return {
                "components": [
                    {
                        "name": "FaultDetector",
                        "responsibility": "Detects the fault.",
                        "requirements": [],
                        "ports": [{
                            "name": "faultStatus",
                            "direction": "out",
                            "type": "FaultStatusPort",
                        }],
                    },
                    {
                        "name": target_name,
                        "responsibility": "Activates recovery.",
                        "requirements": ["REQ_SAFE_001"],
                        "ports": [{
                            "name": "faultStatus",
                            "direction": "in",
                            "type": "FaultStatusPort",
                        }],
                    },
                ],
                "connections": [{
                    "source": {
                        "component": "FaultDetector",
                        "port": "faultStatus",
                    },
                    "target": {
                        "component": target_name,
                        "port": "faultStatus",
                    },
                    "item_type": "FaultStatusPort",
                    "requirements": ["REQ_SAFE_001"],
                }],
                "requirement_realizations": [{
                    "requirement_id": "REQ_SAFE_001",
                    "realization_kind": "CAUSAL_PATH",
                    "trigger_concept": "a fault is detected",
                    "effect_concept": "activate recovery",
                    "connection_path": [{
                        "source_component": "FaultDetector",
                        "source_port": "faultStatus",
                        "target_component": target_name,
                        "target_port": "faultStatus",
                        "item_type": "FaultStatusPort",
                    }],
                }],
                "semantic_bindings": [],
            }

        behavior_plan = BehaviorObligationPlan((
            BehaviorObligation(
                requirement_id="REQ_SAFE_001",
                contract_id="AG_REQ_SAFE_001",
                owner_def="RecoveryController",
                owner_usage="recoveryController",
                realization_kind=INVARIANT,
                stable_behavior_id="RecoveryActivation",
                assumptions=("faultDetected",),
                guarantees=("recoveryActive",),
                invariant_expression=(
                    "not faultDetected or recoveryActive"
                ),
            ),
        ))
        agent = self._make_agent([
            f"```json\n{json.dumps(payload('WrongController'))}\n```",
            f"```json\n{json.dumps(payload('RecoveryController'))}\n```",
        ], monkeypatch)
        agent.allow_legacy_architecture_plan = False
        metadata = self._generate_typed_plan(
            agent,
            "P",
            [requirement],
            behavior_plan=behavior_plan,
        ).metadata

        assert agent.llm.call_count == 2
        assert any(
            "RecoveryController is absent" in issue
            for issue in metadata["step1_plan_attempts"][0]["issues"]
        )
        assert metadata["step1_plan_attempts"][1]["plan_status"] == "PASS"
        assert len(
            metadata["whole_model_generation_plan"][
                "behavior_obligations"
            ]
        ) == 1

    def test_semantic_requirement_is_bound_in_plan_and_routed_to_interfaces(
        self, monkeypatch
    ):

        requirement = (
            "REQ-FUNC-002: The system shall maintain at least 5 metres "
            "of separation while avoiding an obstacle."
        )
        payload = {
            "components": [
                {
                    "name": "Perception",
                    "responsibility": "Measures obstacle separation.",
                    "requirements": ["REQ_FUNC_002"],
                    "ports": [{
                        "name": "obstacleData",
                        "direction": "out",
                        "type": "ObstaclePort",
                    }],
                },
                {
                    "name": "Controller",
                    "responsibility": "Maintains obstacle separation.",
                    "requirements": ["REQ_FUNC_002"],
                    "ports": [{
                        "name": "obstacleData",
                        "direction": "in",
                        "type": "ObstaclePort",
                    }],
                },
            ],
            "connections": [{
                "source": {
                    "component": "Perception",
                    "port": "obstacleData",
                },
                "target": {
                    "component": "Controller",
                    "port": "obstacleData",
                },
                "item_type": "ObstaclePort",
                "requirements": ["REQ_FUNC_002"],
            }],
            "requirement_realizations": [{
                "requirement_id": "REQ_FUNC_002",
                "realization_kind": "CAUSAL_PATH",
                "trigger_concept": "an obstacle",
                "effect_concept": (
                    "maintain at least 5 metres of separation"
                ),
                "connection_path": [{
                    "source_component": "Perception",
                    "source_port": "obstacleData",
                    "target_component": "Controller",
                    "target_port": "obstacleData",
                    "item_type": "ObstaclePort",
                }],
            }],
            "semantic_bindings": [{
                "obligation_id": "SEM_REQ_FUNC_002_001",
                "requirement_id": "REQ_FUNC_002",
                "source": {
                    "component": "Perception",
                    "port": "obstacleData",
                },
                "target": {
                    "component": "Controller",
                    "port": "obstacleData",
                    "runtime_attribute": "currentSeparation",
                },
                "payload": {
                    "port_type": "ObstaclePort",
                    "port_feature": "payload",
                    "item_type": "ObstacleData",
                    "item_feature": "separation",
                    "value_type": "Real",
                    "unit": "m",
                },
                "constraint": {
                    "name": "keepSeparation",
                    "threshold_attribute": "minimumSeparation",
                },
            }],
            "constraints": [{
                "constraint_id": "maintainSeparationConstraint",
                "owner": "Controller",
                "expression": {
                    "lhs": "currentSeparation",
                    "operator": ">=",
                    "rhs": "minimumSeparation",
                },
                "activation": {
                    "kind": "STATE_ACTIVE",
                    "state": "AvoidanceBehavior::AvoidingObstacle",
                },
                "provenance": {
                    "kind": "FROZEN_REQUIREMENT",
                    "requirement_id": "REQ_FUNC_002",
                },
                "verification_tier": "STATE_EXECUTION",
            }],
            "behaviors": [{
                "owner": "Controller",
                "behavior_id": "AvoidanceBehavior",
                "initial_state": "NominalFlight",
                "states": [
                    {
                        "state_id": "NominalFlight",
                        "role": "INITIAL",
                    },
                    {
                        "state_id": "AvoidingObstacle",
                        "role": "RESPONSE",
                        "entry_action": "executeObstacleAvoidance",
                    },
                ],
                "transitions": [{
                    "transition_id": "detectObstacle",
                    "source": "NominalFlight",
                    "target": "AvoidingObstacle",
                    "trigger_kind": "ACCEPT",
                    "trigger": "ObstacleDetectedSignal",
                }],
                "provenance": {
                    "kind": "FROZEN_REQUIREMENT",
                    "requirement_id": "REQ_FUNC_002",
                },
            }],
        }
        agent = self._make_agent(
            [
                f"```json\n{json.dumps(payload)}\n```",
                "```sysml\npart def Perception {}\n"
                "part def Controller {}\n```",
                _INTERFACE_FRAGMENT,
                _BEHAVIOR_FRAGMENT,
                _ASSEMBLED_MODEL,
            ],
            monkeypatch,
        )
        outcome = self._generate_typed_plan(
            agent,
            "DroneSystem",
            [requirement],
        )
        rendered = outcome.architecture_text
        plan = outcome.plan
        authored = self._author_model(
            agent,
            "DroneSystem",
            [requirement],
            rendered,
            plan,
        )
        metadata = authored.metadata

        planning_prompt = agent.llm.messages[0][-1].content
        interface_prompt = agent.llm.messages[2][-1].content
        assert "SOURCE-DERIVED SEMANTIC BINDINGS REQUIRED IN PLAN" in (
            planning_prompt
        )
        assert "SEM_REQ_FUNC_002_001" in planning_prompt
        assert plan.schema_version == "9.0"
        assert plan.status == "PASS"
        assert (
            plan.semantic_bindings[0].constraint_name
            == "maintainSeparationConstraint"
        )
        assert metadata["semantic_interface_requirement_ids"] == [
            "REQ_FUNC_002"
        ]
        assert requirement in interface_prompt

    def test_metadata_contains_generation_steps(self, monkeypatch):
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        responses = [
            "Architecture plan",
            _SYSML_FRAGMENT,
            _INTERFACE_FRAGMENT,   # step 3
            _BEHAVIOR_FRAGMENT,    # step 4
            _ASSEMBLED_MODEL,      # step 5
        ]
        agent = self._make_agent(responses, monkeypatch)

        dummy_model = SysMLModel(name="DroneSystem", description="test")
        dummy_model.part_definitions.append(
            PartDefinition(name="FlightController", short_description="controls flight")
        )
        monkeypatch.setattr(da_module, "build_lite_model", lambda *a, **kw: dummy_model)

        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": self._REQUIREMENTS,
        })

        assert result.metadata["generation_steps_completed"] == 5
        assert "architecture_length" in result.metadata
        assert "parts_fragment_length" in result.metadata
        assert "interfaces_fragment_length" in result.metadata
        assert "behavior_fragment_length" in result.metadata

    def test_step4_retries_unregistered_accept_once_and_remains_fail_closed(
        self, monkeypatch
    ):
        from src.prototyping.generation_plan import (
            ComponentPlan,
            ConnectionPlan,
            ModelGenerationPlan,
            PortPlan,
        )
        from src.prototyping.planned_behavior import (
            PlannedBehavior,
            PlannedState,
            PlannedTransition,
        )

        plan = ModelGenerationPlan(
            components=(
                ComponentPlan(
                    name="Controller",
                    responsibility="Controls the response.",
                    requirements=("REQ_FUNC_001",),
                    ports=(PortPlan("signalOut", "out", "SignalPort"),),
                ),
                ComponentPlan(
                    name="Monitor",
                    responsibility="Receives the response.",
                    requirements=(),
                    ports=(PortPlan("signalIn", "in", "SignalPort"),),
                ),
            ),
            connections=(ConnectionPlan(
                source_component="Controller",
                source_port="signalOut",
                target_component="Monitor",
                target_port="signalIn",
                item_type="SignalPort",
            ),),
            planned_behaviors=(PlannedBehavior(
                owner="Controller",
                behavior_id="ControllerBehavior",
                initial_state="Idle",
                states=(
                    PlannedState("Idle", "INITIAL"),
                    PlannedState(
                        "Responding", "RESPONSE",
                        entry_action="performResponse",
                    ),
                ),
                transitions=(PlannedTransition(
                    "respond",
                    "Idle",
                    "Responding",
                    "ACCEPT",
                    "RegisteredSignal",
                ),),
                source_requirement_id="REQ_FUNC_001",
            ),),
        )
        unregistered = """```sysml
state def ExtraBehavior {
    entry; then Idle;
    state Idle;
    state Acting;
    transition act first Idle accept UnplannedSignal then Acting;
}
```"""
        corrected = "```sysml\naction def supportingAction {}\n```"

        for retry_response, should_pass in (
            (corrected, True),
            (unregistered, False),
        ):
            agent = self._make_agent(
                [
                    "```sysml\npart def Controller {}\n"
                    "part def Monitor {}\n```",
                    _INTERFACE_FRAGMENT,
                    unregistered,
                    retry_response,
                    _ASSEMBLED_MODEL,
                ],
                monkeypatch,
            )
            if not should_pass:
                with pytest.raises(
                    RuntimeError,
                    match="after one targeted retry",
                ):
                    self._author_model(
                        agent,
                        "DroneSystem",
                        ["REQ-FUNC-001: The system shall respond to a signal."],
                        plan.render_for_prompt(),
                        plan,
                    )
                assert agent.llm.call_count == 4
                continue

            outcome = self._author_model(
                agent,
                "DroneSystem",
                ["REQ-FUNC-001: The system shall respond to a signal."],
                plan.render_for_prompt(),
                plan,
            )
            metadata = outcome.metadata
            fragment = outcome.fragments.behavior
            retry_prompt = agent.llm.messages[3][-1].content
            assert agent.llm.call_count == 5
            assert metadata["step4_behavior_retries"] == 1
            assert metadata["planned_behavior_conformance"]["status"] == "PASS"
            assert "RegisteredSignal" in retry_prompt
            assert "UnplannedSignal" in retry_prompt
            assert "UnplannedSignal" not in fragment
            assert "item def RegisteredSignal" in fragment

    def test_step2_retries_once_when_first_response_has_no_part_defs(
        self, monkeypatch
    ):
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        responses = [
            "Architecture plan",
            "```sysml\nitem def NotAPart;\n```",
            _SYSML_FRAGMENT,
            _INTERFACE_FRAGMENT,
            _BEHAVIOR_FRAGMENT,
            _ASSEMBLED_MODEL,
        ]
        agent = self._make_agent(responses, monkeypatch)
        dummy_model = SysMLModel(name="DroneSystem", description="test")
        dummy_model.part_definitions.append(PartDefinition(name="FlightController"))
        monkeypatch.setattr(da_module, "build_lite_model", lambda *a, **kw: dummy_model)

        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": self._REQUIREMENTS,
        })

        assert result.success
        assert agent.llm.call_count == 6
        assert result.metadata["step2_part_retries"] == 1

    def test_step2_retries_cross_kind_or_unplanned_part_def(
        self, monkeypatch
    ):
        from src.prototyping.generation_plan import ModelGenerationPlan

        plan = ModelGenerationPlan.from_payload({
            "components": [
                {
                    "name": "Producer",
                    "responsibility": "Produces data.",
                    "requirements": [],
                    "ports": [{
                        "name": "data",
                        "direction": "out",
                        "type": "DataPort",
                    }],
                },
                {
                    "name": "Consumer",
                    "responsibility": "Consumes data.",
                    "requirements": [],
                    "ports": [{
                        "name": "data",
                        "direction": "in",
                        "type": "DataPort",
                    }],
                },
            ],
            "connections": [{
                "source": {"component": "Producer", "port": "data"},
                "target": {"component": "Consumer", "port": "data"},
                "item_type": "DataPort",
            }],
        })
        agent = self._make_agent([
            """```sysml
            part def Producer {}
            part def Consumer {}
            part def ObstacleData {}
            ```""",
            """```sysml
            part def Producer {}
            part def Consumer {}
            ```""",
            _INTERFACE_FRAGMENT,
            _ASSEMBLED_MODEL,
        ], monkeypatch)

        outcome = self._author_model(
            agent,
            "P",
            [],
            plan.render_for_prompt(),
            plan,
        )
        metadata = outcome.metadata
        fragment = outcome.fragments.parts

        assert agent.llm.call_count == 4
        assert "part def ObstacleData" not in fragment
        assert metadata["step2_definition_contract"]["status"] == "PASS"
        assert "STEP 2 DEFINITION-KIND CORRECTION" in (
            agent.llm.messages[1][-1].content
        )

    def test_step2_fails_after_bounded_empty_structure_retry(self, monkeypatch):
        responses = [
            "Architecture plan",
            "no structural model",
            "still no structural model",
        ]
        agent = self._make_agent(responses, monkeypatch)

        with pytest.raises(RuntimeError, match="Step 2 produced no part definitions"):
            agent.run({
                "system_name": "DroneSystem",
                "requirements": self._REQUIREMENTS,
            })

        assert agent.llm.call_count == 3

    def test_initial_generation_rejects_zero_parseable_parts(self, monkeypatch):
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel

        responses = [
            "Architecture plan",
            _SYSML_FRAGMENT,
            _INTERFACE_FRAGMENT,
            _BEHAVIOR_FRAGMENT,
            _ASSEMBLED_MODEL,
        ]
        agent = self._make_agent(responses, monkeypatch)
        monkeypatch.setattr(
            da_module,
            "build_lite_model",
            lambda *a, **kw: SysMLModel(name="DroneSystem", description="empty"),
        )

        with pytest.raises(RuntimeError, match="no parseable part definitions"):
            agent.run({
                "system_name": "DroneSystem",
                "requirements": self._REQUIREMENTS,
            })

        # Five generation calls plus one bounded common syntax-repair call.
        assert agent.llm.call_count == 6

    def test_initial_parse_failure_gets_one_uniform_syntax_repair(
        self, monkeypatch
    ):
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        responses = [
            "Architecture plan",
            _SYSML_FRAGMENT,
            _INTERFACE_FRAGMENT,
            _BEHAVIOR_FRAGMENT,
            _ASSEMBLED_MODEL,
            _ASSEMBLED_MODEL,
        ]
        agent = self._make_agent(responses, monkeypatch)
        empty = SysMLModel(name="DroneSystem", description="unparseable")
        repaired = SysMLModel(name="DroneSystem", description="repaired")
        repaired.part_definitions.append(PartDefinition(name="FlightController"))
        parsed = iter((empty, repaired))
        monkeypatch.setattr(
            da_module, "build_lite_model", lambda *a, **kw: next(parsed)
        )

        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": self._REQUIREMENTS,
        })

        assert result.success
        assert agent.llm.call_count == 6
        assert result.metadata["initial_parse_repair"]["attempted"] is True
        assert result.metadata["initial_parse_repair"]["successful"] is True

    def test_behavior_step_skipped_without_func_safe_reqs(self, monkeypatch):
        """Only 4 LLM calls when no FUNC or SAFE requirements exist.

        Pipeline: step1(arch) + step2(parts) + step3(interfaces) + step5(assembly)
        Step 4 (behavior) is skipped — no FUNC/SAFE reqs.
        """
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        perf_intf_only = [
            "REQ-PERF-001: The drone shall maintain altitude below 400 m.",
            "REQ-INTF-001: The drone shall exchange telemetry with ground station via MAVLink protocol.",
        ]
        responses = [
            "Architecture plan",  # step 1
            _SYSML_FRAGMENT,      # step 2
            _INTERFACE_FRAGMENT,  # step 3 (interfaces — always runs)
            # step 4 skipped (no FUNC/SAFE)
            _ASSEMBLED_MODEL,     # step 5
        ]
        agent = self._make_agent(responses, monkeypatch)

        dummy_model = SysMLModel(name="DroneSystem", description="test")
        dummy_model.part_definitions.append(
            PartDefinition(name="FlightController", short_description="controls flight")
        )
        monkeypatch.setattr(da_module, "build_lite_model", lambda *a, **kw: dummy_model)

        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": perf_intf_only,
        })

        assert result.success
        assert agent.llm.call_count == 4
        assert result.metadata["behavior_fragment_length"] == 0

    def test_refinement_mode_still_uses_single_step(self, monkeypatch):
        """Refinement mode must not go through the multi-step pipeline."""
        import src.agents.design_agent as da_module
        from src.sysml.model import SysMLModel, PartDefinition

        existing = SysMLModel(name="DroneSystem", description="existing")

        responses = [
            _ASSEMBLED_MODEL,  # single refine_design call
        ]
        agent = self._make_agent(responses, monkeypatch)

        dummy_model = SysMLModel(name="DroneSystem", description="test")
        dummy_model.part_definitions.append(
            PartDefinition(name="FlightController", short_description="controls flight")
        )
        monkeypatch.setattr(da_module, "build_lite_model", lambda *a, **kw: dummy_model)

        result = agent.run({
            "system_name": "DroneSystem",
            "requirements": self._REQUIREMENTS,
            "existing_model": existing,
            "refinement_feedback": "Add safety states",
        })

        assert result.success
        assert agent.llm.call_count == 1


def test_range_floor_is_not_emitted_as_opposite_always_on_constraint():

    text = """package D {
        part def Airframe {
            attribute maxOperationalRange : Real = 5.0;
            attribute currentOperationalRange : Real = 0.0;
            assert constraint operationalRangeBound {
                currentOperationalRange <= maxOperationalRange
            }
        }
    }"""
    fixed, count = fix_capability_semantics(
        text,
        has_range_floor=True,
        has_range_ceiling=False,
    )

    assert count >= 2
    assert "minOperationalRange" in fixed
    assert "currentOperationalRange <=" not in fixed
    assert "forward-flight fidelity" in fixed


def test_missing_part_defs_are_restored_from_structural_fragment():
    assembled = """package DroneSystem {
        requirement def REQ_FUNC_001 { }
    }"""
    parts = """part def FlightController {
        in port commandIn : DataPort;
        attribute controlFrequency : Real = 100.0 [Hz];
    }
    part def PayloadManager {
        out port payloadStatus : DataPort;
        attribute payloadMass : Real = 2.0 [kg];
    }"""

    outcome = _finalize_assembly(assembled, parts=parts)
    restored = outcome.response.extracted_sysml
    names = outcome.metadata["injected_part_defs"]

    assert names == ["FlightController", "PayloadManager"]
    assert restored.count("part def FlightController") == 1
    assert restored.count("part def PayloadManager") == 1


def test_missing_defs_are_restored_into_quoted_package_name():
    assembled = "package 'Autonomous Drone' { requirement def REQ_FUNC_001 { } }"
    parts = "part def FlightController { attribute x : Real = 1.0; }"
    interfaces = "item def TelemetryData;"

    outcome = _finalize_assembly(
        assembled,
        parts=parts,
        interfaces=interfaces,
    )
    restored = outcome.response.extracted_sysml
    part_names = outcome.metadata["injected_part_defs"]
    item_names = outcome.metadata["injected_item_defs"]

    assert part_names == ["FlightController"]
    assert item_names == ["item def TelemetryData"]
    assert "part def FlightController" in restored
    assert "item def TelemetryData" in restored


def test_post_assembly_restores_event_item_before_behavior_compilation():
    from src.prototyping.generation_plan import (
        ComponentPlan,
        ModelGenerationPlan,
    )
    from src.prototyping.planned_behavior import (
        PlannedBehavior,
        PlannedState,
        PlannedTransition,
    )

    plan = ModelGenerationPlan(
        components=(ComponentPlan(
            name="Controller",
            responsibility="Responds to override commands.",
            requirements=(),
            ports=(),
        ),),
        planned_behaviors=(PlannedBehavior(
            owner="Controller",
            behavior_id="ControllerBehavior",
            initial_state="idle",
            states=(
                PlannedState("idle", "INITIAL"),
                PlannedState(
                    "responding", "RESPONSE", entry_action="respond"
                ),
            ),
            transitions=(PlannedTransition(
                "receive",
                "idle",
                "responding",
                "ACCEPT",
                "OverrideCommand",
            ),),
        ),),
    )
    assembled = """package P {
        port def OverrideCommand { in item command; }
        part def Controller {}
    }"""
    interfaces = """item def OverrideCommand {
        attribute overrideActive : Boolean;
    }"""
    outcome = _finalize_assembly(
        assembled,
        interfaces=interfaces,
        behavior="planned behavior fragment",
        generation_plan=plan,
    )
    result = outcome.response
    metadata = outcome.metadata

    assert result.extracted_sysml.count("item def OverrideCommand") == 1
    assert "port def OverrideCommand" not in result.extracted_sysml
    assert "attribute overrideActive : Boolean;" in result.extracted_sysml
    assert "accept OverrideCommand" in result.extracted_sysml
    assert metadata["owned_planned_behavior_conformance"]["status"] == "PASS"


def test_existing_part_defs_are_not_duplicated_during_restore():
    assembled = "package D { part def FlightController { } }"
    parts = "part def FlightController { attribute x : Real = 1.0; }"

    outcome = _finalize_assembly(assembled, parts=parts)
    restored = outcome.response.extracted_sysml
    names = outcome.metadata.get("injected_part_defs", [])

    assert restored == assembled
    assert names == []


def test_range_floor_cleanup_does_not_remove_sensor_range_constraint():
    text = """package D {
        part def PerceptionSystem {
            attribute maxSensorRange : Real = 15.0;
            attribute currentSensorRange : Real = 15.0;
            assert constraint sensorRangeBound {
                currentSensorRange <= maxSensorRange
            }
        }
        part def PropulsionSystem {
            attribute minOperationalRange : Real = 5.0;
            attribute currentRange : Real = 0.0;
            assert constraint operationalRangeBound {
                currentRange >= minOperationalRange
            }
        }
    }"""
    fixed, count = fix_capability_semantics(
        text,
        has_range_floor=True,
        has_range_ceiling=False,
    )

    assert count == 1
    assert "currentSensorRange <= maxSensorRange" in fixed
    assert "currentRange >= minOperationalRange" not in fixed


def test_a_wrong_safety_command_is_no_longer_silently_respelled():
    """`fix_safety_action_semantics` (removed 2026-08-31) rewrote
    `send CMD_LAND()` inside parachute actions into `send CMD_PARACHUTE()`
    and injected the harness's command definition — semantic forgery that
    laundered a real arbitration defect into a pass. The wrong command must
    survive normalization untouched and surface at the linker's traceability
    check (test_parachute_guard_with_land_command_is_traceability_blocked)."""
    from src.sysml import text_normalization

    assert not hasattr(text_normalization, "fix_safety_action_semantics")
    assert "fix_safety_action_semantics" not in (
        text_normalization.NORMALIZATION_RULE_ORDER["design_semantics"]
    )


def test_self_test_satisfy_is_relocated_to_state_machine_owner():
    text = """package D {
        requirement def REQ_FUNC_009 { }
        part def FlightController {
            state def ModeMachine {
                state PhasePowerOn;
                state PhaseSelfTest;
                state PhaseArmed;
                transition initial then PhasePowerOn;
                transition test first PhasePowerOn then PhaseSelfTest;
                transition arm first PhaseSelfTest then PhaseArmed;
            }
        }
        part def SafetyMonitor {
            satisfy requirement REQ_FUNC_009;
        }
    }"""
    requirements = [
        "REQ-FUNC-009: Execute an automated system self-check prior to arming."
    ]

    outcome = _admit_model(text, requirements)
    fixed = outcome.response.extracted_sysml
    count = outcome.metadata["semantic_fixes"][
        "functional_satisfy_ownership"
    ]

    assert count == 1
    assert fixed.count("satisfy requirement REQ_FUNC_009;") == 1
    flight_block = fixed[fixed.index("part def FlightController"):fixed.index("part def SafetyMonitor")]
    assert "satisfy requirement REQ_FUNC_009;" in flight_block


def test_bare_self_test_phase_gets_executable_entry_action():
    text = """package D {
        part def FlightController {
            action def executeSelfTest { }
            state def ModeMachine {
                state PhasePowerOn;
                state PhaseSelfTest;
                state PhaseArmed;
            }
        }
    }"""
    requirements = [
        "REQ-FUNC-009: Execute an automated system self-test prior to arming."
    ]

    outcome = _admit_model(text, requirements)
    fixed = outcome.response.extracted_sysml
    count = outcome.metadata["semantic_fixes"]["self_test_behavior"]

    assert count == 1
    assert "state PhaseSelfTest {" in fixed
    assert "entry action runSelfTest : executeSelfTest;" in fixed
    assert fixed.count("action def executeSelfTest") == 1


def test_functional_satisfy_owner_fix_closes_self_check_audit_gap():
    from src.agents.verification_audit import functional_verification_gap_issues

    text = """package D {
        action def CmdToSelfTest { }
        action def CmdToArmed { }
        requirement def REQ_FUNC_009 {
            doc /* Execute an automated system self-check prior to arming. */
        }
        part def FlightController {
            state def ModeMachine {
                state PhasePowerOn;
                state PhaseSelfTest;
                state PhaseArmed;
                transition initial then PhasePowerOn;
                transition test first PhasePowerOn accept CmdToSelfTest then PhaseSelfTest;
                transition arm first PhaseSelfTest accept CmdToArmed then PhaseArmed;
            }
        }
        part def SafetyMonitor {
            satisfy requirement REQ_FUNC_009;
        }
    }"""
    requirements = [
        "REQ-FUNC-009: Execute an automated system self-check prior to arming."
    ]

    before = functional_verification_gap_issues(text, "D", strict=True)
    fixed = _admit_model(text, requirements).response.extracted_sysml
    after = functional_verification_gap_issues(fixed, "D", strict=True)

    assert any("REQ_FUNC_009" in issue for issue in before)
    assert not after
