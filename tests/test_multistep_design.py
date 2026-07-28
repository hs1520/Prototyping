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


class QueuedMockLLM:
    """Mock LLM that returns responses from a queue in order."""

    def __init__(self, responses: List[str]):
        self._queue: deque = deque(responses)
        self.call_count = 0
        self.messages: List[List[Message]] = []

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> LLMResponse:
        self.call_count += 1
        self.messages.append(list(messages))
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
        # Encourages dynamic thresholds (attribute / arithmetic RHS), now that
        # Layer 1 supports them — not just fixed numeric literals.
        assert "PREFER a dynamic threshold" in prompt
        assert "if batteryCharge <= returnEnergyRequired" in prompt


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

    def _make_agent(self, responses: List[str], monkeypatch):
        """Build a DesignAgent with a queued mock LLM and a no-op parser."""
        import src.agents.design_agent as da_module
        from src.agents.design_agent import DesignAgent

        llm = QueuedMockLLM(responses)
        agent = DesignAgent(llm)

        # Stub out RAG so it returns nothing
        monkeypatch.setattr(agent, "get_augmented_context", lambda *a, **kw: "")

        return agent

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
                    "responsibility": "Controls the system.",
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
                    "responsibility": "Monitors system safety.",
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
        }
        responses = [
            f"```json\n{json.dumps(invalid)}\n```",
            f"```json\n{json.dumps(valid)}\n```",
        ]
        agent = self._make_agent(responses, monkeypatch)
        metadata = {"degraded_steps": []}

        _step, rendered = agent._step1_architecture(
            "DroneSystem",
            self._REQUIREMENTS,
            "",
            "",
            metadata,
            False,
        )

        assert agent.llm.call_count == 2
        assert metadata["step1_plan_retries"] == 1
        assert metadata["whole_model_generation_plan"]["status"] == "PASS"
        assert "Controller.status -> SafetyMonitor.status" in rendered

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
    from src.agents.design_agent import DesignAgent

    text = """package D {
        part def Airframe {
            attribute maxOperationalRange : Real = 5.0;
            attribute currentOperationalRange : Real = 0.0;
            assert constraint operationalRangeBound {
                currentOperationalRange <= maxOperationalRange
            }
        }
    }"""
    fixed, count = DesignAgent._fix_capability_semantics(  # noqa: SLF001
        text,
        ["REQ-PERF-006: The system shall achieve an operational range of at least 5 km."],
    )

    assert count >= 2
    assert "minOperationalRange" in fixed
    assert "currentOperationalRange <=" not in fixed
    assert "forward-flight fidelity" in fixed


def test_missing_part_defs_are_restored_from_structural_fragment():
    from src.agents.design_agent import DesignAgent

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

    restored, names = DesignAgent._inject_missing_part_defs(assembled, parts)

    assert names == ["FlightController", "PayloadManager"]
    assert restored.count("part def FlightController") == 1
    assert restored.count("part def PayloadManager") == 1


def test_missing_defs_are_restored_into_quoted_package_name():
    from src.agents.design_agent import DesignAgent

    assembled = "package 'Autonomous Drone' { requirement def REQ_FUNC_001 { } }"
    parts = "part def FlightController { attribute x : Real = 1.0; }"
    interfaces = "item def TelemetryData;"

    with_parts, part_names = DesignAgent._inject_missing_part_defs(assembled, parts)
    restored, item_names = DesignAgent._inject_missing_item_defs(
        with_parts, interfaces
    )

    assert part_names == ["FlightController"]
    assert item_names == ["item def TelemetryData"]
    assert "part def FlightController" in restored
    assert "item def TelemetryData" in restored


def test_existing_part_defs_are_not_duplicated_during_restore():
    from src.agents.design_agent import DesignAgent

    assembled = "package D { part def FlightController { } }"
    parts = "part def FlightController { attribute x : Real = 1.0; }"

    restored, names = DesignAgent._inject_missing_part_defs(assembled, parts)

    assert restored == assembled
    assert names == []


def test_range_floor_cleanup_does_not_remove_sensor_range_constraint():
    from src.agents.design_agent import DesignAgent

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
    fixed, count = DesignAgent._fix_capability_semantics(  # noqa: SLF001
        text,
        ["REQ-PERF-006: The system shall achieve an operational range of at least 5 km."],
    )

    assert count == 1
    assert "currentSensorRange <= maxSensorRange" in fixed
    assert "currentRange >= minOperationalRange" not in fixed


def test_parachute_action_command_is_repaired_and_declared():
    from src.agents.design_agent import DesignAgent

    text = """package D {
        action def CMD_LAND { }
        part def SafetyMonitor {
            action def deployParachute { send CMD_LAND() to parachuteCmd; }
        }
    }"""
    fixed, count = DesignAgent._fix_safety_action_semantics(text)  # noqa: SLF001

    assert count == 2
    assert "action def CMD_PARACHUTE" in fixed
    assert "send CMD_PARACHUTE() to parachuteCmd" in fixed
    assert "deployParachute { send CMD_LAND" not in fixed


def test_self_test_satisfy_is_relocated_to_state_machine_owner():
    from src.agents.design_agent import DesignAgent

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

    fixed, count = DesignAgent._fix_functional_satisfy_ownership(
        text, requirements
    )

    assert count == 1
    assert fixed.count("satisfy requirement REQ_FUNC_009;") == 1
    flight_block = fixed[fixed.index("part def FlightController"):fixed.index("part def SafetyMonitor")]
    assert "satisfy requirement REQ_FUNC_009;" in flight_block


def test_bare_self_test_phase_gets_executable_entry_action():
    from src.agents.design_agent import DesignAgent

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

    fixed, count = DesignAgent._fix_self_test_behavior_semantics(
        text, requirements
    )

    assert count == 1
    assert "state PhaseSelfTest {" in fixed
    assert "entry action runSelfTest : executeSelfTest;" in fixed
    assert fixed.count("action def executeSelfTest") == 1


def test_functional_satisfy_owner_fix_closes_self_check_audit_gap():
    from src.agents.design_agent import DesignAgent
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
    executable, _ = DesignAgent._fix_self_test_behavior_semantics(text, requirements)
    fixed, _ = DesignAgent._fix_functional_satisfy_ownership(
        executable, requirements
    )
    after = functional_verification_gap_issues(fixed, "D", strict=True)

    assert any("REQ_FUNC_009" in issue for issue in before)
    assert not after
