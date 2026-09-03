from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from src.agents.orchestrator import Orchestrator
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_emitter import ag_event_signals, emit_ag_package
from src.sysml.lite_model import build_lite_model
from src.utils.sysml_text_utils import get_sysml_text
from tests.test_option2_ag_decision import _CORRECT, _DecisionLLM


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("deterministic planning must not call the LLM")

    def chat(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("deterministic planning must not call the LLM")


_REQS = [
    "REQ-SAFE-005: deploy the parachute within 0.5 s of a critical propulsion "
    "failure.",
]


def _orchestrator() -> Orchestrator:
    return Orchestrator(
        _NoCallLLM(),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="DETERMINISTIC_SPEC_EMITTER",
    )


def test_plan_frozen_for_every_step():
    orchestrator = _orchestrator()
    plan = orchestrator._prepare_ag_guided_generation(_REQS)
    assert plan["stage"] == "PRE_GENERATION_A_G_PLANNING"
    assert set(plan["guidance_by_step"]) == {
        "architecture", "parts", "interfaces", "behavior", "assembly",
    }
    assert "SafetyResponseArbiter" in plan["guidance_by_step"]["architecture"]
    assert "criticalPropulsionFailureDetected" in (
        plan["guidance_by_step"]["behavior"]
    )
    assert "Do not emit planning package `REQ_SAFE_005_AG`" in (
        plan["guidance_by_step"]["assembly"]
    )
    assert "requirement def SystemSafe005Contract" not in (
        plan["guidance_by_step"]["assembly"]
    )


def test_frozen_package_stays_input():
    orchestrator = _orchestrator()
    orchestrator._active_ag_generation_plan = (
        orchestrator._prepare_ag_guided_generation(_REQS)
    )
    base = build_lite_model(
        "package DeliveryUAV { requirement def REQ_SAFE_005 { "
        "doc /* deploy the parachute within 0.5 s of a critical propulsion "
        "failure. */ } part def Airframe {} }\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN),
        model_name="DeliveryUAV",
    )
    materialized = orchestrator._materialize_guided_ag_contracts(
        base, "DeliveryUAV"
    )
    text = get_sysml_text(materialized)
    assert "package REQ_SAFE_005_AG" not in text
    assert materialized.metadata["ag_generation_stage"] == (
        "PLANNING_INPUT_ONLY"
    )
    assert materialized.metadata["ag_planning_reports"][0]["verdict"] == "PASS"


def test_terminal_reuses_frozen_plan():
    orchestrator = _orchestrator()
    orchestrator._active_ag_generation_plan = (
        orchestrator._prepare_ag_guided_generation(_REQS)
    )

    def forbidden(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("terminal path must not author a new A/G layer")

    orchestrator._apply_ag_contract_layer = forbidden
    orchestrator.state = SimpleNamespace(system_name="DeliveryUAV")
    event_definitions = "\n".join(
        f"            item def {name};"
        for name in ag_event_signals(REQ_SAFE_005_CHAIN)
    )
    terminal_model = (
        """
        package DeliveryUAV {
"""
        + event_definitions
        + """
            part def SafetyResponseArbiter {
                attribute criticalPropulsionFailureDetected : Boolean = false;
                state def SafetyResponseArbitration { state idle; }
            }
            part safetyResponseArbiter : SafetyResponseArbiter;
            part def RecoveryPowerSupply {
                attribute airborne : Boolean = false;
                attribute recoveryActuationPowerAvailable : Boolean = true;
                assert constraint RecoveryPowerSupplyBehavior {
                    not (airborne) or (recoveryActuationPowerAvailable)
                }
            }
            part recoveryPowerSupply : RecoveryPowerSupply;
            part def RecoverySystem {
                state def RecoverySystemBehavior { state idle; }
            }
            part recoverySystem : RecoverySystem;
        }
        """
    )
    final = orchestrator._reconcile_guided_ag_contract_layer(
        terminal_model,
        _REQS,
    )
    assert "package REQ_SAFE_005_AG" in final
    assert "part def SafetyResponseArbiter;" not in final
    assert orchestrator.last_ag_binding_report["status"] == "PASS"


def test_handoff_carries_frozen_plan():
    orchestrator = _orchestrator()
    orchestrator.last_requirement_input = {
        "mode": "frozen", "requirement_set_digest": "digest",
    }
    orchestrator._active_ag_generation_plan = (
        orchestrator._prepare_ag_guided_generation(_REQS)
    )
    orchestrator._prepare_design_handoff("DeliveryUAV", _REQS)
    envelope = orchestrator._active_design_handoff["envelope"]
    topics = {
        item["topic"] for item in envelope.context_item_provenance
        if item["kind"] == "blackboard_record"
    }
    assert "requirements.authoritative" in topics
    assert "design.ag_generation_plan" in topics


def _decided_orchestrator(payload) -> Orchestrator:
    return Orchestrator(
        _DecisionLLM(payload),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="LLM_DECIDED_SPEC",
    )


def _planning_model() -> str:
    return Orchestrator._requirement_planning_model(_REQS)


def test_gate_admits_correct_decisions():
    """The gate has to be satisfiable at generation stage.

    It checks the candidate before the architecture spends four more provider calls,
    but the emitted package holds no `requirement def REQ_SAFE_005` - at this point
    the requirement lives in the frozen planning inputs. Checking the package alone
    made SOURCE_PROVENANCE_MISSING unsatisfiable, so every decision set was refused.
    """
    orchestrator = _decided_orchestrator(_CORRECT)
    spec = orchestrator._generate_llm_decided_ag_spec(
        REQ_SAFE_005_CHAIN, _planning_model(), generation_stage=True,
    )
    assert spec.source_requirement == "REQ_SAFE_005"


def test_gate_refuses_over_budget():
    over_budget = copy.deepcopy(_CORRECT)
    over_budget["components"][0]["latency_budget_seconds"] = 0.4
    orchestrator = _decided_orchestrator(over_budget)
    with pytest.raises(RuntimeError, match="failed closed"):
        orchestrator._generate_llm_decided_ag_spec(
            REQ_SAFE_005_CHAIN, _planning_model(), generation_stage=True,
        )


def test_gate_reads_input_provenance():
    """Pins why the gate passes: the requirement inputs, not the package.

    Withhold them and the same decisions are refused for missing source provenance.
    """
    orchestrator = _decided_orchestrator(_CORRECT)
    with pytest.raises(RuntimeError, match="SOURCE_PROVENANCE_MISSING"):
        orchestrator._generate_llm_decided_ag_spec(
            REQ_SAFE_005_CHAIN, "package AGPlanningInputs { }",
            generation_stage=True,
        )


def test_guidance_uses_authored_behavior():
    package = emit_ag_package(REQ_SAFE_005_CHAIN).replace(
        "SafetyResponseArbitration", "AuthoredArbitration"
    ).replace(
        "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand",
        "setAuthoredParachuteDeploymentCommand",
    )
    guidance = Orchestrator._ag_guidance_for_specs(
        [REQ_SAFE_005_CHAIN], [package], authored_mode=True
    )
    assert "AuthoredArbitration" in guidance["behavior"]
    assert "SafetyResponseArbitration" not in guidance["behavior"]


def test_planning_is_board_task():
    """§5.3 rule 1, applied to the role that makes the A/G decisions.

    Freezing the plan used to happen before the board existed, so that LLM work had
    no envelope, no session and no transcript.
    """
    import json

    from src.llm.interface import LLMInterface, LLMResponse
    from src.prototyping.task_session import SessionStatus

    class _RealisticDecisionLLM(LLMInterface):
        def _complete_impl(self, messages, temperature, max_tokens):
            return LLMResponse(content=json.dumps(_CORRECT), model="stub")

    orchestrator = Orchestrator(
        _RealisticDecisionLLM(),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="LLM_DECIDED_SPEC",
    )
    orchestrator.last_requirement_input = {
        "mode": "frozen", "requirement_set_digest": "d"
    }
    assert orchestrator._open_collaboration_board("DeliveryUAV", _REQS) is True
    orchestrator._prepare_ag_guided_generation(_REQS)

    sessions = orchestrator.task_sessions.snapshot(
        include_messages=True
    )["sessions"]
    planning = [
        item for item in sessions
        if item["agent_role"] == "AGPlanningAgent"
    ]
    assert len(planning) == 1, "one bounded session for the planning task"
    assert planning[0]["status"] == SessionStatus.COMPLETED.value
    roles = [message["role"] for message in planning[0]["messages"]]
    assert "assistant" in roles and "user" in roles

    results = orchestrator.blackboard.records(
        topic="agent.ag_planning.result"
    )
    assert len(results) == 1
    payload = results[0].payload
    assert payload["transcript_digest"] == planning[0]["transcript_digest"]
    assert payload["context_envelope_digest"]


def test_planning_design_sessions_differ():
    orchestrator = Orchestrator(
        _DecisionLLM(_CORRECT),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="LLM_DECIDED_SPEC",
    )
    orchestrator.last_requirement_input = {
        "mode": "frozen", "requirement_set_digest": "d"
    }
    orchestrator._open_collaboration_board("DeliveryUAV", _REQS)
    orchestrator._active_ag_generation_plan = (
        orchestrator._prepare_ag_guided_generation(_REQS)
    )
    orchestrator._prepare_design_handoff("DeliveryUAV", _REQS)

    sessions = orchestrator.task_sessions.snapshot()["sessions"]
    roles = sorted(item["agent_role"] for item in sessions)
    assert roles == ["AGPlanningAgent", "DesignAgent"]
    assert len({item["session_id"] for item in sessions}) == 2
    assert len({item["task_id"] for item in sessions}) == 2


def test_drafts_archived_not_authority():
    """The board sees the intermediate drafts as evidence, not as input.

    Nothing used to be published between the DesignAgent task opening and the model
    commit, so no knowledge source could subscribe to a step's output and no
    coordination metric covered generation. Building from them would restore the
    removed external authority, so the ContextBuilder refuses the topic.
    """
    from src.agents.orchestrator import Orchestrator
    from src.prototyping.blackboard import RecordType

    orchestrator = _orchestrator()
    orchestrator.last_requirement_input = {"mode": "frozen",
                                           "requirement_set_digest": "d"}
    orchestrator._open_collaboration_board("DeliveryUAV", _REQS)
    task = orchestrator.blackboard.create_task(
        "INITIAL_MODEL_GENERATION", "DesignAgent",
        required_topics=("requirements.authoritative",),
    )
    handoff = {"task": task, "session": None, "captured_llm_calls": 1}
    orchestrator._publish_generation_fragment(handoff, {
        "label": "parts",
        "conversation_id": "conv-x",
        "new_message_offset": 3,
        "response": {"content": "part def FlightController;",
                     "completion_tokens": 7},
    })

    records = orchestrator.blackboard.records(
        topic=Orchestrator.GENERATION_FRAGMENT_TOPIC
    )
    assert len(records) == 1
    payload = records[0].payload
    assert records[0].record_type is RecordType.ANALYSIS
    assert payload["stage"] == "parts"
    assert payload["multi_turn"] is True
    assert payload["fragment"] == "part def FlightController;"
    assert payload["fragment_digest"]
    assert payload["authority"] == "NONE_ARCHIVAL_ONLY"

    with pytest.raises(ValueError, match="authoritative requirements"):
        orchestrator.context_builder.build_design_context(
            task_id=task.task_id,
            system_name="DeliveryUAV",
            source_record_ids=(records[0].record_id,),
        )


def test_plan_owns_boolean_concepts():
    """Two runs died on `attribute airborne : Real = 0.0;`.

    These concepts are typed by the A/G contract but come from the frozen decisions,
    so nothing owned their declared type and the terminal gate saw the contradiction
    first, too late to do anything but fail. Declaring them in the typed plan puts
    them under the planned-attribute materialiser; the terminal gate stays
    fail-closed.
    """
    from src.prototyping.ag_behavior_plan import (
        behavior_boolean_concepts,
        compile_behavior_obligation_plan,
    )
    from src.prototyping.activated_constraint_plan import (
        materialize_planned_attributes,
    )
    from src.prototyping.generation_plan import (
        ModelGenerationPlan,
        attach_ag_behavior_obligations,
    )

    behavior_plan = compile_behavior_obligation_plan([REQ_SAFE_005_CHAIN])
    # only what a component consumes: a guarantee is an output, generation
    # realises it as a directed port, and the terminal binder accepts a port as a
    # carrier of the concept. Planning it as an attribute too made
    # recoveryActuationPowerAvailable both at once, reported as AMBIGUOUS.
    expected = {
        obligation.owner_def: (
            set(behavior_boolean_concepts(obligation))
            & (set(obligation.assumptions) - set(obligation.guarantees))
        )
        for obligation in behavior_plan.obligations
    }
    assert expected.get("RecoveryPowerSupply") == {"airborne"}, (
        "fixture must exercise a consumed concept and exclude the guarantee"
    )

    plan = ModelGenerationPlan.from_payload({
        "components": [
            {
                "name": obligation.owner_def,
                "responsibility": "x",
                "requirements": ["REQ_SAFE_005"],
                "ports": [],
            }
            for obligation in behavior_plan.obligations
        ],
        "connections": [],
    }, source="TEST", requirements=_REQS)
    plan = attach_ag_behavior_obligations(plan, behavior_plan)

    planned = {
        component.name: {
            item.name: item.value_type for item in component.attributes
        }
        for component in plan.components
    }
    for owner, concepts in expected.items():
        for concept in concepts:
            assert planned[owner].get(concept) == "Boolean", (
                f"{owner}::{concept} must be planned Boolean"
            )
    assert "recoveryActuationPowerAvailable" not in (
        planned["RecoveryPowerSupply"]
    )

    component = next(
        item for item in plan.components if item.name == "RecoveryPowerSupply"
    )
    text, report = materialize_planned_attributes(
        "package S {\n"
        "    part def RecoveryPowerSupply {\n"
        "        attribute airborne : Real = 0.0;\n"
        "    }\n"
        "}\n",
        [component],
    )
    assert "attribute airborne : Boolean;" in text
    assert "Real" not in text
    assert "RecoveryPowerSupply.airborne" in report["restored_attributes"]


def test_port_name_not_also_attribute():
    """Guards a collision that a fix once introduced.

    A first version of the plan-ownership fix declared every Boolean A/G concept as
    an attribute, including guaranteed ones. Those are outputs realised as directed
    ports, so the run failed with `recoveryActuationPowerAvailable cannot be both a
    port and an attribute` plus an AMBIGUOUS terminal binding.
    """
    from src.prototyping.ag_behavior_plan import compile_behavior_obligation_plan
    from src.prototyping.generation_plan import (
        ModelGenerationPlan,
        attach_ag_behavior_obligations,
    )

    behavior_plan = compile_behavior_obligation_plan([REQ_SAFE_005_CHAIN])
    plan = ModelGenerationPlan.from_payload({
        "components": [
            {
                "name": obligation.owner_def,
                "responsibility": "x",
                "requirements": ["REQ_SAFE_005"],
                "ports": [
                    {"name": concept, "direction": "out",
                     "type": "StatusPort", "external": False}
                    for concept in obligation.guarantees
                ],
            }
            for obligation in behavior_plan.obligations
        ],
        "connections": [],
    }, source="TEST", requirements=_REQS)

    plan = attach_ag_behavior_obligations(plan, behavior_plan)

    for component in plan.components:
        ports = {item.name for item in component.ports}
        attributes = {item.name for item in component.attributes}
        assert not (ports & attributes), (
            f"{component.name}: {sorted(ports & attributes)} planned as both"
        )
