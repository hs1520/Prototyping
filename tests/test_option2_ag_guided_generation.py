"""R2 A/G decisions must guide generation, not be appended only at the end."""
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


def test_r2_freezes_a_plan_for_every_design_generation_step():
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


def test_frozen_ag_package_remains_planning_input_before_first_commit():
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


def test_terminal_reconciliation_reuses_the_frozen_plan_without_reauthoring():
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


def test_design_handoff_carries_the_frozen_plan_as_a_typed_input():
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


def test_the_pre_generation_gate_admits_a_correct_decision_set():
    """The gate must be satisfiable at generation stage.

    It compiles the candidate and checks it before the ordinary architecture
    spends four more provider calls. But the emitted package never contains
    `requirement def REQ_SAFE_005` — the authoritative requirement lives in the
    model, which at this point is still the frozen planning inputs. Checking the
    package alone made SOURCE_PROVENANCE_MISSING unsatisfiable by construction,
    so every decision set the LLM could return was refused and the arm could not
    generate at all.
    """
    orchestrator = _decided_orchestrator(_CORRECT)
    spec = orchestrator._generate_llm_decided_ag_spec(
        REQ_SAFE_005_CHAIN, _planning_model(), generation_stage=True,
    )
    assert spec.source_requirement == "REQ_SAFE_005"


def test_the_pre_generation_gate_still_refuses_an_over_budget_decision_set():
    """Admitting the correct set must not have cost the gate its teeth."""
    over_budget = copy.deepcopy(_CORRECT)
    over_budget["components"][0]["latency_budget_seconds"] = 0.4
    orchestrator = _decided_orchestrator(over_budget)
    with pytest.raises(RuntimeError, match="failed closed"):
        orchestrator._generate_llm_decided_ag_spec(
            REQ_SAFE_005_CHAIN, _planning_model(), generation_stage=True,
        )


def test_the_gate_reads_provenance_from_the_requirement_inputs_it_is_given():
    """Pins *why* the gate passes: the requirement inputs, not the package.

    Withhold them and the same correct decisions are refused for missing source
    provenance — which is exactly the failure a package-only check produced on
    every run.
    """
    orchestrator = _decided_orchestrator(_CORRECT)
    with pytest.raises(RuntimeError, match="SOURCE_PROVENANCE_MISSING"):
        orchestrator._generate_llm_decided_ag_spec(
            REQ_SAFE_005_CHAIN, "package AGPlanningInputs { }",
            generation_stage=True,
        )


def test_authored_guidance_uses_authored_behavior_not_reviewed_realization_gold():
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
