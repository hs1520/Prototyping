"""LLM-decided A/G specs rendered deterministically (R2 mode LLM_DECIDED_SPEC).

Nine seeds of the LLM-authored-SysML mode agreed with frozen gold on guarantee
allocation 6/6 and on assumption discharge 4/6, yet never once reached a PASS: the
engineering was right and the notation was wrong. This mode takes the notation
away — the model returns decisions, the emitter renders them — so conformance holds
by construction and only the decisions are judged.

The property that matters is that this does not launder wrongness into rightness: a
wrong decision must still produce a well-formed model that scores badly, never a
model that looks correct.
"""
from __future__ import annotations

import json

import pytest

from src.agents.orchestrator import Orchestrator
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_decision import (
    DecisionError,
    build_spec_from_decisions,
    extract_decisions,
    validate_decisions,
)
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_evaluation import evaluate_ag_against_gold
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.architecture_boundary import build_architecture_boundary_draft
from tests.test_option2_ag_evaluation import REQ_SAFE_005_GOLD

_BASE = (
    "package Src { requirement def REQ_SAFE_005 { doc /* deploy the ballistic "
    "recovery parachute within 0.5 s of a critical propulsion failure. */ } }"
)
_BOUNDARY = build_architecture_boundary_draft(REQ_SAFE_005_CHAIN)

_CORRECT = {
    "safety_pattern": "TRIGGERED_TIMED_FAILSAFE_RESPONSE",
    "timing_origin": "criticalPropulsionFailureDetected",
    "deadline_seconds": 0.5,
    "observation": "parachuteDeployed",
    "system_assumptions": ["airborne", "criticalPropulsionFailureDetected"],
    "components": [
        {"component_id": "SafetyResponseArbiterContract",
         "latency_budget_seconds": 0.1, "timing_segment_required": True,
         "assumptions": [{"concept": "airborne", "discharged_by": None},
                         {"concept": "criticalPropulsionFailureDetected",
                          "discharged_by": None}]},
        {"component_id": "RecoveryPowerSupplyContract",
         "timing_segment_required": False,
         "assumptions": [{"concept": "airborne", "discharged_by": None}]},
        {"component_id": "RecoverySystemContract",
         "latency_budget_seconds": 0.35, "timing_segment_required": True,
         "assumptions": [
             {"concept": "parachuteDeploymentCommand",
              "discharged_by": "SafetyResponseArbiterContract"},
             {"concept": "recoveryActuationPowerAvailable",
              "discharged_by": "RecoveryPowerSupplyContract"}]},
    ],
    "priority": {"response_set_id": "FLIGHT_RESPONSES_V1",
                 "members": ["PARACHUTE_DEPLOYMENT", "OTHER_RESPONSE"],
                 "selected_response": "PARACHUTE_DEPLOYMENT"},
}


def _render(decisions) -> str:
    spec = build_spec_from_decisions(decisions, _BOUNDARY)
    return _BASE.rstrip() + "\n\n" + emit_ag_package(spec) + "\n"


class _DecisionLLM:
    def __init__(self, payload):
        self._payload = payload

    def chat(self, prompt, system_prompt=None):
        return f"```json\n{json.dumps(self._payload)}\n```"


def test_decisions_render_to_a_model_that_passes_by_construction():
    report = check_ag_graph(extract_ag_graph(_render(_CORRECT)))
    assert report.verdict == "PASS"
    assert not report.errors()


def test_a_non_gold_response_set_still_passes_the_gold_blind_checker():
    """The measured runs produced a two-member response set because the
    requirement never names the others. That is a wrong answer, not a malformed
    one, so the runtime verdict must accept it and the evaluator must mark it."""
    assert _CORRECT["priority"]["members"] == [
        "PARACHUTE_DEPLOYMENT", "OTHER_RESPONSE"
    ], "fixture must use the non-gold response set"
    assert check_ag_graph(extract_ag_graph(_render(_CORRECT))).verdict == "PASS"


def test_a_wrong_discharge_decision_is_well_formed_but_scores_below_one():
    """The whole point: rendering deterministically must not launder a wrong
    decision into a right one."""
    wrong = json.loads(json.dumps(_CORRECT))
    for component in wrong["components"]:
        if component["component_id"] == "RecoverySystemContract":
            for assumption in component["assumptions"]:
                assumption["discharged_by"] = None  # claim both are environment
    prediction = check_ag_graph(extract_ag_graph(_render(wrong))).to_dict()
    out = evaluate_ag_against_gold(prediction, REQ_SAFE_005_GOLD)
    assert out["assumption_discharge"]["f1"] < 1.0
    assert out["assumption_discharge"]["recall"] < 1.0


def test_a_component_named_by_its_part_is_resolved_not_refused():
    """The boundary shows a component as Contract/part/Def, and a model asked to
    name one reaches for the part — the first live run failed closed on exactly
    that. Naming the same component differently is not extending the
    architecture."""
    by_part = json.loads(json.dumps(_CORRECT))
    by_part["components"][0]["component_id"] = "safetyResponseArbiter"
    by_part["components"][2]["assumptions"][0]["discharged_by"] = (
        "safetyResponseArbiter"
    )
    report = check_ag_graph(extract_ag_graph(_render(by_part)))
    assert report.verdict == "PASS"
    # and it is the same component, not an extra one
    spec = build_spec_from_decisions(by_part, _BOUNDARY)
    assert [item.name for item in spec.components] == [
        item.name for item in build_spec_from_decisions(_CORRECT, _BOUNDARY).components
    ]


def test_the_architecture_may_not_be_extended_by_a_decision():
    invented = json.loads(json.dumps(_CORRECT))
    invented["components"].append(
        {"component_id": "MadeUpContract", "assumptions": []}
    )
    with pytest.raises(DecisionError, match="unknown component"):
        validate_decisions(invented, _BOUNDARY)


def test_every_boundary_component_must_be_decided():
    partial = json.loads(json.dumps(_CORRECT))
    partial["components"] = partial["components"][:2]
    with pytest.raises(DecisionError, match="must be decided"):
        validate_decisions(partial, _BOUNDARY)


def test_a_self_discharging_assumption_is_refused():
    looped = json.loads(json.dumps(_CORRECT))
    looped["components"][2]["assumptions"][0]["discharged_by"] = (
        "RecoverySystemContract"
    )
    with pytest.raises(DecisionError, match="discharges itself"):
        validate_decisions(looped, _BOUNDARY)


def test_an_unknown_pattern_is_refused():
    bogus = json.loads(json.dumps(_CORRECT))
    bogus["safety_pattern"] = "MADE_UP_PATTERN"
    with pytest.raises(DecisionError, match="safety_pattern must be one of"):
        validate_decisions(bogus, _BOUNDARY)


def test_a_timed_pattern_without_a_deadline_is_refused():
    undated = json.loads(json.dumps(_CORRECT))
    undated["deadline_seconds"] = None
    with pytest.raises(DecisionError, match="needs deadline_seconds"):
        validate_decisions(undated, _BOUNDARY)


def test_a_selected_response_outside_the_member_set_is_refused():
    inconsistent = json.loads(json.dumps(_CORRECT))
    inconsistent["priority"]["selected_response"] = "NOT_A_MEMBER"
    with pytest.raises(DecisionError, match="not in members"):
        validate_decisions(inconsistent, _BOUNDARY)


def test_decisions_are_extracted_from_a_fenced_response():
    payload = extract_decisions('noise\n```json\n{"a": {"b": 1}}\n```\ntrailing')
    assert payload == {"a": {"b": 1}}


def test_a_response_without_a_json_object_fails_closed():
    with pytest.raises(DecisionError, match="no JSON object"):
        extract_decisions("I cannot help with that.")


def test_orchestrator_decided_mode_renders_a_passing_package():
    orch = Orchestrator(_DecisionLLM(_CORRECT), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    merged = orch._apply_ag_contract_layer(
        _BASE, ["REQ-SAFE-005: deploy the parachute within 0.5 s"]
    )
    report = check_ag_graph(extract_ag_graph(merged))
    assert report.verdict == "PASS"
    assert not report.errors()


class _SequenceDecisionLLM:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self._calls = 0

    def chat(self, prompt, system_prompt=None):
        self.last_prompt = prompt
        payload = self._payloads[min(self._calls, len(self._payloads) - 1)]
        self._calls += 1
        return json.dumps(payload)


def test_incoherent_decisions_are_fed_back_and_repaired():
    """Two live seeds returned a single-member response set, because the
    requirement never names the other responses. Decisions are small and the
    validator says exactly what is wrong, so that is worth one feedback round
    rather than discarding the whole run."""
    thin = json.loads(json.dumps(_CORRECT))
    thin["priority"]["members"] = ["PARACHUTE_DEPLOYMENT"]
    llm = _SequenceDecisionLLM([thin, _CORRECT])

    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    merged = orch._apply_ag_contract_layer(
        _BASE, ["REQ-SAFE-005: deploy the parachute within 0.5 s"]
    )
    assert check_ag_graph(extract_ag_graph(merged)).verdict == "PASS"
    # the retry carried the validator's actual complaint
    assert "at least two responses" in llm.last_prompt


def test_decisions_that_never_become_coherent_still_fail_closed():
    """The retry must not become a way to eventually accept anything."""
    thin = json.loads(json.dumps(_CORRECT))
    thin["priority"]["members"] = ["PARACHUTE_DEPLOYMENT"]
    orch = Orchestrator(_SequenceDecisionLLM([thin]),
                        revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    with pytest.raises(RuntimeError, match="failed closed after"):
        orch._apply_ag_contract_layer(
            _BASE, ["REQ-SAFE-005: deploy the parachute within 0.5 s"]
        )


def test_orchestrator_decided_mode_fails_closed_on_unusable_decisions():
    orch = Orchestrator(_DecisionLLM({"safety_pattern": "NONSENSE"}),
                        revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    with pytest.raises(RuntimeError, match="failed closed"):
        orch._apply_ag_contract_layer(
            _BASE, ["REQ-SAFE-005: deploy the parachute within 0.5 s"]
        )
