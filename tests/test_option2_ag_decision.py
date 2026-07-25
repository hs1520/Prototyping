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


def test_a_run_reports_the_generation_mode_it_actually_executed():
    """The arm metadata used to hardcode the deterministic intervention, and the
    pipeline dropped the mode entirely. Together that meant a run could execute one
    intervention and record another — mislabelled evidence the pooling gates would
    accept, because they compare the recorded mode rather than observe behaviour.
    The whole suite passed while this was true, so it needs its own test.
    """
    from src.prototyping.experiment_arms import (
        R2_INTERVENTION_VERSION_BY_MODE,
        RevisedExperimentArm,
        revised_arm_metadata,
    )

    for mode, version in R2_INTERVENTION_VERSION_BY_MODE.items():
        meta = revised_arm_metadata(RevisedExperimentArm.SEMANTIC_ASSURANCE, mode)
        assert meta["r2_generation_mode"] == mode
        # the version is derived from the mode, so the two cannot drift apart
        assert meta["r2_intervention_version"] == version

    with pytest.raises(ValueError, match="unknown r2_generation_mode"):
        revised_arm_metadata(RevisedExperimentArm.SEMANTIC_ASSURANCE, "MADE_UP")


def test_the_pipeline_hands_the_mode_to_the_orchestrator():
    """Plumbing test: the pilot configures the mode, but it only takes effect if
    the pipeline forwards it."""
    from src.prototyping.pipeline import PrototypingPipeline

    pipeline = PrototypingPipeline(
        llm=_DecisionLLM(_CORRECT),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="LLM_DECIDED_SPEC",
    )
    assert pipeline.orchestrator.r2_generation_mode == "LLM_DECIDED_SPEC"
    reported = pipeline.orchestrator._build_collaboration_artifacts(
        _render(_CORRECT)
    )["revised_experiment"]
    assert reported["r2_generation_mode"] == "LLM_DECIDED_SPEC"
    assert reported["r2_intervention_version"] == "r2-bbag-llm-decided-spec-v1"


def test_the_r2_assurance_path_produces_all_three_pillars_and_artifacts(tmp_path):
    """End-to-end through the post-design R2 path, not a component call.

    Every earlier validation of this mode called _apply_ag_contract_layer alone,
    which is why a dropped generation mode and a hardcoded metadata field both
    survived a green suite. This drives commit -> assurance -> artifact writing and
    asserts the blackboard, pattern and traceability pillars all arrive, and that
    the run reports the mode it actually executed.
    """
    from types import SimpleNamespace

    from src.prototyping.run_artifacts import write_revised_run_artifacts
    from src.sysml.lite_model import build_lite_model

    # the committed doc text and the published requirement must agree — the
    # blackboard protects source text and rejects a mismatch
    requirement = (
        "REQ-SAFE-005: " + _BASE.split("doc /*", 1)[1].split("*/", 1)[0].strip()
    )

    class _StubLLM:
        def chat(self, user_message, system_prompt="", **kw):
            return f"```json\n{json.dumps(_CORRECT)}\n```"

        def complete(self, messages, **kw):
            from src.llm.interface import LLMResponse
            return LLMResponse(content="package Repair {}", model="stub")

    orch = Orchestrator(_StubLLM(), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", [requirement])
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="gen", metadata={}),
        build_lite_model(_BASE, model_name="DeliveryUAV"),
    )
    merged = orch._apply_ag_contract_layer(_BASE, [requirement])
    orch._commit_terminal_model(merged, producer="llm-decided")
    assurance = orch._build_collaboration_artifacts(merged)

    collaboration = assurance["collaboration"]
    assert collaboration["blackboard"]                      # pillar 2
    assert assurance["pattern_conformance_report"]          # pillar 1
    assert assurance["ag_contract_graph"]
    assert assurance["revised_experiment"]["r2_generation_mode"] == (
        "LLM_DECIDED_SPEC"
    )

    result = dict(assurance)
    result["model_sysml"] = merged
    result["requirements"] = [requirement]
    written = write_revised_run_artifacts(result, tmp_path)
    for artifact in ("blackboard_snapshot", "pattern_conformance_report",
                     "requirement_traceability", "coordination_metrics"):
        assert artifact in written

    trace = json.loads(
        (tmp_path / "requirement_traceability.json").read_text()
    )                                                       # pillar 3
    assert trace["declared_requirements"] == ["REQ_SAFE_005"]
    assert trace["traceability"]["fully_traced"] == 1
    assert trace["pattern_conformance"]["verdict"] == "PASS"


def test_the_prompt_states_every_field_the_validator_can_demand():
    """Six times this session a validator demanded something the generator was
    never told. Here it cost a whole 3x3 pilot: all three R2 runs failed closed
    with "STARTUP_INHIBIT must state at least one invariant" because the decision
    prompt listed no invariants key at all. The previous pilot ran only the timed
    chain, which needs none, so nothing surfaced it.
    """
    from src.prototyping.ag_chains import REQ_SAFE_008_CHAIN
    from src.prototyping.ag_decision import (
        INVARIANT_SOURCE_KINDS,
        KNOWN_PATTERNS,
    )

    captured = {}

    class _Capture:
        def chat(self, prompt, system_prompt=None):
            captured["text"] = f"{system_prompt or ''}\n{prompt}"
            return json.dumps(_CORRECT)

    orch = Orchestrator(_Capture(), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    try:
        orch._generate_llm_decided_ag_spec(REQ_SAFE_008_CHAIN, _BASE)
    except RuntimeError:
        pass  # the stub answers with the wrong chain's decisions; the prompt is
              # what is under test
    text = captured["text"]

    for field in ("safety_pattern", "timing_origin", "deadline_seconds",
                  "observation", "system_assumptions", "components",
                  "discharged_by", "priority", "invariants",
                  "invariant_id", "antecedent", "consequent", "source_kind"):
        assert field in text, f"the validator can demand {field!r}; state it"
    for pattern in KNOWN_PATTERNS:
        assert pattern in text
    for kind in INVARIANT_SOURCE_KINDS:
        assert kind in text
    # and the exclusivity the validator enforces must be stated, not discovered
    assert "no deadline" in text and "no invariants" in text
