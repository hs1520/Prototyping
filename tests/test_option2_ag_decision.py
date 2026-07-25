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

import copy
import itertools
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


def _decision_prompt_text() -> str:
    """The full prompt the decided mode puts in front of the model."""
    from src.prototyping.ag_chains import REQ_SAFE_008_CHAIN

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
    return captured["text"]


def test_the_prompt_states_every_field_the_validator_can_demand():
    """Six times this session a validator demanded something the generator was
    never told. Here it cost a whole 3x3 pilot: all three R2 runs failed closed
    with "STARTUP_INHIBIT must state at least one invariant" because the decision
    prompt listed no invariants key at all. The previous pilot ran only the timed
    chain, which needs none, so nothing surfaced it.
    """
    from src.prototyping.ag_decision import (
        INVARIANT_SOURCE_KINDS,
        KNOWN_PATTERNS,
    )

    text = _decision_prompt_text()

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


def test_the_prompt_states_the_semantic_obligations_not_only_the_field_names():
    """A field name in the schema is not the obligation attached to it.

    The pilot proved the granularity gap: the prompt listed `invariants`, so the
    field-name guardrail above passed — while the checker separately demanded that
    those invariants FILL THE PATTERN'S ROLES, which nothing had ever said. Both
    invariant chains failed INVARIANT_SEMANTICS_INVALID with well-formed invariants
    that simply did not cover the roles. The same held for `observation`: named in
    the schema, but its entailment obligation unstated, and REQ_SAFE_004 answered
    with an invented concept no component produces (DECOMPOSITION_INSUFFICIENT).

    So this asserts against the checker's own tables, not a hand-list: a role added
    to `PATTERN_INVARIANT_ROLES` fails here until the prompt states it.
    """
    from src.prototyping.ag_contracts import PATTERN_INVARIANT_ROLES
    from src.prototyping.ag_convention import (
        DECISION_FIELD_OBLIGATIONS,
        render_decision_field_rules,
        render_invariant_role_rules,
    )

    text = _decision_prompt_text()

    for pattern, roles in PATTERN_INVARIANT_ROLES.items():
        for role in roles:
            assert role in text, (
                f"the checker refuses {pattern} when the {role!r} role is "
                "unfilled; the prompt never says the role exists"
            )
    # single-sourced from ag_convention, so the rule cannot drift between the
    # decided prompt and the authored-SysML one
    assert render_invariant_role_rules() in text
    # the entailment obligation behind `observation`, likewise stated
    assert "PRODUCES" in text
    # and the timed pattern's field obligations, which cost a measured seed its
    # timed chain: `timing_segment_required` was offered as a bare true/false
    assert render_decision_field_rules() in text
    for field, _rule in DECISION_FIELD_OBLIGATIONS:
        assert field in text


def test_all_three_safety_patterns_reach_pass_under_the_decided_mode():
    """The decided mode covers every encoded pattern, not only the timed one.

    Two things had to exist for the invariant patterns to be assemblable at all:
    the lock lifecycle is synthesised from the declared invariants, and the author
    declares which consumed concepts are typed lifecycle EVENTS rather than
    assumptions. Without the second, every consumed concept became an assumption
    and a default-safe mechanism could not be built - a component that assumes its
    power-on event is not safe by default, it is safe once that event happens to
    have occurred.
    """
    from src.prototyping import ag_chains

    def _terms(node):
        if node.get("node") == "Identifier":
            return [{"concept": node["name"]}]
        if node.get("node") == "Not":
            return [{"concept": node["expr"]["name"], "negated": True}]
        out = []
        for item in node.get("operands", ()):
            out += _terms(item)
        return out

    for name in ("REQ_SAFE_004_CHAIN", "REQ_SAFE_008_CHAIN"):
        chain = getattr(ag_chains, name)
        boundary = build_architecture_boundary_draft(chain)
        produced = {
            concept: item["component_id"]
            for item in boundary["components"]
            for concept in item["interfaces"]["produces"]
        }
        events = {c.name: set(c.interface_inputs) for c in chain.components}
        decisions = {
            "safety_pattern": chain.pattern,
            "timing_origin": None, "deadline_seconds": None,
            "observation": chain.observation,
            "system_assumptions": list(chain.system_assumptions),
            "priority": None,
            "components": [
                {
                    "component_id": item["component_id"],
                    "lifecycle_events": sorted(events.get(item["component_id"], ())),
                    "assumptions": [
                        {"concept": concept,
                         "discharged_by": (
                             produced.get(concept)
                             if produced.get(concept) not in (
                                 None, item["component_id"]) else None)}
                        for concept in item["interfaces"]["consumes"]
                    ],
                }
                for item in boundary["components"]
            ],
            "invariants": [
                {"invariant_id": inv.invariant_id,
                 "antecedent": _terms(inv.trigger_or_antecedent_ast),
                 "consequent": _terms(inv.required_consequent_ast),
                 "source_kind": inv.source_kind, "source_id": inv.source_id}
                for inv in chain.invariants
            ],
        }
        assert _verdict(chain, decisions) == ("PASS", []), (
            chain.source_requirement, _verdict(chain, decisions)
        )


def _verdict(chain, decisions):
    """The verdict the *runner* would reach: raw syntax gate, then the A/G check.

    The gate is included because the runner applies it and these tests did not: a
    measured seed failed closed on `1 parser error` from a package every A/G-level
    test here called PASS. An extractor that tolerates a malformed line is not
    evidence that the parser does.
    """
    from src.prototyping.ag_emitter import emit_ag_package
    from src.simulation.syntax_checker import check_syntax

    spec = build_spec_from_decisions(
        decisions, build_architecture_boundary_draft(chain)
    )
    emitted = emit_ag_package(spec)
    gate = check_syntax(emitted, fail_closed=True, filter_stdlib_diagnostics=True)
    if gate.has_errors:
        return "SYNTAX", [
            str(item)[:160] for item in (*gate.parser_errors, *gate.sema_errors)[:3]
        ]
    base = (
        f"package X {{ requirement def {chain.source_requirement} "
        "{ doc /* t */ } }"
    )
    report = check_ag_graph(extract_ag_graph(base + "\n\n" + emitted))
    return report.verdict, [item.code for item in report.errors()]


def _decisions_from_the_published_rules():
    """Decision sets an author could write from the published rules alone.

    Every concept the rules leave free is spelled differently from the reviewed
    chain — the forbidden states, the reset event, the unlocked and power
    concepts. Only the role-carrying concepts are the boundary's, because the
    published rule says the latch/locked concept must be one the architecture
    produces.
    """
    from src.prototyping import ag_chains

    derived = "STUDENT_DERIVED_DESIGN_CONSTRAINT"
    startup_inhibit = {
        "safety_pattern": "STARTUP_INHIBIT",
        "timing_origin": None, "deadline_seconds": None,
        "observation": (
            "armingTransitionInhibited and airborneTransitionInhibited"
        ),
        "system_assumptions": ["powerOnSelfTestActive", "sensorFailureReported"],
        "priority": None,
        "components": [
            {"component_id": "SelfTestStatusLatchContract",
             "lifecycle_events": [],
             "assumptions": [
                 {"concept": "powerOnSelfTestActive", "discharged_by": None},
                 {"concept": "sensorFailureReported", "discharged_by": None}]},
            {"component_id": "ArmingAuthorityContract", "lifecycle_events": [],
             "assumptions": [{"concept": "startupInhibitActive",
                              "discharged_by": "SelfTestStatusLatchContract"}]},
            {"component_id": "FlightModeAuthorityContract",
             "lifecycle_events": [],
             "assumptions": [{"concept": "startupInhibitActive",
                              "discharged_by": "SelfTestStatusLatchContract"}]},
        ],
        "invariants": [
            # (a) the forbidden role — this author's own names for the states
            {"invariant_id": "INV_NoArmOnFailedSelfTest",
             "antecedent": [{"concept": "powerOnSelfTestActive"},
                            {"concept": "sensorFailureReported"}],
             "consequent": [{"concept": "vehicleArmed", "negated": True},
                            {"concept": "vehicleAirborne", "negated": True}],
             "source_kind": "STAKEHOLDER"},
            # (b) the latch and what it inhibits
            {"invariant_id": "INV_LatchEffect",
             "antecedent": [{"concept": "startupInhibitActive"}],
             "consequent": [{"concept": "armingTransitionInhibited"},
                            {"concept": "airborneTransitionInhibited"}],
             "source_kind": derived},
            # (c) the reset event that clears it
            {"invariant_id": "INV_LatchClearedOnPass",
             "antecedent": [{"concept": "selfTestCompletedWithoutFault"}],
             "consequent": [{"concept": "startupInhibitActive", "negated": True}],
             "source_kind": derived},
        ],
    }
    locked_release = {
        "safety_pattern": "LOCKED_UNTIL_AUTHORISED_RELEASE",
        "timing_origin": None, "deadline_seconds": None,
        "observation": "payloadLocked",
        "system_assumptions": ["receivedReleaseCommand", "authorisationDataValid"],
        "priority": None,
        "components": [
            {"component_id": "ReleaseCommandGatewayContract",
             "lifecycle_events": [],
             "assumptions": [
                 {"concept": "receivedReleaseCommand", "discharged_by": None},
                 {"concept": "authorisationDataValid", "discharged_by": None}]},
            # safe by DEFAULT: every consumed concept is an event, so it assumes
            # nothing — the published rule, not this chain's encoding
            {"component_id": "PayloadLockMechanismContract",
             "lifecycle_events": ["powerOnEvent", "powerLostEvent",
                                  "authorisedReleaseCommandReceived"],
             "assumptions": [
                 {"concept": "powerOnEvent", "discharged_by": None},
                 {"concept": "powerLostEvent", "discharged_by": None},
                 {"concept": "authorisedReleaseCommandReceived",
                  "discharged_by": "ReleaseCommandGatewayContract"}]},
        ],
        "invariants": [
            {"invariant_id": "INV_DefaultLockedAtPowerOn",
             "antecedent": [{"concept": "powerOnEvent"}],
             "consequent": [{"concept": "payloadLocked"}],
             "source_kind": "STAKEHOLDER"},
            {"invariant_id": "INV_ReleaseNeedsAuthorisation",
             "antecedent": [{"concept": "payloadReleased"}],
             "consequent": [{"concept": "authorisedReleaseCommandReceived"}],
             "source_kind": derived},
            {"invariant_id": "INV_DeenergiseToLock",
             "antecedent": [{"concept": "lockActuatorPowered", "negated": True}],
             "consequent": [{"concept": "payloadLocked"}],
             "source_kind": derived},
        ],
    }

    return (
        (ag_chains.REQ_SAFE_004_CHAIN, startup_inhibit),
        (ag_chains.REQ_SAFE_008_CHAIN, locked_release),
    )


def test_an_author_following_only_the_published_roles_reaches_pass():
    """Sufficiency of the *rules*, not reproduction of the *answer*.

    The test above feeds the reviewed chains' own invariants back in, so it cannot
    distinguish "the rules are enough" from "the answer was copied". Both chains
    failed INVARIANT_SEMANTICS_INVALID on the measured pilot; if stating the roles
    is the fix, they pass here without either chain's reviewed invariant set.
    """
    for chain, decisions in _decisions_from_the_published_rules():
        assert _verdict(chain, decisions) == ("PASS", []), (
            chain.source_requirement, _verdict(chain, decisions)
        )
        # and none of it may be the reviewed chain's own invariant identity
        reviewed = {
            name
            for item in chain.invariants
            for name in (item.invariant_id, item.source_id)
        }
        assert not reviewed & {
            item["invariant_id"] for item in decisions["invariants"]
        }


def test_an_observation_that_is_an_expression_still_emits_parseable_sysml():
    """The decided mode accepts a Boolean expression as the observation — an
    invariant pattern's system guarantee often is one — but the emitter declares
    one `attribute <concept> : Boolean;` per observation concept. Handed the whole
    expression it wrote `attribute a and b : Boolean;`, which does not parse.

    A measured seed died on exactly that, one parser error, after every A/G-level
    test in this file called the same package PASS.
    """
    from src.prototyping import ag_chains

    for chain, decisions in _decisions_from_the_published_rules():
        spec = build_spec_from_decisions(
            decisions, build_architecture_boundary_draft(chain)
        )
        for concept in spec.system_observation_concepts:
            assert " " not in concept, concept
        assert set(spec.system_observation_concepts) <= set(
            spec.selected_model_elements
        )
    # and the case that actually broke: a two-concept observation
    chain, decisions = _decisions_from_the_published_rules()[0]
    assert " and " in decisions["observation"], "this case must stay conjunctive"
    assert _verdict(chain, decisions) == ("PASS", [])
    assert chain is ag_chains.REQ_SAFE_004_CHAIN


def test_the_timing_segment_field_decides_the_timed_chain():
    """`timing_segment_required` was offered to the author as a bare `true/false`.

    A measured seed set it true for the component whose guarantee is simply
    available at the boundary — a supply that is already on, not something that
    gets triggered — and lost the whole timed chain to four diagnostics at once:
    the budgets no longer fitted the deadline, the emitter built a trigger-response
    machine with no trigger to give it, and the arbitration obligation that names
    that component went unsatisfied.

    Nothing had told the author what the field means. This pins both directions:
    the violation reproduces those diagnostics, and the published rule resolves
    them with every other decision held identical.
    """
    from src.prototyping import ag_chains

    chain = ag_chains.REQ_SAFE_005_CHAIN
    boundary = build_architecture_boundary_draft(chain)
    produced = {
        concept: item["component_id"]
        for item in boundary["components"]
        for concept in item["interfaces"]["produces"]
    }
    budgets = {
        "SafetyResponseArbiterContract": 0.2, "RecoverySystemContract": 0.3,
    }

    def _decisions(segment, budget):
        return {
            "safety_pattern": "TRIGGERED_TIMED_FAILSAFE_RESPONSE",
            "timing_origin": chain.timing_origin,
            "deadline_seconds": 0.5,
            "observation": chain.observation,
            "system_assumptions": list(chain.system_assumptions),
            "invariants": [],
            "priority": {
                "response_set_id": chain.priority.response_set_id,
                "members": list(chain.priority.members),
                "selected_response": chain.priority.selected_response,
            },
            "components": [
                {
                    "component_id": item["component_id"],
                    "timing_segment_required": (
                        segment if "RecoveryPowerSupply" in item["component_id"]
                        else True
                    ),
                    "latency_budget_seconds": (
                        budget if "RecoveryPowerSupply" in item["component_id"]
                        else budgets[item["component_id"]]
                    ),
                    "lifecycle_events": [],
                    "assumptions": [
                        {"concept": concept,
                         "discharged_by": (
                             produced.get(concept)
                             if produced.get(concept) not in (
                                 None, item["component_id"]) else None)}
                        for concept in item["interfaces"]["consumes"]
                    ],
                }
                for item in boundary["components"]
            ],
        }

    verdict, codes = _verdict(chain, _decisions(True, 0.2))
    assert verdict == "FAIL"
    assert {"TIMING_BUDGET_EXCEEDED", "REALIZATION_TRIGGER_MISSING",
            "PRIORITY_TOPOLOGY_INCOMPLETE"} <= set(codes)
    # the only change is the two fields the rule now explains
    assert _verdict(chain, _decisions(False, None)) == ("PASS", [])


def test_the_release_rule_may_be_phrased_over_the_complement_of_locked():
    """`not <locked> => <authorisation>` states the release obligation exactly as
    `<unlocked> => <authorisation>` does — the author simply did not introduce a
    second name for the complement of a concept.

    A measured seed wrote it that way and was failed for it: the derivation read
    every negated antecedent as de-energise-to-lock, so the unlocked and
    authorisation roles came out empty and the model was rejected on phrasing
    rather than on what it asserts. That is the same defect as rejecting an
    element name, which this checker closed for names in ag-bounded-5.
    """
    from src.prototyping.ag_contracts import (
        PATTERN_INVARIANT_ROLES, _invariant_roles,
    )
    from src.prototyping.ag_emitter import emit_ag_package

    chain, decisions = _decisions_from_the_published_rules()[1]
    complement = copy.deepcopy(decisions)
    release = next(
        item for item in complement["invariants"]
        if item["consequent"][0]["concept"] == "authorisedReleaseCommandReceived"
    )
    release["antecedent"] = [{"concept": "payloadLocked", "negated": True}]

    spec = build_spec_from_decisions(
        complement, build_architecture_boundary_draft(chain)
    )
    roles = _invariant_roles(extract_ag_graph(emit_ag_package(spec)))
    unfilled = [
        role for role in PATTERN_INVARIANT_ROLES["LOCKED_UNTIL_AUTHORISED_RELEASE"]
        if not roles.get(role)
    ]
    assert not unfilled, (unfilled, roles)
    assert _verdict(chain, complement) == ("PASS", [])

    # and the phrasing must not become a way to skip the obligation: with the
    # release invariant gone the roles must go unfilled again
    without = copy.deepcopy(complement)
    without["invariants"] = [
        item for item in without["invariants"] if item is not release
        and item["consequent"][0]["concept"] != "authorisedReleaseCommandReceived"
    ]
    verdict, codes = _verdict(chain, without)
    assert verdict != "PASS" and "INVARIANT_SEMANTICS_INVALID" in codes


def test_dropping_any_one_published_obligation_is_still_detected():
    """Publishing the roles must not cost detection strength.

    The rules now say each invariant pattern needs three invariants, so omitting
    any one of them must be caught — including the power-on default, which was NOT
    caught until `power_on` became a required role: locked and power were both
    filled by the de-energise invariant, so a model that never said what the system
    powers up into passed.
    """
    for chain, decisions in _decisions_from_the_published_rules():
        for index in range(len(decisions["invariants"])):
            thinned = copy.deepcopy(decisions)
            dropped = thinned["invariants"].pop(index)["invariant_id"]
            verdict, codes = _verdict(chain, thinned)
            assert verdict != "PASS", (
                f"{chain.source_requirement} passed without {dropped}"
            )
            assert "INVARIANT_SEMANTICS_INVALID" in codes


def test_the_roles_do_not_depend_on_the_order_the_invariants_are_listed_in():
    """Declaration order is not a property of the pattern, and an author has no
    way to know a hidden one. The locked-release roles were read in list order
    until this was pinned, so the same three invariants passed or failed depending
    on which the author wrote first."""
    for chain, decisions in _decisions_from_the_published_rules():
        for order in itertools.permutations(range(len(decisions["invariants"]))):
            reordered = copy.deepcopy(decisions)
            reordered["invariants"] = [
                decisions["invariants"][index] for index in order
            ]
            assert _verdict(chain, reordered) == ("PASS", []), (
                chain.source_requirement, order, _verdict(chain, reordered)
            )


def test_a_declared_lifecycle_event_is_not_an_assumption():
    """The distinction is what makes a default-safe component assemblable: the
    boundary merges both into one `consumes` list, so which is which is the
    author's judgement."""
    from src.prototyping import ag_chains

    chain = ag_chains.REQ_SAFE_008_CHAIN
    boundary = build_architecture_boundary_draft(chain)
    mechanism = next(
        item for item in boundary["components"] if "Mechanism" in item["component_id"]
    )
    consumed = list(mechanism["interfaces"]["consumes"])

    def _spec(events):
        decisions = json.loads(json.dumps({
            "safety_pattern": chain.pattern, "observation": chain.observation,
            "system_assumptions": [], "priority": None,
            "components": [
                {"component_id": item["component_id"],
                 "lifecycle_events": events if item is mechanism else [],
                 "assumptions": [
                     {"concept": c, "discharged_by": None}
                     for c in item["interfaces"]["consumes"]]}
                for item in boundary["components"]
            ],
            "invariants": [
                {"invariant_id": "I1", "antecedent": [{"concept": "powerOnEvent"}],
                 "consequent": [{"concept": "payloadLocked"}],
                 "source_kind": "STAKEHOLDER"},
            ],
        }))
        return build_spec_from_decisions(decisions, boundary)

    without = next(
        c for c in _spec([]).components if "Mechanism" in c.name
    )
    with_events = next(
        c for c in _spec(consumed).components if "Mechanism" in c.name
    )
    assert len(without.assumptions) == len(consumed)
    assert with_events.assumptions == ()
