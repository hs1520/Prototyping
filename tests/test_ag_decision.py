"""LLM-decided A/G specs rendered deterministically (R2 mode LLM_DECIDED_SPEC).

Nine seeds of the LLM-authored-SysML mode agreed with frozen gold on guarantee
allocation 6/6 and assumption discharge 4/6 without ever reaching PASS: the
engineering was right and the notation wrong. Here the model returns decisions
and the emitter renders them, so conformance holds by construction and only the
decisions are judged. A wrong decision still yields a well-formed model that
scores badly, not one that looks correct.
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
    ArchitectureInputRequired,
    DecisionError,
    DecisionFailureDisposition,
    build_spec_from_decisions,
    classify_decision_failure,
    extract_decisions,
    extract_runtime_response_catalog,
    validate_decisions,
)
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_evaluation import evaluate_ag_against_gold
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.architecture_boundary import build_architecture_boundary_draft
from tests.ag_fixtures import REQ_SAFE_005_GOLD

_BASE = (
    "package Src { requirement def REQ_SAFE_005 { doc /* deploy the ballistic "
    "recovery parachute within 0.5 s of a critical propulsion failure. */ } }"
)
_BASE_WITH_RESPONSE_CATALOG = _BASE + """
package ExistingSafetyBehavior {
    action def deployParachute;
    action def inhibitArming;
    state def SafetyArbiter {
        state nominal;
        state parachuteMode {
            entry action onParachute : deployParachute;
        }
        state inhibitMode {
            entry action onInhibit : inhibitArming;
        }
    }
}
"""
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
_CATALOG_CORRECT = copy.deepcopy(_CORRECT)
_CATALOG_CORRECT["priority"] = {
    "response_set_id": "EXISTING_SAFETY_RESPONSES",
    "members": ["deployParachute", "inhibitArming"],
    "selected_response": "deployParachute",
}


def _render(decisions) -> str:
    spec = build_spec_from_decisions(decisions, _BOUNDARY)
    return _BASE.rstrip() + "\n\n" + emit_ag_package(spec) + "\n"


class _DecisionLLM:
    def __init__(self, payload):
        self._payload = payload

    def chat(self, prompt, system_prompt=None):
        return f"```json\n{json.dumps(self._payload)}\n```"

    def complete(self, messages, **kw):
        from src.llm.interface import LLMResponse
        return LLMResponse(
            content=f"```json\n{json.dumps(self._payload)}\n```", model="stub"
        )


def test_decisions_render_passing_model():
    report = check_ag_graph(extract_ag_graph(_render(_CORRECT)))
    assert report.verdict == "PASS"
    assert not report.errors()


def test_one_selected_response_id():
    emitted = emit_ag_package(build_spec_from_decisions(_CORRECT, _BOUNDARY))

    assert "enum PARACHUTE_DEPLOYMENT;" in emitted
    assert "then PARACHUTE_DEPLOYMENT;" in emitted
    assert "state PARACHUTE_DEPLOYMENT {" in emitted
    assert "parachuteDeploymentSelected" not in emitted


def test_non_gold_set_passes():
    """The measured runs produced a two-member response set because the requirement
    never names the others. That is a wrong answer, not a malformed one, so the
    runtime verdict accepts it and the evaluator marks it.
    """
    assert _CORRECT["priority"]["members"] == [
        "PARACHUTE_DEPLOYMENT", "OTHER_RESPONSE"
    ], "fixture must use the non-gold response set"
    assert check_ag_graph(extract_ag_graph(_render(_CORRECT))).verdict == "PASS"


def test_wrong_discharge_scores_low():
    wrong = json.loads(json.dumps(_CORRECT))
    for component in wrong["components"]:
        if component["component_id"] == "RecoverySystemContract":
            for assumption in component["assumptions"]:
                assumption["discharged_by"] = None
    prediction = check_ag_graph(extract_ag_graph(_render(wrong))).to_dict()
    out = evaluate_ag_against_gold(prediction, REQ_SAFE_005_GOLD)
    assert out["assumption_discharge"]["f1"] < 1.0
    assert out["assumption_discharge"]["recall"] < 1.0


def test_part_name_resolved():
    """The boundary shows a component as Contract/part/Def and a model asked to name
    one reaches for the part; the first live run failed closed on that. Naming the
    same component differently is not extending the architecture.
    """
    by_part = json.loads(json.dumps(_CORRECT))
    by_part["components"][0]["component_id"] = "safetyResponseArbiter"
    by_part["components"][2]["assumptions"][0]["discharged_by"] = (
        "safetyResponseArbiter"
    )
    report = check_ag_graph(extract_ag_graph(_render(by_part)))
    assert report.verdict == "PASS"
    spec = build_spec_from_decisions(by_part, _BOUNDARY)
    assert [item.name for item in spec.components] == [
        item.name for item in build_spec_from_decisions(_CORRECT, _BOUNDARY).components
    ]


def test_no_extending_architecture():
    invented = json.loads(json.dumps(_CORRECT))
    invented["components"].append(
        {"component_id": "MadeUpContract", "assumptions": []}
    )
    with pytest.raises(DecisionError, match="unknown component"):
        validate_decisions(invented, _BOUNDARY)


def test_all_components_decided():
    partial = json.loads(json.dumps(_CORRECT))
    partial["components"] = partial["components"][:2]
    with pytest.raises(DecisionError, match="must be decided"):
        validate_decisions(partial, _BOUNDARY)


def test_self_discharge_refused():
    looped = json.loads(json.dumps(_CORRECT))
    looped["components"][2]["assumptions"][0]["discharged_by"] = (
        "RecoverySystemContract"
    )
    with pytest.raises(DecisionError, match="discharges itself"):
        validate_decisions(looped, _BOUNDARY)


def test_unknown_pattern_refused():
    bogus = json.loads(json.dumps(_CORRECT))
    bogus["safety_pattern"] = "MADE_UP_PATTERN"
    with pytest.raises(DecisionError, match="safety_pattern must be one of"):
        validate_decisions(bogus, _BOUNDARY)


def test_missing_deadline_refused():
    undated = json.loads(json.dumps(_CORRECT))
    undated["deadline_seconds"] = None
    with pytest.raises(DecisionError, match="needs deadline_seconds"):
        validate_decisions(undated, _BOUNDARY)


def test_response_not_in_members():
    inconsistent = json.loads(json.dumps(_CORRECT))
    inconsistent["priority"]["selected_response"] = "NOT_A_MEMBER"
    with pytest.raises(DecisionError, match="not in members"):
        validate_decisions(inconsistent, _BOUNDARY)


def test_catalog_from_arbiter_actions():
    catalog = extract_runtime_response_catalog(_BASE_WITH_RESPONSE_CATALOG)
    assert catalog["source"] == "COMMITTED_SYSML_BEHAVIOR"
    assert {
        item["response_id"] for item in catalog["entries"]
    } == {"deployParachute", "inhibitArming"}
    assert all(
        item["source_kind"] == "EXISTING_MODEL_BEHAVIOR"
        and item["source_id"].startswith("SafetyArbiter.")
        for item in catalog["entries"]
    )


def test_members_match_catalog():
    boundary = copy.deepcopy(_BOUNDARY)
    boundary["response_catalog"] = extract_runtime_response_catalog(
        _BASE_WITH_RESPONSE_CATALOG
    )
    # Interface concepts are not selectable responses, though the v5 diagnostic
    # selected one to satisfy cardinality.
    with pytest.raises(DecisionError, match="provenance-backed"):
        validate_decisions(_CORRECT, boundary)
    validate_decisions(_CATALOG_CORRECT, boundary)


def test_extract_fenced_decisions():
    payload = extract_decisions('noise\n```json\n{"a": {"b": 1}}\n```\ntrailing')
    assert payload == {"a": {"b": 1}}


def test_no_json_fails_closed():
    with pytest.raises(DecisionError, match="no JSON object"):
        extract_decisions("I cannot help with that.")


def test_decided_mode_renders_package():
    orch = Orchestrator(_DecisionLLM(_CATALOG_CORRECT),
                        revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    merged = orch._apply_ag_contract_layer(
        _BASE_WITH_RESPONSE_CATALOG,
        ["REQ-SAFE-005: deploy the parachute within 0.5 s"],
    )
    report = check_ag_graph(extract_ag_graph(merged))
    assert report.verdict == "PASS"
    assert not report.errors()


class _SequenceDecisionLLM:
    """Stands in for a provider and records the turn list of every call.

    ``seen_messages`` makes the multi-turn property assertable: a stateless provider
    only knows what the caller resent, so earlier turns are visible here only if
    they were sent.
    """

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self._calls = 0
        self.seen_messages = []

    def _next(self):
        payload = self._payloads[min(self._calls, len(self._payloads) - 1)]
        self._calls += 1
        return json.dumps(payload)

    def chat(self, prompt, system_prompt=None):
        self.last_prompt = prompt
        return self._next()

    def complete(self, messages, **kw):
        from src.llm.interface import LLMResponse
        self.seen_messages.append(
            [(message.role, message.content) for message in messages]
        )
        self.last_prompt = next(
            (
                message.content for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        return LLMResponse(content=self._next(), model="stub")


def test_singleton_priority_stops():
    """The requirement says "all other responses" but never names them.

    A retry previously invented ``OTHER_RESPONSE`` and made the checker pass, which
    is an unsupported architecture fact rather than better engineering.
    """
    thin = json.loads(json.dumps(_CORRECT))
    thin["priority"]["members"] = ["PARACHUTE_DEPLOYMENT"]
    llm = _SequenceDecisionLLM([thin, _CORRECT])

    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    with pytest.raises(
        ArchitectureInputRequired, match="fewer than two provenance-backed"
    ):
        orch._generate_llm_decided_ag_spec(
            REQ_SAFE_005_CHAIN, _BASE, max_decision_attempts=3
        )
    assert llm._calls == 1


def test_failure_classification():
    missing_fact = DecisionError("priority.members needs at least two responses")
    repairable = DecisionError(
        "priority.selected_response 'X' is not in members"
    )
    assert classify_decision_failure(missing_fact) is (
        DecisionFailureDisposition.NEEDS_ARCHITECTURE_INPUT
    )
    assert classify_decision_failure(repairable) is (
        DecisionFailureDisposition.RETRYABLE_VALIDATION_ERROR
    )


def test_retryable_error_feedback():
    inconsistent = json.loads(json.dumps(_CATALOG_CORRECT))
    inconsistent["priority"]["selected_response"] = "NOT_A_MEMBER"
    llm = _SequenceDecisionLLM([inconsistent, _CATALOG_CORRECT])
    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    decided = orch._generate_llm_decided_ag_spec(
        REQ_SAFE_005_CHAIN, _BASE_WITH_RESPONSE_CATALOG,
        max_decision_attempts=2,
    )
    assert emit_ag_package(decided)
    assert llm._calls == 2
    assert "not in members" in llm.last_prompt


def test_retry_resends_rejected_answer():
    """§2 reasoning continuity, as a property rather than a claim.

    The provider is stateless, so the second turn sees the first only if the caller
    resent it; otherwise "keep everything that was already valid" refers to a
    document the model cannot read.
    """
    inconsistent = json.loads(json.dumps(_CATALOG_CORRECT))
    inconsistent["priority"]["selected_response"] = "NOT_A_MEMBER"
    llm = _SequenceDecisionLLM([inconsistent, _CATALOG_CORRECT])
    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    orch._generate_llm_decided_ag_spec(
        REQ_SAFE_005_CHAIN, _BASE_WITH_RESPONSE_CATALOG,
        max_decision_attempts=2,
    )

    first, second = llm.seen_messages
    assert [role for role, _ in first] == ["system", "user"]
    assert ("assistant", json.dumps(inconsistent)) in second
    assert [role for role, _ in second] == [
        "system", "user", "assistant", "user"
    ]
    assert "Approved architecture" not in second[-1][1]
    assert "timing_segment_required" not in second[-1][1]
    assert "not in members" in second[-1][1]


def test_turn_archived_once():
    """Resending history does not re-archive it.

    Without the offset a 3-turn conversation records 1+2+3 turns, inflating the
    transcript digest and the §13 session-growth metric it feeds.
    """
    from src.llm.interface import Conversation, LLMInterface, LLMResponse

    class _Echo(LLMInterface):
        def _complete_impl(self, messages, temperature, max_tokens):
            return LLMResponse(content=f"reply-{len(messages)}", model="stub")

    archived = []
    llm = _Echo()
    llm.add_call_observer(lambda event: archived.extend(
        [
            (m["role"], m["content"])
            for m in event["messages"][int(event["new_message_offset"]):]
        ]
        + [("assistant", event["response"]["content"])]
    ))

    conversation = Conversation(llm, system_prompt="sys")
    conversation.send("first")
    conversation.send("second")
    conversation.send("third")

    assert [content for role, content in archived if role == "user"] == [
        "first", "second", "third"
    ], "each user turn archived exactly once"
    assert sum(1 for role, _ in archived if role == "system") == 1
    assert conversation.assistant_turn_count == 3
    # the stub echoes how many messages it received, so the history grew at the
    # provider: 2 -> 4 -> 6, not 2 -> 2 -> 2
    assert [content for role, content in archived if role == "assistant"] == [
        "reply-2", "reply-4", "reply-6"
    ]


def test_turn_charged_once():
    """A real probe run died here, so the rule is pinned.

    The provider's ``prompt_tokens`` covers the whole resent history, so charging it
    per call grows the session budget quadratically against a linear transcript and
    measures resends rather than accumulated context. Cumulative cost is the
    TokenLedger's job.
    """

    from src.agents.orchestrator import Orchestrator

    charged = []

    class _Session:
        def append(self, role, content, *, token_count=0, **_stamp):
            charged.append((role, token_count))

    orch = Orchestrator.__new__(Orchestrator)
    orch.blackboard = None
    session = _Session()

    reply = "y" * 400
    for turn, offset in enumerate((0, 3, 5)):
        history = [
            {"role": "system", "content": "s" * 400},
            *[
                item
                for index in range(turn)
                for item in (
                    {"role": "user", "content": "u" * 400},
                    {"role": "assistant", "content": reply},
                )
            ],
            {"role": "user", "content": "u" * 400},
        ]
        orch._archive_provider_call(session, {
            "messages": history,
            "new_message_offset": offset,
            # the resent history is already counted in prompt_tokens, so it is not
            # charged to the session again
            "response": {
                "content": reply,
                "prompt_tokens": 1000 * (turn + 1) ** 2,
                "completion_tokens": 100,
            },
        })

    assert [role for role, _ in charged] == [
        "system", "user", "assistant",
        "user", "assistant",
        "user", "assistant",
    ]
    assert [count for _, count in charged] == [100, 100, 100, 100, 100, 100, 100]


def test_incoherent_decisions_fail_closed():
    inconsistent = json.loads(json.dumps(_CATALOG_CORRECT))
    inconsistent["priority"]["selected_response"] = "NOT_A_MEMBER"
    orch = Orchestrator(_SequenceDecisionLLM([inconsistent]),
                        revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    with pytest.raises(RuntimeError, match="failed closed after"):
        orch._apply_ag_contract_layer(
            _BASE_WITH_RESPONSE_CATALOG,
            ["REQ-SAFE-005: deploy the parachute within 0.5 s"],
        )


def test_unusable_decisions_fail_closed():
    orch = Orchestrator(_DecisionLLM({"safety_pattern": "NONSENSE"}),
                        revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    with pytest.raises(RuntimeError, match="failed closed"):
        orch._apply_ag_contract_layer(
            _BASE, ["REQ-SAFE-005: deploy the parachute within 0.5 s"]
        )


def test_arm_metadata_reports_mode():
    """The arm metadata hardcoded the deterministic intervention and the pipeline
    dropped the mode, so a run could execute one intervention and record another.
    The pooling gates compare the recorded mode rather than observe behaviour, so
    they would accept the mislabelled evidence, and the suite passed throughout.
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


def test_pipeline_passes_mode():
    from src.app.pipeline import PrototypingPipeline

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
    assert reported["r2_intervention_version"] == (
        "r2-bbag-whole-model-guided-decided-v8"
    )


def _decision_prompt_text() -> str:
    from src.prototyping.ag_chains import REQ_SAFE_008_CHAIN

    captured = {}

    class _Capture:
        def chat(self, prompt, system_prompt=None):
            captured["text"] = f"{system_prompt or ''}\n{prompt}"
            return json.dumps(_CORRECT)

        def complete(self, messages, **kw):
            from src.llm.interface import LLMResponse
            # Capture the opening turn only: later turns no longer repeat the schema
            # (the conversation holds it), so asserting against them would test the
            # wrong thing.
            captured.setdefault(
                "text", "\n".join(message.content for message in messages)
            )
            return LLMResponse(content=json.dumps(_CORRECT), model="stub")

    orch = Orchestrator(_Capture(), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_DECIDED_SPEC")
    try:
        orch._generate_llm_decided_ag_spec(REQ_SAFE_008_CHAIN, _BASE)
    except RuntimeError:
        pass  # the stub answers with the wrong chain's decisions
    return captured["text"]


def test_prompt_states_all_fields():
    """A validator demanding something the generator was never told cost a 3x3 pilot:
    all three R2 runs failed closed with "STARTUP_INHIBIT must state at least one
    invariant" because the decision prompt listed no invariants key at all. The
    previous pilot ran only the timed chain, which needs none, so nothing surfaced
    it.
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
    assert "no deadline" in text and "no invariants" in text


def test_prompt_states_obligations():
    """A field name in the schema is not the obligation attached to it.

    The prompt listed `invariants`, so the field-name guardrail passed while the
    checker separately demanded those invariants fill the pattern's roles, which
    nothing had said: both invariant chains failed INVARIANT_SEMANTICS_INVALID.
    `observation` was the same, its entailment obligation unstated, and REQ_SAFE_004
    answered with an invented concept no component produces
    (DECOMPOSITION_INSUFFICIENT). So this asserts against the checker's own tables:
    a role added to `PATTERN_INVARIANT_ROLES` fails here until the prompt states it.
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
    assert "PRODUCES" in text
    # and the timed pattern's field obligations, which cost a seed its timed chain:
    # `timing_segment_required` was offered as a bare true/false
    assert render_decision_field_rules() in text
    for field, _rule in DECISION_FIELD_OBLIGATIONS:
        assert field in text


def test_all_patterns_reach_pass():
    """The decided mode covers every encoded pattern, not only the timed one.

    Invariant patterns need two things: the lock lifecycle is synthesised from the
    declared invariants, and the author declares which consumed concepts are typed
    lifecycle events. Without the second, every consumed concept became an
    assumption and no default-safe mechanism could be built.
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
    """The verdict the runner would reach: raw syntax gate, then the A/G check.

    The gate is included because the runner applies it and these tests did not: a
    seed failed closed on `1 parser error` from a package every A/G-level test here
    called PASS.
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
    chain. Only the role-carrying concepts come from the boundary, since the
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
            {"invariant_id": "INV_NoArmOnFailedSelfTest",
             "antecedent": [{"concept": "powerOnSelfTestActive"},
                            {"concept": "sensorFailureReported"}],
             "consequent": [{"concept": "vehicleArmed", "negated": True},
                            {"concept": "vehicleAirborne", "negated": True}],
             "source_kind": "STAKEHOLDER"},
            {"invariant_id": "INV_LatchEffect",
             "antecedent": [{"concept": "startupInhibitActive"}],
             "consequent": [{"concept": "armingTransitionInhibited"},
                            {"concept": "airborneTransitionInhibited"}],
             "source_kind": derived},
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
            # safe by default: every consumed concept is an event, so it assumes nothing
            # (the published rule, not this chain's encoding)
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


def test_published_roles_suffice():
    """Tests whether the published rules suffice, not whether the answer is reproduced.

    The test above feeds the reviewed invariants back in, so it cannot separate the
    two. Both chains failed INVARIANT_SEMANTICS_INVALID on the pilot; if stating the
    roles is the fix, they pass here without either reviewed invariant set.
    """
    for chain, decisions in _decisions_from_the_published_rules():
        assert _verdict(chain, decisions) == ("PASS", []), (
            chain.source_requirement, _verdict(chain, decisions)
        )
        reviewed = {
            name
            for item in chain.invariants
            for name in (item.invariant_id, item.source_id)
        }
        assert not reviewed & {
            item["invariant_id"] for item in decisions["invariants"]
        }


def test_expression_observation_parses():
    """A Boolean expression as the observation still has to emit parseable SysML.

    The emitter declares one `attribute <concept> : Boolean;` per observation
    concept; handed a whole expression it wrote `attribute a and b : Boolean;`,
    which does not parse. A seed died on that one parser error.
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
    chain, decisions = _decisions_from_the_published_rules()[0]
    assert " and " in decisions["observation"], "this case must stay conjunctive"
    assert _verdict(chain, decisions) == ("PASS", [])
    assert chain is ag_chains.REQ_SAFE_004_CHAIN


def test_timing_segment_field():
    """`timing_segment_required` was offered to the author as a bare `true/false`.

    A seed set it true for a component whose guarantee is available at the boundary
    and lost the timed chain to four diagnostics: budgets over deadline, a
    trigger-response machine with no trigger, and an unsatisfied arbitration
    obligation. Pins both directions - the violation reproduces the diagnostics, the
    published rule resolves them with every other decision held identical.
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
    assert _verdict(chain, _decisions(False, None)) == ("PASS", [])


def test_release_rule_over_complement():
    """`not <locked> => <authorisation>` states the same release obligation as
    `<unlocked> => <authorisation>`, without a second name for the complement.

    A seed wrote it that way and failed: the derivation read every negated
    antecedent as de-energise-to-lock, so the unlocked and authorisation roles came
    out empty and the model was rejected on phrasing. Same defect as rejecting an
    element name, closed for names in ag-bounded-5.
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

    # the phrasing does not skip the obligation: with the release invariant gone
    # the roles go unfilled again
    without = copy.deepcopy(complement)
    without["invariants"] = [
        item for item in without["invariants"] if item is not release
        and item["consequent"][0]["concept"] != "authorisedReleaseCommandReceived"
    ]
    verdict, codes = _verdict(chain, without)
    assert verdict != "PASS" and "INVARIANT_SEMANTICS_INVALID" in codes


def test_dropped_invariant_detected():
    """Publishing the roles does not cost detection strength.

    Each invariant pattern needs three invariants, so omitting any one is caught -
    including the power-on default, which was missed until `power_on` became a
    required role: locked and power were both filled by the de-energise invariant.
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


def test_invariant_order_irrelevant():
    """Declaration order is not a property of the pattern.

    The locked-release roles were read in list order until this was pinned, so the
    same three invariants passed or failed depending on which came first.
    """
    for chain, decisions in _decisions_from_the_published_rules():
        for order in itertools.permutations(range(len(decisions["invariants"]))):
            reordered = copy.deepcopy(decisions)
            reordered["invariants"] = [
                decisions["invariants"][index] for index in order
            ]
            assert _verdict(chain, reordered) == ("PASS", []), (
                chain.source_requirement, order, _verdict(chain, reordered)
            )


def test_lifecycle_event_not_assumption():
    """A default-safe component needs the distinction: the boundary merges both into
    one `consumes` list, so which is which is the author's call.
    """
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
