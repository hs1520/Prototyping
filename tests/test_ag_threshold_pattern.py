"""THRESHOLD_TRIGGERED_RESPONSE - a triggered response that states no deadline.

A requirement naming a trigger, a response and a precedence relation but no time
bound fitted neither family: the timed pattern rejects a chain with no budget,
and stating it as a held invariant misdescribes it. REQ_SAFE_002 is that shape -
the timed failsafe's arbitration topology with the timing obligations removed.
Scope is the dominant direction, where the selected response supersedes the
others (competing transitions guarded by ``not <trigger>``); the subordinate
direction (REQ_SAFE_001) needs the opposite precedence and a guard shape the
extractor does not parse, so it stays uncovered.
"""
from __future__ import annotations

import copy

import pytest

from src.prototyping.ag_assurance import check_safety_pattern_conformance
from src.prototyping.ag_chains import REQ_SAFE_002_CHAIN, REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_convention import render_authoring_rules
from src.prototyping.ag_decision import (
    DecisionError,
    INVARIANT_PATTERNS,
    KNOWN_PATTERNS,
    TRIGGERED_PATTERNS,
    build_spec_from_decisions,
    validate_decisions,
)
from src.prototyping.ag_emitter import AGPrioritySpec, emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.architecture_boundary import build_architecture_boundary_draft
from src.simulation.syntax_checker import check_syntax

_BASE = (
    "package Src { requirement def REQ_SAFE_002 { doc /* perform a controlled "
    "descent when the battery state-of-charge falls below 15%. */ } }"
)
_BOUNDARY = build_architecture_boundary_draft(REQ_SAFE_002_CHAIN)

_DECISIONS = {
    "safety_pattern": "THRESHOLD_TRIGGERED_RESPONSE",
    "observation": "controlledDescentEngaged",
    "system_assumptions": ["airborne", "criticalBatteryThresholdReached"],
    "components": [
        {
            "component_id": "SafetyResponseArbiterContract",
            "timing_segment_required": True,
            "assumptions": [
                {"concept": "airborne", "discharged_by": None},
                {"concept": "criticalBatteryThresholdReached",
                 "discharged_by": None},
            ],
        },
        {
            "component_id": "FlightControlSystemContract",
            "timing_segment_required": True,
            "assumptions": [
                {"concept": "controlledDescentCommand",
                 "discharged_by": "SafetyResponseArbiterContract"},
            ],
        },
    ],
    "priority": {
        "response_set_id": "FLIGHT_RESPONSES_V1",
        "members": ["CONTROLLED_DESCENT", "LOW_BATTERY_RETURN_TO_BASE"],
        "selected_response": "CONTROLLED_DESCENT",
        "trigger": "criticalBatteryThresholdReached",
    },
}


def _render(spec) -> str:
    return _BASE.rstrip() + "\n\n" + emit_ag_package(spec) + "\n"


def _decisions(**overrides):
    payload = copy.deepcopy(_DECISIONS)
    payload.update(overrides)
    return payload


def test_reference_chain_clean_model():
    gate = check_syntax(
        _render(REQ_SAFE_002_CHAIN),
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert not gate.has_errors
    assert not gate.warnings


def test_reference_chain_checker_pass():
    report = check_ag_graph(extract_ag_graph(_render(REQ_SAFE_002_CHAIN)))
    assert report.verdict == "PASS"
    assert not report.errors()


def test_decided_spec_checker_pass():
    spec = build_spec_from_decisions(_DECISIONS, _BOUNDARY)
    assert spec.pattern == "THRESHOLD_TRIGGERED_RESPONSE"
    assert spec.deadline is None
    assert spec.timing_origin is None
    text = _render(spec)
    gate = check_syntax(
        text, fail_closed=True, filter_stdlib_diagnostics=False
    )
    assert not gate.has_errors and not gate.warnings
    assert check_ag_graph(extract_ag_graph(text)).verdict == "PASS"


def test_untimed_trigger_stated_once():
    """No timing origin to cross-check against, so the trigger is not duplicated.

    The timed pattern states its trigger twice - priority contract trigger and
    system contract interval origin - and holds them to agreement. An untimed chain
    declares no interval, so there is nothing to cross-check.
    """
    spec = build_spec_from_decisions(_DECISIONS, _BOUNDARY)
    assert spec.priority.trigger == "criticalBatteryThresholdReached"
    graph = extract_ag_graph(_render(spec))
    assert graph.system is not None
    assert graph.system.timing_budget is None
    assert graph.system.timing_origin is None
    assert not check_ag_graph(graph).errors()


def test_deadline_wrong_pattern():
    with pytest.raises(DecisionError, match="must not carry deadline_seconds"):
        validate_decisions(_decisions(deadline_seconds=0.5), _BOUNDARY)


def test_no_component_budget():
    payload = _decisions()
    payload["components"][0]["latency_budget_seconds"] = 0.1
    with pytest.raises(DecisionError, match="latency_budget_seconds"):
        validate_decisions(payload, _BOUNDARY)


def test_untimed_pattern_no_invariants():
    """A triggered pattern, not an invariant one.

    Before the split, everything but the timed pattern was treated as invariant and
    had to state at least one invariant, which misdescribes a trigger->response
    obligation.
    """
    assert "THRESHOLD_TRIGGERED_RESPONSE" not in INVARIANT_PATTERNS
    assert "THRESHOLD_TRIGGERED_RESPONSE" in TRIGGERED_PATTERNS
    payload = _decisions()
    assert "invariants" not in payload
    validate_decisions(payload, _BOUNDARY)


def test_timed_pattern_requires_deadline():
    payload = _decisions(safety_pattern="TRIGGERED_TIMED_FAILSAFE_RESPONSE")
    with pytest.raises(DecisionError, match="needs deadline_seconds"):
        validate_decisions(payload, _BOUNDARY)


def test_conformance_no_timing_criterion():
    graph = extract_ag_graph(_render(REQ_SAFE_002_CHAIN))
    report = check_ag_graph(graph)
    conformance = check_safety_pattern_conformance(graph, report)
    assert conformance["verdict"] == "PASS"
    cases = conformance["cases"]
    assert cases
    assert {case["pattern"] for case in cases} == {
        "THRESHOLD_TRIGGERED_RESPONSE"
    }
    # Passes with no case claiming a timing criterion; under the timed pattern's
    # extra duty these cases would fail.
    assert not any(case["timing_criterion_present"] for case in cases)


def test_pattern_published_to_authors():
    assert "THRESHOLD_TRIGGERED_RESPONSE" in KNOWN_PATTERNS
    rules = render_authoring_rules()
    assert "THRESHOLD_TRIGGERED_RESPONSE" in rules
    assert "no deadline" in rules.lower() or "NO\ndeadline" in rules


def test_arbitration_naming_defaults():
    defaults = AGPrioritySpec(
        response_set_id="X", members=("A", "B"), edges=(("A", "B"),),
        trigger="t", selected_response="A", source_kind="k", source_id="i",
    )
    assert defaults.trigger_signal == "CriticalPropulsionFailureDetectedSignal"
    assert defaults.selected_state == "parachuteDeploymentSelected"
    assert defaults.selection_transition == "selectParachute"
    emitted = emit_ag_package(REQ_SAFE_005_CHAIN)
    assert "transition selectParachute first awaitingResponse" in emitted
    assert "then parachuteDeploymentSelected;" in emitted


def test_two_chains_collide_on_arbiter():
    """The profile supports one arbitrating chain per model.

    A probe selecting all four chains failed in step-1 planning with `duplicate A/G
    behavior contract SafetyResponseArbiterContract`; ``SafetyResponseArbitration``
    is a hard-coded identifier six modules use to recognise the arbiter, so every
    arbitrating chain claims it and the model cannot repair the clash. The pattern
    is therefore standalone, not composable with REQ_SAFE_005; making the arbiter
    identity chain-scoped has to come past this assertion.
    """
    collisions = {}
    for chain in (REQ_SAFE_005_CHAIN, REQ_SAFE_002_CHAIN):
        for component in chain.components:
            collisions.setdefault(
                (component.name, component.behavior), []
            ).append(chain.source_requirement)
    shared = {key: sources for key, sources in collisions.items()
              if len(sources) > 1}
    assert shared == {
        ("SafetyResponseArbiterContract", "SafetyResponseArbitration"): [
            "REQ_SAFE_005", "REQ_SAFE_002",
        ]
    }


def test_subordinate_direction_excluded():
    """REQ_SAFE_001 is out of scope for a structural reason.

    The arbitration encoding guards competing transitions with ``not <trigger>``, so
    the selected response supersedes the others; REQ_SAFE_001's response is
    superseded by them, and encoding it here would invert the requirement.
    """
    priority = REQ_SAFE_002_CHAIN.priority
    assert priority is not None
    dominated = {lower for _higher, lower in priority.edges}
    assert dominated == set(priority.members) - {priority.selected_response}
