"""THRESHOLD_TRIGGERED_RESPONSE — a triggered response that states no deadline.

The bounded pattern set had a structural gap. A requirement that names a trigger,
a response, and a precedence relation but no time bound could be declared under
neither family: the timed pattern rejects a chain with no budget, and stating a
trigger→response obligation as a continuously held invariant misdescribes it. The
only way through was to invent a deadline the stakeholder never wrote, which is
precisely the failure the pattern declaration exists to prevent.

REQ_SAFE_002 ("controlled descent when state-of-charge falls below 15%,
superseding any lower-priority contingency response") is that shape. It is the
timed failsafe's arbitration topology with the timing obligations removed, so the
pattern reuses the arbitration checks and drops only what a deadline pays for.

Scope: this covers the DOMINANT direction, where the selected response supersedes
the others — the direction the arbitration encoding already expresses (competing
transitions guarded by ``not <trigger>``). A SUBORDINATE threshold response
(REQ_SAFE_001, "return to base ... unless a higher-priority response is already in
progress") is NOT covered: it needs the opposite precedence direction and a guard
shape the extractor does not parse. That is a separate extension, and it is left
declared rather than approximated.
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


def test_reference_chain_emits_a_strictly_clean_model():
    """No stdlib-diagnostic allowance and no warnings — the evidence-path gate."""
    gate = check_syntax(
        _render(REQ_SAFE_002_CHAIN),
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert not gate.has_errors
    assert not gate.warnings


def test_reference_chain_round_trips_to_a_checker_pass():
    report = check_ag_graph(extract_ag_graph(_render(REQ_SAFE_002_CHAIN)))
    assert report.verdict == "PASS"
    assert not report.errors()


def test_decided_spec_round_trips_to_a_checker_pass():
    """The production path: decisions in, rendered SysML out, checker PASS."""
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


def test_untimed_trigger_is_stated_once_on_the_priority_contract():
    """No timing origin to cross-check against, so the trigger is not duplicated.

    The timed pattern states its trigger twice — as the priority contract's
    trigger and as the system contract's interval origin — and the checker holds
    them to agreement. An untimed chain declares no interval, so demanding a
    timing origin would be demanding the deadline the pattern does without.
    """
    spec = build_spec_from_decisions(_DECISIONS, _BOUNDARY)
    assert spec.priority.trigger == "criticalBatteryThresholdReached"
    graph = extract_ag_graph(_render(spec))
    assert graph.system is not None
    assert graph.system.timing_budget is None
    assert graph.system.timing_origin is None
    assert not check_ag_graph(graph).errors()


def test_a_deadline_makes_it_the_wrong_pattern():
    with pytest.raises(DecisionError, match="must not carry deadline_seconds"):
        validate_decisions(_decisions(deadline_seconds=0.5), _BOUNDARY)


def test_no_component_may_apportion_a_budget_there_is_nothing_to_apportion():
    payload = _decisions()
    payload["components"][0]["latency_budget_seconds"] = 0.1
    with pytest.raises(DecisionError, match="latency_budget_seconds"):
        validate_decisions(payload, _BOUNDARY)


def test_an_untimed_pattern_does_not_have_to_state_invariants():
    """The gap this closes: it is a triggered pattern, not an invariant one.

    Before the split, everything that was not the timed pattern was treated as an
    invariant pattern and had to state at least one invariant. A trigger→response
    obligation stated as a continuously held implication is a misdescription.
    """
    assert "THRESHOLD_TRIGGERED_RESPONSE" not in INVARIANT_PATTERNS
    assert "THRESHOLD_TRIGGERED_RESPONSE" in TRIGGERED_PATTERNS
    payload = _decisions()
    assert "invariants" not in payload
    validate_decisions(payload, _BOUNDARY)


def test_the_timed_pattern_still_requires_its_deadline():
    """Regression guard on the three-way split, not a new property."""
    payload = _decisions(safety_pattern="TRIGGERED_TIMED_FAILSAFE_RESPONSE")
    with pytest.raises(DecisionError, match="needs deadline_seconds"):
        validate_decisions(payload, _BOUNDARY)


def test_conformance_does_not_re_impose_a_timing_criterion():
    graph = extract_ag_graph(_render(REQ_SAFE_002_CHAIN))
    report = check_ag_graph(graph)
    conformance = check_safety_pattern_conformance(graph, report)
    assert conformance["verdict"] == "PASS"
    cases = conformance["cases"]
    assert cases
    assert {case["pattern"] for case in cases} == {
        "THRESHOLD_TRIGGERED_RESPONSE"
    }
    # It passes without any case claiming a timing criterion: were the timed
    # pattern's extra duty still applied, these cases would fail.
    assert not any(case["timing_criterion_present"] for case in cases)


def test_the_pattern_is_published_to_authors():
    """A checker rule the generator is never told is unsatisfiable by any author."""
    assert "THRESHOLD_TRIGGERED_RESPONSE" in KNOWN_PATTERNS
    rules = render_authoring_rules()
    assert "THRESHOLD_TRIGGERED_RESPONSE" in rules
    # and it says how to tell it apart from the timed one
    assert "no deadline" in rules.lower() or "NO\ndeadline" in rules


def test_arbitration_naming_defaults_keep_the_parachute_chain_identical():
    """The new naming fields must not disturb REQ_SAFE_005 or its frozen gold."""
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


def test_two_arbitrating_chains_collide_on_the_single_global_arbiter():
    """Measured boundary: the profile supports ONE arbitrating chain per model.

    A paid single-arm probe (2026-07-31, seed 0) selected all four chains and
    failed in step-1 planning, not in the A/G layer: six attempts, four of them
    reporting `duplicate A/G behavior contract SafetyResponseArbiterContract` and
    `duplicate stable behavior id SafetyResponseArbiter::SafetyResponseArbitration`.
    The model could not repair it because the duplicate does not come from the
    model — ``SafetyResponseArbitration`` is a hard-coded identifier that six
    modules use to recognise the arbiter, so every arbitrating chain necessarily
    claims it.

    So this pattern is proven STANDALONE and is not composable with REQ_SAFE_005.
    Pinned here so the constraint is rediscovered by a free test rather than by
    another paid run, and so that making the arbiter identity chain-scoped is a
    deliberate change that has to come past this assertion.
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


def test_the_subordinate_direction_is_declared_out_of_scope_not_approximated():
    """REQ_SAFE_001 is not covered, and the reason is structural.

    The arbitration encoding guards competing transitions with ``not <trigger>``,
    which says the selected response supersedes the others. REQ_SAFE_001's
    response is superseded BY the others. Encoding it under this pattern would
    invert what the requirement says, so it stays uncovered.
    """
    priority = REQ_SAFE_002_CHAIN.priority
    assert priority is not None
    dominated = {lower for _higher, lower in priority.edges}
    assert dominated == set(priority.members) - {priority.selected_response}
