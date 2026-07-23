"""A2 evaluator comparison semantics (STUDENT_DESIGN_DECISIONS.md §6).

Exact-decimal quantities, frozen timing-origin aliases, a typed Boolean AST with
canonical normalisation, explicit priority, and per-source-kind invariants — each a
separate agreement category, with derived timing values computed by the evaluator
and never trusted from gold.
"""
from __future__ import annotations

import pytest

from src.prototyping.ag_eval_semantics import (
    SemanticsError,
    canonical_ast_key,
    invariant_agreement,
    priority_agreement,
    quantity_to_seconds,
    resolve_timing_origin,
    timing_agreement,
)


# ── §6.1 quantities/units ────────────────────────────────────────────────────

def test_quantity_converts_units_exactly_and_rejects_binary_float():
    assert quantity_to_seconds({"value": "0.5", "unit": "s"}) == \
        quantity_to_seconds({"value": "500", "unit": "ms"})
    assert str(quantity_to_seconds({"value": "0.05", "unit": "s"})) == "0.05"
    with pytest.raises(SemanticsError, match="binary float"):
        quantity_to_seconds({"value": 0.5, "unit": "s"})
    with pytest.raises(SemanticsError, match="unit"):
        quantity_to_seconds({"value": "1", "unit": "minutes"})


# ── §6.2 frozen timing-origin identity ───────────────────────────────────────

def test_only_the_frozen_alias_resolves_no_fuzzy_matching():
    assert resolve_timing_origin("criticalFailureEvent") == \
        "criticalPropulsionFailureDetected"
    assert resolve_timing_origin("criticalPropulsionFailureDetected") == \
        "criticalPropulsionFailureDetected"
    # a similar-but-unlisted name is NOT coerced
    assert resolve_timing_origin("criticalPropulsionFault") == \
        "criticalPropulsionFault"


# ── §6.3 typed Boolean AST canonicalisation ──────────────────────────────────

def _id(name):
    return {"node": "Identifier", "name": name}


def test_and_flattens_dedups_and_is_order_independent():
    a = {"node": "And", "operands": [_id("x"), {"node": "And", "operands": [_id("y"), _id("z")]}]}
    b = {"node": "And", "operands": [_id("z"), _id("y"), _id("x"), _id("x")]}
    assert canonical_ast_key(a) == canonical_ast_key(b)


def test_double_negation_is_removed_and_implies_is_preserved():
    assert canonical_ast_key({"node": "Not", "expr": {"node": "Not", "expr": _id("p")}}) \
        == canonical_ast_key(_id("p"))
    impl = {"node": "Implies", "antecedent": _id("a"), "consequent": _id("b")}
    assert canonical_ast_key(impl).startswith("implies(")
    # implies is not commutative
    assert canonical_ast_key(impl) != canonical_ast_key(
        {"node": "Implies", "antecedent": _id("b"), "consequent": _id("a")}
    )


def test_unsupported_ast_node_is_rejected():
    with pytest.raises(SemanticsError):
        canonical_ast_key({"node": "Xor", "operands": [_id("a"), _id("b")]})


# ── §6.1/§6.2 timing agreement (evaluator computes derived values) ───────────

def _timing(origin, deadline, segs):
    return {
        "origin": origin,
        "deadline": {"value": deadline, "unit": "s"},
        "segments": [
            {"component": c, "budget": {"value": b, "unit": "s"}} for c, b in segs
        ],
    }


def test_timing_agreement_computes_additive_total_and_within_deadline():
    gold = _timing("criticalPropulsionFailureDetected", "0.5",
                   [("SafetyResponseArbiter", "0.10"), ("RecoverySystem", "0.35")])
    # prediction uses the frozen alias for the origin and matches the budget
    pred = _timing("criticalFailureEvent", "0.5",
                   [("SafetyResponseArbiter", "0.10"), ("RecoverySystem", "0.35")])
    out = timing_agreement(pred, gold)
    assert out["origin_match"] is True          # alias resolves
    assert out["deadline_match"] is True
    assert out["segment_prf"]["f1"] == 1.0
    # the evaluator computed the total; it is not read from gold
    assert out["computed"]["gold"]["additive_total_s"] == "0.45"
    assert out["computed"]["gold"]["within_deadline"] is True


def test_timing_agreement_flags_an_over_budget_prediction():
    gold = _timing("criticalPropulsionFailureDetected", "0.5",
                   [("A", "0.10"), ("B", "0.35")])
    over = _timing("criticalPropulsionFailureDetected", "0.5",
                   [("A", "0.10"), ("B", "0.45")])  # 0.55 > 0.5
    out = timing_agreement(over, gold)
    assert out["computed"]["prediction"]["within_deadline"] is False
    assert out["within_deadline_match"] is False
    assert out["segment_prf"]["f1"] < 1.0


# ── §6.4 priority agreement ──────────────────────────────────────────────────

def test_priority_needs_explicit_edges_not_a_boolean_and_checks_topology():
    gold = {
        "response_set_id": "FLIGHT_RESPONSES_V1",
        "members": ["PARACHUTE_DEPLOYMENT", "LOW_BATTERY_RETURN_TO_BASE"],
        "edges": [{"higher": "PARACHUTE_DEPLOYMENT", "lower": "LOW_BATTERY_RETURN_TO_BASE"}],
        "trigger": "criticalPropulsionFailureDetected",
    }
    pred = dict(gold, arbitration_topology_present=True)
    out = priority_agreement(pred, gold)
    assert out["response_set_id_match"] and out["members_match"]
    assert out["edge_prf"]["f1"] == 1.0 and out["trigger_match"]
    assert out["arbitration_topology_present"] is True
    # a bare claim with no extracted arbitration structure does not count
    assert priority_agreement(dict(gold), gold)["arbitration_topology_present"] is False


# ── §6.5 invariants: separate stakeholder / student-derived denominators ─────

def _inv(iid, ante, cons, kind):
    return {
        "invariant_id": iid, "scope": "system",
        "trigger_or_antecedent_ast": _id(ante),
        "required_consequent_ast": _id(cons),
        "source_kind": kind, "source_id": iid,
    }


def test_invariants_are_scored_separately_by_source_kind():
    gold = {"invariants": [
        _inv("i1", "powerOnInitialisation", "payloadLocked", "STAKEHOLDER"),
        _inv("i2", "notActuatorPowerAvailable", "payloadLocked",
             "STUDENT_DERIVED_DESIGN_CONSTRAINT"),
    ]}
    pred = {"invariants": [
        _inv("i1", "powerOnInitialisation", "payloadLocked", "STAKEHOLDER"),
    ]}
    out = invariant_agreement(pred, gold)
    # stakeholder invariant matched; student-derived one missing — never pooled
    assert out["stakeholder"]["f1"] == 1.0
    assert out["student_derived_design_constraint"]["recall"] == 0.0
    assert "f1" not in out  # there is no composite score


# ── evaluator wiring: separate categories surface, never a merged F1 ─────────

def test_evaluator_reports_each_category_separately_when_present():
    from src.prototyping.ag_evaluation import evaluate_ag_against_gold

    timing = _timing("criticalPropulsionFailureDetected", "0.5",
                     [("SafetyResponseArbiter", "0.10"), ("RecoverySystem", "0.35")])
    priority = {
        "response_set_id": "FLIGHT_RESPONSES_V1",
        "members": ["PARACHUTE_DEPLOYMENT"],
        "edges": [],
        "trigger": "criticalPropulsionFailureDetected",
        "arbitration_topology_present": True,
    }
    invariants = [_inv("i1", "powerOnInitialisation", "payloadLocked", "STAKEHOLDER")]

    prediction = {
        "artifact_role": "RUNTIME_A_G_PREDICTION",
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "configuration": "R2-BBAG",
        "graph": {
            "allocations": [], "discharge_edges": [],
            "timing": timing, "priority": priority, "invariants": invariants,
        },
    }
    gold = {
        "artifact_role": "EVALUATOR_GOLD",
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "status": "FROZEN",
        "reviewer": "Dr. Supervisor",
        "reviewed_date": "2026-07-30",
        "review_protocol": {
            "blind_to_runtime_verdict": True, "independent_human_review": True,
        },
        "allocations": [],
        "timing": timing, "priority": priority, "invariants": invariants,
    }
    out = evaluate_ag_against_gold(prediction, gold)
    # each A2 category present, on its own, with no merged/composite F1 anywhere
    assert out["timing_agreement"]["category"] == "timing_agreement"
    assert out["priority_agreement"]["category"] == "priority_agreement"
    assert out["invariant_agreement"]["category"] == "invariant_agreement"
    assert "f1" not in out
    # allocation/discharge remain their own separate categories too
    assert "guarantee_allocation" in out and "assumption_discharge" in out
