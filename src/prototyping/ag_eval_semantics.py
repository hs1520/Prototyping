"""Evaluator comparison semantics for the A2 agreement categories.

Implements the student-fixed comparison policy (STUDENT_DESIGN_DECISIONS.md §6):
exact-decimal quantities and unit conversion, frozen timing-origin aliases, a typed
Boolean-expression AST with canonical normalisation, priority as an explicit
response-set + precedence edges + trigger, and invariant normalisation. The
evaluator *computes* derived timing values (`additive_total`, `within_deadline`);
they are never read from gold. Timing, priority, and invariant agreement are
reported as **separate** categories and never merged into a composite F1.

This module is evaluator-only and F3-safe: it imports neither ``ag_extractor`` nor
``ag_contracts``. It uses :class:`decimal.Decimal` (never binary float) as the
arithmetic authority.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Tuple


class SemanticsError(ValueError):
    """A gold/prediction artifact violates the fixed comparison policy."""


# ── §6.1 exact decimal quantities and unit conversion ───────────────────────

_UNIT_TO_SECONDS: Dict[str, Decimal] = {
    "s": Decimal(1),
    "ms": Decimal("0.001"),
    "us": Decimal("0.000001"),
}


def quantity_to_seconds(quantity: Mapping[str, Any]) -> Decimal:
    """Exact seconds from an atomic quantity ``{"value": "0.5", "unit": "s"}``.

    Binary floats are rejected as evaluator authority (§6.1): a tool must archive
    the source decimal literal (a string) or a declared uncertainty. Conversion
    uses exact decimal powers of ten and the comparison tolerance is exactly zero.
    """
    if not isinstance(quantity, Mapping):
        raise SemanticsError("quantity must be a mapping {value, unit}")
    value = quantity.get("value")
    unit = quantity.get("unit")
    if isinstance(value, float):
        raise SemanticsError(
            "binary float is not accepted as evaluator authority; archive the "
            "source decimal literal as a string or a declared uncertainty"
        )
    if unit not in _UNIT_TO_SECONDS:
        raise SemanticsError(f"unsupported duration unit {unit!r}; use s/ms/us")
    try:
        magnitude = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise SemanticsError(f"invalid decimal value {value!r}")
    return magnitude * _UNIT_TO_SECONDS[unit]


# ── §6.2 frozen timing-origin identity (no fuzzy matching) ───────────────────

# The only accepted historical alias, as an explicit frozen mapping. Anything else
# must match exactly; name/text similarity is never inferred.
_TIMING_ORIGIN_ALIASES: Dict[str, str] = {
    "criticalfailureevent": "criticalPropulsionFailureDetected",
}


def resolve_timing_origin(name: Any) -> str:
    key = str(name if name is not None else "").strip()
    return _TIMING_ORIGIN_ALIASES.get(key.lower(), key)


# ── §6.3 typed Boolean expression AST ────────────────────────────────────────

_BOOL_NODES = {"Identifier", "Not", "And", "Or", "Implies"}


def _normalise_ast(node: Any) -> Dict[str, Any]:
    if not isinstance(node, Mapping) or node.get("node") not in _BOOL_NODES:
        raise SemanticsError(f"unsupported Boolean AST node: {node!r}")
    kind = node["node"]
    if kind == "Identifier":
        name = node.get("name")
        if not name:
            raise SemanticsError("Identifier node requires a name")
        return {"node": "Identifier", "name": str(name)}
    if kind == "Not":
        inner = _normalise_ast(node.get("expr"))
        if inner.get("node") == "Not":  # double negation removed
            return inner["expr"]
        return {"node": "Not", "expr": inner}
    if kind in ("And", "Or"):
        flattened: List[Dict[str, Any]] = []
        for operand in node.get("operands", ()):
            child = _normalise_ast(operand)
            if child.get("node") == kind:  # flatten same-kind nesting
                flattened.extend(child["operands"])
            else:
                flattened.append(child)
        # deduplicate and sort by canonical serialisation
        unique: Dict[str, Dict[str, Any]] = {}
        for child in flattened:
            unique[_serialise_ast(child)] = child
        operands = [unique[key] for key in sorted(unique)]
        if not operands:
            raise SemanticsError(f"{kind} requires at least one operand")
        if len(operands) == 1:
            return operands[0]
        return {"node": kind, "operands": operands}
    # Implies stays an AST node; it is rendered to SysML as `not a or b` elsewhere.
    return {
        "node": "Implies",
        "antecedent": _normalise_ast(node.get("antecedent")),
        "consequent": _normalise_ast(node.get("consequent")),
    }


def _serialise_ast(node: Mapping[str, Any]) -> str:
    kind = node["node"]
    if kind == "Identifier":
        return f"id:{node['name']}"
    if kind == "Not":
        return f"not({_serialise_ast(node['expr'])})"
    if kind in ("And", "Or"):
        inner = ",".join(sorted(_serialise_ast(o) for o in node["operands"]))
        return f"{kind.lower()}({inner})"
    return (
        f"implies({_serialise_ast(node['antecedent'])},"
        f"{_serialise_ast(node['consequent'])})"
    )


def canonical_ast_key(node: Any) -> str:
    """Canonical, comparison-stable string for a Boolean AST (normalised first)."""
    return _serialise_ast(_normalise_ast(node))


# ── set-agreement helper (local, to avoid importing the evaluator) ───────────

def _prf(predicted: set, gold: set) -> Dict[str, Any]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── §6.1/§6.2 timing agreement (evaluator computes the derived values) ───────

def _timing_facts(block: Mapping[str, Any]) -> Dict[str, Any]:
    origin = resolve_timing_origin(block.get("origin"))
    deadline_s = quantity_to_seconds(block["deadline"]) if block.get("deadline") else None
    segments = []
    for seg in block.get("segments", ()) or ():
        segments.append((str(seg.get("component")), quantity_to_seconds(seg["budget"])))
    additive_total = sum((b for _c, b in segments), Decimal(0))
    within = None if deadline_s is None else (additive_total <= deadline_s)
    return {
        "origin": origin,
        "deadline_s": deadline_s,
        "segments": segments,
        "additive_total_s": additive_total,
        "within_deadline": within,
    }


def timing_agreement(
    prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Separate timing category: origin identity, deadline, and per-segment budgets.

    ``additive_total`` / ``within_deadline`` are recomputed for each side here and
    never taken from the artifact, so a self-contradictory gold cannot slip through.
    """
    pred = _timing_facts(prediction)
    ref = _timing_facts(gold)
    pred_segments = {(c, str(b)) for c, b in pred["segments"]}
    ref_segments = {(c, str(b)) for c, b in ref["segments"]}
    return {
        "category": "timing_agreement",
        "origin_match": pred["origin"] == ref["origin"],
        "deadline_match": pred["deadline_s"] == ref["deadline_s"],
        "segment_prf": _prf(pred_segments, ref_segments),
        "computed": {
            "prediction": {
                "additive_total_s": str(pred["additive_total_s"]),
                "within_deadline": pred["within_deadline"],
            },
            "gold": {
                "additive_total_s": str(ref["additive_total_s"]),
                "within_deadline": ref["within_deadline"],
            },
        },
        "within_deadline_match": pred["within_deadline"] == ref["within_deadline"],
    }


# ── §6.4 priority agreement (explicit set + edges + trigger, never a Boolean) ─

def _edge_set(edges) -> set:
    return {
        (str(e.get("higher")), str(e.get("lower")))
        for e in (edges or ())
    }


def priority_agreement(
    prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Separate priority category from an explicit response set, precedence edges,
    trigger, and the separately extracted arbitration topology — never inferred
    from a bare ``priority=true`` claim (§6.4)."""
    return {
        "category": "priority_agreement",
        "response_set_id_match": (
            prediction.get("response_set_id") == gold.get("response_set_id")
        ),
        "members_match": (
            set(map(str, prediction.get("members", ())))
            == set(map(str, gold.get("members", ())))
        ),
        "edge_prf": _prf(
            _edge_set(prediction.get("edges")), _edge_set(gold.get("edges"))
        ),
        "trigger_match": (
            resolve_timing_origin(prediction.get("trigger"))
            == resolve_timing_origin(gold.get("trigger"))
        ),
        # A structural claim only counts if the prediction actually carries the
        # extracted arbitration topology; a Boolean flag is not evidence.
        "arbitration_topology_present": bool(
            prediction.get("arbitration_topology_present")
        ),
    }


# ── §6.5 invariant agreement (separate stakeholder / student denominators) ───

def _invariant_key(inv: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(inv.get("scope") or ""),
        canonical_ast_key(inv.get("trigger_or_antecedent_ast")),
        canonical_ast_key(inv.get("required_consequent_ast")),
    )


_INVARIANT_SOURCE_KINDS = ("STAKEHOLDER", "STUDENT_DERIVED_DESIGN_CONSTRAINT")


def invariant_agreement(
    prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Separate invariant category, reported **per source kind** so stakeholder and
    student-derived invariants are never pooled into one denominator (§6.5)."""
    def by_kind(items, kind):
        out = set()
        for inv in items or ():
            if str(inv.get("source_kind")) == kind:
                out.add(_invariant_key(inv))
        return out

    result: Dict[str, Any] = {"category": "invariant_agreement"}
    pred_items = prediction.get("invariants")
    gold_items = gold.get("invariants")
    for kind in _INVARIANT_SOURCE_KINDS:
        result[kind.lower()] = _prf(
            by_kind(pred_items, kind), by_kind(gold_items, kind)
        )
    return result
