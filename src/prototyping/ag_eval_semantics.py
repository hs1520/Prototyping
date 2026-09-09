"""Evaluator comparison semantics for the A2 agreement categories.

Implements the student-fixed comparison policy (STUDENT_DESIGN_DECISIONS.md §6):
exact-decimal quantities and unit conversion, frozen timing-origin aliases, a typed
Boolean-expression AST with canonical normalisation, priority as response-set +
precedence edges + trigger, and invariant normalisation. Derived timing values
(`additive_total`, `within_deadline`) are computed here, not read from gold, and
timing, priority and invariant agreement stay separate categories rather than one
composite F1. Evaluator-only and F3-safe: imports neither ``ag_extractor`` nor
``ag_contracts``, and uses :class:`decimal.Decimal` rather than binary float as the
arithmetic authority.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Tuple

from .ag_profile import INVARIANT_SOURCE_KINDS


class SemanticsError(ValueError):
    """A gold/prediction artifact violates the fixed comparison policy."""


_UNIT_TO_SECONDS: Dict[str, Decimal] = {
    "s": Decimal(1),
    "ms": Decimal("0.001"),
    "us": Decimal("0.000001"),
}


def quantity_to_seconds(quantity: Mapping[str, Any]) -> Decimal:
    """Exact seconds from an atomic quantity ``{"value": "0.5", "unit": "s"}``.

    Binary floats are rejected as evaluator authority (§6.1); a tool archives the
    source decimal literal (a string) or a declared uncertainty. Conversion uses
    decimal powers of ten and the comparison tolerance is zero.
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
    if not isinstance(value, (str, Decimal)):
        raise SemanticsError(
            "quantity value must be an archived decimal string, not a computed number"
        )
    if unit not in _UNIT_TO_SECONDS:
        raise SemanticsError(f"unsupported duration unit {unit!r}; use s/ms/us")
    try:
        magnitude = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise SemanticsError(f"invalid decimal value {value!r}")
    if not magnitude.is_finite() or magnitude < 0:
        raise SemanticsError("duration value must be finite and non-negative")
    return magnitude * _UNIT_TO_SECONDS[unit]


# The only accepted historical alias, as a frozen mapping. Everything else
# matches exactly; name/text similarity is not inferred.
_TIMING_ORIGIN_ALIASES: Dict[str, str] = {
    "criticalfailureevent": "criticalPropulsionFailureDetected",
}


def resolve_timing_origin(name: Any) -> str:
    key = str(name if name is not None else "").strip()
    return _TIMING_ORIGIN_ALIASES.get(key.lower(), key)


_BOOL_NODES = {"Identifier", "Not", "And", "Or", "Implies"}


def _normalise_ast(
    node: Any, selected_model_elements: set[str] | None = None
) -> Dict[str, Any]:
    if not isinstance(node, Mapping) or node.get("node") not in _BOOL_NODES:
        raise SemanticsError(f"unsupported Boolean AST node: {node!r}")
    kind = node["node"]
    if kind == "Identifier":
        name = node.get("name")
        if not name:
            raise SemanticsError("Identifier node requires a name")
        if (
            selected_model_elements is not None
            and str(name) not in selected_model_elements
        ):
            raise SemanticsError(
                f"Identifier {name!r} does not resolve to a selected model element"
            )
        return {"node": "Identifier", "name": str(name)}
    if kind == "Not":
        inner = _normalise_ast(node.get("expr"), selected_model_elements)
        if inner.get("node") == "Not":
            return inner["expr"]
        return {"node": "Not", "expr": inner}
    if kind in ("And", "Or"):
        flattened: List[Dict[str, Any]] = []
        for operand in node.get("operands", ()):
            child = _normalise_ast(operand, selected_model_elements)
            if child.get("node") == kind:
                flattened.extend(child["operands"])
            else:
                flattened.append(child)
        unique: Dict[str, Dict[str, Any]] = {}
        for child in flattened:
            unique[_serialise_ast(child)] = child
        operands = [unique[key] for key in sorted(unique)]
        if not operands:
            raise SemanticsError(f"{kind} requires at least one operand")
        if len(operands) == 1:
            return operands[0]
        return {"node": kind, "operands": operands}
    return {
        "node": "Implies",
        "antecedent": _normalise_ast(
            node.get("antecedent"), selected_model_elements
        ),
        "consequent": _normalise_ast(
            node.get("consequent"), selected_model_elements
        ),
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


def canonical_ast_key(
    node: Any, selected_model_elements: Any = None
) -> str:
    """Canonical, comparison-stable string for a Boolean AST (normalised first)."""
    resolved = None
    if selected_model_elements is not None:
        if not isinstance(selected_model_elements, (list, tuple, set, frozenset)):
            raise SemanticsError("selected_model_elements must be a sequence")
        resolved = {str(item) for item in selected_model_elements}
        if not resolved:
            raise SemanticsError("selected_model_elements must be non-empty")
    return _serialise_ast(_normalise_ast(node, resolved))


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


def _timing_facts(block: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(block, Mapping):
        raise SemanticsError("timing facts must be an object")
    origin = resolve_timing_origin(block.get("origin"))
    deadline_s = quantity_to_seconds(block["deadline"]) if block.get("deadline") else None
    segments = []
    for index, seg in enumerate(block.get("segments", ()) or ()):
        if not isinstance(seg, Mapping):
            raise SemanticsError(f"timing segment[{index}] must be an object")
        if not str(seg.get("component") or ""):
            raise SemanticsError(f"timing segment[{index}] component must be set")
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

    ``additive_total`` / ``within_deadline`` are recomputed per side rather than read
    from the artifact, so a self-contradictory gold is caught.
    """
    pred = _timing_facts(prediction)
    ref = _timing_facts(gold)
    # Segments are ordered, additive facts; the ordinal keeps duplicates from
    # collapsing into a set and changing the total.
    pred_segments = {
        (index, component, str(budget))
        for index, (component, budget) in enumerate(pred["segments"])
    }
    ref_segments = {
        (index, component, str(budget))
        for index, (component, budget) in enumerate(ref["segments"])
    }
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
        "additive_total_match": (
            pred["additive_total_s"] == ref["additive_total_s"]
        ),
        "within_deadline_match": pred["within_deadline"] == ref["within_deadline"],
    }


def _edge_set(edges) -> set:
    result = set()
    for index, edge in enumerate(edges or ()):
        if not isinstance(edge, Mapping):
            raise SemanticsError(f"priority edge[{index}] must be an object")
        result.add((str(edge.get("higher")), str(edge.get("lower"))))
    return result


def priority_agreement(
    prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Separate priority category from the response set, precedence edges, trigger and
    the separately extracted arbitration topology, not from a bare ``priority=true``
    claim (§6.4).
    """
    if not isinstance(prediction, Mapping) or not isinstance(gold, Mapping):
        raise SemanticsError("priority facts must be objects")
    members = set(map(str, prediction.get("members", ())))
    gold_members = set(map(str, gold.get("members", ())))
    prediction_edges = _edge_set(prediction.get("edges"))
    gold_edges = _edge_set(gold.get("edges"))
    trigger = resolve_timing_origin(prediction.get("trigger"))
    gold_trigger = resolve_timing_origin(gold.get("trigger"))

    topology = prediction.get("arbitration_topology")
    topology_conforms = False
    topology_problems: List[str] = []
    if not isinstance(topology, Mapping):
        topology_problems.append("explicit extracted arbitration_topology is missing")
    else:
        topology_members = set(map(str, topology.get("members", ())))
        topology_edges = _edge_set(topology.get("edges"))
        if topology.get("response_set_id") != gold.get("response_set_id"):
            topology_problems.append("topology response_set_id mismatch")
        if topology_members != gold_members:
            topology_problems.append("topology members mismatch")
        if topology_edges != gold_edges:
            topology_problems.append("topology precedence edges mismatch")
        if resolve_timing_origin(topology.get("trigger")) != gold_trigger:
            topology_problems.append("topology trigger mismatch")

        higher = {edge[0] for edge in gold_edges}
        selected = next(iter(higher)) if len(higher) == 1 else None
        selection = topology.get("selection") or {}
        if (
            not selected
            or selection.get("selected_response") != selected
            or resolve_timing_origin(selection.get("when")) != gold_trigger
        ):
            topology_problems.append("critical-trigger response selection is missing")

        expected_lowers = {edge[1] for edge in gold_edges}
        guards = {
            str(item.get("response")): item.get("guard_ast")
            for item in (topology.get("competing_transition_guards") or ())
            if isinstance(item, Mapping)
        }
        expected_guard = {
            "node": "Not",
            "expr": {"node": "Identifier", "name": str(gold.get("trigger"))},
        }
        allowed = topology.get("selected_model_elements")
        try:
            expected_guard_key = canonical_ast_key(expected_guard, allowed)
            guarded = {
                response
                for response, guard in guards.items()
                if canonical_ast_key(guard, allowed) == expected_guard_key
            }
        except SemanticsError as exc:
            guarded = set()
            topology_problems.append(str(exc))
        if guarded != expected_lowers:
            topology_problems.append("competing transition guards are incomplete")
        for field in (
            "parachute_transition_reachable",
            "selection_action_connected",
            "deployment_action_connected",
            "observation_connected",
        ):
            if topology.get(field) is not True:
                topology_problems.append(f"{field} must be true")
        topology_conforms = not topology_problems

    return {
        "category": "priority_agreement",
        "response_set_id_match": (
            prediction.get("response_set_id") == gold.get("response_set_id")
        ),
        "members_match": (
            members == gold_members
        ),
        "edge_prf": _prf(
            prediction_edges, gold_edges
        ),
        "trigger_match": trigger == gold_trigger,
        "arbitration_topology_conforms": topology_conforms,
        "arbitration_topology_problems": topology_problems,
    }


def _invariant_key(
    inv: Mapping[str, Any], selected_model_elements: Any
) -> Tuple[str, str, str, str, str]:
    invariant_id = str(inv.get("invariant_id") or "")
    source_id = str(inv.get("source_id") or "")
    if not invariant_id or not source_id:
        raise SemanticsError("invariant_id and source_id must be set")
    return (
        invariant_id,
        str(inv.get("scope") or ""),
        canonical_ast_key(
            inv.get("trigger_or_antecedent_ast"), selected_model_elements
        ),
        canonical_ast_key(
            inv.get("required_consequent_ast"), selected_model_elements
        ),
        source_id,
    )


def invariant_agreement(
    prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Separate invariant category, reported per source kind so stakeholder and
    student-derived invariants keep separate denominators (§6.5).
    """
    def by_kind(items, kind, selected_model_elements):
        out = set()
        for index, inv in enumerate(items or ()):
            if not isinstance(inv, Mapping):
                raise SemanticsError(f"invariant[{index}] must be an object")
            source_kind = str(inv.get("source_kind"))
            if source_kind not in INVARIANT_SOURCE_KINDS:
                raise SemanticsError(
                    f"unsupported invariant source_kind {source_kind!r}"
                )
            if source_kind == kind:
                out.add(_invariant_key(inv, selected_model_elements))
        return out

    result: Dict[str, Any] = {"category": "invariant_agreement"}
    pred_items = prediction.get("invariants")
    gold_items = gold.get("invariants")
    pred_elements = prediction.get("selected_model_elements")
    gold_elements = gold.get("selected_model_elements")
    for kind in INVARIANT_SOURCE_KINDS:
        result[kind.lower()] = _prf(
            by_kind(pred_items, kind, pred_elements),
            by_kind(gold_items, kind, gold_elements),
        )
    return result
