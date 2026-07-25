"""Where a generated model's facts came from, and where they diverge.

The three robustness pillars cannot tell the two emitter-rendered R2 modes apart.
Both `DETERMINISTIC_SPEC_EMITTER` and `LLM_DECIDED_SPEC` render through
`ag_emitter`, so conformance is guaranteed by the renderer rather than earned by
the decisions: a full 3x3 pilot of each produced pattern-conformance reports
differing only in the model digest. Everything that actually differed — the timing
apportionment, the response-set enumeration — was invisible to every pillar.

This measures that difference. It is deliberately reported as **divergence, not
accuracy**, because for the facts at issue there is no ground truth to be accurate
against:

``REQUIREMENT_DETERMINED``
    The requirement text fixes the answer — the safety pattern it instantiates,
    what starts the timing, the stated deadline. A divergence here is a defect: the
    author misread a requirement that says what it means.
``DESIGNER_SUPPLIED``
    The requirement text does not contain the answer — how the deadline divides
    across components, which responses exist and in what order. A divergence here
    is a different design decision, not an error. Apportioning 0.2 + 0.3 and
    0.1 + 0.35 are both internally consistent and both meet a 0.5 s deadline; one
    holds margin, the other does not, and the requirement is silent on margin.

Comparing against the committed chain spec is not gold scoring: that spec is the
student-approved decomposition, and for DESIGNER_SUPPLIED facts it is one defensible
choice rather than the correct one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

DIVERGENCE_SCHEMA_VERSION = "1.0"

REQUIREMENT_DETERMINED = "REQUIREMENT_DETERMINED"
DESIGNER_SUPPLIED = "DESIGNER_SUPPLIED"

#: Why each fact carries its provenance class. Stated here so the classification is
#: reviewable rather than implicit in the comparison code.
FACT_PROVENANCE: Dict[str, tuple] = {
    "safety_pattern": (
        REQUIREMENT_DETERMINED,
        "the requirement describes a triggered timed response, a startup "
        "inhibit, or a locked-until-authorised release",
    ),
    "timing_origin": (
        REQUIREMENT_DETERMINED,
        "the requirement names the event the deadline runs from",
    ),
    "deadline_seconds": (
        REQUIREMENT_DETERMINED,
        "the requirement states the deadline",
    ),
    "priority_trigger": (
        REQUIREMENT_DETERMINED,
        "the requirement names the condition under which the response wins",
    ),
    "latency_apportionment": (
        DESIGNER_SUPPLIED,
        "the requirement bounds the total, never the per-component split or how "
        "much margin to retain",
    ),
    "response_set_members": (
        DESIGNER_SUPPLIED,
        "'precedence over all other safety responses' does not name them",
    ),
    "selected_response": (
        DESIGNER_SUPPLIED,
        "the winning response's identifier is a naming decision, though which "
        "behaviour must win is requirement-determined",
    ),
    "precedence_edges": (
        DESIGNER_SUPPLIED,
        "the ordering follows from the response set, which the requirement does "
        "not enumerate",
    ),
}


def _facts_from_graph(graph: Any) -> Dict[str, Any]:
    system = getattr(graph, "system", None)
    priority = dict(getattr(graph, "priority", None) or {})
    apportionment = {
        item.name: item.timing_value_literal or (
            str(item.timing_budget) if item.timing_budget is not None else None
        )
        for item in (getattr(graph, "components", ()) or ())
        if item.timing_budget is not None
    }
    return {
        "safety_pattern": getattr(system, "declared_pattern", None),
        "timing_origin": getattr(system, "timing_origin", None),
        "deadline_seconds": (
            getattr(system, "timing_value_literal", None)
            or (str(system.timing_budget) if system and system.timing_budget
                is not None else None)
        ),
        "priority_trigger": priority.get("trigger"),
        # nested under the extracted arbitration topology, not at the top level;
        # reading it from the top level returned None for every model and made the
        # deterministic emitter — which renders the reviewed spec verbatim — report
        # a divergence from it
        "selected_response": (
            (priority.get("arbitration_topology") or {}).get("selection") or {}
        ).get("selected_response"),
        "latency_apportionment": {
            key: apportionment[key] for key in sorted(apportionment)
        } or None,
        "response_set_members": (
            sorted(str(item) for item in (priority.get("members") or ()))
            or None
        ),
        "precedence_edges": sorted(
            f"{edge.get('higher')}>{edge.get('lower')}"
            for edge in (priority.get("edges") or ())
            if isinstance(edge, Mapping)
        ) or None,
    }


def _facts_from_spec(spec: Any) -> Dict[str, Any]:
    priority = getattr(spec, "priority", None)
    apportionment = {
        item.name: (
            str(item.latency_budget) if item.latency_budget is not None else None
        )
        for item in (getattr(spec, "components", ()) or ())
        if item.latency_budget is not None
    }
    return {
        "safety_pattern": getattr(spec, "pattern", None),
        "timing_origin": getattr(spec, "timing_origin", None),
        "deadline_seconds": (
            str(spec.deadline) if getattr(spec, "deadline", None) is not None
            else None
        ),
        "priority_trigger": getattr(priority, "trigger", None),
        "latency_apportionment": {
            key: apportionment[key] for key in sorted(apportionment)
        } or None,
        "response_set_members": (
            sorted(str(item) for item in getattr(priority, "members", ()) or ())
            or None
        ),
        "selected_response": getattr(priority, "selected_response", None),
        "precedence_edges": sorted(
            f"{higher}>{lower}"
            for higher, lower in (getattr(priority, "edges", ()) or ())
        ) or None,
    }


def _comparable(value: Any) -> Any:
    """Numeric strings compare by value, so '0.5' and '0.50' are not a divergence."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    return value


def compute_divergence(graph: Any, reference_spec: Any) -> Dict[str, Any]:
    """Fact-by-fact divergence of a committed model from the reviewed spec."""
    produced = _facts_from_graph(graph)
    reference = _facts_from_spec(reference_spec)

    facts: List[Dict[str, Any]] = []
    for name, (provenance, rationale) in FACT_PROVENANCE.items():
        left, right = produced.get(name), reference.get(name)
        facts.append({
            "fact": name,
            "provenance": provenance,
            "rationale": rationale,
            "produced": left,
            "reviewed": right,
            "diverged": _comparable(left) != _comparable(right),
        })

    determined = [f for f in facts if f["provenance"] == REQUIREMENT_DETERMINED]
    supplied = [f for f in facts if f["provenance"] == DESIGNER_SUPPLIED]
    return {
        "schema_version": DIVERGENCE_SCHEMA_VERSION,
        "artifact_role": "AG_DECISION_DIVERGENCE",
        "source_requirement": getattr(
            getattr(graph, "system", None), "source_requirement", None
        ),
        "metric_interpretation": (
            "divergence from the student-approved decomposition, NOT accuracy. A "
            "REQUIREMENT_DETERMINED divergence is a defect — the text fixes the "
            "answer. A DESIGNER_SUPPLIED divergence is a different design "
            "decision, because the text does not contain the answer at all."
        ),
        "requirement_determined": {
            "facts": len(determined),
            "diverged": sum(1 for f in determined if f["diverged"]),
            "diverged_facts": [f["fact"] for f in determined if f["diverged"]],
        },
        "designer_supplied": {
            "facts": len(supplied),
            "diverged": sum(1 for f in supplied if f["diverged"]),
            "diverged_facts": [f["fact"] for f in supplied if f["diverged"]],
        },
        "facts": facts,
    }


def format_divergence(report: Mapping[str, Any]) -> str:
    lines = [
        f"{'fact':<24} {'provenance':<22} {'diverged':>8}",
        "-" * 56,
    ]
    for fact in report.get("facts", ()):
        lines.append(
            f"{fact['fact']:<24} {fact['provenance']:<22} "
            f"{('YES' if fact['diverged'] else 'no'):>8}"
        )
    determined = report.get("requirement_determined", {})
    supplied = report.get("designer_supplied", {})
    lines += [
        "",
        f"requirement-determined divergences: {determined.get('diverged')}"
        f"/{determined.get('facts')}  (a divergence here is a DEFECT)",
        f"designer-supplied divergences:      {supplied.get('diverged')}"
        f"/{supplied.get('facts')}  (a divergence here is a CHOICE)",
    ]
    return "\n".join(lines)
