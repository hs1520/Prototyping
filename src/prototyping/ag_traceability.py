"""Per-requirement implementation traceability — gold-free (design §13, §9).

Answers the question the project actually asks: *for each requirement, is it
implemented, and how completely?* Every link in the chain is a property of the
committed model itself, so none of it needs human gold or blind review:

  requirement -> contract -> decomposition -> ownership -> discharge
             -> realizing behaviour -> verification observation

This is deliberately not an accuracy measure. It never compares against a reviewed
answer, so it cannot tell you the decomposition is *right*; it tells you whether the
implementation chain is *there*. That is what "more robust" means here — fewer
missing links, more of the requirement actually carried into the model — and it is
measurable on every arm, including the ones that carry no A/G contracts at all and
therefore trace nothing.

The reason this exists as its own measure: allocation and discharge agreement
against frozen gold turned out to be largely determined by the architecture
boundary (see `docs/R2_GENERATION_FINDINGS.md` §2), so more human freezing buys
little. Traceability is the part of the claim that is both load-bearing and free.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

TRACEABILITY_SCHEMA_VERSION = "1.1"

#: The links checked for every requirement, in dependency order. A requirement is
#: fully traced only when all of them hold.
TRACE_LINKS = (
    "contract_present",
    "decomposed_to_components",
    "guarantees_owned",
    "assumptions_discharged",
    "behaviours_realized",
    "observation_linked",
)


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _link_ok(value: Any) -> bool:
    """A link holds when it is True, or a fraction that reached 1.0.

    A link with nothing to measure (denominator 0) does not hold: a requirement
    whose contract owns no guarantee at all has not been traced to an
    implementation, it has simply been left empty.
    """
    if isinstance(value, bool):
        return value
    return value == 1.0


def trace_requirement(graph: Any, realization_links: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """Traceability of one extracted chain, keyed by its source requirement."""
    system = getattr(graph, "system", None)
    components = list(getattr(graph, "components", ()) or ())
    edges = list(getattr(graph, "edges", ()) or ())
    requirement = getattr(system, "source_requirement", None) if system else None

    guarantees = [
        guarantee
        for item in components
        for guarantee in getattr(item, "guarantees", ())
    ]
    owned = sum(
        len(getattr(item, "guarantees", ()))
        for item in components
        if len(set(getattr(item, "owners", ()))) == 1
    )
    decomposed = sum(
        1
        for item in components
        if system is not None
        and sum(
            getattr(edge, "kind", "") == "decomposes"
            and getattr(edge, "src", "") == system.name
            and getattr(edge, "dst", "") == item.name
            for edge in edges
        ) == 1
    )
    assumptions = [
        assumption
        for item in components
        for assumption in getattr(item, "assumptions", ())
    ]
    produced_by = {
        str(concept).strip().lower(): item.name
        for item in components
        for concept in item.boolean_guarantee_concepts()
    }
    explicit_discharge = {
        (
            str(getattr(edge, "src", "")),
            str(getattr(edge, "dst", "")),
            str(getattr(edge, "subject", "") or "").strip().lower(),
        )
        for edge in edges
        if getattr(edge, "kind", "") == "discharges"
    }
    discharged = sum(
        1
        for item in components
        for assumption in getattr(item, "assumptions", ())
        if (
            getattr(assumption, "is_environment", False)
            or (
                (
                    produced_by.get(
                        str(assumption.concept).strip().lower()
                    ),
                    item.name,
                    str(assumption.concept).strip().lower(),
                )
                in explicit_discharge
            )
        )
    )
    realized_names = {
        str(link.get("contract"))
        for link in realization_links
        if (
            isinstance(link, Mapping)
            and link.get("behavior")
            and link.get("status") == "PASS"
        )
    }
    realized = sum(1 for item in components if item.name in realized_names)
    # The link is the observe EDGE, not the presence of a verification element:
    # a `verification def` nobody points at observes nothing, and checking only
    # for its existence made this link true for a model with the dependency
    # deleted.
    verification_targets = dict(getattr(graph, "verification_targets", None) or {})
    observation_linked = any(
        getattr(edge, "kind", "") == "observed_by"
        and getattr(edge, "src", "") == getattr(system, "name", None)
        and getattr(system, "name", None)
        in verification_targets.get(getattr(edge, "dst", ""), ())
        for edge in edges
    )

    links: Dict[str, Any] = {
        "contract_present": system is not None,
        "decomposed_to_components": bool(components)
        and decomposed == len(components),
        "guarantees_owned": _fraction(owned, len(guarantees)),
        "assumptions_discharged": _fraction(discharged, len(assumptions)),
        "behaviours_realized": _fraction(realized, len(components)),
        "observation_linked": observation_linked,
    }
    complete = [name for name in TRACE_LINKS if _link_ok(links[name])]
    return {
        "source_requirement": requirement,
        "links": links,
        "links_complete": len(complete),
        "links_total": len(TRACE_LINKS),
        "trace_completeness": _fraction(len(complete), len(TRACE_LINKS)),
        "fully_traced": len(complete) == len(TRACE_LINKS),
        "missing_links": [name for name in TRACE_LINKS if name not in complete],
    }


#: A requirement is in scope for the bounded A/G layer when it instantiates one of
#: the encoded safety patterns: an event-triggered timed response, or a state
#: invariant (startup inhibit, locked-until-authorised release). A continuous
#: control envelope — "maintain at least 5 metres of separation while avoiding it" —
#: has no trigger, no deadline and no invariant state, and is not something this
#: layer is built to decompose.
#:
#: Scope is a DECLARED design decision, never inferred from the text here. An
#: undeclared requirement counts as in scope, so excluding one always requires an
#: explicit recorded decision and the denominator cannot be quietly shrunk.
IN_SCOPE = "IN_SCOPE"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

#: The declarations themselves, for the frozen requirement set this project runs.
#: Kept here as DATA with its reason attached, so a shrunk denominator is always
#: reviewable: the reason is what a reader checks, not the number.
#:
#: Recorded 2026-07-26. Until then the runner passed no declaration at all, so a
#: requirement the layer is not built to decompose was reported as an
#: implementation gap (3/4 = 0.75) — the mirror image of scoring an arm 0.00 for
#: carrying no A/G layer, and wrong for the same reason.
DECLARED_OUT_OF_SCOPE: Mapping[str, str] = {
    "REQ_FUNC_002": (
        "continuous control envelope (maintain separation while avoiding an "
        "obstacle): no trigger event, no deadline and no invariant state, so it "
        "instantiates none of the three encoded bounded safety patterns"
    ),
}


def compute_traceability(
    graphs: Sequence[Any],
    *,
    realization_links: Sequence[Mapping[str, Any]] = (),
    declared_requirements: Sequence[str] = (),
    out_of_scope: Mapping[str, str] = {},
) -> Dict[str, Any]:
    """Traceability across every selected chain, plus the requirements with none.

    ``declared_requirements`` is the set the run was asked to implement. A
    requirement that produced no A/G chain at all is the most important case and
    the easiest to hide, so it is reported explicitly as untraced rather than
    silently excluded from the denominator — which is exactly how an arm that
    carries no contracts would otherwise appear to score perfectly.
    """
    traced = [trace_requirement(graph, realization_links) for graph in graphs]
    # A chain whose provenance line is missing cannot be attributed to any
    # requirement. It must not occupy a row of its own *and* leave the requirement
    # it belongs to counted as untraced, or the denominator inflates and every
    # score drops for a single defect. Report it separately; the requirement it
    # should have cited is reported untraced below, which is the honest reading.
    unattributed = [item for item in traced if not item["source_requirement"]]
    per_requirement: List[Dict[str, Any]] = [
        item for item in traced if item["source_requirement"]
    ]
    traced_ids = {item["source_requirement"] for item in per_requirement}
    # An out-of-scope requirement is not an implementation gap: counting it would
    # penalise a requirement for lacking a mechanism it was never eligible for —
    # the same error as scoring R0/R1 at 0.0 for carrying no A/G layer.
    excluded = [
        str(item) for item in declared_requirements if str(item) in out_of_scope
    ]
    in_scope = [
        str(item) for item in declared_requirements if str(item) not in out_of_scope
    ]
    untraced = [item for item in in_scope if item not in traced_ids]
    for requirement in untraced:
        per_requirement.append({
            "source_requirement": requirement,
            "links": {name: False for name in TRACE_LINKS},
            "links_complete": 0,
            "links_total": len(TRACE_LINKS),
            "trace_completeness": 0.0,
            "fully_traced": False,
            "missing_links": list(TRACE_LINKS),
        })

    total = len(per_requirement)
    fully = sum(1 for item in per_requirement if item["fully_traced"])
    scores = [
        item["trace_completeness"] for item in per_requirement
        if item["trace_completeness"] is not None
    ]
    return {
        "schema_version": TRACEABILITY_SCHEMA_VERSION,
        "artifact_role": "REQUIREMENT_TRACEABILITY",
        "measurement_boundary": (
            "committed model only; no human gold, no blind review, no comparison "
            "against a reviewed answer"
        ),
        "metric_interpretation": (
            "presence and completeness of the requirement->implementation chain, "
            "NOT correctness of the decomposition"
        ),
        "requirements": total,
        "fully_traced": fully,
        "fully_traced_rate": _fraction(fully, total),
        "mean_trace_completeness": (
            round(sum(scores) / len(scores), 4) if scores else None
        ),
        "untraced_requirements": untraced,
        # reported, never silently dropped: an excluded requirement must be visible
        # with the reason it was excluded, or the denominator is unauditable
        "out_of_scope_requirements": [
            {"requirement": item, "reason": out_of_scope[item]} for item in excluded
        ],
        # Kept in full, not just counted: a chain can carry most of its links and
        # still be untraceable because it never cites a requirement. Losing that
        # detail would make "0.0" look like nothing was built, when what actually
        # failed was provenance — which is a different defect with a different fix.
        "unattributed_chains": unattributed,
        "per_requirement": per_requirement,
    }
