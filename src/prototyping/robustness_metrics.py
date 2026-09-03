"""Robustness metrics for the bounded A/G assurance enhancement.

A/G assurance makes a generated SysML v2 model more robust - fewer errors, more
complete, better requirement fulfilment. This module quantifies that as a
detection + bounded repair measure:

- DETECTION: the A/G check surfaces incompleteness (undischarged assumptions,
  unrealised guarantees, missing owners, non-conformant patterns) that a
  no-contract pipeline (R0-CURRENT / R1-BBCTX) cannot see at all.
- BOUNDED REPAIR: the dependency-closed surgical loop repairs the
  model-semantic subset it is authorised to; integration/decomposition gaps are
  routed as explicit BLOCKED for human/upstream repair. The delta separates
  auto-repaired from routed-blocked, and is not a full auto-fixing claim.

Computed from an R2 A/G trace (the `_build_ag_trace` result); imports no LLM,
no runtime extractor/checker and no gold. Thesis baseline/evidence code,
exercised by its own tests and invoked on demand rather than wired into the
runtime pipeline. Do not remove as dead code.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .ag_traceability import _fraction

ROBUSTNESS_ROLE = "AG_ROBUSTNESS_METRICS"


def compute_robustness_metrics(assurance: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute the A/G robustness delta from one R2 assurance trace."""
    graph = dict(assurance.get("ag_contract_graph") or {})
    pattern = dict(assurance.get("pattern_conformance_report") or {})
    failures = dict(assurance.get("failure_diagnostics") or {})
    repair = dict(assurance.get("repair_decisions") or {})

    history = list(failures.get("analysis_history") or [])
    failure_list = list(failures.get("failures") or [])
    round0 = history[0] if history else {}
    final = history[-1] if history else {}

    errors_before = len(round0.get("failure_ids") or [])
    errors_after = len(final.get("failure_ids") or [])
    auto_repaired = max(0, errors_before - errors_after)

    by_class: Dict[str, int] = {}
    for item in failure_list:
        key = str(item.get("classification") or "UNCLASSIFIED")
        by_class[key] = by_class.get(key, 0) + 1

    decisions = list(repair.get("decisions") or [])
    routed_blocked = sum(
        1 for d in decisions if str(d.get("status")).upper() == "BLOCKED"
    )

    completeness = graph.get("component_completeness") or {}
    ready = sum(1 for v in completeness.values() if str(v).upper() == "READY")
    inner = graph.get("graph") or {}
    discharge_edges = list(inner.get("discharge_edges") or [])
    discharged = sum(
        1 for e in discharge_edges
        if e.get("by") not in (None, "", "undischarged")
    )
    realization_links = list(inner.get("realization_links") or [])
    realized = sum(
        1 for r in realization_links if str(r.get("status")).upper() == "PASS"
    )

    return {
        "artifact_role": ROBUSTNESS_ROLE,
        "measurement_boundary": "INTERVENTION",
        "measurement": "detection + bounded repair (NOT full auto-fix)",
        "detection": {
            "errors_detected": errors_before,
            "by_class": by_class,
            "note": (
                "R0-CURRENT / R1-BBCTX carry no A/G contracts and cannot detect "
                "these; surfacing hidden A/G incompleteness is itself a robustness "
                "gain"
            ),
        },
        "repair": {
            "analysis_rounds": len(history),
            "auto_repaired": auto_repaired,
            "routed_blocked": routed_blocked,
            "verdict_initial": round0.get("verdict"),
            "verdict_final": final.get("verdict"),
        },
        "completeness_final": {
            "components_ready": _fraction(ready, len(completeness)),
            "assumption_discharge": _fraction(discharged, len(discharge_edges)),
            "guarantee_realization": _fraction(realized, len(realization_links)),
        },
        "requirement_fulfillment_final": {
            "ag_verdict": graph.get("verdict"),
            "pattern_conformance": pattern.get("verdict"),
        },
        "robustness_delta": {
            "errors_before_assurance_repair": errors_before,
            "errors_after_assurance_repair": errors_after,
            "errors_resolved_by_auto_repair": auto_repaired,
            "errors_routed_for_human_upstream": routed_blocked,
        },
    }
