"""Increment-3 safety-pattern conformance and typed A/G failure routing.

This layer consumes only the graph extracted from committed SysML and the
runtime check report. It does not load evaluator gold and it does not assert a
formal Assume/Guarantee proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Tuple

from .ag_contracts import AGDiagnostic, AGGraph, AGReport


class FailureClass(str, Enum):
    CONTRACT_INCOMPLETENESS = "CONTRACT_INCOMPLETENESS"
    INTEGRATION_DECOMPOSITION_GAP = "INTEGRATION_DECOMPOSITION_GAP"
    MODEL_SEMANTIC_FAULT = "MODEL_SEMANTIC_FAULT"
    ARCHITECTURE_DESIGN_ISSUE = "ARCHITECTURE_DESIGN_ISSUE"
    VERIFIER_LIMITATION = "VERIFIER_LIMITATION"


class FailureRoute(str, Enum):
    CLARIFICATION_OR_BLOCKED = "CLARIFICATION_OR_BLOCKED"
    UPSTREAM_INTEGRATION_REPAIR = "UPSTREAM_INTEGRATION_REPAIR"
    DEPENDENCY_CLOSED_SURGICAL_REPAIR = "DEPENDENCY_CLOSED_SURGICAL_REPAIR"
    ARCHITECTURE_ANALYSIS_OR_DSE = "ARCHITECTURE_ANALYSIS_OR_DSE"
    PARTIAL_OR_UNASSIGNED = "PARTIAL_OR_UNASSIGNED"


@dataclass(frozen=True)
class PatternCase:
    contract: str
    pattern: str
    trigger_present: bool
    reachable_response: bool
    entry_action_present: bool
    timing_criterion_present: bool
    invariant_preserved: bool

    @property
    def status(self) -> str:
        return "PASS" if all((
            self.trigger_present,
            self.reachable_response,
            self.entry_action_present,
            self.timing_criterion_present,
            self.invariant_preserved,
        )) else "FAIL"


def check_safety_pattern_conformance(
    graph: AGGraph, report: AGReport
) -> Dict[str, Any]:
    """Check the bounded triggered timed-failsafe topology for REQ_SAFE_005."""
    by_contract = {
        item["contract"]: item for item in report.realization_links
    }
    cases = []
    for component in graph.components:
        realization = by_contract.get(component.name, {})
        reachable = list(realization.get("reachable_states") or ())
        actions = list(realization.get("response_actions") or ())
        case = PatternCase(
            contract=component.name,
            pattern="TRIGGERED_TIMED_FAILSAFE_RESPONSE",
            trigger_present=bool(realization.get("trigger_ok")),
            reachable_response=len(reachable) >= 2,
            entry_action_present=bool(actions),
            timing_criterion_present=(component.timing_budget is not None),
            # In the bounded profile the invariant is that no response PASS is
            # possible without a reachable trigger and response entry action.
            invariant_preserved=(
                bool(realization.get("trigger_ok")) and bool(actions)
            ),
        )
        cases.append({**asdict(case), "status": case.status})
    verdict = "PASS" if cases and all(c["status"] == "PASS" for c in cases) else "FAIL"
    return {
        "schema_version": "1.0",
        "artifact_role": "INTERVENTION_PATTERN_CONFORMANCE",
        "producing_stage": "R2_PATTERN_CONFORMANCE",
        "measurement_boundary": "INTERVENTION",
        "source_model_revision": report.revision,
        "source_model_digest": report.model_digest,
        "selected_scope": report.source_requirement,
        "formal_proof": False,
        "verdict": verdict,
        "cases": cases,
    }


_CONTRACT_CODES = {
    "CONTRACT_INCOMPLETE", "CONTRACT_UNSUPPORTED",
    "SOURCE_PROVENANCE_MISSING",
}
_INTEGRATION_CODES = {
    "GUARANTEE_NO_OWNER", "GUARANTEE_MULTIPLE_OWNERS",
    "DECOMPOSITION_MISSING", "ASSUMPTION_UNDISCHARGED",
    "CIRCULAR_ASSUMPTION", "DISCHARGE_EDGE_MISSING",
    "DECOMPOSITION_INSUFFICIENT",
}
_MODEL_CODES = {
    "REALIZATION_MISSING", "REALIZATION_UNREACHABLE",
    "REALIZATION_TRIGGER_MISSING", "REALIZATION_ACTION_MISSING",
    "PATTERN_NONCONFORMANT",
}
_DESIGN_CODES = {
    "UNIT_INCOMPATIBLE", "TIMING_BUDGET_EXCEEDED", "TIMING_BUDGET_MISSING",
}
_VERIFIER_CODES = {"OBSERVATION_MISSING"}


def _classification(code: str) -> Tuple[FailureClass, FailureRoute, bool]:
    if code in _MODEL_CODES:
        return (
            FailureClass.MODEL_SEMANTIC_FAULT,
            FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR,
            True,
        )
    if code in _INTEGRATION_CODES:
        return (
            FailureClass.INTEGRATION_DECOMPOSITION_GAP,
            FailureRoute.UPSTREAM_INTEGRATION_REPAIR,
            False,
        )
    if code in _DESIGN_CODES:
        return (
            FailureClass.ARCHITECTURE_DESIGN_ISSUE,
            FailureRoute.ARCHITECTURE_ANALYSIS_OR_DSE,
            False,
        )
    if code in _VERIFIER_CODES:
        return (
            FailureClass.VERIFIER_LIMITATION,
            FailureRoute.PARTIAL_OR_UNASSIGNED,
            False,
        )
    return (
        FailureClass.CONTRACT_INCOMPLETENESS,
        FailureRoute.CLARIFICATION_OR_BLOCKED,
        False,
    )


def route_failure_diagnostics(
    diagnostics: Iterable[AGDiagnostic],
    *,
    source_requirement: str | None,
    realization_links: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    behavior_by_contract = {
        str(item.get("contract")): item.get("behavior")
        for item in realization_links
    }
    failures = []
    for index, diagnostic in enumerate(
        diagnostics, 1
    ):
        failure_class, route, authorised = _classification(diagnostic.code)
        affected = [item for item in (
            diagnostic.contract,
            behavior_by_contract.get(str(diagnostic.contract)),
        ) if item]
        failures.append({
            "failure_id": f"failure-{index:04d}",
            "diagnostic_code": diagnostic.code,
            "message": diagnostic.message,
            "source_requirement": source_requirement,
            "contract": diagnostic.contract,
            "subject": diagnostic.subject,
            "classification": failure_class.value,
            "route": route.value,
            "repair_authorized": authorised,
            "affected_elements": affected,
            "provenance": dict(diagnostic.provenance),
        })
    return {
        "schema_version": "1.0",
        "artifact_role": "INTERVENTION_FAILURE_ROUTING",
        "producing_stage": "R2_FAILURE_ROUTING",
        "measurement_boundary": "INTERVENTION",
        "failures": failures,
    }
