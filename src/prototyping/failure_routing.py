"""Deterministic Option 2 failure classification and repair authorization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contract_types import INCOMPLETE, UNSUPPORTED, contract_bundle_from_dict


REQUIREMENT_AMBIGUITY = "REQUIREMENT_AMBIGUITY"
MODEL_SEMANTIC_FAULT = "MODEL_SEMANTIC_FAULT"
DESIGN_OR_REALIZATION_NONCOMPLIANCE = "DESIGN_OR_REALIZATION_NONCOMPLIANCE"
VERIFIER_LIMITATION = "VERIFIER_LIMITATION"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
NO_FAILURE = "NO_FAILURE"


@dataclass(frozen=True)
class Diagnostic:
    diagnostic_id: str
    req_id: str
    source_stage: str
    finding_code: str
    expected: Mapping[str, Any]
    observed: Mapping[str, Any]
    evidence_refs: tuple[str, ...] = ()
    affected_elements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    req_id: str
    failure_class: str
    repair_authorised: bool
    route: str
    repair_scope: tuple[str, ...]
    protected_artifacts: tuple[str, ...]
    rationale_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_failure(
    req_id: str,
    *,
    contract_status: str,
    diagnostics: Iterable[Diagnostic] = (),
    verifier_available: bool = True,
    infrastructure_ok: bool = True,
    oracle_executed: bool = False,
    oracle_passed: bool | None = None,
) -> RouteDecision:
    """Apply the declared fail-closed classification order without an LLM."""
    findings = tuple(diagnostics)
    codes = tuple(dict.fromkeys(item.finding_code for item in findings))
    affected = tuple(dict.fromkeys(
        element for item in findings for element in item.affected_elements
    ))
    protected = (
        "requirement_source", "requirement_digest", "contract",
        "platform_binding", "verification_oracle", "unaffected_ast_elements",
    )

    if contract_status == INCOMPLETE:
        return RouteDecision(
            req_id, REQUIREMENT_AMBIGUITY, False, "clarification/block", (),
            protected, codes or ("CONTRACT_INCOMPLETE",),
        )
    if contract_status == UNSUPPORTED:
        return RouteDecision(
            req_id, VERIFIER_LIMITATION, False, "PARTIAL/UNASSIGNED", (),
            protected, codes or ("CONTRACT_UNSUPPORTED",),
        )
    if findings:
        return RouteDecision(
            req_id, MODEL_SEMANTIC_FAULT, True, "bounded_surgical_repair",
            affected, protected, codes,
        )
    if not infrastructure_ok:
        return RouteDecision(
            req_id, INFRASTRUCTURE_FAILURE, False, "retry/inconclusive", (),
            protected, ("EXECUTION_INFRASTRUCTURE_FAILED",),
        )
    if not verifier_available:
        return RouteDecision(
            req_id, VERIFIER_LIMITATION, False, "PARTIAL/UNASSIGNED", (),
            protected, ("VERIFIER_UNAVAILABLE",),
        )
    if oracle_executed and oracle_passed is False:
        return RouteDecision(
            req_id, DESIGN_OR_REALIZATION_NONCOMPLIANCE, False,
            "DSE/realization/configuration_report", (), protected,
            ("ENGINEERING_ORACLE_FAILED",),
        )
    return RouteDecision(
        req_id, NO_FAILURE, False, "preserve_result", (), protected, (),
    )


def route_report(decisions: Iterable[RouteDecision]) -> dict[str, Any]:
    items = tuple(decisions)
    counts: dict[str, int] = {}
    for item in items:
        counts[item.failure_class] = counts.get(item.failure_class, 0) + 1
    return {"decisions": [item.to_dict() for item in items], "counts": counts}


def diagnostic_from_dict(value: Mapping[str, Any]) -> Diagnostic:
    return Diagnostic(
        diagnostic_id=str(value.get("diagnostic_id", "")),
        req_id=str(value.get("req_id", "")),
        source_stage=str(value.get("source_stage", "semantic_trace")),
        finding_code=str(value.get("finding_code", "UNKNOWN")),
        expected=dict(value.get("expected", {})),
        observed=dict(value.get("observed", {})),
        evidence_refs=tuple(value.get("evidence_refs", ())),
        affected_elements=tuple(value.get("affected_elements", ())),
    )


def classify_bundle_evidence(
    contract_bundle: Mapping[str, Any] | Any,
    semantic_trace_report: Mapping[str, Any] | Any,
    evidence_by_req: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-run routing after terminal behavioral/SITL/Gazebo evidence exists."""
    bundle = contract_bundle_from_dict(contract_bundle)
    traces = getattr(semantic_trace_report, "traces", ())
    if isinstance(semantic_trace_report, Mapping):
        traces = semantic_trace_report.get("traces", ())
    findings_by_req: dict[str, tuple[Diagnostic, ...]] = {}
    for trace in traces:
        if isinstance(trace, Mapping):
            req_id = str(trace.get("req_id", ""))
            findings_by_req[req_id] = tuple(
                diagnostic_from_dict(item) for item in trace.get("findings", ())
            )
        else:
            findings_by_req[str(getattr(trace, "req_id", ""))] = tuple(
                getattr(trace, "findings", ())
            )
    decisions = []
    for contract in bundle.contracts:
        evidence = evidence_by_req.get(contract.req_id, {})
        decisions.append(classify_failure(
            contract.req_id,
            contract_status=contract.completeness,
            diagnostics=findings_by_req.get(contract.req_id, ()),
            verifier_available=bool(evidence.get("verifier_available", True)),
            infrastructure_ok=bool(evidence.get("infrastructure_ok", True)),
            oracle_executed=bool(evidence.get("oracle_executed", False)),
            oracle_passed=evidence.get("oracle_passed"),
        ))
    return route_report(decisions)
