"""Terminal qualification gate for generated SysML v2 prototypes.

Qualification is separate from the continuous quality score, so a high score
cannot hide a syntax error, a missing planned connection or a failed
safety-assurance chain.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..utils.req_id import first_req_id, normalise_req_id


def _declared_requirement_ids(requirements: Sequence[str]) -> list[str]:
    result: list[str] = []
    for requirement in requirements:
        match = first_req_id(str(requirement))
        if match:
            req_id = normalise_req_id(match)
            if req_id not in result:
                result.append(req_id)
    return result


def build_model_qualification(
    *,
    model_text: str,
    requirements: Sequence[str],
    syntax_result: Any,
    simulation_result: Any,
    terminal_consistency: Mapping[str, Any],
    structural_obligation_report: Mapping[str, Any] | None = None,
    semantic_fidelity_report: Mapping[str, Any] | None = None,
    generation_plan_conformance: Mapping[str, Any] | None = None,
    ag_contract_graph: Mapping[str, Any] | None = None,
    pattern_conformance_report: Mapping[str, Any] | None = None,
    ag_binding_report: Mapping[str, Any] | None = None,
    ag_non_degradation: Mapping[str, Any] | None = None,
    ag_expected: bool = False,
    generation_plan_expected: bool = False,
    semantic_fidelity_expected: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, evidence: Mapping[str, Any]) -> None:
        checks.append({
            "name": name,
            "status": (
                "NOT_APPLICABLE"
                if passed is None else ("PASS" if passed else "FAIL")
            ),
            "evidence": dict(evidence),
        })

    syntax_errors = int(syntax_result.total_errors())
    syntax_warnings = list(getattr(syntax_result, "warnings", ()) or ())
    parser_errors = list(
        getattr(syntax_result, "parser_errors", ()) or ()
    )
    semantic_errors = list(
        getattr(syntax_result, "sema_errors", ()) or ()
    )
    add(
        "SYSML_SYNTAX_AND_SEMANTICS",
        syntax_errors == 0 and not syntax_warnings,
        {
        "error_count": syntax_errors,
        "parser_error_count": len(parser_errors),
        "semantic_error_count": len(semantic_errors),
        "warning_count": len(syntax_warnings),
        "warning_codes": sorted({
            str(item.get("code") or "unknown")
            for item in syntax_warnings
            if isinstance(item, Mapping)
        }),
        "parser_errors": parser_errors,
        "semantic_errors": semantic_errors,
        "warnings": syntax_warnings,
        "policy": (
            "fail closed on every parser error, semantic error, and warning "
            "reported for the committed user model"
        ),
    })
    from .namespace_integrity import check_user_namespace_integrity

    namespace_integrity = check_user_namespace_integrity(model_text)
    add(
        "USER_NAMESPACE_INTEGRITY",
        namespace_integrity["status"] == "PASS",
        namespace_integrity,
    )

    add(
        "TERMINAL_EVIDENCE_SAME_REVISION",
        terminal_consistency.get("status") == "PASS",
        {"status": terminal_consistency.get("status")},
    )

    requirement_ids = _declared_requirement_ids(requirements)
    missing_requirement_defs = [
        req_id for req_id in requirement_ids
        if not re.search(
            rf"\brequirement\s+def\s+{re.escape(req_id)}\b", model_text
        )
    ]
    missing_satisfy = [
        req_id for req_id in requirement_ids
        if not re.search(
            rf"\bsatisfy\s+requirement\b[^;\n]*\b{re.escape(req_id)}\b",
            model_text,
        )
    ]
    add(
        "REQUIREMENT_REALIZATION_COVERAGE",
        not missing_requirement_defs and not missing_satisfy,
        {
            "requirements": len(requirement_ids),
            "missing_requirement_defs": missing_requirement_defs,
            "missing_satisfy": missing_satisfy,
        },
    )

    if generation_plan_conformance is None:
        add(
            "TYPED_GENERATION_PLAN_CONFORMANCE",
            False if generation_plan_expected else None,
            {"reason": "no typed whole-model plan attached"},
        )
    else:
        add(
            "TYPED_GENERATION_PLAN_CONFORMANCE",
            generation_plan_conformance.get("status") == "PASS",
            generation_plan_conformance,
        )
        reserved_identity = generation_plan_conformance.get(
            "ag_reserved_identity_conformance"
        )
        if (
            isinstance(reserved_identity, Mapping)
            and reserved_identity.get("status") != "NOT_APPLICABLE"
        ):
            add(
                "A_G_RESERVED_IDENTITY_CONFORMANCE",
                reserved_identity.get("status") == "PASS",
                reserved_identity,
            )
        else:
            add(
                "A_G_RESERVED_IDENTITY_CONFORMANCE",
                None,
                {"reason": "no applicable frozen A/G identities"},
            )
        event_symbols = generation_plan_conformance.get(
            "planned_event_symbol_conformance"
        )
        if (
            isinstance(event_symbols, Mapping)
            and event_symbols.get("status") != "NOT_APPLICABLE"
        ):
            add(
                "PLANNED_EVENT_SYMBOL_CONFORMANCE",
                event_symbols.get("status") == "PASS",
                event_symbols,
            )
        else:
            add(
                "PLANNED_EVENT_SYMBOL_CONFORMANCE",
                None,
                {"reason": "no applicable frozen event identities"},
            )

    if structural_obligation_report is not None:
        add(
            "REQUIREMENT_STRUCTURAL_OBLIGATIONS",
            structural_obligation_report.get("status") == "PASS",
            structural_obligation_report,
        )
    else:
        add(
            "REQUIREMENT_STRUCTURAL_OBLIGATIONS",
            False if generation_plan_expected else None,
            {"reason": "no frozen structural obligation report"},
        )
    if semantic_fidelity_report is None:
        add(
            "REQUIREMENT_MODEL_SEMANTIC_FIDELITY",
            False if semantic_fidelity_expected else None,
            {"reason": "no frozen semantic fidelity report"},
        )
    elif semantic_fidelity_report.get("status") == "UNVERIFIED":
        add(
            "REQUIREMENT_MODEL_SEMANTIC_FIDELITY",
            None,
            semantic_fidelity_report,
        )
    else:
        add(
            "REQUIREMENT_MODEL_SEMANTIC_FIDELITY",
            semantic_fidelity_report.get("status") == "PASS",
            semantic_fidelity_report,
        )
    scenarios = list(
        getattr(simulation_result, "scenario_results", ()) or ()
    )
    passed_scenarios = list(simulation_result.passed_scenarios())
    advisory_evidence_factory = getattr(
        simulation_result, "advisory_structural_evidence", None
    )
    advisory_evidence = (
        advisory_evidence_factory()
        if callable(advisory_evidence_factory) else {
            "evidence_kind": "LEGACY_ADVISORY",
            "qualification_effect": "NONE",
            "role_assignments": {},
            "weakly_connected_components": [],
            "scenarios": [],
        }
    )
    checks.append({
        "name": "HEURISTIC_STRUCTURAL_DIAGNOSTIC",
        "status": "ADVISORY",
        "evidence": {
            "passed": len(passed_scenarios),
            "total": len(scenarios),
            "reachability_score": simulation_result.reachability_score,
            **advisory_evidence,
        },
    })

    behavioral = getattr(simulation_result, "behavioral_result", None)
    extracted = int(getattr(behavioral, "extracted_sm_count", 0) or 0)
    behavioral_scenarios = list(
        getattr(behavioral, "scenario_results", ()) or ()
    )
    activated_constraint_report = (
        generation_plan_conformance.get(
            "activated_constraint_conformance"
        )
        if isinstance(generation_plan_conformance, Mapping)
        else None
    )
    plan_aware_behavior = isinstance(
        activated_constraint_report, Mapping
    )

    def tagged(tag: str) -> list[Any]:
        return [
            item for item in behavioral_scenarios
            if tag in (getattr(item, "tags", ()) or ())
        ]

    def add_execution_check(name: str, selected: Sequence[Any]) -> None:
        if not selected:
            add(name, None, {"reason": "no applicable planned scenarios"})
            return
        failed_items = [
            item for item in selected if not getattr(item, "passed", False)
        ]
        add(name, not failed_items, {
            "passed": len(selected) - len(failed_items),
            "total": len(selected),
            "failed_scenarios": [
                str(getattr(item, "name", "unknown"))
                for item in failed_items
            ],
        })

    if plan_aware_behavior:
        requirement_scenarios = tagged("requirement_behavior")
        ag_scenarios = tagged("ag_behavior")
        design_scenarios = [
            item for item in tagged("design_constraint")
            if item not in requirement_scenarios
            and item not in ag_scenarios
        ]
        add_execution_check(
            "REQUIREMENT_BEHAVIOR_EXECUTION",
            requirement_scenarios,
        )
        add_execution_check("A_G_BEHAVIOR_EXECUTION", ag_scenarios)
        add_execution_check(
            "DESIGN_CONSTRAINT_CONSISTENCY",
            design_scenarios,
        )
        readiness = activated_constraint_report.get(
            "external_verification_readiness"
        )
        if (
            isinstance(readiness, Mapping)
            and readiness.get("status") != "NOT_APPLICABLE"
        ):
            add(
                "EXTERNAL_VERIFICATION_READINESS",
                readiness.get("status") == "PASS",
                readiness,
            )
        else:
            add("EXTERNAL_VERIFICATION_READINESS", None, {
                "reason": "no external-analysis constraint is planned",
            })
        checks.append({
            "name": "BEHAVIORAL_EXECUTION",
            "status": "ADVISORY",
            "evidence": {
                "extracted_state_machines": extracted,
                "passed": sum(
                    1 for item in behavioral_scenarios
                    if getattr(item, "passed", False)
                ),
                "total": len(behavioral_scenarios),
                "qualification_effect": (
                    "NONE; replaced by provenance-specific checks"
                ),
            },
        })
    elif behavioral is not None and extracted:
        failed_behavioral = (
            list(behavioral.failed_scenarios())
            if callable(getattr(behavioral, "failed_scenarios", None))
            else [item for item in behavioral_scenarios
                  if not getattr(item, "passed", False)]
        )
        add("BEHAVIORAL_EXECUTION", not failed_behavioral, {
            "extracted_state_machines": extracted,
            "passed": len(behavioral_scenarios) - len(failed_behavioral),
            "total": len(behavioral_scenarios),
        })
    else:
        add("BEHAVIORAL_EXECUTION", None, {
            "reason": "no executable state-machine scenarios",
        })

    if ag_contract_graph is not None:
        add("BOUNDED_A_G_ASSURANCE", ag_contract_graph.get("verdict") == "PASS", {
            "verdict": ag_contract_graph.get("verdict"),
            "checker_version": ag_contract_graph.get("checker_version"),
        })
    else:
        add("BOUNDED_A_G_ASSURANCE", None, {
            "reason": "no applicable bounded A/G layer",
        })

    if pattern_conformance_report is not None:
        add(
            "SAFETY_PATTERN_CONFORMANCE",
            pattern_conformance_report.get("verdict") == "PASS",
            {"verdict": pattern_conformance_report.get("verdict")},
        )
    else:
        add("SAFETY_PATTERN_CONFORMANCE", None, {
            "reason": "no applicable safety-pattern layer",
        })

    if ag_binding_report is not None:
        add(
            "A_G_TERMINAL_BINDING",
            ag_binding_report.get("status") == "PASS",
            ag_binding_report,
        )
    elif ag_expected:
        add("A_G_TERMINAL_BINDING", False, {
            "reason": "A/G was planned but no terminal binding evidence exists",
        })
    else:
        add("A_G_TERMINAL_BINDING", None, {
            "reason": "no applicable bounded A/G layer",
        })

    if ag_non_degradation is not None:
        add(
            "A_G_NON_DEGRADATION",
            ag_non_degradation.get("status") == "PASS",
            ag_non_degradation,
        )
    elif ag_expected:
        add("A_G_NON_DEGRADATION", False, {
            "reason": "A/G was planned but no non-degradation comparison exists",
        })
    else:
        add("A_G_NON_DEGRADATION", None, {
            "reason": "no applicable bounded A/G layer",
        })

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    return {
        "schema_version": "1.0",
        "artifact_role": "TERMINAL_MODEL_QUALIFICATION",
        "status": "QUALIFIED" if not failed else "NOT_QUALIFIED",
        "score_semantics": (
            "qualification is a hard gate; final_score ranks quality but cannot "
            "override a failed qualification check"
        ),
        "failed_checks": failed,
        "checks": checks,
    }
