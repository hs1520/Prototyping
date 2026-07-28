"""Terminal qualification gate for generated SysML v2 prototypes.

Qualification is deliberately separate from the continuous quality score:
a high weighted score cannot hide a syntax error, a missing planned connection,
or a failed applicable safety-assurance chain.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..utils.req_id import normalise_req_id


_REQ_ID = re.compile(r"\bREQ[-_][A-Za-z]+[-_]\d+\b", re.IGNORECASE)


def _declared_requirement_ids(requirements: Sequence[str]) -> list[str]:
    result: list[str] = []
    for requirement in requirements:
        match = _REQ_ID.search(str(requirement))
        if match:
            req_id = normalise_req_id(match.group(0))
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
    generation_plan_conformance: Mapping[str, Any] | None = None,
    ag_contract_graph: Mapping[str, Any] | None = None,
    pattern_conformance_report: Mapping[str, Any] | None = None,
    ag_binding_report: Mapping[str, Any] | None = None,
    ag_non_degradation: Mapping[str, Any] | None = None,
    ag_expected: bool = False,
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
    add("SYSML_SYNTAX_AND_SEMANTICS", syntax_errors == 0, {
        "error_count": syntax_errors,
    })

    digest_values = {
        terminal_consistency.get("model_digest"),
        terminal_consistency.get("simulation_source_model_digest"),
        terminal_consistency.get("evaluation_source_model_digest"),
    }
    add(
        "TERMINAL_EVIDENCE_SAME_REVISION",
        terminal_consistency.get("status") == "PASS"
        and len(digest_values) == 1
        and None not in digest_values,
        {"digests": sorted(str(item) for item in digest_values)},
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
        add("TYPED_GENERATION_PLAN_CONFORMANCE", None, {
            "reason": "no typed whole-model plan attached",
        })
    else:
        add(
            "TYPED_GENERATION_PLAN_CONFORMANCE",
            generation_plan_conformance.get("status") == "PASS",
            generation_plan_conformance,
        )

    scenarios = list(
        getattr(simulation_result, "scenario_results", ()) or ()
    )
    passed_scenarios = list(simulation_result.passed_scenarios())
    if scenarios:
        add(
            "STRUCTURAL_REACHABILITY",
            len(passed_scenarios) == len(scenarios),
            {
                "passed": len(passed_scenarios),
                "total": len(scenarios),
                "reachability_score": simulation_result.reachability_score,
            },
        )
    else:
        add("STRUCTURAL_REACHABILITY", None, {
            "reason": "no applicable structural scenarios",
        })

    behavioral = getattr(simulation_result, "behavioral_result", None)
    extracted = int(getattr(behavioral, "extracted_sm_count", 0) or 0)
    behavioral_scenarios = list(
        getattr(behavioral, "scenario_results", ()) or ()
    )
    if behavioral is not None and extracted:
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
