"""Uniform, read-only evaluation of completed B0/B1/B2 models.

The post-hoc evaluator is deliberately outside the generation intervention:
it receives a frozen requirement artifact and an already completed model, uses
one versioned measurement stack for every configuration, and never returns a
model or invokes an LLM/repair path.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .contract_types import ContractBundle, contract_bundle_from_dict
from .failure_routing import classify_failure
from .platform_semantics import PLATFORM_BINDING_VERSION
from .requirement_contracts import build_contract_bundle, contract_summary
from .requirement_inputs import (
    requirement_records,
    resolve_frozen_requirement_set,
)
from .safety_patterns import (
    PATTERN_LIBRARY_VERSION,
    bindings_to_dict,
    select_patterns,
)
from .semantic_trace import build_semantic_trace
from ..simulation.controlled_scenarios import evaluate_controlled_scenarios


POSTHOC_SCHEMA_VERSION = "1.1"


def _validated_measurement_bundle(
    requirements: list[str],
    supplied: ContractBundle | Mapping[str, Any] | None,
) -> ContractBundle:
    bundle = (
        contract_bundle_from_dict(supplied)
        if supplied is not None
        else build_contract_bundle(requirements)
    )
    expected = {
        item["req_id"]: item["source_digest"]
        for item in requirement_records(requirements)
    }
    observed = {item.req_id: item.source_digest for item in bundle.contracts}
    if observed != expected:
        raise ValueError(
            "post-hoc measurement contracts must cover the exact frozen "
            "requirement set with matching source digests"
        )
    return bundle


def build_uniform_posthoc_evaluation(
    *,
    model_text: str,
    model_name: str,
    frozen_requirements: Iterable[str] | Mapping[str, Any],
    measurement_contract_bundle: ContractBundle | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a completed model without mutating or feeding back into it."""
    requirements, frozen = resolve_frozen_requirement_set(frozen_requirements)
    bundle = _validated_measurement_bundle(
        requirements, measurement_contract_bundle
    )
    patterns = select_patterns(bundle)
    trace = build_semantic_trace(
        model_text,
        bundle,
        pattern_bindings=patterns,
        model_name=model_name,
    )
    trace_dict = trace.to_dict()
    controlled_scenarios = evaluate_controlled_scenarios(
        model_text, model_name=model_name
    )
    diagnostics = [item.to_dict() for item in trace.diagnostics()]
    decisions = []
    trace_by_req = {item.req_id: item for item in trace.traces}
    for contract in bundle.contracts:
        item = classify_failure(
            contract.req_id,
            contract_status=contract.completeness,
            diagnostics=tuple(trace_by_req[contract.req_id].findings),
        ).to_dict()
        # A common evaluator may identify a repair candidate, but it must not
        # authorise or perform a mutation in any experimental arm.
        item["repair_eligible"] = bool(item.pop("repair_authorised"))
        item["recommended_route"] = item.pop("route")
        item["intervention_applied"] = False
        decisions.append(item)
    classification_counts: dict[str, int] = {}
    for item in decisions:
        failure_class = str(item["failure_class"])
        classification_counts[failure_class] = (
            classification_counts.get(failure_class, 0) + 1
        )

    provenance = {
        "artifact_role": "UNIFORM_POSTHOC_MEASUREMENT",
        "measurement_only": True,
        "mutation_permitted": False,
        "independent_of_run_intervention": True,
        "requirement_set_digest": frozen["requirement_set_digest"],
        "model_digest": trace.model_digest,
        "contract_library_version": bundle.library_version,
        "pattern_library_version": PATTERN_LIBRARY_VERSION,
        "platform_binding_version": PLATFORM_BINDING_VERSION,
    }
    contract_artifact = bundle.to_dict()
    contract_artifact.update(provenance)
    binding_artifact = bindings_to_dict(patterns)
    binding_artifact.update(provenance)
    trace_dict.update(provenance)
    supported = sum(
        trace_dict["counts"].get(status, 0)
        for status in ("PASS", "FAIL", "BLOCKED")
    )
    passed = trace_dict["counts"].get("PASS", 0)
    return {
        "schema_version": POSTHOC_SCHEMA_VERSION,
        "artifact_type": "OPTION2_UNIFORM_POSTHOC_EVALUATION",
        **provenance,
        "requirement_input": frozen,
        "requirement_contracts": contract_artifact,
        "safety_pattern_bindings": binding_artifact,
        "semantic_trace_report": trace_dict,
        "controlled_scenario_evaluation": controlled_scenarios,
        "failure_diagnostics": {
            **provenance,
            "diagnostics": diagnostics,
        },
        "failure_classifications": {
            **provenance,
            "decisions": decisions,
            "counts": classification_counts,
        },
        "metrics": {
            **provenance,
            "contract_counts": contract_summary(bundle)["counts"],
            "pattern_binding_count": len(patterns),
            "supported_trace_count": supported,
            "semantic_trace_pass_count": passed,
            "semantic_trace_pass_rate": passed / supported if supported else None,
            "diagnostic_count": len(diagnostics),
            "controlled_scenario_count": controlled_scenarios["scenario_count"],
            "controlled_scenario_pass_count": controlled_scenarios["counts"]["PASS"],
            "controlled_scenario_pass_rate": controlled_scenarios["pass_rate"],
            "failure_classification_counts": classification_counts,
        },
    }
