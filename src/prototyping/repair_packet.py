"""Build minimal, authoritative context for Option 2 semantic repair calls."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any, Iterable, Mapping

from .contract_types import contract_bundle_from_dict, normalise_req_id
from .platform_semantics import PLATFORM_BINDING_VERSION, platform_binding
from .safety_patterns import PatternBinding, pattern_by_id


REPAIR_PACKET_SCHEMA_VERSION = "1.0"


def _trace_dict(trace: Any) -> dict[str, Any]:
    if isinstance(trace, Mapping):
        return dict(trace)
    to_dict = getattr(trace, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else {}


def _semantic_issue_req_ids(issues: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for issue in issues:
        text = str(issue)
        if not text.startswith("[SEMANTIC-TRACE]"):
            continue
        result.update(
            normalise_req_id(match)
            for match in re.findall(
                r"\bREQ(?:[_-][A-Z0-9]+){2,}\b", text, re.IGNORECASE
            )
        )
    return result


def build_scoped_repair_packet(
    contract_bundle: Any,
    semantic_trace_report: Any,
    pattern_bindings: Iterable[PatternBinding] = (),
    *,
    issues: Iterable[str] = (),
) -> dict[str, Any]:
    """Select only failing requirements and their authoritative repair context.

    The complete contract library and trace report remain local.  The returned
    packet contains the smallest source-grounded slice needed by the LLM, plus
    a digest that can be recorded with the repair attempt.
    """
    bundle = contract_bundle_from_dict(contract_bundle)
    traces = getattr(semantic_trace_report, "traces", ())
    if isinstance(semantic_trace_report, Mapping):
        traces = semantic_trace_report.get("traces", ())
    trace_dicts = [_trace_dict(trace) for trace in traces]
    failing_ids = {
        normalise_req_id(trace.get("req_id", ""))
        for trace in trace_dicts
        if trace.get("findings")
    }
    requested_ids = _semantic_issue_req_ids(issues)
    target_ids = failing_ids & requested_ids if requested_ids else failing_ids
    target_ids.discard("")
    if not target_ids:
        return {}

    selected_contracts = [
        contract for contract in bundle.contracts if contract.req_id in target_ids
    ]
    contracts = [contract.to_dict() for contract in selected_contracts]
    selected_traces = [
        trace
        for trace in trace_dicts
        if normalise_req_id(trace.get("req_id", "")) in target_ids
    ]

    affected_elements: set[str] = set()
    for trace in selected_traces:
        for finding in trace.get("findings", ()):
            affected_elements.update(
                str(item) for item in finding.get("affected_elements", ()) if item
            )

    pattern_constraints: list[dict[str, Any]] = []
    for binding in pattern_bindings:
        if hasattr(binding, "to_dict"):
            binding_dict = binding.to_dict()
        elif isinstance(binding, Mapping):
            binding_dict = dict(binding)
        else:
            continue
        req_id = normalise_req_id(binding_dict.get("req_id", ""))
        if req_id not in target_ids:
            continue
        pattern = pattern_by_id(str(binding_dict.get("pattern_id", "")))
        pattern_constraints.append({
            "binding": binding_dict,
            "pattern": (
                {
                    "pattern_id": pattern.pattern_id,
                    "version": pattern.version,
                    "required_topology": asdict(pattern.required_topology),
                    "required_response_concepts": list(
                        pattern.required_response_concepts
                    ),
                    "invariants": [asdict(item) for item in pattern.invariants],
                    "verification_capabilities": list(
                        pattern.verification_capabilities
                    ),
                    "prompt_guidance": pattern.prompt_guidance,
                }
                if pattern is not None else None
            ),
        })

    platform_bindings: list[dict[str, Any]] = []
    bound_concepts: set[str] = set()
    for contract in selected_contracts:
        for obligation in contract.obligations:
            concept = obligation.response.concept if obligation.response else ""
            if not concept or concept in bound_concepts:
                continue
            binding = platform_binding(concept)
            if binding is not None:
                platform_bindings.append(binding.to_dict())
                bound_concepts.add(concept)

    packet: dict[str, Any] = {
        "artifact_type": "SCOPED_SEMANTIC_REPAIR_PACKET",
        "schema_version": REPAIR_PACKET_SCHEMA_VERSION,
        "contract_schema_version": bundle.schema_version,
        "contract_library_version": bundle.library_version,
        "platform_binding_version": PLATFORM_BINDING_VERSION,
        "scope": {
            "req_ids": sorted(target_ids),
            "affected_elements": sorted(affected_elements),
            "scope_rule": (
                "Edit only the affected top-level blocks, or the minimum owner "
                "block required to restore a missing trace link."
            ),
        },
        "contracts": contracts,
        "traces": selected_traces,
        "platform_bindings": platform_bindings,
        "pattern_constraints": pattern_constraints,
        "edit_policy": {
            "immutable": [
                "requirement source_text and source_digest",
                "contract trigger, response, criterion, envelope and qualifiers",
                "platform binding and verification intent",
                "previously passing trace links and model behavior",
            ],
            "required_result": (
                "Reduce the listed semantic diagnostics without introducing new "
                "diagnostics, syntax failures, simulation regressions, lost "
                "connections, lost satisfy links or score regression."
            ),
        },
    }
    canonical = json.dumps(
        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    packet["packet_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return packet
