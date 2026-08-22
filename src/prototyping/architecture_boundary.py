"""Architecture-boundary draft builder and frozen-artifact validator.

Freeze-review finding (round 1, P1): guarantee ownership cannot be derived from the
stakeholder requirement text alone, because the component decomposition
(`propulsionMonitor`, `armingAuthority`, …) is a **design decision**, not part of
the requirement. So the architecture is frozen as its own authoritative artifact,
with full provenance, and the evaluator gold cites its digest. Owner-allocation
gold then measures **conformance to an approved architecture**; it does not, on its
own, become an architecture-generation accuracy metric.

Like the gold this is DRAFT until a supervisor **independently** reviews and
freezes it. It authors nothing from the runtime checker (F3): it imports neither
``ag_extractor`` nor ``ag_contracts``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .ag_emitter import AGChainSpec
from .experiment_arms import REVISED_EXPERIMENT_NAMESPACE
from .frozen_artifact_protocol import (
    canonical_artifact_digest,
    has_review_markers,
    is_sha256,
    validate_frozen_envelope,
)
from ..utils.req_id import normalise_req_id

ARCHITECTURE_BOUNDARY_ROLE = "ARCHITECTURE_BOUNDARY"
BOUNDARY_STATUS_DRAFT = "DRAFT_FOR_SUPERVISOR_REVIEW"
BOUNDARY_STATUS_FROZEN = "FROZEN"
_NAMESPACE = REVISED_EXPERIMENT_NAMESPACE
_SCHEMA_VERSION = "1.0"
def architecture_boundary_digest(boundary: Mapping[str, Any]) -> str:
    """Digest the boundary content, excluding the self-referential digest field.

    The gold cites this digest, so it must be stable and independent of the
    ``artifact_digest`` slot itself.
    """
    return canonical_artifact_digest(boundary)


def build_architecture_boundary_draft(
    spec: AGChainSpec, *, requirement_set_digest: str | None = None
) -> Dict[str, Any]:
    """A review-ready DRAFT architecture boundary for one chain.

    Component ids, interfaces (consumes/produces/trigger), and owner→guarantee
    allocations are pre-filled from the student-approved decomposition candidate
    so the reviewer
    confirms rather than transcribes; each component's *responsibility* is left for
    the reviewer to state as a design decision (not from the stakeholder text).
    """
    components = [
        {
            "component_id": comp.name,
            "owner_usage": comp.owner_usage,
            "owner_def": comp.owner_def,
            "responsibility": None,
            "_responsibility_review": (
                "state this component's single responsibility as a frozen design "
                "decision, not from the stakeholder requirement text"
            ),
            "interfaces": {
                "consumes": list(dict.fromkeys([
                    *(a.concept for a in comp.assumptions),
                    *comp.interface_inputs,
                ])),
                "produces": list(comp.guarantees),
                "trigger": comp.trigger_signal,
            },
        }
        for comp in spec.components
    ]
    allocations = [
        {"owner": comp.owner_usage, "contract": comp.name, "guarantee": guarantee}
        for comp in spec.components
        for guarantee in comp.guarantees
    ]
    return {
        "artifact_role": ARCHITECTURE_BOUNDARY_ROLE,
        "experiment_namespace": _NAMESPACE,
        "schema_version": _SCHEMA_VERSION,
        "status": BOUNDARY_STATUS_DRAFT,
        "chain_id": normalise_req_id(spec.source_requirement),
        "source_requirement": spec.source_requirement,
        "requirement_set_digest": requirement_set_digest,
        "components": components,
        "allocations": allocations,
        "reviewer": None,
        "reviewed_date": None,
        "review_protocol": {"independent_architecture_review": False},
        "review_instructions": (
            "Freeze the component decomposition as a design decision, independent "
            "of the stakeholder requirement and blind to the runtime verdict. "
            "Confirm each component's responsibility and interface boundary and each "
            "owner->guarantee allocation; drop the _review markers; set reviewer / "
            "reviewed_date and independent_architecture_review=true; set "
            f"artifact_digest = architecture_boundary_digest(this); set status "
            f"{BOUNDARY_STATUS_FROZEN!r}. The evaluator gold must cite this digest."
        ),
        "artifact_digest": None,
    }


def validate_frozen_boundary(boundary: Mapping[str, Any]) -> List[str]:
    """Return the freeze-completeness problems of an architecture-boundary file.

    Empty ⇒ the file is a complete, self-consistent frozen boundary: evaluator
    role/namespace, FROZEN status, a named reviewer + date + independent-review
    flag, every component's responsibility stated, non-empty allocations, no
    leftover review markers, and an ``artifact_digest`` that matches the content.
    Structure only — it never authors or second-guesses the architecture (F3).
    """
    problems = validate_frozen_envelope(
        boundary,
        role=ARCHITECTURE_BOUNDARY_ROLE,
        schema_version=_SCHEMA_VERSION,
        namespace=_NAMESPACE,
        review_flags=("independent_architecture_review",),
    )
    requirement_set_digest = str(boundary.get("requirement_set_digest") or "")
    if not is_sha256(requirement_set_digest):
        problems.append("requirement_set_digest must be a lowercase SHA-256 digest")
    components = boundary.get("components")
    if not isinstance(components, list) or not components:
        problems.append("components must be a non-empty list")
        components = []
    component_ids: set[str] = set()
    expected_allocations: set[tuple[str, str, str]] = set()
    for index, comp in enumerate(components):
        if not isinstance(comp, Mapping):
            problems.append(f"components[{index}] must be an object")
            continue
        component_id = str(comp.get("component_id") or "")
        owner_usage = str(comp.get("owner_usage") or "")
        owner_def = str(comp.get("owner_def") or "")
        if not component_id:
            problems.append(f"components[{index}].component_id must be set")
        elif component_id in component_ids:
            problems.append(f"duplicate component_id {component_id!r}")
        component_ids.add(component_id)
        if not owner_usage:
            problems.append(f"{component_id or f'components[{index}]'}: owner_usage missing")
        if not owner_def:
            problems.append(f"{component_id or f'components[{index}]'}: owner_def missing")
        if not comp.get("responsibility"):
            problems.append(
                f"{component_id or f'components[{index}]'}: responsibility must be stated"
            )
        interfaces = comp.get("interfaces")
        if not isinstance(interfaces, Mapping):
            problems.append(
                f"{component_id or f'components[{index}]'}: interfaces must be an object"
            )
            continue
        consumes = interfaces.get("consumes")
        produces = interfaces.get("produces")
        if not isinstance(consumes, list):
            problems.append(f"{component_id}: interfaces.consumes must be a list")
        if not isinstance(produces, list) or not produces:
            problems.append(f"{component_id}: interfaces.produces must be non-empty")
        elif len(produces) != len(set(map(str, produces))):
            problems.append(f"{component_id}: interfaces.produces contains duplicates")
        if "trigger" not in interfaces:
            problems.append(f"{component_id}: interfaces.trigger must be present")
        for guarantee in produces if isinstance(produces, list) else []:
            expected_allocations.add((owner_usage, component_id, str(guarantee)))

    allocations = boundary.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        problems.append("allocations must be non-empty")
        allocations = []
    actual_allocations: set[tuple[str, str, str]] = set()
    for index, allocation in enumerate(allocations):
        if not isinstance(allocation, Mapping):
            problems.append(f"allocations[{index}] must be an object")
            continue
        item = (
            str(allocation.get("owner") or ""),
            str(allocation.get("contract") or ""),
            str(allocation.get("guarantee") or ""),
        )
        if not all(item):
            problems.append(f"allocations[{index}] must set owner/contract/guarantee")
        if item in actual_allocations:
            problems.append(f"duplicate allocation {item!r}")
        actual_allocations.add(item)
        if item[1] and item[1] not in component_ids:
            problems.append(
                f"allocation {item!r} references an unknown component contract"
            )
    if actual_allocations != expected_allocations:
        problems.append(
            "allocations must exactly match component owner_usage/interface guarantees"
        )
    if has_review_markers(boundary):
        problems.append(
            "leftover _review markers remain — drop them after confirming each field"
        )
    digest = boundary.get("artifact_digest")
    if not digest:
        problems.append(
            "artifact_digest must be set to architecture_boundary_digest(this)"
        )
    elif digest != architecture_boundary_digest(boundary):
        problems.append(
            "artifact_digest does not match the content (tampered or stale)"
        )
    return problems
