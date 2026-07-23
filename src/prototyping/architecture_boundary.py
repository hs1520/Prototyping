"""Independently reviewed architecture boundary for the bounded A/G chains.

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

import hashlib
import json
from typing import Any, Dict, List, Mapping

from .ag_emitter import AGChainSpec
from ..utils.req_id import normalise_req_id

ARCHITECTURE_BOUNDARY_ROLE = "ARCHITECTURE_BOUNDARY"
BOUNDARY_STATUS_DRAFT = "DRAFT_FOR_SUPERVISOR_REVIEW"
BOUNDARY_STATUS_FROZEN = "FROZEN"
_NAMESPACE = "BLACKBOARD_AG_V1"
_SCHEMA_VERSION = "1.0"


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _has_review_markers(obj: Any) -> bool:
    if isinstance(obj, dict):
        if any(str(k).startswith("_") and "review" in str(k) for k in obj):
            return True
        return any(_has_review_markers(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_review_markers(v) for v in obj)
    return False


def architecture_boundary_digest(boundary: Mapping[str, Any]) -> str:
    """Digest the boundary content, excluding the self-referential digest field.

    The gold cites this digest, so it must be stable and independent of the
    ``artifact_digest`` slot itself.
    """
    content = {k: v for k, v in boundary.items() if k != "artifact_digest"}
    return _canonical_digest(content)


def build_architecture_boundary_draft(
    spec: AGChainSpec, *, requirement_set_digest: str | None = None
) -> Dict[str, Any]:
    """A review-ready DRAFT architecture boundary for one chain.

    Component ids, interfaces (consumes/produces/trigger), and owner→guarantee
    allocations are pre-filled from the reviewed decomposition so the reviewer
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
                "consumes": [a.concept for a in comp.assumptions],
                "produces": [comp.guarantee],
                "trigger": comp.trigger_signal,
            },
        }
        for comp in spec.components
    ]
    allocations = [
        {"owner": comp.owner_usage, "contract": comp.name, "guarantee": comp.guarantee}
        for comp in spec.components
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
    problems: List[str] = []
    if boundary.get("artifact_role") != ARCHITECTURE_BOUNDARY_ROLE:
        problems.append(f"artifact_role must be {ARCHITECTURE_BOUNDARY_ROLE!r}")
    if boundary.get("experiment_namespace") != _NAMESPACE:
        problems.append(f"experiment_namespace must be {_NAMESPACE!r}")
    if boundary.get("status") != BOUNDARY_STATUS_FROZEN:
        problems.append(f"status must be {BOUNDARY_STATUS_FROZEN!r} (still a draft?)")
    if not boundary.get("reviewer"):
        problems.append("reviewer must be set to the reviewing supervisor")
    if not boundary.get("reviewed_date"):
        problems.append("reviewed_date must be set")
    review = boundary.get("review_protocol") or {}
    if review.get("independent_architecture_review") is not True:
        problems.append("review_protocol.independent_architecture_review must be true")
    if not boundary.get("allocations"):
        problems.append("allocations must be non-empty")
    for comp in boundary.get("components") or []:
        if not comp.get("responsibility"):
            problems.append(
                f"{comp.get('component_id')}: responsibility must be stated"
            )
    if _has_review_markers(boundary):
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
