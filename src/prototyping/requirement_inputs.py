"""Immutable requirement-input and approved-contract boundaries.

Controlled robustness experiments must vary generation, not the stakeholder
requirement set.  This module creates/verifies frozen requirement artifacts and
keeps human-approved contracts distinct from evaluation gold datasets.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .contract_types import ContractBundle, contract_bundle_from_dict, source_digest


FROZEN_REQUIREMENT_SCHEMA_VERSION = "1.0"
APPROVED_CONTRACT_PROTOCOL_VERSION = "1.0"
_REQ_ID_RE = re.compile(r"\b(REQ[-_][A-Z]+[-_]\d+)\b", re.IGNORECASE)


def normalise_requirement_id(value: str) -> str:
    match = _REQ_ID_RE.search(value or "")
    token = match.group(1) if match else value
    return (token or "").upper().replace("-", "_")


def requirement_records(requirements: Iterable[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(requirements):
        text = str(raw)
        req_id = normalise_requirement_id(text)
        if not _REQ_ID_RE.search(text):
            raise ValueError(f"frozen requirement[{index}] has no valid requirement id")
        if req_id in seen:
            raise ValueError(f"frozen requirements contain duplicate id {req_id}")
        seen.add(req_id)
        records.append({
            "req_id": req_id,
            "source_text": text,
            "source_digest": source_digest(text),
        })
    if not records:
        raise ValueError("frozen requirement set must not be empty")
    return records


def requirement_set_digest(requirements: Iterable[str]) -> str:
    """Order-sensitive digest of exact IDs and stakeholder-owned source text."""
    records = requirement_records(requirements)
    payload = json.dumps(
        [(item["req_id"], item["source_text"]) for item in records],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_frozen_requirement_set(
    requirements: Iterable[str],
    *,
    name: str = "frozen-requirements",
    source: str = "manual",
) -> dict[str, Any]:
    items = list(requirements)
    records = requirement_records(items)
    return {
        "schema_version": FROZEN_REQUIREMENT_SCHEMA_VERSION,
        "artifact_type": "FROZEN_REQUIREMENT_SET",
        "name": name,
        "source": source,
        "requirement_set_digest": requirement_set_digest(items),
        "requirements": [item["source_text"] for item in records],
        "source_digests": {
            item["req_id"]: item["source_digest"] for item in records
        },
    }


def resolve_frozen_requirement_set(
    value: Iterable[str] | Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate a frozen artifact or wrap an explicit list as one."""
    if isinstance(value, Mapping):
        if value.get("artifact_type") != "FROZEN_REQUIREMENT_SET":
            raise ValueError("frozen input must have artifact_type=FROZEN_REQUIREMENT_SET")
        if value.get("schema_version") != FROZEN_REQUIREMENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported frozen requirement schema_version: "
                f"{value.get('schema_version')!r}"
            )
        requirements = [str(item) for item in value.get("requirements", ())]
        canonical = build_frozen_requirement_set(
            requirements,
            name=str(value.get("name") or "frozen-requirements"),
            source=str(value.get("source") or "manual"),
        )
        if value.get("requirement_set_digest") != canonical["requirement_set_digest"]:
            raise ValueError("frozen requirement_set_digest does not match contents")
        if value.get("source_digests") != canonical["source_digests"]:
            raise ValueError("frozen requirement source_digests do not match contents")
        return requirements, canonical
    requirements = [str(item) for item in value]
    return requirements, build_frozen_requirement_set(requirements)


def validate_approved_contract_bundle(
    value: Mapping[str, Any],
    requirements: Iterable[str],
) -> tuple[ContractBundle, dict[str, Any]]:
    """Validate a human-approved contract bundle without accepting gold files."""
    if value.get("review_mode") or value.get("artifact_type") == "GOLD_DATASET":
        raise ValueError(
            "evaluation gold cannot be used as a pipeline contract input; "
            "provide an approved_contract_bundle artifact"
        )
    approval = value.get("approval")
    if not isinstance(approval, Mapping):
        raise ValueError("approved_contract_bundle requires approval metadata")
    if approval.get("protocol_version") != APPROVED_CONTRACT_PROTOCOL_VERSION:
        raise ValueError("unsupported approved-contract protocol_version")
    if approval.get("status") != "APPROVED":
        raise ValueError("approved_contract_bundle status must be APPROVED")
    if not approval.get("reviewer") or not approval.get("approved_at"):
        raise ValueError("approved_contract_bundle requires reviewer and approved_at")

    source_items = list(requirements)
    expected_set_digest = requirement_set_digest(source_items)
    if approval.get("requirement_set_digest") != expected_set_digest:
        raise ValueError("approved contract bundle targets a different requirement set")
    source_by_id = {
        item["req_id"]: item for item in requirement_records(source_items)
    }
    bundle = contract_bundle_from_dict(value)
    if not bundle.contracts:
        raise ValueError("approved_contract_bundle must contain at least one contract")
    contract_ids = [contract.req_id for contract in bundle.contracts]
    if len(contract_ids) != len(set(contract_ids)):
        raise ValueError("approved_contract_bundle contains duplicate requirement ids")
    for contract in bundle.contracts:
        source = source_by_id.get(contract.req_id)
        if source is None:
            raise ValueError(
                f"approved contract {contract.req_id} is absent from frozen requirements"
            )
        if contract.source_text != source["source_text"]:
            raise ValueError(
                f"approved contract {contract.req_id} changes stakeholder source text"
            )
        if contract.source_digest != source["source_digest"]:
            raise ValueError(
                f"approved contract {contract.req_id} source digest mismatch"
            )
    provenance = {
        "mode": "approved_contract_bundle",
        "protocol_version": APPROVED_CONTRACT_PROTOCOL_VERSION,
        "reviewer": str(approval["reviewer"]),
        "approved_at": str(approval["approved_at"]),
        "requirement_set_digest": expected_set_digest,
        "approved_requirement_ids": contract_ids,
    }
    return bundle, provenance


def build_approved_contract_bundle(
    bundle: ContractBundle | Mapping[str, Any],
    requirements: Iterable[str],
    *,
    reviewer: str,
    approved_at: str,
) -> dict[str, Any]:
    """Attach explicit human approval metadata, then validate the artifact."""
    if not reviewer.strip() or not approved_at.strip():
        raise ValueError("reviewer and approved_at must be non-empty")
    raw = bundle.to_dict() if isinstance(bundle, ContractBundle) else dict(bundle)
    source_items = list(requirements)
    raw["approval"] = {
        "protocol_version": APPROVED_CONTRACT_PROTOCOL_VERSION,
        "status": "APPROVED",
        "reviewer": reviewer.strip(),
        "approved_at": approved_at.strip(),
        "requirement_set_digest": requirement_set_digest(source_items),
    }
    validate_approved_contract_bundle(raw, source_items)
    return raw
