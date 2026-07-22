"""Immutable stakeholder requirement-input boundary.

Controlled experiments must vary generation, not the stakeholder requirement set.
This module creates and verifies digest-checked frozen requirement artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping


FROZEN_REQUIREMENT_SCHEMA_VERSION = "1.0"
_REQ_ID_RE = re.compile(r"\b(REQ[-_][A-Z]+[-_]\d+)\b", re.IGNORECASE)


def source_digest(text: str) -> str:
    """Stable digest of the stakeholder-owned source text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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
