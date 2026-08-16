"""Shared freeze, review and tamper-evidence protocol for human artifacts."""
from __future__ import annotations

from datetime import date
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ..utils.req_id import normalise_req_id


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_artifact_digest(
    artifact: Mapping[str, Any],
    *,
    excluded_fields: Sequence[str] = ("artifact_digest",),
) -> str:
    """Hash canonical JSON after excluding self-referential envelope fields."""
    excluded = frozenset(excluded_fields)
    content = {key: value for key, value in artifact.items() if key not in excluded}
    raw = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "")))


def valid_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def has_review_markers(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(
            str(key).startswith("_") and "review" in str(key)
            for key in value
        ):
            return True
        return any(has_review_markers(item) for item in value.values())
    if isinstance(value, list):
        return any(has_review_markers(item) for item in value)
    return False


def validate_chain_source_binding(artifact: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    chain_id = str(artifact.get("chain_id") or "")
    source_requirement = str(artifact.get("source_requirement") or "")
    if not chain_id:
        problems.append("chain_id must be set")
    if not source_requirement:
        problems.append("source_requirement must be set")
    elif chain_id and normalise_req_id(source_requirement) != chain_id:
        problems.append("chain_id must match the normalised source_requirement")
    return problems


def validate_frozen_envelope(
    artifact: Mapping[str, Any],
    *,
    role: str,
    schema_version: str,
    namespace: str,
    review_flags: Sequence[str],
) -> list[str]:
    """Validate the common human-freeze envelope, not its domain payload."""
    problems: list[str] = []
    if artifact.get("schema_version") != schema_version:
        problems.append(f"schema_version must be {schema_version!r}")
    if artifact.get("artifact_role") != role:
        problems.append(f"artifact_role must be {role!r}")
    if artifact.get("experiment_namespace") != namespace:
        problems.append(f"experiment_namespace must be {namespace!r}")
    if artifact.get("status") != "FROZEN":
        problems.append("status must be 'FROZEN' (still a draft?)")
    problems.extend(validate_chain_source_binding(artifact))
    if not artifact.get("reviewer"):
        problems.append("reviewer must be set to the reviewing supervisor")
    if not valid_iso_date(artifact.get("reviewed_date")):
        problems.append("reviewed_date must be ISO YYYY-MM-DD")
    review = artifact.get("review_protocol")
    if not isinstance(review, Mapping):
        problems.append("review_protocol must be an object")
        review = {}
    for flag in review_flags:
        if review.get(flag) is not True:
            problems.append(f"review_protocol.{flag} must be true")
    return problems
