"""Requirement-ID normalisation, with no contract-layer dependencies.

Moved here in the Layer-2 excision so verification and SITL code can normalise
requirement identifiers without the removed external-contract modules.
"""
from __future__ import annotations

import re
from typing import Iterator, Sequence


_REQ_ID = re.compile(r"\bREQ[-_][A-Za-z]+[-_]\d+\b", re.IGNORECASE)


def normalise_req_id(value: str) -> str:
    """Canonical requirement-id form: upper-case with hyphens as underscores."""
    return (value or "").upper().replace("-", "_")


def iter_req_ids(value: str) -> Iterator[str]:
    """Yield requirement IDs in source order using their original spelling."""
    yield from _REQ_ID.findall(str(value or ""))


def first_req_id(value: str) -> str | None:
    """Return the first requirement ID in *value*, or ``None``."""
    match = _REQ_ID.search(str(value or ""))
    return match.group(0) if match else None


def strip_req_ids(value: str) -> str:
    """Remove requirement IDs while leaving the surrounding source text intact."""
    return _REQ_ID.sub("", str(value or ""))


def source_requirements_by_id(requirements: Sequence[str]) -> dict[str, str]:
    """Return canonical requirement ids mapped to their complete source text."""
    result: dict[str, str] = {}
    for raw in requirements:
        source = str(raw or "").strip()
        for match in iter_req_ids(source):
            result[normalise_req_id(match)] = source
    return result
