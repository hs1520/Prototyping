"""Requirement-ID normalisation — a neutral utility with no contract-layer deps.

Relocated here during the Layer-2 excision so that verification and SITL code can
normalise requirement identifiers without the removed external-contract modules.
"""
from __future__ import annotations


def normalise_req_id(value: str) -> str:
    """Canonical requirement-id form: upper-case with hyphens as underscores."""
    return (value or "").upper().replace("-", "_")
