"""Requirement-ID normalisation — a neutral utility with no contract-layer deps.

Relocated out of the legacy ``prototyping.contract_types`` module so that
verification and SITL code can normalise requirement identifiers without
importing the external-contract layer (scheduled for Layer-2 excision after the
R2-BBAG checker lands). ``contract_types`` re-exports this function for backward
compatibility with the remaining legacy importers.
"""
from __future__ import annotations


def normalise_req_id(value: str) -> str:
    """Canonical requirement-id form: upper-case with hyphens as underscores."""
    return (value or "").upper().replace("-", "_")
