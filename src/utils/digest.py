"""Shared SHA-256 text fingerprint.

Canonical-JSON digests stay with their owning protocol modules
(``frozen_artifact_protocol``, ``blackboard``): their ``default=`` fallbacks
and failure modes belong to each protocol's contract.
"""
from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Hex SHA-256 of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
