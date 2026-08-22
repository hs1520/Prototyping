"""Shared SHA-256 text fingerprint.

Canonical-JSON digests deliberately stay with their owning protocol modules
(``frozen_artifact_protocol``, ``blackboard``): their ``default=`` fallbacks and
failure modes are part of each protocol's contract, not a generic utility.
"""
from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Hex SHA-256 of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
