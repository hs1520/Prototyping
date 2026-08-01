"""Shared policy primitives for deterministic SysML text writers.

This module does not parse or normalise authored model text. It owns choices
made while structured input is being emitted, before any text exists.
"""
from __future__ import annotations

import re
from enum import Enum


class EmissionMode(str, Enum):
    """Select which facts a structured emitter is authorised to assert."""

    COMPLETE = "complete"
    CONTRACTS_ONLY = "contracts_only"


def identifier(
    value: str,
    *,
    fallback: str,
    max_length: int | None = None,
) -> str:
    """Spell a structured source name as its historical SysML identifier."""
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not token:
        token = fallback
    elif token[0].isdigit():
        token = "_" + token
    return token[:max_length] if max_length is not None else token
