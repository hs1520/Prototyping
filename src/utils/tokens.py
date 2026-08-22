"""Crude token estimation shared by transcript and session accounting."""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text* at ~4 characters per token, minimum 1."""
    return max(1, len(text) // 4)
