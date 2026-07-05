"""Process-local visibility for intentionally suppressed exceptions.

The pipeline deliberately keeps many best-effort phases non-fatal. This module
preserves that control flow while making those fallbacks visible in run reports.
"""
from __future__ import annotations

from typing import Any, Dict

_SUPPRESSED: Dict[str, Dict[str, Any]] = {}


def record_suppressed(where: str, exc: BaseException) -> None:
    """Record a swallowed exception without ever raising from the recorder."""
    try:
        key = str(where or "unknown")
        try:
            exc_type = type(exc).__name__
        except Exception:
            exc_type = "Exception"
        try:
            message = str(exc)
        except Exception:
            message = "<unprintable exception>"
        last = f"{exc_type}: {message[:200]}"
        entry = _SUPPRESSED.setdefault(key, {"count": 0, "last": ""})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last"] = last
    except Exception:
        pass


def suppressed_summary() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the suppressed-exception summary."""
    try:
        return {
            str(where): {"count": int(data.get("count", 0)), "last": str(data.get("last", ""))}
            for where, data in _SUPPRESSED.items()
        }
    except Exception:
        return {}


def reset_suppressed() -> None:
    """Clear the process-local recorder."""
    try:
        _SUPPRESSED.clear()
    except Exception:
        pass
