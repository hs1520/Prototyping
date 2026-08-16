"""Shared text-level utilities for SysML source manipulation."""
from __future__ import annotations

import re


def find_block_end(text: str, start: int) -> int:
    """Return the index of the closing '}' matching the '{' at *start*.

    Scans forward from *start*, tracking brace depth.
    Returns -1 if no matching closing brace is found.
    """
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def get_sysml_text(model) -> str:
    """Return SysML text for *model*, preferring the cached metadata string.

    Falls back to ``model.to_sysml_text()`` when metadata is absent.
    """
    return (
        (getattr(model, "metadata", None) or {}).get("last_sysml_text")
        or model.to_sysml_text()
    )


def set_sysml_text(model, text: str) -> None:
    """Replace the authoritative cached SysML text on *model*."""
    if getattr(model, "metadata", None) is None:
        model.metadata = {}
    model.metadata["last_sysml_text"] = str(text)


def remove_named_package(model_text: str, package_name: str) -> str:
    """Remove every balanced ``package <name> { ... }`` block."""
    pattern = re.compile(rf"\bpackage\s+{re.escape(package_name)}\s*\{{")
    text = str(model_text)
    while True:
        match = pattern.search(text)
        if match is None:
            return text
        brace = text.find("{", match.start())
        end = find_block_end(text, brace)
        if end == -1:
            return text
        text = text[:match.start()] + text[end + 1:]
