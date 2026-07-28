"""User-defined namespace integrity checks for generated SysML v2.

Syside reports namespace-distinguishability warnings, but a blanket warning gate
also catches unrelated standard-library shadowing.  This module checks the
unambiguous defect we own: two definitions with the same name in the same
user-defined namespace.  Definitions in different owning scopes remain legal.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from ..utils.sysml_text_utils import find_block_end


_DEFINITION_RE = re.compile(
    r"\b(?P<kind>"
    r"package|"
    r"(?:action|attribute|constraint|enum|item|part|port|requirement|state|"
    r"verification)\s+def"
    r")\s+(?P<name>[A-Za-z_]\w*)"
)


def _mask_comments_and_strings(text: str) -> str:
    """Preserve offsets while hiding braces and declarations in prose."""
    result = list(text)
    index = 0
    mode: str | None = None
    while index < len(text):
        if mode == "block":
            if text.startswith("*/", index):
                result[index:index + 2] = "  "
                index += 2
                mode = None
            else:
                if text[index] != "\n":
                    result[index] = " "
                index += 1
            continue
        if mode == "line":
            if text[index] == "\n":
                mode = None
            else:
                result[index] = " "
            index += 1
            continue
        if mode == "string":
            if text[index] == "\\" and index + 1 < len(text):
                result[index:index + 2] = "  "
                index += 2
            elif text[index] == '"':
                result[index] = " "
                index += 1
                mode = None
            else:
                if text[index] != "\n":
                    result[index] = " "
                index += 1
            continue
        if text.startswith("/*", index):
            result[index:index + 2] = "  "
            index += 2
            mode = "block"
        elif text.startswith("//", index):
            result[index:index + 2] = "  "
            index += 2
            mode = "line"
        elif text[index] == '"':
            result[index] = " "
            index += 1
            mode = "string"
        else:
            index += 1
    return "".join(result)


def _depths(masked: str) -> list[int]:
    depth = 0
    result: list[int] = []
    for char in masked:
        result.append(depth)
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
    return result


def _scan_namespace(
    source: str,
    masked: str,
    *,
    scope: tuple[str, ...],
    findings: list[dict[str, Any]],
) -> None:
    depths = _depths(masked)
    direct = [
        match for match in _DEFINITION_RE.finditer(masked)
        if depths[match.start()] == 0
    ]
    by_name: dict[str, list[re.Match[str]]] = {}
    for match in direct:
        by_name.setdefault(match.group("name"), []).append(match)

    for name, matches in sorted(by_name.items()):
        if len(matches) < 2:
            continue
        findings.append({
            "scope": "::".join(scope) if scope else "<root>",
            "name": name,
            "kinds": [match.group("kind") for match in matches],
            "count": len(matches),
        })

    for match in direct:
        brace = masked.find("{", match.end())
        semicolon = masked.find(";", match.end())
        if brace == -1 or (semicolon != -1 and semicolon < brace):
            continue
        closing = find_block_end(masked, brace)
        if closing == -1:
            continue
        _scan_namespace(
            source[brace + 1:closing],
            masked[brace + 1:closing],
            scope=(*scope, match.group("name")),
            findings=findings,
        )


def check_user_namespace_integrity(model_text: str) -> dict[str, Any]:
    """Reject duplicate definition names in the same user-owned scope."""
    text = str(model_text or "")
    findings: list[dict[str, Any]] = []
    _scan_namespace(
        text,
        _mask_comments_and_strings(text),
        scope=(),
        findings=findings,
    )
    return {
        "schema_version": "1.0",
        "artifact_role": "USER_NAMESPACE_INTEGRITY",
        "status": "PASS" if not findings else "FAIL",
        "source_model_digest": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "duplicate_members": findings,
    }
