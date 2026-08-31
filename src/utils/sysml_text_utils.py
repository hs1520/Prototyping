"""Shared text-level utilities for SysML source manipulation."""
from __future__ import annotations

import re

#: A bare SysML identifier (whole-string match via ``fullmatch``).
IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
#: Header tail between a definition name and its body brace. Tolerates
#: specialization (``:> Super``) and any other header text, but never crosses a
#: ``;`` (a bodiless declaration), another brace, or a NEWLINE. The DSE
#: variation layer emits ``part def LidarSuite :> SensorSuite { ... }``, so a
#: lookup that requires the brace to follow the name immediately silently
#: misses real blocks in variated models. The newline bound matters just as
#: much: prose such as ``// "Every part def MUST have >= 1 satisfy link"``
#: otherwise mints a phantom def named ``MUST`` that swallows the NEXT block's
#: braces (observed in two archived pilot models). Emitters write single-line
#: headers, so the bound costs nothing.
_DEF_HEADER_TAIL = r"\b[^{;\n]*\{"
#: ``part def <Name> ... {`` — named part definition with an opening body brace.
PART_DEF_RE = re.compile(r"\bpart\s+def\s+(\w+)" + _DEF_HEADER_TAIL)
#: ``state def <Name> ... {`` — named state definition with an opening body brace.
STATE_DEF_RE = re.compile(r"\bstate\s+def\s+(\w+)" + _DEF_HEADER_TAIL)


def named_def_pattern(kind: str, name: str) -> re.Pattern[str]:
    """``<kind> def <name> ... {`` for one specific definition name.

    The single supertype-tolerant convention for locating a named definition's
    body. ``kind`` may be an alternation such as ``"(?:part|item)"``.
    """
    return re.compile(rf"\b{kind}\s+def\s+{re.escape(name)}{_DEF_HEADER_TAIL}")


def named_block_span(text: str, kind: str, name: str) -> tuple[int, int] | None:
    """``(opening_brace, closing_brace)`` of ``<kind> def <name> { ... }``.

    Returns ``None`` when the definition is absent, bodiless, or unbalanced.
    The body is ``text[opening + 1:closing]``.
    """
    match = named_def_pattern(kind, name).search(text)
    if match is None:
        return None
    opening = match.end() - 1
    closing = find_block_end(text, opening)
    if closing == -1:
        return None
    return opening, closing


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


_SEMANTIC_STOP_WORDS = frozenset({
    "a", "all", "an", "and", "any", "condition", "during", "if", "in", "is",
    "of", "or", "state", "system", "the", "to", "when", "whenever", "while",
})


def semantic_terms(text: str) -> frozenset[str]:
    """Content words of an identifier or phrase (camelCase split, stopworded).

    Lives in utils so that packages below `prototyping` in the dependency
    order (simulation's verification binding, dse) can match identities by
    meaning without importing that package. Re-exported from
    prototyping.verification_obligations for its existing callers.
    """
    separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("-", " ")
    return frozenset(
        token for token in re.findall(r"[A-Za-z]+", separated.lower())
        if token not in _SEMANTIC_STOP_WORDS
    )


_PART_DEF_BASES_RE = re.compile(
    r"\bpart\s+def\s+(\w+)\s*:>\s*([\w:,\s]+?)\s*[{;]"
)


def part_def_bases(model_text: str) -> dict[str, tuple[str, ...]]:
    """Map each part definition to the definitions it specialises (`:>`)."""
    bases: dict[str, tuple[str, ...]] = {}
    for match in _PART_DEF_BASES_RE.finditer(model_text or ""):
        names = tuple(
            item.strip().rsplit("::", 1)[-1]
            for item in match.group(2).split(",")
            if item.strip()
        )
        if names:
            bases[match.group(1)] = names
    return bases


PASSIVE_MARKER_RE = re.compile(
    r"(?m)^[ \t]*//\s*PLAN-PASSIVE\s+(?P<name>[A-Za-z_]\w*)\s*:"
)


def passive_components_in_text(sysml_text: str) -> set[str]:
    """Part-definition names the committed model itself declares passive.

    The declaration is a `// PLAN-PASSIVE <PartDef>: <reason>` comment inside
    the part def, written by ``materialize_passive_components`` from the
    generation plan. It lives in the model text so that every downstream
    reader -- the reachability simulator in particular -- sees the same
    decision the planner recorded, without needing the plan object.
    """
    return {
        m.group("name") for m in PASSIVE_MARKER_RE.finditer(sysml_text or "")
    }


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
