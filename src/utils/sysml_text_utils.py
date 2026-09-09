"""Shared text-level utilities for SysML source manipulation."""
from __future__ import annotations

import re

IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
# Header tail between a definition name and its body brace. Tolerates
# specialization (``:> Super``) and other header text, but does not cross a
# ``;`` (a bodiless declaration), another brace, or a newline. The DSE variation
# layer emits ``part def LidarSuite :> SensorSuite { ... }``, so requiring the
# brace right after the name would miss real blocks. The newline bound stops
# prose such as ``// "Every part def MUST have >= 1 satisfy link"`` from minting
# a phantom def named ``MUST`` that swallows the next block's braces (seen in
# two archived pilot models). Emitters write single-line headers.
_DEF_HEADER_TAIL = r"\b[^{;\n]*\{"
PART_DEF_RE = re.compile(r"\bpart\s+def\s+(\w+)" + _DEF_HEADER_TAIL)
STATE_DEF_RE = re.compile(r"\bstate\s+def\s+(\w+)" + _DEF_HEADER_TAIL)


def named_def_pattern(kind: str, name: str) -> re.Pattern[str]:
    """``<kind> def <name> ... {`` for one specific definition name."""
    return re.compile(rf"\b{kind}\s+def\s+{re.escape(name)}{_DEF_HEADER_TAIL}")


def named_block_span(text: str, kind: str, name: str) -> tuple[int, int] | None:
    """``(opening_brace, closing_brace)`` of ``<kind> def <name> { ... }``."""
    match = named_def_pattern(kind, name).search(text)
    if match is None:
        return None
    opening = match.end() - 1
    closing = find_block_end(text, opening)
    if closing == -1:
        return None
    return opening, closing


def find_block_end(text: str, start: int) -> int:
    """Return the index of the closing '}' matching the '{' at *start*."""
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

    In utils so packages below `prototyping` in the dependency order
    (simulation's verification binding, dse) can match identities by meaning
    without importing it. Re-exported from prototyping.verification_obligations.
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
    """Part-definition names the committed model declares passive.

    The declaration is a `// PLAN-PASSIVE <PartDef>: <reason>` comment inside the
    part def, written by ``materialize_passive_components`` from the generation
    plan. Keeping it in the model text lets downstream readers (the reachability
    simulator) see the planner's decision without the plan object.
    """
    return {
        m.group("name") for m in PASSIVE_MARKER_RE.finditer(sysml_text or "")
    }


def get_sysml_text(model) -> str:
    """Return SysML text for *model*, preferring the cached metadata string."""
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


def close_truncated_blocks(text: str) -> tuple[str, int]:
    """Deterministically close blocks an output-budget truncation cut off.

    In run3 the assembly response stopped mid-file with every statement present
    complete and only the enclosing closing braces missing, and the syntax gate
    bought a full LLM window repair that only appended ``}`` lines. Balancing
    applies only to that shape: the text must end at a statement/block boundary
    (``;``, ``{``, ``}``, or a line comment). A tail cut mid-token, mid-string or
    mid-block-comment lost content and is refused untouched, as is text whose
    braces already balance or over-close. Returns (text, braces_appended).
    """
    depth = 0
    in_line_comment = in_block_comment = in_string = False
    last_code_char = ""
    index = 0
    while index < len(text):
        char = text[index]
        pair = text[index:index + 2]
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            if pair == "*/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if pair == "//":
            in_line_comment = True
            index += 2
            continue
        if pair == "/*":
            in_block_comment = True
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return text, 0
        if not char.isspace():
            last_code_char = char
        index += 1

    if depth <= 0 or in_block_comment or in_string:
        return text, 0
    stripped = text.rstrip()
    boundary = last_code_char in {";", "{", "}"} or (
        in_line_comment and stripped.rsplit("\n", 1)[-1].lstrip().startswith("//")
    )
    if not boundary:
        return text, 0
    return stripped + "\n" + "}\n" * depth, depth
