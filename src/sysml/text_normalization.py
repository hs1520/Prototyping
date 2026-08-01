"""Deterministic normalization seam for provider-authored SysML text.

These rules repair text whose upstream authority is an LLM or a repair round.
They are deliberately not part of the structured SysML writer.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence, Tuple

from ..utils.sysml_text_utils import find_block_end


# These tuples record the historical order at each call-site boundary.  Callers
# retain their gates and validation between steps; this seam owns the rules,
# not their scheduling.
NORMALIZATION_RULE_ORDER = {
    "syntax_gate": (
        "strip_readonly_keyword",
        "fix_keyword_item_names",
    ),
    "design_semantics": (
        "fix_capability_semantics",
        "fix_safety_action_semantics",
    ),
    "design_post_assembly": (
        "fix_doc_syntax",
        "strip_invalid_requirement_attrs",
        "normalise_connect_syntax",
    ),
    "surgical_repair": ("strip_code_fences",),
    "ag_authored_planning": ("strip_ag_implementation",),
    "ag_terminal_binding": ("strip_named_item_definitions",),
}


def fix_keyword_item_names(sysml_text: str) -> str:
    """
    Quote SysML reserved words used as item-usage names inside port/part defs.

    Pattern:  (in|out|inout) item <keyword> :
    Fix:      (in|out|inout) item '<keyword>' :

    syside rejects bare reserved words like `message`, `flow`, `connect`,
    `accept`, `send`, `loop`, `state` as item-usage names (parser error
    "Unexpected 'item'").  Quoting them makes the syntax valid.
    """
    _SYSML_KW = re.compile(
        r'\b(in|out|inout)\s+item\s+'
        r'(message|flow|connect|accept|send|loop|state|item|if|then|first|else)\s*:',
        re.IGNORECASE,
    )
    return _SYSML_KW.sub(lambda m: f"{m.group(1)} item '{m.group(2)}' :", sysml_text)


_READONLY_ATTR_RE = re.compile(r'\breadonly\s+(attribute\b)')


def strip_readonly_keyword(sysml_text: str) -> str:
    """
    Drop the `readonly` modifier before `attribute`.

    syside's parser rejects `readonly attribute X : ...` ("Unexpected
    identifier"), and the official SysML v2 corpus never uses `readonly`
    — constants are plain `attribute name : T = value [unit];`.  The
    design-limit vs runtime-state distinction is carried by naming
    convention (max/min/limit vs current*) and assert constraints, not by
    this keyword.  The generation prompt no longer teaches `readonly`;
    this is a deterministic safety net for residual LLM emissions so they
    cost no syntax-gate LLM round.
    """
    return _READONLY_ATTR_RE.sub(r'\1', sysml_text)


def strip_code_fences(text: str) -> str:
    """
    去除 LLM 返回内容中常见的 markdown 代码围栏。

    处理以下格式：
      ```sysml ... ```
      ```         ... ```
      ~~~sysml    ... ~~~
    """
    # 去掉开头的围栏行（```sysml、```、~~~sysml 等）
    text = re.sub(r'^[ \t]*(?:```|~~~)\w*[ \t]*\n', '', text, flags=re.MULTILINE)
    # 去掉结尾的围栏行
    text = re.sub(r'^[ \t]*(?:```|~~~)[ \t]*$', '', text, flags=re.MULTILINE)
    return text.strip('\n')


# ──────────────────────────────────────────────────────────────────────
# doc = "string" → doc /* string */ syntax normaliser
# ──────────────────────────────────────────────────────────────────────

def normalise_connect_syntax(sysml_text: str) -> Tuple[str, int]:
    """Convert `connect a::b to c::d;` → `connect a.b to c.d;`.

    SysML v2 uses dot notation for connect endpoints (verified against the
    official SysML-v2-release-src/examples corpus).  The `::` operator is
    for namespace-qualified names (`Package::Element`), not feature access
    in connect statements.  LLMs sometimes emit the `::` form anyway;
    normalising here ensures every downstream consumer sees the canonical
    SysML v2 syntax.

    Only `::` occurrences inside `connect ... to ...;` are touched — any
    other use (e.g. `Package::Element` qualified names) is preserved.

    Returns (normalised_text, count_of_substitutions).
    """
    # Match a complete connect statement that uses `::` on either side.
    # The capture groups isolate part / port pieces so we can rewrite with `.`.
    connect_pat = re.compile(
        r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)::(\w+)\s*;",
        re.IGNORECASE,
    )

    # Also handle the asymmetric forms (one side `::`, the other `.`).
    connect_mixed_left = re.compile(
        r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)\.(\w+)\s*;",
        re.IGNORECASE,
    )
    connect_mixed_right = re.compile(
        r"\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)::(\w+)\s*;",
        re.IGNORECASE,
    )

    count = 0

    def _rewrite(m: re.Match) -> str:  # type: ignore[type-arg]
        return f"connect {m.group(1)}.{m.group(2)} to {m.group(3)}.{m.group(4)};"

    for pat in (connect_pat, connect_mixed_left, connect_mixed_right):
        n = len(pat.findall(sysml_text))
        if n:
            sysml_text = pat.sub(_rewrite, sysml_text)
            count += n

    return sysml_text, count


def fix_doc_syntax(sysml_text: str) -> Tuple[str, int]:
    """Convert invalid ``doc = "string";`` to valid ``doc /* string */``.

    The correct SysML v2 doc-comment syntax is ``doc /* text */``.
    LLMs sometimes generate ``doc = "text";`` (or ``doc = "text"``) which
    is not valid SysML v2 — it is parsed by Syside as a feature-usage
    named ``doc`` of type String, causing round-trip serialization issues
    (the feature leaks into ``top_level_usages`` as a spurious
    ``requirement req : String = "..."`` line).

    Returns:
        (fixed_text, count_of_substitutions)
    """
    # Match: optional leading whitespace, `doc`, optional whitespace,
    # `=`, optional whitespace, a double-quoted string, optional `;`
    doc_eq_re = re.compile(
        r'\bdoc\s*=\s*"((?:[^"\\]|\\.)*)"[ \t]*;?',
    )
    count = len(doc_eq_re.findall(sysml_text))
    fixed = doc_eq_re.sub(lambda m: f'doc /* {m.group(1)} */', sysml_text)
    return fixed, count


# ──────────────────────────────────────────────────────────────────────
# Invalid requirement-attribute cleanup (post-assembly sanitiser)
# ──────────────────────────────────────────────────────────────────────

def strip_invalid_requirement_attrs(sysml_text: str) -> Tuple[str, int]:
    """Remove invalid ``requirement <name> : <Type> = "...";`` lines.

    These are not valid SysML v2.  The refinement LLM sometimes produces
    them when it tries to "document" a requirement inline — but the correct
    construct is ``satisfy requirement REQ_ID;`` inside a part def, or
    ``requirement def REQ_ID { doc /* ... */ }`` at package level.

    Only lines that match the specific bogus pattern are removed; all valid
    ``requirement def`` and ``satisfy requirement`` constructs are preserved.

    Returns:
        (cleaned_text, count_of_removed_lines)
    """
    # Matches: optional leading whitespace, `requirement`, identifier,
    # colon, type name, equals, a double-quoted string, semicolon.
    # Does NOT match `requirement def` (the `def` keyword breaks the pattern).
    invalid_re = re.compile(
        r"^[ \t]*requirement[ \t]+(?!def\b)\w+[ \t]*:[ \t]*\w+[ \t]*"
        r"=[ \t]*\"[^\"]*\"[ \t]*;[ \t]*$",
        re.MULTILINE,
    )
    matches = invalid_re.findall(sysml_text)
    cleaned = invalid_re.sub("", sysml_text)
    return cleaned, len(matches)


def fix_capability_semantics(
    sysml_text: str,
    *,
    has_range_floor: bool,
    has_range_ceiling: bool,
) -> Tuple[str, int]:
    """Repair range-floor naming and remove invalid always-on invariants.

    Operational range is a mission-end capability.  When the structured
    requirement is a lower bound, ``currentRange <= maxRange`` verifies the
    opposite property, while ``currentRange >= minRange`` is false at
    startup.  Keep the design target as ``min*Range`` and leave evaluation
    to the forward-flight fidelity tier.
    """
    if not has_range_floor or has_range_ceiling:
        return sysml_text, 0

    fixes = 0

    def _rename(match: re.Match) -> str:
        nonlocal fixes
        fixes += 1
        token = match.group(0)
        return ("min" if token.startswith("max") else "Min") + token[3:]

    result = re.sub(
        r"\b(?:max|Max)(?:Operational)?Range\b",
        _rename,
        sysml_text,
    )

    constraint_re = re.compile(
        r"(?ms)^(?P<indent>[ \t]*)assert\s+constraint\s+\w+\s*\{"
        r"(?P<body>[^{}]*(?:current\w*Range|distance\w*)[^{}]*)\}\s*"
    )

    def _drop_constraint(match: re.Match) -> str:
        nonlocal fixes
        body = match.group("body")
        is_mission_range = re.search(
            r"\b(?:current(?:Operational)?Range|distanceTravelled)\b",
            body,
            re.IGNORECASE,
        ) and re.search(
            r"\b(?:min|max)(?:Operational)?Range\b",
            body,
            re.IGNORECASE,
        )
        if not is_mission_range:
            return match.group(0)
        fixes += 1
        return (
            f"{match.group('indent')}// Operational range is a mission-end "
            "capability evaluated by forward-flight fidelity, not an invariant.\n"
        )

    return constraint_re.sub(_drop_constraint, result), fixes


def fix_safety_action_semantics(sysml_text: str) -> Tuple[str, int]:
    """Prevent a parachute action from sending a flight-mode LAND command."""
    action_re = re.compile(
        r"(?P<head>action\s+def\s+\w*(?:parachute|chute)\w*\s*\{)"
        r"(?P<body>[^{}]*)(?P<tail>\})",
        re.IGNORECASE,
    )
    fixes = 0

    def _fix_action(match: re.Match) -> str:
        nonlocal fixes
        body, n = re.subn(
            r"\bsend\s+CMD_(?:LAND|RTL|AUTO|GUIDED|LOITER|POSHOLD)\s*\(\)",
            "send CMD_PARACHUTE()",
            match.group("body"),
            flags=re.IGNORECASE,
        )
        fixes += n
        return match.group("head") + body + match.group("tail")

    result = action_re.sub(_fix_action, sysml_text)
    if fixes and not re.search(r"\baction\s+def\s+CMD_PARACHUTE\b", result):
        first_part = re.search(r"(?m)^[ \t]*part\s+def\s+", result)
        if first_part:
            result = (
                result[:first_part.start()]
                + "    action def CMD_PARACHUTE { }\n\n"
                + result[first_part.start():]
            )
            fixes += 1
    return result, fixes


def _remove_named_block(text: str, keyword: str, name: str) -> str:
    pattern = re.compile(
        rf"\b{re.escape(keyword)}\s+{re.escape(name)}\s*\{{"
    )
    result = str(text)
    while True:
        match = pattern.search(result)
        if match is None:
            return result
        brace = result.find("{", match.start())
        end = find_block_end(result, brace)
        if end == -1:
            return result
        result = result[:match.start()] + result[end + 1:]


def strip_ag_implementation(
    package_text: str,
    behavior_names: Iterable[str],
    components: Iterable[Tuple[str, str, str]],
) -> str:
    """Remove implementation claims from an authored A/G package."""
    text = str(package_text)

    # A planning artifact may describe the behavior obligation by name in the
    # typed AGChainSpec, but no concrete state definition exists until the main
    # model has been generated.
    for behavior in dict.fromkeys(behavior_names):
        text = _remove_named_block(text, "state def", behavior)

    for owner_def, owner_usage, contract_name in components:
        text = re.sub(
            rf"(?m)^\s*part\s+def\s+"
            rf"{re.escape(owner_def)}\s*;\s*\n?",
            "",
            text,
        )
        text = re.sub(
            rf"(?m)^\s*part\s+{re.escape(owner_usage)}\s*:\s*"
            rf"{re.escape(owner_def)}\s*;\s*\n?",
            "",
            text,
        )
        text = re.sub(
            rf"(?m)^\s*satisfy\s+requirement\s+\w+\s*:\s*"
            rf"{re.escape(contract_name)}\s+by\s+"
            rf"{re.escape(owner_usage)}\s*;\s*\n?",
            "",
            text,
        )

    # Realization edges would assert that the removed behavior exists.  The
    # terminal binder recreates these edges against qualified main-model paths.
    text = re.sub(
        r"(?m)^\s*dependency\s+realize\w+\s+from\s+\w+\s+to\s+\w+\s*;\s*\n?",
        "",
        text,
    )
    return text


def strip_named_item_definitions(
    package_text: str,
    event_names: Sequence[str],
) -> str:
    """Remove standalone event types before terminal canonical imports.

    A terminal A/G package imports the exact system event classifiers and must
    not retain package-local classifiers with merely equal simple names.
    """
    text = str(package_text)
    for event_name in event_names:
        text = re.sub(
            rf"(?m)^\s*item\s+def\s+"
            rf"{re.escape(event_name)}\s*;\s*\n?",
            "",
            text,
        )
    return text
