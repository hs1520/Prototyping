"""Deterministic normalization seam for provider-authored SysML text.

These rules repair text authored by an LLM or a repair round; they are not
part of the structured SysML writer.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence, Tuple

from ..utils.sysml_text_utils import find_block_end


# These tuples record the rule order at each call-site boundary. Callers keep
# their gates between steps and do the scheduling; this seam owns the rules.
NORMALIZATION_RULE_ORDER = {
    "syntax_gate": (
        "fix_doc_syntax",
        "strip_readonly_keyword",
        "fix_keyword_item_names",
        "fix_c_style_negation",
    ),
    # 2026-08-31: `fix_safety_action_semantics` removed from this tier - it
    # rewrote wrong safety commands toward the harness vocabulary. The defect
    # now surfaces at the linker's traceability check.
    "design_semantics": (
        "fix_capability_semantics",
    ),
    "design_post_assembly": (
        "fix_doc_syntax",
        "strip_invalid_requirement_attrs",
        "normalise_connect_syntax",
    ),
    "surgical_repair": ("strip_code_fences",),
    "ag_authored_planning": ("strip_ag_implementation",),
    "ag_terminal_binding": ("strip_named_item_definitions",),
    "terminal_commit": ("strip_redundant_inherited_ports",),
}


def fix_keyword_item_names(sysml_text: str) -> str:
    """Quote SysML reserved words used as item-usage names inside port/part defs.

    Pattern:  (in|out|inout) item <keyword> :
    Fix:      (in|out|inout) item '<keyword>' :

    syside rejects bare reserved words such as `message`, `flow`, `connect`,
    `accept`, `send`, `loop`, `state` here (parser error "Unexpected 'item'").
    """
    _SYSML_KW = re.compile(
        r'\b(in|out|inout)\s+item\s+'
        r'(message|flow|connect|accept|send|loop|state|item|if|then|first|else)\s*:',
        re.IGNORECASE,
    )
    return _SYSML_KW.sub(lambda m: f"{m.group(1)} item '{m.group(2)}' :", sysml_text)


_READONLY_ATTR_RE = re.compile(r'\breadonly\s+(attribute\b)')


def strip_readonly_keyword(sysml_text: str) -> str:
    """Drop the `readonly` modifier before `attribute`.

    syside rejects `readonly attribute X : ...` ("Unexpected identifier") and the
    SysML v2 corpus writes constants as `attribute name : T = value [unit];`. The
    design-limit vs runtime-state distinction is carried by naming (max/min/limit
    vs current*) and assert constraints. The generation prompt no longer teaches
    `readonly`; this catches residual LLM emissions without a syntax-gate round.
    """
    return _READONLY_ATTR_RE.sub(r'\1', sysml_text)


def strip_code_fences(text: str) -> str:
    """去除 LLM 返回内容中常见的 markdown 代码围栏。"""
    text = re.sub(r'^[ \t]*(?:```|~~~)\w*[ \t]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]*(?:```|~~~)[ \t]*$', '', text, flags=re.MULTILINE)
    return text.strip('\n')


def normalise_connect_syntax(sysml_text: str) -> Tuple[str, int]:
    """Convert `connect a::b to c::d;` -> `connect a.b to c.d;`.

    SysML v2 uses dot notation for connect endpoints (checked against the
    official SysML-v2-release-src/examples corpus); `::` is for
    namespace-qualified names, which LLMs sometimes emit here anyway. Only `::`
    inside `connect ... to ...;` is touched, so `Package::Element` is preserved.

    Returns (normalised_text, count_of_substitutions).
    """
    connect_pat = re.compile(
        r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)::(\w+)\s*;",
        re.IGNORECASE,
    )

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
    """Convert quoted ``doc`` bodies to the legal ``doc /* text */`` form.

    SysML v2 doc bodies are comments, not strings. LLMs emit two invalid
    spellings: ``doc = "text";``, which Syside parses as a String feature-usage
    named ``doc`` and leaks into ``top_level_usages``; and ``doc 'text';``
    without ``=``, a parser error the three-attempt syntax gate could not clear
    on the 2026-08-30 attempt. Both rewrites are mechanical, so they cost no
    syntax-gate LLM round. Single-line bodies only; ``*/`` inside a body is
    defused.

    Returns:
        (fixed_text, count_of_substitutions)
    """
    doc_quoted_re = re.compile(
        r"""\bdoc\s*(?:=\s*)?(?P<quote>['"])"""
        r"""(?P<body>(?:\\.|(?!(?P=quote))[^\n\\])*)"""
        r"""(?P=quote)[ \t]*;?""",
    )

    def _swap(match: re.Match) -> str:
        body = match.group("body").replace("*/", "* /").strip()
        return f"doc /* {body} */"

    count = len(doc_quoted_re.findall(sysml_text))
    fixed = doc_quoted_re.sub(_swap, sysml_text)
    return fixed, count


def strip_invalid_requirement_attrs(sysml_text: str) -> Tuple[str, int]:
    """Remove invalid ``requirement <name> : <Type> = "...";`` lines.

    The refinement LLM emits these when documenting a requirement inline; the
    valid constructs are ``satisfy requirement REQ_ID;`` inside a part def and
    ``requirement def REQ_ID { doc /* ... */ }`` at package level. Only the
    bogus pattern is removed.

    Returns:
        (cleaned_text, count_of_removed_lines)
    """
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

    Operational range is a mission-end capability. For a lower-bound requirement,
    ``currentRange <= maxRange`` checks the opposite property and
    ``currentRange >= minRange`` is false at startup, so keep the design target as
    ``min*Range`` and leave evaluation to the forward-flight fidelity tier.
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
        r"(?ms)^(?P<indent>[ \t]*)assert\s+constraint\s+(?P<name>\w+)\s*\{"
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
        # The marker keeps the delegation auditable: semantic fidelity records
        # the obligation as DELEGATED to the named tier rather than failing it
        # for the assert removed here.
        return (
            f"{match.group('indent')}// DELEGATED-CONSTRAINT "
            f"{match.group('name')} tier=FORWARD_FLIGHT_FIDELITY "
            "reason=mission-end-capability\n"
            f"{match.group('indent')}// Operational range is a mission-end "
            "capability evaluated by forward-flight fidelity, not an invariant.\n"
        )

    return constraint_re.sub(_drop_constraint, result), fixes


# `fix_safety_action_semantics` was removed 2026-08-31. It rewrote
# `send CMD_LAND()` inside parachute-named actions into `send CMD_PARACHUTE()`
# and injected the harness's command definition, turning an arbitration defect
# into a pass. The defect now surfaces at the requirement linker's traceability
# check, which blocks a parachute guard that sends a LAND command
# (test_parachute_land_command_blocked); a
# correctly-commanded model with its own spelling is accepted via the plan's
# causal-path route rather than a CHUTE substring.


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

    # The typed AGChainSpec may name the behavior obligation, but no state
    # definition exists until the main model is generated.
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

    A terminal A/G package imports the system event classifiers, so package-local
    classifiers sharing a simple name are dropped.
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


# Boolean negation written the C way. `!=` is a legal SysML v2 inequality the
# A/G emitter produces, so the lookahead excluding `=` is required. Negation is
# rewritten only where an identifier follows - the only position `not` is valid.
_C_NEGATION_RE = re.compile(r"(?<![!=<>])!(?!=)\s*(?=[A-Za-z_])")


def fix_c_style_negation(sysml_text: str) -> str:
    """Rewrite `!flag` as `not flag`.

    SysML v2 spells boolean negation `not`; syside rejects `!` with "Unexpected
    token '!'". In pilot_n6_20260802/seed-3/R0-CURRENT a single
    `if !sensorFailure` at line 125 was the model's only parser error and failed
    the run's qualification gate - one occurrence across eight archived pilots,
    on provider output the deterministic A/G emitter already spells correctly.
    """
    return _C_NEGATION_RE.sub("not ", sysml_text)


_PART_DEF_HEADER_RE = re.compile(
    r"\bpart\s+def\s+(?P<name>\w+)\s*"
    r"(?::>\s*(?P<parents>\w+(?:\s*,\s*\w+)*))?\s*\{"
)
_PORT_DECLARATION_RE = re.compile(
    r"\b(?P<direction>in|out|inout)\s+port\s+(?P<name>\w+)\s*"
    r"(?::\s*(?P<type>[\w:]+))?\s*;"
)


def strip_redundant_inherited_ports(sysml_text: str) -> Tuple[str, int]:
    """Remove port redeclarations identical to an inherited declaration.

    A part def specialising a host (``X :> Host``) inherits its ports;
    redeclaring one with the same name, direction and type is redundant and trips
    syside's namespace-distinguishability warning per copy. Run 00e4d333 carried
    ten of them and lost the terminal zero-warning gate. Only exact matches are
    removed; a declaration differing in direction or type is left to the checker.
    """
    text = str(sysml_text or "")

    definitions: list[tuple[str, int, int, tuple[str, ...]]] = []
    for header in _PART_DEF_HEADER_RE.finditer(text):
        opening = text.find("{", header.start(), header.end() + 1)
        closing = find_block_end(text, opening)
        if closing == -1:
            continue
        parents = tuple(
            parent.strip()
            for parent in (header.group("parents") or "").split(",")
            if parent.strip()
        )
        definitions.append(
            (header.group("name"), opening + 1, closing, parents)
        )

    def innermost_owner(position: int) -> int | None:
        owner, owner_size = None, None
        for index, (_name, start, end, _parents) in enumerate(definitions):
            if start <= position < end:
                size = end - start
                if owner_size is None or size < owner_size:
                    owner, owner_size = index, size
        return owner

    ports_by_definition: dict[int, dict[str, tuple[str, str, int, int]]] = {}
    for match in _PORT_DECLARATION_RE.finditer(text):
        owner = innermost_owner(match.start())
        if owner is None:
            continue
        ports_by_definition.setdefault(owner, {})[match.group("name")] = (
            match.group("direction"),
            match.group("type") or "",
            match.start(),
            match.end(),
        )

    by_name = {
        definitions[index][0]: index for index in range(len(definitions))
    }

    def inherited_ports(index: int) -> dict[str, tuple[str, str]]:
        collected: dict[str, tuple[str, str]] = {}
        frontier = list(definitions[index][3])
        seen: set[str] = set()
        while frontier:
            parent = frontier.pop()
            if parent in seen or parent not in by_name:
                continue
            seen.add(parent)
            parent_index = by_name[parent]
            for name, (direction, port_type, _s, _e) in (
                ports_by_definition.get(parent_index, {})
            ).items():
                collected.setdefault(name, (direction, port_type))
            frontier.extend(definitions[parent_index][3])
        return collected

    removals: list[tuple[int, int]] = []
    for index, (_name, _start, _end, parents) in enumerate(definitions):
        if not parents:
            continue
        inherited = inherited_ports(index)
        for name, (direction, port_type, start, end) in (
            ports_by_definition.get(index, {})
        ).items():
            if inherited.get(name) == (direction, port_type):
                removals.append((start, end))

    if not removals:
        return text, 0
    output = text
    for start, end in sorted(removals, reverse=True):
        tail = output[end:]
        output = output[:start].rstrip(" ") + tail.lstrip(" ") \
            if tail[:1] not in ("\n", "") else output[:start].rstrip(" ") + tail
    return output, len(removals)
