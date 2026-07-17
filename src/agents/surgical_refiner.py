"""Surgical (block-level) refinement — replaces the whole-model rewrite.

The legacy refinement asked the LLM to re-emit the ENTIRE model each
iteration; its known failure mode is dropping ``connect`` statements on
components it wasn't even asked to touch (reachability collapse), caught only
after the fact by the connectivity floor — wasting the whole LLM call.

This module inverts the contract: the LLM returns ONLY the top-level blocks it
changes (complete ``part def`` / ``requirement def`` / … elements) plus bare
package-level statements to add; the code merges them into the original text
by exact block replacement.  Everything the LLM does not mention is untouched
by construction, and the output is an order of magnitude smaller than a
full-model rewrite.

Gates (all local, no LLM): the merged model must pass ``check_syntax`` and
must not shed ``connect`` statements.  On any failure the caller falls back to
the legacy whole-model rewrite, so this path can only improve on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import find_block_end

# Top-level definition keywords we can identify for replace-by-name merging.
_DEF_KEYWORDS = (
    "part", "requirement", "port", "item", "action", "enum",
    "state", "calc", "attribute", "analysis", "verification", "constraint",
)

_DEF_HEADER_RE = re.compile(
    r"^[ \t]*(?:abstract\s+)?(?:variation\s+)?"
    rf"({'|'.join(_DEF_KEYWORDS)})\s+def\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)

_FENCE_RE = re.compile(r"```(?:sysml)?\s*\n(.*?)```", re.DOTALL)


SURGICAL_SYSTEM_PROMPT = """You are a SysML v2 surgical editor. You fix the reported issues by rewriting ONLY the affected top-level blocks of the model.

OUTPUT RULES (strict):
  - Return one or more ```sysml fenced blocks and NOTHING else (no prose).
  - Each fenced block contains only COMPLETE top-level elements:
      * a full `part def <Name> { ... }` (or requirement/port/item/action/enum def)
        with the SAME name as an existing element — it REPLACES that element;
      * a NEW definition to add to the package;
      * bare package-level statements to add (e.g. `connect a.x to b.y;`).
  - Do NOT return the whole model. Do NOT return blocks you did not change.
  - Never rename existing elements. Keep every existing port, attribute, and
    connect inside a block you rewrite unless an issue explicitly requires
    changing it.

KEY CONSTRUCT RULES (same as generation):
  satisfy requirement <REQ_ID>;   — INSIDE a part def body; REQ_ID uses underscores
  connect <partA>.<portA> to <partB>.<portB>;   — SysML v2 dot notation
  // State machines: `action def emergencyStop { }` at the part-def top level
  // (NOT inline inside an entry); transitions use canonical keywords:
  state def <Name> {
      state nominal;
      state fault { entry action stop : emergencyStop; }
      transition initial then nominal;
      transition <name>Fault first nominal if <condition> then fault;
  }
  // Initialization/default-state requirements are NOT fault monitors:
  // - declare the required initial state and a consistent Boolean attribute;
  // - a single-state invariant may contain only `transition initial`;
  // - if multiple states are declared, every state MUST be reachable through
  //   real guarded/accept transitions (never add an empty Locked/Unlocked shell);
  // - connect state entry actions to the required actuator/default response.
  // Functional response requirements are NOT satisfied by an action declaration alone:
  // - add a reachable response state whose entry action invokes the required action;
  // - model the real incoming event/condition on the transition (for example a valid
  //   waypoint-modification command or automated-landing-completed event);
  // - preserve explicit qualifiers such as Valid and LandingCompleted in event names;
  // - for `within N seconds`, add max/current latency attributes and an assert constraint
  //   linking the runtime latency to that bound.
  Numeric guards never use `==`; enum guards use `if mode == Type::VALUE`."""


@dataclass
class SurgicalOutcome:
    """A validated block-level merge."""
    merged_text: str
    replaced: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    statements_added: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.replaced:
            parts.append(f"replaced {len(self.replaced)} block(s) "
                         f"({', '.join(self.replaced[:4])})")
        if self.added:
            parts.append(f"added {len(self.added)} block(s)")
        if self.statements_added:
            parts.append(f"added {self.statements_added} statement(s)")
        return "; ".join(parts) or "no-op"


def build_surgical_prompt(
    model_text: str,
    issues: List[str],
    feedback: str = "",
) -> str:
    """User prompt: the full model (read-only context) + the issues to fix."""
    numbered = "\n".join(f"{i}. {iss}" for i, iss in enumerate(issues, 1))
    hint = ", ".join(_affected_names(model_text, issues)) or "(infer from the issues)"
    sections = [
        "CURRENT MODEL (read-only context — return only the blocks you change):",
        f"```sysml\n{model_text}\n```",
        f"ISSUES TO FIX:\n{numbered}",
    ]
    if feedback.strip():
        sections.append(f"ADDITIONAL GUIDANCE:\n{feedback.strip()}")
    sections.append(f"Likely affected elements: {hint}")
    sections.append(
        "Return ONLY the changed/new blocks in ```sysml fences, per the output rules."
    )
    return "\n\n".join(sections)


def _affected_names(model_text: str, issues: List[str]) -> List[str]:
    """Definition names mentioned by the issues (a hint, not a restriction)."""
    names = [m.group(2) for m in _DEF_HEADER_RE.finditer(model_text)]
    blob = " ".join(issues)
    return [n for n in dict.fromkeys(names) if re.search(rf"\b{re.escape(n)}\b", blob)]


# ---------------------------------------------------------------------------
# LLM-output parsing
# ---------------------------------------------------------------------------

def extract_sysml_blocks(raw: str) -> List[str]:
    """Top-level SysML elements from the LLM output (fenced or bare)."""
    raw = raw or ""
    chunks = _FENCE_RE.findall(raw)
    if not chunks:
        # tolerate fence-less output that starts straight with SysML
        stripped = raw.strip()
        if _DEF_HEADER_RE.match(stripped) or stripped.startswith(("connect", "package")):
            chunks = [stripped]
    elements: List[str] = []
    for chunk in chunks:
        elements.extend(_split_top_level(chunk))
    # defensive: unwrap a whole-package wrapper the LLM shouldn't have produced
    unwrapped: List[str] = []
    for el in elements:
        if re.match(r"^\s*package\s+\w+", el):
            brace = el.find("{")
            end = find_block_end(el, brace) if brace != -1 else -1
            if brace != -1 and end != -1:
                unwrapped.extend(_split_top_level(el[brace + 1:end]))
                continue
        unwrapped.append(el)
    return [e for e in unwrapped if e.strip()]


def _split_top_level(text: str) -> List[str]:
    """Split text into top-level elements (brace blocks or `;` statements)."""
    items: List[str] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        if text.startswith("//", i):  # standalone comment line → skip
            nl = text.find("\n", i)
            i = nl + 1 if nl != -1 else n
            continue
        start = i
        depth = 0
        while i < n:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif c == ";" and depth == 0:
                i += 1
                break
            i += 1
        item = text[start:i].strip()
        if item:
            items.append(item)
    return items


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _def_key(element: str) -> Optional[Tuple[str, str]]:
    m = _DEF_HEADER_RE.match(element)
    return (m.group(1), m.group(2)) if m else None


def _find_def_span(base: str, kw: str, name: str) -> Optional[Tuple[int, int]]:
    """(start, end) span of the `<kw> def <name>` element in *base*, or None."""
    pat = re.compile(
        rf"^[ \t]*(?:abstract\s+)?(?:variation\s+)?{kw}\s+def\s+{re.escape(name)}\b",
        re.MULTILINE,
    )
    m = pat.search(base)
    if not m:
        return None
    brace = base.find("{", m.end())
    semi = base.find(";", m.end())
    if brace != -1 and (semi == -1 or brace < semi):
        end = find_block_end(base, brace)
        return (m.start(), end + 1) if end != -1 else None
    if semi != -1:
        return (m.start(), semi + 1)
    return None


def _normalise(stmt: str) -> str:
    return re.sub(r"\s+", "", stmt)


def merge_blocks(base: str, elements: List[str]) -> Optional[SurgicalOutcome]:
    """Merge LLM elements into *base* by exact block replacement / append.

    Returns None when nothing merges (caller falls back).  No validation here —
    gates run in :func:`attempt_surgical_refinement`.
    """
    out = SurgicalOutcome(merged_text=base)
    text = base
    new_blocks: List[str] = []
    new_statements: List[str] = []

    for el in elements:
        key = _def_key(el)
        if key is not None:
            span = _find_def_span(text, *key)
            if span is not None:
                start, end = span
                text = text[:start] + el + text[end:]
                out.replaced.append(key[1])
            else:
                new_blocks.append(el)
                out.added.append(key[1])
        elif el.endswith(";"):
            if _normalise(el) not in {_normalise(s) for s in re.findall(r"[^\n;{}]+;", text)}:
                new_statements.append(el)
        else:
            out.notes.append(f"unrecognised element skipped: {el[:60]!r}")

    if new_blocks or new_statements:
        closing = text.rfind("}")
        if closing == -1:
            return None
        addition = "".join(
            "\n    " + b.replace("\n", "\n    ") for b in new_blocks
        ) + "".join("\n    " + s for s in new_statements)
        text = text[:closing] + addition + "\n" + text[closing:]
        out.statements_added = len(new_statements)

    if text == base:
        return None  # nothing changed — not a usable refinement
    out.merged_text = text
    return out


# ---------------------------------------------------------------------------
# Gated attempt (the orchestrator entry point)
# ---------------------------------------------------------------------------

def _count_connects(text: str) -> int:
    return len(re.findall(r"\bconnect\b", text, re.IGNORECASE))


_REQ_DEF_RE = re.compile(r"\brequirement\s+def\s+([A-Za-z_]\w*)")


def _requirement_defs(text: str) -> set:
    return set(_REQ_DEF_RE.findall(text))


def _count_satisfies(text: str) -> int:
    return len(re.findall(r"\bsatisfy\b", text, re.IGNORECASE))


def _gates_ok(base: str, merged: str) -> Tuple[bool, str]:
    if check_syntax(merged).has_errors:
        return False, "merged model fails syntax check"
    if _count_connects(merged) < _count_connects(base):
        return False, "merge would shed connect statements"
    # Semantic-surgery gates: refinement fixes the DESIGN, never the SPEC.
    # A surgical answer must not add or drop requirement definitions (that would
    # rewrite the problem statement), and must not shed satisfy links (the same
    # silent-loss failure mode the connect gate exists for).
    if _requirement_defs(merged) != _requirement_defs(base):
        return False, "merge would change the requirement def set (refinement must not rewrite the spec)"
    if _count_satisfies(merged) < _count_satisfies(base):
        return False, "merge would shed satisfy links"
    return True, ""


def attempt_surgical_refinement(
    llm,
    model_text: str,
    issues: List[str],
    feedback: str = "",
    verbose: bool = False,
) -> Optional[SurgicalOutcome]:
    """One surgical refinement attempt; None means "fall back to full rewrite".

    Uses temperature escalation when the provider supports it: a low-temperature
    answer that fails to parse/merge/validate is retried warmer before giving up.
    """
    if not model_text.strip() or not issues:
        return None
    prompt = build_surgical_prompt(model_text, issues, feedback)
    cache: Dict[str, Optional[SurgicalOutcome]] = {}

    def _try(content: str) -> Optional[SurgicalOutcome]:
        if content in cache:
            return cache[content]
        outcome: Optional[SurgicalOutcome] = None
        elements = extract_sysml_blocks(content)
        if elements:
            merged = merge_blocks(model_text, elements)
            if merged is not None:
                ok, why = _gates_ok(model_text, merged.merged_text)
                if ok:
                    outcome = merged
                elif verbose:
                    print(f"  [surgical] rejected: {why}")
        cache[content] = outcome
        return outcome

    escalate = getattr(llm, "chat_with_escalation", None)
    if callable(escalate):
        content, ok = escalate(
            prompt,
            system_prompt=SURGICAL_SYSTEM_PROMPT,
            validate=lambda c: _try(c) is not None,
            temperatures=(0.2, 0.6),
        )
        return _try(content) if ok else None
    return _try(str(llm.chat(prompt, system_prompt=SURGICAL_SYSTEM_PROMPT)))
