"""Behavioural verification of SAFETY requirements — upgrades safety `satisfy` from mere
allocation to real (three-state) behavioural evidence, reusing the state-machine extractor.

For each SAFE requirement we find the part(s) that satisfy it, take their state machines, and
check whether a FAIL-SAFE state is actually REACHABLE from the initial state (the safety
response can really be entered). Outcomes:
  behaviorally-verified : owner part has a state machine and a fail-safe state is reachable.
  behaviorally-violated : a state machine exists but NO fail-safe state is reachable (the
                          safety behaviour is named/allocated but the SM can't get there —
                          "fake safety", a real finding a keyword/structural scorer misses).
  behavior-absent       : the satisfying part has no state machine → nothing to verify
                          (the behaviour was never generated → flags a generation gap).

Scope (honest): this verifies fail-safe *reachability* of the safety state machine. Full
per-requirement scenario semantics + timing are richer (behavioral_sim / grounded_eval for
the redundancy-failsafe subclass) and remain roadmap items.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from ..simulation.state_extractor import extract_state_machines
from ..utils.sysml_text_utils import find_block_end

BEHAVIORALLY_VERIFIED = "behaviorally-verified"
BEHAVIORALLY_VIOLATED = "behaviorally-violated"
BEHAVIOR_ABSENT = "behavior-absent"

_SAFE_STATE_KW = ("failsafe", "fail_safe", "safe", "abort", "lock", "disarm", "rtb",
                  "return", "land", "hold", "emergency", "parachute", "contingency")
_SAFETY_REQ_KW = ("safe", "fail", "abort", "lock", "disarm", "emergency", "parachute",
                  "prohibit", "inhibit", "contingency", "geofence")
_REQ_ID_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")
_SATISFY_RE = re.compile(r"satisfy\s+(?:requirement\s+)?(\w*REQ[_-]\w+)", re.IGNORECASE)


def _norm(rid: str) -> str:
    return rid.upper().replace("_", "-")


def is_safety_req(rid: str, text: str) -> bool:
    return "SAFE" in rid.upper() or any(k in text.lower() for k in _SAFETY_REQ_KW)


def _req_owner_parts(model_text: str) -> Dict[str, Set[str]]:
    """{req_id: {part def names whose body declares `satisfy <req>`}}."""
    out: Dict[str, Set[str]] = {}
    for m in re.finditer(r"\bpart\s+def\s+(\w+)\s*(?::>[^{]*)?\{", model_text):
        brace = model_text.index("{", m.start())
        end = find_block_end(model_text, brace)
        body = model_text[brace + 1:end] if end != -1 else ""
        for sm in _SATISFY_RE.finditer(body):
            out.setdefault(_norm(sm.group(1)), set()).add(m.group(1))
    return out


def reachable_states(sm) -> Set[str]:
    """State names reachable from the initial state via transitions. If the initial state
    can't be determined (extractor limitation), treat all states as reachable (lenient — so
    we never falsely claim a behaviour is unreachable)."""
    starts: Set[str] = set()
    if sm.initial_state:
        starts.add(sm.initial_state)
    starts |= {t.target for t in sm.transitions if t.is_initial and t.target}
    if not starts:
        return {s.name for s in sm.states}
    adj: Dict[str, List[str]] = {}
    for t in sm.transitions:
        if t.source and t.target:
            adj.setdefault(t.source, []).append(t.target)
    seen: Set[str] = set()
    stack = list(starts)
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        stack.extend(adj.get(s, []))
    return seen


def _failsafe_reachable(sm) -> str:
    """'reachable' | 'unreachable' | 'no_safe_state' for one state machine."""
    safe = {s.name for s in sm.states if any(k in s.name.lower() for k in _SAFE_STATE_KW)}
    if not safe:
        return "no_safe_state"
    return "reachable" if (reachable_states(sm) & safe) else "unreachable"


def safety_behavior_status(model_text: str, requirements: List[str],
                           dynamic_fire: Dict[str, str] = None) -> Dict[str, str]:
    """{safety req_id: behavioural status} for every SAFE requirement satisfied in the model.

    If ``dynamic_fire`` ({owner_part: 'fired'|'failed'}, from dynamic_behavior) is given, the
    DYNAMIC verdict takes precedence over structural reachability: a part whose guarded
    transition actually fires → verified; one whose scenario fails (guard present but never
    fires) → violated (caught dynamically, not by structural reachability)."""
    dynamic_fire = dynamic_fire or {}
    by_part: Dict[str, list] = {}
    for sm in extract_state_machines(model_text):
        by_part.setdefault(sm.owner_part, []).append(sm)
    text = {m.group(0).replace("_", "-"): r for r in requirements
            for m in [_REQ_ID_RE.search(r)] if m}

    out: Dict[str, str] = {}
    for rid, parts in _req_owner_parts(model_text).items():
        if not is_safety_req(rid, text.get(rid, rid)):
            continue
        sms = [sm for p in parts for sm in by_part.get(p, [])]
        verdicts = {dynamic_fire[p] for p in parts if p in dynamic_fire}
        # structural: is a FAIL-SAFE state reachable? (dynamic firing of a non-safe transition
        # must NOT count as a safety response — so the safe-state requirement gates everything.)
        structural_ok = any(_failsafe_reachable(sm) == "reachable" for sm in sms)
        if not sms:
            out[rid] = BEHAVIOR_ABSENT
        elif not structural_ok:
            out[rid] = BEHAVIORALLY_VIOLATED        # no reachable fail-safe state
        elif "failed" in verdicts:
            out[rid] = BEHAVIORALLY_VIOLATED        # safe state reachable but dynamically NEVER
            #                                         fires → "fake safety" (dynamic-only catch)
        else:
            out[rid] = BEHAVIORALLY_VERIFIED        # fail-safe reachable (+ fires if dynamic ran)
    return out
