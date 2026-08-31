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
from .requirement_trace import extract_requirement_trace

BEHAVIORALLY_VERIFIED = "behaviorally-verified"
BEHAVIORALLY_VIOLATED = "behaviorally-violated"
BEHAVIOR_ABSENT = "behavior-absent"
RESPONSE_COLLAPSED = "response-collapsed"

# distinct mandated safety RESPONSES → keywords (in requirement text AND in action names).
# If two DIFFERENT response categories emit the SAME command, the arbitration is lost at the
# behaviour interface (e.g. battery-low LAND, comm-loss LAND, propulsion PARACHUTE all collapse
# to one CmdToEmergency → the flight controller can't do the right thing per cause).
_RESPONSE_CATEGORIES = {
    "parachute": ("parachute", "ballistic recovery", "chute"),
    "rtb":       ("return-to-base", "return to base", "return-to-home", "rtb", "return trajectory"),
    "land":      ("controlled descent", "safe landing", "land", "landing", "touchdown", "descend"),
    "lock":      ("lock", "locked", "inhibit release", "mechanically locked"),
    # Degraded-continue: the mandated response is to KEEP FLYING under a
    # fault (one motor inoperative), not to enter a land/rtb/lock state.
    "continue":  ("controlled flight", "controlledflight", "maintain controlled"),
}


def _response_category(text: str):
    """The distinct safety RESPONSE a requirement mandates
    (parachute/rtb/land/lock/continue), or None. Order matters: parachute &
    rtb are checked before land (a parachute/RTB req may also say 'land'),
    and 'continue' last ('controlled descent' is a landing, not a
    continuation)."""
    t = text.lower()
    for cat in ("parachute", "rtb", "land", "lock", "continue"):
        if any(k in t for k in _RESPONSE_CATEGORIES[cat]):
            return cat
    return None


def _name_category(name: str):
    return _response_category(name)


def collapsed_response_categories(model_text: str) -> Set[str]:
    """Response categories that COLLAPSE — i.e. a single emitted command is sent by actions of
    ≥2 distinct response categories (the arbitration distinction is lost at the interface)."""
    cats_by_cmd: Dict[str, Set[str]] = {}
    for m in re.finditer(r"action\s+def\s+(\w+)\s*\{(.*?)\}", model_text, re.DOTALL):
        cat = _name_category(m.group(1))
        if not cat:
            continue
        for cmd in re.findall(r"send\s+(\w+)", m.group(2)):
            cats_by_cmd.setdefault(cmd, set()).add(cat)
    collapsed: Set[str] = set()
    for cmd, cats in cats_by_cmd.items():
        if len(cats) >= 2:                  # one command serves ≥2 distinct responses → collapse
            collapsed |= cats
    return collapsed

_SAFE_STATE_KW = ("failsafe", "fail_safe", "safe", "abort", "lock", "disarm", "rtb",
                  "return", "land", "hold", "emergency", "parachute", "contingency")
_SAFETY_REQ_KW = ("safe", "fail", "abort", "lock", "disarm", "emergency", "parachute",
                  "prohibit", "inhibit", "contingency", "geofence")
def is_safety_req(rid: str, text: str) -> bool:
    return "SAFE" in rid.upper() or any(k in text.lower() for k in _SAFETY_REQ_KW)


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


#: Word-prefix vocabulary for the RESPONSE a state EXECUTES. A fail-safe
#: state is usually named for the safety condition it represents, but
#: REQ-SAFE-007's mandated response is to keep flying: the state is named for
#: the fault (SingleMotorFailure) and the whole response lives in its
#: do-action (maintainControlledFlight). Grading state NAMES only produced
#: the literally-false verdict "no reachable fail-safe state" on run
#: 2026-08-31 — nothing was unreachable; the ontology had no word for a
#: degraded-continue response. What a state does is evidence of the same
#: rank as what it is called. Matching is on camelCase-split word PREFIXES,
#: not substrings, so `checkThresholds` does not smuggle in "hold" and
#: `unlockPayload` does not smuggle in "lock".
_SAFE_ACTION_WORD_PREFIXES = (
    "failsafe", "safe", "abort", "lock", "disarm", "rtb", "return", "land",
    "hold", "emergency", "parachute", "contingency", "controlled",
)


def _executes_safe_response(state) -> bool:
    words: Set[str] = set()
    for name in (
        getattr(state, "entry_action", None),
        getattr(state, "entry_action_def", None),
        getattr(state, "do_action", None),
        getattr(state, "do_action_def", None),
    ):
        if name:
            separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
            words.update(w.lower() for w in re.findall(r"[A-Za-z]+", separated))
    return any(
        word.startswith(prefix)
        for word in words
        for prefix in _SAFE_ACTION_WORD_PREFIXES
    )


def _failsafe_reachable(sm) -> str:
    """'reachable' | 'unreachable' | 'no_safe_state' for one state machine."""
    safe = {
        s.name for s in sm.states
        if any(k in s.name.lower() for k in _SAFE_STATE_KW)
        or _executes_safe_response(s)
    }
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
    trace = extract_requirement_trace(model_text, requirements)
    text = trace.source_by_id
    collapsed = collapsed_response_categories(model_text)   # categories sharing one command

    out: Dict[str, str] = {}
    for rid, parts in trace.owners.items():
        if not is_safety_req(rid, text.get(rid, rid)):
            continue
        sms = [sm for p in parts for sm in by_part.get(p, [])]
        verdicts = {dynamic_fire[p] for p in parts if p in dynamic_fire}
        # structural: is a FAIL-SAFE state reachable? (dynamic firing of a non-safe transition
        # must NOT count as a safety response — so the safe-state requirement gates everything.)
        structural_ok = any(_failsafe_reachable(sm) == "reachable" for sm in sms)
        cat = _response_category(text.get(rid, rid))
        if not sms:
            out[rid] = BEHAVIOR_ABSENT
        elif not structural_ok:
            out[rid] = BEHAVIORALLY_VIOLATED        # no reachable fail-safe state
        elif cat in collapsed:
            out[rid] = RESPONSE_COLLAPSED           # this response shares one command with a
            #   DIFFERENT mandated response → arbitration lost at the interface (e.g. LAND, RTB
            #   and PARACHUTE all emit the same CmdToEmergency → flight ctrl can't act per cause)
        elif "failed" in verdicts:
            out[rid] = BEHAVIORALLY_VIOLATED        # safe state reachable but dynamically NEVER
            #                                         fires → "fake safety" (dynamic-only catch)
        else:
            out[rid] = BEHAVIORALLY_VERIFIED        # fail-safe reachable (+ fires if dynamic ran)
    return out
