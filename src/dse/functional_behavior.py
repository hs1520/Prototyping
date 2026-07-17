"""Behavioural verification of FUNCTIONAL actuation/sequencing requirements (roadmap ①).

A functional requirement that mandates a RESPONSE (release the payload, return to base, land,
report health, navigate) has real behavioural evidence only if a reachable state actually
PRODUCES that response — an entry action or a `send <Cmd> to <port>`. We check the response
ACTION (not just a phase name), so "PhaseDelivery exists" doesn't masquerade as "release logic
verified". Outcomes per functional requirement with a recognised intent:
  behaviorally-verified : a reachable state's entry-action / sent-command matches the response.
  behavior-absent       : no reachable state anywhere produces the response (not modelled —
                          honestly flags the gap, e.g. release/health-report logic missing).
Requirements with no recognised functional intent are left to the base classifier.

Scope (honest): checks that the functional RESPONSE ACTION is reachable. For event-driven
waypoint-update and post-flight-report requirements it also checks the named causal trigger
and requires an explicit seconds-valued latency constraint. Other continuous guard semantics
(e.g. delivery distance and abort inhibition) remain scenario-execution roadmap items.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

from ..simulation.state_extractor import extract_state_machines
from .safety_behavior import (BEHAVIOR_ABSENT, BEHAVIORALLY_VERIFIED, _norm,
                              is_safety_req, reachable_states)

# functional intent → (keywords that mark the intent in the requirement text,
#                       markers that mark the produced response in a state's action/send/name)
_FUNC_INTENT = {
    "release":  (("release", "deliver", "drop", "payload release"),
                 ("release", "deliver", "drop", "payloadcmd", "payloadrelease")),
    "return":   (("return to base", "return-to-base", "rtb", "return trajectory", "return-to-home"),
                 ("rtb", "returnhome", "returntobase", "gohome")),
    "land":     (("land", "landing", "touchdown"), ("land", "touchdown")),
    "navigate": (("navigate", "waypoint", "gps waypoint", "flight plan"),
                 ("navigate", "waypoint", "gotowaypoint", "nav")),
    "report":   (("health report", "status report", "post-flight", "health and status report"),
                 ("report", "healthreport", "postflight", "telemetryreport")),
    "self_test": (("self-test", "self test", "self-check", "self check"),
                  ("selftest", "selfcheck")),
}
# More-specific response intents must win over context words.  For example,
# "transmit a post-flight health report after landing" requires REPORT; a
# reachable LAND action is only the trigger/context and must not satisfy it.
_FUNC_INTENT_PRIORITY = (
    "report", "self_test", "release", "return", "land", "navigate"
)
_REQ_ID_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")
_SATISFY_RE = re.compile(r"satisfy\s+(?:requirement\s+)?(\w*REQ[_-]\w+)", re.IGNORECASE)


@dataclass(frozen=True)
class _ProducedResponse:
    names: frozenset[str]
    trigger_context: str


def _produced_response_records(model_text: str) -> List[_ProducedResponse]:
    """Responses produced by reachable states, with their incoming trigger context."""
    out: List[_ProducedResponse] = []
    for sm in extract_state_machines(model_text):
        reach = reachable_states(sm)
        for s in sm.states:
            if s.name not in reach:
                continue
            names: Set[str] = set()
            if s.entry_action:
                names.add(s.entry_action.lower())
            if s.entry_action_def:
                names.add(s.entry_action_def.lower())
            for cmd, _port in s.sends:
                names.add(cmd.lower())
            if not names:
                continue
            incoming = [
                t for t in sm.transitions
                if not t.is_initial and t.target == s.name and t.source in reach
            ]
            context = " ".join(
                str(value or "")
                for t in incoming
                for value in (
                    t.name, t.source, t.target, t.accept_trigger,
                    " ".join(g.description() for g in t.guards),
                )
            ).lower()
            out.append(_ProducedResponse(frozenset(names), context))
    return out


def _has_required_trigger(req_text: str, record: _ProducedResponse) -> bool:
    """Reject a response action reached through an unrelated event.

    This is intentionally narrow: only causal qualifiers present in the two
    functional sequencing families are enforced. Other functional intents keep
    their existing response-reachability semantics.
    """
    text = req_text.lower()
    context = re.sub(r"[^a-z0-9]+", "", record.trigger_context)
    if "health report" in text and any(k in text for k in ("landing", "post-flight")):
        return "land" in context and "complet" in context
    if "waypoint" in text and any(k in text for k in (
        "modification command", "waypoint-modification", "revised waypoint",
    )):
        waypoint_event = "waypoint" in context and any(
            k in context for k in ("modification", "revision", "revised", "update")
        )
        valid_qualified = "valid" not in text or "valid" in context
        return waypoint_event and valid_qualified
    if any(k in text for k in ("self-test", "self test", "self-check", "self check")):
        return (
            any(k in context for k in ("selftest", "selfcheck"))
            and any(k in context for k in ("poweron", "startup", "start"))
        )
    return True


_SECONDS_RE = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", re.IGNORECASE)
_TIME_ATTR_RE = re.compile(
    r"attribute\s+(\w+)\s*:\s*Real\s*=\s*(\d+(?:\.\d+)?)\s*\[s\]\s*;",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(r"assert\s+constraint\s+\w+\s*\{([^{}]+)\}", re.IGNORECASE)


def _has_required_timing_anchor(model_text: str, req_text: str) -> bool:
    """Require an explicit seconds-valued bound linked by an assert constraint."""
    match = _SECONDS_RE.search(req_text)
    if not match:
        return True
    expected = float(match.group(1))
    low = req_text.lower()
    family = "report" if "health report" in low else "waypoint" if "waypoint" in low else ""
    constraints = " ".join(_CONSTRAINT_RE.findall(model_text)).lower()
    for name, value in _TIME_ATTR_RE.findall(model_text):
        key = name.lower()
        if abs(float(value) - expected) > 1e-9:
            continue
        if family and family not in key:
            continue
        if not any(k in key for k in ("latency", "delay", "time")):
            continue
        if key in constraints:
            return True
    return False


def functional_behavior_status(model_text: str, requirements: List[str]) -> Dict[str, str]:
    """{functional req_id: behaviour status} for FUNC requirements with a recognised
    actuation/sequencing intent (excludes safety reqs — those go through safety_behavior)."""
    produced = _produced_response_records(model_text)
    has_state_machines = bool(extract_state_machines(model_text))
    text = {m.group(0).replace("_", "-"): r for r in requirements
            for m in [_REQ_ID_RE.search(r)] if m}
    satisfied = {_norm(m.group(1)) for m in _SATISFY_RE.finditer(model_text)}

    out: Dict[str, str] = {}
    for rid in satisfied:
        txt = text.get(rid, rid).lower()
        if "FUNC" not in rid.upper() or is_safety_req(rid, txt):
            continue
        markers: Set[str] = set()
        for intent in _FUNC_INTENT_PRIORITY:
            kws, resp = _FUNC_INTENT[intent]
            if any(k in txt for k in kws):
                markers = set(resp)
                break
        if not markers:
            continue                                       # no recognised functional intent
        response_records = [
            record for record in produced
            if any(marker in name for name in record.names for marker in markers)
        ]
        hit = (
            any(_has_required_trigger(txt, record) for record in response_records)
            and _has_required_timing_anchor(model_text, txt)
        )
        if hit:
            out[rid] = BEHAVIORALLY_VERIFIED
        elif has_state_machines:
            out[rid] = BEHAVIOR_ABSENT                     # response action not produced anywhere
    return out
