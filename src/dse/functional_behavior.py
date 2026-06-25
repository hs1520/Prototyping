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

Scope (honest): checks the functional RESPONSE ACTION is reachable; it does NOT verify the
full guard condition / timing (e.g. "within 1 m", "within 1 s") — those need scenario
execution (behavioral_sim) and are roadmap items.
"""
from __future__ import annotations

import re
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
}
_REQ_ID_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")
_SATISFY_RE = re.compile(r"satisfy\s+(?:requirement\s+)?(\w*REQ[_-]\w+)", re.IGNORECASE)


def _produced_responses(model_text: str) -> Set[str]:
    """Lowercased response markers actually produced by any REACHABLE state — from its entry
    action name and the command types it sends."""
    out: Set[str] = set()
    for sm in extract_state_machines(model_text):
        reach = reachable_states(sm)
        for s in sm.states:
            if s.name not in reach:
                continue
            if s.entry_action:
                out.add(s.entry_action.lower())
            for cmd, _port in s.sends:
                out.add(cmd.lower())
    return out


def functional_behavior_status(model_text: str, requirements: List[str]) -> Dict[str, str]:
    """{functional req_id: behaviour status} for FUNC requirements with a recognised
    actuation/sequencing intent (excludes safety reqs — those go through safety_behavior)."""
    produced = _produced_responses(model_text)
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
        for kws, resp in _FUNC_INTENT.values():
            if any(k in txt for k in kws):
                markers |= set(resp)
        if not markers:
            continue                                       # no recognised functional intent
        hit = any(marker in p for p in produced for marker in markers)
        if hit:
            out[rid] = BEHAVIORALLY_VERIFIED
        elif has_state_machines:
            out[rid] = BEHAVIOR_ABSENT                     # response action not produced anywhere
    return out
