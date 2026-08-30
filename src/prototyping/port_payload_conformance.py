"""Send-payload vs port-payload type conformance for generated SysML v2.

Measured on 8 of 24 archived authoritative runs (5af6c666 et al.): the model
declares a correctly typed command port (``port parachuteCmd :
ParachuteCmdPort`` whose payload item is ``ParachuteCmdData``) yet the response
action sends the *detected-failure event itself* through it
(``send CriticalPropulsionFailure() to parachuteCmd``).  Syntax and the
zero-warning qualification both accept this, and the defect then surfaces only
at the SITL traceability gate as a blocked evidence row.  The type
inconsistency is fully decidable from the model text, so it belongs in the
refinement loop while the author is still in session.

Deliberately conservative: a finding is emitted only when EVERY link in the
chain resolves — the send sits inside a part definition, the target is a port
declared on that part, the port's type is a port definition found in the model,
that definition declares at least one typed payload item, and the sent payload
name is itself a known item definition.  Anything unresolved is silence, never
a guess: a false positive here would send the repair loop chasing a healthy
model.
"""
from __future__ import annotations

import re

from ..utils.sysml_text_utils import find_block_end
from .namespace_integrity import _mask_comments_and_strings

_PART_DEF_RE = re.compile(r"\bpart\s+def\s+(?P<name>[A-Za-z_]\w*)\s*\{")
_PORT_DEF_RE = re.compile(r"\bport\s+def\s+(?P<name>[A-Za-z_]\w*)\s*\{")
_ACTION_DEF_RE = re.compile(r"\baction\s+def\s+(?P<name>[A-Za-z_]\w*)\s*\{")
_ITEM_DEF_RE = re.compile(r"\bitem\s+def\s+(?P<name>[A-Za-z_]\w*)")
_PORT_USAGE_RE = re.compile(
    r"\b(?:in\s+|out\s+|inout\s+)?port\s+(?!def\b)(?P<name>[A-Za-z_]\w*)\s*:\s*"
    r"(?P<conj>~?)\s*(?P<type>[A-Za-z_][\w:]*)"
)
_PORT_ITEM_RE = re.compile(
    r"\bitem\s+(?!def\b)(?:[A-Za-z_]\w*)\s*:\s*(?P<type>[A-Za-z_][\w:]*)"
)
_SEND_RE = re.compile(
    r"\bsend\s+(?P<payload>[A-Za-z_]\w*)\s*(?:\([^()]*\))?\s+to\s+"
    r"(?P<target>[A-Za-z_]\w*)\s*;"
)


def _short(name: str) -> str:
    return name.split("::")[-1]


def _blocks(masked: str, pattern: re.Pattern) -> list[tuple[str, int, int]]:
    """(name, body_start, body_end) for every ``<kind> def Name { ... }``."""
    found = []
    for match in pattern.finditer(masked):
        opening = masked.find("{", match.end() - 1)
        if opening == -1:
            continue
        closing = find_block_end(masked, opening)
        if closing == -1:
            continue
        found.append((match.group("name"), opening + 1, closing))
    return found


def check_port_payload_conformance(model_text: str) -> dict:
    """Findings for sends whose payload type contradicts the port's item type."""
    source = str(model_text or "")
    masked = _mask_comments_and_strings(source)

    item_defs = {m.group("name") for m in _ITEM_DEF_RE.finditer(masked)}
    port_def_payloads: dict[str, set[str]] = {}
    for name, start, end in _blocks(masked, _PORT_DEF_RE):
        body = masked[start:end]
        payloads = {_short(m.group("type")) for m in _PORT_ITEM_RE.finditer(body)}
        if payloads:
            port_def_payloads[name] = payloads

    findings: list[dict] = []
    for part_name, part_start, part_end in _blocks(masked, _PART_DEF_RE):
        body = masked[part_start:part_end]
        ports = {
            m.group("name"): _short(m.group("type"))
            for m in _PORT_USAGE_RE.finditer(body)
        }
        actions = [
            (name, start, end)
            for name, start, end in _blocks(body, _ACTION_DEF_RE)
        ]
        for send in _SEND_RE.finditer(body):
            payload = send.group("payload")
            target = send.group("target")
            port_type = ports.get(target)
            if port_type is None:
                continue                      # not a port on this part
            declared = port_def_payloads.get(port_type)
            if not declared:
                continue                      # port def missing or untyped payload
            if payload not in item_defs:
                continue                      # sent name is not a known item def
            if payload in declared:
                continue                      # conformant
            action_name = next(
                (name for name, start, end in actions
                 if start <= send.start() < end),
                None,
            )
            findings.append({
                "part": part_name,
                "action": action_name,
                "port": target,
                "port_type": port_type,
                "sent": payload,
                "declared": sorted(declared),
            })
    return {"send_port_mismatches": findings}


def port_payload_conformance_issues(model_text: str) -> list[str]:
    """Refinement-actionable issues for send/port payload-type mismatches.

    Feeds the defect the SITL traceability gate later blocks (a response
    action sending the trigger event instead of the port's command payload)
    into the refinement loop while the author is still in session.  A
    deterministic rewrite is deliberately not attempted: whether the fix is
    to send the declared payload or to retype the port is the author's
    intent to state.
    """
    report = check_port_payload_conformance(model_text)
    issues: list[str] = []
    for finding in report["send_port_mismatches"]:
        where = (
            f"action def '{finding['action']}'" if finding["action"]
            else "a behaviour"
        )
        declared = "' or '".join(finding["declared"])
        issues.append(
            f"[PORT-PAYLOAD] In part '{finding['part']}', {where} sends "
            f"'{finding['sent']}' to port '{finding['port']}', but that "
            f"port's type '{finding['port_type']}' declares payload item "
            f"type '{declared}'. The receiving side matches on the port's "
            f"declared payload, so this send cannot be consumed: send "
            f"'{declared}' through '{finding['port']}' (keeping the "
            f"triggering event as the transition's accept), or retype the "
            "port if the intent changed."
        )
    return issues
