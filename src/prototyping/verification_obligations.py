"""Clause-level obligations used to gate requirement verification status.

The verification matrix combines evidence produced at several fidelities.  A
PASS at one fidelity is not automatically evidence for every clause of a
compound requirement.  This module provides the small, deterministic data model
used to make that boundary explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Tuple


# An inhibition requirement ("shall not transition...", "shall block arming
# while...") is anchored by the response that is withheld -- a guard or a
# fault-path transition -- never by initial-state semantics. Every reader
# that routes requirements by this distinction uses this one predicate; two
# modules once kept diverging keyword lists, and a phrasing in their
# difference would have been routed to different evidence standards.
_INHIBITION_RE = re.compile(
    r"\bshall\s+not\b|\binhibit\w*\b|\bprevent\w*\b|\bsuppress\w*\b"
    r"|\bblock\w*\b|\block\s*-?\s*out\b|\blockout\b",
    re.IGNORECASE,
)


def is_inhibition_requirement(text: str) -> bool:
    """True when the requirement is phrased as an inhibition."""
    return bool(_INHIBITION_RE.search(text or ""))


_QUANTITY_RE = re.compile(
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?:consecutive\s+)?"
    r"(?P<unit>milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|min|"
    r"kilometres?|kilometers?|km/h|km|metres?|meters?|m/s|m|"
    r"percent|%|hertz|hz|kilohertz|khz|kilograms?|kg|grams?|g|"
    r"degrees?|deg|°\s*c|celsius|rpm|volts?|v|amps?|a)(?![A-Za-z])",
    re.IGNORECASE,
)


#: Terms whose presence makes a clause inspection/analysis work — no simulator
#: can test them. Kept in step with verification_matrix._INSPECTION_KWS.
INSPECTION_TERMS = (
    "comply", "compliance", "regulation", "easa", "faa", "astm", "ip54", "ip5",
    "ingress", "temperature", "certif", "material", "encrypt", "aes",
)

#: Connectives that introduce the MEANS or MEDIUM a capability runs over. A
#: sentence that names an untestable medium for an otherwise testable
#: capability is asserting two things with different verification means, and a
#: single obligation over the whole sentence lets either half misrepresent the
#: other: the untestable half drags the capability out of scope, and a test of
#: the capability would appear to close the untestable half. Splitting there is
#: the only way both can be reported truthfully.
#:
#: Deliberately NOT included: "in accordance with", "compliant with", " across "
#: and " through ". Those qualify HOW or UNDER WHAT CONDITIONS the same
#: capability must behave rather than naming a separate medium — "operate
#: across an ambient temperature range" is one obligation, not two — so
#: splitting them would invent an obligation the requirement never asserted.
_MEDIUM_CONNECTIVES = (" over ", " via ", " using ")

#: A capability clause has to survive the cut as a requirement in its own
#: right. "The system shall operate" is what is left when a condition is
#: mistaken for a medium, and it asserts nothing testable.
_MIN_CAPABILITY_WORDS = 8


def _inspection_positions(low: str) -> list:
    return sorted(
        position
        for term in INSPECTION_TERMS
        for position in [low.find(term)]
        if position >= 0
    )


def split_capability_and_medium(text: str):
    """Split "<capability> over <untestable medium>" into its two clauses.

    Returns ``(capability, medium)``, or ``None`` when the sentence does not
    have that shape — which is the common case, and stays a single obligation.
    """
    source = _clean(text)
    low = source.lower()
    positions = _inspection_positions(low)
    if not positions:
        return None
    first = positions[0]

    cut = -1
    connective = ""
    for token in _MEDIUM_CONNECTIVES:
        index = low.rfind(token, 0, first)
        if index > cut:
            cut, connective = index, token
    if cut < 0:
        return None                      # the untestable term is in the main clause

    capability = source[:cut].strip(" .,;")
    medium = source[cut + len(connective):].strip(" .,;")
    if "shall" not in capability.lower() or not medium:
        return None                      # left side no longer reads as a requirement
    if len(capability.split()) < _MIN_CAPABILITY_WORDS:
        return None                      # degenerate stub, not a capability
    if _inspection_positions(capability.lower()):
        return None                      # the split failed to isolate the untestable half
    return capability, medium


@dataclass(frozen=True)
class VerificationObligation:
    obligation_id: str
    clause: str
    kind: str


@dataclass(frozen=True)
class EvidenceClaim:
    """A structured claim made by one concrete verification result."""

    description: str
    status: str  # verified | failed | planned | partial | out-of-sim-scope
    kinds: FrozenSet[str] = field(default_factory=frozenset)
    all_obligations: bool = False
    #: Restrict the claim to obligations whose CLAUSE mentions one of these
    #: terms. An inspection finding about encryption must not be stamped over a
    #: protocol clause that a test does cover.
    clause_terms: FrozenSet[str] = field(default_factory=frozenset)
    #: The inverse: a claim that must never close a clause carrying one of
    #: these terms. A wire-level protocol test says nothing about encryption
    #: even when both live in the same requirement.
    clause_exclude_terms: FrozenSet[str] = field(default_factory=frozenset)

    def covers(self, obligation: "VerificationObligation") -> bool:
        clause = (obligation.clause or "").lower()
        if self.clause_exclude_terms and any(t in clause for t in self.clause_exclude_terms):
            return False
        if self.clause_terms and not any(t in clause for t in self.clause_terms):
            return False
        return self.all_obligations or obligation.kind in self.kinds


@dataclass(frozen=True)
class ObligationResult:
    obligation_id: str
    clause: str
    kind: str
    status: str
    evidence: Tuple[str, ...] = field(default_factory=tuple)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _quantity_kind(
    text: str, start: int, end: int, unit: str, previous_end: int
) -> str:
    low = text.lower()
    local = low[max(previous_end, start - 65):min(len(low), end + 24)]
    if unit.lower().replace(" ", "") in {"°c", "celsius"}:
        return "temperature"
    if any(word in local for word in (
        "take-off mass", "takeoff mass", "mtow", "total mass",
    )):
        return "mass"
    if "hover throttle" in local or "throttle margin" in local:
        return "hover_throttle_margin"
    if "roll and pitch" in local or "pitch rms" in local or "attitude" in local:
        return "attitude_rms"
    if "circular error" in local or "cep" in local or "position" in local:
        return "position_accuracy"
    if "payload" in local and unit.lower() in {"kg", "kilogram", "kilograms"}:
        return "payload"
    if "gross mass" in local:
        return "mass"
    if "loop" in local and unit.lower() in {"hz", "hertz", "khz", "kilohertz"}:
        return "loop_rate"
    if "endurance" in local or "sustain flight" in local or "flight for" in local:
        return "endurance"
    if "altitude" in local or "above ground" in local or "agl" in local:
        return "altitude"
    if "temperature" in local or "ambient" in local or "celsius" in local:
        return "temperature"
    if any(word in local for word in ("state-of-charge", "state of charge", "battery")):
        return "battery_threshold"
    if "separation" in local or "clearance" in local:
        return "separation"
    if "detect" in local and any(word in local for word in (
        "obstacle", "range", "metre", "meter",
    )):
        return "detection_range"
    if "range" in local and unit.lower() in {
        "m", "metre", "metres", "meter", "meters", "km", "kilometre",
        "kilometres", "kilometer", "kilometers",
    }:
        return "range"
    if any(word in local for word in ("headwind", "tailwind", "crosswind", "wind speed")):
        return "wind_speed"
    if any(word in local for word in ("speed", "closing")):
        return "speed"
    if unit.lower() in {
        "s", "sec", "secs", "second", "seconds", "ms", "millisecond",
        "milliseconds",
    }:
        return "response_time"
    return "quantitative_constraint"


def _clause_around(text: str, start: int, end: int) -> str:
    low = text.lower()
    left_candidates = [text.rfind(",", 0, start), text.rfind(";", 0, start)]
    for marker in (" and ", " while "):
        left_candidates.append(low.rfind(marker, 0, start))
    left = max(left_candidates) + 1
    right_candidates = [
        position
        for position in (text.find(",", end), text.find(";", end))
        if position >= 0
    ]
    for marker in (" and ", " while "):
        position = low.find(marker, end)
        if position >= 0:
            right_candidates.append(position)
    terminal = re.search(r"[.](?:\s|$)", text[end:])
    if terminal:
        right_candidates.append(end + terminal.start())
    right = min(right_candidates) if right_candidates else len(text)
    return _clean(text[left:right]).strip(" .")


def compile_verification_obligations(
    req_id: str, text: str
) -> Tuple[VerificationObligation, ...]:
    """Compile the mandatory behaviour plus every explicit physical threshold.

    This is intentionally conservative and source-only.  It does not infer
    obligations from design documents or generated model structure.
    """
    source = _clean(text)
    if not source:
        return tuple()

    split = split_capability_and_medium(source)
    if split is None:
        obligations = [VerificationObligation(
            obligation_id=f"OBL_{req_id}_001",
            clause=source,
            kind="behavior",
        )]
    else:
        capability, medium = split
        obligations = [
            VerificationObligation(
                obligation_id=f"OBL_{req_id}_001",
                clause=capability,
                kind="behavior",
            ),
            VerificationObligation(
                obligation_id=f"OBL_{req_id}_001M",
                clause=medium,
                kind="behavior",
            ),
        ]
    previous_end = 0
    for index, match in enumerate(_QUANTITY_RE.finditer(source), 2):
        obligations.append(VerificationObligation(
            obligation_id=f"OBL_{req_id}_{index:03d}",
            clause=_clause_around(source, match.start(), match.end()),
            kind=_quantity_kind(
                source, match.start(), match.end(), match.group("unit"),
                previous_end,
            ),
        ))
        previous_end = match.end()
    return tuple(obligations)


_STATUS_PRIORITY = {
    "failed": 5,
    "verified": 4,
    "partial": 3,
    "planned": 2,
    "out-of-sim-scope": 1,
}


def evaluate_obligations(
    obligations: Iterable[VerificationObligation],
    claims: Iterable[EvidenceClaim],
    *,
    blocked: bool = False,
) -> Tuple[ObligationResult, ...]:
    results = []
    claim_list = tuple(claims)
    for obligation in obligations:
        matching = tuple(claim for claim in claim_list if claim.covers(obligation))
        if blocked:
            status = "blocked"
        elif matching:
            status = max(
                (claim.status for claim in matching),
                key=lambda value: _STATUS_PRIORITY.get(value, 0),
            )
        else:
            status = "unverified"
        results.append(ObligationResult(
            obligation_id=obligation.obligation_id,
            clause=obligation.clause,
            kind=obligation.kind,
            status=status,
            evidence=tuple(claim.description for claim in matching),
        ))
    return tuple(results)


def all_obligations_verified(results: Iterable[ObligationResult]) -> bool:
    values = tuple(results)
    return bool(values) and all(item.status == "verified" for item in values)
