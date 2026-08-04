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


_QUANTITY_RE = re.compile(
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?:consecutive\s+)?"
    r"(?P<unit>milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|min|"
    r"kilometres?|kilometers?|km/h|km|metres?|meters?|m/s|m|"
    r"percent|%|hertz|hz|kilohertz|khz|kilograms?|kg|grams?|g|"
    r"degrees?|deg|°\s*c|celsius|rpm|volts?|v|amps?|a)(?![A-Za-z])",
    re.IGNORECASE,
)


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

    obligations = [VerificationObligation(
        obligation_id=f"OBL_{req_id}_001",
        clause=source,
        kind="behavior",
    )]
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
        matching = tuple(
            claim for claim in claim_list
            if claim.all_obligations or obligation.kind in claim.kinds
        )
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
