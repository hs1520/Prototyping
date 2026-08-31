"""Clause-level obligations used to gate requirement verification status.

The verification matrix combines evidence produced at several fidelities.  A
PASS at one fidelity is not automatically evidence for every clause of a
compound requirement.  This module provides the small, deterministic data model
used to make that boundary explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Optional, Tuple


# An inhibition requirement ("shall not transition...", "shall block arming
# while...") is anchored by the response that is withheld -- a guard or a
# fault-path transition -- never by initial-state semantics. Every reader
# that routes requirements by this distinction uses this one predicate; two
# modules once kept diverging keyword lists, and a phrasing in their
# difference would have been routed to different evidence standards.
_INHIBITION_RE = re.compile(
    r"\bshall\s+not\s+(?:transition|initiate|enter|arm|release|deploy|start|unlock|open)\w*\b"
    r"|\binhibit\w*\b|\bprevent\w*\b|\bsuppress\w*\b"
    r"|\bblock\w*\b|\block\s*-?\s*out\b|\blockout\b",
    re.IGNORECASE,
)

_POSITIVE_INHIBITION_RE = re.compile(
    r"\bshall\s+(?:maintain|keep|remain)\s+"
    r"(?P<state>.+?)\s+(?:whenever|while|when|during)\s+"
    r"(?P<condition>.+?)(?:,|\.|$)",
    re.IGNORECASE,
)

_NEGATIVE_TRANSITION_RE = re.compile(
    r"\bshall\s+not\s+transition\s+to\s+(?P<state>.+?)\s+"
    r"(?:if|when|while|during)\s+(?P<condition>.+?)(?:,|\.|$)",
    re.IGNORECASE,
)

_BLOCK_TRANSITION_RE = re.compile(
    r"\bshall\s+block\s+(?:the\s+)?transition\s+to\s+"
    r"(?P<state>.+?)\s+(?:if|when|while|during)\s+"
    r"(?P<condition>.+?)(?:,|\.|$)",
    re.IGNORECASE,
)

_SAFE_STATE_TERMS = frozenset({"locked", "secured", "disabled", "disarmed", "closed"})
_CONDITION_SIGNAL_TERMS = frozenset({
    "abort", "failure", "fault", "unsafe", "emergency", "sensor", "open", "loss",
})
_SEMANTIC_STOP_WORDS = frozenset({
    "a", "all", "an", "and", "any", "condition", "during", "if", "in", "is",
    "of", "or", "state", "system", "the", "to", "when", "whenever", "while",
})


class RequirementIntentKind(str, Enum):
    BEHAVIOR = "behavior"
    INHIBITION = "inhibition"


class ObligationKind(str, Enum):
    BEHAVIOR = "behavior"
    INHIBITION = "inhibition"
    PROTOCOL_CONFORMANCE = "protocol_conformance"
    ACTIVE_ROUTE_CHANGE = "active_route_change"
    CONTROLLED_FLIGHT = "controlled_flight"
    PRECEDENCE = "precedence"
    RESPONSE_TIME = "response_time"
    POSITION_ACCURACY = "position_accuracy"
    ATTITUDE_RMS = "attitude_rms"
    MASS = "mass"
    ENDURANCE = "endurance"
    SPEED = "speed"
    TEMPERATURE = "temperature"
    HOVER_THROTTLE_MARGIN = "hover_throttle_margin"
    PAYLOAD = "payload"
    LOOP_RATE = "loop_rate"
    ALTITUDE = "altitude"
    BATTERY_THRESHOLD = "battery_threshold"
    SEPARATION = "separation"
    DETECTION_RANGE = "detection_range"
    RANGE = "range"
    WIND_SPEED = "wind_speed"
    QUANTITATIVE_CONSTRAINT = "quantitative_constraint"


class EvidenceCapability(str, Enum):
    BEHAVIOR_OBSERVED = "behavior_observed"
    INHIBITION_BEHAVIOR_OBSERVED = "inhibition_behavior_observed"
    PHYSICAL_INHIBITION_OBSERVED = "physical_inhibition_observed"
    WIRE_PROTOCOL_OBSERVED = "wire_protocol_observed"
    ACTIVE_ROUTE_CHANGE_OBSERVED = "active_route_change_observed"
    CONTROLLED_FLIGHT_OBSERVED = "controlled_flight_observed"
    SAFETY_PRECEDENCE_OBSERVED = "safety_precedence_observed"
    RESPONSE_TIME_MEASURED = "response_time_measured"
    POSITION_ERROR_MEASURED = "position_error_measured"
    ATTITUDE_RMS_MEASURED = "attitude_rms_measured"
    MASS_MEASURED = "mass_measured"
    ENDURANCE_ANALYSED = "endurance_analysed"
    SPEED_MEASURED = "speed_measured"
    TEMPERATURE_VERIFIED = "temperature_verified"
    HOVER_THROTTLE_MARGIN_MEASURED = "hover_throttle_margin_measured"
    PAYLOAD_STATE_OBSERVED = "payload_state_observed"
    LOOP_RATE_MEASURED = "loop_rate_measured"
    ALTITUDE_MEASURED = "altitude_measured"
    BATTERY_THRESHOLD_VERIFIED = "battery_threshold_verified"
    SEPARATION_MEASURED = "separation_measured"
    DETECTION_RANGE_MEASURED = "detection_range_measured"
    RANGE_MEASURED = "range_measured"
    WIND_SPEED_MEASURED = "wind_speed_measured"
    QUANTITATIVE_CONSTRAINT_VERIFIED = "quantitative_constraint_verified"
    MODE_TRANSITION_OBSERVED = "mode_transition_observed"
    SENSOR_STATE_OBSERVED = "sensor_state_observed"
    ACTUATOR_COMMAND_OBSERVED = "actuator_command_observed"
    MISSION_STORAGE_READBACK_OBSERVED = "mission_storage_readback_observed"


_CAPABILITY_ENTAILS = {
    EvidenceCapability.BEHAVIOR_OBSERVED: frozenset({ObligationKind.BEHAVIOR}),
    EvidenceCapability.INHIBITION_BEHAVIOR_OBSERVED: frozenset({ObligationKind.INHIBITION}),
    EvidenceCapability.PHYSICAL_INHIBITION_OBSERVED: frozenset({ObligationKind.INHIBITION}),
    EvidenceCapability.WIRE_PROTOCOL_OBSERVED: frozenset({ObligationKind.PROTOCOL_CONFORMANCE}),
    EvidenceCapability.ACTIVE_ROUTE_CHANGE_OBSERVED: frozenset({ObligationKind.ACTIVE_ROUTE_CHANGE}),
    EvidenceCapability.CONTROLLED_FLIGHT_OBSERVED: frozenset({ObligationKind.CONTROLLED_FLIGHT}),
    EvidenceCapability.SAFETY_PRECEDENCE_OBSERVED: frozenset({ObligationKind.PRECEDENCE}),
    EvidenceCapability.RESPONSE_TIME_MEASURED: frozenset({ObligationKind.RESPONSE_TIME}),
    EvidenceCapability.POSITION_ERROR_MEASURED: frozenset({ObligationKind.POSITION_ACCURACY}),
    EvidenceCapability.ATTITUDE_RMS_MEASURED: frozenset({ObligationKind.ATTITUDE_RMS}),
    EvidenceCapability.MASS_MEASURED: frozenset({ObligationKind.MASS}),
    EvidenceCapability.ENDURANCE_ANALYSED: frozenset({ObligationKind.ENDURANCE}),
    EvidenceCapability.SPEED_MEASURED: frozenset({ObligationKind.SPEED}),
    EvidenceCapability.TEMPERATURE_VERIFIED: frozenset({ObligationKind.TEMPERATURE}),
    EvidenceCapability.HOVER_THROTTLE_MARGIN_MEASURED: frozenset({ObligationKind.HOVER_THROTTLE_MARGIN}),
    EvidenceCapability.PAYLOAD_STATE_OBSERVED: frozenset({ObligationKind.PAYLOAD}),
    EvidenceCapability.LOOP_RATE_MEASURED: frozenset({
        ObligationKind.BEHAVIOR,
        ObligationKind.LOOP_RATE,
    }),
    EvidenceCapability.ALTITUDE_MEASURED: frozenset({ObligationKind.ALTITUDE}),
    EvidenceCapability.BATTERY_THRESHOLD_VERIFIED: frozenset({ObligationKind.BATTERY_THRESHOLD}),
    EvidenceCapability.SEPARATION_MEASURED: frozenset({ObligationKind.SEPARATION}),
    EvidenceCapability.DETECTION_RANGE_MEASURED: frozenset({ObligationKind.DETECTION_RANGE}),
    EvidenceCapability.RANGE_MEASURED: frozenset({ObligationKind.RANGE}),
    EvidenceCapability.WIND_SPEED_MEASURED: frozenset({ObligationKind.WIND_SPEED}),
    EvidenceCapability.QUANTITATIVE_CONSTRAINT_VERIFIED: frozenset({ObligationKind.QUANTITATIVE_CONSTRAINT}),
    EvidenceCapability.MODE_TRANSITION_OBSERVED: frozenset({ObligationKind.BEHAVIOR}),
    EvidenceCapability.SENSOR_STATE_OBSERVED: frozenset(),
    EvidenceCapability.ACTUATOR_COMMAND_OBSERVED: frozenset(),
    EvidenceCapability.MISSION_STORAGE_READBACK_OBSERVED: frozenset(),
}


@dataclass(frozen=True)
class RequirementIntent:
    kind: RequirementIntentKind
    condition_terms: FrozenSet[str] = field(default_factory=frozenset)
    required_state_terms: FrozenSet[str] = field(default_factory=frozenset)
    forbidden_state_terms: FrozenSet[str] = field(default_factory=frozenset)


def semantic_terms(text: str) -> FrozenSet[str]:
    separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("-", " ")
    return frozenset(
        token for token in re.findall(r"[A-Za-z]+", separated.lower())
        if token not in _SEMANTIC_STOP_WORDS
    )


def _condition_terms(text: str) -> FrozenSet[str]:
    terms = semantic_terms(text)
    signals = terms & _CONDITION_SIGNAL_TERMS
    return signals or terms


def parse_requirement_intent(text: str) -> RequirementIntent:
    source = " ".join(text.split())
    positive = _POSITIVE_INHIBITION_RE.search(source)
    if positive:
        required = semantic_terms(positive.group("state")) & _SAFE_STATE_TERMS
        condition = _condition_terms(positive.group("condition"))
        if required and condition:
            return RequirementIntent(
                kind=RequirementIntentKind.INHIBITION,
                condition_terms=condition,
                required_state_terms=required,
            )

    for pattern in (_NEGATIVE_TRANSITION_RE, _BLOCK_TRANSITION_RE):
        negative = pattern.search(source)
        if negative:
            return RequirementIntent(
                kind=RequirementIntentKind.INHIBITION,
                condition_terms=_condition_terms(negative.group("condition")),
                forbidden_state_terms=semantic_terms(negative.group("state")),
            )

    if _INHIBITION_RE.search(source):
        return RequirementIntent(kind=RequirementIntentKind.INHIBITION)
    return RequirementIntent(kind=RequirementIntentKind.BEHAVIOR)


def classify_requirement_intent(text: str) -> RequirementIntentKind:
    """Classify the response shape asserted by a requirement sentence."""
    return parse_requirement_intent(text).kind


def is_inhibition_requirement(text: str) -> bool:
    """True when the requirement is phrased as an inhibition."""
    return classify_requirement_intent(text) is RequirementIntentKind.INHIBITION


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
    "ingress", "temperature", "certification", "certified", "certificate",
    "material", "materials", "encrypt", "encrypted", "encryption", "aes",
)

#: Connectives that introduce the MEANS or MEDIUM a capability runs over. A
#: sentence that names an untestable medium for an otherwise testable
#: capability is asserting two things with different verification means, and a
#: single obligation over the whole sentence lets either half misrepresent the
#: other: the untestable half drags the capability out of scope, and a test of
#: the capability would appear to close the untestable half. Splitting there is
#: the only way both can be reported truthfully.
#:
#: Deliberately NOT included:
#:
#: " using " — it introduces the MEANS by which the capability is achieved, not
#: a medium the capability runs over, and a means is not separable from it.
#: "authenticate every operator command … using an AES challenge" split into a
#: testable "authenticate every operator command" and an untestable "AES
#: challenge", so a protocol-level test could close the authentication clause
#: while the mechanism that IS the authentication went untested. Contrast
#: " over an AES-256 encrypted RF channel": the channel is a medium, and
#: MAVLink v2 conformance over it is a real, separate capability.
#:
#: "in accordance with", "compliant with", " across ", " through " — those
#: qualify HOW or UNDER WHAT CONDITIONS the same capability must behave.
#: "operate across an ambient temperature range" is one obligation, not two.
_MEDIUM_CONNECTIVES = (" over ", " via ")

#: A capability clause has to survive the cut as a requirement in its own
#: right. "The system shall operate" is what is left when a condition is
#: mistaken for a medium, and it asserts nothing testable.
_MIN_CAPABILITY_WORDS = 8


def _inspection_positions(low: str) -> list:
    return sorted(
        match.start()
        for term in INSPECTION_TERMS
        for match in [re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", low)]
        if match is not None
    )


def contains_term(text: str, term: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(term.lower())}(?![A-Za-z0-9])",
        text.lower(),
    ) is not None


def contains_any_term(text: str, terms: Iterable[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


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


def states_acceptance_threshold(text: str) -> bool:
    """Does the requirement itself name a measurable acceptance threshold?

    This is what separates a PASS that applied the REQUIREMENT's criterion from
    one that applied an interpretation of its own. "deploy the parachute within
    0.5 seconds" states the bar; "maintain controlled flight" states none, so
    any verdict on it embeds a definition the requirement never gave — and a
    result that does not declare that definition cannot be read as closing it.
    """
    return bool(_QUANTITY_RE.search(_clean(text)))


@dataclass(frozen=True)
class VerificationObligation:
    obligation_id: str
    clause: str
    kind: ObligationKind


class CriterionSource(str, Enum):
    REQUIREMENT = "requirement"
    DERIVED_FROM_REQUIREMENT = "derived_from_requirement"
    VERIFICATION_POLICY = "verification_policy"
    ENGINEERING_JUDGEMENT = "engineering_judgement"


@dataclass(frozen=True)
class VerificationCriterion:
    metric: str
    operator: str
    threshold: float
    unit: str
    source: CriterionSource
    basis: str
    accepted_for_requirement: bool


@dataclass(frozen=True)
class CriterionEvaluation:
    interpretation: str
    passed_runs: int
    total_runs: int


@dataclass(frozen=True)
class EvidenceClaim:
    """A structured claim made by one concrete verification result."""

    description: str
    status: str  # verified | failed | planned | partial | out-of-sim-scope
    capabilities: FrozenSet[EvidenceCapability] = field(default_factory=frozenset)
    applies_to_matching_clauses: bool = False
    criterion: Optional[VerificationCriterion] = None
    sensitivity: Tuple[CriterionEvaluation, ...] = field(default_factory=tuple)
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
        if self.clause_exclude_terms and contains_any_term(clause, self.clause_exclude_terms):
            return False
        if self.clause_terms and not contains_any_term(clause, self.clause_terms):
            return False
        if self.applies_to_matching_clauses:
            return True
        entailed = frozenset().union(*(
            _CAPABILITY_ENTAILS[capability]
            for capability in self.capabilities
        )) if self.capabilities else frozenset()
        return obligation.kind in entailed


@dataclass(frozen=True)
class ObligationResult:
    obligation_id: str
    clause: str
    kind: ObligationKind
    status: str
    evidence: Tuple[str, ...] = field(default_factory=tuple)
    criteria: Tuple[VerificationCriterion, ...] = field(default_factory=tuple)
    sensitivity: Tuple[CriterionEvaluation, ...] = field(default_factory=tuple)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _quantity_kind(
    text: str, start: int, end: int, unit: str, previous_end: int
) -> ObligationKind:
    low = text.lower()
    local = low[max(previous_end, start - 65):min(len(low), end + 24)]
    if unit.lower().replace(" ", "") in {"°c", "celsius"}:
        return ObligationKind.TEMPERATURE
    if any(word in local for word in (
        "take-off mass", "takeoff mass", "mtow", "total mass",
    )):
        return ObligationKind.MASS
    if "hover throttle" in local or "throttle margin" in local:
        return ObligationKind.HOVER_THROTTLE_MARGIN
    if "roll and pitch" in local or "pitch rms" in local or "attitude" in local:
        return ObligationKind.ATTITUDE_RMS
    if "circular error" in local or "cep" in local or "position" in local:
        return ObligationKind.POSITION_ACCURACY
    if "payload" in local and unit.lower() in {"kg", "kilogram", "kilograms"}:
        return ObligationKind.PAYLOAD
    if "gross mass" in local:
        return ObligationKind.MASS
    if "loop" in local and unit.lower() in {"hz", "hertz", "khz", "kilohertz"}:
        return ObligationKind.LOOP_RATE
    if "endurance" in local or "sustain flight" in local or "flight for" in local:
        return ObligationKind.ENDURANCE
    if "altitude" in local or "above ground" in local or "agl" in local:
        return ObligationKind.ALTITUDE
    if "temperature" in local or "ambient" in local or "celsius" in local:
        return ObligationKind.TEMPERATURE
    if any(word in local for word in ("state-of-charge", "state of charge", "battery")):
        return ObligationKind.BATTERY_THRESHOLD
    if "separation" in local or "clearance" in local:
        return ObligationKind.SEPARATION
    if "detect" in local and any(word in local for word in (
        "obstacle", "range", "metre", "meter",
    )):
        return ObligationKind.DETECTION_RANGE
    if "range" in local and unit.lower() in {
        "m", "metre", "metres", "meter", "meters", "km", "kilometre",
        "kilometres", "kilometer", "kilometers",
    }:
        return ObligationKind.RANGE
    if any(word in local for word in ("headwind", "tailwind", "crosswind", "wind speed")):
        return ObligationKind.WIND_SPEED
    if any(word in local for word in ("speed", "closing")):
        return ObligationKind.SPEED
    if unit.lower() in {
        "s", "sec", "secs", "second", "seconds", "ms", "millisecond",
        "milliseconds",
    }:
        return ObligationKind.RESPONSE_TIME
    return ObligationKind.QUANTITATIVE_CONSTRAINT


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


def _response_obligation_kind(text: str) -> ObligationKind:
    intent = classify_requirement_intent(text)
    if intent is RequirementIntentKind.INHIBITION:
        return ObligationKind.INHIBITION
    low = text.lower()
    if "mavlink" in low and "protocol" in low:
        return ObligationKind.PROTOCOL_CONFORMANCE
    if "revised waypoint" in low and "active flight plan" in low:
        return ObligationKind.ACTIVE_ROUTE_CHANGE
    if "controlled flight" in low:
        return ObligationKind.CONTROLLED_FLIGHT
    return ObligationKind.BEHAVIOR


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
    response_kind = _response_obligation_kind(source)
    if split is None:
        obligations = [VerificationObligation(
            obligation_id=f"OBL_{req_id}_001",
            clause=source,
            kind=response_kind,
        )]
    else:
        capability, medium = split
        obligations = [
            VerificationObligation(
                obligation_id=f"OBL_{req_id}_001",
                clause=capability,
                kind=_response_obligation_kind(capability),
            ),
            VerificationObligation(
                obligation_id=f"OBL_{req_id}_001M",
                clause=medium,
                kind=ObligationKind.BEHAVIOR,
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
    precedence = re.search(
        r"\b(?:taking\s+precedence|takes\s+precedence|priorit(?:y|ised|ized))\b[^.]*",
        source,
        re.IGNORECASE,
    )
    if precedence:
        obligations.append(VerificationObligation(
            obligation_id=f"OBL_{req_id}_{len(obligations) + 1:03d}",
            clause=_clean(precedence.group(0)).strip(" .,"),
            kind=ObligationKind.PRECEDENCE,
        ))
    return tuple(obligations)


_STATUS_PRIORITY = {
    "failed": 5,
    "verified": 4,
    "partial": 3,
    "planned": 2,
    "out-of-sim-scope": 1,
}


def evaluate_evidence(
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
                (
                    "partial"
                    if claim.criterion is not None
                    and not claim.criterion.accepted_for_requirement
                    and claim.status in {"verified", "failed"}
                    else claim.status
                    for claim in matching
                ),
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
            criteria=tuple(
                claim.criterion for claim in matching
                if claim.criterion is not None
            ),
            sensitivity=tuple(
                evaluation
                for claim in matching
                for evaluation in claim.sensitivity
            ),
        ))
    return tuple(results)


def all_obligations_verified(results: Iterable[ObligationResult]) -> bool:
    values = tuple(results)
    return bool(values) and all(item.status == "verified" for item in values)
