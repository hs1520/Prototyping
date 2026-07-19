"""Executable verification contracts derived from natural-language requirements.

The quantitative DSE extractor answers questions such as "how much endurance?".
Verification needs a different structure: stimulus, operational envelope, response
threshold, and acceptance oracle.  Keeping that structure in one auditable module
prevents a downstream test from silently strengthening or weakening a requirement.

The original obstacle contract remains backward compatible.  Option 2 adds a
typed, multi-obligation contract bundle for a deliberately bounded family set;
unsupported requirements remain explicit rather than acquiring invented fields.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional

from .contract_types import (
    CONTRACT_SCHEMA_VERSION,
    CriterionSpec,
    ContractBundle,
    EnvelopeSpec,
    FieldProvenance,
    INCOMPLETE,
    READY,
    ResponseSpec,
    RequirementContract,
    RequirementObligation,
    TriggerSpec,
    UNSUPPORTED,
    VerificationIntent,
    normalise_req_id,
    source_digest,
)


_REQ_ID_RE = re.compile(r"\b(REQ[-_][A-Z]+[-_]\d+)\b", re.IGNORECASE)
_NUMBER = r"(\d+(?:\.\d+)?)"
_METRE = r"(?:m|metres?|meters?)"
_MPS = r"(?:m\s*/\s*s|mps|metres?\s+per\s+second|meters?\s+per\s+second)"


def _req_id(text: str) -> str:
    match = _REQ_ID_RE.search(text or "")
    return match.group(1).upper().replace("_", "-") if match else ""


def _first_number(patterns: Iterable[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1))
    return None


@dataclass(frozen=True)
class ObstacleAvoidanceContract:
    """Requirement semantics used to configure and judge one obstacle scenario.

    Distances are measured from the sensor/airframe toward the obstacle while the
    vehicle is approaching it, so a response is timely when
    ``response_onset_distance_m >= response_threshold_m``.
    """

    req_id: str
    requirement_text: str
    check: str = "obstacle_avoidance"
    detection_range_m: float | None = None
    response_threshold_m: float | None = None
    minimum_separation_m: float | None = None
    max_closing_speed_mps: float | None = None
    geometry: str | None = None
    response_semantics: str = "initiation"
    contract_ready: bool = False
    semantic_gaps: tuple[str, ...] = ()
    semantic_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["semantic_gaps"] = list(self.semantic_gaps)
        value["semantic_notes"] = list(self.semantic_notes)
        return value


def obstacle_avoidance_contract(requirement: str) -> ObstacleAvoidanceContract | None:
    """Parse an obstacle/collision requirement without inventing acceptance criteria."""
    text = " ".join((requirement or "").split())
    low = text.lower()
    if not any(word in low for word in ("obstacle", "collision", "avoidance")):
        return None

    detection_range = _first_number(
        (
            rf"{_NUMBER}\s*[- ]?{_METRE}\s+(?:sensor\s+)?(?:detection\s+)?range",
            rf"detect[^.;]{{0,100}}?within(?:\s+(?:a|the))?\s+{_NUMBER}\s*[- ]?{_METRE}",
            rf"detect[^.;]{{0,120}}?no\s+later\s+than\s+{_NUMBER}\s*[- ]?{_METRE}",
            rf"detection[^.;]{{0,100}}?no\s+later\s+than\s+{_NUMBER}\s*[- ]?{_METRE}",
            rf"(?:obstacle|collision\s+threat)[^.;]{{0,100}}?within\s+{_NUMBER}\s*[- ]?{_METRE}",
            rf"no\s+(?:farther|further)\s+than\s+{_NUMBER}\s*[- ]?{_METRE}",
        ),
        low,
    )

    response_threshold = _first_number(
        (
            rf"(?:initiat|begin|start|command)[^.;]{{0,160}}?before[^.;]{{0,100}}?{_NUMBER}\s*[- ]?{_METRE}",
            rf"(?:initiat|begin|start|command)[^.;]{{0,160}}?no\s+later\s+than[^.;]{{0,100}}?{_NUMBER}\s*[- ]?{_METRE}",
            rf"before[^.;]{{0,120}}?(?:separation|clearance|distance)[^.;]{{0,60}}?{_NUMBER}\s*[- ]?{_METRE}",
        ),
        low,
    )

    minimum_separation = _first_number(
        (
            rf"(?:maintain|keep|preserve|ensure)[^.;]{{0,120}}?(?:separation|clearance)[^.;]{{0,60}}?(?:at\s+least|no\s+less\s+than|minimum(?:\s+of)?)\s+{_NUMBER}\s*[- ]?{_METRE}",
            rf"(?:separation|clearance)[^.;]{{0,60}}?(?:remain|be)\s+(?:at\s+least|no\s+less\s+than)\s+{_NUMBER}\s*[- ]?{_METRE}",
            rf"(?:shall\s+not|must\s+not)[^.;]{{0,100}}?(?:separation|clearance)[^.;]{{0,50}}?(?:fall|drop)\s+below\s+{_NUMBER}\s*[- ]?{_METRE}",
        ),
        low,
    )

    closing_speed = _first_number(
        (
            rf"(?:closing|approach)\s+speed[^.;]{{0,80}}?(?:no\s+greater\s+than|not\s+exceed(?:ing)?|up\s+to|maximum(?:\s+of)?|<=?)\s*{_NUMBER}\s*{_MPS}",
            rf"(?:while|when)[^.;]{{0,100}}?approach(?:ing)?[^.;]{{0,80}}?{_NUMBER}\s*{_MPS}",
        ),
        low,
    )

    geometry = None
    if any(term in low for term in ("forward field of view", "forward sensor field", "directly ahead", "ahead of")):
        geometry = "forward_sensor_axis"

    semantics = "maintain_clearance" if minimum_separation is not None else "initiation"
    gaps: list[str] = []
    notes: list[str] = []
    if detection_range is None:
        gaps.append("missing measurable obstacle-detection range")
    if response_threshold is None and minimum_separation is None:
        gaps.append("missing measurable avoidance-response threshold")
    if closing_speed is None:
        gaps.append("missing maximum closing/approach speed for the verification envelope")
    if geometry is None:
        gaps.append("missing obstacle geometry or sensor field-of-view condition")
    if response_threshold is not None and minimum_separation is None:
        notes.append(
            "requirement asks when avoidance must start; it does not require the verifier "
            "to infer that the same threshold is a maintained minimum clearance"
        )

    return ObstacleAvoidanceContract(
        req_id=_req_id(text),
        requirement_text=text,
        detection_range_m=detection_range,
        response_threshold_m=response_threshold,
        minimum_separation_m=minimum_separation,
        max_closing_speed_mps=closing_speed,
        geometry=geometry,
        response_semantics=semantics,
        contract_ready=not gaps,
        semantic_gaps=tuple(gaps),
        semantic_notes=tuple(notes),
    )


def analyse_requirements(
    requirements: Iterable[str],
    *,
    generalized: bool = False,
) -> dict[str, Any]:
    """Return serialisable contracts and fail-closed semantic diagnostics."""
    requirements = tuple(requirements or ())
    contracts: list[dict[str, Any]] = []
    gaps: list[str] = []
    notes: list[str] = []
    for requirement in requirements or []:
        contract = obstacle_avoidance_contract(requirement)
        if contract is None:
            continue
        contracts.append(contract.to_dict())
        label = contract.req_id or "obstacle requirement"
        gaps.extend(f"{label}: {gap}" for gap in contract.semantic_gaps)
        notes.extend(f"{label}: {note}" for note in contract.semantic_notes)
    result = {
        "verification_contracts": contracts,
        "semantic_gaps": gaps,
        "semantic_notes": notes,
        "verification_ready": not gaps,
    }
    if generalized:
        bundle = build_contract_bundle(requirements)
        result["contract_bundle"] = bundle.to_dict()
        result["contract_summary"] = contract_summary(bundle)
    return result


# ---------------------------------------------------------------------------
# Option 2 typed contract builders
# ---------------------------------------------------------------------------

_CATEGORY_RE = re.compile(r"\bREQ[-_]([A-Z]+)[-_]\d+\b", re.IGNORECASE)
_SECONDS_RE = re.compile(
    rf"\bwithin\s+{_NUMBER}\s*(seconds?|secs?|s)\b", re.IGNORECASE
)


def _category(text: str) -> str:
    match = _CATEGORY_RE.search(text or "")
    return match.group(1).upper() if match else "UNKNOWN"


def _span(text: str, match: re.Match[str] | None) -> tuple[int, int] | None:
    return match.span() if match else None


def _prov(
    field: str,
    text: str,
    match: re.Match[str] | None,
    *,
    source: str = "requirement_text",
) -> FieldProvenance:
    evidence = match.group(0) if match else ""
    return FieldProvenance(
        field=field,
        source=source,
        evidence=evidence,
        source_span=_span(text, match),
    )


def _find(text: str, pattern: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.IGNORECASE | re.DOTALL)


_RESPONSE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("deploy_parachute", r"\b(?:deploy|release|activate)\b[^.;]{0,80}\b(?:parachute|ballistic recovery)\b"),
    ("prevent_arming", r"\b(?:prevent|inhibit|block|shall not transition)\b[^.;]{0,100}\b(?:arm|arming|armed|airborne)\b"),
    ("alert_gcs", r"\b(?:alert|notify|issue\s+(?:a\s+)?failure alert)\b[^.;]{0,100}\b(?:gcs|ground control)?\b"),
    ("lock_payload", r"\b(?:maintain|keep|place|lock|default)\b[^.;]{0,100}\bpayload\b[^.;]{0,80}\b(?:lock|locked|mechanically locked)\b|\b(?:lock|locked)\b[^.;]{0,80}\bpayload\b|\bpayload(?:[- ]release)?(?:\s+actuator)?\b[^.;]{0,100}\bdefault\b[^.;]{0,80}\b(?:lock|locked|mechanically locked)\b"),
    # Keep the action distinct from noun phrases such as "payload-release
    # actuator".  A release obligation needs an explicit modal/action phrase.
    ("release_payload", r"\b(?:shall|must|may)\b[^.;]{0,50}\brelease\b[^.;]{0,80}\bpayload\b|\brelease\s+the\s+payload\b|\bpayload\b[^.;]{0,50}\bshall\s+be\s+released\b|\bpayload\s+release\s+actuation\b[^.;]{0,80}\bshall\s+complete\b"),
    ("revise_waypoint_sequence", r"\b(?:incorporate|apply|update)\b[^.;]{0,120}\b(?:revised waypoint|waypoint sequence|flight plan)\b"),
    ("transmit_health_report", r"\b(?:transmit|send|issue)\b[^.;]{0,100}\b(?:health report|system health)\b"),
    ("return_to_base", r"\b(?:return-to-base|return to base|return-to-home|return to home|\brtl\b|\brtb\b)"),
    ("controlled_landing", r"\b(?:controlled descent|safe landing|autonomous safe landing|perform[^.;]{0,60}\bland(?:ing)?)\b"),
    ("maintain_controlled_flight", r"\bmaintain\b[^.;]{0,80}\bcontrolled flight\b"),
    ("avoid_obstacle", r"\b(?:avoidance manoeuvre|avoidance maneuver|avoid obstacles?|collision avoidance)\b"),
    ("perform_self_test", r"\b(?:execute|perform|run)\b[^.;]{0,100}\b(?:self-test|self test|self-check|self check)\b"),
)


def _response_matches(text: str) -> list[tuple[str, re.Match[str]]]:
    matches: list[tuple[str, re.Match[str]]] = []
    for concept, pattern in _RESPONSE_PATTERNS:
        match = _find(text, pattern)
        if match:
            matches.append((concept, match))
    # Avoid treating "prevent arming and alert GCS" as one response while still
    # retaining both obligations.  Exact concept de-duplication is deterministic.
    seen: set[str] = set()
    return [item for item in matches if not (item[0] in seen or seen.add(item[0]))]


def _trigger(text: str) -> tuple[TriggerSpec | None, FieldProvenance | None]:
    patterns: tuple[tuple[str, str, str, Optional[str]], ...] = (
        ("critical_propulsion_failure", r"\bcritical\b[^.;]{0,80}\bpropulsion\b[^.;]{0,50}\bfail(?:ure|ed)?\b|\bpropulsion\b[^.;]{0,60}\bcritical\b[^.;]{0,50}\bfail(?:ure|ed)?\b", "event", "propulsionCriticalFailure"),
        ("single_motor_failure", r"\b(?:single|one)\b[^.;]{0,50}\b(?:motor|propulsion motor)\b[^.;]{0,50}\b(?:failure|failed|inoperative|out)\b|\bfailure\b[^.;]{0,60}\b(?:single|one)\b[^.;]{0,40}\bmotor\b", "event", None),
        ("delivery_waypoint_proximity", r"\bwithin\s+\d+(?:\.\d+)?\s*(?:m|metres?|meters?)\b[^.;]{0,100}\bdelivery waypoint\b", "threshold", "waypointDistance"),
        ("delivery_coordinate_condition_satisfied", r"\bdelivery\s+coordinate\s+condition\b[^.;]{0,80}\b(?:is\s+)?satisfied\b", "event", None),
        ("delivery_abort_condition", r"\bdelivery[- ]abort condition\b|\babort condition\b", "boolean", "deliveryAbortConditionActive"),
        ("sensor_self_test_failure", r"\b(?:sensor|onboard sensor)\b[^.;]{0,100}\b(?:failure|fails?|failed|reports? a failure)\b[^.;]{0,80}\b(?:self-test|self test)\b|\b(?:self-test|self test)\b[^.;]{0,100}\b(?:failure|fails?|failed)\b", "boolean", "sensorSelfTestFailed"),
        ("gcs_link_absent", r"\b(?:gcs|ground control)[^.;]{0,80}\b(?:uplink|link|communication)?\b[^.;]{0,80}\b(?:absent|lost|loss|unavailable|disconnect(?:ed)?)\b", "duration", "commLossTime"),
        ("battery_state_of_charge", r"\bbattery\b[^.;]{0,100}\b(?:state[- ]of[- ]charge|charge|soc)\b|\bstate[- ]of[- ]charge\b", "threshold", "batterySoc"),
        ("valid_waypoint_modification_command", r"\bvalid\b[^.;]{0,80}\bwaypoint[- ]modification command\b|\bwaypoint[- ]modification command\b[^.;]{0,80}\bvalid\b", "event", None),
        ("automated_landing_completed", r"\b(?:completing|completion of|completed)\b[^.;]{0,80}\b(?:automated )?landing\b|\blanding[- ]completed\b", "event", None),
        ("contingency_condition", r"\bcontingency condition\b", "event", None),
        ("power_on", r"\bpower[- ]on\b|\bstartup\b|\bstart-up\b", "event", None),
    )
    for concept, pattern, kind, variable in patterns:
        match = _find(text, pattern)
        if not match:
            continue
        comparator = value = unit = None
        if concept == "battery_state_of_charge":
            number = _find(text, rf"(?:reaches?|below|less than|at most)\s*{_NUMBER}\s*%")
            if number:
                value = float(number.group(1))
                unit = "percent"
                phrase = number.group(0).lower()
                comparator = "<" if "below" in phrase or "less" in phrase else "<="
        elif concept == "gcs_link_absent":
            number = _find(text, rf"(?:more than|over|for)\s*{_NUMBER}\s*(?:seconds?|secs?|s)\b")
            if number:
                value = float(number.group(1))
                unit = "s"
                comparator = ">"
        elif concept == "delivery_waypoint_proximity":
            number = _find(text, rf"within\s+{_NUMBER}\s*(?:m|metres?|meters?)")
            if number:
                value = float(number.group(1))
                unit = "m"
                comparator = "<="
        qualifiers: tuple[str, ...] = ()
        if concept == "delivery_waypoint_proximity" and _find(
            text, r"\bno\s+delivery[- ]abort condition\s+is\s+active\b"
        ):
            qualifiers = ("delivery_abort_inactive",)
        spec = TriggerSpec(
            concept=concept,
            condition_kind=kind,
            variable=variable,
            comparator=comparator,
            value=value,
            unit=unit,
            qualifiers=qualifiers,
        )
        return spec, _prov("trigger", text, match)
    return None, None


def _deadline(text: str) -> tuple[CriterionSpec | None, FieldProvenance | None]:
    match = _SECONDS_RE.search(text)
    if not match:
        return None, None
    return (
        CriterionSpec(
            metric="response_latency",
            comparator="<=",
            value=float(match.group(1)),
            unit="s",
            timing_semantics="after_trigger",
        ),
        _prov("criterion", text, match),
    )


def _contract_base(text: str) -> tuple[str, str, tuple[FieldProvenance, ...]]:
    req_id = normalise_req_id(_req_id(text))
    match = _REQ_ID_RE.search(text)
    provenance = (_prov("source_text", text, match),) if match else ()
    return req_id, _category(text), provenance


def _typed_obstacle_contract(text: str) -> RequirementContract | None:
    legacy = obstacle_avoidance_contract(text)
    if legacy is None:
        return None
    req_id, category, provenance = _contract_base(text)
    trigger = TriggerSpec(
        concept="obstacle_detected",
        condition_kind="geometry_threshold",
        variable="obstacleDistance",
        comparator="<=",
        value=legacy.detection_range_m,
        unit="m",
    )
    criterion = None
    if legacy.minimum_separation_m is not None:
        criterion = CriterionSpec(
            metric="minimum_separation",
            comparator=">=",
            value=legacy.minimum_separation_m,
            unit="m",
            timing_semantics="throughout_avoidance",
        )
    elif legacy.response_threshold_m is not None:
        criterion = CriterionSpec(
            metric="response_onset_distance",
            comparator=">=",
            value=legacy.response_threshold_m,
            unit="m",
            timing_semantics="response_initiation",
        )
    obligation = RequirementObligation(
        obligation_id=f"{req_id}.O1",
        kind="obstacle_avoidance",
        subject="system",
        trigger=trigger,
        response=ResponseSpec("avoid_obstacle"),
        criterion=criterion,
        verification_intent=VerificationIntent(
            method="simulation",
            observation_concept=(
                "minimum_airframe_obstacle_separation"
                if legacy.minimum_separation_m is not None
                else "avoidance_response_onset_distance"
            ),
            # The existing deterministic trajectory/scenario executor can
            # evaluate the contract cheaply; Gazebo remains optional fidelity.
            preferred_tier="behavioral",
        ),
    )
    return RequirementContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        req_id=req_id,
        source_text=text,
        source_digest=source_digest(text),
        category=category,
        kinds=("obstacle_avoidance",),
        envelope=EnvelopeSpec(
            operating_states=("approaching_obstacle",),
            geometry=legacy.geometry,
            max_closing_speed_mps=legacy.max_closing_speed_mps,
        ),
        obligations=(obligation,),
        completeness=READY if legacy.contract_ready else INCOMPLETE,
        gaps=legacy.semantic_gaps,
        provenance=provenance,
    )


def _typed_behavior_contract(text: str) -> RequirementContract | None:
    responses = _response_matches(text)
    trigger, trigger_provenance = _trigger(text)
    criterion, criterion_provenance = _deadline(text)
    low = text.lower()
    compound_trigger_match = _find(
        text,
        r"\bcontingency\b[^.;]{0,80}\((?P<triggers>[^)]*(?:,|\bor\b)[^)]*)\)",
    )
    compound_trigger_items: tuple[str, ...] = ()
    if compound_trigger_match:
        body = compound_trigger_match.group("triggers").lower()
        detected = []
        if "gcs" in body or "link loss" in body:
            detected.append("gcs_link_absent")
        if "geofence" in body:
            detected.append("geofence_breach")
        if "battery" in body or "state-of-charge" in body:
            detected.append("battery_state_of_charge")
        compound_trigger_items = tuple(detected)
        if len(compound_trigger_items) > 1:
            trigger = TriggerSpec(
                concept="compound_contingency",
                condition_kind="compound_any",
                qualifiers=compound_trigger_items,
            )
            trigger_provenance = _prov(
                "trigger", text, compound_trigger_match
            )
    invariant = any(phrase in low for phrase in (
        "whenever", "shall not transition", "must not transition",
        "default to", "remain locked", "maintain the payload",
    ))
    if not responses:
        return None
    if trigger is None and not invariant:
        return None

    req_id, category, top_provenance = _contract_base(text)
    design_invariant = any(concept == "maintain_controlled_flight" for concept, _ in responses)
    timed_actuation = bool(criterion and "actuation" in low)
    kind = (
        "state_invariant" if invariant or design_invariant
        else (
            "timed_actuation" if timed_actuation
            else ("timed_response" if criterion else "triggered_response")
        )
    )
    gaps: list[str] = []
    if len(compound_trigger_items) > 1:
        gaps.append(
            "compound alternative triggers require separate reviewed obligations; "
            "the MVP builder must not select only one branch"
        )
        if _find(text, r"\bbattery\b[^.;]{0,100}\breturn threshold\b"):
            gaps.append(
                "battery return threshold is defined only by cross-reference, not "
                "quantified in this requirement"
            )
        if _find(text, r"\bdoes not require immediate landing\b"):
            gaps.append(
                "the no-immediate-landing qualifier does not enumerate the cases "
                "in which return-to-base remains applicable"
            )
    if kind in {"triggered_response", "timed_response", "timed_actuation"} and trigger is None:
        gaps.append("missing explicit trigger/event for required response")
    obligations: list[RequirementObligation] = []
    for index, (concept, response_match) in enumerate(responses, 1):
        local_provenance = [_prov("response", text, response_match)]
        if trigger_provenance:
            local_provenance.append(trigger_provenance)
        if criterion_provenance:
            local_provenance.append(criterion_provenance)
        observation = {
            "deploy_parachute": "parachute_deployed",
            "prevent_arming": "arming_rejected",
            "alert_gcs": "gcs_alert_observed",
            "lock_payload": "payload_locked",
            "release_payload": "payload_released",
            "revise_waypoint_sequence": "active_flight_plan_updated",
            "transmit_health_report": "health_report_received_by_gcs",
            "return_to_base": "vehicle_enters_rtl",
            "controlled_landing": "vehicle_enters_land",
            "perform_self_test": "self_test_completed",
            "maintain_controlled_flight": "controlled_flight_stability",
        }.get(concept, f"{concept}_observed")
        response_qualifiers: tuple[str, ...] = ()
        if concept == "return_to_base" and _find(
            text,
            r"\bunless\b[^.;]{0,100}\bhigher[- ]priority\b[^.;]{0,100}"
            r"\bsafety response\b[^.;]{0,80}\bin progress\b",
        ):
            response_qualifiers = (
                "unless_higher_priority_safety_response_in_progress",
            )
        elif concept == "deploy_parachute" and _find(
            text,
            r"\b(?:taking|takes?)\s+precedence\b[^.;]{0,100}"
            r"\b(?:other|all)\b[^.;]{0,80}\bsafety responses?\b",
        ):
            response_qualifiers = (
                "takes_precedence_over_other_safety_responses",
            )
        preferred_tier = "sitl" if concept in {
            "deploy_parachute", "prevent_arming", "alert_gcs", "lock_payload",
            "return_to_base", "controlled_landing",
        } else (
            "gazebo" if concept == "maintain_controlled_flight"
            else "behavioral"
        )
        obligations.append(RequirementObligation(
            obligation_id=f"{req_id}.O{index}",
            kind=kind,
            subject="system",
            trigger=trigger,
            response=ResponseSpec(
                concept=concept,
                qualifiers=response_qualifiers,
            ),
            criterion=criterion,
            verification_intent=VerificationIntent(
                method="test",
                observation_concept=observation,
                preferred_tier=preferred_tier,
            ),
            provenance=tuple(local_provenance),
        ))

    states: list[str] = []
    if "during flight" in low or "during airborne" in low:
        states.append("airborne")
    if (
        "prior to arming" in low
        or "before arming" in low
        or "before any arming" in low
    ):
        states.append("pre_arm")
    return RequirementContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        req_id=req_id,
        source_text=text,
        source_digest=source_digest(text),
        category=category,
        kinds=(kind,),
        envelope=EnvelopeSpec(operating_states=tuple(states)),
        obligations=tuple(obligations),
        completeness=READY if not gaps else INCOMPLETE,
        gaps=tuple(gaps),
        provenance=top_provenance,
    )


ContractBuilder = Callable[[str], RequirementContract | None]
_CONTRACT_BUILDERS: tuple[ContractBuilder, ...] = (
    _typed_obstacle_contract,
    _typed_behavior_contract,
)


def build_requirement_contract(requirement: str) -> RequirementContract:
    """Build one typed contract, returning an explicit UNSUPPORTED result."""
    # Preserve stakeholder text byte-for-byte for the immutable source digest.
    # Individual builders may normalize a private parsing view, never this value.
    text = requirement or ""
    for builder in _CONTRACT_BUILDERS:
        contract = builder(text)
        if contract is not None:
            return contract
    req_id, category, provenance = _contract_base(text)
    return RequirementContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        req_id=req_id,
        source_text=text,
        source_digest=source_digest(text),
        category=category,
        kinds=(),
        envelope=EnvelopeSpec(),
        obligations=(),
        completeness=UNSUPPORTED,
        gaps=("no implemented contract family matches this requirement",),
        provenance=provenance,
    )


def build_contract_bundle(requirements: Iterable[str]) -> ContractBundle:
    contracts = tuple(build_requirement_contract(req) for req in requirements or ())
    return ContractBundle(contracts=contracts)


def contract_summary(bundle: ContractBundle) -> dict[str, Any]:
    counts = {READY: 0, INCOMPLETE: 0, UNSUPPORTED: 0}
    for contract in bundle.contracts:
        counts[contract.completeness] = counts.get(contract.completeness, 0) + 1
    return {
        "total": len(bundle.contracts),
        "counts": counts,
        "supported": counts[READY] + counts[INCOMPLETE],
        "ready": counts[READY],
        "library_version": bundle.library_version,
    }
