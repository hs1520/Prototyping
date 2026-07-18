"""Executable verification contracts derived from natural-language requirements.

The quantitative DSE extractor answers questions such as "how much endurance?".
Verification needs a different structure: stimulus, operational envelope, response
threshold, and acceptance oracle.  Keeping that structure in one auditable module
prevents a downstream test from silently strengthening or weakening a requirement.

Only obstacle avoidance is implemented today.  The public ``analyse_requirements``
shape is deliberately generic so other high-fidelity checks can add contract types
without returning to scattered keyword/number parsing in their runners.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


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


def analyse_requirements(requirements: Iterable[str]) -> dict[str, Any]:
    """Return serialisable contracts and fail-closed semantic diagnostics."""
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
    return {
        "verification_contracts": contracts,
        "semantic_gaps": gaps,
        "semantic_notes": notes,
        "verification_ready": not gaps,
    }
