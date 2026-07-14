"""Shared requirement verdict helpers for realization closure."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..dse.domain_objective import _emergent_for_family
from ..dse.physics_estimator import estimate
from ..dse.requirement_spec import (
    ALTITUDE,
    ENDURANCE,
    MASS_MTOW,
    PAYLOAD,
    RANGE,
    SPEED,
    ReqSpec,
    extract_requirements,
)
from .bottom_up import RealizedMetrics


# Datasheet realization can decide only quantities that emerge from component choice
# with ZERO assumed constants (the closure admission rule): hover/endurance from
# motor+prop bench curves and pack capacity, total mass from component masses, and
# payload-carrying capacity as the hover-throttle margin at the rated delivery
# payload (realized metrics are computed carrying it; throttle comes from bench-curve
# interpolation). Forward-flight speed/range need an assumed drag area, so they are
# evaluated in a separate lumped forward-flight tier; all unknown future families
# stay deferred. Attitude/behaviour clauses inside a payload requirement are NOT
# covered at this tier (flight-dynamics evidence).
CLOSURE_SCOPE_FAMILIES = {"time", "mass", "payload"}
FORWARD_FLIGHT_SCOPE_FAMILIES = {"speed", "range"}


@dataclass(frozen=True)
class RequirementVerdict:
    req_id: str
    family: str
    target: float
    estimator_value: float
    realized_value: float | None
    met: bool | None
    scope: str = "closure"
    fidelity: str | None = None
    note: str = ""


def closure_scope(verdict: RequirementVerdict) -> bool:
    return verdict.scope == "closure"


def forward_flight_scope(verdict: RequirementVerdict) -> bool:
    return verdict.scope == "forward_flight"


def _scope_for_family(fam: str) -> str:
    if fam in CLOSURE_SCOPE_FAMILIES:
        return "closure"
    if fam in FORWARD_FLIGHT_SCOPE_FAMILIES:
        return "forward_flight"
    return "deferred"


def _realized_metric_for_family(fam: str, metrics: RealizedMetrics, design) -> float:
    return {
        "speed": design.cruise_speed_mps,
        "time": metrics.endurance_min,
        "range": metrics.range_m,
        "mass": metrics.total_mass_kg,
    }.get(fam, 0.0)


def _family_for_spec(spec: ReqSpec) -> str:
    return {
        ENDURANCE: "time",
        MASS_MTOW: "mass",
        PAYLOAD: "payload",
        RANGE: "range",
        SPEED: "speed",
        ALTITUDE: "altitude",
    }.get(spec.quantity, spec.quantity)


def _actionable_spec(spec: ReqSpec) -> bool:
    if spec.quantity == ENDURANCE:
        return spec.operator == ">="
    if spec.quantity in {MASS_MTOW, PAYLOAD}:
        return spec.operator == "<="
    if spec.quantity in {RANGE, SPEED}:
        return spec.operator == ">="
    return False


def _structured_targets(requirements: List[str]) -> Tuple[ReqSpec, ...]:
    return tuple(
        s for s in extract_requirements(requirements)
        if s.quantity in {ENDURANCE, MASS_MTOW, PAYLOAD, RANGE, SPEED, ALTITUDE}
    )


# "hover throttle margin of at least 30 percent" → 0.30 (fraction)
_MARGIN_RE = re.compile(
    r"margin of at least\s+(\d+(?:\.\d+)?)\s*(?:percent|%)", re.IGNORECASE)


def _margin_target(requirements: List[str], req_id: str) -> Optional[float]:
    """Explicit hover-throttle margin stated in the payload requirement, if any."""
    rid = req_id.replace("_", "-")
    for line in requirements or []:
        if rid in str(line).replace("_", "-"):
            m = _MARGIN_RE.search(str(line))
            if m:
                return float(m.group(1)) / 100.0
    return None


def requirement_verdicts(design, metrics: RealizedMetrics,
                         requirements: List[str],
                         forward_flight: Dict[Tuple[str, str, float], object] | None = None
                         ) -> Tuple[RequirementVerdict, ...]:
    est = estimate(design)
    verdicts = []
    forward_flight = forward_flight or {}
    seen = set()
    for spec in _structured_targets(requirements):
        rid = spec.req_id.replace("_", "-")
        fam = _family_for_spec(spec)
        target = spec.value
        key = (rid, fam, target)
        if key in seen:
            continue
        seen.add(key)
        if not _actionable_spec(spec):
            verdicts.append(RequirementVerdict(
                req_id=rid,
                family=fam,
                target=target,
                estimator_value=0.0,
                realized_value=None,
                met=None,
                scope="deferred",
                note=f"{spec.quantity} {spec.operator} is not a datasheet/forward-flight capability check",
            ))
            continue
        note_override = ""
        if fam == "mass":
            estimator_value = est.get("total_mass_kg", 0.0)
            realized_value = metrics.total_mass_kg
            met = realized_value <= target
        elif fam == "payload":
            # Closure-admissible with zero assumed constants: realized metrics are
            # computed CARRYING the rated delivery payload (payload_split), and
            # hover throttle comes from bench-curve interpolation. The verdict is
            # the hover-throttle margin at that load; the estimator has no
            # throttle model, so estimator_value stays 0.0. Attitude/behaviour
            # clauses in the same requirement are NOT covered at this tier.
            margin_target = _margin_target(requirements, spec.req_id)
            realized_value = max(0.0, 1.0 - metrics.hover_throttle)
            estimator_value = 0.0
            if margin_target is not None:
                target = margin_target
                met = realized_value >= margin_target
                note_override = (
                    f"hover-throttle margin at the rated delivery payload "
                    f"({spec.value:g} kg aboard, bench-curve interpolation); "
                    "attitude/behaviour clauses are not covered at the datasheet tier")
            else:
                target = 0.0
                met = realized_value > 0.0
                note_override = (
                    f"no explicit margin stated; datasheet evidence = hovers within "
                    f"the bench curve carrying the rated {spec.value:g} kg")
        else:
            estimator_value = _emergent_for_family(fam, est)
            ff = forward_flight.get(key)
            if ff is not None:
                realized_value = ff.realized_value
                met = ff.met
            else:
                realized_value = _realized_metric_for_family(fam, metrics, design)
                met = realized_value >= target
        ff = forward_flight.get(key)
        verdicts.append(RequirementVerdict(
            req_id=rid,
            family=fam,
            target=target,
            estimator_value=estimator_value,
            realized_value=realized_value,
            met=met,
            scope=_scope_for_family(fam),
            fidelity=getattr(ff, "fidelity", None),
            note=note_override or getattr(ff, "note", ""),
        ))
    return tuple(verdicts)


def requirement_verdicts_met(design, metrics: RealizedMetrics, requirements: List[str]) -> bool:
    verdicts = requirement_verdicts(design, metrics, requirements)
    scoped = [v for v in verdicts if closure_scope(v)]
    return bool(scoped) and all(v.met for v in scoped)
