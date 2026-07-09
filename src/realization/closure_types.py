"""Shared requirement verdict helpers for realization closure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..dse.domain_objective import _emergent_for_family
from ..dse.physics_estimator import estimate
from ..dse.requirement_spec import (
    ALTITUDE,
    ENDURANCE,
    MASS_MTOW,
    RANGE,
    SPEED,
    ReqSpec,
    extract_requirements,
)
from .bottom_up import RealizedMetrics


# Datasheet realization can decide only quantities that emerge from component choice:
# hover/endurance from motor+prop bench curves and pack capacity, and total mass from
# component masses. Forward-flight speed/range are L1/SITL/Gazebo concerns; static
# hover bench data cannot honestly close them. Speed/range are evaluated in a
# separate lumped forward-flight tier; all unknown future families stay deferred.
CLOSURE_SCOPE_FAMILIES = {"time", "mass"}
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
        RANGE: "range",
        SPEED: "speed",
        ALTITUDE: "altitude",
    }.get(spec.quantity, spec.quantity)


def _actionable_spec(spec: ReqSpec) -> bool:
    if spec.quantity == ENDURANCE:
        return spec.operator == ">="
    if spec.quantity == MASS_MTOW:
        return spec.operator == "<="
    if spec.quantity in {RANGE, SPEED}:
        return spec.operator == ">="
    return False


def _structured_targets(requirements: List[str]) -> Tuple[ReqSpec, ...]:
    return tuple(
        s for s in extract_requirements(requirements)
        if s.quantity in {ENDURANCE, MASS_MTOW, RANGE, SPEED, ALTITUDE}
    )


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
        if fam == "mass":
            estimator_value = est.get("total_mass_kg", 0.0)
            realized_value = metrics.total_mass_kg
            met = realized_value <= target
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
            note=getattr(ff, "note", ""),
        ))
    return tuple(verdicts)


def requirement_verdicts_met(design, metrics: RealizedMetrics, requirements: List[str]) -> bool:
    verdicts = requirement_verdicts(design, metrics, requirements)
    scoped = [v for v in verdicts if closure_scope(v)]
    return bool(scoped) and all(v.met for v in scoped)
