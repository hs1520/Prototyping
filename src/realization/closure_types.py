"""Shared requirement verdict helpers for realization closure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..dse.domain_objective import _emergent_for_family, mass_limit, requirement_targets
from ..dse.physics_estimator import estimate
from .bottom_up import RealizedMetrics


# Datasheet realization can decide only quantities that emerge from component choice:
# hover/endurance from motor+prop bench curves and pack capacity, and total mass from
# component masses. Forward-flight speed/range are L1/SITL/Gazebo concerns; static
# hover bench data cannot honestly close them, so all other families are deferred.
CLOSURE_SCOPE_FAMILIES = {"time", "mass"}


@dataclass(frozen=True)
class RequirementVerdict:
    req_id: str
    family: str
    target: float
    estimator_value: float
    realized_value: float
    met: bool
    scope: str = "closure"


def closure_scope(verdict: RequirementVerdict) -> bool:
    return verdict.scope == "closure"


def _realized_metric_for_family(fam: str, metrics: RealizedMetrics, design) -> float:
    return {
        "speed": design.cruise_speed_mps,
        "time": metrics.endurance_min,
        "range": metrics.range_m,
        "mass": metrics.total_mass_kg,
    }.get(fam, 0.0)


def requirement_verdicts(design, metrics: RealizedMetrics,
                         requirements: List[str]) -> Tuple[RequirementVerdict, ...]:
    est = estimate(design)
    verdicts = []
    mtow_id, mtow_target = mass_limit(requirements)
    mtow_id = mtow_id.replace("_", "-") if mtow_id else None
    seen = set()
    for rid, targets in requirement_targets(requirements).items():
        for fam, target in targets:
            if fam == "mass" and rid != mtow_id:
                continue
            if fam == "mass":
                estimator_value = est.get("total_mass_kg", 0.0)
                realized_value = metrics.total_mass_kg
                met = realized_value <= target
            else:
                estimator_value = _emergent_for_family(fam, est)
                realized_value = _realized_metric_for_family(fam, metrics, design)
                met = realized_value >= target
            key = (rid, fam, target)
            if key in seen:
                continue
            seen.add(key)
            verdicts.append(RequirementVerdict(
                req_id=rid,
                family=fam,
                target=target,
                estimator_value=estimator_value,
                realized_value=realized_value,
                met=met,
                scope="closure" if fam in CLOSURE_SCOPE_FAMILIES else "deferred",
            ))
    if mtow_id and (mtow_id, "mass", mtow_target) not in seen:
        verdicts.append(RequirementVerdict(
            req_id=mtow_id,
            family="mass",
            target=mtow_target,
            estimator_value=est.get("total_mass_kg", 0.0),
            realized_value=metrics.total_mass_kg,
            met=metrics.total_mass_kg <= mtow_target,
            scope="closure",
        ))
    return tuple(verdicts)


def requirement_verdicts_met(design, metrics: RealizedMetrics, requirements: List[str]) -> bool:
    verdicts = requirement_verdicts(design, metrics, requirements)
    scoped = [v for v in verdicts if closure_scope(v)]
    return bool(scoped) and all(v.met for v in scoped)
