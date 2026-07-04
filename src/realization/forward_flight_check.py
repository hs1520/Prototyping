"""Forward-flight realization checks for speed/range requirements.

This is a distinct fidelity tier from datasheet hover closure. It uses a lumped
momentum-theory model plus an assumed equivalent drag area, so results are
reported separately and never decide the datasheet CLOSED/INFEASIBLE verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..dse.domain_objective import requirement_targets
from .bottom_up import RealizedDesign, realized_total_mass_kg
from .forward_flight import DEFAULT_DRAG_AREA, G, RHO, range_estimate

FORWARD_FLIGHT_SCOPE_FAMILIES = {"speed", "range"}
FIDELITY = "lumped_forward_flight"
NOTE = (
    "lumped forward-flight momentum model; equivalent drag area assumed "
    f"{DEFAULT_DRAG_AREA:g} m^2; Gazebo/SITL can calibrate this fidelity tier"
)


@dataclass(frozen=True)
class ForwardFlightRequirement:
    req_id: str
    family: str
    target: float
    realized_value: float
    met: bool
    fidelity: str = FIDELITY
    note: str = NOTE


def forward_flight_verdicts(
    rd: RealizedDesign,
    requirements: List[str],
) -> Dict[Tuple[str, str, float], ForwardFlightRequirement]:
    """Return speed/range checks keyed by ``(req_id, family, target)``."""
    relevant = [
        (rid, fam, target)
        for rid, targets in requirement_targets(requirements).items()
        for fam, target in targets
        if fam in FORWARD_FLIGHT_SCOPE_FAMILIES
    ]
    if not relevant:
        return {}

    mass_kg = realized_total_mass_kg(rd)
    rotor_radius_m = rd.combo.prop_diameter_in * 0.0254 / 2.0
    range_result = range_estimate(
        mass_kg,
        rd.rotor_count,
        rotor_radius_m,
        rd.pack.capacity_mah,
        rd.pack.cells,
    )
    realized_range_m = range_result.range_km * 1000.0
    max_speed = max_sustainable_speed_mps(rd, mass_kg, rotor_radius_m)
    out: Dict[Tuple[str, str, float], ForwardFlightRequirement] = {}
    for rid, fam, target in relevant:
        if fam == "range":
            realized = realized_range_m
        else:
            realized = max_speed
        out[(rid, fam, target)] = ForwardFlightRequirement(
            req_id=rid,
            family=fam,
            target=target,
            realized_value=realized,
            met=realized >= target,
        )
    return out


def max_sustainable_speed_mps(
    rd: RealizedDesign,
    mass_kg: float | None = None,
    rotor_radius_m: float | None = None,
    drag_area: float = DEFAULT_DRAG_AREA,
) -> float:
    """Largest grid speed whose required tilted thrust fits the combo max thrust."""
    mass = realized_total_mass_kg(rd) if mass_kg is None else mass_kg
    radius = rd.combo.prop_diameter_in * 0.0254 / 2.0 if rotor_radius_m is None else rotor_radius_m
    _ = radius  # radius is part of the public calculation context; thrust limit is drag/mass based.
    feasible = []
    for speed in (0.5 * i for i in range(1, 60)):
        drag = 0.5 * RHO * speed ** 2 * drag_area
        thrust_n = math.hypot(mass * G, drag)
        per_motor_thrust_g = thrust_n / rd.rotor_count / G * 1000.0
        if per_motor_thrust_g <= rd.combo.max_thrust_g():
            feasible.append(speed)
    return max(feasible) if feasible else 0.0
