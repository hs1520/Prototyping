"""Forward-flight realization checks for speed/range requirements.

This is a distinct fidelity tier from datasheet hover closure. It uses a lumped
momentum-theory model plus an assumed equivalent drag area, so results are
reported separately and never decide the datasheet CLOSED/INFEASIBLE verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..dse.requirement_spec import RANGE, SPEED, extract_requirements
from .bottom_up import RealizedDesign, realized_total_mass_kg
from .forward_flight import DEFAULT_DRAG_AREA, G, RHO, power_at_speed, range_estimate, speed_grid

FIDELITY = "lumped_forward_flight"
NOTE = (
    "lumped forward-flight momentum model; equivalent drag area assumed "
    f"{DEFAULT_DRAG_AREA:g} m^2; max speed is capped by tilted thrust and "
    "datasheet electrical power (published bench power plus pack C-rate); "
    "continuous thermal limits remain unavailable; Gazebo/SITL can calibrate this fidelity tier"
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
        (spec.req_id.replace("_", "-"), spec.quantity, spec.value)
        for spec in extract_requirements(requirements)
        if spec.quantity in {SPEED, RANGE} and spec.operator == ">="
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
        # Li-ion packs are 3.6 V/cell nominal — the 3.7 default silently rated
        # them as LiPo, inflating usable energy (and range verdicts) by ~2.8%.
        cell_v=rd.pack.operating_voltage_v() / rd.pack.cells,
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
    """Largest grid speed fitting both thrust and datasheet electrical-power limits."""
    mass = realized_total_mass_kg(rd) if mass_kg is None else mass_kg
    radius = rd.combo.prop_diameter_in * 0.0254 / 2.0 if rotor_radius_m is None else rotor_radius_m
    power_limit_w = _forward_power_limit_w(rd)
    feasible = []
    for speed in speed_grid():
        per_motor_thrust_g = _required_per_motor_thrust_g(mass, rd.rotor_count, speed, drag_area)
        required_power_w = power_at_speed(mass, rd.rotor_count, radius, speed, drag_area).power_w
        if per_motor_thrust_g <= rd.combo.max_thrust_g() and required_power_w <= power_limit_w:
            feasible.append(speed)
    return max(feasible) if feasible else 0.0


def _required_per_motor_thrust_g(
    mass_kg: float,
    rotor_count: int,
    speed_mps: float,
    drag_area: float,
) -> float:
    drag = 0.5 * RHO * speed_mps ** 2 * drag_area
    thrust_n = math.hypot(mass_kg * G, drag)
    return thrust_n / rotor_count / G * 1000.0


def _forward_power_limit_w(rd: RealizedDesign) -> float:
    """Electrical power ceiling from published bench rows and pack C-rate.

    The catalog does not carry thermal continuous-power ratings. We therefore use
    the maximum official bench-table electrical power as a motor-side upper bound
    and the pack C-rating as a battery-side upper bound. This remains a lumped
    fidelity check, but avoids claiming speeds that exceed the real component
    power envelope.
    """
    motor_limit_w = max(p.power_w for p in rd.combo.curve) * rd.rotor_count
    pack_voltage_v = rd.pack.operating_voltage_v()
    pack_limit_w = (rd.pack.capacity_mah / 1000.0) * rd.pack.c_rating * pack_voltage_v
    return min(motor_limit_w, pack_limit_w)
