"""Datasheet operating-point recomputation for realized designs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..dse.domain_objective import DESIGN_DEFAULTS, evaluation_overrides
from ..dse.physics_estimator import AVIONICS_POWER_W, G, USABLE
from .catalog import BatteryPack, Frame, MotorPropCombo


@dataclass(frozen=True)
class RealizedDesign:
    combo: MotorPropCombo
    pack: BatteryPack
    frame: Frame
    rotor_count: int
    delivery_payload_kg: float
    equipment_mass_kg: float


@dataclass(frozen=True)
class RealizedMetrics:
    total_mass_kg: float
    hover_thrust_per_motor_g: float
    hover_current_per_motor_a: float
    hover_throttle: float
    total_hover_current_a: float
    twr_max: float
    endurance_min: float
    range_m: float
    cost: float
    notes: Tuple[str, ...] = ()


def payload_split(design, requirements: List[str]) -> Tuple[float, float]:
    """Split DSE added mass into delivery payload and onboard equipment mass."""
    delivery_kg = (
        evaluation_overrides(requirements).get("payload_mass_kg", 0.0)
        or DESIGN_DEFAULTS["payload_mass_kg"]
    )
    equipment_kg = max(0.0, float(design.payload_mass_kg) - delivery_kg)
    return float(delivery_kg), equipment_kg


def realize_from_design(design, requirements: List[str],
                        combo: MotorPropCombo, pack: BatteryPack,
                        frame: Frame) -> RealizedDesign:
    delivery_kg, equipment_kg = payload_split(design, requirements)
    return RealizedDesign(
        combo=combo,
        pack=pack,
        frame=frame,
        rotor_count=frame.arms,
        delivery_payload_kg=delivery_kg,
        equipment_mass_kg=equipment_kg,
    )


def realized_total_mass_kg(rd: RealizedDesign) -> float:
    """All-up mass from real component masses only."""
    return (
        rd.frame.mass_g / 1000.0
        + rd.rotor_count * (rd.combo.motor_mass_g + rd.combo.prop_mass_g) / 1000.0
        + rd.pack.mass_g / 1000.0
        + rd.delivery_payload_kg
        + rd.equipment_mass_kg
    )


def realized_metrics(rd: RealizedDesign, cruise_speed_mps: float = 0.0,
                     cost_axis: str = "mass") -> RealizedMetrics:
    total_mass = realized_total_mass_kg(rd)
    thrust_g = (total_mass * 1000.0) / rd.rotor_count
    current_a, _power_w = rd.combo.interp_at_thrust(thrust_g)
    throttle = rd.combo.throttle_at_thrust(thrust_g)
    avionics_a = AVIONICS_POWER_W / rd.combo.voltage_v
    total_a = current_a * rd.rotor_count + avionics_a
    usable_ah = (rd.pack.capacity_mah / 1000.0) * USABLE
    endurance_min = (usable_ah / total_a) * 60.0 if total_a > 0 else 0.0
    twr_max = (rd.combo.max_thrust_g() / 1000.0 * G * rd.rotor_count) / (total_mass * G)
    notes = []
    cost = total_mass
    if cost_axis == "price":
        prices = (rd.combo.price_usd, rd.pack.price_usd, rd.frame.price_usd)
        if all(p is not None for p in prices):
            cost = float(sum(p for p in prices if p is not None))
        else:
            notes.append("cost_axis=price requested but a component price is missing; used mass proxy")
    elif cost_axis != "mass":
        notes.append(f"unknown cost_axis={cost_axis!r}; used mass proxy")
    return RealizedMetrics(
        total_mass_kg=total_mass,
        hover_thrust_per_motor_g=thrust_g,
        hover_current_per_motor_a=current_a,
        hover_throttle=throttle,
        total_hover_current_a=total_a,
        twr_max=twr_max,
        endurance_min=endurance_min,
        range_m=endurance_min * 60.0 * cruise_speed_mps,
        cost=cost,
        notes=tuple(notes),
    )

