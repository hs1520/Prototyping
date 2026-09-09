"""Datasheet operating-point recomputation for realized designs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..dse.domain_objective import DESIGN_DEFAULTS, evaluation_overrides
from ..dse.physics_estimator import AVIONICS_POWER_W, G, USABLE
from .catalog import (
    BatteryPack,
    Frame,
    IntegrationBundle,
    MotorPropCombo,
    NO_INTEGRATION_BUNDLE,
)


@dataclass(frozen=True)
class RealizedDesign:
    combo: MotorPropCombo
    pack: BatteryPack
    frame: Frame
    rotor_count: int
    delivery_payload_kg: float
    equipment_mass_kg: float
    integration_bundle: IntegrationBundle = NO_INTEGRATION_BUNDLE


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
    pack_voltage_v: float
    voltage_ratio: float
    derated_max_thrust_per_motor_g: float
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
                        frame: Frame,
                        integration_bundle: IntegrationBundle = NO_INTEGRATION_BUNDLE,
                        ) -> RealizedDesign:
    delivery_kg, equipment_kg = payload_split(design, requirements)
    return RealizedDesign(
        combo=combo,
        pack=pack,
        frame=frame,
        rotor_count=frame.arms,
        delivery_payload_kg=delivery_kg,
        equipment_mass_kg=equipment_kg,
        integration_bundle=integration_bundle,
    )


def realized_total_mass_kg(rd: RealizedDesign) -> float:
    """All-up mass from real component masses only."""
    return (
        rd.frame.mass_g / 1000.0
        + rd.rotor_count * (rd.combo.motor_mass_g + rd.combo.prop_mass_g) / 1000.0
        + rd.pack.mass_g / 1000.0
        + rd.delivery_payload_kg
        + rd.equipment_mass_kg
        + rd.integration_bundle.mass_g / 1000.0
    )


def realized_metrics(rd: RealizedDesign, cruise_speed_mps: float = 0.0,
                     cost_axis: str = "mass") -> RealizedMetrics:
    total_mass = realized_total_mass_kg(rd)
    thrust_g = (total_mass * 1000.0) / rd.rotor_count
    pack_voltage_v = rd.pack.operating_voltage_v()
    if pack_voltage_v <= 0 or rd.combo.voltage_v <= 0:
        raise ValueError("pack and motor-curve voltages must be positive")

    # Same S-count does not imply the same operating voltage. A curve is not
    # scaled above its published test point. At lower pack voltage, maximum static
    # thrust is derated with V^2; the same hover point needs proportionally more
    # throttle, and its published electrical power converts to pack current at the
    # actual voltage.
    voltage_ratio = pack_voltage_v / rd.combo.voltage_v
    conservative_ratio = min(1.0, voltage_ratio)
    derated_max_thrust_g = rd.combo.max_thrust_g() * conservative_ratio ** 2
    if thrust_g > derated_max_thrust_g:
        raise ValueError(
            f"required {thrust_g:.0f} g/motor exceeds voltage-derated "
            f"{rd.combo.name} max {derated_max_thrust_g:.0f} g at "
            f"{pack_voltage_v:.1f}V"
        )
    _curve_current_a, power_w = rd.combo.interp_at_thrust(thrust_g)
    current_a = power_w / pack_voltage_v
    throttle = rd.combo.throttle_at_thrust(thrust_g) / conservative_ratio
    avionics_a = AVIONICS_POWER_W / pack_voltage_v
    total_a = current_a * rd.rotor_count + avionics_a
    usable_ah = (rd.pack.capacity_mah / 1000.0) * USABLE
    endurance_min = (usable_ah / total_a) * 60.0 if total_a > 0 else 0.0
    twr_max = (derated_max_thrust_g / 1000.0 * G * rd.rotor_count) / (total_mass * G)
    notes = []
    if abs(voltage_ratio - 1.0) > 1e-9:
        notes.append(
            f"motor curve {rd.combo.voltage_v:.1f}V evaluated at pack nominal "
            f"{pack_voltage_v:.1f}V; max thrust and hover operating point derated"
        )
    cost = total_mass
    if cost_axis == "price":
        prices = (
            rd.combo.price_usd,
            rd.pack.price_usd,
            rd.frame.price_usd,
            rd.integration_bundle.price_usd,
        )
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
        pack_voltage_v=pack_voltage_v,
        voltage_ratio=voltage_ratio,
        derated_max_thrust_per_motor_g=derated_max_thrust_g,
        notes=tuple(notes),
    )
