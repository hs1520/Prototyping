"""Datasheet-grounded hover endurance, an independent cross-check of the analytic estimator.

Pipeline: design all-up mass -> hover thrust per motor (m*g / N) -> per-motor hover current from
the motor+prop curve (component_data) -> endurance from usable battery capacity, plus a constant
avionics hotel load. It replaces the estimator's lumped FOM/ETA_DRIVE with a measured operating
point, so a divergence between the two points at the lumped efficiency. Scope: static-bench
current (no ground effect, forward flight or pack sag), so expect ~10-20% off real flight; Gazebo
covers the trimmed-hover dynamics instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.dse.physics_estimator import AVIONICS_POWER_W, G, USABLE

from .component_data import MN5008_KV340_18x61, MotorProp


@dataclass(frozen=True)
class DatasheetEndurance:
    motor: str
    total_mass_kg: float
    hover_thrust_per_motor_g: float
    hover_current_per_motor_a: float
    total_hover_current_a: float
    twr_max: float
    endurance_min: float
    source_url: str


def datasheet_endurance(total_mass_kg: float, rotor_count: int,
                        battery_capacity_mah: float,
                        motor: MotorProp = MN5008_KV340_18x61) -> DatasheetEndurance:
    """Hover endurance from the motor+prop curve."""
    thrust_per_motor_g = (total_mass_kg * 1000.0) / rotor_count
    cur_a, _pwr_w = motor.interp_at_thrust(thrust_per_motor_g)
    avionics_a = AVIONICS_POWER_W / motor.voltage_v
    total_a = cur_a * rotor_count + avionics_a
    usable_ah = (battery_capacity_mah / 1000.0) * USABLE
    endurance_min = (usable_ah / total_a) * 60.0
    twr_max = (motor.max_thrust_g() / 1000.0 * G * rotor_count) / (total_mass_kg * G)
    return DatasheetEndurance(
        motor=motor.name,
        total_mass_kg=total_mass_kg,
        hover_thrust_per_motor_g=thrust_per_motor_g,
        hover_current_per_motor_a=cur_a,
        total_hover_current_a=total_a,
        twr_max=twr_max,
        endurance_min=endurance_min,
        source_url=motor.source_url,
    )
