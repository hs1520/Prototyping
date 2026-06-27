"""Forward-flight power → range (stage 6) — momentum-theory model of the regime where Gazebo
genuinely beats "hover thrust = weight".

In hover, power is just induced power (T·v_h) and the datasheet nails it. In forward flight three
terms compete and give a U-shaped power-vs-speed curve:

  P(V) = T·v_i(V)            induced  — DROPS with speed (Glauert): v_i = v_h² / sqrt(V²+v_i²)
       + 0.5·ρ·V³·f          parasite — GROWS with speed (body drag, f = equivalent flat-plate area)
       + P_profile           blade profile drag — ~constant

The minimum gives the best-endurance speed; the tangent-from-origin to P(V) gives the best-range
speed (max distance per energy). Range = usable_energy / P(V*) · V*.

Honest scope/assumptions (this is a LUMPED model, the thing Gazebo forward flight is validated
against — not ground truth):
  - f (equivalent flat-plate drag area): assumed 0.05 m² for a small quad — the per-design value
    is exactly what a Gazebo forward-flight measurement would pin down.
  - P_profile taken from the hover figure-of-merit loss (FOM=0.62), consistent with the estimator.
  - thrust tilts to balance drag: T = sqrt((m·g)² + D²).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO = 1.225
G = 9.81
FOM = 0.62                 # rotor figure of merit (matches src/dse/physics_estimator)
DEFAULT_DRAG_AREA = 0.05   # m², equivalent flat-plate area f for a small quad (assumption)


def _induced_velocity_forward(V: float, v_h: float) -> float:
    """Glauert forward-flight induced velocity (fixed-point solve of v_i = v_h²/sqrt(V²+v_i²))."""
    v_i = v_h
    for _ in range(100):
        new = v_h * v_h / math.sqrt(V * V + v_i * v_i)
        if abs(new - v_i) < 1e-6:
            break
        v_i = new
    return v_i


@dataclass(frozen=True)
class ForwardPower:
    speed_mps: float
    power_w: float
    induced_w: float
    parasite_w: float
    profile_w: float


def power_at_speed(mass_kg: float, rotor_count: int, rotor_radius_m: float, speed_mps: float,
                   drag_area: float = DEFAULT_DRAG_AREA) -> ForwardPower:
    A = rotor_count * math.pi * rotor_radius_m ** 2          # total disc area
    drag = 0.5 * RHO * speed_mps ** 2 * drag_area
    T = math.hypot(mass_kg * G, drag)                        # thrust tilts to balance drag
    v_h = math.sqrt(T / (2.0 * RHO * A))                     # hover induced velocity
    v_i = _induced_velocity_forward(speed_mps, v_h)
    induced = T * v_i
    parasite = 0.5 * RHO * speed_mps ** 3 * drag_area
    profile = (1.0 / FOM - 1.0) * (mass_kg * G) * math.sqrt(mass_kg * G / (2.0 * RHO * A))
    return ForwardPower(speed_mps, induced + parasite + profile, induced, parasite, profile)


def effective_drag_area_from_power(power_w: float, speed_mps: float, mass_kg: float,
                                   rotor_count: int, rotor_radius_m: float) -> float:
    """Back out the airframe's equivalent flat-plate drag area f from a MEASURED forward-flight
    power (e.g. from Gazebo): f = (P - P_induced - P_profile) / (0.5·ρ·V³). This is the parameter
    the lumped model assumes — a forward-flight simulation/flight pins it down for the real body."""
    base = power_at_speed(mass_kg, rotor_count, rotor_radius_m, speed_mps, drag_area=0.0)
    non_parasite = base.induced_w + base.profile_w
    return max(0.0, (power_w - non_parasite) / (0.5 * RHO * speed_mps ** 3))


@dataclass(frozen=True)
class RangeResult:
    hover_power_w: float
    best_endurance_speed_mps: float
    min_power_w: float
    best_range_speed_mps: float
    range_km: float
    endurance_at_range_speed_min: float


def range_estimate(mass_kg: float, rotor_count: int, rotor_radius_m: float,
                   battery_capacity_mah: float, battery_cells: int,
                   drag_area: float = DEFAULT_DRAG_AREA,
                   usable: float = 0.8, cell_v: float = 3.7) -> RangeResult:
    """Best-range / best-endurance speeds and range from the forward-flight power curve."""
    usable_wh = (battery_capacity_mah / 1000.0) * battery_cells * cell_v * usable
    speeds = [0.5 * i for i in range(1, 60)]                 # 0.5 .. 29.5 m/s
    curve = [power_at_speed(mass_kg, rotor_count, rotor_radius_m, v, drag_area) for v in speeds]
    hover = power_at_speed(mass_kg, rotor_count, rotor_radius_m, 0.01, drag_area).power_w
    best_end = min(curve, key=lambda p: p.power_w)           # min power → max endurance
    best_rng = max(curve, key=lambda p: p.speed_mps / p.power_w)   # max V/P → max range
    range_m = (usable_wh * 3600.0 / best_rng.power_w) * best_rng.speed_mps   # J/W·(m/s)=m
    return RangeResult(
        hover_power_w=hover,
        best_endurance_speed_mps=best_end.speed_mps, min_power_w=best_end.power_w,
        best_range_speed_mps=best_rng.speed_mps, range_km=range_m / 1000.0,
        endurance_at_range_speed_min=usable_wh / best_rng.power_w * 60.0,
    )
