"""Blade-element / momentum prop theory — the independent analytical model that Gazebo's
measured hover rotor RPM is cross-validated against (stage 5).

Static thrust:  T = Ct · ρ · n² · D⁴   (n in rev/s, D = rotor diameter).
So the rotor speed needed to hover (T = m·g/N per rotor) is  n = sqrt(T / (Ct·ρ·D⁴)).

Ct: thrust coefficient. ~0.115 is a realistic full-throttle figure for a T-Motor multirotor
prop (Tyto Robotics, "How to Calculate & Measure Propeller Thrust",
https://www.tytorobotics.com/blogs/articles/how-to-calculate-propeller-thrust). Ct varies with
RPM/prop; we report the cross-check across a Ct band so the agreement isn't hostage to one value.

IMPORTANT: use the DESIGN's rotor diameter (2·rotor_radius), not an arbitrary motor's prop —
the earlier "Gazebo 4204 vs 2900 RPM gap" was an error from using an 18" prop against a design
that specs 0.19 m radius (15" / 0.38 m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO = 1.225            # kg/m³, ISA sea level
G = 9.81
PROP_CT = 0.115        # cited typical T-Motor multirotor prop (Tyto Robotics)
PROP_CT_SOURCE = "https://www.tytorobotics.com/blogs/articles/how-to-calculate-propeller-thrust"


FOM = 0.62             # figure of merit (matches src/dse/physics_estimator)


# iris LiftDrag thrust constant κ (T = κ·area·ω²), back-calculated from the validated quad flight:
# hover thrust 13.49 N at ω=440 rad/s, area=0.00733 → κ = 13.49/(0.00733·440²) ≈ 9.506e-3.
IRIS_LIFTDRAG_KAPPA = 9.506e-3


def calibrated_area(diameter_m: float, ct: float = PROP_CT) -> float:
    """LiftDrag `area` that makes Gazebo's thrust-vs-ω curve equal a real prop's (Ct): set
    κ·area = Ct·ρ·D⁴/(4π²). Then the trimmed hover RPM equals the real prop's RPM."""
    kappa_real = ct * RHO * diameter_m ** 4 / (4.0 * math.pi ** 2)
    return kappa_real / IRIS_LIFTDRAG_KAPPA


def calibrated_max_rad_s(area: float, motor_max_thrust_n: float) -> float:
    """Max rotor speed (ArduPilotPlugin multiplier) so full throttle gives the real motor's max
    thrust: T_max = κ·area·ω_max² → ω_max = sqrt(T_max/(κ·area)). Sets a realistic T/W."""
    return math.sqrt(motor_max_thrust_n / (IRIS_LIFTDRAG_KAPPA * area))


def power_coefficient(ct: float = PROP_CT, fom: float = FOM) -> float:
    """Cp from Ct via the figure-of-merit identity FM = Ct^1.5 / (sqrt(2)·Cp).
    For Ct=0.115, FM=0.62 → Cp≈0.045 (typical multirotor prop)."""
    return ct ** 1.5 / (math.sqrt(2.0) * fom)


def mechanical_power_w(rpm: float, diameter_m: float, cp: float = None) -> float:
    """Shaft power of one rotor: P = Cp·ρ·n³·D⁵ (n in rev/s)."""
    cp = power_coefficient() if cp is None else cp
    n = rpm / 60.0
    return cp * RHO * n ** 3 * diameter_m ** 5


def prop_hover_rpm(thrust_per_rotor_n: float, diameter_m: float, ct: float = PROP_CT) -> float:
    """Rotor speed (RPM) a real prop of this diameter needs to make the given thrust (momentum
    theory, static hover)."""
    n_rev_s = math.sqrt(thrust_per_rotor_n / (ct * RHO * diameter_m ** 4))
    return n_rev_s * 60.0


@dataclass(frozen=True)
class RpmCrossCheck:
    gazebo_rpm: float
    diameter_m: float
    theory_rpm: float            # at PROP_CT
    theory_rpm_lo: float         # Ct band (high Ct → low RPM)
    theory_rpm_hi: float
    pct_diff: float              # gazebo vs theory(PROP_CT)
    within_ct_band: bool         # gazebo RPM falls inside the [lo, hi] Ct band
    ct_implied: float            # the Ct that would make theory match the measured RPM


def rpm_cross_check(gazebo_rpm: float, mass_kg: float, rotor_count: int, rotor_radius_m: float,
                    ct: float = PROP_CT, ct_band=(0.10, 0.13)) -> RpmCrossCheck:
    """Cross-validate Gazebo's measured hover RPM against prop theory at the DESIGN diameter.
    Independent of the simulator's aero — if they agree, the Gazebo rotor aero is physically
    consistent with a real prop of the design's size."""
    d = 2.0 * rotor_radius_m
    thrust = mass_kg * G / rotor_count
    theory = prop_hover_rpm(thrust, d, ct)
    rpm_hi = prop_hover_rpm(thrust, d, ct_band[0])    # low Ct → high RPM
    rpm_lo = prop_hover_rpm(thrust, d, ct_band[1])
    # Ct implied by the measured RPM: Ct = T/(ρ D⁴ (rpm/60)²)
    ct_implied = thrust / (RHO * d ** 4 * (gazebo_rpm / 60.0) ** 2)
    return RpmCrossCheck(
        gazebo_rpm=gazebo_rpm, diameter_m=d, theory_rpm=theory,
        theory_rpm_lo=rpm_lo, theory_rpm_hi=rpm_hi,
        pct_diff=(gazebo_rpm / theory - 1.0) * 100.0,
        within_ct_band=rpm_lo <= gazebo_rpm <= rpm_hi,
        ct_implied=ct_implied,
    )
