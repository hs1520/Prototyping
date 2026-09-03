"""Blade-element / momentum prop theory - the analytical model Gazebo's measured hover rotor RPM is
cross-validated against (stage 5).

Static thrust T = Ct . ρ . n² . D⁴ (n in rev/s, D = rotor diameter), so the hover rotor speed
(T = m*g/N per rotor) is n = sqrt(T / (Ct*ρ*D⁴)). Ct ~0.115 is a full-throttle figure for a
T-Motor multirotor prop (Tyto Robotics, "How to Calculate & Measure Propeller Thrust",
https://www.tytorobotics.com/blogs/articles/how-to-calculate-propeller-thrust); it varies with
RPM and prop, so the cross-check is reported across a Ct band. Use the design's rotor diameter
(2*rotor_radius): the earlier "Gazebo 4204 vs 2900 RPM gap" came from an 18" prop against a
design specifying 0.19 m radius (15" / 0.38 m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO = 1.225
G = 9.81
PROP_CT = 0.115
PROP_CT_SOURCE = "https://www.tytorobotics.com/blogs/articles/how-to-calculate-propeller-thrust"


FOM = 0.62


IRIS_LIFTDRAG_KAPPA = 9.506e-3


def calibrated_area(diameter_m: float, ct: float = PROP_CT) -> float:
    """LiftDrag `area` that makes Gazebo's thrust-vs-ω curve equal a real prop's (Ct): set κ*area = Ct*ρ*D⁴/(4π²)."""
    kappa_real = ct * RHO * diameter_m ** 4 / (4.0 * math.pi ** 2)
    return kappa_real / IRIS_LIFTDRAG_KAPPA


def calibrated_max_rad_s(area: float, motor_max_thrust_n: float) -> float:
    """Max rotor speed (ArduPilotPlugin multiplier) so full throttle gives the real motor's max thrust:
    T_max = κ*area*ω_max² -> ω_max = sqrt(T_max/(κ*area)).
    """
    return math.sqrt(motor_max_thrust_n / (IRIS_LIFTDRAG_KAPPA * area))


def power_coefficient(ct: float = PROP_CT, fom: float = FOM) -> float:
    """Cp from Ct via the figure-of-merit identity FM = Ct^1.5 / (sqrt(2).Cp)."""
    return ct ** 1.5 / (math.sqrt(2.0) * fom)


def mechanical_power_w(rpm: float, diameter_m: float, cp: float = None) -> float:
    """Shaft power of one rotor: P = Cp*ρ*n³*D⁵ (n in rev/s)."""
    cp = power_coefficient() if cp is None else cp
    n = rpm / 60.0
    return cp * RHO * n ** 3 * diameter_m ** 5


def prop_hover_rpm(thrust_per_rotor_n: float, diameter_m: float, ct: float = PROP_CT) -> float:
    """Rotor speed (RPM) a prop of this diameter needs for the given thrust (momentum theory, static hover).
    """
    n_rev_s = math.sqrt(thrust_per_rotor_n / (ct * RHO * diameter_m ** 4))
    return n_rev_s * 60.0


@dataclass(frozen=True)
class RpmCrossCheck:
    gazebo_rpm: float
    diameter_m: float
    theory_rpm: float
    theory_rpm_lo: float
    theory_rpm_hi: float
    pct_diff: float
    within_ct_band: bool
    ct_implied: float


def rpm_cross_check(gazebo_rpm: float, mass_kg: float, rotor_count: int, rotor_radius_m: float,
                    ct: float = PROP_CT, ct_band=(0.10, 0.13)) -> RpmCrossCheck:
    """Cross-validate Gazebo's measured hover RPM against prop theory at the design diameter.

    Independent of the simulator's aero: agreement means the Gazebo rotor aero is consistent with a
    prop of the design's size.
    """
    d = 2.0 * rotor_radius_m
    thrust = mass_kg * G / rotor_count
    theory = prop_hover_rpm(thrust, d, ct)
    rpm_hi = prop_hover_rpm(thrust, d, ct_band[0])    # low Ct -> high RPM
    rpm_lo = prop_hover_rpm(thrust, d, ct_band[1])
    ct_implied = thrust / (RHO * d ** 4 * (gazebo_rpm / 60.0) ** 2)
    return RpmCrossCheck(
        gazebo_rpm=gazebo_rpm, diameter_m=d, theory_rpm=theory,
        theory_rpm_lo=rpm_lo, theory_rpm_hi=rpm_hi,
        pct_diff=(gazebo_rpm / theory - 1.0) * 100.0,
        within_ct_band=rpm_lo <= gazebo_rpm <= rpm_hi,
        ct_implied=ct_implied,
    )
