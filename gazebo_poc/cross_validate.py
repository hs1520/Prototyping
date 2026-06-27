"""Cross-validate the two independent fidelity sources (Gazebo dynamics vs datasheet power)
on the one quantity where they genuinely meet: HOVER THRUST PER ROTOR (stage 4).

Why not throttle-%: we tried comparing the simulator's trimmed hover throttle to the datasheet
throttle and found it INVALID — ArduCopter's VFR_HUD throttle is a normalised/learned value, not
the physical motor command (scaling the rotor LiftDrag area 2.4× left the reported hover throttle
unchanged at 37%). The physically comparable quantity would be rotor RPM, but the iris model's
joint-state publisher does not expose the rotor joints, so RPM isn't observable without an SDF
change (documented as the remaining refinement).

What DOES cross-validate: at hover, thrust per rotor = m·g/N by force balance — both sources must
honour this. Gazebo confirms the airframe reaches that thrust in STABLE controlled flight
(dynamics feasibility); the datasheet says the chosen motor produces that thrust within its
envelope and reports the current → endurance (power feasibility). Consistency = both feasible.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.dse.physics_estimator import G

from .component_data import MN5008_KV340_18x61, MotorProp
from .datasheet_endurance import datasheet_endurance


@dataclass(frozen=True)
class CrossValidation:
    hover_thrust_per_rotor_n: float
    hover_thrust_per_rotor_g: float
    gazebo_stable: bool
    datasheet_within_envelope: bool
    twr_margin: float
    hover_current_per_motor_a: float
    endurance_min: float
    datasheet_hover_throttle_pct: float    # reported for context only — NOT compared to sim
    consistent: bool
    note: str


def cross_validate(mass_kg: float, rotor_count: int, capacity_mah: float,
                   gazebo_stable: bool,
                   motor: MotorProp = MN5008_KV340_18x61) -> CrossValidation:
    thrust_n = mass_kg * G / rotor_count
    thrust_g = thrust_n / G * 1000.0
    within = thrust_g <= motor.max_thrust_g()
    if not within:                                  # motor can't hover it → no power figures
        return CrossValidation(
            hover_thrust_per_rotor_n=thrust_n, hover_thrust_per_rotor_g=thrust_g,
            gazebo_stable=gazebo_stable, datasheet_within_envelope=False,
            twr_margin=motor.max_thrust_g() / 1000.0 * rotor_count / mass_kg,
            hover_current_per_motor_a=float("inf"), endurance_min=0.0,
            datasheet_hover_throttle_pct=100.0, consistent=False,
            note="INCONSISTENT: thrust exceeds motor max — can't hover")
    ds = datasheet_endurance(mass_kg, rotor_count, capacity_mah, motor)
    consistent = gazebo_stable and within
    note = ("Gazebo: stable hover at this thrust (dynamics OK); datasheet: motor produces it "
            "within envelope (power OK) → design feasible by BOTH sources."
            if consistent else
            "INCONSISTENT: " + ("; ".join(
                ([] if gazebo_stable else ["Gazebo did not reach stable hover"])
                + ([] if within else ["thrust exceeds motor max — can't hover"]))))
    return CrossValidation(
        hover_thrust_per_rotor_n=thrust_n,
        hover_thrust_per_rotor_g=thrust_g,
        gazebo_stable=gazebo_stable,
        datasheet_within_envelope=within,
        twr_margin=ds.twr_max,
        hover_current_per_motor_a=ds.hover_current_per_motor_a,
        endurance_min=ds.endurance_min,
        datasheet_hover_throttle_pct=motor.throttle_at_thrust(thrust_g) * 100.0,
        consistent=consistent,
        note=note,
    )


def report(cv: CrossValidation) -> str:
    return (
        f"cross-validation @ hover thrust {cv.hover_thrust_per_rotor_g:.0f} g/rotor "
        f"({cv.hover_thrust_per_rotor_n:.1f} N):\n"
        f"  Gazebo (dynamics): stable hover = {cv.gazebo_stable}\n"
        f"  datasheet (power): within envelope = {cv.datasheet_within_envelope} "
        f"(TWR {cv.twr_margin:.1f}), {cv.hover_current_per_motor_a:.1f} A/motor, "
        f"endurance {cv.endurance_min:.1f} min\n"
        f"  [context] datasheet hover throttle ~{cv.datasheet_hover_throttle_pct:.0f}% "
        f"(NOT comparable to sim's normalised throttle)\n"
        f"  consistent: {cv.consistent} — {cv.note}"
    )
