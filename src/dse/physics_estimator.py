"""Physics estimator: multirotor design inputs → emergent performance.

This is the bridge that makes SITL calibration meaningful (DSE→SITL layer 2/3):
the DSE recommends *design inputs* (battery, mass, rotor geometry, cruise speed);
this module predicts the *emergent* metrics (endurance, range) ANALYTICALLY, while
SITL produces the same metrics by SIMULATION from the same inputs. Calibration then
compares analytic vs simulated rankings.

Objectivity: every constant below is a standard physical or accepted small-multirotor
engineering value (annotated with its meaning), not a fitted/tuned knob. The endurance
model is hover power from momentum theory divided into usable battery energy — the
textbook first-order estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ── Physical / engineering constants (standard values, not tuned) ──────────────
G = 9.81          # m/s^2, gravitational acceleration
RHO = 1.225       # kg/m^3, air density at sea level, ISA
CELL_V = 3.7      # V, nominal voltage per LiPo cell
USABLE = 0.8      # usable battery fraction (avoid deep discharge below ~20%)
FOM = 0.62        # rotor figure of merit (hover efficiency), typical small multirotor
ETA_DRIVE = 0.75  # combined motor + ESC electrical→mechanical efficiency
AVIONICS_POWER_W = 12.0  # autopilot + sensors + payload electronics (constant hotel load)
ENERGY_DENSITY_WH_KG = 150.0  # LiPo pack gravimetric energy density (typical)
BASE_FRAME_KG = 0.5           # bare frame + avionics (propulsion mass is separate, below)
ROTOR_MASS_COEF = 2.5         # propulsion-group mass (motors+props+arms) per m² disk area


@dataclass(frozen=True)
class DesignInputs:
    """SITL-settable multirotor design inputs (what the DSE chooses).

    Airframe all-up MASS is NOT an input — it EMERGES from battery pack mass
    (cells × capacity ÷ energy density) + payload + a fixed frame base. So a bigger
    battery correctly pays a weight penalty (the real battery-endurance trade-off):
    more energy, but also more mass to lift."""
    payload_mass_kg: float
    battery_capacity_mah: float
    battery_cells: int
    rotor_count: int
    rotor_radius_m: float
    cruise_speed_mps: float = 0.0   # 0 = pure hover endurance


def disk_area_m2(rotor_count: int, rotor_radius_m: float) -> float:
    """Total rotor disk area."""
    return rotor_count * math.pi * rotor_radius_m ** 2


def hover_power_w(mass_kg: float, area_m2: float, fom: float = None) -> float:
    """Ideal momentum-theory hover power divided by the rotor figure of merit.

    P = T^1.5 / (sqrt(2 ρ A) · FOM), with thrust T = m g (hover). ``fom`` defaults to
    the module constant FOM, read at CALL time (so FOM stays tunable/sensitivity-able)."""
    if area_m2 <= 0:
        return float("inf")
    f = FOM if fom is None else fom
    thrust = mass_kg * G
    return thrust ** 1.5 / (math.sqrt(2.0 * RHO * area_m2) * f)


def battery_energy_wh(capacity_mah: float, cells: int) -> float:
    """Nominal pack energy = (Ah) · (pack voltage)."""
    return (capacity_mah / 1000.0) * cells * CELL_V


def battery_mass_kg(capacity_mah: float, cells: int) -> float:
    """Pack mass from energy ÷ gravimetric energy density."""
    return battery_energy_wh(capacity_mah, cells) / ENERGY_DENSITY_WH_KG


def propulsion_mass_kg(rotor_count: int, rotor_radius_m: float) -> float:
    """Motors + props + arms scale with total disk area — bigger / more rotors are
    heavier. This gives rotor sizing a real COST so the outer search doesn't collapse
    to the largest rotor (otherwise larger rotors would be free efficiency)."""
    return ROTOR_MASS_COEF * disk_area_m2(rotor_count, rotor_radius_m)


def total_mass_kg(d: DesignInputs) -> float:
    """Emergent all-up mass = frame base + propulsion group + battery pack + payload."""
    return (BASE_FRAME_KG
            + propulsion_mass_kg(d.rotor_count, d.rotor_radius_m)
            + battery_mass_kg(d.battery_capacity_mah, d.battery_cells)
            + d.payload_mass_kg)


def electrical_power_w(mass_kg: float, area_m2: float) -> float:
    """Electrical draw at hover = mechanical (aero/FOM) power ÷ drive efficiency +
    the constant avionics/payload hotel load. This is what actually drains the pack,
    so endurance must use it (mechanical power alone is optimistic)."""
    mech = hover_power_w(mass_kg, area_m2)
    if math.isinf(mech):
        return float("inf")
    return mech / ETA_DRIVE + AVIONICS_POWER_W


def endurance_min(d: DesignInputs) -> float:
    """First-order hover endurance: usable battery energy / ELECTRICAL hover power,
    lifting the EMERGENT all-up mass (battery self-weight + drive losses + hotel load
    all paid for)."""
    energy = battery_energy_wh(d.battery_capacity_mah, d.battery_cells) * USABLE
    power = electrical_power_w(total_mass_kg(d), disk_area_m2(d.rotor_count, d.rotor_radius_m))
    if power <= 0 or math.isinf(power):
        return 0.0
    return (energy / power) * 60.0


def range_m(d: DesignInputs) -> float:
    """Cruise range = endurance × cruise speed (first-order; ignores cruise-vs-hover
    power difference, conservative)."""
    return endurance_min(d) * 60.0 * d.cruise_speed_mps


def estimate(d: DesignInputs) -> dict:
    """All emergent metrics for a design (the analytic side of the calibration)."""
    return {
        "endurance_min": round(endurance_min(d), 3),
        "range_m": round(range_m(d), 1),
        "total_mass_kg": round(total_mass_kg(d), 3),
        "electrical_power_w": round(electrical_power_w(total_mass_kg(d), disk_area_m2(d.rotor_count, d.rotor_radius_m)), 1),
        "cruise_speed_mps": d.cruise_speed_mps,
    }
