"""Physics estimator: multirotor design inputs -> emergent performance.

Bridges DSE to SITL calibration (layer 2/3): the DSE recommends design inputs
(battery, mass, rotor geometry, cruise speed), this module predicts the emergent
metrics (endurance, range) analytically, and SITL produces the same metrics by
simulation from the same inputs, so calibration compares the two rankings. Every
constant below is a standard physical or accepted small-multirotor engineering
value, annotated with its meaning; the endurance model is momentum-theory hover
power divided into usable battery energy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81
RHO = 1.225
CELL_V = 3.7
USABLE = 0.8      # usable battery fraction (no discharge below ~20%)
FOM = 0.62
ETA_DRIVE = 0.75
AVIONICS_POWER_W = 12.0
ENERGY_DENSITY_WH_KG = 150.0
BASE_FRAME_KG = 0.5
ROTOR_MASS_COEF = 2.5

# ── Catalog calibration overrides (F1) ──────────────────────────────
# The constants above are generic engineering values and run ~36% low on
# endurance against the datasheet layer (experiment B). set_calibration() lets
# the orchestrator inject catalog-derived effective values around the search
# (fitted in src/realization/estimator_calibration.py; this module has no dependency
# on the realization layer). State is explicit and restorable; when empty, behavior
# is bit-identical to the uncalibrated estimator.
_CALIBRATION: dict = {}
_CALIBRATION_KEYS = frozenset(
    {"fom", "eta_drive", "energy_density_wh_kg", "base_frame_kg", "rotor_mass_coef"}
)


def set_calibration(**overrides: float) -> None:
    """Install effective-constant overrides; unknown keys raise ValueError."""
    bad = set(overrides) - _CALIBRATION_KEYS
    if bad:
        raise ValueError(f"unknown calibration key(s): {sorted(bad)}")
    _CALIBRATION.update({k: float(v) for k, v in overrides.items()})


def clear_calibration() -> None:
    _CALIBRATION.clear()


def calibration_active() -> dict:
    return dict(_CALIBRATION)


class calibrated:
    """Context manager: apply overrides, restore the previous state on exit."""

    def __init__(self, **overrides: float) -> None:
        self._overrides = overrides
        self._saved: dict = {}

    def __enter__(self) -> "calibrated":
        self._saved = dict(_CALIBRATION)
        clear_calibration()
        set_calibration(**self._overrides)
        return self

    def __exit__(self, *exc) -> None:
        clear_calibration()
        _CALIBRATION.update(self._saved)


@dataclass(frozen=True)
class DesignInputs:
    """SITL-settable multirotor design inputs (what the DSE chooses).

    All-up mass is not an input: it emerges from battery pack mass (cells x capacity
    ÷ energy density) + payload + a fixed frame base, so a bigger battery pays a
    weight penalty - more energy, but more mass to lift.
    """
    payload_mass_kg: float
    battery_capacity_mah: float
    battery_cells: int
    rotor_count: int
    rotor_radius_m: float
    cruise_speed_mps: float = 0.0


def disk_area_m2(rotor_count: int, rotor_radius_m: float) -> float:
    """Total rotor disk area."""
    return rotor_count * math.pi * rotor_radius_m ** 2


def hover_power_w(mass_kg: float, area_m2: float, fom: float = None) -> float:
    """Ideal momentum-theory hover power divided by the rotor figure of merit.

    P = T^1.5 / (sqrt(2 ρ A) . FOM), with thrust T = m g (hover). ``fom`` defaults to
    the module constant FOM, read at call time so FOM stays tunable.
    """
    if area_m2 <= 0:
        return float("inf")
    f = _CALIBRATION.get("fom", FOM) if fom is None else fom
    thrust = mass_kg * G
    return thrust ** 1.5 / (math.sqrt(2.0 * RHO * area_m2) * f)


def battery_energy_wh(capacity_mah: float, cells: int) -> float:
    """Nominal pack energy = (Ah) . (pack voltage)."""
    return (capacity_mah / 1000.0) * cells * CELL_V


def battery_mass_kg(capacity_mah: float, cells: int) -> float:
    """Pack mass from energy ÷ gravimetric energy density."""
    return battery_energy_wh(capacity_mah, cells) / _CALIBRATION.get(
        "energy_density_wh_kg", ENERGY_DENSITY_WH_KG)


def propulsion_mass_kg(rotor_count: int, rotor_radius_m: float) -> float:
    """Motors + props + arms scale with total disk area, so bigger or more rotors weigh
    more. Rotor sizing therefore carries a cost and the outer search does not
    collapse to the largest rotor.
    """
    return _CALIBRATION.get("rotor_mass_coef", ROTOR_MASS_COEF) * disk_area_m2(
        rotor_count, rotor_radius_m)


def total_mass_kg(d: DesignInputs) -> float:
    """Emergent all-up mass = frame base + propulsion group + battery pack + payload."""
    return (_CALIBRATION.get("base_frame_kg", BASE_FRAME_KG)
            + propulsion_mass_kg(d.rotor_count, d.rotor_radius_m)
            + battery_mass_kg(d.battery_capacity_mah, d.battery_cells)
            + d.payload_mass_kg)


def electrical_power_w(mass_kg: float, area_m2: float) -> float:
    """Electrical draw at hover = mechanical (aero/FOM) power ÷ drive efficiency + the
    constant avionics/payload hotel load. This is what drains the pack, so endurance
    uses it; mechanical power alone is optimistic.
    """
    mech = hover_power_w(mass_kg, area_m2)
    if math.isinf(mech):
        return float("inf")
    return mech / _CALIBRATION.get("eta_drive", ETA_DRIVE) + AVIONICS_POWER_W


def endurance_min(d: DesignInputs) -> float:
    """First-order hover endurance: usable battery energy / electrical hover power,
    lifting the emergent all-up mass (battery self-weight, drive losses, hotel load).
    """
    energy = battery_energy_wh(d.battery_capacity_mah, d.battery_cells) * USABLE
    power = electrical_power_w(total_mass_kg(d), disk_area_m2(d.rotor_count, d.rotor_radius_m))
    if power <= 0 or math.isinf(power):
        return 0.0
    return (energy / power) * 60.0


def range_m(d: DesignInputs) -> float:
    """Cruise range = endurance x cruise speed (first-order; ignores cruise-vs-hover power difference, conservative)."""
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
