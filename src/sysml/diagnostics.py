"""SysML standard-library diagnostic classification shared by parsers."""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# SysML v2 standard library types — implicitly available in all conformant
# tools (SysML v2 Pilot, Cameo, Rhapsody).  syside flags these as undefined
# because it doesn't inject them automatically, but they are NOT real model
# errors in practice.
# ---------------------------------------------------------------------------

_STDLIB_TYPE_NAMES = {
    # ScalarValues (KerML)
    "Real", "Integer", "Boolean", "String", "Rational", "Complex",
    "ScalarValue", "NumericalValue",
    # ISQ / SI units.  Only names that actually exist in the standard library
    # belong here: suppressing a nonexistent name (AngleValue, VelocityValue,
    # VoltageValue, CurrentValue, ChargeValue once sat here) hides a REAL
    # reference error from the repair loop, so it survives silently until the
    # unfiltered terminal qualification fails the run.  Membership is pinned by
    # tests/test_stdlib_vocabulary.py.
    "ISQ", "SI", "LengthValue", "MassValue", "TimeValue", "DurationValue",
    "SpeedValue", "AccelerationValue", "ForceValue",
    "EnergyValue", "PowerValue", "FrequencyValue", "AngularMeasureValue",
    "PlaneAngleValue", "ThermodynamicTemperatureValue", "TemperatureValue",
    "ElectricPotentialValue", "ElectricCurrentValue", "ElectricChargeValue",
    "DimensionOneValue",
    # SysML standard packages
    "SysML", "KerML", "ScalarValues", "Quantities",
    "Occurrences", "Transfers", "Connections",
    "Requirements", "Constraints", "Parts", "Ports",
    "Interfaces", "Flows", "Actions", "States", "UseCases",
    "Allocations", "Geometries",
}

# SI / ISQ unit feature names used in attribute definitions like `= 15.0 [m/s]`
# syside reports these as "No Feature named 'X' found." — also false positives.
_STDLIB_UNIT_NAMES = {
    # Length / area / volume
    "m", "km", "cm", "mm", "um", "nm",
    # Time
    "s", "ms", "us", "ns", "min", "h", "hr",
    # Mass
    "kg", "g", "mg",
    # Angle
    "deg", "rad", "grad",
    # Frequency
    "Hz", "kHz", "MHz", "GHz",
    # Speed
    "m_s", "km_h", "knot",
    # Acceleration
    "m_s2",
    # Force / pressure
    "N", "kN", "Pa", "kPa", "MPa", "bar",
    # Energy / power
    "J", "kJ", "W", "kW", "MW",
    # Voltage / current / charge
    "V", "mV", "kV", "A", "mA", "C", "Ah",
    # Temperature  (Cel = UCUM/SI symbol for degree Celsius — the SI library's
    # canonical name; degC/degF are LLM-friendly aliases)
    "K", "degC", "Cel", "degF",
    # Energy / charge capacity & rotation (common in drone/EV domains)
    "Wh", "kWh", "mAh", "rpm", "Nm",
    # Sound
    "dB", "dBA",
    # Percentage / dimensionless
    "pct", "percent",
    # Data / information units (LLM commonly annotates comms attributes with these)
    "bit", "bits", "byte", "bytes", "B",
    "kbit", "Kbit", "Mbit", "Gbit",
    "kB", "MB", "GB", "TB",
    "bps", "kbps", "Kbps", "Mbps", "Gbps", "baud",
    # Misc
    "G", "g_force", "lx", "lm", "cd",
    # Compound unit names that LLM might use
    "mm_hr", "m_s2", "rad_s",
    # SysML unit packages
    "SI", "ISQ",
    # Archived/best-effort callers still parse legacy state spellings. Evidence
    # and terminal paths always pass ``filter_stdlib_diagnostics=False`` and
    # therefore cannot use this compatibility allowance.
    "initial", "final", "done", "accept",
}


def is_stdlib_sema_error(message: str) -> bool:
    """
    Return True if this sema error is purely about a missing standard-library
    type or unit — i.e. a false positive caused by syside's strict parsing mode.

    Patterns matched:
      "No Type named '<X>' found."      — stdlib type (Real, Boolean …)
      "No Namespace named '<X>' found." — stdlib package (SI, ISQ …)
      "No Feature named '<X>' found."   — SI unit symbol (m, Hz, deg …)
    """
    m = re.search(r"No (?:Type|Namespace|Feature) named '([^']+)' found", message)
    if not m:
        return False
    name = m.group(1)
    root = name.split("::")[0]
    return root in _STDLIB_TYPE_NAMES or root in _STDLIB_UNIT_NAMES
