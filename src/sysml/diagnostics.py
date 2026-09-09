"""SysML standard-library diagnostic classification shared by parsers."""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# SysML v2 standard library types, implicitly available in conformant tools
# (SysML v2 Pilot, Cameo, Rhapsody). syside does not inject them and flags
# them as undefined; they are false positives.
# ---------------------------------------------------------------------------

_STDLIB_TYPE_NAMES = {
    "Real", "Integer", "Boolean", "String", "Rational", "Complex",
    "ScalarValue", "NumericalValue",
    # ISQ / SI units. Only names that exist in the standard library belong here:
    # suppressing a nonexistent one (AngleValue, VelocityValue, VoltageValue,
    # CurrentValue and ChargeValue once sat here) hides a reference error from the
    # repair loop until terminal qualification fails the run. Membership is pinned
    # by tests/test_stdlib_vocabulary.py.
    "ISQ", "SI", "LengthValue", "MassValue", "TimeValue", "DurationValue",
    "SpeedValue", "AccelerationValue", "ForceValue",
    "EnergyValue", "PowerValue", "FrequencyValue", "AngularMeasureValue",
    "PlaneAngleValue", "ThermodynamicTemperatureValue", "TemperatureValue",
    "ElectricPotentialValue", "ElectricCurrentValue", "ElectricChargeValue",
    "DimensionOneValue",
    "SysML", "KerML", "ScalarValues", "Quantities",
    "Occurrences", "Transfers", "Connections",
    "Requirements", "Constraints", "Parts", "Ports",
    "Interfaces", "Flows", "Actions", "States", "UseCases",
    "Allocations", "Geometries",
}

# SI / ISQ unit feature names used in attribute definitions like `= 15.0 [m/s]`
# syside reports these as "No Feature named 'X' found." - also false positives.
_STDLIB_UNIT_NAMES = {
    "m", "km", "cm", "mm", "um", "nm",
    "s", "ms", "us", "ns", "min", "h", "hr",
    "kg", "g", "mg",
    "deg", "rad", "grad",
    "Hz", "kHz", "MHz", "GHz",
    "m_s", "km_h", "knot",
    "m_s2",
    "N", "kN", "Pa", "kPa", "MPa", "bar",
    "J", "kJ", "W", "kW", "MW",
    "V", "mV", "kV", "A", "mA", "C", "Ah",
    # Temperature (Cel = the SI library's canonical degree-Celsius symbol;
    # degC/degF are aliases)
    "K", "degC", "Cel", "degF",
    "Wh", "kWh", "mAh", "rpm", "Nm",
    "dB", "dBA",
    "pct", "percent",
    "bit", "bits", "byte", "bytes", "B",
    "kbit", "Kbit", "Mbit", "Gbit",
    "kB", "MB", "GB", "TB",
    "bps", "kbps", "Kbps", "Mbps", "Gbps", "baud",
    "G", "g_force", "lx", "lm", "cd",
    "mm_hr", "m_s2", "rad_s",
    "SI", "ISQ",
    # Archived/best-effort callers still parse legacy state spellings. Evidence
    # and terminal paths pass ``filter_stdlib_diagnostics=False``, so they do not
    # get this allowance.
    "initial", "final", "done", "accept",
}


def is_stdlib_sema_error(message: str) -> bool:
    """Return True if the sema error only reports a missing standard-library type
    or unit - a false positive from syside's strict parsing mode.
    """
    m = re.search(r"No (?:Type|Namespace|Feature) named '([^']+)' found", message)
    if not m:
        return False
    name = m.group(1)
    root = name.split("::")[0]
    return root in _STDLIB_TYPE_NAMES or root in _STDLIB_UNIT_NAMES
