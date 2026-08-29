"""Single authority for unit identity across the pipeline.

The ablation pilots exposed the cost of fragmenting this knowledge: at least
four partial tables (semantic compiler canonicalisation, import/resolution
injection, plan token acceptance, SysML emission) each knew a different slice
of "what is a unit", and `m/s` fell through the cracks — planned as ``m/s``,
authored as ``[m_s]`` (the prompt's identifier-safe bracket convention), never
given a resolution alias, and compared unequal by the frozen-threshold check.
`deg` had the same disease one incident earlier.

Every unit the pipeline can accept, emit, or compare now lives here, once:

- ``canonical``  — the requirement/plan-facing spelling ("m/s", "%", "deg").
- ``emission``   — the identifier-safe token written inside model brackets
  (``[m_s]``): bracket readers throughout the pipeline depend on word
  characters, so slashes and signs never reach model text.
- ``resolution`` — the SysML construct injected when the emission token is not
  bare-resolvable under the standard imports (alias onto the SI unit where one
  exists, a conversion-defined unit where none does).
- ``quantity_type`` — the syside-verified value type binding validation
  requires for the unit.
- ``spellings``  — every accepted input spelling, lowercased.

The derived views at the bottom feed the pre-existing table names in
``requirement_semantics``, ``generation_plan`` and ``activated_constraint_plan``
so their consumers keep working; tests/test_stdlib_vocabulary.py drives every
entry through syside end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class UnitSpec:
    canonical: str
    emission: str
    quantity_type: Optional[str] = None
    spellings: Tuple[str, ...] = ()
    #: SysML construct that makes ``emission`` resolvable; empty when the
    #: token already resolves bare under ``import SI::*`` et al.
    resolution: str = ""
    resolution_namespaces: Tuple[str, ...] = ()


_MS_DEFINITION = (
    "attribute ms : DurationUnit {\n"
    "        :>> unitConversion : ConversionByPrefix {\n"
    "            :>> prefix = milli;\n"
    "            :>> referenceUnit = s;\n"
    "        }\n"
    "    }"
)
_PERCENT_DEFINITION = (
    "attribute percent : DimensionOneUnit {\n"
    "        :>> unitConversion : ConversionByConvention {\n"
    "            :>> referenceUnit = one;\n"
    "            :>> conversionFactor = 0.01;\n"
    "        }\n"
    "    }"
)

UNITS: Tuple[UnitSpec, ...] = (
    # ── time ──────────────────────────────────────────────────────────────
    UnitSpec("s", "s", "DurationValue", ("second", "seconds")),
    UnitSpec(
        "ms", "ms", "DurationValue", ("millisecond", "milliseconds"),
        # SI declares prefixed units selectively (mm/cm/km exist; ms does
        # not), so the millisecond follows SI's own ConversionByPrefix
        # pattern instead of an alias.
        resolution=_MS_DEFINITION,
        resolution_namespaces=("MeasurementReferences", "ISQ"),
    ),
    UnitSpec("min", "min", "DurationValue", ("minute", "minutes")),
    UnitSpec("h", "h", None, ("hour", "hours")),
    # ── length ────────────────────────────────────────────────────────────
    UnitSpec("m", "m", "LengthValue", ("metre", "metres", "meter", "meters")),
    UnitSpec("cm", "cm"),
    UnitSpec("mm", "mm"),
    UnitSpec(
        "km", "km", "LengthValue",
        ("kilometre", "kilometres", "kilometer", "kilometers"),
    ),
    # ── speed (slash never reaches model text) ────────────────────────────
    UnitSpec(
        "m/s", "m_s", "SpeedValue", ("m_s",),
        resolution="alias m_s for SI::'m/s';",
    ),
    UnitSpec(
        "km/h", "km_h", "SpeedValue", ("km_h",),
        resolution="alias km_h for SI::'km/h';",
    ),
    # ── mass ──────────────────────────────────────────────────────────────
    UnitSpec("kg", "kg", "MassValue", ("kilogram", "kilograms")),
    UnitSpec("g", "g"),
    # ── angle (SI names the unit `degree`, symbol °) ──────────────────────
    UnitSpec(
        "deg", "deg", "AngularMeasureValue", ("degree", "degrees", "°"),
        resolution="alias deg for SI::degree;",
    ),
    UnitSpec("rad", "rad", "AngularMeasureValue"),
    # ── temperature ───────────────────────────────────────────────────────
    UnitSpec(
        "degC", "degC", None, ("°c", "degc", "celsius"),
        resolution=(
            "alias degC for SI::'degree celsius (temperature difference)';"
        ),
    ),
    UnitSpec("K", "K"),
    # ── dimensionless (SI defines no percent unit at all) ─────────────────
    UnitSpec(
        "%", "percent", "DimensionOneValue", ("percent",),
        resolution=_PERCENT_DEFINITION,
        resolution_namespaces=("MeasurementReferences",),
    ),
    # ── electrical / mechanical symbols ───────────────────────────────────
    UnitSpec("Hz", "Hz", "FrequencyValue", ("hertz", "hz")),
    UnitSpec("A", "A"),
    UnitSpec("C", "C"),
    UnitSpec("V", "V"),
    UnitSpec("W", "W"),
    UnitSpec("J", "J"),
    UnitSpec("N", "N"),
    UnitSpec("Pa", "Pa"),
)


def _spelling_keys(unit: UnitSpec) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(
        spelling.lower()
        for spelling in (unit.canonical, unit.emission, *unit.spellings)
    ))


#: Every token the pipeline may write inside model brackets.
EMISSION_TOKENS: FrozenSet[str] = frozenset(unit.emission for unit in UNITS)

#: emission token -> (construct, "ns,ns") for tokens that are not
#: bare-resolvable under the standard imports.
RESOLUTIONS: Dict[str, Tuple[str, str]] = {
    unit.emission: (unit.resolution, ",".join(unit.resolution_namespaces))
    for unit in UNITS
    if unit.resolution
}

#: lowercased accepted spelling -> canonical spelling.
CANONICAL_BY_SPELLING: Dict[str, str] = {
    key: unit.canonical
    for unit in UNITS
    for key in _spelling_keys(unit)
}

#: lowercased accepted spelling -> emission token, only where they differ
#: (identity lookups fall through unchanged, preserving prior behaviour for
#: tokens outside the registry).
EMISSION_BY_SPELLING: Dict[str, str] = {
    key: unit.emission
    for unit in UNITS
    for key in _spelling_keys(unit)
    if key != unit.emission
}

#: canonical spelling -> syside-verified quantity type.
QUANTITY_TYPE_BY_CANONICAL: Dict[str, str] = {
    unit.canonical: unit.quantity_type
    for unit in UNITS
    if unit.quantity_type
}
