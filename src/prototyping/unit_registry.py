"""Single authority for unit identity across the pipeline.

Unit knowledge used to be split across four partial tables (semantic compiler
canonicalisation, import/resolution injection, plan token acceptance, SysML
emission), and `m/s` fell through: planned as ``m/s``, authored as ``[m_s]``,
never given a resolution alias, and compared unequal by the frozen-threshold
check; `deg` failed the same way earlier. Every unit the pipeline can accept,
emit or compare now lives here once:

- ``canonical``  - the requirement/plan-facing spelling ("m/s", "%", "deg").
- ``emission``   - the identifier-safe token written inside model brackets
  (``[m_s]``), since bracket readers depend on word characters.
- ``resolution`` - the SysML construct injected when the emission token is not
  bare-resolvable under the standard imports.
- ``quantity_type`` - the syside-verified value type binding for the unit.
- ``spellings``  - every accepted input spelling, lowercased.

The derived views at the bottom feed the table names in ``requirement_semantics``,
``generation_plan`` and ``activated_constraint_plan``; tests/test_stdlib_vocabulary.py
drives every entry through syside.
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
    UnitSpec("m", "m", "LengthValue", ("metre", "metres", "meter", "meters")),
    UnitSpec("cm", "cm"),
    UnitSpec("mm", "mm"),
    UnitSpec(
        "km", "km", "LengthValue",
        ("kilometre", "kilometres", "kilometer", "kilometers"),
    ),
    UnitSpec(
        "m/s", "m_s", "SpeedValue", ("m_s",),
        resolution="alias m_s for SI::'m/s';",
    ),
    UnitSpec(
        "km/h", "km_h", "SpeedValue", ("km_h",),
        resolution="alias km_h for SI::'km/h';",
    ),
    UnitSpec("kg", "kg", "MassValue", ("kilogram", "kilograms")),
    UnitSpec("g", "g"),
    UnitSpec(
        "deg", "deg", "AngularMeasureValue", ("degree", "degrees", "°"),
        resolution="alias deg for SI::degree;",
    ),
    UnitSpec("rad", "rad", "AngularMeasureValue"),
    UnitSpec(
        "degC", "degC", None, ("°c", "degc", "celsius"),
        resolution=(
            "alias degC for SI::'degree celsius (temperature difference)';"
        ),
    ),
    UnitSpec("K", "K"),
    UnitSpec(
        "%", "percent", "DimensionOneValue", ("percent",),
        resolution=_PERCENT_DEFINITION,
        resolution_namespaces=("MeasurementReferences",),
    ),
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


EMISSION_TOKENS: FrozenSet[str] = frozenset(unit.emission for unit in UNITS)

RESOLUTIONS: Dict[str, Tuple[str, str]] = {
    unit.emission: (unit.resolution, ",".join(unit.resolution_namespaces))
    for unit in UNITS
    if unit.resolution
}

CANONICAL_BY_SPELLING: Dict[str, str] = {
    key: unit.canonical
    for unit in UNITS
    for key in _spelling_keys(unit)
}

# lowercased accepted spelling -> emission token, only where they differ;
# identity lookups fall through unchanged for tokens outside the registry.
EMISSION_BY_SPELLING: Dict[str, str] = {
    key: unit.emission
    for unit in UNITS
    for key in _spelling_keys(unit)
    if key != unit.emission
}

QUANTITY_TYPE_BY_CANONICAL: Dict[str, str] = {
    unit.canonical: unit.quantity_type
    for unit in UNITS
    if unit.quantity_type
}
