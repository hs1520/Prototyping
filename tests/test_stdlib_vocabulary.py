"""Every claimed standard-library name resolves under syside.

``generation_plan``'s tables decide which imports make a planned name
resolvable; ``diagnostics``' tables decide which reference errors the in-loop
checker treats as false positives. A nonexistent name in either fails silently:
materialisation writes it, the repair loop never sees the error, and the run
fails only at the unfiltered terminal qualification (the 2026-08-29 pilot was
NOT_QUALIFIED on ``AngleValue``, one of five phantom names). Each table is
pinned to syside with one probe model per table kind.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from src.prototyping.activated_constraint_plan import RESOLVABLE_VALUE_TYPES
from src.prototyping.generation_plan import (
    _SI_UNIT_NAMES,
    _STANDARD_LIBRARY_TYPES,
    _UNIT_RESOLUTIONS,
)
from src.simulation.syntax_checker import check_syntax
from src.sysml.diagnostics import _STDLIB_TYPE_NAMES

_PROBE_HEADER = (
    "package VocabularyProbe {\n"
    "    private import ISQ::*;\n"
    "    private import MeasurementReferences::*;\n"
    "    private import SI::*;\n"
    "    private import ScalarValues::*;\n"
    "    part def P {\n"
)
_PROBE_FOOTER = "    }\n}\n"

_SCALAR_TYPE_NAMES = {
    "Real", "Integer", "Boolean", "String", "Rational", "Complex",
    "Natural", "ScalarValue", "NumericalValue",
}


def _failures_by_name(
    names: Sequence[str], lines: Sequence[str]
) -> Dict[str, List[str]]:
    header_lines = _PROBE_HEADER.count("\n")
    text = (
        _PROBE_HEADER
        + "".join(f"        {line}\n" for line in lines)
        + _PROBE_FOOTER
    )
    result = check_syntax(text, filter_stdlib_diagnostics=False)
    failures: Dict[str, List[str]] = {}
    for error in list(result.parser_errors) + list(result.sema_errors):
        index = int(error.get("line", 0)) - header_lines - 1
        name = names[index] if 0 <= index < len(names) else f"<line {error.get('line')}>"
        failures.setdefault(name, []).append(str(error.get("message")))
    return failures


def _type_failures(names: Sequence[str]) -> Dict[str, List[str]]:
    return _failures_by_name(
        names, [f"attribute t{i} : {name};" for i, name in enumerate(names)]
    )


def test_plan_type_table_resolves():
    names = sorted(
        name
        for members in _STANDARD_LIBRARY_TYPES.values()
        for name in members
    )
    failures = _type_failures(names)
    assert not failures, (
        "phantom entries in generation_plan._STANDARD_LIBRARY_TYPES "
        f"(import computation blesses names syside cannot resolve): {failures}"
    )


def test_value_types_resolve():
    failures = _type_failures(sorted(RESOLVABLE_VALUE_TYPES))
    assert not failures, (
        "phantom entries in RESOLVABLE_VALUE_TYPES — the gate that exists "
        f"precisely to reject unresolvable planned types: {failures}"
    )


def test_suppressed_names_exist():
    # Package names (SysML, KerML, Quantities, ...) suppress import-absence
    # complaints and are not attribute types, so probe only the type-shaped
    # entries; a phantom among them hides an error from the repair loop.
    names = sorted(
        name for name in _STDLIB_TYPE_NAMES
        if name.endswith("Value") or name in _SCALAR_TYPE_NAMES
    )
    failures = _type_failures(names)
    assert not failures, (
        "phantom entries in diagnostics._STDLIB_TYPE_NAMES — suppressing a "
        f"nonexistent name silences a genuine reference error: {failures}"
    )


def test_every_unit_resolves():
    bare_units = sorted(_SI_UNIT_NAMES - set(_UNIT_RESOLUTIONS))
    failures = _failures_by_name(
        bare_units,
        [
            f"attribute u{i} : Real = 1.0 [{unit}];"
            for i, unit in enumerate(bare_units)
        ],
    )
    assert not failures, (
        "units in _SI_UNIT_NAMES that neither resolve under `import SI::*` "
        f"nor carry a _UNIT_RESOLUTIONS alias/definition: {failures}"
    )


def test_resolutions_within_vocabulary():
    orphans = set(_UNIT_RESOLUTIONS) - _SI_UNIT_NAMES
    assert not orphans, (
        f"_UNIT_RESOLUTIONS entries missing from _SI_UNIT_NAMES: {orphans}"
    )


# ---------------------------------------------------------------------------
# Unit registry: one authority, end-to-end through syside. Pilot 2 failed on
# `m_s` because emission, resolution and comparison each had a partial table.
# ---------------------------------------------------------------------------


def test_registry_units_survive_syside():
    from src.prototyping.unit_registry import RESOLUTIONS, UNITS

    constructs = "\n    ".join(
        construct for construct, _ns in RESOLUTIONS.values()
    )
    lines = [
        f"attribute u{i} : Real = 1.0 [{unit.emission}];"
        for i, unit in enumerate(UNITS)
    ]
    lines += [
        f"attribute q{i} : {unit.quantity_type} = 1.0 [{unit.emission}];"
        for i, unit in enumerate(UNITS)
        if unit.quantity_type
    ]
    text = (
        "package RegistryProbe {\n"
        "    private import ISQ::*;\n"
        "    private import MeasurementReferences::*;\n"
        "    private import SI::*;\n"
        "    private import ScalarValues::*;\n"
        f"    {constructs}\n"
        "    part def P {\n"
        + "".join(f"        {line}\n" for line in lines)
        + "    }\n}\n"
    )
    result = check_syntax(text, filter_stdlib_diagnostics=False)
    messages = [
        error.get("message")
        for error in list(result.parser_errors) + list(result.sema_errors)
    ]
    assert not messages, (
        f"registry emission tokens / quantity pairings failed syside: {messages}"
    )


def test_registry_views_agree():
    from src.prototyping.activated_constraint_plan import sysml_unit_name
    from src.prototyping.requirement_semantics import (
        _normalise_unit,
        quantity_type_for_unit,
    )
    from src.prototyping.unit_registry import (
        EMISSION_TOKENS,
        RESOLUTIONS,
        UNITS,
    )

    assert _SI_UNIT_NAMES == set(EMISSION_TOKENS)
    assert _UNIT_RESOLUTIONS == RESOLUTIONS
    for unit in UNITS:
        assert sysml_unit_name(unit.canonical) == unit.emission
        assert _normalise_unit(unit.emission) == unit.canonical
        assert quantity_type_for_unit(unit.emission) == unit.quantity_type
        assert quantity_type_for_unit(unit.canonical) == unit.quantity_type
