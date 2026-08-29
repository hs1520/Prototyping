"""Every claimed standard-library name must actually resolve under syside.

The vocabulary tables serve two opposite roles: ``generation_plan``'s tables
decide which imports make a planned name resolvable, and ``diagnostics``'
tables decide which reference errors the in-loop checker may treat as false
positives.  A plausible-but-nonexistent name in either table produces the same
silent failure: materialisation writes it faithfully, the repair loop never
sees the error (it is "suppressed as a stdlib false positive"), and the run
fails only at the unfiltered terminal qualification.  Measured: the ablation
pilot of 2026-08-29 was NOT_QUALIFIED on ``AngleValue`` — one of five phantom
type names the tables carried.  These tests pin every table to syside ground
truth with one probe model per table kind.
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

#: KerML/ScalarValues scalar type names that are legitimately usable as
#: attribute types but do not end in "Value".
_SCALAR_TYPE_NAMES = {
    "Real", "Integer", "Boolean", "String", "Rational", "Complex",
    "Natural", "ScalarValue", "NumericalValue",
}


def _failures_by_name(
    names: Sequence[str], lines: Sequence[str]
) -> Dict[str, List[str]]:
    """One syside check over ``lines`` (one probe per name, same order).

    Returns {probed name: [unfiltered error messages]}, attributing each
    parser/semantic error to the name whose line produced it.
    """
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


def test_generation_plan_type_table_matches_the_standard_library():
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


def test_constraint_plan_value_types_all_resolve():
    failures = _type_failures(sorted(RESOLVABLE_VALUE_TYPES))
    assert not failures, (
        "phantom entries in RESOLVABLE_VALUE_TYPES — the gate that exists "
        f"precisely to reject unresolvable planned types: {failures}"
    )


def test_suppression_table_only_hides_names_that_really_exist():
    # Package names (SysML, KerML, Quantities, …) suppress import-absence
    # complaints and are not attribute types; probe only the type-shaped
    # entries — a phantom among them hides a REAL error from the repair loop.
    names = sorted(
        name for name in _STDLIB_TYPE_NAMES
        if name.endswith("Value") or name in _SCALAR_TYPE_NAMES
    )
    failures = _type_failures(names)
    assert not failures, (
        "phantom entries in diagnostics._STDLIB_TYPE_NAMES — suppressing a "
        f"nonexistent name silences a genuine reference error: {failures}"
    )


def test_every_unit_resolves_bare_or_has_an_explicit_resolution():
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


def test_unit_resolutions_stay_within_the_unit_vocabulary():
    orphans = set(_UNIT_RESOLUTIONS) - _SI_UNIT_NAMES
    assert not orphans, (
        f"_UNIT_RESOLUTIONS entries missing from _SI_UNIT_NAMES: {orphans}"
    )
