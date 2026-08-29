"""Redundant inherited-port redeclarations are stripped, nothing else is.

Measured: run 00e4d333 carried ten namespace-distinguishability warnings —
an LLM refinement pass decorated every catalog variant with the
``propulsionStatus`` port each already inherits from PropulsionSystem — and
the terminal qualification's zero-warning policy failed the run on them.
The normaliser removes only declarations identical to an inherited one
(name, direction, and type); anything that differs stays for the checker.
"""
from __future__ import annotations

from pathlib import Path

from src.simulation.syntax_checker import check_syntax
from src.sysml.text_normalization import strip_redundant_inherited_ports

_REPO = Path(__file__).resolve().parents[1]


def test_exact_inherited_redeclaration_is_stripped():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Impl :> Host { out port status : StatusPort; attribute n : Real = 4.0; }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 1
    assert out.count("out port status : StatusPort;") == 1
    assert "attribute n : Real = 4.0;" in out


def test_differing_direction_or_type_is_left_for_the_checker():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def A :> Host { in port status : StatusPort; }
    part def B :> Host { out port status : RichStatusPort; }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text


def test_definitions_without_parents_are_untouched():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Standalone { out port status : StatusPort; }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text


def test_transitive_inheritance_is_recognised():
    text = """package P {
    part def Base { out port status : StatusPort; }
    part def Mid :> Base { attribute x : Real = 1.0; }
    part def Leaf :> Mid { out port status : StatusPort; }
}"""
    _out, count = strip_redundant_inherited_ports(text)
    assert count == 1


def test_nested_definition_ports_are_owned_by_the_innermost_def():
    # The nested def has no parent, so its port must survive even though the
    # OUTER def inherits an identical declaration.
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Outer :> Host {
        part def Inner { out port status : StatusPort; }
    }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text


def test_normaliser_is_idempotent():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Impl :> Host { out port status : StatusPort; }
}"""
    once, first = strip_redundant_inherited_ports(text)
    twice, second = strip_redundant_inherited_ports(once)
    assert first == 1 and second == 0
    assert twice == once


def test_archived_00e4d333_loses_all_ten_warnings_and_no_semantics():
    text = (
        _REPO / "examples/output/runs"
        / "00e4d333-d87a-4f33-bc5c-b7768fdfeb79/final_model.sysml"
    ).read_text()
    out, count = strip_redundant_inherited_ports(text)
    assert count == 10
    result = check_syntax(out, filter_stdlib_diagnostics=False)
    assert result.total_errors() == 0
    assert len(result.warnings) == 0


def test_archived_pilot4_is_untouched():
    text = (
        _REPO / "experiments/ablation/results"
        / "20260829_174841_pilot4/runs/FULL_seed0.final.sysml"
    ).read_text()
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text
