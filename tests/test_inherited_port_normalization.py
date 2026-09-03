"""Redundant inherited-port redeclarations are stripped, nothing else is.

Run 00e4d333 carried ten namespace-distinguishability warnings: a refinement
pass added the inherited ``propulsionStatus`` port to every catalog variant, and
the terminal zero-warning policy failed the run. Only declarations identical to
an inherited one (name, direction, type) are removed; anything differing stays
for the checker.
"""
from __future__ import annotations

from pathlib import Path

from src.simulation.syntax_checker import check_syntax
from src.sysml.text_normalization import strip_redundant_inherited_ports

_REPO = Path(__file__).resolve().parents[1]


def test_exact_redeclaration_stripped():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Impl :> Host { out port status : StatusPort; attribute n : Real = 4.0; }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 1
    assert out.count("out port status : StatusPort;") == 1
    assert "attribute n : Real = 4.0;" in out


def test_differing_direction_kept():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def A :> Host { in port status : StatusPort; }
    part def B :> Host { out port status : RichStatusPort; }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text


def test_no_parent_untouched():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Standalone { out port status : StatusPort; }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text


def test_transitive_inheritance():
    text = """package P {
    part def Base { out port status : StatusPort; }
    part def Mid :> Base { attribute x : Real = 1.0; }
    part def Leaf :> Mid { out port status : StatusPort; }
}"""
    _out, count = strip_redundant_inherited_ports(text)
    assert count == 1


def test_nested_def_owns_ports():
    # The nested def has no parent, so its port survives even though the outer def
    # inherits an identical declaration.
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Outer :> Host {
        part def Inner { out port status : StatusPort; }
    }
}"""
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text


def test_idempotent():
    text = """package P {
    part def Host { out port status : StatusPort; }
    part def Impl :> Host { out port status : StatusPort; }
}"""
    once, first = strip_redundant_inherited_ports(text)
    twice, second = strip_redundant_inherited_ports(once)
    assert first == 1 and second == 0
    assert twice == once


def test_archived_00e4d333_no_warnings():
    text = (
        _REPO / "examples/output/runs"
        / "00e4d333-d87a-4f33-bc5c-b7768fdfeb79/final_model.sysml"
    ).read_text()
    out, count = strip_redundant_inherited_ports(text)
    assert count == 10
    result = check_syntax(out, filter_stdlib_diagnostics=False)
    assert result.total_errors() == 0
    assert len(result.warnings) == 0


def test_archived_pilot4_untouched():
    text = (
        _REPO / "experiments/ablation/results"
        / "20260829_174841_pilot4/runs/FULL_seed0.final.sysml"
    ).read_text()
    out, count = strip_redundant_inherited_ports(text)
    assert count == 0
    assert out == text
