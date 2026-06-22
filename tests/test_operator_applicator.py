"""Tests for the operator-based architecture applicator (Item F, step 4)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.dse.design_space import DesignConfiguration
from src.dse.operator_applicator import apply_architecture
from src.simulation.syntax_checker import check_syntax

_MODEL = """package DroneSystem {
    part def FlightController { attribute mass : Real = 1.0; }
    part def Battery;
}"""


def _model(text=_MODEL):
    return SimpleNamespace(metadata={"last_sysml_text": text})


def _cfg(level):
    return DesignConfiguration(name="b", parameters={"redundancy_level": level})


@pytest.mark.parametrize("level", ["dual", "triple"])
def test_merged_model_is_valid_by_construction(level):
    m = _model()
    applied = apply_architecture(m, _cfg(level))
    assert applied  # something was applied
    assert not check_syntax(m.metadata["last_sysml_text"]).has_errors


def test_applies_voting_machine_as_addressable_part_usage():
    m = _model()
    apply_architecture(m, _cfg("triple"))
    txt = m.metadata["last_sysml_text"]
    assert "part def BdseTripleModularRedundancy" in txt   # prefixed local type (no clashes)
    assert "failedChannels >= 2" in txt                    # real 2oo3 voting guard
    # addressable part usage with a LOCAL type → recognised by the behavioral extractor
    assert "part bdseSafetyMonitor : BdseTripleModularRedundancy;" in txt


def test_original_model_preserved():
    m = _model()
    apply_architecture(m, _cfg("triple"))
    txt = m.metadata["last_sysml_text"]
    assert "FlightController" in txt and "Battery" in txt


def test_none_redundancy_applies_nothing():
    m = _model()
    assert apply_architecture(m, _cfg("none")) == []
    assert m.metadata["last_sysml_text"] == _MODEL  # untouched


def test_unmergeable_model_left_untouched():
    """A model that cannot be merged into safely is never corrupted."""
    m = _model("garbage without braces")
    assert apply_architecture(m, _cfg("triple")) == []
    assert m.metadata["last_sysml_text"] == "garbage without braces"


def test_merge_skipped_if_it_would_break_parsing():
    """If the merge produced invalid SysML, the model is left untouched."""
    # an already-invalid model: merge result also won't parse → skip
    m = _model("package P { part def X { ")  # unbalanced
    before = m.metadata["last_sysml_text"]
    apply_architecture(m, _cfg("triple"))
    assert m.metadata["last_sysml_text"] == before
