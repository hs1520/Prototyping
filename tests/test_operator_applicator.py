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


# ── Protocol application (validated retype — replaces the regex injector) ───

_PROTO_MODEL = """package DroneSystem {
    part def FlightController {
        in port gnssIn : DataPort;
        out port telemetryOut : RfPort;
        in port powerIn : PowerPort;
    }
    port def DataPort;
    port def RfPort;
    port def PowerPort;
}"""


def _proto_cfg(protocol):
    return DesignConfiguration(
        name="b", parameters={"communication_protocol": protocol}
    )


def test_protocol_retype_is_valid_by_construction():
    m = _model(_PROTO_MODEL)
    applied = apply_architecture(m, _proto_cfg("MAVLink"))
    assert any("protocol=MAVLink" in a for a in applied)
    txt = m.metadata["last_sysml_text"]
    assert not check_syntax(txt).has_errors
    # data ports retyped to the rich catalog signal
    assert "in port gnssIn : MAVLinkSignal;" in txt
    assert "out port telemetryOut : MAVLinkSignal;" in txt
    # rich def with item-typed payload (what the old empty `port def X;` lacked)
    assert "port def MAVLinkSignal :> BdseSignal { in item payload : MAVLinkFrame; }" in txt


def test_protocol_leaves_power_ports_untouched():
    m = _model(_PROTO_MODEL)
    apply_architecture(m, _proto_cfg("CAN"))
    txt = m.metadata["last_sysml_text"]
    assert "in port powerIn : PowerPort;" in txt
    assert "port def PowerPort;" in txt


def test_protocol_removes_stale_generic_defs():
    m = _model(_PROTO_MODEL)
    apply_architecture(m, _proto_cfg("Ethernet"))
    txt = m.metadata["last_sysml_text"]
    assert "port def DataPort;" not in txt
    assert "port def RfPort;" not in txt


def test_unknown_protocol_applies_nothing():
    m = _model(_PROTO_MODEL)
    assert apply_architecture(m, _proto_cfg("Zigbee")) == []
    assert m.metadata["last_sysml_text"] == _PROTO_MODEL


def test_redundancy_and_protocol_compose():
    m = _model(_PROTO_MODEL)
    applied = apply_architecture(
        m,
        DesignConfiguration(name="b", parameters={
            "redundancy_level": "triple",
            "communication_protocol": "MAVLink",
        }),
    )
    assert len(applied) == 2
    txt = m.metadata["last_sysml_text"]
    assert not check_syntax(txt).has_errors
    assert "bdseSafetyMonitor" in txt and "MAVLinkSignal" in txt
