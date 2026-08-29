"""Reference-chain initializers yield None, not a suppressed TypeError.

The typed semantic bindings write ``attribute currentX : T =
channel.payload.feature;`` — syside's Compiler evaluates that initializer to
the referenced AttributeUsage node, not a scalar.  Three evaluation sites
called ``float()`` on it blindly; pilot 2 recorded 369 suppressed TypeErrors
across them (lite_model, syside_utils, requirement_linker).  A reference
initializer has no static scalar: that is data, not an error.
"""
from __future__ import annotations

from pathlib import Path

from src.utils.suppressed import reset_suppressed, suppressed_summary
from src.utils.syside_utils import coerce_static_number, extract_attr_values

_PILOT2_MODEL = (
    Path(__file__).resolve().parents[1]
    / "experiments/ablation/results/20260829_131825_pilot2/runs"
    / "FULL_seed0.final.sysml"
)

_EVAL_CHANNELS = (
    "sysml.lite_model.attr_compiler_eval",
    "utils.syside_utils.attr_eval",
    "sitl.requirement_linker.syside_attr_node",
)


def test_coercion_accepts_scalars_and_refuses_nodes():
    assert coerce_static_number(120) == 120.0
    assert coerce_static_number(0.5) == 0.5
    assert coerce_static_number("18.0") == 18.0
    assert coerce_static_number(None) is None
    assert coerce_static_number(True) is None
    assert coerce_static_number("payload.feature") is None
    assert coerce_static_number(object()) is None


_REFERENCE_CHAIN_MODEL = """package P {
    private import ScalarValues::*;
    item def AirspeedData {
        attribute cruiseAirspeed : Real;
    }
    port def AirspeedPort {
        in item payload : AirspeedData;
    }
    part def Controller {
        in port airspeed : AirspeedPort;
        attribute plainLiteral : Real = 42.0;
        attribute currentCruiseAirspeed : Real =
            airspeed.payload.cruiseAirspeed;
    }
}"""


def test_reference_initializers_are_skipped_silently_literals_still_evaluate():
    reset_suppressed()
    values = extract_attr_values(_REFERENCE_CHAIN_MODEL)

    # The literal proves the sweep actually ran; the reference chain is
    # legitimately absent (no static scalar) instead of a suppressed error.
    assert values.get("plainLiteral") == 42.0
    assert "currentCruiseAirspeed" not in values
    noise = {
        key: entry["count"]
        for key, entry in (suppressed_summary() or {}).items()
        if key in _EVAL_CHANNELS
    }
    assert noise == {}, (
        f"reference initializers leaked TypeErrors into suppression: {noise}"
    )


def test_pilot2_sweep_keeps_the_eval_channels_silent():
    # The archived pilot-2 model carries the real binding attributes that
    # produced 369 suppressed TypeErrors; a full sweep must stay silent.
    # (Unit-bearing initializers are Compiler-FATAL by long-standing
    # behaviour, so an empty result is acceptable here — the literal case
    # above proves non-vacuity.)
    reset_suppressed()
    extract_attr_values(_PILOT2_MODEL.read_text().replace("*/; }", "*/ }"))
    noise = {
        key: entry["count"]
        for key, entry in (suppressed_summary() or {}).items()
        if key in _EVAL_CHANNELS
    }
    assert noise == {}
