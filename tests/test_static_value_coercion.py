"""Reference-chain initializers yield None rather than a suppressed TypeError.

Typed semantic bindings write ``attribute currentX : T =
channel.payload.feature;`` and syside's Compiler evaluates that to the
referenced AttributeUsage node. Three sites called ``float()`` on it
(lite_model, syside_utils, requirement_linker); pilot 2 recorded 369
suppressed TypeErrors. A reference initializer has no static scalar.
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


def test_scalars_accepted_nodes_refused():
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


def test_references_skipped_literals_kept():
    reset_suppressed()
    values = extract_attr_values(_REFERENCE_CHAIN_MODEL)

    # The literal shows the sweep ran; the reference chain is absent (no static
    # scalar) rather than a suppressed error.
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


def test_pilot2_sweep_stays_silent():
    # The archived pilot-2 model carries the binding attributes that produced 369
    # suppressed TypeErrors; a full sweep stays silent. Unit-bearing initializers
    # are Compiler-FATAL, so an empty result is fine here - the literal case above
    # covers non-vacuity.
    reset_suppressed()
    extract_attr_values(_PILOT2_MODEL.read_text().replace("*/; }", "*/ }"))
    noise = {
        key: entry["count"]
        for key, entry in (suppressed_summary() or {}).items()
        if key in _EVAL_CHANNELS
    }
    assert noise == {}
