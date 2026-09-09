from __future__ import annotations

from src.dse.variation_introducer import (
    VariantSpec,
    connected_components,
    introduce_variation,
)
from src.dse.variation_parser import admitted, parse_variation_points, port_safe, resolve_model
from src.simulation.syntax_checker import check_syntax

_MODEL = """package Drone {
    port def Sig;
    part def SensorSuite { out port data : Sig; }
    part def FlightController { in port sensorIn : Sig; }
    part def Airframe {
        part sensorSuite : SensorSuite;
        part flightController : FlightController;
        connect sensorSuite.data to flightController.sensorIn;
    }
}"""

_VARIANTS = [
    VariantSpec("lidar", "LidarSuite", "attribute rangeM : Real = 100.0;"),
    VariantSpec("vision", "VisionSuite", "attribute massKg : Real = 0.2;"),
]


def test_connected_components_found():
    comps = dict(connected_components(_MODEL))
    assert comps["sensorSuite"] == "SensorSuite"
    assert comps["flightController"] == "FlightController"


def test_introduce_keeps_connect_safe():
    text, ok = introduce_variation(
        _MODEL, "sensorSuite", "SensorSuite", _VARIANTS, "range vs mass", ["REQ-PERF-001"]
    )
    assert ok and not check_syntax(text).has_errors
    assert "connect sensorSuite.data to flightController.sensorIn;" in text
    vp = admitted(parse_variation_points(text))[0][0]
    assert vp.point_id == "sensorSuite"
    assert port_safe(vp, text)


def test_resolution_keeps_connect():
    text, _ = introduce_variation(
        _MODEL, "sensorSuite", "SensorSuite", _VARIANTS, "range vs mass", ["REQ-PERF-001"]
    )
    vps = admitted(parse_variation_points(text))[0]
    for choice in ("lidar", "vision"):
        concrete = resolve_model(text, vps, {"sensorSuite": choice})
        assert not check_syntax(concrete).has_errors
        assert "connect sensorSuite.data to flightController.sensorIn;" in concrete


def test_one_variant_noop():
    text, ok = introduce_variation(
        _MODEL, "sensorSuite", "SensorSuite", _VARIANTS[:1], "x", ["REQ-PERF-001"]
    )
    assert not ok and text == _MODEL


def test_missing_usage_noop():
    text, ok = introduce_variation(
        _MODEL, "nonexistent", "SensorSuite", _VARIANTS, "x", ["REQ-PERF-001"]
    )
    assert not ok and text == _MODEL
