"""Ports inherited via `:>` specialisation are visible to the connectivity auditor
and the behavioral extractor.

Resolving a variation point binds the usage to a variant type specialising the
host (`part def Variant :> Host`), and the referenced ports live on Host. A port
model ignoring inheritance prunes every such connect as "no port", isolating the
component (reachability -> 0).
"""
from __future__ import annotations

from src.simulation.connectivity_fixer import audit_connects, build_port_directory
from src.simulation.extractor import extract_behavioral_graph

_RESOLVED = """package Drone {
    port def Sig;
    part def FlightController { in port sensorIn : Sig; out port motorCmd : Sig; }
    part def FCUltraLight :> FlightController { attribute massKg : Real = 0.1; }
    part def SensorSuite { out port data : Sig; }
    part def Motor { in port cmd : Sig; }
    part def Airframe {
        part sensorSuite : SensorSuite;
        part flightController : FCUltraLight;
        part motor : Motor;
        connect sensorSuite.data to flightController.sensorIn;
        connect flightController.motorCmd to motor.cmd;
    }
}"""


def test_directory_exposes_inherited_ports():
    d = build_port_directory(_RESOLVED)
    ports = d.instances.get("flightController", {})
    assert "sensorIn" in ports and "motorCmd" in ports


def test_auditor_keeps_inherited_connects():
    res = audit_connects(_RESOLVED)
    assert res.n_removed == 0, [v.summary() for v in res.violations]


def test_extractor_reaches_inherited_ports():
    bg = extract_behavioral_graph(_RESOLVED)
    fc = bg.parts.get("flightController")
    assert fc is not None
    assert any(p.endswith(".sensorIn") for p in fc.port_ids)
    assert any(p.endswith(".motorCmd") for p in fc.port_ids)


def test_untyped_inherited_ports_resolve():
    text = """package P {
        part def Base { in port a; out port b; }
        part def V :> Base { }
        part def Sys { part x : V; part y : Base; connect y.b to x.a; }
    }"""
    assert "a" in build_port_directory(text).instances.get("x", {})
    assert audit_connects(text).n_removed == 0
