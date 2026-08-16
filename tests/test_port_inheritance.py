"""Ports inherited via `:>` specialisation must be visible to the connectivity
auditor and the behavioral extractor.

When a variation point is resolved, its usage is bound to a variant type that
specialises the host type (`part def Variant :> Host`).  The ports the host's
connects reference live on Host and are inherited.  If the simulator's port
model ignored inheritance, every such connect would be pruned as "no port",
isolating the component (reachability → 0).
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
    assert "sensorIn" in ports and "motorCmd" in ports  # inherited from FlightController


def test_auditor_keeps_connects_through_inheritance():
    res = audit_connects(_RESOLVED)
    assert res.n_removed == 0, [v.summary() for v in res.violations]


def test_extractor_reaches_inherited_port_component():
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
