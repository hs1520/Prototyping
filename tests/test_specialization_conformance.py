from __future__ import annotations

from src.prototyping.generation_plan import (
    ModelGenerationPlan,
    apply_generation_plan,
)
from src.prototyping.structural_obligations import (
    part_def_bases,
    specializes,
)
from src.simulation.syntax_checker import check_syntax

_MODEL = """package P {
    item def CmdData;
    port def CmdPort { item payload : CmdData; }
    part def Controller { out port cmd : CmdPort; }
    part def Propulsion {
        in port cmd : CmdPort;
        out port status : CmdPort;
    }
    part def CatalogPropulsionImpl :> Propulsion { attribute rotors : Real = 6.0; }
    part controller : Controller;
    part propulsion : CatalogPropulsionImpl;
    connect controller.cmd to propulsion.cmd;
    part analysisBlock {
        part propulsion : CatalogPropulsionImpl;
    }
}"""

_PAYLOAD = {
    "components": [
        {"name": "Controller", "responsibility": "commands",
         "requirements": ["REQ-FUNC-001"],
         "ports": [{"name": "cmd", "direction": "out", "type": "CmdPort"}],
         "attributes": []},
        {"name": "Propulsion", "responsibility": "thrust",
         "requirements": ["REQ-FUNC-001"],
         "ports": [{"name": "cmd", "direction": "in", "type": "CmdPort"},
                   {"name": "status", "direction": "out", "type": "CmdPort"}],
         "attributes": []},
    ],
    "connections": [
        {"source": {"component": "Controller", "port": "cmd"},
         "target": {"component": "Propulsion", "port": "cmd"},
         "item_type": "CmdData"},
    ],
}


def test_specializes_walks_the_chain():
    bases = part_def_bases(_MODEL)
    assert specializes("CatalogPropulsionImpl", "Propulsion", bases)
    assert not specializes("Controller", "Propulsion", bases)


def test_resolves_subtyped_usage():
    plan = ModelGenerationPlan.from_payload(_PAYLOAD)
    updated, conformance = apply_generation_plan(_MODEL, plan)
    issues = conformance.get("issues") or []
    assert not any("cannot uniquely resolve" in i for i in issues), issues
    assert conformance.get("missing_components") in (None, []), conformance.get(
        "missing_components")
    assert conformance.get("missing_ports") in (None, []), conformance.get(
        "missing_ports")
    assert "in port cmd : CmdPort;" not in updated.split(
        "CatalogPropulsionImpl :> Propulsion")[1].split("}")[0]


def test_verification_warning_free():
    from src.dse import analysis_emitter
    import inspect
    src = inspect.getsource(analysis_emitter)
    assert "verification def {rid}_check" not in src
    probe = """package P {
        private import ScalarValues::*;
        requirement def REQ_A { doc /* x */ }
        part def S { attribute m : Real = 1.0; }
        part s : S {
            requirement req_a : REQ_A;
            verification req_a_check { objective req_a_obj { verify req_a; } }
        }
    }"""
    r = check_syntax(probe, fail_closed=True, filter_stdlib_diagnostics=False)
    assert not r.has_errors and not r.warnings


def test_untyped_port_retyped():
    """`in port environmentExposure;` is legal grammar (the type is optional).

    An untyped planned port is retyped rather than reported as a missing/unplanned
    pair, the shape seen in an archived run's terminal audit.
    """
    from src.prototyping.generation_plan import normalise_planned_port_types

    model = """package P {
        port def EnvironmentPort;
        part def Airframe {
            in port environmentExposure;
        }
        part airframe : Airframe;
    }"""
    payload = {
        "components": [
            {"name": "Airframe", "responsibility": "structure",
             "requirements": ["REQ-CONS-002"],
             "ports": [{"name": "environmentExposure", "direction": "in",
                        "type": "EnvironmentPort"}],
             "attributes": []},
        ],
        "connections": [],
    }
    plan = ModelGenerationPlan.from_payload(payload)
    updated, changes = normalise_planned_port_types(model, plan.components)
    assert "in port environmentExposure : EnvironmentPort;" in updated
    assert changes
