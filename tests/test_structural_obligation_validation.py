from types import SimpleNamespace

from src.prototyping.generation_plan import ModelGenerationPlan
from src.prototyping.model_qualification import build_model_qualification
from src.prototyping.structural_obligations import (
    StructuralObligation,
    validate_structural_obligations,
)
from src.simulation.validator import SimulationValidator


_PAYLOAD = {
    "components": [
        {
            "name": "Source",
            "responsibility": "Produces a typed signal.",
            "requirements": ["REQ_FUNC_001"],
            "ports": [{
                "name": "signal",
                "direction": "out",
                "type": "SignalPort",
                "external": False,
            }],
        },
        {
            "name": "Sink",
            "responsibility": "Consumes a typed signal.",
            "requirements": ["REQ_FUNC_001"],
            "ports": [{
                "name": "signal",
                "direction": "in",
                "type": "SignalPort",
                "external": False,
            }],
        },
    ],
    "connections": [{
        "source": {"component": "Source", "port": "signal"},
        "target": {"component": "Sink", "port": "signal"},
        "item_type": "SignalPort",
        "requirements": ["REQ_FUNC_001"],
    }],
}

_CONNECTED = """package P {
    port def SignalPort;
    requirement def REQ_FUNC_001 { doc /* signal propagation */ }
    part def Source {
        out port signal : SignalPort;
        satisfy requirement REQ_FUNC_001;
    }
    part def Sink { in port signal : SignalPort; }
    part source : Source;
    part sink : Sink;
    connect source.signal to sink.signal;
}"""


def _plan():
    return ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ_FUNC_001: propagate the signal"],
    )


def test_terminal_validation_uses_frozen_requirement_path():
    report = validate_structural_obligations(
        _CONNECTED,
        _plan().structural_obligations,
        model_name="P",
    )

    assert report["status"] == "PASS"
    assert report["scenario_set_fixed"] is True
    assert report["passed"] == report["total"] == 1
    assert report["results"][0]["requirement_id"] == "REQ_FUNC_001"
    assert report["results"][0]["observed_path"] == [
        "source",
        "source.signal",
        "sink.signal",
        "sink",
    ]


def test_missing_planned_edge_fails_the_named_obligation():
    report = validate_structural_obligations(
        _CONNECTED.replace(
            "    connect source.signal to sink.signal;\n",
            "",
        ),
        _plan().structural_obligations,
        model_name="P",
    )

    assert report["status"] == "FAIL"
    assert report["results"][0]["obligation_id"] == (
        "STRUCT_REQ_FUNC_001_001"
    )
    assert report["results"][0]["missing_connections"]


def test_heuristic_scenarios_are_advisory_not_a_qualification_gate():
    structural = validate_structural_obligations(
        _CONNECTED,
        _plan().structural_obligations,
        model_name="P",
    )
    simulation = SimulationValidator().validate(
        _CONNECTED,
        model_name="P",
    )
    qualification = build_model_qualification(
        model_text=_CONNECTED,
        requirements=["REQ_FUNC_001: propagate the signal"],
        syntax_result=SimpleNamespace(total_errors=lambda: 0),
        simulation_result=simulation,
        terminal_consistency={
            "status": "PASS",
            "model_digest": "same",
            "simulation_source_model_digest": "same",
            "evaluation_source_model_digest": "same",
        },
        structural_obligation_report=structural,
        generation_plan_conformance={"status": "PASS"},
    )

    checks = {item["name"]: item for item in qualification["checks"]}
    assert checks["REQUIREMENT_STRUCTURAL_OBLIGATIONS"]["status"] == "PASS"
    assert checks["HEURISTIC_STRUCTURAL_DIAGNOSTIC"]["status"] == "ADVISORY"
    heuristic = checks["HEURISTIC_STRUCTURAL_DIAGNOSTIC"]["evidence"]
    assert heuristic["evidence_kind"] == "UNTRACED_ADVISORY"
    assert heuristic["qualification_effect"] == "NONE"
    assert heuristic["role_assignments"]
    assert heuristic["weakly_connected_components"]
    assert all(
        scenario["provenance"] == "AUTO_ROLE_HEURISTIC"
        for scenario in heuristic["scenarios"]
    )
    assert "STRUCTURAL_REACHABILITY" not in checks


def test_local_behavior_realization_requires_the_named_sysml_behavior():
    obligation = StructuralObligation(
        obligation_id="STRUCT_REQ_OPER_001_001",
        requirement_id="REQ_OPER_001",
        source_component="Controller",
        target_component="Controller",
        required_components=("Controller",),
        required_connections=(),
        entry_kind="SOURCE_ANCHORED_LOCAL_BEHAVIOR",
        trigger_concept="standby mode",
        effect_concept="active mode",
        provenance="FROZEN_REQUIREMENT_REALIZATION",
        realization_kind="LOCAL_BEHAVIOR",
        behavior_kind="STATE_DEF",
        behavior_name="OperatingModes",
    )
    model = """package P {
        part def Controller {
            state def OperatingModes {
                state standby;
                state active;
                transition activate first standby then active;
            }
        }
        part controller : Controller;
    }"""

    passed = validate_structural_obligations(
        model, (obligation,), model_name="P"
    )
    failed = validate_structural_obligations(
        model.replace("state def OperatingModes", "state def OtherModes"),
        (obligation,),
        model_name="P",
    )

    assert passed["status"] == "PASS"
    assert passed["results"][0]["observed_path"] == [
        "controller",
        "STATE_DEF OperatingModes",
    ]
    assert failed["status"] == "FAIL"
    assert failed["results"][0]["missing_behavior_elements"] == [
        "STATE_DEF Controller.OperatingModes"
    ]
