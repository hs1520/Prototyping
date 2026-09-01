"""A component the plan declares passive plans no ports, is materialised as a
PLAN-PASSIVE marker, and takes no part in reachability scenarios."""
from src.prototyping.generation_plan import (
    ModelGenerationPlan, materialize_passive_components, passive_components_in_text,
)
from src.simulation.extractor import BehavioralGraph, PartNode
from src.simulation.scenarios import auto_detect_scenarios


def _payload(passive_block):
    return {
        "components": [
            {"name": "PowerSystem", "responsibility": "supplies power",
             "requirements": ["REQ_FUNC_001"],
             "ports": [{"name": "power", "direction": "out", "type": "PowerPort", "external": False}]},
            {"name": "FlightController", "responsibility": "controls",
             "requirements": ["REQ_FUNC_001"],
             "ports": [{"name": "power", "direction": "in", "type": "PowerPort", "external": False}]},
            passive_block,
        ],
        "connections": [{
            "source": {"component": "PowerSystem", "port": "power"},
            "target": {"component": "FlightController", "port": "power"},
            "item_type": "PowerPort", "requirements": ["REQ_FUNC_001"],
        }],
    }


def test_passive_component_may_plan_no_ports_but_needs_a_reason():
    ok = ModelGenerationPlan.from_payload(_payload({
        "name": "Airframe", "responsibility": "carries the parts",
        "requirements": [], "ports": [], "passive": True,
        "passive_rationale": "structural body; no signals, commands or power in this model",
    }))
    assert not any("Airframe has no planned ports" in i for i in ok.issues)
    no_reason = ModelGenerationPlan.from_payload(_payload({
        "name": "Airframe", "responsibility": "carries the parts",
        "requirements": [], "ports": [], "passive": True,
    }))
    assert any("passive_rationale" in i for i in no_reason.issues)
    with_ports = ModelGenerationPlan.from_payload(_payload({
        "name": "Airframe", "responsibility": "carries the parts",
        "requirements": [], "passive": True, "passive_rationale": "x",
        "ports": [{"name": "power", "direction": "in", "type": "PowerPort", "external": False}],
    }))
    assert any("declared passive but plans" in i for i in with_ports.issues)


def test_passive_marker_is_written_once_and_read_back():
    plan = ModelGenerationPlan.from_payload(_payload({
        "name": "Airframe", "responsibility": "carries the parts",
        "requirements": [], "ports": [], "passive": True,
        "passive_rationale": "structural body only",
    }))
    text = "package P {\n    part def Airframe {\n    }\n    part def PowerSystem {\n    }\n}\n"
    once, written = materialize_passive_components(text, plan.components)
    assert written == ["Airframe"]
    assert "// PLAN-PASSIVE Airframe: structural body only" in once
    twice, written_again = materialize_passive_components(once, plan.components)
    assert twice == once and written_again == []
    assert passive_components_in_text(once) == {"Airframe"}


def test_declared_passive_part_generates_no_scenarios():
    def graph(passive):
        bg = BehavioralGraph()
        bg.parts["powerSystem"] = PartNode(id="powerSystem", def_name="PowerSystem")
        bg.parts["airframe"] = PartNode(id="airframe", def_name="Airframe")
        if passive:
            bg.passive_defs = {"Airframe"}
        return bg
    with_expectation = auto_detect_scenarios(graph(passive=False))
    without = auto_detect_scenarios(graph(passive=True))
    assert any("airframe" in s.target_nodes for s in with_expectation)
    assert not any("airframe" in s.target_nodes or "airframe" in s.entry_nodes for s in without)


_UNCONNECTED_AIRFRAME = """package P {{
    port def DataPort;
    part def Airframe {{{marker}
    }}
    part def PerceptionSystem {{
        out port data : DataPort;
    }}
    part def FlightController {{
        in port data : DataPort;
    }}
    part def System {{
        part airframe : Airframe;
        part perceptionSystem : PerceptionSystem;
        part flightController : FlightController;
        connect perceptionSystem.data to flightController.data;
    }}
}}"""


def test_declared_passive_part_is_not_charged_the_isolation_penalty():
    """Measured on the s0v5 anchor: a perfect run (19/19 requirement paths,
    20/20 advisory scenarios) read reachability 0.9 because the plan-passive
    airframe has no connect statements and the scorer charged it the
    isolation penalty — while the isolated-parts feedback simultaneously
    told refinement to wire it in, against the plan's own passivity
    discipline. Passivity is the model's recorded decision; the scorer must
    read it. An UNDECLARED unconnected part keeps the penalty."""
    from src.simulation.validator import SimulationValidator

    passive = SimulationValidator().validate(
        _UNCONNECTED_AIRFRAME.format(
            marker="\n        // PLAN-PASSIVE Airframe: structural body only"
        ),
        model_name="P",
    )
    assert passive.passive_unconnected_parts == ["airframe"]
    assert passive.isolated_parts == []
    assert passive.reachability_score == 1.0

    undeclared = SimulationValidator().validate(
        _UNCONNECTED_AIRFRAME.format(marker=""), model_name="P",
    )
    assert undeclared.isolated_parts == ["airframe"]
    assert undeclared.passive_unconnected_parts == []
    assert undeclared.reachability_score < 1.0


def test_passive_component_plans_exactly_the_structural_mount():
    """A passive body plans one structural attachment port -- fixed name and
    type -- because a mounting interface is a legitimate connection point (the
    standard's ports are interaction points, mechanical ones included), and
    leaving it unplanned made every generator that sensibly wrote one
    non-conformant (measured: two consecutive end-to-end draws added an
    unplanned airframe mount port). The emitted port definition has no
    features, so nothing can be exchanged through it and the passive marker
    stays true."""
    from src.prototyping.generation_plan import (
        STRUCTURAL_MOUNT_PORT_NAME,
        STRUCTURAL_MOUNT_PORT_TYPE,
        ModelGenerationPlan,
    )
    payload = {
        "components": [
            {
                "name": "Airframe",
                "responsibility": "carry the parts",
                "requirements": ["REQ-CONS-001"],
                "ports": [],
                "attributes": [],
                "passive": True,
                "passive_rationale": "structural body; exchanges nothing",
            },
            {
                "name": "Controller",
                "responsibility": "control",
                "requirements": ["REQ-FUNC-001"],
                "ports": [{"name": "cmd", "direction": "out",
                           "type": "DataPort"}],
                "attributes": [],
            },
        ],
        "connections": [],
    }
    plan = ModelGenerationPlan.from_payload(payload)
    airframe = next(c for c in plan.components if c.name == "Airframe")
    assert [p.name for p in airframe.ports] == [STRUCTURAL_MOUNT_PORT_NAME]
    assert airframe.ports[0].port_type == STRUCTURAL_MOUNT_PORT_TYPE
    assert airframe.ports[0].direction == "inout"
    # a passive component planning any OTHER port is an issue
    payload["components"][0]["ports"] = [
        {"name": "power", "direction": "in", "type": "DataPort"}
    ]
    plan2 = ModelGenerationPlan.from_payload(payload)
    assert any("passive" in issue and "power" in issue
               for issue in plan2.issues)
