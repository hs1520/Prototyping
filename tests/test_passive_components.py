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
