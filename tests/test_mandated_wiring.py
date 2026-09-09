"""Prompt-mandated wiring is planned by construction and cannot drift.

Both authoring prompts mandate the SafetyMonitor interconnect; the step-1 plan
sometimes omitted it, and conformance then flagged the mandated structure as
unplanned (four ports + two connections on ablation pilot 2 and run 219eb9bb).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.llm.chain_of_thought import (
    ARCHITECTURE_DECOMPOSITION_TEMPLATE,
    PART_DEFINITIONS_TEMPLATE,
)
from src.prototyping.mandated_wiring import (
    MANDATED_LINKS,
    PART_SIDE_RULES_BLOCK,
    PLAN_SIDE_RULES_BLOCK,
    augment_architecture_payload,
)

_REPO = Path(__file__).resolve().parents[1]
_RUN = _REPO / "tests/fixtures/run_219eb9bb"


def _payload(**overrides):
    components = [
        {"name": "FlightController", "responsibility": "controls",
         "requirements": ["REQ_FUNC_001"], "ports": []},
        {"name": "SafetyMonitor", "responsibility": "monitors",
         "requirements": ["REQ_SAFE_001"], "ports": []},
        {"name": "CommunicationSystem", "responsibility": "links",
         "requirements": ["REQ_INTF_001"], "ports": []},
        {"name": "PerceptionSystem", "responsibility": "senses",
         "requirements": ["REQ_FUNC_002"], "ports": []},
    ]
    payload = {"components": components, "connections": []}
    payload.update(overrides)
    return payload


def test_templates_carry_blocks_verbatim():
    assert PLAN_SIDE_RULES_BLOCK in ARCHITECTURE_DECOMPOSITION_TEMPLATE
    assert PART_SIDE_RULES_BLOCK in PART_DEFINITIONS_TEMPLATE


def test_links_stated_in_both_blocks():
    for link in MANDATED_LINKS:
        for block in (PLAN_SIDE_RULES_BLOCK, PART_SIDE_RULES_BLOCK):
            assert f"out {'port ' if 'DataPort' in block else ''}{link.port_name}" \
                .replace("port  ", "port ") .strip() in block.replace("`", ""), (
                f"{link.port_name} (out side) missing from a rules block"
            )
            assert f"in {'port ' if 'DataPort' in block else ''}{link.port_name}" \
                .replace("port  ", "port ").strip() in block.replace("`", ""), (
                f"{link.port_name} (in side) missing from a rules block"
            )


def test_missing_wiring_is_planned():
    augmented, notes = augment_architecture_payload(_payload())

    ports = {
        (component["name"], port["name"], port["direction"])
        for component in augmented["components"]
        for port in component["ports"]
    }
    assert ("SafetyMonitor", "overrideCmd", "out") in ports
    assert ("FlightController", "overrideCmd", "in") in ports
    assert ("CommunicationSystem", "commStatus", "out") in ports
    assert ("SafetyMonitor", "commStatus", "in") in ports
    assert ("PerceptionSystem", "sensorStatus", "out") in ports
    assert ("SafetyMonitor", "sensorStatus", "in") in ports
    assert len(augmented["connections"]) == 3
    assert len(notes) == 3
    assert all("planned by construction" in note for note in notes)


def test_augmentation_is_idempotent():
    first, _ = augment_architecture_payload(_payload())
    second, notes = augment_architecture_payload(first)
    assert notes == []
    assert second == first


def test_absent_roles_leave_payload():
    payload = _payload(components=[
        {"name": "PowerSystem", "responsibility": "powers",
         "requirements": [], "ports": []},
    ])
    augmented, notes = augment_architecture_payload(payload)
    assert notes == []
    assert augmented["components"][0]["ports"] == []
    assert augmented["connections"] == []


def test_conflict_not_overwritten():
    payload = _payload()
    payload["components"][1]["ports"] = [
        {"name": "overrideCmd", "direction": "in", "type": "DataPort",
         "external": False},
    ]
    augmented, notes = augment_architecture_payload(payload)
    conflict_notes = [note for note in notes if "conflict" in note]
    assert len(conflict_notes) == 1 and "overrideCmd" in conflict_notes[0]
    safety = next(c for c in augmented["components"]
                  if c["name"] == "SafetyMonitor")
    override_ports = [p for p in safety["ports"] if p["name"] == "overrideCmd"]
    assert override_ports == payload["components"][1]["ports"]
    assert not any(
        (connection["source"]["port"] == "overrideCmd")
        for connection in augmented["connections"]
    )


def test_passive_endpoint_skips_link():
    payload = _payload()
    payload["components"][3]["passive"] = True
    _augmented, notes = augment_architecture_payload(payload)
    assert any("passive" in note and "sensorStatus" in note for note in notes)


def test_ambiguous_role_skips_link():
    payload = _payload()
    payload["components"].append(
        {"name": "BackupController", "responsibility": "backup",
         "requirements": [], "ports": []},
    )
    _augmented, notes = augment_architecture_payload(payload)
    assert any("more than one" in note for note in notes)


# ---------------------------------------------------------------------------
# Archived regression: run 219eb9bb's plan omitted the wiring its model carried;
# with the wiring planned by construction conformance no longer flags it.
# ---------------------------------------------------------------------------


def test_archived_219eb9bb_conforms():
    from src.prototyping.generation_plan import (
        ModelGenerationPlan,
        apply_generation_plan,
    )

    report = json.loads((_RUN / "canonical_run.json").read_text())
    plan_dict = report["pipeline_report"]["whole_model_generation_plan"]
    model_text = (_RUN / "final_model.sysml").read_text()

    augmented, notes = augment_architecture_payload(plan_dict)
    assert notes, "the archived plan omitted the wiring; augmentation must act"
    plan = ModelGenerationPlan.from_dict(augmented)

    _text, conformance = apply_generation_plan(model_text, plan)
    wiring_issues = [
        issue for issue in conformance.get("issues", [])
        if any(port in issue for port in
               ("overrideCmd", "commStatus", "sensorStatus"))
    ]
    assert wiring_issues == [], wiring_issues
