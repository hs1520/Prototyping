from __future__ import annotations

import copy
import sys
from types import SimpleNamespace
from types import ModuleType


if "dotenv" not in sys.modules:
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub
if "pinecone" not in sys.modules:
    pinecone_stub = ModuleType("pinecone")
    pinecone_stub.Pinecone = type(
        "Pinecone", (), {"__init__": lambda self, **kwargs: None}
    )
    sys.modules["pinecone"] = pinecone_stub
from src.prototyping.generation_plan import (
    ModelGenerationPlan,
    apply_generation_plan,
    materialize_standard_library_imports,
    validate_part_definition_fragment,
)
from src.simulation.syntax_checker import check_syntax
from src.prototyping.model_qualification import build_model_qualification
from src.simulation.validator import SimulationResult


_PAYLOAD = {
    "components": [
        {
            "name": "Producer",
            "responsibility": "Produces status.",
            "requirements": ["REQ_FUNC_001"],
            "ports": [
                {
                    "name": "status",
                    "direction": "out",
                    "type": "DataPort",
                    "external": False,
                }
            ],
        },
        {
            "name": "Consumer",
            "responsibility": "Consumes status.",
            "requirements": [],
            "ports": [
                {
                    "name": "status",
                    "direction": "in",
                    "type": "DataPort",
                    "external": False,
                }
            ],
        },
    ],
    "connections": [
        {
            "source": {"component": "Producer", "port": "status"},
            "target": {"component": "Consumer", "port": "status"},
            "item_type": "DataPort",
            "requirements": ["REQ_FUNC_001"],
        }
    ],
}


def test_typed_plan_validates_and_deterministically_materialises_connections():
    plan = ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ-FUNC-001: propagate status."],
    )
    assert plan.status == "PASS"

    model = """package P {
        port def DataPort;
        part def Producer { out port status : DataPort; }
        part def Consumer { in port status : DataPort; }
        part def System {
            part producer : Producer;
            part consumer : Consumer;
        }
    }"""
    updated, report = apply_generation_plan(model, plan)

    assert report["status"] == "PASS"
    assert report["realized_connection_count"] == 1
    assert "connect producer.status to consumer.status;" in updated


def test_terminal_compiler_closes_root_standard_library_imports():
    model = """package P {
        part def Controller {
            attribute enabled : Boolean = true;
            attribute separation : LengthValue = 5 [m];
        }
        part controller : Controller;
    }"""

    updated, report = materialize_standard_library_imports(model)

    assert report["status"] == "PASS"
    assert report["added_imports"] == [
        "private import ISQ::*;",
        "private import SI::*;",
        "private import ScalarValues::*;",
    ]
    strict = check_syntax(
        updated,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert not strict.has_errors, strict.short_summary()
    assert not strict.warnings


def test_terminal_compiler_does_not_duplicate_sufficient_member_imports():
    model = """package P {
        private import ScalarValues::Boolean;
        part def Controller {
            attribute enabled : Boolean = true;
        }
        part controller : Controller;
    }"""

    updated, report = materialize_standard_library_imports(model)

    assert updated == model
    assert report["added_imports"] == []


def test_typed_plan_rejects_multiple_drivers_before_sysml_generation():
    payload = {
        **_PAYLOAD,
        "components": [
            *_PAYLOAD["components"],
            {
                "name": "Backup",
                "responsibility": "Also produces status.",
                "requirements": [],
                "ports": [{
                    "name": "status",
                    "direction": "out",
                    "type": "DataPort",
                    "external": False,
                }],
            },
        ],
        "connections": [
            *_PAYLOAD["connections"],
            {
                "source": {"component": "Backup", "port": "status"},
                "target": {"component": "Consumer", "port": "status"},
                "item_type": "DataPort",
            },
        ],
    }
    plan = ModelGenerationPlan.from_payload(payload)

    assert plan.status == "INVALID"
    assert any("more than one planned driver" in item for item in plan.issues)


def test_step1_plan_rejects_owner_port_as_accept_event_classifier():
    payload = copy.deepcopy(_PAYLOAD)
    payload["schema_version"] = "9.0"
    payload["behaviors"] = [{
        "owner": "Consumer",
        "behavior_id": "ConsumerBehavior",
        "initial_state": "idle",
        "states": [
            {"state_id": "idle", "role": "INITIAL"},
            {
                "state_id": "responding",
                "role": "RESPONSE",
                "entry_action": "respond",
            },
        ],
        "transitions": [{
            "transition_id": "receive",
            "source": "idle",
            "target": "responding",
            "trigger_kind": "ACCEPT",
            "trigger": "status",
        }],
        "provenance": {"kind": "DESIGN_DECISION"},
    }]

    plan = ModelGenerationPlan.from_payload(payload)

    assert plan.status == "INVALID"
    assert any(
        "not an owner port usage" in issue for issue in plan.issues
    )


def test_typed_plan_rejects_item_type_that_disagrees_with_endpoints():
    payload = {
        **_PAYLOAD,
        "connections": [{
            **_PAYLOAD["connections"][0],
            "item_type": "WrongPort",
        }],
    }

    plan = ModelGenerationPlan.from_payload(payload)

    assert plan.status == "INVALID"
    assert any("does not match endpoint type" in item for item in plan.issues)


def test_terminal_conformance_rejects_unplanned_ports_and_connections():
    plan = ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ_FUNC_001: propagate status."],
    )
    model = """package P {
        port def DataPort;
        part def Producer {
            out port status : DataPort;
            out port invented : DataPort;
        }
        part def Consumer {
            in port status : DataPort;
            in port invented : DataPort;
        }
        part producer : Producer;
        part consumer : Consumer;
        connect producer.status to consumer.status;
        connect producer.invented to consumer.invented;
    }"""

    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "FAIL"
    assert len(report["unplanned_ports"]) == 2
    assert report["unplanned_connections"] == [
        "producer.invented -> consumer.invented"
    ]


def test_step2_definition_contract_rejects_unplanned_part_definition():
    plan = ModelGenerationPlan.from_payload(_PAYLOAD)
    report = validate_part_definition_fragment(
        """part def Producer {}
        part def Consumer {}
        part def StatusData {}""",
        plan,
    )

    assert report["status"] == "FAIL"
    assert report["unplanned_part_definitions"] == ["StatusData"]


def test_plan_rejects_port_attribute_member_name_collision():
    payload = copy.deepcopy(_PAYLOAD)
    payload["components"][0]["attributes"] = [{
        "name": "status",
        "value_type": "Boolean",
        "unit": "1",
    }]

    plan = ModelGenerationPlan.from_payload(payload)

    assert plan.status == "INVALID"
    assert (
        "Producer direct member 'status' cannot be both a port "
        "and an attribute"
    ) in plan.issues


def test_plan_allows_same_feature_name_on_different_components():
    assert ModelGenerationPlan.from_payload(_PAYLOAD).status == "PASS"


def test_terminal_conformance_rejects_unplanned_non_container_part_def():
    plan = ModelGenerationPlan.from_payload(_PAYLOAD)
    model = """package P {
        port def DataPort;
        part def Producer { out port status : DataPort; }
        part def Consumer { in port status : DataPort; }
        part def StatusData { attribute value : Real; }
        part producer : Producer;
        part consumer : Consumer;
        connect producer.status to consumer.status;
    }"""

    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "FAIL"
    assert report["definition_contract"][
        "unplanned_part_definitions"
    ] == ["StatusData"]


def test_external_plan_port_cannot_be_silently_internalized():
    payload = {
        "components": [
            {
                "name": "InternalSource",
                "responsibility": "Produces a request.",
                "requirements": ["REQ_FUNC_001"],
                "ports": [{
                    "name": "request",
                    "direction": "out",
                    "type": "DataPort",
                    "external": True,
                }],
            },
            {
                "name": "Gateway",
                "responsibility": "Accepts an external request.",
                "requirements": ["REQ_FUNC_001"],
                "ports": [{
                    "name": "request",
                    "direction": "in",
                    "type": "DataPort",
                    "external": True,
                }],
            },
        ],
        "connections": [{
            "source": {
                "component": "InternalSource",
                "port": "request",
            },
            "target": {"component": "Gateway", "port": "request"},
            "item_type": "DataPort",
            "requirements": ["REQ_FUNC_001"],
        }],
    }
    plan = ModelGenerationPlan.from_payload(payload)
    model = """package P {
        port def DataPort;
        part def InternalSource { out port request : DataPort; }
        part def Gateway { in port request : DataPort; }
        part internalSource : InternalSource;
        part gateway : Gateway;
        connect internalSource.request to gateway.request;
    }"""

    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "FAIL"
    assert report["internalized_external_ports"] == [
        "internalSource.request drives an internal connection",
        "gateway.request receives an internal connection",
    ]


def test_missing_planned_port_and_connection_are_restored_from_plan_only():
    plan = ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ_FUNC_001: propagate status."],
    )
    incomplete = """package P {
        port def DataPort;
        part def Producer { out port status : DataPort; }
        part def Consumer { }
        part producer : Producer;
        part consumer : Consumer;
    }"""

    repaired, report = apply_generation_plan(incomplete, plan)

    assert report["status"] == "PASS"
    assert report["deterministically_added_ports"] == [
        "Consumer.status (in:DataPort)"
    ]
    assert report["deterministically_added_connections"] == [
        "connect producer.status to consumer.status;"
    ]
    assert "in port status : DataPort;" in repaired
    assert "connect producer.status to consumer.status;" in repaired


def test_terminal_qualification_is_independent_of_continuous_score():
    model_text = """package P {
        requirement def REQ_FUNC_001 { doc /* propagate status */ }
        part def Producer {
            satisfy requirement REQ_FUNC_001;
        }
    }"""
    digest = "same"
    qualification = build_model_qualification(
        model_text=model_text,
        requirements=["REQ_FUNC_001: propagate status."],
        syntax_result=SimpleNamespace(total_errors=lambda: 0),
        simulation_result=SimulationResult(model_name="P"),
        terminal_consistency={
            "status": "PASS",
            "model_digest": digest,
            "simulation_source_model_digest": digest,
            "evaluation_source_model_digest": digest,
        },
        generation_plan_conformance={"status": "PASS"},
        ag_contract_graph={"verdict": "FAIL", "checker_version": "test"},
        pattern_conformance_report={"verdict": "PASS"},
    )

    assert qualification["status"] == "NOT_QUALIFIED"
    assert "BOUNDED_A_G_ASSURANCE" in qualification["failed_checks"]


def test_expected_generation_plan_is_fail_closed_when_metadata_is_lost():
    model_text = """package P {
        requirement def REQ_FUNC_001 { doc /* propagate status */ }
        part def Producer { satisfy requirement REQ_FUNC_001; }
    }"""
    qualification = build_model_qualification(
        model_text=model_text,
        requirements=["REQ_FUNC_001: propagate status."],
        syntax_result=SimpleNamespace(total_errors=lambda: 0),
        simulation_result=SimulationResult(model_name="P"),
        terminal_consistency={
            "status": "PASS",
            "model_digest": "same",
            "simulation_source_model_digest": "same",
            "evaluation_source_model_digest": "same",
        },
        generation_plan_expected=True,
    )

    assert qualification["status"] == "NOT_QUALIFIED"
    assert qualification["failed_checks"] == [
        "TYPED_GENERATION_PLAN_CONFORMANCE",
        "REQUIREMENT_STRUCTURAL_OBLIGATIONS",
    ]
