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


def test_planned_port_retype_materialises_the_planned_definition():
    """Enforcing a planned port type must not create a dangling reference.
    Measured before this existed: retyping DataPort -> SensorStatusPort left
    two `No Type named 'SensorStatusPort'` errors on a committed model,
    because the planned type had no definition in the text."""
    from src.prototyping.generation_plan import (
        materialize_planned_port_definitions,
    )

    class _Port:
        def __init__(self, name, port_type):
            self.name = name
            self.direction = "out"
            self.port_type = port_type

    class _Comp:
        def __init__(self, name, ports):
            self.name = name
            self.ports = ports

    class _Conn:
        def __init__(self, sc, sp, tc, tp, item):
            self.source_component = sc
            self.source_port = sp
            self.target_component = tc
            self.target_port = tp
            self.item_type = item

    model = """package P {
        item def SensorStatusData;
        part def Perception { out port sensorStatus : SensorStatusPort; }
        part def Monitor { in port sensorStatus : SensorStatusPort; }
    }"""
    comps = [
        _Comp("Perception", [_Port("sensorStatus", "SensorStatusPort")]),
        _Comp("Monitor", [_Port("sensorStatus", "SensorStatusPort")]),
    ]
    conns = [_Conn("Perception", "sensorStatus", "Monitor", "sensorStatus",
                   "SensorStatusData")]

    updated, added = materialize_planned_port_definitions(model, comps, conns)

    assert added == [
        "port def SensorStatusPort { item payload : SensorStatusData; }"
    ]
    strict = check_syntax(
        updated, fail_closed=True, filter_stdlib_diagnostics=False,
    )
    assert not strict.has_errors, strict.short_summary()

    # idempotent, and an unplanned type is never legalised
    again, more = materialize_planned_port_definitions(updated, comps, conns)
    assert again == updated and more == []
    stray = updated.replace(
        "item def SensorStatusData;",
        "item def SensorStatusData;\n"
        "        part def Rogue { out port x : InventedPort; }",
    )
    _, added2 = materialize_planned_port_definitions(stray, comps, conns)
    assert all("InventedPort" not in line for line in added2)


def test_terminal_compiler_resolves_project_unit_tokens():
    """`deg`, `degC` and `percent` are project tokens, not SI-library names:
    the SI library names the angle unit `degree` and defines no percent unit,
    so `import SI::*` alone leaves `[deg]`/`[degC]`/`[percent]` as reference
    errors (measured on the archived extraction run: qualification failed
    SYSML_SYNTAX_AND_SEMANTICS on exactly those). The import closure must emit
    the aliases and the conversion-defined percent, and the result must be
    clean under the strict, unfiltered check."""
    model = """package P {
        part def Controller {
            attribute heading : Real = 30.0 [deg];
            attribute temp : Real = 5.0 [degC];
            attribute soc : Real = 20.0 [percent];
        }
        part controller : Controller;
    }"""

    updated, report = materialize_standard_library_imports(model)

    assert "alias deg for SI::degree;" in updated
    assert "alias degC for SI::'degree celsius (temperature difference)';" \
        in updated
    assert "attribute percent : DimensionOneUnit" in updated
    assert "private import MeasurementReferences::*;" in updated
    assert report["added_unit_resolutions"]
    strict = check_syntax(
        updated, fail_closed=True, filter_stdlib_diagnostics=False,
    )
    assert not strict.has_errors, strict.short_summary()

    again, second = materialize_standard_library_imports(updated)
    assert again == updated
    assert second["added_unit_resolutions"] == []


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


def test_step1_plan_rejects_event_name_used_as_planned_port_type():
    payload = copy.deepcopy(_PAYLOAD)
    payload["schema_version"] = "9.0"
    for component in payload["components"]:
        component["ports"][0]["type"] = "OverrideCommand"
    payload["connections"][0]["item_type"] = "OverrideCommand"
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
            "trigger": "OverrideCommand",
        }],
        "provenance": {"kind": "DESIGN_DECISION"},
    }]

    plan = ModelGenerationPlan.from_payload(payload)

    assert plan.status == "INVALID"
    assert (
        "planned event OverrideCommand collides with planned port "
        "definition type"
    ) in plan.issues


def test_step1_requires_plan_owned_timed_functional_evidence_chain():
    requirement = (
        "REQ-FUNC-006: The system shall incorporate a revised waypoint "
        "sequence into the active flight plan within 1.0 second of receiving "
        "a valid waypoint-modification command from the GCS."
    )
    payload = copy.deepcopy(_PAYLOAD)
    payload["components"][0]["requirements"] = ["REQ_FUNC_006"]
    payload["connections"][0]["requirements"] = ["REQ_FUNC_006"]

    missing = ModelGenerationPlan.from_payload(
        payload,
        requirements=[requirement],
        require_source_anchored_paths=True,
    )

    assert any(
        "REQ_FUNC_006 timed functional plan requires a source-linked "
        "reachable waypoint response" in issue
        for issue in missing.issues
    )
    assert any(
        "REQ_FUNC_006 timed functional plan requires a source-linked "
        "waypoint latency bound" in issue
        for issue in missing.issues
    )

    payload["components"][0]["attributes"] = [
        {
            "name": "currentWaypointLatency",
            "value_type": "Real",
            "unit": "s",
            "role": "LOCAL_STATE",
            "initial_value": "0.0",
            "provenance": "DESIGN_DECISION",
        },
        {
            "name": "maxWaypointLatency",
            "value_type": "Real",
            "unit": "s",
            "role": "FROZEN_THRESHOLD",
            "initial_value": "1.0",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_FUNC_006",
        },
    ]
    payload["constraints"] = [{
        "constraint_id": "waypointLatencyBound",
        "owner": "Producer",
        "expression": {
            "lhs": "currentWaypointLatency",
            "operator": "<=",
            "rhs": "maxWaypointLatency",
        },
        "activation": {"kind": "ALWAYS"},
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_006",
        },
        "verification_tier": "PARAMETRIC_SWEEP",
    }]
    payload["behaviors"] = [{
        "owner": "Producer",
        "behavior_id": "WaypointUpdateBehavior",
        "initial_state": "waiting",
        "states": [
            {"state_id": "waiting", "role": "INITIAL"},
            {
                "state_id": "updating",
                "role": "RESPONSE",
                "entry_action": "reviseWaypointSequence",
            },
        ],
        "transitions": [{
            "transition_id": "receiveUpdate",
            "source": "waiting",
            "target": "updating",
            "trigger_kind": "ACCEPT",
            "trigger": "ValidWaypointModificationCommand",
        }],
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_006",
        },
    }]

    covered = ModelGenerationPlan.from_payload(
        payload,
        requirements=[requirement],
        require_source_anchored_paths=True,
    )

    assert not any(
        "REQ_FUNC_006 timed functional plan requires" in issue
        for issue in covered.issues
    )


def test_step1_requires_the_response_the_closure_gate_will_demand():
    """The deadlock this prevents: the gate demands an accept-triggered
    navigate response, but only plan-declared symbols are legal accept
    targets, so a plan without the behavior can never be repaired into one."""
    requirement = (
        "REQ-FUNC-001: The drone shall navigate to GPS waypoints with "
        "< 1 m precision"
    )
    payload = copy.deepcopy(_PAYLOAD)

    missing = ModelGenerationPlan.from_payload(
        payload,
        requirements=[requirement],
        require_source_anchored_paths=True,
    )

    assert any(
        "REQ_FUNC_001 needs a planned navigate response" in issue
        for issue in missing.issues
    )

    payload["behaviors"] = [{
        "owner": "Producer",
        "behavior_id": "WaypointNavigationBehavior",
        "initial_state": "idle",
        "states": [
            {"state_id": "idle", "role": "INITIAL"},
            {
                "state_id": "navigating",
                "role": "RESPONSE",
                "entry_action": "navigateToWaypoint",
            },
        ],
        "transitions": [{
            "transition_id": "startNavigation",
            "source": "idle",
            "target": "navigating",
            "trigger_kind": "ACCEPT",
            "trigger": "WaypointMissionAcceptedSignal",
        }],
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_001",
        },
    }]

    covered = ModelGenerationPlan.from_payload(
        payload,
        requirements=[requirement],
        require_source_anchored_paths=True,
    )

    assert not any(
        "REQ_FUNC_001 needs a planned" in issue for issue in covered.issues
    )
    # the trigger the repair would have had to invent is now a legal symbol
    assert "WaypointMissionAcceptedSignal" in {
        symbol.name for symbol in covered.planned_event_symbols
    }


def test_a_response_state_the_plan_cannot_reach_does_not_discharge_the_intent():
    requirement = (
        "REQ-FUNC-001: The drone shall navigate to GPS waypoints with "
        "< 1 m precision"
    )
    payload = copy.deepcopy(_PAYLOAD)
    payload["behaviors"] = [{
        "owner": "Producer",
        "behavior_id": "WaypointNavigationBehavior",
        "initial_state": "idle",
        "states": [
            {"state_id": "idle", "role": "INITIAL"},
            {
                "state_id": "navigating",
                "role": "RESPONSE",
                "entry_action": "navigateToWaypoint",
            },
        ],
        "transitions": [],          # nothing leads to the response state
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_001",
        },
    }]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=[requirement],
        require_source_anchored_paths=True,
    )

    assert any(
        "REQ_FUNC_001 needs a planned navigate response" in issue
        for issue in plan.issues
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


def test_a_planned_port_written_with_the_wrong_type_is_retyped():
    """One measured run failed with four ports reported as BOTH missing and
    unplanned: same component, same name, same direction, different type.

    "Does this port exist?" looked only at the name, so nothing added the port
    and nothing corrected it. The plan owns a port's type exactly as it owns an
    attribute's, so the mismatch is repaired rather than reported twice.
    """
    from types import SimpleNamespace

    from src.prototyping.generation_plan import normalise_planned_port_types

    component = SimpleNamespace(
        name="FlightController",
        ports=(
            SimpleNamespace(
                name="overrideCmd", direction="in", port_type="CommandPort"
            ),
            SimpleNamespace(
                name="telemetry", direction="out", port_type="StatusPort"
            ),
        ),
    )
    model = (
        "package S {\n"
        "    part def FlightController {\n"
        "        in port overrideCmd : DataPort;\n"
        "        out port telemetry : StatusPort;\n"
        "    }\n"
        "}\n"
    )

    text, changes = normalise_planned_port_types(model, [component])

    assert "in port overrideCmd : CommandPort;" in text
    assert "DataPort" not in text
    # the port that already agreed is untouched, and not reported
    assert "out port telemetry : StatusPort;" in text
    assert changes == ["FlightController.overrideCmd (DataPort -> CommandPort)"]


def test_a_wrong_direction_is_left_alone_because_it_is_a_design_question():
    """Retyping is a notation repair. A direction reversal changes what the
    connections mean, so it stays a reported mismatch rather than a silent edit.
    """
    from types import SimpleNamespace

    from src.prototyping.generation_plan import normalise_planned_port_types

    component = SimpleNamespace(
        name="FlightController",
        ports=(SimpleNamespace(
            name="overrideCmd", direction="in", port_type="CommandPort"
        ),),
    )
    model = (
        "package S {\n"
        "    part def FlightController {\n"
        "        out port overrideCmd : DataPort;\n"
        "    }\n"
        "}\n"
    )

    text, changes = normalise_planned_port_types(model, [component])

    assert text == model
    assert changes == []


def test_terminal_conformance_admits_justified_conservative_extension():
    """Extension contract (schema 4.1): a specialization-consistent addition
    carrying an in-body doc /* rationale; satisfies REQ_... */ is a
    JUSTIFIED_EXTENSION, not a violation."""
    plan = ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ_FUNC_001: propagate status."],
    )
    model = """package P {
        port def DataPort;
        part def Producer {
            out port status : DataPort;
            out port extra : DataPort { doc /* status feedback the plan missed; satisfies REQ_FUNC_001 */ }
        }
        part def Consumer {
            in port status : DataPort;
            in port extraIn : DataPort { doc /* receives the declared feedback; satisfies REQ_FUNC_001 */ }
        }
        part producer : Producer;
        part consumer : Consumer;
        connect producer.status to consumer.status;
        connect producer.extra to consumer.extraIn;
    }"""

    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "PASS"
    assert len(report["justified_extension_ports"]) == 2
    assert report["justified_extension_connections"] == [
        "producer.extra -> consumer.extraIn"
    ]
    # raw inventory keeps its meaning; the ISSUES are what got reclassified
    assert len(report["unplanned_ports"]) == 2
    assert not any("unplanned" in issue for issue in report["issues"])


def test_extension_without_requirement_link_still_fails():
    plan = ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ_FUNC_001: propagate status."],
    )
    model = """package P {
        port def DataPort;
        part def Producer {
            out port status : DataPort;
            out port extra : DataPort { doc /* felt like it */ }
        }
        part def Consumer { in port status : DataPort; }
        part producer : Producer;
        part consumer : Consumer;
        connect producer.status to consumer.status;
    }"""

    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "FAIL"
    assert any("unplanned port" in issue for issue in report["issues"])
    assert report["justified_extension_ports"] == []


def test_rewiring_planned_ports_is_not_an_extension():
    plan = ModelGenerationPlan.from_payload(
        _PAYLOAD,
        requirements=["REQ_FUNC_001: propagate status."],
    )
    # an unplanned connection between purely PLANNED ports alters the planned
    # information flow — never justifiable as an extension
    model = """package P {
        port def DataPort;
        part def Producer { out port status : DataPort; }
        part def Consumer {
            in port status : DataPort;
            in port statusIn2 : DataPort { doc /* second sink; satisfies REQ_FUNC_001 */ }
        }
        part producer : Producer;
        part consumer : Consumer;
        connect producer.status to consumer.status;
        connect producer.status to consumer.statusIn2;
    }"""

    _, report = apply_generation_plan(model, plan)
    # the port itself is justified; the connection is too (it serves the
    # declared extension through planned source) — but a connection touching
    # NO extension port must fail:
    model2 = model.replace(
        " { doc /* second sink; satisfies REQ_FUNC_001 */ }", ";"
    )
    _, report2 = apply_generation_plan(model2, plan)
    assert report2["status"] == "FAIL"
    assert any("unplanned connection" in i for i in report2["issues"])


# ---------------------------------------------------------------------------
# Declared (out-of-vocabulary) response intents and the unverifiable record
# ---------------------------------------------------------------------------

_ALERT_REQ = (
    "REQ-FUNC-001: The system shall alert the operators within 5 minutes "
    "of an equipment fault."
)


def _alert_payload() -> dict:
    """A plan whose FUNC requirement obliges a response outside the built-in
    intent table (alert), realized locally with a reachable alert action."""
    payload = copy.deepcopy(_PAYLOAD)
    payload["components"][0]["responsibility"] = (
        "Monitors equipment and alerts the operators."
    )
    payload["requirement_realizations"] = [{
        "requirement_id": "REQ_FUNC_001",
        "realization_kind": "LOCAL_BEHAVIOR",
        "trigger_concept": "an equipment fault",
        "effect_concept": "alert the operators",
        "owner_component": "Producer",
        "behavior_kind": "STATE_DEF",
        "behavior_name": "AlertBehavior",
        "response_intent": "alert",
        "response_markers": ["alert"],
        "connection_path": [],
    }]
    payload["behaviors"] = [{
        "owner": "Producer",
        "behavior_id": "AlertBehavior",
        "initial_state": "monitoring",
        "states": [
            {"state_id": "monitoring", "role": "INITIAL"},
            {
                "state_id": "alerting",
                "role": "RESPONSE",
                "entry_action": "alertOperators",
            },
        ],
        "transitions": [{
            "transition_id": "onFault",
            "source": "monitoring",
            "target": "alerting",
            "trigger_kind": "ACCEPT",
            "trigger": "EquipmentFaultDetected",
        }],
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_001",
        },
    }]
    return payload


def test_declared_intent_with_anchored_markers_and_reachable_response_passes():
    plan = ModelGenerationPlan.from_payload(
        _alert_payload(),
        requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )
    assert not plan.issues


def test_out_of_vocabulary_intent_without_markers_is_a_plan_defect():
    payload = _alert_payload()
    payload["requirement_realizations"][0]["response_markers"] = []

    plan = ModelGenerationPlan.from_payload(
        payload, requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )

    assert any(
        "REQ_FUNC_001 response_intent 'alert'" in issue
        and "declares no response_markers" in issue
        for issue in plan.issues
    )


def test_a_marker_not_anchored_in_the_effect_phrase_is_refused():
    """The anti-self-grading rule: the planner cannot declare a marker its
    own behaviours happen to satisfy unless the requirement's effect phrase
    names it."""
    payload = _alert_payload()
    payload["requirement_realizations"][0]["response_markers"] = ["hovering"]

    plan = ModelGenerationPlan.from_payload(
        payload, requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )

    assert any(
        "share no content word with the copied effect phrase" in issue
        for issue in plan.issues
    )


def test_declared_intent_still_demands_a_reachable_declared_response():
    """Declared markers buy checkability, not a pass: the plan must still
    carry a reachable state producing the declared response."""
    payload = _alert_payload()
    payload["behaviors"][0]["states"][1] = {
        "state_id": "alerting", "role": "RESPONSE",
    }  # response state no longer produces the alert action

    plan = ModelGenerationPlan.from_payload(
        payload, requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )

    assert any(
        "REQ_FUNC_001 needs a planned alert response" in issue
        for issue in plan.issues
    )


def test_unverifiable_requires_a_rationale_and_then_obliges_no_behavior():
    payload = _alert_payload()
    realization = payload["requirement_realizations"][0]
    realization["response_intent"] = "unverifiable"
    realization["response_markers"] = []
    payload["behaviors"] = []

    missing = ModelGenerationPlan.from_payload(
        payload, requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )
    assert any(
        "records response_intent 'unverifiable' without a "
        "response_intent_rationale" in issue
        for issue in missing.issues
    )

    realization["response_intent_rationale"] = (
        "the alert deadline is a latency property no reachable-action "
        "marker can evidence"
    )
    reasoned = ModelGenerationPlan.from_payload(
        payload, requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )
    assert not any("response_intent" in issue for issue in reasoned.issues)
    assert not any("needs a planned" in issue for issue in reasoned.issues)


def test_response_markers_survive_the_plan_serialisation_round_trip():
    plan = ModelGenerationPlan.from_payload(
        _alert_payload(), requirements=[_ALERT_REQ],
        require_source_anchored_paths=True,
    )
    realization = plan.requirement_realizations[0].to_dict()
    assert realization["response_markers"] == ["alert"]


# ─────────────────────────────────────────────────────────────────────────────
#  strip_unplanned_additions — the conformance-scoped salvage knife
# ─────────────────────────────────────────────────────────────────────────────

def test_strip_unplanned_additions_removes_each_additive_class():
    from src.prototyping.generation_plan import strip_unplanned_additions

    text = """package P {
    port def SignalPort;
    part def Source {
        out port signal : SignalPort;
        out port invented : SignalPort;
        assert constraint altitudeBound { maxAltitude <= 120.0 }
    }
    part def Sink { in port signal : SignalPort; }
    part def Gadget {
        in port stuff : SignalPort;
    }
    part source : Source;
    part sink : Sink;
    part gadget : Gadget;
    connect source.signal to sink.signal;
    connect source.invented to gadget.stuff;
}"""
    stripped, removed = strip_unplanned_additions(text, {
        "connections": [["source", "invented", "gadget", "stuff"]],
        "ports": [["Source", "invented", "out", "SignalPort"]],
        "component_usages": [["gadget", "Gadget"]],
        "part_definitions": ["Gadget"],
    })

    assert len(removed) == 4, removed
    assert "invented" not in stripped
    assert "Gadget" not in stripped
    assert "gadget" not in stripped
    # the in-plan edits survive intact
    assert "out port signal : SignalPort" in stripped
    assert "assert constraint altitudeBound" in stripped
    assert "connect source.signal to sink.signal" in stripped
    # no dangling header remains from the part-def deletion
    assert "part def\n" not in stripped and "part def ;" not in stripped


def test_strip_sweeps_connects_referencing_a_dropped_usage():
    from src.prototyping.generation_plan import strip_unplanned_additions

    text = """package P {
    part def A { out port p : X; }
    part def B { in port q : X; }
    part a : A;
    part b : B;
    connect a.p to b.q;
}"""
    stripped, removed = strip_unplanned_additions(text, {
        "component_usages": [["b", "B"]],
    })

    assert "part b : B" not in stripped
    assert "connect a.p to b.q" not in stripped
    assert any("dropped usage b" in item for item in removed)


def test_conformance_report_carries_salvage_targets():
    payload = {
        "components": [
            {
                "name": "Source",
                "responsibility": "Produces a signal.",
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
                "responsibility": "Consumes a signal.",
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
    plan = ModelGenerationPlan.from_payload(
        payload, requirements=["REQ_FUNC_001: propagate signal"],
    )
    model = """package P {
        port def SignalPort;
        part def Source {
            out port signal : SignalPort;
            out port invented : SignalPort;
        }
        part def Sink { in port signal : SignalPort; }
        part source : Source;
        part sink : Sink;
        connect source.signal to sink.signal;
    }"""
    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "FAIL"
    targets = report["salvage_targets"]
    assert ["Source", "invented", "out", "SignalPort"] in targets["ports"]
    # and the salvage round-trips to PASS
    from src.prototyping.generation_plan import strip_unplanned_additions
    stripped, removed = strip_unplanned_additions(model, targets)
    assert removed
    _, recheck = apply_generation_plan(stripped, plan)
    assert recheck["status"] == "PASS", recheck["issues"]
