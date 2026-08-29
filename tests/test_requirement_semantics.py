from src.prototyping.activated_constraint_plan import (
    ConstraintPlanningContext,
    compile_constraint_plan,
)
from src.prototyping.generation_plan import (
    ComponentPlan,
    ModelGenerationPlan,
    PortPlan,
)
from src.prototyping.requirement_semantics import (
    NUMERIC_INVARIANT,
    RequirementSemanticObligation,
    compile_requirement_semantic_obligations,
)


_REQUIREMENT = (
    "REQ-FUNC-002: The system shall detect a stationary obstacle directly "
    "ahead within the forward sensor field of view and maintain at least "
    "5 metres of separation while avoiding it."
)


def _payload():
    return {
        "components": [
            {
                "name": "Perception",
                "responsibility": "Measures obstacle separation.",
                "requirements": ["REQ_FUNC_002"],
                "ports": [{
                    "name": "obstacleData",
                    "direction": "out",
                    "type": "ObstaclePort",
                }],
            },
            {
                "name": "Controller",
                "responsibility": "Maintains obstacle separation.",
                "requirements": ["REQ_FUNC_002"],
                "ports": [{
                    "name": "obstacleData",
                    "direction": "in",
                    "type": "ObstaclePort",
                }],
            },
        ],
        "connections": [{
            "source": {
                "component": "Perception",
                "port": "obstacleData",
            },
            "target": {
                "component": "Controller",
                "port": "obstacleData",
            },
            "item_type": "ObstaclePort",
            "requirements": ["REQ_FUNC_002"],
        }],
        "semantic_bindings": [{
            "obligation_id": "SEM_REQ_FUNC_002_001",
            "requirement_id": "REQ_FUNC_002",
            "source": {
                "component": "Perception",
                "port": "obstacleData",
            },
            "target": {
                "component": "Controller",
                "port": "obstacleData",
                "runtime_attribute": "currentSeparation",
            },
            "payload": {
                "port_type": "ObstaclePort",
                "port_feature": "payload",
                "item_type": "ObstacleData",
                "item_feature": "separation",
                "value_type": "Real",
                "unit": "m",
            },
            "constraint": {
                "name": "keepSeparation",
                "threshold_attribute": "minimumSeparation",
            },
        }],
        "constraints": [{
            "constraint_id": "maintainSeparationConstraint",
            "owner": "Controller",
            "expression": {
                "lhs": "currentSeparation",
                "operator": ">=",
                "rhs": "minimumSeparation",
            },
            "activation": {
                "kind": "STATE_ACTIVE",
                "state": "FlightBehavior::AvoidingObstacle",
            },
            "provenance": {
                "kind": "FROZEN_REQUIREMENT",
                "requirement_id": "REQ_FUNC_002",
            },
            "verification_tier": "STATE_EXECUTION",
        }],
        "behaviors": [{
            "owner": "Controller",
            "behavior_id": "FlightBehavior",
            "initial_state": "NominalFlight",
            "states": [
                {
                    "state_id": "NominalFlight",
                    "role": "INITIAL",
                },
                {
                    "state_id": "AvoidingObstacle",
                    "role": "RESPONSE",
                    "entry_action": "executeObstacleAvoidance",
                },
            ],
            "transitions": [{
                "transition_id": "detectObstacle",
                "source": "NominalFlight",
                "target": "AvoidingObstacle",
                "trigger_kind": "ACCEPT",
                "trigger": "ObstacleDetectedSignal",
            }],
            "provenance": {
                "kind": "FROZEN_REQUIREMENT",
                "requirement_id": "REQ_FUNC_002",
            },
        }],
    }


def test_explicit_lower_bound_compiles_from_frozen_source():
    obligations = compile_requirement_semantic_obligations((_REQUIREMENT,))

    assert len(obligations) == 1
    obligation = obligations[0]
    assert obligation.obligation_id == "SEM_REQ_FUNC_002_001"
    assert obligation.requirement_id == "REQ_FUNC_002"
    assert obligation.kind == NUMERIC_INVARIANT
    assert obligation.subject_terms == ("separation",)
    assert obligation.operator == ">="
    assert obligation.threshold == 5.0
    assert obligation.unit == "m"
    assert obligation.activation_kind == "CONTEXTUAL"
    assert obligation.activation_clause == "while avoiding it"
    assert obligation.source_clause.endswith("while avoiding it")
    assert len(obligation.source_digest) == 64


def test_non_matching_or_timed_requirement_is_not_overinterpreted():
    requirements = (
        "REQ-SAFE-005: deploy within 0.5 seconds after failure.",
        "REQ-FUNC-003: carry a payload.",
    )

    assert compile_requirement_semantic_obligations(requirements) == ()


def test_constraint_compiler_owns_binding_reconciliation_and_attributes():
    producer_port = PortPlan("obstacleData", "out", "ObstaclePort")
    controller_port = PortPlan("obstacleData", "in", "ObstaclePort")
    components = (
        ComponentPlan(
            "Perception",
            "Measures obstacle separation.",
            ("REQ_FUNC_002",),
            (producer_port,),
        ),
        ComponentPlan(
            "Controller",
            "Maintains obstacle separation.",
            ("REQ_FUNC_002",),
            (controller_port,),
        ),
    )

    compiled = compile_constraint_plan(
        _payload(),
        requirements=(_REQUIREMENT,),
        context=ConstraintPlanningContext(
            components=components,
            port_lookup={
                ("Perception", "obstacleData"): producer_port,
                ("Controller", "obstacleData"): controller_port,
            },
            connection_keys=frozenset({(
                "Perception",
                "obstacleData",
                "Controller",
                "obstacleData",
            )}),
            allocated_component_requirements=frozenset({
                ("Perception", "REQ_FUNC_002"),
                ("Controller", "REQ_FUNC_002"),
            }),
        ),
    )

    assert compiled.issues == ()
    assert compiled.semantic_bindings[0].constraint_name == (
        "maintainSeparationConstraint"
    )
    assert compiled.identity_reconciliations == (
        "Controller.keepSeparation -> maintainSeparationConstraint "
        "(SEM_REQ_FUNC_002_001)",
    )
    controller = next(
        item for item in compiled.components if item.name == "Controller"
    )
    assert {item.name for item in controller.attributes} == {
        "currentSeparation",
        "minimumSeparation",
    }


def test_plan_carries_and_round_trips_frozen_semantic_obligation():
    plan = ModelGenerationPlan.from_payload(
        _payload(),
        requirements=(_REQUIREMENT,),
    )

    assert plan.status == "PASS"
    assert plan.schema_version == "9.0"
    assert len(plan.semantic_obligations) == 1
    assert len(plan.semantic_bindings) == 1
    assert len(plan.constraint_plans) == 1
    assert (
        plan.semantic_bindings[0].constraint_name
        == "maintainSeparationConstraint"
    )
    assert plan.constraint_identity_reconciliations == (
        "Controller.keepSeparation -> maintainSeparationConstraint "
        "(SEM_REQ_FUNC_002_001)",
    )
    restored = ModelGenerationPlan.from_dict(plan.to_dict())
    assert restored.semantic_obligations == plan.semantic_obligations
    assert restored.semantic_bindings == plan.semantic_bindings
    assert restored.constraint_plans == plan.constraint_plans
    assert restored.to_dict()["semantic_obligations"][0][
        "claim_boundary"
    ] == "MODEL_SEMANTIC_FIDELITY_NOT_PHYSICAL_PROOF"


def test_archived_schema7_plan_without_behavior_ir_remains_readable():
    payload = _payload()
    payload.pop("behaviors")
    payload["schema_version"] = "7.0"

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=(_REQUIREMENT,),
        source="ARCHIVED",
    )

    assert plan.status == "PASS"
    assert plan.schema_version == "7.0"
    assert plan.planned_behaviors == ()


def test_semantic_obligation_deserialisation_preserves_source_facts():
    original = compile_requirement_semantic_obligations((_REQUIREMENT,))[0]

    restored = RequirementSemanticObligation.from_dict(original.to_dict())

    assert restored == original


def test_plan_rejects_semantic_obligation_without_typed_binding():
    payload = _payload()
    payload["semantic_bindings"] = []

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=(_REQUIREMENT,),
    )

    assert plan.status == "INVALID"
    assert any(
        "has no typed semantic binding" in issue
        for issue in plan.issues
    )


def test_plan_rejects_generic_port_for_measured_semantic_binding():
    payload = _payload()
    for component in payload["components"]:
        component["ports"][0]["type"] = "DataPort"
    payload["connections"][0]["item_type"] = "DataPort"
    payload["semantic_bindings"][0]["payload"]["port_type"] = "DataPort"

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=(_REQUIREMENT,),
    )

    assert plan.status == "INVALID"
    assert any("not generic DataPort" in issue for issue in plan.issues)


def test_plan_rejects_definition_kind_name_collision():
    payload = _payload()
    payload["semantic_bindings"][0]["payload"]["item_type"] = "Controller"

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=(_REQUIREMENT,),
    )

    assert plan.status == "INVALID"
    assert any(
        "item_type collides with planned component Controller" in issue
        for issue in plan.issues
    )


def test_semantic_realization_must_end_at_constraint_owner_not_actuator():
    payload = _payload()
    payload["components"][1]["ports"].append({
        "name": "flightCommand",
        "direction": "out",
        "type": "FlightCommandPort",
    })
    payload["components"].append({
        "name": "PropulsionSystem",
        "responsibility": "Executes flight commands.",
        "requirements": ["REQ_FUNC_002"],
        "ports": [{
            "name": "flightCommand",
            "direction": "in",
            "type": "FlightCommandPort",
        }],
    })
    payload["connections"].append({
        "source": {
            "component": "Controller",
            "port": "flightCommand",
        },
        "target": {
            "component": "PropulsionSystem",
            "port": "flightCommand",
        },
        "item_type": "FlightCommandPort",
        "requirements": ["REQ_FUNC_002"],
    })
    payload["requirement_realizations"] = [{
        "requirement_id": "REQ_FUNC_002",
        "realization_kind": "CAUSAL_PATH",
        "trigger_concept": "a stationary obstacle directly ahead",
        "effect_concept": (
            "maintain at least 5 metres of separation while avoiding it"
        ),
        "connection_path": [
            {
                "source_component": "Perception",
                "source_port": "obstacleData",
                "target_component": "Controller",
                "target_port": "obstacleData",
                "item_type": "ObstaclePort",
            },
            {
                "source_component": "Controller",
                "source_port": "flightCommand",
                "target_component": "PropulsionSystem",
                "target_port": "flightCommand",
                "item_type": "FlightCommandPort",
            },
        ],
    }]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=(_REQUIREMENT,),
        require_source_anchored_paths=True,
    )

    assert plan.status == "INVALID"
    assert (
        "source-derived semantic realization REQ_FUNC_002 must end at "
        "semantic binding target Controller.obstacleData, found "
        "PropulsionSystem.flightCommand"
    ) in plan.issues


def test_an_existing_declaration_is_rebound_not_duplicated():
    """Measured twice: `attribute currentSeparation : LengthValue [m] = ...`.

    The existence check required an initializer and did not allow a unit suffix
    after the type, so it could not see a declaration written either way and
    appended a second one — a duplicate in the same part def, which
    USER_NAMESPACE_INTEGRITY and Syside's namespace-distinguishability warning
    both correctly rejected. This is the same blind spot that
    `activated_constraint_plan` had; two independent materialisers write these
    attributes, and fixing one left the other.
    """
    import re

    def existing(name: str, body: str):
        return re.search(
            rf"\battribute\s+{re.escape(name)}"
            rf"(?:\s*:\s*[A-Za-z_][\w:]*(?:\s*\[[^\]{{}}]*\])?)?"
            rf"(?:\s*=\s*[^;{{}}]+)?\s*;",
            body,
        )

    written_forms = [
        "attribute currentSeparation : LengthValue = obstacle.payload.sep;",
        "attribute currentSeparation : LengthValue [m] = obstacle.payload.sep;",
        "attribute currentSeparation : LengthValue;",
        "attribute currentSeparation;",
    ]
    for form in written_forms:
        assert existing("currentSeparation", f"    {form}\n") is not None, (
            f"an existing declaration written as {form!r} must be seen, "
            "or it is appended a second time"
        )
    # and a different attribute is still not mistaken for this one
    assert existing(
        "currentSeparation", "    attribute minSeparationThreshold : Real = 5;\n"
    ) is None


# ---------------------------------------------------------------------------
# Compiler coverage on the flagship requirement style (ablation pilot finding:
# the base-form maintain/keep/ensure pattern compiled ZERO obligations from
# all 29 drone_v2 requirements, so the semantic-fidelity layer ran vacuously
# on the main experiment input).
# ---------------------------------------------------------------------------


def test_inflected_verbs_and_within_compile():
    obligations = compile_requirement_semantic_obligations([
        "REQ-PERF-001: The system shall maintain roll and pitch attitude "
        "deviations within 0.5 degree RMS during steady cruise flight.",
        "REQ-FUNC-002: ..., maintaining an airframe-to-obstacle separation "
        "of at least 5 metres.",
    ])
    by_req = {item.requirement_id: item for item in obligations}
    attitude = by_req["REQ_PERF_001"]
    assert (attitude.operator, attitude.threshold, attitude.unit) == ("<=", 0.5, "deg")
    assert attitude.activation_kind == "CONTEXTUAL"
    assert "and" not in attitude.subject_terms
    separation = by_req["REQ_FUNC_002"]
    assert (separation.operator, separation.threshold, separation.unit) == (">=", 5.0, "m")
    assert "to" not in separation.subject_terms


def test_embedded_minmax_and_not_exceed_compile():
    obligations = compile_requirement_semantic_obligations([
        "REQ-PERF-004: The system shall maintain a minimum forward ground "
        "speed of 2 m/s when operating in sustained headwinds.",
        "REQ-CONS-001: The system shall not exceed a flight altitude of "
        "120 metres above ground level at any point.",
        "REQ-CONS-003: The system maximum take-off mass, including payload "
        "and battery, shall not exceed 8.0 kg.",
    ])
    by_req = {item.requirement_id: item for item in obligations}
    speed = by_req["REQ_PERF_004"]
    assert (speed.operator, speed.threshold, speed.unit) == (">=", 2.0, "m/s")
    altitude = by_req["REQ_CONS_001"]
    assert (altitude.operator, altitude.threshold, altitude.unit) == ("<=", 120.0, "m")
    assert set(altitude.subject_terms) == {"flight", "altitude"}
    mass = by_req["REQ_CONS_003"]
    assert (mass.operator, mass.threshold, mass.unit) == ("<=", 8.0, "kg")
    assert "mass" in mass.subject_terms


def test_triggers_and_scenario_envelopes_still_compile_to_nothing():
    obligations = compile_requirement_semantic_obligations([
        # Scenario envelope, no bound verb — NOT an invariant on the system.
        "REQ-FUNC-002: When approaching a threat at a closing speed no "
        "greater than 1.5 m/s, the system shall execute a manoeuvre.",
        # Battery trigger threshold, not a maintained bound.
        "REQ-SAFE-001: The system shall initiate return-to-base when the "
        "battery state-of-charge reaches 25%.",
        # Event-driven response latency, deliberately out of scope.
        "REQ-SAFE-005: The system shall deploy the parachute within 0.5 "
        "seconds of detecting a critical propulsion failure.",
    ])
    assert obligations == ()


def test_flagship_drone_v2_requirements_compile_nonzero():
    import sys
    from pathlib import Path
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "examples")
    )
    from drone_system_v2 import DRONE_REQUIREMENTS

    obligations = compile_requirement_semantic_obligations(DRONE_REQUIREMENTS)
    compiled_requirements = {item.requirement_id for item in obligations}
    # The exact set the flagship input supports today; extending coverage may
    # grow it, but silently regressing to vacuous must fail loudly.
    assert compiled_requirements >= {
        "REQ_FUNC_002", "REQ_FUNC_003", "REQ_PERF_001", "REQ_PERF_002",
        "REQ_PERF_003", "REQ_PERF_004", "REQ_PERF_006",
        "REQ_CONS_001", "REQ_CONS_003",
    }
    from src.prototyping.requirement_semantics import quantity_type_for_unit
    for item in obligations:
        assert quantity_type_for_unit(item.unit) is not None, (
            f"{item.obligation_id}: unit {item.unit!r} lacks the "
            "quantity-type mapping binding validation requires"
        )
