from types import SimpleNamespace

from src.prototyping.generation_plan import (
    ModelGenerationPlan,
    apply_generation_plan,
)
from src.prototyping.activated_constraint_plan import (
    ConstraintPlan,
    materialize_planned_constraints,
)
from src.prototyping.model_qualification import build_model_qualification
from src.simulation.behavioral_sim import (
    BehavioralScenarioResult,
    BehavioralSimResult,
    run_behavioral_simulation,
)
from src.simulation.constraint_checker import extract_constraints
from src.simulation.syntax_checker import check_syntax
from src.simulation.validator import SimulationResult


def _payload(*, attributes=(), constraints=(), behaviors=()):
    return {
        "components": [
            {
                "name": "Producer",
                "responsibility": "Produces a signal.",
                "requirements": ["REQ_SAFE_005"],
                "ports": [{
                    "name": "signal",
                    "direction": "out",
                    "type": "DataPort",
                    "external": False,
                }],
            },
            {
                "name": "Controller",
                "responsibility": "Consumes and checks the signal.",
                "requirements": ["REQ_SAFE_005"],
                "ports": [{
                    "name": "signal",
                    "direction": "in",
                    "type": "DataPort",
                    "external": False,
                }],
                "attributes": list(attributes),
            },
        ],
        "connections": [{
            "source": {"component": "Producer", "port": "signal"},
            "target": {"component": "Controller", "port": "signal"},
            "item_type": "DataPort",
            "requirements": ["REQ_SAFE_005"],
        }],
        "constraints": list(constraints),
        "behaviors": list(behaviors),
    }


def _flight_behavior(
    *,
    trigger_kind="ACCEPT",
    trigger="EngageCruiseSignal",
    entry_action="maintainCruise",
):
    return {
        "owner": "Controller",
        "behavior_id": "FlightBehavior",
        "initial_state": "Ground",
        "states": [
            {"state_id": "Ground", "role": "INITIAL"},
            {
                "state_id": "Cruise",
                "role": "RESPONSE",
                "entry_action": entry_action,
            },
        ],
        "transitions": [{
            "transition_id": "engage",
            "source": "Ground",
            "target": "Cruise",
            "trigger_kind": trigger_kind,
            "trigger": trigger,
        }],
        "provenance": {"kind": "DESIGN_DECISION"},
    }


def _model(controller_body: str) -> str:
    return f"""package P {{
        port def DataPort;
        part def Producer {{ out port signal : DataPort; }}
        part def Controller {{
            in port signal : DataPort;
{controller_body}
        }}
        part producer : Producer;
        part controller : Controller;
        connect producer.signal to controller.signal;
    }}"""


def test_frozen_constraint_cannot_invent_a_500_newton_threshold():
    requirement = (
        "REQ-SAFE-005: The recovery system shall deploy within "
        "0.5 seconds after a critical propulsion failure."
    )
    attributes = [
        {
            "name": "currentDeploymentForce",
            "value_type": "Real",
            "unit": "N",
            "role": "LOCAL_STATE",
            "initial_value": "0 [N]",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_SAFE_005",
        },
        {
            "name": "minDeploymentForce",
            "value_type": "Real",
            "unit": "N",
            "role": "FROZEN_THRESHOLD",
            "initial_value": "500 [N]",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_SAFE_005",
        },
    ]
    constraints = [{
        "constraint_id": "deploymentForceMin",
        "owner": "Controller",
        "expression": {
            "lhs": "currentDeploymentForce",
            "operator": ">=",
            "rhs": "minDeploymentForce",
        },
        "activation": {"kind": "ALWAYS"},
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_SAFE_005",
        },
        "verification_tier": "PARAMETRIC_SWEEP",
    }]

    plan = ModelGenerationPlan.from_payload(
        _payload(attributes=attributes, constraints=constraints),
        requirements=[requirement],
    )

    assert plan.status == "INVALID"
    assert any(
        "numeric bound is absent from frozen source" in item
        for item in plan.issues
    )
    assert any(
        "unit 'N' is absent from frozen source" in item
        for item in plan.issues
    )


def test_requirement_identifier_digits_are_not_accepted_as_a_bound():
    attributes = [
        {
            "name": "currentDelay",
            "unit": "s",
            "role": "LOCAL_STATE",
            "initial_value": "0",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_SAFE_005",
        },
        {
            "name": "maxDelay",
            "unit": "s",
            "role": "FROZEN_THRESHOLD",
            "initial_value": "5",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_SAFE_005",
        },
    ]
    constraints = [{
        "constraint_id": "delayBound",
        "owner": "Controller",
        "expression": {
            "lhs": "currentDelay",
            "operator": "<=",
            "rhs": "maxDelay",
        },
        "activation": {"kind": "ALWAYS"},
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_SAFE_005",
        },
        "verification_tier": "PARAMETRIC_SWEEP",
    }]
    plan = ModelGenerationPlan.from_payload(
        _payload(attributes=attributes, constraints=constraints),
        requirements=[
            "REQ-SAFE-005: deploy within 0.5 seconds after failure."
        ],
    )

    assert any(
        "numeric bound is absent from frozen source" in item
        for item in plan.issues
    )


def test_state_active_property_is_serialized_inside_owning_state():
    attributes = [
        {
            "name": "currentGroundSpeed",
            "initial_value": "0",
            "role": "LOCAL_STATE",
            "provenance": "DESIGN_DECISION",
        },
        {
            "name": "minGroundSpeed",
            "initial_value": "5",
            "role": "DESIGN_PARAMETER",
            "provenance": "DESIGN_DECISION",
        },
    ]
    constraints = [{
        "constraint_id": "groundSpeedMin",
        "owner": "Controller",
        "expression": {
            "lhs": "currentGroundSpeed",
            "operator": ">=",
            "rhs": "minGroundSpeed",
        },
        "activation": {
            "kind": "STATE_ACTIVE",
            "state": "FlightBehavior::Cruise",
        },
        "provenance": {"kind": "DESIGN_DECISION"},
        "verification_tier": "STATE_EXECUTION",
    }]
    plan = ModelGenerationPlan.from_payload(
        _payload(
            attributes=attributes,
            constraints=constraints,
            behaviors=[_flight_behavior()],
        )
    )
    model = _model("""
            state def FlightBehavior {
                state Ground;
                state Cruise;
                transition initial then Cruise;
            }
            assert constraint groundSpeedMin {
                currentGroundSpeed >= minGroundSpeed
            }""")

    updated, report = apply_generation_plan(model, plan)

    assert report["status"] == "PASS"
    assert updated.count("assert constraint groundSpeedMin") == 1
    cruise_body = updated.split("state Cruise {", 1)[1].split("}", 1)[0]
    assert "assert constraint groundSpeedMin" in cruise_body
    parsed = next(
        item for item in extract_constraints(updated)
        if item.name == "groundSpeedMin"
    )
    assert parsed.activation_ref == "FlightBehavior::Cruise"
    assert parsed.containing_behavior == "FlightBehavior"
    assert parsed.containing_state == "Cruise"
    constraint_report = report["activated_constraint_conformance"]
    assert constraint_report["removed_unplanned_constraints"] == [
        "Controller.groundSpeedMin"
    ]
    assert constraint_report["planned_state_active_count"] == 1


def test_state_active_constraint_executes_without_fake_runtime_initial_value():
    attributes = [
        {
            "name": "currentGroundSpeed",
            "value_type": "Real",
            "role": "RUNTIME_MEASUREMENT",
            "input_binding": "signal.payload.value",
            "provenance": "DESIGN_DECISION",
        },
        {
            "name": "minGroundSpeed",
            "value_type": "Real",
            "initial_value": "5",
            "role": "DESIGN_PARAMETER",
            "provenance": "DESIGN_DECISION",
        },
        {
            "name": "engageCruise",
            "value_type": "Boolean",
            "initial_value": "true",
            "role": "LOCAL_STATE",
            "provenance": "DESIGN_DECISION",
        },
    ]
    constraints = [{
        "constraint_id": "groundSpeedMin",
        "owner": "Controller",
        "expression": {
            "lhs": "currentGroundSpeed",
            "operator": ">=",
            "rhs": "minGroundSpeed",
        },
        "activation": {
            "kind": "STATE_ACTIVE",
            "state": "FlightBehavior::Cruise",
        },
        "provenance": {"kind": "DESIGN_DECISION"},
        "verification_tier": "STATE_EXECUTION",
    }]
    plan = ModelGenerationPlan.from_payload(
        _payload(
            attributes=attributes,
            constraints=constraints,
            behaviors=[_flight_behavior(
                trigger_kind="GUARD",
                trigger="engageCruise",
                entry_action="maintainCruiseSpeed",
            )],
        )
    )
    model = """package P {
        item def SignalData { attribute value : Real; }
        port def DataPort { in item payload : SignalData; }
        action def maintainCruiseSpeed;
        part def Producer { out port signal : DataPort; }
        part def Controller {
            in port signal : DataPort;
            state def FlightBehavior {
                state Ground;
                state Cruise {
                    entry action maintain : maintainCruiseSpeed;
                }
                transition initial then Ground;
                transition engage first Ground if engageCruise then Cruise;
            }
        }
        part producer : Producer;
        part controller : Controller;
        connect producer.signal to controller.signal;
    }"""

    updated, report = apply_generation_plan(model, plan)
    simulation = run_behavioral_simulation(updated, "P")
    constraint_result = next(
        item for item in simulation.scenario_results
        if item.name == "state_constraint_groundSpeedMin"
    )

    assert report["status"] == "PASS"
    assert constraint_result.passed is True
    assert "no initial value" not in " ".join(constraint_result.violations)


def test_materialized_state_owned_constraint_is_syside_valid():
    model = """package P {
        private import ScalarValues::*;
        part def Controller {
            attribute measured : Real = 5.0;
            attribute minimum : Real = 5.0;
            attribute hazard : Boolean = true;
            state def Behavior {
                entry; then Nominal;
                state Nominal;
                state Avoiding;
                transition go first Nominal if hazard then Avoiding;
            }
        }
    }"""
    constraint = ConstraintPlan.from_dict({
        "constraint_id": "bound",
        "owner": "Controller",
        "expression": {
            "lhs": "measured",
            "operator": ">=",
            "rhs": "minimum",
        },
        "activation": {
            "kind": "STATE_ACTIVE",
            "state": "Behavior::Avoiding",
        },
        "verification_tier": "STATE_EXECUTION",
    })

    updated, report = materialize_planned_constraints(
        model, (constraint,)
    )
    syntax = check_syntax(
        updated,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )

    assert report["status"] == "PASS"
    assert syntax.parser_errors == []
    assert syntax.sema_errors == []


def test_state_active_reference_must_be_behavior_qualified_and_reachable():
    attributes = [
        {
            "name": "currentGroundSpeed",
            "initial_value": "0",
            "role": "LOCAL_STATE",
            "provenance": "DESIGN_DECISION",
        },
        {
            "name": "minGroundSpeed",
            "initial_value": "5",
            "role": "DESIGN_PARAMETER",
            "provenance": "DESIGN_DECISION",
        },
    ]
    constraints = [{
        "constraint_id": "groundSpeedMin",
        "owner": "Controller",
        "expression": {
            "lhs": "currentGroundSpeed",
            "operator": ">=",
            "rhs": "minGroundSpeed",
        },
        "activation": {
            "kind": "STATE_ACTIVE",
            "state": "FlightBehavior::Cruise",
        },
        "provenance": {"kind": "DESIGN_DECISION"},
        "verification_tier": "STATE_EXECUTION",
    }]
    plan = ModelGenerationPlan.from_payload(
        _payload(attributes=attributes, constraints=constraints)
    )
    model = _model("""
            state def FlightBehavior {
                state Ground;
                state Cruise;
                transition initial then Ground;
            }""")

    _, report = apply_generation_plan(model, plan)

    assert report["status"] == "FAIL"
    assert report["activated_constraint_conformance"][
        "activation_issues"
    ] == [
        "Controller.groundSpeedMin: activation state "
        "FlightBehavior::Cruise is unreachable from initial"
    ]


def test_plan_restores_owned_invariant_and_removes_unplanned_assertion():
    attributes = [
        {
            "name": "currentLoad",
            "initial_value": "0",
            "role": "LOCAL_STATE",
            "provenance": "DESIGN_DECISION",
        },
        {
            "name": "maxLoad",
            "initial_value": "10",
            "role": "DESIGN_PARAMETER",
            "provenance": "DESIGN_DECISION",
        },
    ]
    constraints = [{
        "constraint_id": "loadBound",
        "owner": "Controller",
        "expression": {
            "lhs": "currentLoad",
            "operator": "<=",
            "rhs": "maxLoad",
        },
        "activation": {"kind": "ALWAYS"},
        "provenance": {"kind": "DESIGN_DECISION"},
        "verification_tier": "PARAMETRIC_SWEEP",
    }]
    plan = ModelGenerationPlan.from_payload(
        _payload(attributes=attributes, constraints=constraints)
    )
    model = _model("""
            attribute currentLoad : Real = 99;
            attribute maxLoad : Real = 10;
            assert constraint loadBound { currentLoad >= maxLoad }
            assert constraint inventedBound { currentLoad >= 500 }""")

    updated, report = apply_generation_plan(model, plan)

    assert report["status"] == "PASS"
    assert "attribute currentLoad : Real = 0;" in updated
    assert "currentLoad <= maxLoad" in updated
    assert "inventedBound" not in updated
    assert (
        "// PLAN-CONSTRAINT loadBound provenance=DESIGN_DECISION "
        "activation=ALWAYS verification=PARAMETRIC_SWEEP"
    ) in updated


def test_qualified_owner_and_plan_annotation_control_parametric_evidence():
    text = """package P {
        part def Controller {
            attribute currentLoad : Real = 0;
            attribute maxLoad : Real = 10;
            // PLAN-CONSTRAINT loadBound provenance=DESIGN_DECISION activation=ALWAYS verification=PARAMETRIC_SWEEP
            assert constraint loadBound { currentLoad <= maxLoad }
            assert constraint inventedBound { currentLoad >= 500 }
        }
    }"""

    parsed = extract_constraints(text)
    result = run_behavioral_simulation(text, model_name="P")

    assert [item.owner_part for item in parsed] == [
        "Controller",
        "Controller",
    ]
    assert len(result.scenario_results) == 1
    assert result.scenario_results[0].name == "constraint_loadBound"
    assert "design_constraint" in result.scenario_results[0].tags


def test_qualification_uses_provenance_specific_behavior_denominators():
    requirement_pass = BehavioralScenarioResult(
        name="requiredBehavior",
        state_machine="Controller::RequiredBehavior",
        description="required behavior",
        passed=True,
        tags=["requirement_behavior"],
    )
    unrelated_fail = BehavioralScenarioResult(
        name="heuristicFailure",
        state_machine="Controller::HeuristicBehavior",
        description="unplanned heuristic",
        passed=False,
    )
    behavioral = BehavioralSimResult(
        model_name="P",
        scenario_results=[requirement_pass, unrelated_fail],
        extracted_sm_count=2,
    )
    simulation = SimulationResult(
        model_name="P",
        behavioral_result=behavioral,
    )
    digest = "same"
    qualification = build_model_qualification(
        model_text="""package P {
            requirement def REQ_SAFE_005 { doc /* required */ }
            part def Controller {
                satisfy requirement REQ_SAFE_005;
            }
        }""",
        requirements=["REQ-SAFE-005: required."],
        syntax_result=SimpleNamespace(total_errors=lambda: 0, warnings=[]),
        simulation_result=simulation,
        terminal_consistency={
            "status": "PASS",
            "model_digest": digest,
            "simulation_source_model_digest": digest,
            "evaluation_source_model_digest": digest,
        },
        generation_plan_conformance={
            "status": "PASS",
            "activated_constraint_conformance": {
                "status": "PASS",
                "external_verification_readiness": {
                    "status": "NOT_APPLICABLE",
                    "planned": 0,
                },
            },
        },
    )

    checks = {item["name"]: item for item in qualification["checks"]}
    assert checks["REQUIREMENT_BEHAVIOR_EXECUTION"]["status"] == "PASS"
    assert checks["BEHAVIORAL_EXECUTION"]["status"] == "ADVISORY"
    assert "REQUIREMENT_BEHAVIOR_EXECUTION" not in qualification["failed_checks"]


def test_a_unit_suffixed_type_is_normalised_rather_than_duplicated():
    """A real pilot run failed qualification on exactly this.

    The model declared `attribute currentSeparation : LengthValue [m] = ...`.
    The materialiser's attribute pattern did not accept a unit after the type,
    found no match by that name, and appended its own declaration — leaving the
    same attribute twice in one part def, which USER_NAMESPACE_INTEGRITY and the
    Syside namespace-distinguishability warning both correctly rejected.

    Normalising loses nothing: the project writes units on the value
    (`= 5 [m]`), never on the type, and the quantity is carried by the type name
    plus the separately tracked semantic binding.
    """
    from src.prototyping.activated_constraint_plan import (
        materialize_planned_attributes,
    )

    model = (
        "package S {\n"
        "    part def Controller {\n"
        "        in port signal : DataPort;\n"
        "        attribute currentSeparation : LengthValue [m] = "
        "signal.payload.separation;\n"
        "        attribute minSeparationThreshold : LengthValue [m] = 5 [m];\n"
        "    }\n"
        "}\n"
    )
    component = SimpleNamespace(
        name="Controller",
        attributes=(
            SimpleNamespace(
                name="currentSeparation", value_type="LengthValue", unit="m",
                initial_value="", input_binding="signal.payload.separation",
            ),
            SimpleNamespace(
                name="minSeparationThreshold", value_type="LengthValue",
                unit="m", initial_value="5", input_binding="",
            ),
        ),
    )

    text, report = materialize_planned_attributes(model, [component])

    assert report["status"] == "PASS"
    assert report["materialized_attributes"] == [], "nothing may be appended"
    for name in ("currentSeparation", "minSeparationThreshold"):
        assert text.count(f"attribute {name} :") == 1, f"{name} declared twice"
    # normalised to the project's form: unit on the value, not on the type
    assert "LengthValue [m]" not in text
    assert "attribute minSeparationThreshold : LengthValue = 5 [m];" in text
