from types import SimpleNamespace

from src.prototyping.generation_plan import (
    ModelGenerationPlan,
    apply_generation_plan,
)
from src.prototyping.activated_constraint_plan import (
    ConstraintPlan,
    _unit_tokens,
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


def test_no_invented_500_n_threshold():
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


def test_stated_units_both_spellings():
    """A stated unit is accepted in every spelling of itself.

    'degree' and 'minutes' were absent from the vocabulary, so a plan satisfied
    neither 'deg' nor 'degree' and alternated between them until the attempt
    budget ran out.
    """
    assert {"degree", "deg"} <= _unit_tokens(
        "roll and pitch RMS within 1.0 degree"
    )
    assert {"minutes", "min"} <= _unit_tokens(
        "sustain flight for a minimum of 25 minutes"
    )
    assert {"m/s", "m_s"} <= _unit_tokens("a closing speed of 1.5 m/s")
    assert "N" not in _unit_tokens("within 1.0 degree")


def test_req_id_digits_not_a_bound():
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


def test_state_property_inside_state():
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
        # INSPECTION rather than STATE_EXECUTION: the subject is unbound, so the
        # state executor cannot sweep it and the plan validator says so (see
        # activated_constraint_plan._state_execution_obstacle). The test still
        # checks that the constraint is serialised inside the owning state.
        "verification_tier": "INSPECTION",
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


def test_no_fake_runtime_initial_value():
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


def test_state_owned_constraint_valid():
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


def test_reference_qualified_reachable():
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


def test_restores_owned_invariant():
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


def test_annotation_gates_parametric():
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


def test_provenance_denominators():
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


def test_unit_suffixed_type_not_duped():
    """A pilot run failed qualification on this.

    The model declared `attribute currentSeparation : LengthValue [m] = ...`. The
    materialiser's attribute pattern did not accept a unit after the type, matched
    nothing, and appended its own declaration, leaving the attribute twice in one
    part def - rejected by USER_NAMESPACE_INTEGRITY and Syside's
    namespace-distinguishability warning. Normalising is safe: units are written
    on the value (`= 5 [m]`), never on the type, and the quantity is carried by
    the type name plus the separately tracked semantic binding.
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
    assert "LengthValue [m]" not in text
    assert "attribute minSeparationThreshold : LengthValue = 5 [m];" in text


def test_unknown_value_type_fails():
    """One archived run failed Syside with "No Type named 'StateEnum' found."

    The plan had `lockState : StateEnum = Locked` and validation only checked that
    the name was a well-formed identifier, which StateEnum is. The plan cannot
    declare a type, so anything outside the library set is a reference to nothing;
    failing the plan lets generation retry inside its bounded budget instead of
    committing a model that cannot parse.
    """
    from src.prototyping.activated_constraint_plan import (
        AttributePlan, validate_constraint_plan,
    )

    def plan_with(value_type: str):
        return AttributePlan(
            name="lockState", value_type=value_type, unit="1",
            role="LOCAL_STATE", initial_value=None,
            provenance="DESIGN_DECISION",
        )

    component = SimpleNamespace(
        name="PayloadMechanism", attributes=(plan_with("StateEnum"),)
    )
    issues = validate_constraint_plan([], [component], [])
    assert any("StateEnum" in issue for issue in issues), (
        f"an unresolvable value type must fail the plan; got {issues}"
    )

    for legal in ("Boolean", "Real", "LengthValue", "ISQ::DurationValue"):
        ok = SimpleNamespace(
            name="PayloadMechanism", attributes=(plan_with(legal),)
        )
        assert not validate_constraint_plan([], [ok], []), (
            f"{legal} is emitted by this pipeline and must be accepted"
        )


def test_equality_claims_inspection():
    """The plan does not promise execution evidence the executor cannot produce.

    `behavioral_sim` probes a constraint's satisfaction boundary by perturbing the
    right-hand value and requiring one side to satisfy and the other not to.
    Equality fails that by construction, so every `==` STATE_ACTIVE constraint was
    reported "boundary is not live" regardless of the model - measured on
    pilot_n6_20260802/seed-3, which lost the qualification gate.
    """
    from src.prototyping.activated_constraint_plan import (
        ConstraintPlan,
        _state_execution_obstacle,
    )

    def constraint(operator: str) -> ConstraintPlan:
        return ConstraintPlan(
            constraint_id="c",
            owner="Controller",
            lhs="payloadLocked",
            operator=operator,
            rhs="defaultLockState",
            activation_kind="STATE_ACTIVE",
            activation_ref="B::S",
        )

    for operator in ("<=", ">=", "<", ">"):
        assert _state_execution_obstacle(constraint(operator), None) is None

    obstacle = _state_execution_obstacle(constraint("=="), None)
    assert obstacle is not None
    assert "live satisfaction boundary" in obstacle


def test_executor_rejects_equality():
    """Pins the executor behaviour the rule above depends on.

    If the liveness probe learns to handle `==`, this fails and the plan rule
    should be revisited rather than the constraint routed around.
    """
    from src.simulation.behavioral_sim import eval_op

    rhs = 1.0
    epsilon = max(abs(rhs) * 0.01, 0.01)
    for operator in ("<=", ">=", "<", ">"):
        above, below = rhs + epsilon, rhs - epsilon
        valid, invalid = (
            (above, below) if operator in {">=", ">"} else (below, above)
        )
        assert eval_op(valid, operator, rhs)
        assert not eval_op(invalid, operator, rhs)

    # Equality: both perturbed sides violate, so no valid/invalid assignment
    # exists and the probe reports no live boundary.
    assert not eval_op(rhs + epsilon, "==", rhs)
    assert not eval_op(rhs - epsilon, "==", rhs)


def test_constant_subject_reported():
    """The plan may claim executable evidence its executor cannot produce.

    A STATE_ACTIVE constraint whose subject carries only an initial value compares
    that constant to itself, so the behavioural executor refuses it for want of a
    bound runtime measurement and the requirement lands unanchored.
    `_state_execution_obstacle` records why the validator does not reject this
    yet; this test keeps the disagreement visible.
    """
    requirement = (
        "REQ_FUNC_006: The system shall incorporate a revised waypoint "
        "sequence into the active flight plan within 1.0 second."
    )
    attributes = [{
        "name": "waypointModificationLatency",
        "value_type": "Real",
        "unit": "s",
        "role": "FROZEN_THRESHOLD",
        "initial_value": "1.0",
        "provenance": "FROZEN_REQUIREMENT",
        "source_requirement_id": "REQ_FUNC_006",
    }]
    constraints = [{
        "constraint_id": "waypointModificationLatencyConstraint",
        "owner": "Controller",
        "expression": {
            "lhs": "waypointModificationLatency",
            "operator": "<=",
            "rhs": "1.0",
        },
        "activation": {
            "kind": "STATE_ACTIVE",
            "reference": "WaypointModificationBehavior::ModifyingWaypoint",
        },
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_006",
        },
        "verification_tier": "STATE_EXECUTION",
    }]

    plan = ModelGenerationPlan.from_payload(
        _payload(attributes=attributes, constraints=constraints),
        requirements=[requirement],
    )

    assert any(
        "claims STATE_EXECUTION" in item
        and "waypointModificationLatency" in item
        for item in plan.advisories
    )
    assert plan.advisories == tuple(plan.to_dict()["advisories"])
    assert not any("claims STATE_EXECUTION" in item for item in plan.issues)


def test_bound_measurement_no_advisory():
    requirement = (
        "REQ_FUNC_006: The system shall incorporate a revised waypoint "
        "sequence into the active flight plan within 1.0 second."
    )
    attributes = [
        {
            "name": "measuredLatency",
            "value_type": "Real",
            "unit": "s",
            "role": "RUNTIME_MEASUREMENT",
            "input_binding": "signal.latency",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_FUNC_006",
        },
        {
            "name": "latencyLimit",
            "value_type": "Real",
            "unit": "s",
            "role": "FROZEN_THRESHOLD",
            "initial_value": "1.0",
            "provenance": "FROZEN_REQUIREMENT",
            "source_requirement_id": "REQ_FUNC_006",
        },
    ]
    constraints = [{
        "constraint_id": "waypointModificationLatencyConstraint",
        "owner": "Controller",
        "expression": {
            "lhs": "measuredLatency",
            "operator": "<=",
            "rhs": "latencyLimit",
        },
        "activation": {
            "kind": "STATE_ACTIVE",
            "reference": "WaypointModificationBehavior::ModifyingWaypoint",
        },
        "provenance": {
            "kind": "FROZEN_REQUIREMENT",
            "requirement_id": "REQ_FUNC_006",
        },
        "verification_tier": "STATE_EXECUTION",
    }]

    plan = ModelGenerationPlan.from_payload(
        _payload(attributes=attributes, constraints=constraints),
        requirements=[requirement],
    )

    assert plan.advisories == ()


def test_percent_unit_keeps_machines():
    """`= 25 [%]` is a parse error, and parse errors do not stay local.

    syside reparents every declaration after the unclosed expression, so state
    machines of later parts lose their owning part. The downstream crash was
    swallowed as non-fatal, disabling the behavioural tier and leaving timed
    requirements unanchored.
    """
    from src.simulation.state_extractor import extract_state_machines
    from src.simulation.syntax_checker import check_syntax
    from src.prototyping.activated_constraint_plan import sysml_unit_name

    def model(unit: str) -> str:
        return f"""package P {{
            part def Battery {{
                attribute soc : Real = 25 [{unit}];
                assert constraint socFloor {{ soc == 25 }}
            }}
            part def Monitor {{
                state def RtbMonitor {{ state Idle; entry; then Idle; }}
            }}
        }}"""

    raw = model("%")
    assert check_syntax(raw).has_errors
    assert any(
        machine.owner_part is None or machine.owner_part == "__unknown__"
        for machine in extract_state_machines(raw)
    )

    emitted = model(sysml_unit_name("%"))
    assert not check_syntax(emitted).has_errors
    machines = extract_state_machines(emitted)
    assert [m.owner_part for m in machines] == ["Monitor"]


def test_long_unit_spellings_resolve():
    from src.prototyping.activated_constraint_plan import sysml_unit_name
    from src.simulation.syntax_checker import check_syntax

    for stated, expected in (
        ("degree", "deg"), ("minutes", "min"), ("seconds", "s"),
        ("metres", "m"), ("%", "percent"),
    ):
        assert sysml_unit_name(stated) == expected
        src = (
            "package P { part def B { attribute a : Real = 1 "
            f"[{expected}]; assert constraint K {{ a == 1 }} }} }}"
        )
        assert not check_syntax(src).has_errors, expected
    # Slash units are rewritten: `[m/s]` breaks word-character bracket readers,
    # so the registry emits the identifier-safe token and resolves it with
    # `alias m_s for SI::'m/s'` (ablation pilot 2 failed on the passthrough).
    assert sysml_unit_name("m/s") == "m_s"
    assert sysml_unit_name("km/h") == "km_h"
    assert sysml_unit_name("kg") == "kg"
