from __future__ import annotations

from src.prototyping.activated_constraint_plan import (
    ConstraintPlan,
    materialize_planned_constraints,
)
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    PlannedState,
    PlannedTransition,
    check_planned_behavior_conformance,
    materialize_owned_planned_behaviors,
    materialize_planned_behaviors,
    validate_planned_behaviors,
)
from src.simulation.state_extractor import extract_state_machines
from src.simulation.syntax_checker import check_syntax


def _behavior() -> PlannedBehavior:
    return PlannedBehavior(
        owner="FlightController",
        behavior_id="FlightControllerBehavior",
        initial_state="NominalFlight",
        states=(
            PlannedState("NominalFlight", "INITIAL"),
            PlannedState(
                "AvoidingObstacle",
                "RESPONSE",
                "executeObstacleAvoidance",
            ),
        ),
        transitions=(
            PlannedTransition(
                "detectObstacle",
                "NominalFlight",
                "AvoidingObstacle",
                "ACCEPT",
                "ObstacleDetectedSignal",
            ),
        ),
        provenance="FROZEN_REQUIREMENT",
        source_requirement_id="REQ_FUNC_002",
    )


def _constraint() -> ConstraintPlan:
    return ConstraintPlan(
        constraint_id="maintainMinSeparation",
        owner="FlightController",
        lhs="currentSeparation",
        operator=">=",
        rhs="minSeparationThreshold",
        activation_kind="STATE_ACTIVE",
        activation_ref="FlightControllerBehavior::AvoidingObstacle",
        provenance="FROZEN_REQUIREMENT",
        verification_tier="STATE_EXECUTION",
        source_requirement_id="REQ_FUNC_002",
    )


def test_state_active_constraint_requires_exact_typed_behavior_identity():
    issues = validate_planned_behaviors(
        (),
        component_names={"FlightController"},
        requirements=(
            "REQ-FUNC-002: maintain at least 5 metres while avoiding it",
        ),
        state_active_constraints=(_constraint(),),
    )

    assert issues == [
        "FlightController.maintainMinSeparation activation "
        "FlightControllerBehavior::AvoidingObstacle has no typed "
        "behavior/state declaration"
    ]


def test_behavior_plan_rejects_unreachable_activation_state():
    behavior = PlannedBehavior(
        owner="FlightController",
        behavior_id="FlightControllerBehavior",
        initial_state="NominalFlight",
        states=(
            PlannedState("NominalFlight", "INITIAL"),
            PlannedState("AvoidingObstacle", "RESPONSE"),
        ),
        transitions=(),
        source_requirement_id="REQ_FUNC_002",
    )

    issues = validate_planned_behaviors(
        (behavior,),
        component_names={"FlightController"},
        requirements=(
            "REQ-FUNC-002: maintain at least 5 metres while avoiding it",
        ),
        state_active_constraints=(_constraint(),),
    )

    assert (
        "behaviors[0] state AvoidingObstacle is unreachable from "
        "NominalFlight"
    ) in issues


def test_response_state_requires_executable_entry_or_do_action():
    behavior = PlannedBehavior(
        owner="FlightController",
        behavior_id="FlightControllerBehavior",
        initial_state="NominalFlight",
        states=(
            PlannedState("NominalFlight", "INITIAL"),
            PlannedState("AvoidingObstacle", "RESPONSE"),
        ),
        transitions=(
            PlannedTransition(
                "detectObstacle",
                "NominalFlight",
                "AvoidingObstacle",
                "ACCEPT",
                "ObstacleDetectedSignal",
            ),
        ),
    )

    issues = validate_planned_behaviors(
        (behavior,),
        component_names={"FlightController"},
        requirements=(),
        state_active_constraints=(),
    )

    assert any(
        "role RESPONSE requires an executable entry_action or do_action"
        in issue for issue in issues
    )


def test_do_action_is_preserved_as_executable_state_behavior():
    behavior = PlannedBehavior(
        owner="FlightController",
        behavior_id="FlightControllerBehavior",
        initial_state="NominalFlight",
        states=(
            PlannedState("NominalFlight", "INITIAL"),
            PlannedState(
                "AvoidingObstacle",
                "RESPONSE",
                do_action="maintainObstacleSeparation",
            ),
        ),
        transitions=(
            PlannedTransition(
                "detectObstacle",
                "NominalFlight",
                "AvoidingObstacle",
                "ACCEPT",
                "ObstacleDetectedSignal",
            ),
        ),
    )

    compiled, report = materialize_planned_behaviors("", (behavior,))

    assert report["status"] == "PASS"
    assert "do action maintainObstacleSeparation;" in compiled
    assert "action def maintainObstacleSeparation {}" in compiled
    assert "item def ObstacleDetectedSignal;" in compiled
    assert check_syntax(compiled).has_errors is False
    machines = extract_state_machines(compiled)
    assert len(machines) == 1
    assert machines[0].do_action_for_state("AvoidingObstacle") == (
        "maintainObstacleSeparation"
    )
    assert machines[0].response_action_for_state("AvoidingObstacle") == (
        "maintainObstacleSeparation"
    )


def test_fragment_compiler_replaces_llm_state_identity_drift():
    fragment = """
// OWNER: FlightController
state def FlightControllerBehavior {
    transition initial then FCNominalFlight;
    state FCNominalFlight;
    state FC_AvoidingObstacle;
    transition detectObstacle
        first FCNominalFlight
        accept ObstacleDetectedSignal
        then FC_AvoidingObstacle;
}
"""

    compiled, report = materialize_planned_behaviors(
        fragment, (_behavior(),)
    )

    assert report["status"] == "PASS"
    assert "state AvoidingObstacle {" in compiled
    assert "state FC_AvoidingObstacle" not in compiled
    assert "first NominalFlight" in compiled
    assert "then AvoidingObstacle;" in compiled
    assert compiled.count("state def FlightControllerBehavior") == 1


def test_owned_compiler_replaces_assembly_identity_drift():
    assembled = """
package DeliveryUAV {
    part def FlightController {
        attribute currentSeparation : Real;
        attribute minSeparationThreshold : Real = 5.0;
        state def FlightControllerBehavior {
            transition initial then FCNominalFlight;
            state FCNominalFlight;
            state FCAvoidingObstacle;
        }
    }
}
"""

    compiled, report = materialize_owned_planned_behaviors(
        assembled, (_behavior(),)
    )

    assert report["status"] == "PASS"
    assert "state AvoidingObstacle {" in compiled
    assert "state FCAvoidingObstacle" not in compiled
    assert compiled.count("state def FlightControllerBehavior") == 1
    assert "item def ObstacleDetectedSignal;" in compiled
    assert "action def ObstacleDetectedSignal" not in compiled
    assert "action def executeObstacleAvoidance {}" in compiled


def test_constraint_compiler_is_the_only_state_constraint_writer():
    assembled = """
package DeliveryUAV {
    item def ObstacleDetectedSignal;
    part def FlightController {
        attribute currentSeparation : Real;
        attribute minSeparationThreshold : Real = 5.0;
    }
}
"""
    behavior_text, behavior_report = materialize_owned_planned_behaviors(
        assembled, (_behavior(),)
    )
    final_text, constraint_report = materialize_planned_constraints(
        behavior_text, (_constraint(),)
    )

    assert behavior_report["status"] == "PASS"
    assert constraint_report["status"] == "PASS"
    assert final_text.count(
        "assert constraint maintainMinSeparation"
    ) == 1
    state_report = check_planned_behavior_conformance(
        final_text, (_behavior(),), owned=True
    )
    assert state_report["status"] == "PASS"
    assert check_syntax(final_text).has_errors is False
