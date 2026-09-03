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
    emit_planned_behavior,
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


def test_constraint_needs_typed_behavior():
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


def test_unreachable_state_rejected():
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


def test_response_state_needs_action():
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


def test_do_action_preserved():
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
    # Named, typed usage spelling: the bare `do action X;` declared a nested member
    # X that shadowed the part-level `action def X` - the shadowing source of the
    # 2026-08-30 draws.
    assert (
        "do action runAvoidingObstacle : maintainObstacleSeparation;"
        in compiled
    )
    assert "do action maintainObstacleSeparation;" not in compiled
    assert "action def maintainObstacleSeparation {}" in compiled
    assert "item def ObstacleDetectedSignal;" in compiled
    assert check_syntax(compiled).has_errors is False
    machines = extract_state_machines(compiled)
    assert len(machines) == 1
    assert machines[0].do_action_for_state("AvoidingObstacle") == (
        "runAvoidingObstacle"
    )
    assert machines[0].response_action_for_state("AvoidingObstacle") == (
        "runAvoidingObstacle"
    )


def test_fragment_compiler_fixes_drift():
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


def test_owned_compiler_fixes_drift():
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


def test_only_compiler_writes_constraints():
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


_INHIBITION_REQ = (
    "REQ-SAFE-006: The system shall maintain the payload in the mechanically "
    "locked state whenever a delivery-abort condition is active, regardless of "
    "geographic proximity to the delivery waypoint."
)


def _release_plan(guard: str = "") -> PlannedBehavior:
    return PlannedBehavior(
        owner="PayloadMechanism",
        behavior_id="PayloadReleaseBehavior",
        initial_state="Locked",
        states=(
            PlannedState("Locked", role="INITIAL"),
            PlannedState("Releasing", role="RESPONSE",
                         entry_action="actuateRelease"),
        ),
        transitions=(PlannedTransition(
            transition_id="toReleasing", source="Locked", target="Releasing",
            trigger_kind="ACCEPT", trigger="DeliveryCoordinateSatisfied",
            guard=guard,
        ),),
        source_requirement_id="REQ_SAFE_006",
    )


def _validate(behavior: PlannedBehavior) -> list[str]:
    return validate_planned_behaviors(
        [behavior],
        component_names={"PayloadMechanism"},
        requirements=[_INHIBITION_REQ],
        state_active_constraints=[],
    )


def test_held_state_needs_guard():
    """The generated plan, verbatim. Every structural check passed - states exist,
    identifiers legal, response state reachable - yet the vehicle separated its
    payload during an active abort. Nothing checked whether the held state could be
    left unconditionally.
    """
    issues = _validate(_release_plan())

    assert issues
    assert "leaves Locked unconditionally" in issues[0]
    assert "REQ_SAFE_006" in issues[0]
    assert "abort" in issues[0]


def test_matching_guard_satisfies():
    assert _validate(_release_plan("not deliveryAbortActive")) == []


def test_other_guard_not_enough():
    issues = _validate(_release_plan("not batteryLow"))
    assert issues and "leaves Locked unconditionally" in issues[0]


def test_guard_emitted_with_accept():
    """An inhibition is accept plus guard: the event still arrives and the transition
    does not fire while the condition holds. Emitting only one makes "release on
    arrival" and "release on arrival unless aborted" the same model.
    """
    text = emit_planned_behavior(_release_plan("not deliveryAbortActive"))

    assert "accept DeliveryCoordinateSatisfied" in text
    assert "if not deliveryAbortActive" in text
    assert text.index("accept") < text.index("if not") < text.index("then Releasing")


def test_dropped_guard_caught():
    """Dropping a guard leaves a model that parses, keeps every state reachable and
    fires unconditionally, so the conformance check has to name it.
    """
    behavior = _release_plan("not deliveryAbortActive")
    unguarded = emit_planned_behavior(_release_plan())
    report = check_planned_behavior_conformance(
        unguarded, [behavior], owned=False)

    assert report["status"] == "FAIL"
    assert any("dropped its guard" in issue for issue in report["issues"])


def test_non_inhibition_needs_no_guard():
    issues = validate_planned_behaviors(
        [_release_plan()],
        component_names={"PayloadMechanism"},
        requirements=[
            "REQ-FUNC-005: The system shall release the payload within 1.0 m "
            "of the designated delivery waypoint."
        ],
        state_active_constraints=[],
    )
    assert not any("unconditionally" in issue for issue in issues)
