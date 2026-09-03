from src.simulation.transition_fixer import (
    build_state_machine_summary,
    validate_transitions,
)


def _model(event_declaration: str) -> str:
    return f"""package P {{
    {event_declaration}
    action def performResponse {{}}
    state def ControllerBehavior {{
        entry; then idle;
        state idle;
        state responding;
        state fallback;
        transition respond first idle accept FaultSignal then responding;
    }}
}}"""


def test_accepts_item_def_event():
    summary = build_state_machine_summary(
        _model("item def FaultSignal;"),
        "ControllerBehavior",
    )
    assert summary is not None

    result = validate_transitions(
        (
            "transition respond first idle accept FaultSignal "
            "then fallback;",
        ),
        summary,
    )

    assert len(result.accepted) == 1
    assert result.rejected == []


def test_rejects_action_def_event():
    summary = build_state_machine_summary(
        _model("action def FaultSignal {}"),
        "ControllerBehavior",
    )
    assert summary is not None

    result = validate_transitions(
        (
            "transition respond first idle accept FaultSignal "
            "then fallback;",
        ),
        summary,
    )

    assert result.accepted == []
    assert "package-level item def" in result.rejected[0][1]
