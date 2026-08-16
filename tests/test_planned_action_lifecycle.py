"""The planned-action lifecycle is exercised through its two-entry interface."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from src.agents.planned_action_lifecycle import (
    prepare_planned_actions,
    observe_terminal_actions,
)
from src.prototyping.action_effects import PlannedActionEffect, SEND_EVENT


def _effect(**overrides) -> PlannedActionEffect:
    values = {
        "requirement_id": "REQ_SAFE_005",
        "owner_def": "PayloadController",
        "owner_behavior": "PayloadControllerBehavior",
        "response_state": "Releasing",
        "action_def": "releasePayload",
        "usage_label": "onRelease",
        "effect_kind": SEND_EVENT,
        "event_type": "PayloadReleaseRequestedSignal",
        "sender_port": "payloadCommand",
        "consumer_owner_def": "PayloadActuator",
        "consumer_behavior": "PayloadActuatorBehavior",
        "accept_transition": "acceptPayloadReleaseRequestedSignal",
        "target_state": "Released",
    }
    values.update(overrides)
    return PlannedActionEffect(**values)


def _ag_plan() -> dict:
    producer = SimpleNamespace(
        owner_def="PayloadController",
        behavior="PayloadControllerBehavior",
        response_state="Releasing",
        response_action="releasePayload",
        guarantee="payloadReleaseRequested",
        trigger_signal="ReleaseCommand",
    )
    consumer = SimpleNamespace(
        owner_def="PayloadActuator",
        behavior="PayloadActuatorBehavior",
        response_state="Released",
        response_action="releasePayloadAtActuator",
        guarantee="payloadReleased",
        trigger_signal="PayloadReleaseRequestedSignal",
    )
    return {
        "specs": (
            SimpleNamespace(
                source_requirement="REQ_SAFE_005",
                components=(producer, consumer),
            ),
        ),
    }


def _model_plan(
    *,
    source_port: str = "payloadCommand",
    action_effects: tuple[PlannedActionEffect, ...] = (),
) -> dict:
    payload = {
        "components": (
            {
                "name": "PayloadController",
                "ports": (
                    {"name": "payloadCommand", "direction": "out"},
                ),
            },
            {
                "name": "PayloadActuator",
                "ports": (
                    {"name": "payloadCommandIn", "direction": "in"},
                ),
            },
        ),
        "connections": (
            {
                "source_component": "PayloadController",
                "source_port": source_port,
                "target_component": "PayloadActuator",
                "target_port": "payloadCommandIn",
            },
        ),
    }
    if action_effects:
        payload["action_effects"] = tuple(
            effect.to_dict() for effect in action_effects
        )
    return payload


def _model(
    *,
    action_body: str = "",
    invocation: str = "entry action releasePayload;",
) -> str:
    return f"""
package PayloadChain {{
    item def ReleaseCommand;
    item def ExistingSignal;
    item def PayloadReleaseRequestedSignal;

    port def CommandPort {{
        in item release : PayloadReleaseRequestedSignal;
    }}

    part def PayloadController {{
        out port payloadCommand : CommandPort;
        action def releasePayload {{ {action_body} }}
        action def plannedOnly {{ }}

        state def PayloadControllerBehavior {{
            entry; then Idle;
            state Idle;
            state Releasing {{ {invocation} }}
            transition doRelease first Idle accept ReleaseCommand then Releasing;
        }}
    }}

    part def PayloadActuator {{
        in port payloadCommandIn : CommandPort;
        state def PayloadActuatorBehavior {{
            entry; then Held;
            state Held;
            state Released;
            transition acceptRelease first Held
                accept PayloadReleaseRequestedSignal then Released;
        }}
    }}

    part def DeliverySystem {{
        part controller : PayloadController;
        part actuator : PayloadActuator;
        connect controller.payloadCommand to actuator.payloadCommandIn;
    }}
}}
"""


def test_runtime_effects_materialize_while_plan_effects_control_the_audit():
    plan_effect = _effect(
        requirement_id="REQ_PLAN_ONLY",
        action_def="plannedOnly",
        usage_label="onPlannedOnly",
    )
    prepared = prepare_planned_actions(
        _model(),
        model_plan=_model_plan(action_effects=(plan_effect,)),
        ag_plan=_ag_plan(),
    )

    assert "send PayloadReleaseRequestedSignal() to payloadCommand;" in (
        prepared.model_text
    )
    planned_body = prepared.model_text.split("action def plannedOnly", 1)[1]
    assert "send" not in planned_body.split("}", 1)[0]

    observed = observe_terminal_actions(
        prepared,
        prepared.model_text,
        requirements=(),
        profile="ENFORCE_V1",
    )
    chains = observed.to_artifact_dict()["requirement_chains"]
    assert [chain["action_def"] for chain in chains] == ["plannedOnly"]
    assert observed.to_artifact_dict()["status"] == "FAIL"


def test_nonempty_body_is_preserved_while_bare_usage_is_still_typed():
    original_body = "send ExistingSignal() to payloadCommand;"
    prepared = prepare_planned_actions(
        _model(action_body=original_body),
        model_plan=_model_plan(),
        ag_plan=_ag_plan(),
    )

    assert original_body in prepared.model_text
    assert "send PayloadReleaseRequestedSignal()" not in prepared.model_text
    assert "entry action onReleasePayload : releasePayload;" in prepared.model_text
    codes = {item.code for item in prepared.diagnostics}
    assert "NONEMPTY_BODY" in codes
    assert "TYPED_USAGE_APPLIED" in codes


def test_preparation_is_byte_idempotent():
    first = prepare_planned_actions(
        _model(),
        model_plan=_model_plan(),
        ag_plan=_ag_plan(),
    )
    second = prepare_planned_actions(
        first.model_text,
        model_plan=_model_plan(),
        ag_plan=_ag_plan(),
    )

    assert second.model_text == first.model_text
    assert second.changed is False
    assert second.syntax_disposition == "NOT_CHECKED"


def test_syntax_regression_rolls_back_the_complete_candidate():
    original = _model()
    prepared = prepare_planned_actions(
        original,
        model_plan=_model_plan(source_port="invalid-port"),
        ag_plan=_ag_plan(),
    )

    assert prepared.model_text == original
    assert prepared.syntax_disposition == "REJECTED_SYNTAX_REGRESSION"
    assert "SYNTAX_REGRESSION_ROLLBACK" in {
        item.code for item in prepared.diagnostics
    }


def test_observation_reads_the_final_terminal_text_not_the_prepared_candidate():
    prepared = prepare_planned_actions(
        _model(),
        model_plan=_model_plan(),
        ag_plan=_ag_plan(),
    )
    repaired_after_commit = prepared.model_text.replace(
        "send PayloadReleaseRequestedSignal() to payloadCommand;", ""
    )

    observed = observe_terminal_actions(
        prepared,
        repaired_after_commit,
        requirements=(),
        profile="ENFORCE_V1",
    )

    artifact = observed.to_artifact_dict()
    assert artifact["source_model_sha256"] == hashlib.sha256(
        repaired_after_commit.encode("utf-8")
    ).hexdigest()
    assert artifact["status"] == "FAIL"


def test_artifact_projection_is_a_fresh_copy():
    prepared = prepare_planned_actions(
        _model(),
        model_plan=_model_plan(),
        ag_plan=_ag_plan(),
    )
    observed = observe_terminal_actions(
        prepared,
        prepared.model_text,
        requirements=(),
        profile="LEGACY_AUDIT",
    )

    first = observed.to_artifact_dict()
    first["status"] = "MUTATED"
    assert observed.to_artifact_dict()["status"] == "ADVISORY"
