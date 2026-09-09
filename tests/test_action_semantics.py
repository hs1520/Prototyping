"""Terminal action-semantics audit: what an action does.

The behavioural simulator credits a response when the target state carries an
action label, and the only reader of an action body produces no scenario for
an empty one. These tests pin the distinctions the audit has to make,
including those the archived evidence never exercises.
"""
from __future__ import annotations

import pytest

from src.prototyping.action_effects import (
    EXTERNAL_OR_UNSUPPORTED,
    LEGACY_AUDIT,
    OFF,
    SEND_EVENT,
    PlannedActionEffect,
)
from src.prototyping.action_semantics import (
    ARTIFACT_ROLE,
    BARE_INVOCATION,
    LABEL_ONLY,
    NO_ACCEPT_TRANSITION,
    NO_CONNECT_PATH,
    ORPHAN,
    REALISED_BY_BODY,
    REALISED_BY_TYPED_REF,
    UNSUPPORTED_BODY,
    analyze_action_semantics,
)


def _model(*, controller_body: str, consumer_transition: str,
           connect: str = "connect controller.payloadCommand to "
                          "actuator.payloadCommandIn;") -> str:
    return f"""
package PayloadChain {{
    item def ReleaseCommand;
    item def PayloadReleaseRequested;

    port def CommandPort {{
        in item release : PayloadReleaseRequested;
    }}

    part def PayloadController {{
        out port payloadCommand : CommandPort;

        action def releasePayload {{{controller_body}}}

        state def PayloadControllerBehavior {{
            entry; then Idle;
            state Idle;
            state Releasing {{ entry action onRelease : releasePayload; }}
            transition doRelease
                first Idle
                accept ReleaseCommand
                then Releasing;
        }}
    }}

    part def PayloadActuator {{
        in port payloadCommandIn : CommandPort;

        state def PayloadActuatorBehavior {{
            entry; then Held;
            state Held;
            state Released;
            {consumer_transition}
        }}
    }}

    part def DeliverySystem {{
        part controller : PayloadController;
        part actuator : PayloadActuator;
        {connect}
    }}
}}
"""


_SEND = "\n            send PayloadReleaseRequested() to payloadCommand;\n        "
_ACCEPT = (
    "transition acceptRelease first Held "
    "accept PayloadReleaseRequested then Released;"
)
_COMPLETE = _model(controller_body=_SEND, consumer_transition=_ACCEPT)


def _effect(**overrides) -> PlannedActionEffect:
    base = dict(
        requirement_id="REQ_FUNC_005",
        owner_def="PayloadController",
        owner_behavior="PayloadControllerBehavior",
        response_state="Releasing",
        action_def="releasePayload",
        usage_label="onRelease",
        effect_kind=SEND_EVENT,
        event_type="PayloadReleaseRequested",
        sender_port="payloadCommand",
        consumer_owner_def="PayloadActuator",
        consumer_behavior="PayloadActuatorBehavior",
        accept_transition="acceptRelease",
        target_state="Released",
    )
    base.update(overrides)
    return PlannedActionEffect(**base)


def _chain(model: str, effect: PlannedActionEffect) -> dict:
    report = analyze_action_semantics(
        model, action_effect_plan=[effect], profile="ENFORCE_V1"
    )
    return report.requirement_chains[0]


def test_complete_chain_passes():
    from src.simulation.syntax_checker import check_syntax

    assert check_syntax(_COMPLETE).total_errors() == 0, (
        "the fixture must be a legal model, or every failure below is vacuous"
    )
    chain = _chain(_COMPLETE, _effect())
    assert chain["status"] == "PASS", chain


def test_empty_response_action_fails():
    chain = _chain(_model(controller_body=" ", consumer_transition=_ACCEPT),
                   _effect())
    assert chain["status"] == "FAIL"
    assert UNSUPPORTED_BODY in chain["failures"]


def test_empty_event_item_allowed():
    report = analyze_action_semantics(_COMPLETE)
    assert report.summary["definitions_total"] == 1, (
        "only the response action is an action definition; the event is an item"
    )
    assert report.summary["supported_effects"] == 1


def test_bare_invocation_fails():
    bare = _COMPLETE.replace(
        "entry action onRelease : releasePayload;",
        "entry action releasePayload;",
    )
    chain = _chain(bare, _effect())
    assert chain["status"] == "FAIL"
    assert BARE_INVOCATION in chain["failures"]


def test_no_accepting_consumer_fails():
    chain = _chain(
        _model(controller_body=_SEND,
               consumer_transition="transition idle first Held "
                                   "accept ReleaseCommand then Released;"),
        _effect(),
    )
    assert chain["status"] == "FAIL"
    assert NO_ACCEPT_TRANSITION in chain["failures"]


def test_no_connect_path_fails():
    chain = _chain(
        _model(controller_body=_SEND, consumer_transition=_ACCEPT, connect=""),
        _effect(),
    )
    assert chain["status"] == "FAIL"
    assert NO_CONNECT_PATH in chain["failures"]


def test_foreign_owner_fails():
    chain = _chain(_COMPLETE, _effect(owner_def="SomeOtherController"))
    assert chain["status"] == "FAIL"


def test_unsupported_effect_recorded():
    chain = _chain(
        _COMPLETE,
        _effect(effect_kind=EXTERNAL_OR_UNSUPPORTED, event_type="",
                sender_port="", consumer_owner_def="", consumer_behavior="",
                accept_transition="", target_state=""),
    )
    assert chain["status"] == EXTERNAL_OR_UNSUPPORTED
    assert chain["failures"] == []


def test_four_realisation_states():
    model = """
package P {
    part def Monitor {
        action def withBody { send Ping() to out1; }
        action def typedButEmpty { }
        action def labelOnly { }
        action def nobodyNamesMe { }
        out port out1 : PingPort;
        state def Behavior {
            entry; then Idle;
            state Idle;
            state A { entry action onA : withBody; }
            state B { entry action onB : typedButEmpty; }
            state C { entry action labelOnly; }
        }
    }
}
"""
    report = analyze_action_semantics(model)
    kinds = {a.name: a.realisation for a in report.actions}
    assert kinds["withBody"] == REALISED_BY_BODY
    assert kinds["typedButEmpty"] == REALISED_BY_TYPED_REF
    assert kinds["labelOnly"] == LABEL_ONLY
    assert kinds["nobodyNamesMe"] == ORPHAN


def test_audit_profile_invents_no_chains():
    """LEGACY_AUDIT keeps archived evidence reproducible: it records the numbers,
    asserts nothing, and with no plan claims no chains instead of rebuilding
    them from action names.
    """
    report = analyze_action_semantics(_COMPLETE, profile=LEGACY_AUDIT)
    assert report.status == "ADVISORY"
    assert report.requirement_chains == []
    assert report.summary["planned_response_actions"] == 0
    assert report.summary["complete_requirement_chains"] == 0


def test_off_profile_no_work():
    report = analyze_action_semantics(_COMPLETE, profile=OFF)
    assert report.actions == []
    assert report.summary["definitions_total"] == 0


def test_report_carries_model_digest():
    import hashlib

    report = analyze_action_semantics(_COMPLETE)
    assert report.to_dict()["artifact_role"] == ARTIFACT_ROLE
    assert report.source_model_sha256 == hashlib.sha256(
        _COMPLETE.encode("utf-8")
    ).hexdigest()


def test_default_prefix_not_counted():
    model = """
package P {
    part def Latch {
        action def realOne { }
        state def Behavior {
            entry; then Off;
            state Off { entry action defaultToLockedState; }
        }
    }
}
"""
    report = analyze_action_semantics(model)
    assert report.summary["definitions_total"] == 1


@pytest.mark.parametrize("path,expected_total,expected_empty", [
    ("examples/output/pilot_n6_v8_full_20260808_1029", 155, 152),
    ("examples/output/runs/82c1e7ad-8e26-4f71-946c-83e7fe9b684e", 25, 19),
])
def test_archived_counts_reproduce(
    path, expected_total, expected_empty
):
    import pathlib

    root = pathlib.Path(path)
    if not root.exists():          # evidence archives are optional in a checkout
        return
    models = sorted(root.glob("seed-*/*/shared_model_final.sysml")) or [
        root / "final_model.sysml"
    ]
    total = empty = 0
    for model in models:
        summary = analyze_action_semantics(
            model.read_text(encoding="utf-8")
        ).summary
        total += summary["definitions_total"]
        empty += summary["empty_definitions"]
    assert (total, empty) == (expected_total, expected_empty)


# ---------------------------------------------------------------------------
# Strict reading of functional evidence. Advisory: `functional_behavior_status`
# is unchanged, so archived runs score as before.
# ---------------------------------------------------------------------------

def _archived_authoritative():
    import json
    import pathlib

    root = pathlib.Path(
        "examples/output/runs/82c1e7ad-8e26-4f71-946c-83e7fe9b684e"
    )
    model = root / "final_model.sysml"
    canonical = root / "canonical_run.json"
    if not (model.exists() and canonical.exists()):
        return None
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    requirements = [
        item if isinstance(item, str) else str(item.get("text") or "")
        for item in payload["requirements"]
    ]
    return model.read_text(encoding="utf-8"), requirements


def test_legacy_rule_credits_by_name():
    """Pinned against the authoritative run rather than a fixture.

    REQ-FUNC-001 asks for navigation to GPS waypoints. `navigateToGpsWaypoints` is
    defined but referenced nowhere, so the only crediting name is REQ-FUNC-006's
    `incorporateRevisedWaypointSequence`, matched on the shared substring
    `waypoint`. Neither REQ-FUNC-001 nor REQ-FUNC-008 hits the requirement-text
    fallback in `verification_matrix`, so their archived `partial` rows rest on
    this alone.
    """
    from src.dse.functional_behavior import functional_behavior_status

    archived = _archived_authoritative()
    if archived is None:
        return
    model, requirements = archived
    status = functional_behavior_status(model, requirements)
    assert status.get("REQ-FUNC-001") == "behaviorally-verified"
    assert status.get("REQ-FUNC-008") == "behaviorally-verified"


def test_strict_withdraws_name_credit():
    from src.dse.functional_behavior import functional_behavior_diagnosis
    from src.prototyping.action_semantics import analyze_action_semantics

    archived = _archived_authoritative()
    if archived is None:
        return
    model, requirements = archived
    diagnosis = functional_behavior_diagnosis(
        model, requirements, analyze_action_semantics(model).actions
    )
    for rid in ("REQ-FUNC-001", "REQ-FUNC-008"):
        assert diagnosis[rid]["legacy_status"] == "behaviorally-verified"
        assert diagnosis[rid]["strict_status"] == "behavior-absent"
        assert diagnosis[rid]["status_difference_reason"]


def test_strict_leaves_legacy_verdict():
    from src.dse.functional_behavior import (
        functional_behavior_diagnosis,
        functional_behavior_status,
    )
    from src.prototyping.action_semantics import analyze_action_semantics

    archived = _archived_authoritative()
    if archived is None:
        return
    model, requirements = archived
    before = dict(functional_behavior_status(model, requirements))
    functional_behavior_diagnosis(
        model, requirements, analyze_action_semantics(model).actions
    )
    assert functional_behavior_status(model, requirements) == before


# ---------------------------------------------------------------------------
# Schema 10: the plan carries the response chain, so a requirement's evidence
# is an identity to resolve rather than a name to match.
# ---------------------------------------------------------------------------

def _schema_10_payload() -> dict:
    return {
        "schema_version": "9.0",
        "components": [{"name": "PayloadController", "responsibility": "x"}],
        "connections": [{
            "source_component": "PayloadController",
            "source_port": "payloadCommand",
            "target_component": "PayloadActuator",
            "target_port": "payloadCommandIn",
        }],
        "action_effects": [_effect().to_dict()],
    }


def test_action_effects_declare_schema_10():
    from src.prototyping.generation_plan import ModelGenerationPlan

    plan = ModelGenerationPlan.from_dict(_schema_10_payload())
    assert plan.schema_version == "10.0"
    assert len(plan.action_effects) == 1
    assert plan.to_dict()["action_effects"][0]["requirement_id"] == "REQ_FUNC_005"


def test_no_action_effects_old_schema():
    from src.prototyping.generation_plan import ModelGenerationPlan

    payload = _schema_10_payload()
    payload.pop("action_effects")
    plan = ModelGenerationPlan.from_dict(payload)
    assert plan.action_effects == ()
    assert plan.schema_version != "10.0"
    assert "action_effects" not in plan.to_dict()


def test_effect_missing_identity_fails():
    from src.prototyping.generation_plan import ModelGenerationPlan

    payload = _schema_10_payload()
    payload["action_effects"][0]["consumer"]["accept_transition"] = ""
    plan = ModelGenerationPlan.from_dict(payload)
    assert any("does not name every element" in issue for issue in plan.issues)


def test_archived_plans_round_trip():
    import glob
    import json

    from src.prototyping.generation_plan import ModelGenerationPlan

    reports = sorted(glob.glob(
        "examples/output/pilot_n6_v8_full_20260808_1029/*/*/run_report.json"
    ))
    if not reports:                 # evidence archives are optional in a checkout
        return
    checked = 0
    for path in reports:
        archived = json.loads(
            open(path, encoding="utf-8").read()
        ).get("whole_model_generation_plan")
        if not archived:
            continue
        checked += 1
        rebuilt = ModelGenerationPlan.from_dict(archived).to_dict()
        assert rebuilt["schema_version"] == archived.get("schema_version")
        assert "action_effects" not in rebuilt
    assert checked


def test_unrelated_state_invocation_fails():
    """`some state somewhere types it` is not the obligation.

    The usage-label requirement was dropped because the label is a local name that
    three runs of one configuration spelled three ways, but the check still binds
    the invocation to the planned response state, or an unrelated machine
    satisfies it.
    """
    from src.prototyping.action_semantics import BARE_INVOCATION

    elsewhere = _COMPLETE.replace(
        "state Releasing { entry action onRelease : releasePayload; }",
        "state Releasing;",
    ).replace(
        "state Held;",
        "state Held { entry action stray : releasePayload; }",
    )
    chain = _chain(elsewhere, _effect())
    assert chain["status"] == "FAIL"
    assert BARE_INVOCATION in chain["failures"]


def test_any_usage_label_passes():
    for label in ("onRelease", "releasePayloadAction", "doIt"):
        model = _COMPLETE.replace(
            "entry action onRelease : releasePayload;",
            f"entry action {label} : releasePayload;",
        )
        assert _chain(model, _effect())["status"] == "PASS", label
