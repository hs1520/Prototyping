"""The terminal action-semantics audit: what an action actually does.

The behavioural simulator credits a response when the target state carries an
action label, and the only reader of an action body produces no scenario when
there is nothing to read.  These tests pin the distinctions that audit has to
make, including the ones the archived evidence never exercises.
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


def test_the_complete_chain_passes():
    from src.simulation.syntax_checker import check_syntax

    assert check_syntax(_COMPLETE).total_errors() == 0, (
        "the fixture must be a legal model, or every failure below is vacuous"
    )
    chain = _chain(_COMPLETE, _effect())
    assert chain["status"] == "PASS", chain


def test_an_empty_response_action_fails():
    """The archived pilot's ordinary case: a named response that does nothing."""
    chain = _chain(_model(controller_body=" ", consumer_transition=_ACCEPT),
                   _effect())
    assert chain["status"] == "FAIL"
    assert UNSUPPORTED_BODY in chain["failures"]


def test_an_empty_event_item_definition_is_allowed():
    """`item def PayloadReleaseRequested;` has no body and needs none — the
    strict rule must not demand an implementation of a command type."""
    report = analyze_action_semantics(_COMPLETE)
    assert report.summary["definitions_total"] == 1, (
        "only the response action is an action definition; the event is an item"
    )
    assert report.summary["supported_effects"] == 1


def test_a_bare_invocation_fails_even_though_the_name_matches():
    bare = _COMPLETE.replace(
        "entry action onRelease : releasePayload;",
        "entry action releasePayload;",
    )
    chain = _chain(bare, _effect())
    assert chain["status"] == "FAIL"
    assert BARE_INVOCATION in chain["failures"]


def test_a_send_with_no_accepting_consumer_fails():
    chain = _chain(
        _model(controller_body=_SEND,
               consumer_transition="transition idle first Held "
                                   "accept ReleaseCommand then Released;"),
        _effect(),
    )
    assert chain["status"] == "FAIL"
    assert NO_ACCEPT_TRANSITION in chain["failures"]


def test_a_send_with_no_connect_path_fails():
    chain = _chain(
        _model(controller_body=_SEND, consumer_transition=_ACCEPT, connect=""),
        _effect(),
    )
    assert chain["status"] == "FAIL"
    assert NO_CONNECT_PATH in chain["failures"]


def test_a_response_owned_by_another_part_does_not_satisfy_the_requirement():
    """Name matching is what this profile replaces: an identically named action
    in a different owner must not be credited."""
    chain = _chain(_COMPLETE, _effect(owner_def="SomeOtherController"))
    assert chain["status"] == "FAIL"


def test_an_unsupported_effect_is_recorded_rather_than_faked():
    chain = _chain(
        _COMPLETE,
        _effect(effect_kind=EXTERNAL_OR_UNSUPPORTED, event_type="",
                sender_port="", consumer_owner_def="", consumer_behavior="",
                accept_transition="", target_state=""),
    )
    assert chain["status"] == EXTERNAL_OR_UNSUPPORTED
    assert chain["failures"] == []


def test_the_four_realisation_states_are_distinguished():
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


def test_the_audit_profile_never_fails_and_reports_no_invented_chains():
    """LEGACY_AUDIT must leave archived evidence reproducible: it records the
    numbers and asserts nothing, and with no plan it claims no chains rather
    than reconstructing them from action names."""
    report = analyze_action_semantics(_COMPLETE, profile=LEGACY_AUDIT)
    assert report.status == "ADVISORY"
    assert report.requirement_chains == []
    assert report.summary["planned_response_actions"] == 0
    assert report.summary["complete_requirement_chains"] == 0


def test_the_off_profile_does_no_work():
    report = analyze_action_semantics(_COMPLETE, profile=OFF)
    assert report.actions == []
    assert report.summary["definitions_total"] == 0


def test_the_report_carries_a_digest_of_the_model_it_read():
    import hashlib

    report = analyze_action_semantics(_COMPLETE)
    assert report.to_dict()["artifact_role"] == ARTIFACT_ROLE
    assert report.source_model_sha256 == hashlib.sha256(
        _COMPLETE.encode("utf-8")
    ).hexdigest()


def test_the_action_def_count_is_not_inflated_by_a_default_prefixed_usage():
    """`entry action defaultToLockedState;` contains the substring `action def`
    and inflates a naive count by three across the archived pilot."""
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
def test_the_archived_evidence_reproduces_its_known_counts(
    path, expected_total, expected_empty
):
    """Pinned against the frozen evidence: the audit is only useful if it
    reproduces the numbers the report will quote."""
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


def test_the_audit_reaches_the_run_report_and_its_own_artifact():
    """Two carriers, because one has failed before.

    `generation_metadata["semantic_fixes"]` and its four sibling counters reach
    no archived artefact at all — not in the frozen pilot, not anywhere — and a
    repair-refusal audit was reverted in `a8d7052` for the same reason.  The
    run report is a hand-maintained projection of the run result, so a field
    present in the result is not thereby present in the report.
    """
    from src.app.pipeline import PrototypingPipeline
    from src.prototyping.run_artifacts import write_revised_run_artifacts

    payload = {"artifact_role": ARTIFACT_ROLE, "summary": {}}
    report = PrototypingPipeline.build_run_report(
        {"action_semantics_audit": payload}
    )
    assert report["action_semantics_audit"] == payload, (
        "the run report projection drops the audit"
    )

    import inspect

    source = inspect.getsource(write_revised_run_artifacts)
    assert '"action_semantics_audit.json"' in source, (
        "the audit has no artefact of its own to fall back on"
    )


# ---------------------------------------------------------------------------
# The strict reading of functional evidence.  Advisory: `functional_behavior_status`
# is unchanged so every archived run still scores exactly as it did.
# ---------------------------------------------------------------------------

def _archived_authoritative():
    """The frozen authoritative run, or None in a checkout without archives."""
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


def test_the_legacy_rule_credits_a_requirement_from_another_requirements_action():
    """Pinned against the authoritative run rather than a fixture.

    REQ-FUNC-001 asks for navigation to GPS waypoints. The model's own
    `navigateToGpsWaypoints` is an orphan — defined, referenced nowhere — and
    the only crediting name is `incorporateRevisedWaypointSequence`, which is
    REQ-FUNC-006's response, matched on the shared substring `waypoint`.
    Neither REQ-FUNC-001 nor REQ-FUNC-008 hits the requirement-text fallback in
    `verification_matrix`, so their archived `partial` rows rest on this alone.
    """
    from src.dse.functional_behavior import functional_behavior_status

    archived = _archived_authoritative()
    if archived is None:
        return
    model, requirements = archived
    status = functional_behavior_status(model, requirements)
    assert status.get("REQ-FUNC-001") == "behaviorally-verified"
    assert status.get("REQ-FUNC-008") == "behaviorally-verified"


def test_the_strict_reading_withdraws_what_only_a_name_supported():
    from src.dse.functional_behavior import functional_behavior_diagnosis

    archived = _archived_authoritative()
    if archived is None:
        return
    model, requirements = archived
    diagnosis = functional_behavior_diagnosis(model, requirements)
    for rid in ("REQ-FUNC-001", "REQ-FUNC-008"):
        assert diagnosis[rid]["legacy_status"] == "behaviorally-verified"
        assert diagnosis[rid]["strict_status"] == "behavior-absent"
        assert diagnosis[rid]["status_difference_reason"]


def test_the_strict_reading_leaves_the_legacy_verdict_untouched():
    """The audit must not change a number any archived run reported."""
    from src.dse.functional_behavior import (
        functional_behavior_diagnosis,
        functional_behavior_status,
    )

    archived = _archived_authoritative()
    if archived is None:
        return
    model, requirements = archived
    before = dict(functional_behavior_status(model, requirements))
    functional_behavior_diagnosis(model, requirements)
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


def test_a_plan_that_carries_action_effects_declares_schema_10():
    from src.prototyping.generation_plan import ModelGenerationPlan

    plan = ModelGenerationPlan.from_dict(_schema_10_payload())
    assert plan.schema_version == "10.0"
    assert len(plan.action_effects) == 1
    assert plan.to_dict()["action_effects"][0]["requirement_id"] == "REQ_FUNC_005"


def test_a_plan_without_action_effects_serialises_exactly_as_before():
    """Schema 1-9 plans must round-trip unchanged, or every archived run's
    plan artefact stops being comparable with the one the code now produces."""
    from src.prototyping.generation_plan import ModelGenerationPlan

    payload = _schema_10_payload()
    payload.pop("action_effects")
    plan = ModelGenerationPlan.from_dict(payload)
    assert plan.action_effects == ()
    assert plan.schema_version != "10.0"
    assert "action_effects" not in plan.to_dict()


def test_an_action_effect_missing_an_identity_fails_the_plan_closed():
    from src.prototyping.generation_plan import ModelGenerationPlan

    payload = _schema_10_payload()
    payload["action_effects"][0]["consumer"]["accept_transition"] = ""
    plan = ModelGenerationPlan.from_dict(payload)
    assert any("does not name every element" in issue for issue in plan.issues)


def test_the_archived_plans_round_trip_without_gaining_a_schema_10_key():
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


def test_an_invocation_from_an_unrelated_state_does_not_discharge_the_plan():
    """`some state somewhere types it` is not the obligation.

    Dropping the usage-label requirement was right — the label is a local name
    and three runs of one configuration spelled it three ways — but the check
    still has to bind the invocation to the planned response state, or an
    unrelated machine satisfies it.
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


def test_the_planned_response_state_may_spell_its_usage_label_any_way():
    for label in ("onRelease", "releasePayloadAction", "doIt"):
        model = _COMPLETE.replace(
            "entry action onRelease : releasePayload;",
            f"entry action {label} : releasePayload;",
        )
        assert _chain(model, _effect())["status"] == "PASS", label
