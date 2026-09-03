from __future__ import annotations

import pytest

from src.prototyping.generation_plan import (
    SYSML_RESERVED_WORDS,
    ModelGenerationPlan,
)

try:
    from src.simulation.syntax_checker import check_syntax
    _HAS_SYSIDE = True
except Exception:
    _HAS_SYSIDE = False


def _payload(entry_action="initiateRtb", component="FlightController",
             port="statusOut", state="Returning", trigger="RtbCommand"):
    return {
        "components": [
            {"name": component, "responsibility": "control",
             "requirements": ["REQ-FUNC-001"],
             "ports": [{"name": port, "direction": "out",
                        "type": "StatusPort"}],
             "attributes": []},
        ],
        "connections": [],
        "behaviors": [
            {"owner": component, "behavior_id": "RtbControl",
             "initial_state": "Idle",
             "states": [
                 {"state_id": "Idle"},
                 {"state_id": state, "entry_action": entry_action},
             ],
             "transitions": [
                 {"transition_id": "onRtb", "source": "Idle",
                  "target": state, "trigger_kind": "ACCEPT",
                  "trigger": trigger},
             ]},
        ],
    }


def _reserved_issues(plan):
    return [i for i in plan.issues if "reserved word" in i]


def test_reserved_entry_action_issue():
    plan = ModelGenerationPlan.from_payload(_payload(entry_action="return"))
    issues = _reserved_issues(plan)
    assert issues and "'return'" in issues[0]
    assert plan.status != "PASS"


def test_reserved_names_flagged():
    for kwargs in (
        {"component": "part"},
        {"port": "flow"},
        {"state": "entry"},
        {"trigger": "accept"},
    ):
        plan = ModelGenerationPlan.from_payload(_payload(**kwargs))
        assert _reserved_issues(plan), kwargs


def test_legal_identifiers_pass():
    plan = ModelGenerationPlan.from_payload(_payload())
    assert not _reserved_issues(plan)


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_word_list_matches_parser():
    positions = (
        "package P {{ private import ScalarValues::*; "
        "attribute {w} : Real; }}",
        "package P {{ action def {w} {{ }} }}",
        "package P {{ item def {w}; }}",
    )
    for word in sorted(SYSML_RESERVED_WORDS):
        rejected = any(
            check_syntax(
                template.format(w=word),
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            ).has_errors
            for template in positions
        )
        assert rejected, f"'{word}' is listed but the parser accepts it"
    for word in ("returnToBase", "navigate", "lockPayload", "sendStatus"):
        assert word not in SYSML_RESERVED_WORDS
        accepted = not check_syntax(
            positions[0].format(w=word),
            fail_closed=True,
            filter_stdlib_diagnostics=False,
        ).has_errors
        assert accepted, word


def test_ag_decision_rejects_reserved():
    from src.prototyping.ag_decision import DecisionError, _identifier
    import pytest as _pytest
    assert _identifier("deployParachute", "f") == "deployParachute"
    with _pytest.raises(DecisionError):
        _identifier("return", "f")
    with _pytest.raises(DecisionError):
        _identifier("state", "f")


def test_ag_emitter_sanitises_reserved():
    from src.prototyping.ag_emitter import _sysml_identifier
    assert _sysml_identifier("state") == "id_state"
    assert _sysml_identifier("safe-mode") == "safe_mode"
    assert _sysml_identifier("3phase") == "id_3phase"
