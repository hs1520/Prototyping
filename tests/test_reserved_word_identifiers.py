"""Plan identifiers are rendered verbatim into SysML text by materialisation
steps that run after the last syntax gate, so a reserved word in the plan
becomes a parser error in the committed model (observed: an entry action
named `return` produced `action def return {}` and two parser errors on an
otherwise qualifying run). Planning time is the cheapest place to refuse
one, and a plan issue routes back through the existing retry-with-feedback
loop."""
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


def test_reserved_entry_action_is_a_plan_issue():
    plan = ModelGenerationPlan.from_payload(_payload(entry_action="return"))
    issues = _reserved_issues(plan)
    assert issues and "'return'" in issues[0]
    assert plan.status != "PASS"


def test_reserved_state_component_port_and_trigger_are_flagged():
    for kwargs in (
        {"component": "part"},
        {"port": "flow"},
        {"state": "entry"},
        {"trigger": "accept"},
    ):
        plan = ModelGenerationPlan.from_payload(_payload(**kwargs))
        assert _reserved_issues(plan), kwargs


def test_legal_identifiers_raise_no_reserved_issue():
    plan = ModelGenerationPlan.from_payload(_payload())
    assert not _reserved_issues(plan)


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_reserved_word_list_agrees_with_the_parser():
    """Every listed word must actually be refused by the parser in at least
    one declaration position the plan renders, and common legal identifiers
    must not be listed."""
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


def test_ag_decision_identifier_rejects_reserved_words():
    from src.prototyping.ag_decision import DecisionError, _identifier
    import pytest as _pytest
    assert _identifier("deployParachute", "f") == "deployParachute"
    with _pytest.raises(DecisionError):
        _identifier("return", "f")
    with _pytest.raises(DecisionError):
        _identifier("state", "f")


def test_ag_emitter_sanitiser_never_emits_a_reserved_word():
    from src.prototyping.ag_emitter import _sysml_identifier
    assert _sysml_identifier("state") == "id_state"
    assert _sysml_identifier("safe-mode") == "safe_mode"
    assert _sysml_identifier("3phase") == "id_3phase"
