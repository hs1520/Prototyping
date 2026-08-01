from __future__ import annotations

from src.simulation.surgical_refiner import SURGICAL_SYSTEM_PROMPT
from src.agents.verification_audit import verification_gap_issues
from src.prototyping.verification_matrix import build_matrix
from src.simulation.behavioral_sim import run_behavioral_simulation
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model


_REQ = """
    requirement def REQ_SAFE_008 {
        doc /* The payload-release actuator shall default to the mechanically
        locked state upon power-on, before any arming or flight authorisation. */
    }
"""


def _model(part_body: str):
    text = f"""package D {{
        {_REQ}
        part def PayloadMechanism {{
            {part_body}
            satisfy requirement REQ_SAFE_008;
        }}
    }}"""
    return text, build_lite_model(text, model_name="D")


def test_single_state_default_invariant_is_a_valid_initialization_scenario():
    text, _ = _model("""
        attribute isLocked : Boolean = true;
        state def PayloadLockMachine {
            state Locked;
            transition initial then Locked;
        }
    """)

    result = run_behavioral_simulation(text, model_name="D")
    scenario = next(r for r in result.scenario_results
                    if r.state_machine == "PayloadLockMachine")

    assert scenario.passed
    assert "initialization" in scenario.tags
    assert any("isLocked=True" in line for line in scenario.timeline)


def test_bare_single_state_declaration_has_no_initialization_evidence():
    text, model = _model("""
        state def PayloadLockMachine {
            state Locked;
            transition initial then Locked;
        }
    """)

    result = run_behavioral_simulation(text, model_name="D")
    scenario = next(r for r in result.scenario_results
                    if r.state_machine == "PayloadLockMachine")
    row = build_matrix(model, None, RequirementLinker(model))[0]

    assert not scenario.passed
    assert any("no observable initialization semantics" in violation
               for violation in scenario.violations)
    assert row.status == "failed"
    assert "behavioral_sim_failed" in row.tiers


def test_initial_entry_action_is_valid_initialization_evidence():
    text, _ = _model("""
        action def defaultToLockedState { }
        state def PayloadLockMachine {
            state Locked { entry action lock : defaultToLockedState; }
            transition initial then Locked;
        }
    """)

    result = run_behavioral_simulation(text, model_name="D")
    scenario = next(r for r in result.scenario_results
                    if r.state_machine == "PayloadLockMachine")

    assert scenario.passed, scenario.violations
    assert "lock" in scenario.fired_actions


def test_declaration_only_locked_unlocked_shell_fails_as_unreachable():
    text, model = _model("""
        attribute isLocked : Boolean = true;
        action def defaultToLockedState { }
        state def PayloadLockMachine {
            state Locked;
            state Unlocked;
            transition initial then Locked;
        }
    """)

    result = run_behavioral_simulation(text, model_name="D")
    scenario = next(r for r in result.scenario_results
                    if r.state_machine == "PayloadLockMachine")
    row = build_matrix(model, None, RequirementLinker(model))[0]

    assert not scenario.passed
    assert any("unreachable" in violation.lower() for violation in scenario.violations)
    assert row.status == "failed"
    assert "behavioral_sim_failed" in row.tiers
    assert "behavioral_sim" not in row.tiers


def test_complete_lock_lifecycle_produces_real_behavioral_evidence():
    text, model = _model("""
        attribute isLocked : Boolean = true;
        attribute releaseAuthorized : Boolean = false;
        attribute deliveryAbortConditionActive : Boolean = false;
        action def releasePayload { }
        action def defaultToLockedState { }
        state def PayloadLockMachine {
            state Locked { entry action lock : defaultToLockedState; }
            state Unlocked { entry action release : releasePayload; }
            transition initial then Locked;
            transition release first Locked if releaseAuthorized then Unlocked;
            transition relock first Unlocked if deliveryAbortConditionActive then Locked;
        }
    """)

    result = run_behavioral_simulation(text, model_name="D")
    scenario = next(r for r in result.scenario_results
                    if r.state_machine == "PayloadLockMachine")
    row = build_matrix(model, None, RequirementLinker(model))[0]

    assert scenario.passed, scenario.violations
    assert row.status == "partial"  # behavioral PASS; generated L2 check still planned
    assert "behavioral_sim" in row.tiers
    assert "behavioral_sim_failed" not in row.tiers


def test_failed_behavioral_anchor_remains_a_refinement_issue():
    text, _ = _model("""
        attribute isLocked : Boolean = true;
        state def PayloadLockMachine {
            state Locked;
            state Unlocked;
            transition initial then Locked;
        }
    """)

    issues = verification_gap_issues(text, model_name="D")

    assert any("REQ_SAFE_008" in issue for issue in issues)
    assert any("scenario FAILS" in issue for issue in issues)
    assert any("declaration-only" in issue for issue in issues)


def test_surgical_contract_forbids_empty_multi_state_shells():
    assert "every state MUST be reachable" in SURGICAL_SYSTEM_PROMPT
    assert "empty Locked/Unlocked shell" in SURGICAL_SYSTEM_PROMPT
