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
    row = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]

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
    row = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]

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
    row = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]

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


def test_unnamed_attribute_redefinition_does_not_break_the_linker():
    """A redefinition is legal SysML and reports no name of its own.

    Keying the linker's attribute map on that name put a None into every
    later keyword scan, so one such line anywhere in the model crashed the
    verification audit — which fails closed — and blocked the whole run.
    """
    text, model = _model("""
        attribute maxMass : Real = 5.0;
        attribute :>> maxMass = 7.0;
        satisfy requirement REQ_SAFE_008;
    """)

    linker = RequirementLinker(model)

    assert all(
        name is not None
        for attrs in linker._attr_map.values()
        for name in attrs
    )
    # the audit must reach a verdict rather than raise
    verification_gap_issues(text, "D", strict=True)
    build_matrix(model, None, linker.compile_evidence())


# ─────────────────────────────────────────────────────────────────────────────
#  Power-on shape: initial phase state, every exit driven onto the default
# ─────────────────────────────────────────────────────────────────────────────

def _extract_machine(text, name="PayloadLockMachine"):
    from src.simulation.state_extractor import extract_state_machines
    return next(sm for sm in extract_state_machines(text) if sm.name == name)


_POWER_ON_BODY = """
    action def lockPayload {}
    state def PayloadLockMachine {
        entry; then PowerOn;
        state PowerOn;
        state MechanicallyLocked {
            entry action onLocked : lockPayload;
        }
        transition initializeLock
            first PowerOn
            accept PowerOnEvent
            then MechanicallyLocked;
    }
"""


def test_power_on_shaped_machine_drives_the_event_and_passes():
    """s0v15's shape: initial PowerOn --accept PowerOnEvent--> Locked{entry}.
    The check drives the power event instead of failing the phase state."""
    from src.simulation.behavioral_sim import run_initialization_scenario

    text, _ = _model(_POWER_ON_BODY)
    scenario = run_initialization_scenario(
        _extract_machine(text), required_state_terms={"locked"}
    )

    assert scenario.passed, scenario.violations
    assert any("Drove power event" in line for line in scenario.timeline)
    assert "onLocked" in scenario.fired_actions
    # And the in-loop audit that spent s0v15's whole iteration budget:
    assert not [
        i for i in verification_gap_issues(text, model_name="D")
        if "REQ_SAFE_008" in i
    ]


def test_power_on_shape_with_escape_edge_still_fails():
    """An exit from the initial state to a non-default state means the machine
    can leave power-on without passing the default — "before any arming"."""
    from src.simulation.behavioral_sim import run_initialization_scenario

    text, _ = _model("""
        action def lockPayload {}
        action def releasePayload {}
        state def PayloadLockMachine {
            entry; then PowerOn;
            state PowerOn;
            state MechanicallyLocked {
                entry action onLocked : lockPayload;
            }
            state Released {
                entry action onReleased : releasePayload;
            }
            transition initializeLock
                first PowerOn
                accept PowerOnEvent
                then MechanicallyLocked;
            transition earlyRelease
                first PowerOn
                accept ReleaseCommand
                then Released;
            transition release
                first MechanicallyLocked
                accept ReleaseCommand
                then Released;
        }
    """)
    scenario = run_initialization_scenario(
        _extract_machine(text), required_state_terms={"locked"}
    )

    assert not scenario.passed
    assert any("no observable initialization semantics" in v
               for v in scenario.violations)


def test_power_on_shape_with_only_guarded_exits_still_fails():
    """A default reached only through a guard is conditional, not a default."""
    from src.simulation.behavioral_sim import run_initialization_scenario

    text, _ = _model("""
        attribute selfTestPassed : Boolean = false;
        action def lockPayload {}
        state def PayloadLockMachine {
            entry; then PowerOn;
            state PowerOn;
            state MechanicallyLocked {
                entry action onLocked : lockPayload;
            }
            transition initializeLock
                first PowerOn
                if selfTestPassed
                then MechanicallyLocked;
        }
    """)
    scenario = run_initialization_scenario(
        _extract_machine(text), required_state_terms={"locked"}
    )

    assert not scenario.passed
    assert any("guarded" in v for v in scenario.violations)


def test_power_on_shape_needs_semantics_on_the_landed_state():
    """Driving the event onto a bare state name is still an empty shell."""
    from src.simulation.behavioral_sim import run_initialization_scenario

    text, _ = _model("""
        state def PayloadLockMachine {
            entry; then PowerOn;
            state PowerOn;
            state MechanicallyLocked;
            transition initializeLock
                first PowerOn
                accept PowerOnEvent
                then MechanicallyLocked;
        }
    """)
    scenario = run_initialization_scenario(
        _extract_machine(text), required_state_terms={"locked"}
    )

    assert not scenario.passed
    assert any("no observable semantics" in v for v in scenario.violations)


def test_plan_binding_selects_the_machine_vocabulary_excludes():
    """Initial state 'BootPhase' appears nowhere in the requirement text, so
    the vocabulary fallback cannot select the machine; the plan's recorded
    requirement->behavior binding must."""
    text, model = _model("""
        action def lockPayload {}
        state def PayloadLockMachine {
            entry; then BootPhase;
            state BootPhase;
            state MechanicallyLocked {
                entry action onLocked : lockPayload;
            }
            transition initializeLock
                first BootPhase
                accept PowerOnEvent
                then MechanicallyLocked;
        }
    """)

    _INIT_DETAIL = "initial/default-state invariant exercised at behavioral-sim tier"

    unbound = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]
    # Without the binding the row may still earn behavioral_sim from the
    # generic requirement-linked scenario route; what it cannot earn is the
    # initialization-invariant evidence — no machine was selected for it.
    assert not any(_INIT_DETAIL in e for e in unbound.evidence)

    # NB: assign the plan as a key on the existing metadata dict — the lite
    # model's metadata already carries last_sysml_text, which to_sysml_text()
    # serialises from; replacing the dict silently empties the model.
    model.metadata["whole_model_generation_plan"] = {
        "requirement_realizations": [{
            "requirement_id": "REQ_SAFE_008",
            "owner_component": "PayloadMechanism",
            "behavior_name": "PayloadLockMachine",
        }],
    }
    bound = build_matrix(model, None, RequirementLinker(model).compile_evidence())[0]

    assert any(_INIT_DETAIL in e and "(PASS)" in e for e in bound.evidence), (
        bound.evidence
    )
    assert "behavioral_sim_failed" not in bound.tiers
