"""Pipeline writers do not manufacture namespace defects.

On the 2026-08-30 authoritative draws, 59 of 61 shadowing warnings sat beside
injection markers: the emitter's bare ``do action X;`` spelling shadowed the
injected part-level ``action def X``, and both the assembly state-def injector
and the planned-behavior materializer re-declared behaviours already present in
the extractor-invisible usage spelling. Cures: typed usage spelling,
spelling-robust presence checks, usage replacement instead of sibling injection,
and a shadow-fingerprint guard that reverts a piece leaving more duplicates or
shadows than before.
"""
from __future__ import annotations

from src.agents.assembly_finalization import AssemblyFinalizer
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    PlannedState,
    PlannedTransition,
    _shadow_fingerprint,
    materialize_owned_planned_behaviors,
    validate_planned_behaviors,
)
from src.simulation.state_extractor import extract_state_machines
from src.simulation.syntax_checker import check_syntax


def _behavior(owner="SafetyMonitor", behavior_id="LockdownBehavior",
              entry_action="lockActuator"):
    return PlannedBehavior(
        owner=owner,
        behavior_id=behavior_id,
        initial_state="Nominal",
        source_requirement_id="REQ-SAFE-006",
        states=(
            PlannedState("Nominal"),
            PlannedState("Locked", entry_action=entry_action),
        ),
        transitions=(
            PlannedTransition(
                "toLocked", "Nominal", "Locked", "ACCEPT", "AbortSignal",
            ),
        ),
    )


_OWNER_SHELL = """package M {{
    item def AbortSignal;
    part def SafetyMonitor {{
{body}
    }}
}}
"""


def test_materialization_shadow_free():
    text, report = materialize_owned_planned_behaviors(
        _OWNER_SHELL.format(body=""), (_behavior(),),
    )
    assert report["status"] == "PASS"
    assert _shadow_fingerprint(text) == (0, 0)
    assert "entry action onLocked : lockActuator;" in text
    assert "entry action lockActuator;" not in text
    machines = extract_state_machines(text)
    assert [m.name for m in machines] == ["LockdownBehavior"]
    assert not check_syntax(text).has_errors


def test_bodied_usage_replaced():
    """Draws #1/#4: the behaviour already existed as a part-level bodied usage
    (`state Name { ... }`), so injecting a def beside it double-declared the name.
    The usage is replaced by the plan-blessed def.
    """
    usage_body = (
        "        state LockdownBehavior {\n"
        "            state Anything;\n"
        "        }\n"
    )
    text, report = materialize_owned_planned_behaviors(
        _OWNER_SHELL.format(body=usage_body), (_behavior(),),
    )
    assert report["status"] == "PASS"
    assert text.count("LockdownBehavior") >= 1
    assert "state def LockdownBehavior" in text
    assert "state LockdownBehavior {\n            state Anything;" not in text
    assert _shadow_fingerprint(text) == (0, 0)


def test_existing_action_not_redeclared():
    body = "        action def lockActuator {}\n"
    text, report = materialize_owned_planned_behaviors(
        _OWNER_SHELL.format(body=body), (_behavior(),),
    )
    assert report["status"] == "PASS"
    assert text.count("action def lockActuator") == 1
    assert _shadow_fingerprint(text) == (0, 0)


def test_guard_reverts_shadowing():
    """Bypass shape: the owner already declares an action def named like the behaviour
    (draw #3's cross-kind pair, arriving via materialization).

    The state-spelling pre-checks cannot see an action def, so adding the state def
    creates the same-name sibling pair syside flags; the fingerprint guard reverts
    the piece and reports it.
    """
    body = "        action def LockdownBehavior {}\n"
    before = _OWNER_SHELL.format(body=body)
    text, report = materialize_owned_planned_behaviors(before, (_behavior(),))
    assert report.get("reverted"), report
    assert "LockdownBehavior" in report["reverted"][0]["behavior"]
    assert "state def LockdownBehavior" not in text
    assert _shadow_fingerprint(text) == _shadow_fingerprint(before)
    assert any("Lockdown" in issue for issue in report["issues"])


def test_assembly_injector_reverts():
    inject = AssemblyFinalizer._inject_missing_state_defs

    fragment = (
        "// OWNER: SafetyMonitor\n"
        "state def GuardBehavior {\n"
        "    entry; then Idle;\n"
        "    state Idle;\n"
        "    state Acting { do action guardAct; }\n"
        "    transition t first Idle accept Go then Acting;\n"
        "}\n"
    )

    assembled_usage = (
        "package M {\n"
        "    part def SafetyMonitor {\n"
        "        state GuardBehavior {\n"
        "            state Idle;\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    result, injected = inject(assembled_usage, fragment)
    assert injected == []
    assert "injected by pipeline" not in result

    # 2. absent -> injected, and the bare `do action guardAct;` fragment does
    #    not collide because no part-level guardAct exists
    assembled_absent = (
        "package M {\n"
        "    part def SafetyMonitor {\n"
        "        attribute x : Real = 1.0;\n"
        "    }\n"
        "}\n"
    )
    result, injected = inject(assembled_absent, fragment)
    assert injected == ["GuardBehavior"]
    assert "// (injected by pipeline)" in result
    assert _shadow_fingerprint(result) == (0, 0)

    # 3. the part declares `action def GuardBehavior`; a same-name state def is the
    #    cross-kind sibling pair the terminal gate rejects, and no state-spelling
    #    pre-check sees an action def -> the fingerprint guard reverts the piece
    assembled_colliding = (
        "package M {\n"
        "    part def SafetyMonitor {\n"
        "        action def GuardBehavior {}\n"
        "    }\n"
        "}\n"
    )
    result, injected = inject(assembled_colliding, fragment)
    assert injected == []
    assert result == assembled_colliding


def test_plan_rejects_id_collision():
    colliding = (
        _behavior(behavior_id="LockdownBehavior", entry_action="lockActuator"),
        _behavior(behavior_id="OtherBehavior",
                  entry_action="LockdownBehavior"),
    )
    issues = validate_planned_behaviors(
        colliding,
        component_names={"SafetyMonitor"},
        requirements=["REQ-SAFE-006: keep the payload locked on abort."],
        state_active_constraints=(),
        require_executable_responses=False,
    )
    assert any(
        "LockdownBehavior" in issue and "share one name" in issue
        for issue in issues
    )

    different_owner = (
        _behavior(owner="A", behavior_id="LockdownBehavior"),
        _behavior(owner="B", behavior_id="OtherBehavior",
                  entry_action="LockdownBehavior"),
    )
    issues = validate_planned_behaviors(
        different_owner,
        component_names={"A", "B"},
        requirements=["REQ-SAFE-006: keep the payload locked on abort."],
        state_active_constraints=(),
        require_executable_responses=False,
    )
    assert not any("share one name" in issue for issue in issues)
