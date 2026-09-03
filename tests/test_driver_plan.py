"""Layer 2 tests: multi-variable driver plan.

For a bare `A OP B` guard (both sides variables) the simulator generates two
driver plans: drive_LHS sweeps A across B's held value, drive_RHS sweeps B so
the relation reverses.

Run with:
    python tests/test_driver_plan.py
"""

from __future__ import annotations

import sys

from src.simulation.state_extractor import (
    GuardCondition, StateMachineDef, StateNode, TransitionDef,
    VarRef, Const, BinOp, extract_state_machines, _SYSIDE_OK,
)
from src.simulation.behavioral_sim import (
    _build_driver_plans, _guard_endpoints, _build_test_sequence,
    run_behavioral_simulation,
)


_PASS = 0
_FAIL = 0


def ok(name: str, cond: bool, msg: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        print(f"  PASS  {name}")
        _PASS += 1
    else:
        print(f"  FAIL  {name}  {msg}")
        _FAIL += 1
    assert cond, f"{name}: {msg}"


def _sm_with_guard(guard: GuardCondition, initial_values: dict) -> StateMachineDef:
    sm = StateMachineDef(
        name="Beh", owner_part="Owner",
        initial_state="Nominal",
        initial_values=initial_values,
    )
    sm.states = [
        StateNode(name="Nominal"),
        StateNode(name="Fault", entry_action="onFault"),
    ]
    sm.transitions = [
        TransitionDef(name=None, source=None, target="Nominal",
                      guards=[], is_initial=True),
        TransitionDef(name="f", source="Nominal", target="Fault",
                      guards=[guard], is_initial=False),
    ]
    return sm


def test_guard_endpoints():
    print("T1  _guard_endpoints")
    g = GuardCondition(kind="comparison", operator="<=",
                       lhs=VarRef("a"), rhs=VarRef("b"))
    ok("both_vars", _guard_endpoints(g) == ("a", "b"))

    g2 = GuardCondition(kind="comparison", operator="<",
                        lhs=VarRef("a"), rhs=Const(10.0))
    ok("lhs_only", _guard_endpoints(g2) == ("a", None))

    g3 = GuardCondition(kind="comparison", operator=">",
                        lhs=VarRef("a"),
                        rhs=BinOp("+", VarRef("b"), Const(5.0)))
    ok("rhs_arith_not_bare", _guard_endpoints(g3) == ("a", None))


def test_two_plans_for_var_vs_var():
    print("T2  A OP B → 两条驱动轨迹")
    guard = GuardCondition(
        kind="comparison", operator="<=",
        lhs=VarRef("a"), rhs=VarRef("b"),
    )
    sm = _sm_with_guard(guard, initial_values={"a": 100.0, "b": 30.0})
    plans = _build_driver_plans(sm)

    ok("two_plans", len(plans) == 2, f"got {len(plans)}")
    names = [p.name for p in plans]
    ok("has_drive_lhs", "drive_LHS" in names)
    ok("has_drive_rhs", "drive_RHS" in names)

    p_lhs = [p for p in plans if p.name == "drive_LHS"][0]
    p_rhs = [p for p in plans if p.name == "drive_RHS"][0]
    ok("lhs_sweeps_a", p_lhs.swept_var == "a")
    ok("rhs_sweeps_b", p_rhs.swept_var == "b")

    held_b_values = {s.get("b") for s in p_lhs.sequence}
    ok("lhs_holds_b", held_b_values == {30.0}, f"b values seen: {held_b_values}")

    held_a_values = {s.get("a") for s in p_rhs.sequence}
    ok("rhs_holds_a", held_a_values == {100.0}, f"a values seen: {held_a_values}")


def test_one_plan_for_const_rhs():
    print("T3  A OP <const> → 单驱动轨迹(向后兼容)")
    guard = GuardCondition(
        kind="comparison", operator="<", attribute="a",
        lhs=VarRef("a"), rhs=Const(15.0), threshold=15.0,
    )
    sm = _sm_with_guard(guard, initial_values={"a": 100.0})
    plans = _build_driver_plans(sm)
    ok("one_plan", len(plans) == 1, f"got {len(plans)}")
    ok("name", plans[0].name == "drive_LHS")
    ok("swept_a", plans[0].swept_var == "a")


def test_unresolved_threshold_no_plan():
    guard = GuardCondition(
        kind="comparison",
        operator="<=",
        attribute="currentSeparation",
        lhs=VarRef("currentSeparation"),
        rhs=BinOp("+", VarRef("minimumSeparation"), Const(5.0)),
    )
    sm = _sm_with_guard(guard, initial_values={})

    assert _build_driver_plans(sm) == []


def test_unresolved_threshold_fails_closed():
    if not _SYSIDE_OK:
        return
    sysml = """\
package T {
    part def Mon {
        attribute currentSeparation : Real = 20 [m];
        attribute minimumSeparation : Real;
        action def onF { }
        state def Beh {
            state Nominal;
            state Fault { entry action ea : onF; }
            transition initial then Nominal;
            transition f first Nominal
                if currentSeparation <= minimumSeparation + 5.0
                then Fault;
        }
    }
}
"""

    simulation = run_behavioral_simulation(sysml, model_name="T")
    scenario = next(
        item for item in simulation.scenario_results
        if item.name == "Beh"
    )

    assert not scenario.passed
    assert scenario.violations == [
        "Could not resolve guard threshold from declared numeric initial "
        "values; simulation did not assume 0.0 for: minimumSeparation"
    ]


def test_quantity_default_resolves_guard():
    if not _SYSIDE_OK:
        return
    sysml = """\
package T {
    item def ObstacleData {
        attribute separation : Real = 0.0 [m];
    }
    port def ObstaclePort {
        item payload : ObstacleData;
    }
    part def Mon {
        in port obstacleData : ObstaclePort;
        attribute currentSeparation : Real =
            obstacleData.payload.separation;
        attribute minimumSeparation : Real = 5 [m];
        action def onF { }
        state def Beh {
            state Nominal;
            state Fault { entry action ea : onF; }
            transition initial then Nominal;
            transition f first Nominal
                if currentSeparation <= minimumSeparation + 5.0
                then Fault;
        }
    }
}
"""
    machines = extract_state_machines(sysml)
    assert len(machines) == 1
    machine = machines[0]
    assert machine.initial_values["minimumSeparation"] == 5.0
    assert "currentSeparation" not in machine.initial_values

    plans = _build_driver_plans(machine)
    assert len(plans) == 1
    assert machine.fault_transitions()[0].guards[0].threshold == 10.0
    assert all(
        step["minimumSeparation"] == 5.0
        for step in plans[0].sequence
    )

    simulation = run_behavioral_simulation(sysml, model_name="T")
    scenario = next(
        item for item in simulation.scenario_results
        if item.name == "Beh"
    )
    assert scenario.passed
    assert any(
        "threshold: <= 10.0" in line
        for line in scenario.timeline
    )


def test_bool_compat():
    print("T4  bool / compound 兼容")
    g_bool = GuardCondition(kind="bool_true", attribute="flag")
    sm = _sm_with_guard(g_bool, initial_values={})
    plans = _build_driver_plans(sm)
    ok("bool_one_plan", len(plans) == 1)
    ok("bool_name", plans[0].name == "flip_bool")
    seq = _build_test_sequence(sm)
    ok("shim_returns_seq", len(seq) == 20, f"len={len(seq)}")


def test_bool_false_driver_crosses():
    guard = GuardCondition(kind="bool_false", attribute="systemSafe")
    sm = _sm_with_guard(guard, initial_values={"systemSafe": True})

    plans = _build_driver_plans(sm)

    assert len(plans) == 1
    assert plans[0].name == "flip_bool_false"
    assert plans[0].sequence[0] == {"systemSafe": True}
    assert plans[0].sequence[-1] == {"systemSafe": False}


def test_mixed_guard_executes():
    if not _SYSIDE_OK:
        return
    sysml = """\
package T {
    part def PayloadManager {
        attribute waypointDistance : Real = 10.0;
        attribute abortActive : Boolean = false;
        action def releasePayload {}
        state def ReleaseMachine {
            state locked;
            state released {
                entry action release : releasePayload;
            }
            transition initial then locked;
            transition release first locked
                if waypointDistance <= 1.0 and not abortActive
                then released;
        }
    }
}
"""

    simulation = run_behavioral_simulation(sysml, model_name="T")
    scenario = next(
        item for item in simulation.scenario_results
        if item.name == "ReleaseMachine"
    )

    assert scenario.passed
    assert any(
        "abortActive=False" in line
        for line in scenario.timeline
    ) or scenario.fired_actions


def test_e2e_both_sides_fire():
    print("T5  端到端:A<=B 两个方向都能触发(关系真的被验证)")
    if not _SYSIDE_OK:
        ok("syside_available", False, "syside not importable — skipping E2E")
        return

    sysml = """\
package T {
    part def Mon {
        attribute a : Real = 100.0;
        attribute b : Real = 30.0;
        action def onF { }
        state def Beh {
            state Nominal;
            state Fault { entry action ea : onF; }
            transition initial then Nominal;
            transition f first Nominal if a <= b then Fault;
        }
    }
}
"""
    sim = run_behavioral_simulation(sysml, model_name="T")
    rs = [r for r in sim.scenario_results if r.name == "Beh"]
    ok("scenario_present", len(rs) == 1, f"got {[r.name for r in sim.scenario_results]}")
    r = rs[0]
    ok("scenario_passes", r.passed, f"violations={r.violations}")
    tl = " ".join(r.timeline)
    ok("drive_lhs_in_timeline", "drive_LHS" in tl, f"timeline:\n{tl}")
    ok("drive_rhs_in_timeline", "drive_RHS" in tl, f"timeline:\n{tl}")


def test_e2e_one_sided_relation():
    """With a=100, b=200 and guard `a <= b` the relation already holds.

    drive_LHS fires on the first variable binding; drive_RHS pushes b below 100
    because the flipped operator drives b down. Either way at least one plan fires.
    """
    print("T6  端到端:初值已满足关系 — 仍 pass")
    if not _SYSIDE_OK:
        ok("syside_available", False, "syside not importable — skipping E2E")
        return

    sysml = """\
package T {
    part def Mon {
        attribute a : Real = 100.0;
        attribute b : Real = 200.0;
        action def onF { }
        state def Beh {
            state Nominal;
            state Fault { entry action ea : onF; }
            transition initial then Nominal;
            transition f first Nominal if a <= b then Fault;
        }
    }
}
"""
    sim = run_behavioral_simulation(sysml, model_name="T")
    rs = [r for r in sim.scenario_results if r.name == "Beh"]
    ok("scenario_present", len(rs) == 1)
    ok("scenario_passes", rs[0].passed, f"violations={rs[0].violations}")


if __name__ == "__main__":
    test_guard_endpoints()
    test_two_plans_for_var_vs_var()
    test_one_plan_for_const_rhs()
    test_bool_compat()
    test_e2e_both_sides_fire()
    test_e2e_one_sided_relation()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
