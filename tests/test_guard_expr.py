"""
tests/test_guard_expr.py

Layer 1 tests: guard expression trees (variable / arithmetic RHS).

  • Expr nodes (Const / VarRef / BinOp) evaluate + report vars
  • _extract_guard builds lhs/rhs for variable-RHS and arithmetic-RHS guards
  • resolve_threshold partial-evaluates RHS against initial_values
  • end-to-end: a state machine with `batteryCharge <= returnEnergyRequired`
    now extracts a fault transition and FIRES (regression that previously
    failed with "No fault transitions found")

Run with:
    python tests/test_guard_expr.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.state_extractor import (
    Const, VarRef, BinOp, GuardCondition,
    extract_state_machines, _SYSIDE_OK,
)
from src.simulation.state_executor import StateMachineInstance
from src.simulation.behavioral_sim import run_behavioral_simulation


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


# ---------------------------------------------------------------------------
# T1 — Expr nodes
# ---------------------------------------------------------------------------

def test_expr_nodes():
    print("T1  Expr nodes")
    ok("const_eval",  Const(5.0).eval({}) == 5.0)
    ok("var_eval",    VarRef("x").eval({"x": 3.0}) == 3.0)
    ok("var_missing", VarRef("x").eval({}) is None)
    ok("binop_add",   BinOp("+", VarRef("a"), Const(300.0)).eval({"a": 10.0}) == 310.0)
    ok("binop_sub",   BinOp("-", Const(10.0), Const(4.0)).eval({}) == 6.0)
    ok("binop_div0",  BinOp("/", Const(1.0), Const(0.0)).eval({}) is None)
    ok("vars_collect",
       BinOp("+", VarRef("a"), VarRef("b")).vars() == ["a", "b"])


# ---------------------------------------------------------------------------
# T2 — GuardCondition.eval with expression trees
# ---------------------------------------------------------------------------

def test_guard_eval():
    print("T2  GuardCondition.eval (var vs var)")
    g = GuardCondition(
        kind="comparison", operator="<=",
        lhs=VarRef("batteryCharge"), rhs=VarRef("returnEnergyRequired"),
    )
    ok("fires_when_le",
       g.eval({"batteryCharge": 10.0, "returnEnergyRequired": 20.0}) is True)
    ok("no_fire_when_gt",
       g.eval({"batteryCharge": 30.0, "returnEnergyRequired": 20.0}) is False)
    # arithmetic RHS
    g2 = GuardCondition(
        kind="comparison", operator=">",
        lhs=VarRef("commLossTime"), rhs=BinOp("+", VarRef("timeToHub"), Const(300.0)),
    )
    ok("arith_rhs_fire",
       g2.eval({"commLossTime": 350.0, "timeToHub": 0.0}) is True)
    ok("arith_rhs_nofire",
       g2.eval({"commLossTime": 100.0, "timeToHub": 0.0}) is False)
    ok("involved_both_sides",
       set(g2.involved_attributes()) == {"commLossTime", "timeToHub"})


# ---------------------------------------------------------------------------
# T3 — resolve_threshold
# ---------------------------------------------------------------------------

def test_resolve_threshold():
    print("T3  resolve_threshold")
    g = GuardCondition(kind="comparison", operator="<=",
                       lhs=VarRef("batteryCharge"), rhs=VarRef("returnEnergyRequired"))
    ok("resolves_var", g.resolve_threshold({"returnEnergyRequired": 25.0}) == 25.0)
    ok("unresolved_when_missing", g.resolve_threshold({}) is None)
    g2 = GuardCondition(kind="comparison", operator=">",
                        lhs=VarRef("t"), rhs=BinOp("+", VarRef("timeToHub"), Const(300.0)))
    ok("resolves_arith", g2.resolve_threshold({"timeToHub": 50.0}) == 350.0)


# ---------------------------------------------------------------------------
# T4 — end-to-end extraction + execution (needs syside)
# ---------------------------------------------------------------------------

_SYSML = """\
package T {
    part def SafetyMonitor {
        attribute batteryCharge : Real = 100.0;
        attribute returnEnergyRequired : Real = 30.0;
        action def returnToBase { }
        state def RtbBatterySafetyBehavior {
            state RtbNominal;
            state RtbFault {
                entry action onFault : returnToBase;
            }
            transition initial then RtbNominal;
            transition rtbFault
                first RtbNominal
                if batteryCharge <= returnEnergyRequired
                then RtbFault;
        }
    }
}
"""


def test_end_to_end_variable_rhs():
    print("T4  end-to-end: variable-RHS guard extracts + fires")
    if not _SYSIDE_OK:
        ok("syside_available", False, "syside not importable — skipping E2E")
        return

    sms = extract_state_machines(_SYSML)
    ok("one_sm", len(sms) == 1, f"got {len(sms)}")
    sm = sms[0]

    # The guard must now be extracted (previously dropped → no fault transition)
    ft = sm.fault_transitions()
    ok("fault_transition_present", len(ft) == 1, f"ft={len(ft)}")
    if ft:
        g = ft[0].guards[0]
        ok("is_comparison", g.kind == "comparison")
        ok("has_expr_trees", g.lhs is not None and g.rhs is not None)
        # threshold resolved from returnEnergyRequired's default (30.0)
        ok("threshold_resolved", abs(g.threshold - 30.0) < 1e-9, f"th={g.threshold}")
        ok("involved_vars",
           set(g.involved_attributes()) == {"batteryCharge", "returnEnergyRequired"})

    # Behavioral simulation should now PASS this state machine
    sim = run_behavioral_simulation(_SYSML, model_name="T")
    rtb = [r for r in sim.scenario_results if "Rtb" in r.name]
    ok("rtb_scenario_exists", len(rtb) == 1, f"results={[r.name for r in sim.scenario_results]}")
    if rtb:
        ok("rtb_passes", rtb[0].passed,
           f"violations={rtb[0].violations}")


if __name__ == "__main__":
    test_expr_nodes()
    test_guard_eval()
    test_resolve_threshold()
    test_end_to_end_variable_rhs()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
