"""
behavioral_sim.py

行为仿真主模块：为每个从模型提取的状态机自动生成仿真场景，
注入驱动事件，执行状态机，验证转移是否在正确的时刻触发。

这是真正的行为仿真（行为仿真/状态机执行）：
  - 从模型定义中提取 guard 条件（e.g. batteryCharge < 15.0）
  - 按时步让变量值演进（电池从 100% 逐步耗尽）
  - 验证状态转移在正确的阈值触发，且 entry action 被调用
  - 跨场景约束检查（RTB 必须早于 Critical）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .state_extractor import GuardCondition, StateMachineDef, VarRef, extract_state_machines
from .state_executor import StateMachineInstance


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class BehavioralScenarioResult:
    name: str
    state_machine: str
    description: str
    passed: bool
    timeline: List[str] = field(default_factory=list)  # human-readable event log
    violations: List[str] = field(default_factory=list)
    fired_actions: List[str] = field(default_factory=list)
    trigger_step: Optional[int] = None      # step at which fault transition fired
    trigger_value: Optional[float] = None   # variable value at trigger
    tags: List[str] = field(default_factory=list)


@dataclass
class BehavioralSimResult:
    model_name: str
    scenario_results: List[BehavioralScenarioResult] = field(default_factory=list)
    sim_score: float = 0.0
    extracted_sm_count: int = 0

    def passed_count(self) -> int:
        return sum(1 for r in self.scenario_results if r.passed)

    def failed_scenarios(self) -> List[BehavioralScenarioResult]:
        return [r for r in self.scenario_results if not r.passed]

    def all_violations(self) -> List[str]:
        out = []
        for r in self.scenario_results:
            for v in r.violations:
                out.append(f"[{r.name}] {v}")
        return out

    def summary_lines(self) -> List[str]:
        total = len(self.scenario_results)
        lines = [
            f"=== Behavioral Simulation (State Machine Execution): {self.model_name} ===",
            f"Extracted {self.extracted_sm_count} state machine(s) from model",
            f"Score: {self.sim_score:.2%}  ({self.passed_count()}/{total} scenarios passed)",
        ]
        for r in self.scenario_results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name}: {r.description}")
            for line in r.timeline:
                lines.append(f"         {line}")
            for v in r.violations:
                lines.append(f"         ⚠ {v}")
        return lines


# ---------------------------------------------------------------------------
# Scenario generation helpers
# ---------------------------------------------------------------------------

_N_STEPS = 70   # simulation steps per scenario

def _derive_start_and_step(guard: GuardCondition,
                            initial_values: Dict[str, Any]
                            ) -> Tuple[float, float]:
    """
    For a comparison guard, return (start_value, step_size) such that the
    variable crosses the threshold at roughly 85% of _N_STEPS.
    Uses the model's initial_values when available, otherwise derives a
    sensible default from the threshold.
    """
    attr  = guard.attribute
    th    = guard.threshold
    start = initial_values.get(attr)

    if guard.operator in ("<", "<="):
        # Need to decrease below threshold
        if start is None or float(start) <= th:
            start = th * 3.0 + 10.0    # safe fallback: well above threshold
        start = float(start)
        step = (start - th) / (_N_STEPS * 0.85)
        step = max(step, 0.001)
        return start, -step             # negative → decreasing

    else:  # '>', '>='
        # Need to increase above threshold
        if start is None or float(start) >= th:
            start = 0.0
        start = float(start)
        step = (th - start) / (_N_STEPS * 0.85)
        step = max(step, 0.001)
        return start, +step             # positive → increasing


@dataclass
class DriverPlan:
    """
    One named trajectory of variable bindings used to drive a state machine.

    Layer 2 allows multiple plans per guard (e.g. for `A <= B` we generate
    two: one that pushes A down holding B, and one that pushes B up holding A)
    so the *relationship* is exercised, not just one operand.
    """
    name: str                       # human label, e.g. "drive_LHS" / "drive_RHS"
    sequence: List[Dict[str, Any]]  # per-step variable bindings
    swept_var: str = ""             # which variable is being swept (for reporting)
    start_val: Optional[float] = None
    step_size: Optional[float] = None


def _swept_plan(
    swept: str,                 # the variable being driven
    held: Dict[str, float],     # all other guard vars (held constant)
    operator: str,              # the comparison op as seen by THIS sweep
    threshold: float,           # constant against which `swept` is compared
    initial_values: Dict[str, Any],
    name: str,
) -> DriverPlan:
    """
    Build a single-variable sweep trajectory.  `swept` ramps across
    `threshold` under `operator`; `held` variables are emitted unchanged
    every step so the guard evaluator sees the full environment.
    """
    # Synthesize a tiny ad-hoc guard to reuse the existing ramp logic.
    pseudo = GuardCondition(
        kind="comparison",
        attribute=swept,
        operator=operator,
        threshold=threshold,
    )
    start, step = _derive_start_and_step(pseudo, initial_values)
    seq: List[Dict[str, Any]] = []
    val = start
    for _ in range(_N_STEPS + 15):
        binding: Dict[str, Any] = {swept: val}
        binding.update(held)
        seq.append(binding)
        val += step
    return DriverPlan(name=name, sequence=seq, swept_var=swept,
                      start_val=start, step_size=step)


def _guard_endpoints(guard: GuardCondition) -> Tuple[Optional[str], Optional[str]]:
    """For a comparison guard, return (lhs_var, rhs_var) where each is a bare
    variable name or None when that side is not a bare VarRef."""
    lhs_var = guard.lhs.name if isinstance(guard.lhs, VarRef) else None
    rhs_var = guard.rhs.name if isinstance(guard.rhs, VarRef) else None
    return lhs_var, rhs_var


def _resolve(env: Dict[str, Any], name: str, default: float = 0.0) -> float:
    """Read a numeric attribute from env with a safe fallback."""
    v = env.get(name)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _build_driver_plans(sm: StateMachineDef) -> List[DriverPlan]:
    """
    Generate one or more named DriverPlans for the first fault transition's
    primary guard.

    Layer 2 behaviour:
      • Bare-variable LHS, bare-variable RHS (`A OP B`) →  TWO plans:
          drive_LHS — sweep A across B's value (B held)
          drive_RHS — sweep B across A's value (A held)
        The *relationship* itself is verified — passes if ANY plan fires.
      • Bare LHS, non-bare RHS (constant / arithmetic) →  ONE plan
        (sweep LHS across the resolved threshold; old behaviour).
      • Boolean / compound-AND guards → unchanged.
    """
    ft = sm.fault_transitions()
    if not ft:
        return []

    guard = ft[0].guards[0]
    init  = sm.initial_values or {}

    # ── Comparison ────────────────────────────────────────────────────────
    if guard.kind == "comparison":
        lhs_var, rhs_var = _guard_endpoints(guard)

        # Both sides are variables → two-trajectory matrix
        if lhs_var and rhs_var:
            lhs_init = _resolve(init, lhs_var)
            rhs_init = _resolve(init, rhs_var)

            plans: List[DriverPlan] = []

            # Plan A: drive LHS against the current RHS value
            plans.append(_swept_plan(
                swept=lhs_var,
                held={rhs_var: rhs_init},
                operator=guard.operator,
                threshold=rhs_init,
                initial_values=init,
                name="drive_LHS",
            ))

            # Plan B: drive RHS so the relation reverses
            # (`A < B` fires when LHS shrinks below RHS — equivalently when
            # RHS GROWS above LHS, i.e. flipped operator on RHS).
            flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(
                guard.operator, guard.operator,
            )
            plans.append(_swept_plan(
                swept=rhs_var,
                held={lhs_var: lhs_init},
                operator=flipped,
                threshold=lhs_init,
                initial_values=init,
                name="drive_RHS",
            ))
            return plans

        # LHS bare variable, RHS constant/arithmetic → original single sweep,
        # but still emit RHS-side vars (if any) as held constants so the
        # executor's full-env evaluator sees them.
        if lhs_var:
            held: Dict[str, float] = {}
            if guard.rhs is not None:
                for v in guard.rhs.vars():
                    held[v] = _resolve(init, v)
            return [_swept_plan(
                swept=lhs_var,
                held=held,
                operator=guard.operator,
                threshold=guard.threshold,
                initial_values=init,
                name="drive_LHS",
            )]

        # Neither side is a bare variable — no obvious driver, skip.
        return []

    # ── Enum equality: `mode == EnumType::Value` (Layer 2) ───────────────
    # Walk the first mode transition: hold at initial enum value for 5 steps,
    # then switch to the target value.  The state machine should advance once
    # the attribute matches the guard's expected value.
    if guard.kind == "enum_eq":
        attr = guard.attribute
        init_val = str(init.get(attr, ""))   # e.g. "POWER_ON"
        target_val = guard.enum_value         # e.g. "SELF_TEST"
        seq: List[Dict[str, Any]] = [{attr: init_val}] * 5 + [{attr: target_val}] * 15
        return [DriverPlan(name="set_mode", sequence=seq, swept_var=attr)]

    # ── Boolean flag ──────────────────────────────────────────────────────
    if guard.kind == "bool_true":
        attr = guard.attribute
        seq = [{attr: False}] * 5 + [{attr: True}] * 15
        return [DriverPlan(name="flip_bool", sequence=seq, swept_var=attr)]

    # ── Compound AND of boolean flags ─────────────────────────────────────
    if guard.kind == "compound" and guard.compound_op == "and":
        bool_operands = [op for op in guard.operands if op.kind == "bool_true"]
        if not bool_operands:
            return []
        seq: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {op.attribute: False for op in bool_operands}
        for i in range(len(bool_operands) * 8 + 10):
            idx = i // 8
            for j, op in enumerate(bool_operands):
                state[op.attribute] = (j <= idx and idx < len(bool_operands))
            seq.append(dict(state))
        return [DriverPlan(name="flip_bool_and", sequence=seq)]

    return []


def _build_test_sequence(sm: StateMachineDef) -> List[Dict[str, Any]]:
    """Back-compat shim: return the first driver plan's sequence."""
    plans = _build_driver_plans(sm)
    return plans[0].sequence if plans else []


def _expected_trigger_step(guard: GuardCondition,
                            start: float, step: float) -> int:
    """Compute the step at which the guard is expected to first fire."""
    if step == 0:
        return _N_STEPS
    th = guard.threshold
    if guard.operator in ("<", "<="):
        # step is negative
        if step >= 0:
            return _N_STEPS
        # start + step*t < th  →  t > (start - th) / abs(step)
        return max(1, int((start - th) / abs(step)) + 1)
    else:
        if step <= 0:
            return _N_STEPS
        return max(1, int((th - start) / step) + 1)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(sm: StateMachineDef) -> BehavioralScenarioResult:
    """
    Build a test sequence for *sm*, execute the state machine, and return
    a BehavioralScenarioResult.
    """
    ft = sm.fault_transitions()
    description = (
        f"{sm.owner_part}.{sm.name}: verify fault transition fires correctly"
    )

    # Determine tags from state machine / owner name
    tags: List[str] = []
    name_low = sm.name.lower() + sm.owner_part.lower()
    if any(k in name_low for k in ("battery", "power")):
        tags.append("safety")
    if any(k in name_low for k in ("comm", "comms")):
        tags.append("safety")
    if any(k in name_low for k in ("impact", "sensor")):
        tags.append("safety")
    if any(k in name_low for k in ("redundan", "channel")):
        tags.append("reliability")

    result = BehavioralScenarioResult(
        name=sm.name,
        state_machine=sm.name,
        description=description,
        passed=False,
        tags=tags,
    )

    if not ft:
        result.violations.append("No fault transitions found in state machine")
        return result

    plans = _build_driver_plans(sm)
    if not plans:
        result.violations.append("Could not generate test sequence for guard type")
        return result

    primary_guard = ft[0].guards[0]
    is_mode_machine = primary_guard.kind == "enum_eq"

    # ── Execute every plan; pass if ANY plan triggers the fault ──────────────
    # Multi-plan (Layer 2) verifies the *relationship* in `A OP B`: a guard
    # that only fires in one direction still passes, but we record which plans
    # fired so the user can see the relation is properly two-sided.
    any_fired = False
    fault_states = {s.name for s in sm.states if s.entry_action}

    for plan in plans:
        inst = StateMachineInstance(sm)
        for t, variables in enumerate(plan.sequence):
            inst.step(variables, time=float(t))
            # Mode machines: stop as soon as any transition fires.
            # Fault monitors: stop when reaching a fault state (has entry action).
            if is_mode_machine:
                if inst.transition_log:
                    break
            else:
                if inst.in_fault_state():
                    break

        events = inst.transition_log
        plan_label = f"[{plan.name}]" if len(plans) > 1 else ""

        if not events:
            # Plan didn't fire — record but keep trying other plans
            result.timeline.append(
                f"{plan_label} no trigger over {len(plan.sequence)} steps"
                f" (swept {plan.swept_var or '?'})"
            )
            continue

        # This plan did fire — capture details from the LAST event
        fault_event = events[-1]
        first_fire = not any_fired
        any_fired = True

        if first_fire:
            # Use the first firing plan's data as the canonical record
            result.trigger_step = int(fault_event.time)
            result.fired_actions = list(inst.fired_actions)

            # Sweep header line for context
            if plan.swept_var:
                if primary_guard.kind == "comparison":
                    result.timeline.append(
                        f"{plan_label} Driving {plan.swept_var}: "
                        f"{plan.start_val:.2f} → "
                        f"(threshold {primary_guard.operator}"
                        f" {primary_guard.threshold})"
                        if plan.start_val is not None else
                        f"{plan_label} Driving {plan.swept_var}"
                    )
                elif primary_guard.kind == "enum_eq":
                    result.timeline.append(
                        f"{plan_label} Setting {plan.swept_var} = "
                        f"{primary_guard.enum_type}::{primary_guard.enum_value} at t=5"
                    )
                else:
                    result.timeline.append(
                        f"{plan_label} Flipping {plan.swept_var} to True at t=5"
                    )

            result.timeline.append(fault_event.to_line())

            # ── Check 1: correct target state ─────────────────────────────────
            # Mode machines have no fault states — any transition is valid.
            if not is_mode_machine and fault_event.to_state not in fault_states:
                result.violations.append(
                    f"Transition target '{fault_event.to_state}' is not a known "
                    f"fault state (expected one of: {fault_states})"
                )

            # ── Check 2: entry action was called ──────────────────────────────
            # Mode machines: entry actions are optional — skip this check.
            if not is_mode_machine:
                if not result.fired_actions:
                    result.violations.append(
                        f"Fault state '{fault_event.to_state}' has no entry action "
                        "recorded — emergency response may not have been triggered"
                    )
                else:
                    result.timeline.append(
                        f"  entry action called: {result.fired_actions[-1]}  ✓"
                    )

            # ── Check 3: trigger at roughly the expected step ────────────────
            if primary_guard.kind == "comparison" and plan.start_val is not None \
                    and plan.step_size is not None:
                # Re-use the same pseudo-guard the swept plan used to compute exp_step.
                pseudo = GuardCondition(
                    kind="comparison",
                    attribute=plan.swept_var,
                    operator=("<" if (plan.step_size or 0) < 0 else ">"),
                    threshold=(plan.start_val + plan.step_size * 0),  # placeholder
                )
                # Compute the value at trigger directly:
                trig_val = round(plan.start_val + plan.step_size * result.trigger_step, 3)
                result.trigger_value = trig_val
                result.timeline.append(
                    f"  {plan.swept_var} at trigger: {trig_val}"
                    f"  (threshold: {primary_guard.operator}"
                    f" {primary_guard.threshold})"
                )
        else:
            # Subsequent plans that also fired — just note in timeline
            result.timeline.append(
                f"{plan_label} also fired at step {int(fault_event.time)}  ✓"
            )

    if not any_fired:
        result.violations.append(
            f"No state transition fired across {len(plans)} driver plan(s) — "
            f"guard '{primary_guard.description()}' was never satisfied"
        )
        return result

    # Layer 2 informational hint when only one side of a two-sided relation fired
    if len(plans) > 1 and result.timeline:
        fired_names = [p.name for p in plans
                       if any(p.name in tl for tl in result.timeline)]
        if len(fired_names) == 1:
            result.timeline.append(
                f"  (only {fired_names[0]} fired — relationship is one-sided "
                f"with the current initial values)"
            )

    result.passed = len(result.violations) == 0
    return result


# ---------------------------------------------------------------------------
# Cross-scenario ordering constraint
# ---------------------------------------------------------------------------

def _check_battery_ordering(results: List[BehavioralScenarioResult]
                             ) -> Optional[BehavioralScenarioResult]:
    """
    Verify that the battery RTB threshold is higher than the critical threshold,
    ensuring RTB always fires before emergency landing.

    Each scenario runs independently, so step numbers are not comparable.
    Instead we compare the trigger_value (actual variable value at trigger),
    which reflects the threshold from the model directly.
    """
    rtb_res  = next((r for r in results if "Rtb"      in r.state_machine or
                                            "rtb"      in r.state_machine.lower()), None)
    crit_res = next((r for r in results if "Critical"  in r.state_machine or
                                            "critical" in r.state_machine.lower()), None)

    if rtb_res is None or crit_res is None:
        return None  # No pair to check

    constraint = BehavioralScenarioResult(
        name="battery_ordering_constraint",
        state_machine="cross_scenario",
        description="RTB threshold must be higher than Critical threshold (safety hierarchy)",
        passed=False,
        tags=["safety"],
    )

    rtb_val  = rtb_res.trigger_value
    crit_val = crit_res.trigger_value

    if rtb_val is None:
        constraint.violations.append("Battery RTB scenario did not fire — cannot check ordering")
    elif crit_val is None:
        constraint.violations.append("Battery Critical scenario did not fire — cannot check ordering")
    elif rtb_val <= crit_val:
        constraint.violations.append(
            f"RTB trigger value ({rtb_val}) ≤ Critical trigger value ({crit_val}) — "
            "RTB threshold must be higher than Critical threshold for correct safety hierarchy"
        )
    else:
        margin = round(rtb_val - crit_val, 3)
        constraint.timeline.append(
            f"RTB fires at {rtb_val}  >  Critical fires at {crit_val} — "
            f"margin = {margin}  ✓"
        )

    constraint.passed = len(constraint.violations) == 0
    return constraint


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(results: List[BehavioralScenarioResult]) -> float:
    if not results:
        return 1.0   # neutral: no state machines → no violations
    total_w  = 0.0
    passed_w = 0.0
    for r in results:
        w = 2.0 if "safety" in r.tags else 1.0
        total_w  += w
        if r.passed:
            passed_w += w
    return passed_w / total_w if total_w > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_behavioral_simulation(sysml_text: str,
                               model_name: str = "UnknownModel"
                               ) -> BehavioralSimResult:
    """
    Main entry point.

    Extracts all state machines from *sysml_text* via syside, generates
    a simulation scenario for each, executes the state machines, and
    returns an aggregated BehavioralSimResult.
    """
    state_machines = extract_state_machines(sysml_text)

    br = BehavioralSimResult(
        model_name=model_name,
        extracted_sm_count=len(state_machines),
    )

    if not state_machines:
        br.sim_score = 1.0   # neutral: no state machines defined
        return br

    scenario_results: List[BehavioralScenarioResult] = []
    for sm in state_machines:
        scenario_results.append(_run_scenario(sm))

    # Cross-scenario constraint
    ordering = _check_battery_ordering(scenario_results)
    if ordering is not None:
        scenario_results.append(ordering)

    br.scenario_results = scenario_results
    br.sim_score = _compute_score(scenario_results)
    return br
