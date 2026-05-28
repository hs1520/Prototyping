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

from .state_extractor import GuardCondition, StateMachineDef, extract_state_machines
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


def _build_test_sequence(sm: StateMachineDef) -> List[Dict[str, Any]]:
    """
    Auto-generate a sequence of variable dicts that will drive the
    first fault transition's guard condition from safe → fault.
    """
    ft = sm.fault_transitions()
    if not ft:
        return []

    guard = ft[0].guards[0]   # primary guard of the first fault transition

    if guard.kind == "comparison":
        start, step = _derive_start_and_step(guard, sm.initial_values)
        seq = []
        val = start
        for _ in range(_N_STEPS + 15):
            seq.append({guard.attribute: val})
            val += step
        return seq

    if guard.kind == "bool_true":
        # Flip to True at step 5
        attr = guard.attribute
        return [{attr: False}] * 5 + [{attr: True}] * 15

    if guard.kind == "compound" and guard.compound_op == "and":
        # Flip each boolean operand in sequence (5 steps apart)
        bool_operands = [op for op in guard.operands if op.kind == "bool_true"]
        if not bool_operands:
            return []
        seq: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {op.attribute: False for op in bool_operands}
        for i in range(len(bool_operands) * 8 + 10):
            idx = i // 8   # flip next operand every 8 steps
            for j, op in enumerate(bool_operands):
                state[op.attribute] = (j <= idx and idx < len(bool_operands))
            seq.append(dict(state))
        return seq

    return []


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

    seq = _build_test_sequence(sm)
    if not seq:
        result.violations.append("Could not generate test sequence for guard type")
        return result

    # Compute expected trigger step for the primary guard
    primary_guard = ft[0].guards[0]
    exp_step: Optional[int] = None
    start_val: Optional[float] = None
    step_size: Optional[float] = None

    if primary_guard.kind == "comparison":
        s, st = _derive_start_and_step(primary_guard, sm.initial_values)
        start_val = s
        step_size = st
        exp_step = _expected_trigger_step(primary_guard, s, st)
        result.timeline.append(
            f"Driving {primary_guard.attribute}: "
            f"{s:.2f} → (threshold {primary_guard.operator} {primary_guard.threshold})"
        )
    elif primary_guard.kind == "bool_true":
        exp_step = 5
        result.timeline.append(
            f"Flipping {primary_guard.attribute} to True at t=5"
        )
    elif primary_guard.kind == "compound":
        bool_ops = [op for op in primary_guard.operands if op.kind == "bool_true"]
        exp_step = len(bool_ops) * 8
        result.timeline.append(
            f"Flipping boolean flags: {primary_guard.description()}"
        )

    # ── Execute ───────────────────────────────────────────────────────────────
    inst = StateMachineInstance(sm)

    for t, variables in enumerate(seq):
        inst.step(variables, time=float(t))
        if inst.in_fault_state():
            break   # fault state reached — stop simulation

    # ── Evaluate ──────────────────────────────────────────────────────────────
    events = inst.transition_log
    result.fired_actions = list(inst.fired_actions)

    if not events:
        result.violations.append(
            f"No state transition fired in {len(seq)} simulation steps — "
            f"guard '{primary_guard.description()}' was never satisfied"
        )
        return result

    # The last event is the fault transition
    fault_event = events[-1]
    result.trigger_step = int(fault_event.time)
    result.timeline.append(fault_event.to_line())

    # Check 1: correct target state
    fault_states = {s.name for s in sm.states if s.entry_action}
    if fault_event.to_state not in fault_states:
        result.violations.append(
            f"Transition target '{fault_event.to_state}' is not a known fault state "
            f"(expected one of: {fault_states})"
        )

    # Check 2: entry action was called
    if not result.fired_actions:
        result.violations.append(
            f"Fault state '{fault_event.to_state}' has no entry action recorded — "
            "emergency response may not have been triggered"
        )
    else:
        result.timeline.append(f"  entry action called: {result.fired_actions[-1]}  ✓")

    # Check 3: trigger at roughly the expected step (tolerance ±30%)
    if exp_step and primary_guard.kind == "comparison":
        actual  = result.trigger_step
        tolerance = max(3, int(exp_step * 0.30))
        if abs(actual - exp_step) > tolerance:
            result.violations.append(
                f"Transition fired at step {actual} but expected near step {exp_step} "
                f"(tolerance ±{tolerance}). Guard threshold may be misconfigured."
            )

    # Record the variable value at trigger (for numeric guards)
    if primary_guard.kind == "comparison" and start_val is not None and step_size is not None:
        trig_val = start_val + step_size * result.trigger_step
        result.trigger_value = round(trig_val, 3)
        result.timeline.append(
            f"  {primary_guard.attribute} at trigger: {result.trigger_value}"
            f"  (threshold: {primary_guard.operator} {primary_guard.threshold})"
        )
        # Verify the value actually satisfies the guard
        op = primary_guard.operator
        th = primary_guard.threshold
        tv = result.trigger_value
        satisfied = (
            (op == "<"  and tv <  th) or
            (op == "<=" and tv <= th) or
            (op == ">"  and tv >  th) or
            (op == ">=" and tv >= th) or
            (op == "==" and tv == th)
        )
        if not satisfied:
            result.violations.append(
                f"Guard not satisfied at trigger: {tv} {op} {th} is False"
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
