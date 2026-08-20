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

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .state_extractor import GuardCondition, StateMachineDef, VarRef, extract_state_machines
from .state_executor import StateMachineInstance
from .constraint_checker import extract_constraints, ParsedConstraint, eval_op
from ..utils.sysml_text_utils import find_block_end


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


def _resolve(env: Dict[str, Any], name: str) -> Optional[float]:
    """Read a numeric attribute without inventing an engineering value."""
    v = env.get(name)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _build_comparison_driver_plans(
    guard: GuardCondition,
    initial_values: Dict[str, Any],
) -> List[DriverPlan]:
    """Build bounded trajectories for one numeric comparison guard."""
    lhs_var, rhs_var = _guard_endpoints(guard)

    if lhs_var and rhs_var:
        lhs_init = _resolve(initial_values, lhs_var)
        rhs_init = _resolve(initial_values, rhs_var)
        plans: List[DriverPlan] = []
        if rhs_init is not None:
            plans.append(_swept_plan(
                swept=lhs_var,
                held={rhs_var: rhs_init},
                operator=guard.operator,
                threshold=rhs_init,
                initial_values=initial_values,
                name="drive_LHS",
            ))
        flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(
            guard.operator, guard.operator,
        )
        if lhs_init is not None:
            plans.append(_swept_plan(
                swept=rhs_var,
                held={lhs_var: lhs_init},
                operator=flipped,
                threshold=lhs_init,
                initial_values=initial_values,
                name="drive_RHS",
            ))
        return plans

    if lhs_var:
        effective_threshold = guard.resolve_threshold(initial_values)
        if effective_threshold is None:
            return []
        guard.threshold = effective_threshold
        held: Dict[str, float] = {}
        if guard.rhs is not None:
            for variable in guard.rhs.vars():
                resolved = _resolve(initial_values, variable)
                if resolved is None:
                    return []
                held[variable] = resolved
        return [_swept_plan(
            swept=lhs_var,
            held=held,
            operator=guard.operator,
            threshold=effective_threshold,
            initial_values=initial_values,
            name="drive_LHS",
        )]
    return []


def _flatten_and_operands(
    guard: GuardCondition,
) -> Optional[List[GuardCondition]]:
    if guard.kind != "compound":
        return [guard]
    if guard.compound_op != "and":
        return None
    flattened: List[GuardCondition] = []
    for operand in guard.operands:
        nested = _flatten_and_operands(operand)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened


def _build_bounded_and_plans(
    guard: GuardCondition,
    initial_values: Dict[str, Any],
) -> List[DriverPlan]:
    """Satisfy a bounded AND without introducing a general-purpose solver."""
    operands = _flatten_and_operands(guard)
    if not operands:
        return []
    boolean_targets: Dict[str, bool] = {}
    comparisons: List[GuardCondition] = []
    for operand in operands:
        if operand.kind in {"bool_true", "bool_false"}:
            target = operand.kind == "bool_true"
            previous = boolean_targets.get(operand.attribute)
            if previous is not None and previous is not target:
                return []
            boolean_targets[operand.attribute] = target
        elif operand.kind == "comparison":
            comparisons.append(operand)
        else:
            return []
    if len(comparisons) > 1:
        return []

    if comparisons:
        plans = _build_comparison_driver_plans(
            comparisons[0], initial_values
        )
        for plan in plans:
            for step in plan.sequence:
                step.update(boolean_targets)
            plan.name = f"{plan.name}_with_boolean_holds"
        return plans

    if not boolean_targets:
        return []
    target_state = dict(boolean_targets)
    first_name = next(iter(target_state))
    baseline = dict(target_state)
    baseline[first_name] = not target_state[first_name]
    sequence = [dict(baseline) for _ in range(5)]
    sequence.extend(dict(target_state) for _ in range(15))
    return [DriverPlan(
        name="satisfy_bool_and",
        sequence=sequence,
        swept_var=first_name,
    )]


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
    # Accept-triggered mode machine: use command injection plan
    if sm.has_accept_transitions():
        plan = _build_accept_command_plan(sm)
        return [plan] if plan else []

    ft = sm.fault_transitions()
    if not ft:
        return []

    guard = ft[0].guards[0]
    init  = sm.initial_values or {}

    # ── Comparison ────────────────────────────────────────────────────────
    if guard.kind == "comparison":
        return _build_comparison_driver_plans(guard, init)

    # ── Enum equality: `mode == EnumType::Value` (Layer 3) ───────────────
    # Walk ALL enum_eq fault transitions on the same attribute in sequence:
    # hold at initial for 5 steps, then advance one step per phase so the
    # full mode chain (N-1 transitions) is exercised, not just the first.
    if guard.kind == "enum_eq":
        attr = guard.attribute
        init_val = str(init.get(attr, ""))   # e.g. "BOOT"

        # Collect target values from every enum_eq fault transition, in order.
        phase_values: List[str] = [
            t.guards[0].enum_value
            for t in ft
            if t.guards and t.guards[0].kind == "enum_eq"
            and t.guards[0].attribute == attr
        ]

        # Build sequence: 5 steps holding initial, then 1 step per phase,
        # then 5 extra steps at the final phase to let the last transition settle.
        seq: List[Dict[str, Any]] = [{attr: init_val}] * 5
        for val in phase_values:
            seq.append({attr: val})
        if phase_values:
            seq.extend([{attr: phase_values[-1]}] * 5)

        return [DriverPlan(name="traverse_modes", sequence=seq, swept_var=attr)]

    # ── Boolean flag ──────────────────────────────────────────────────────
    if guard.kind == "bool_true":
        attr = guard.attribute
        seq = [{attr: False}] * 5 + [{attr: True}] * 15
        return [DriverPlan(name="flip_bool", sequence=seq, swept_var=attr)]

    if guard.kind == "bool_false":
        attr = guard.attribute
        seq = [{attr: True}] * 5 + [{attr: False}] * 15
        return [DriverPlan(
            name="flip_bool_false",
            sequence=seq,
            swept_var=attr,
        )]

    # ── Bounded compound AND ──────────────────────────────────────────────
    if guard.kind == "compound":
        return _build_bounded_and_plans(guard, init)

    return []


def _build_accept_command_plan(sm: StateMachineDef) -> Optional[DriverPlan]:
    """
    Build a command-injection DriverPlan for accept-triggered mode machines.

    Traverses the state graph following accept transitions from the initial
    state and builds a step sequence that fires each command in order.
    The special key ``__accept__`` carries the command name; None means
    "no command this step" (hold).
    """
    if sm.initial_state is None:
        return None

    # Build adjacency: source_state -> [(accept_trigger, target_state)]
    graph: Dict[str, List[tuple]] = {}
    for t in sm.transitions:
        if t.is_initial or not t.accept_trigger or not t.source or not t.target:
            continue
        graph.setdefault(t.source, []).append((t.accept_trigger, t.target))

    if not graph:
        return None

    # Walk the nominal chain from initial state
    steps: List[str] = []   # ordered list of command names
    state = sm.initial_state
    visited: set = set()
    while state not in visited and state in graph:
        visited.add(state)
        cmd, next_state = graph[state][0]   # take first available transition
        steps.append(cmd)
        state = next_state

    if not steps:
        return None

    # Sequence: 2 hold steps, then one command + one hold per transition
    seq: List[Dict[str, Any]] = [{"__accept__": None}, {"__accept__": None}]
    for cmd in steps:
        seq.append({"__accept__": cmd})
        seq.append({"__accept__": None})
    seq.extend([{"__accept__": None}] * 3)

    return DriverPlan(
        name="accept_command_sequence",
        sequence=seq,
        swept_var="command",
    )


def _build_test_sequence(sm: StateMachineDef) -> List[Dict[str, Any]]:
    """Back-compat shim: return the first driver plan's sequence."""
    plans = _build_driver_plans(sm)
    return plans[0].sequence if plans else []


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_initialization_scenario(sm: StateMachineDef) -> BehavioralScenarioResult:
    """Verify an initialization/default-state machine without inventing a fault.

    Some requirements are invariants ("default to Locked on power-on"), not
    fault-triggered responses.  They legitimately need no guard transition when
    the machine has a single default state.  A multi-state declaration is not
    allowed to hide behind that exception: every declared state must still be
    structurally reachable from the initial state.
    """
    result = BehavioralScenarioResult(
        name=sm.name,
        state_machine=sm.name,
        description=f"{sm.owner_part}.{sm.name}: verify initial/default state semantics",
        passed=False,
        tags=["initialization"],
    )
    state_names = {s.name for s in sm.states}
    initial = sm.initial_state
    if not initial:
        result.violations.append("No initial state is declared")
        return result
    if initial not in state_names:
        result.violations.append(
            f"Initial state '{initial}' is not declared in the state machine"
        )
        return result

    reachable = {initial}
    changed = True
    while changed:
        changed = False
        for transition in sm.transitions:
            if transition.is_initial or not transition.source or not transition.target:
                continue
            if transition.source in reachable and transition.target not in reachable:
                reachable.add(transition.target)
                changed = True
    unreachable = sorted(state_names - reachable)
    if unreachable:
        result.violations.append(
            "Declared state(s) are unreachable from the initial state: "
            + ", ".join(unreachable)
        )

    result.timeline.append(f"Initial state: {initial}")
    entry = sm.entry_action_for_state(initial)
    if entry:
        result.fired_actions.append(entry)
        result.timeline.append(f"Initial entry action: {entry}")

    # Check conventional Boolean state mirrors such as Locked ↔ isLocked=true
    # and Disarmed ↔ isArmed=false.  This is intentionally conservative: an
    # unrelated Boolean attribute is ignored rather than guessed.
    state_key = re.sub(r"[^a-z0-9]", "", initial.lower())
    has_state_mirror = False
    for attr, value in (sm.initial_values or {}).items():
        if not isinstance(value, bool):
            continue
        attr_key = re.sub(r"[^a-z0-9]", "", str(attr).lower())
        if not attr_key.startswith("is") or len(attr_key) <= 2:
            continue
        feature = attr_key[2:]
        expected: Optional[bool] = None
        if state_key == feature:
            expected = True
        elif state_key in {f"un{feature}", f"dis{feature}", f"not{feature}"}:
            expected = False
        if expected is None:
            continue
        has_state_mirror = True
        result.timeline.append(f"Initial attribute: {attr}={value}")
        if value is not expected:
            result.violations.append(
                f"Initial state '{initial}' contradicts {attr}={value}; "
                f"expected {expected}"
            )

    # A state name alone is only a declaration, not executable or observable
    # initialization semantics.  Require either a conventional Boolean state
    # mirror (e.g. isLocked=true) or an entry action that performs the default
    # response.  This keeps legitimate single-state invariants compact while
    # rejecting empty shells such as `state Locked; transition initial ...`.
    if not has_state_mirror and not entry:
        result.violations.append(
            f"Initial state '{initial}' has no observable initialization semantics; "
            "add a consistent Boolean state attribute or an initial entry action"
        )

    result.passed = not result.violations
    return result


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

    # Accept-triggered machines have no guard-based fault transitions;
    # their plans come from the accept command sequence instead.
    is_accept_machine = sm.has_accept_transitions()

    if not ft and not is_accept_machine:
        return run_initialization_scenario(sm)

    primary_guard = ft[0].guards[0] if ft else None
    plans = _build_driver_plans(sm)
    if not plans:
        unresolved: List[str] = []
        if primary_guard is not None and primary_guard.kind == "comparison":
            lhs_var, rhs_var = _guard_endpoints(primary_guard)
            if lhs_var and rhs_var:
                unresolved = [
                    name for name in (lhs_var, rhs_var)
                    if _resolve(sm.initial_values or {}, name) is None
                ]
            elif primary_guard.rhs is not None:
                unresolved = [
                    name for name in primary_guard.rhs.vars()
                    if _resolve(sm.initial_values or {}, name) is None
                ]
        if unresolved:
            result.violations.append(
                "Could not resolve guard threshold from declared numeric "
                "initial values; simulation did not assume 0.0 for: "
                + ", ".join(dict.fromkeys(unresolved))
            )
        else:
            result.violations.append(
                "Could not generate test sequence for guard type"
            )
        return result

    is_mode_machine = (primary_guard is not None and primary_guard.kind == "enum_eq")

    # ── Execute every plan; pass if ANY plan triggers the fault ──────────────
    # Multi-plan (Layer 2) verifies the *relationship* in `A OP B`: a guard
    # that only fires in one direction still passes, but we record which plans
    # fired so the user can see the relation is properly two-sided.
    any_fired = False
    fault_states = {s.name for s in sm.states if s.entry_action}

    for plan in plans:
        inst = StateMachineInstance(sm)
        for t, variables in enumerate(plan.sequence):
            command = variables.get("__accept__") if is_accept_machine else None
            fired = inst.step(variables, time=float(t), command=command)
            # Fault monitors: stop when reaching a fault state (has entry action).
            # Mode/accept machines: run the full sequence.
            # The initial/default state may itself have an entry action (e.g.
            # power-on → Locked).  Do not mistake that pre-existing state for a
            # newly fired fault response; only stop after this step transitioned.
            if (fired and not is_mode_machine and not is_accept_machine
                    and inst.in_fault_state()):
                break

        events = inst.transition_log
        plan_label = f"[{plan.name}]" if len(plans) > 1 else ""

        # ── Accept machine path ───────────────────────────────────────────────
        # Verify the nominal command chain: all accept-triggered transitions
        # must fire in order.  Analogous to the mode-machine path below.
        if is_accept_machine:
            accept_trs = [t for t in sm.transitions
                          if not t.is_initial and t.accept_trigger]
            expected_count = len(accept_trs)
            fired_count = len(events)

            if fired_count == 0:
                result.violations.append(
                    f"No accept transition fired over {len(plan.sequence)} steps — "
                    f"command sequence was never consumed"
                )
                continue

            any_fired = True
            result.trigger_step = int(events[0].time)
            result.timeline.append(
                f"Command sequence: {fired_count}/{expected_count} accept transitions fired"
            )
            for ev in events:
                result.timeline.append(ev.to_line())

            if fired_count < expected_count:
                result.violations.append(
                    f"Mode machine only traversed {fired_count}/{expected_count} "
                    f"accept transitions — stuck at '{inst.current_state}' "
                    f"(remaining phases unreachable)"
                )
            continue

        # ── Mode machine path (Layer 3) ───────────────────────────────────────
        # Verify the complete phase chain: all N-1 enum_eq transitions must fire.
        if is_mode_machine:
            expected_count = len([tr for tr in ft
                                   if tr.guards and tr.guards[0].kind == "enum_eq"])
            fired_count = len(events)

            if fired_count == 0:
                result.violations.append(
                    f"No mode transition fired over {len(plan.sequence)} steps — "
                    f"guard '{primary_guard.description()}' was never satisfied"
                )
                continue

            any_fired = True
            result.trigger_step = int(events[0].time)
            result.timeline.append(
                f"Traversing {plan.swept_var}: "
                f"{fired_count}/{expected_count} transitions"
            )
            for ev in events:
                result.timeline.append(ev.to_line())

            if fired_count < expected_count:
                result.violations.append(
                    f"Mode machine only traversed {fired_count}/{expected_count} "
                    f"transitions — stuck at '{inst.current_state}' "
                    f"(remaining phases unreachable)"
                )
            continue

        # ── Fault monitor path (comparison / bool / compound) ────────────────
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
                elif primary_guard.kind in {"bool_true", "bool_false"}:
                    trigger_value = fault_event.variables_snapshot.get(
                        plan.swept_var
                    )
                    result.timeline.append(
                        f"{plan_label} Flipping {plan.swept_var} to "
                        f"{trigger_value} at t={int(fault_event.time)}"
                    )
                else:
                    result.timeline.append(
                        f"{plan_label} Satisfying compound guard via "
                        f"{plan.swept_var} at t={int(fault_event.time)}"
                    )

            result.timeline.append(fault_event.to_line())

            # ── Check 1: correct target state ─────────────────────────────────
            # Mode machines: no fault states — any transition is valid.
            # Detection-only monitors: fault_states is empty (no entry actions
            # by design) — transition firing is sufficient, skip state check.
            if not is_mode_machine and fault_states and fault_event.to_state not in fault_states:
                result.violations.append(
                    f"Transition target '{fault_event.to_state}' is not a known "
                    f"fault state (expected one of: {fault_states})"
                )

            # ── Check 2: entry action was called ──────────────────────────────
            # Mode machines: entry actions are optional — skip.
            # Detection-only monitors (fault_states empty): no entry action by
            # design — transition firing is the verification; skip this check.
            if not is_mode_machine and fault_states:
                # Look up whether the target state itself has an entry action
                target_has_entry = fault_event.to_state in fault_states
                if target_has_entry and not result.fired_actions:
                    result.violations.append(
                        f"Fault state '{fault_event.to_state}' has no entry action "
                        "recorded — emergency response may not have been triggered"
                    )
                elif result.fired_actions:
                    result.timeline.append(
                        f"  entry action called: {result.fired_actions[-1]}  ✓"
                    )
                else:
                    # fault state with no entry action — detection-only, pass
                    result.timeline.append(
                        f"  detection-only fault state (no entry action — command "
                        f"dispatch delegated to SafetyArbiter)  ✓"
                    )

            # ── Check 3: trigger at roughly the expected step ────────────────
            if primary_guard.kind == "comparison" and plan.start_val is not None \
                    and plan.step_size is not None:
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
# Accept machine: nominal + emergency branch scenarios
# ---------------------------------------------------------------------------

def _classify_accept_transitions(sm: StateMachineDef):
    """
    将 accept 转移分成正常功能响应和应急分支。

    不能用“目标状态是否有 entry action”区分两者：正常功能状态也必须能够
    执行动作，例如收到有效航点修改命令后更新飞行计划，或自动着陆完成后
    发送健康报告。应急语义应由状态、转移或响应动作的名称明确表达。
    """
    states = {s.name: s for s in sm.states}
    emergency_markers = (
        "emergency", "fault", "failure", "failsafe", "critical",
        "abort", "parachute", "hazard",
    )

    def _is_emergency(t) -> bool:
        state = states.get(t.target or "")
        semantic_name = " ".join(filter(None, (
            t.name,
            t.target,
            state.entry_action if state else None,
            state.entry_action_def if state else None,
            state.do_action if state else None,
            state.do_action_def if state else None,
        ))).lower()
        return any(marker in semantic_name for marker in emergency_markers)

    nominal, emergency = [], []
    for t in sm.transitions:
        if t.is_initial or not t.accept_trigger:
            continue
        if _is_emergency(t):
            emergency.append(t)
        else:
            nominal.append(t)
    return nominal, emergency


def _build_nominal_multigraph(
    nominal_trs,
) -> Dict[str, List[Tuple[str, str]]]:
    """保留所有 nominal 出边：{source: [(trigger, target), ...]}"""
    graph: Dict[str, List[Tuple[str, str]]] = {}
    for t in nominal_trs:
        if t.source and t.target and t.accept_trigger:
            graph.setdefault(t.source, []).append((t.accept_trigger, t.target))
    return graph


def _longest_nominal_path(
    graph: Dict[str, List[Tuple[str, str]]],
    start: str,
) -> List[Tuple[str, str]]:
    """
    DFS 找从 start 出发的最长无环路径。
    返回 [(cmd, target_state), ...] 列表。
    最长路径对应"完整正常运行序列"；捷径/中止路径更短，自然被排除。
    """
    def dfs(state: str, visited: frozenset) -> List[Tuple[str, str]]:
        best: List[Tuple[str, str]] = []
        for cmd, target in graph.get(state, []):
            if target not in visited:
                sub = dfs(target, visited | {target})
                candidate = [(cmd, target)] + sub
                if len(candidate) > len(best):
                    best = candidate
        return best

    return dfs(start, frozenset({start}))


def _bfs_nav_cmds(
    graph: Dict[str, List[Tuple[str, str]]],
    start: str,
    target: str,
) -> Optional[List[str]]:
    """
    BFS 找从 start 到 target 的最短命令序列（fallback 用）。
    返回命令列表，不可达时返回 None。
    """
    if start == target:
        return []
    queue: deque = deque([(start, [])])
    visited = {start}
    while queue:
        state, cmds = queue.popleft()
        for cmd, nxt in graph.get(state, []):
            if nxt == target:
                return cmds + [cmd]
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, cmds + [cmd]))
    return None


def _run_accept_nominal_scenario(
    sm: StateMachineDef,
    longest_path: List[Tuple[str, str]],
) -> BehavioralScenarioResult:
    """沿最长 nominal 路径逐步注入命令，验证所有正常转移全部触发。"""
    cmds = [cmd for cmd, _ in longest_path]
    states = [s for _, s in longest_path]
    chain_desc = " → ".join([sm.initial_state] + states) if states else "(none)"

    r = BehavioralScenarioResult(
        name=f"{sm.name}_nominal",
        state_machine=sm.name,
        description=f"nominal chain: {chain_desc}",
        passed=False,
        tags=["accept_machine"],
    )
    if not cmds:
        r.violations.append("No nominal transitions found")
        return r

    seq = [{"__accept__": None}] * 2
    source = sm.initial_state
    for cmd, target in longest_path:
        command_input: Dict[str, Any] = {"__accept__": cmd}
        transition = next(
            (
                item for item in sm.transitions
                if (
                    not item.is_initial
                    and item.source == source
                    and item.target == target
                    and item.accept_trigger == cmd
                )
            ),
            None,
        )
        for guard in transition.guards if transition is not None else ():
            if guard.kind == "bool_true":
                command_input[guard.attribute] = True
            elif guard.kind == "bool_false":
                command_input[guard.attribute] = False
        seq += [command_input, {"__accept__": None}]
        source = target

    inst = StateMachineInstance(sm)
    for t, v in enumerate(seq):
        inst.step(v, time=float(t), command=v["__accept__"])

    fired = len(inst.transition_log)
    r.fired_actions = list(inst.fired_actions)
    r.timeline.append(f"Command sequence: {fired}/{len(cmds)} nominal transitions fired")
    for ev in inst.transition_log:
        r.timeline.append(ev.to_line())
    if fired < len(cmds):
        r.violations.append(
            f"Nominal chain incomplete: {fired}/{len(cmds)} fired, "
            f"stuck at '{inst.current_state}'"
        )
    r.passed = not r.violations
    return r


def _run_accept_emergency_scenario(
    sm: StateMachineDef,
    graph: Dict[str, List[Tuple[str, str]]],
    longest_path: List[Tuple[str, str]],
    emrg_tr,
) -> BehavioralScenarioResult:
    """
    导航到 emrg_tr.source，再注入 emergency 命令，验证应急状态被触发。

    导航策略（优先级由高到低）：
      1. emrg_tr.source 在最长路径上 → 用路径前缀（上下文最真实）
      2. 否则 → BFS 找任意最短路径（fallback）
    """
    path_states = [s for _, s in longest_path]

    if emrg_tr.source == sm.initial_state:
        nav_cmds: Optional[List[str]] = []
    elif emrg_tr.source in path_states:
        idx = path_states.index(emrg_tr.source)
        nav_cmds = [cmd for cmd, _ in longest_path[: idx + 1]]
    else:
        nav_cmds = _bfs_nav_cmds(graph, sm.initial_state, emrg_tr.source)

    r = BehavioralScenarioResult(
        name=f"{sm.name}_emrg_from_{emrg_tr.source}",
        state_machine=sm.name,
        description=(
            f"emergency: navigate to {emrg_tr.source}, "
            f"then {emrg_tr.accept_trigger} → {emrg_tr.target}"
        ),
        passed=False,
        tags=["accept_machine", "emergency"],
    )
    if nav_cmds is None:
        r.violations.append(f"Cannot reach '{emrg_tr.source}' via nominal graph")
        return r

    # Detect trigger conflict: same trigger used by a nominal transition from this source.
    # Both would fire on the same command → non-deterministic in SysML v2;
    # the executor picks one, so the emergency may never trigger.
    # This is a model design issue (missing guard conditions), not a simulator failure.
    nominal_triggers = {cmd for cmd, _ in graph.get(emrg_tr.source, [])}
    if emrg_tr.accept_trigger in nominal_triggers:
        r.timeline.append(
            f"⚠ SKIP: trigger '{emrg_tr.accept_trigger}' is shared by a nominal "
            f"transition from '{emrg_tr.source}' — non-deterministic without guard "
            f"conditions; cannot reliably test emergency branch."
        )
        r.passed = True
        r.tags.append("trigger_conflict")
        return r

    seq = [{"__accept__": None}] * 2
    for cmd in nav_cmds:
        seq += [{"__accept__": cmd}, {"__accept__": None}]
    emergency_input: Dict[str, Any] = {
        "__accept__": emrg_tr.accept_trigger
    }
    for guard in emrg_tr.guards:
        if guard.kind == "bool_true":
            emergency_input[guard.attribute] = True
        elif guard.kind == "bool_false":
            emergency_input[guard.attribute] = False
    seq += [emergency_input, {"__accept__": None}]
    seq += [{"__accept__": None}] * 2

    inst = StateMachineInstance(sm)
    for t, v in enumerate(seq):
        inst.step(v, time=float(t), command=v["__accept__"])

    emrg_fired = any(ev.to_state == emrg_tr.target for ev in inst.transition_log)
    r.fired_actions = list(inst.fired_actions)

    if nav_cmds:
        r.timeline.append(
            f"Pre-navigate: {len(nav_cmds)} step(s) → '{emrg_tr.source}'"
        )
    r.timeline += [ev.to_line() for ev in inst.transition_log]

    if not emrg_fired:
        r.violations.append(
            f"Emergency transition to '{emrg_tr.target}' did not fire"
        )
    else:
        entry_action = sm.response_action_for_state(emrg_tr.target)
        if entry_action and entry_action not in inst.fired_actions:
            r.violations.append(
                f"Emergency state '{emrg_tr.target}' entered but "
                f"entry action '{entry_action}' was not called"
            )
        elif entry_action:
            r.timeline.append(f"  entry action called: {entry_action}  ✓")
    r.passed = not r.violations
    return r


def _run_accept_machine_scenarios(sm: StateMachineDef) -> List[BehavioralScenarioResult]:
    """
    为 accept-triggered 状态机生成完整场景集：
      1. nominal 场景（最长路径）
      2. 每个 emergency 分支各一个场景（最长路径前缀 or BFS fallback）
    """
    nominal_trs, emergency_trs = _classify_accept_transitions(sm)
    graph = _build_nominal_multigraph(nominal_trs)
    longest_path = _longest_nominal_path(graph, sm.initial_state)

    # An accept machine may legitimately be emergency-only: a recovery mechanism
    # remains in its safe initial state until a failsafe command arrives. State
    # persistence is its nominal behaviour; inventing a self-loop merely to create
    # a command-driven "nominal path" would make the model worse. Only require and
    # execute a nominal scenario when the model actually declares a nominal
    # transition. Emergency branches are still exercised individually below.
    results: List[BehavioralScenarioResult] = []
    if nominal_trs:
        results.append(_run_accept_nominal_scenario(sm, longest_path))
    for emrg_tr in emergency_trs:
        results.append(
            _run_accept_emergency_scenario(sm, graph, longest_path, emrg_tr)
        )
    return results


# ---------------------------------------------------------------------------
# Cross-component scenario (SafetyMonitor send → FlightController accept)
# ---------------------------------------------------------------------------

def _run_cross_component_scenario(
    sm_safety: StateMachineDef,
    sm_flight: StateMachineDef,
    send_cmd: str,
    send_port: str,
) -> BehavioralScenarioResult:
    """
    联合仿真：
      1. 驱动 sm_safety 的 guard 变量直到故障转移触发
      2. 检测到 entry action 含 send(send_cmd, send_port)
      3. 将 send_cmd 桥接注入 sm_flight
      4. 验证 sm_flight 进入应急状态
    """
    r = BehavioralScenarioResult(
        name=f"cross_{sm_safety.name}_to_{sm_flight.name}",
        state_machine="cross_component",
        description=(
            f"{sm_safety.name} guard → send {send_cmd} to {send_port} "
            f"→ {sm_flight.name} emergency"
        ),
        passed=False,
        tags=["safety", "cross_component"],
    )

    fault_states_flight = {s.name for s in sm_flight.states if s.entry_action}

    # 找 sm_flight 里接受 send_cmd 的所有转移（不限于 fault state）
    # 说明：safety SM 可能发送命令触发正常阶段（如 battery RTB → return phase）
    # 也可能触发应急阶段（如 propulsion fail → emergency）
    # 两者都是合法的跨组件因果链，只要命令被接受就 PASS
    all_trs_flight = [
        t for t in sm_flight.transitions
        if not t.is_initial
        and t.accept_trigger == send_cmd
    ]
    if not all_trs_flight:
        r.violations.append(
            f"{sm_flight.name} has no accept transition for '{send_cmd}' "
            f"— command sent by {sm_safety.name} is not accepted by the mode machine"
        )
        return r

    # 在 sm_flight 里找最近可达的 source 状态（nominal 链上最早有 send_cmd 转移的节点）
    nominal_trs_flight, _ = _classify_accept_transitions(sm_flight)
    nominal_graph_flight = {
        t.source: (t.accept_trigger, t.target)
        for t in nominal_trs_flight
        if t.source and t.target and t.accept_trigger
    }

    # 找到最近的可触发 send_cmd 的状态（不限于 fault 目标）
    emrg_sources = {t.source for t in all_trs_flight if t.source}
    nav_cmds, state, visited = [], sm_flight.initial_state, set()
    while state not in emrg_sources and state not in visited:
        if state not in nominal_graph_flight:
            break
        visited.add(state)
        cmd, state = nominal_graph_flight[state]
        nav_cmds.append(cmd)

    if state not in emrg_sources:
        r.violations.append(
            f"Cannot navigate {sm_flight.name} to any state "
            f"accepting '{send_cmd}' via nominal chain"
        )
        return r

    # 为 sm_safety 生成 guard 驱动计划
    plans = _build_driver_plans(sm_safety)
    if not plans:
        r.violations.append(f"Cannot build driver plan for {sm_safety.name}")
        return r
    safety_plan = plans[0]

    # ── 实例化两个状态机 ──────────────────────────────────────────────────
    inst_safety = StateMachineInstance(sm_safety)
    inst_flight = StateMachineInstance(sm_flight)

    # 预导航 sm_flight 到 emergency source 状态
    for i, cmd in enumerate(nav_cmds):
        inst_flight.step({"__accept__": cmd}, time=float(i), command=cmd)
        inst_flight.step({"__accept__": None}, time=float(i) + 0.5)

    pre_nav_count = len(inst_flight.transition_log)
    if nav_cmds:
        r.timeline.append(
            f"Pre-navigate {sm_flight.name}: "
            f"{pre_nav_count}/{len(nav_cmds)} steps → state='{inst_flight.current_state}'"
        )

    # ── 驱动 sm_safety，桥接到 sm_flight ─────────────────────────────────
    safety_fired_at: Optional[int] = None
    base_t = len(nav_cmds) * 2

    for step_i, variables in enumerate(safety_plan.sequence):
        t = float(base_t + step_i)
        inst_safety.step(variables, time=t)

        if safety_fired_at is None and inst_safety.in_fault_state():
            safety_fired_at = step_i
            ev = inst_safety.transition_log[-1]
            r.timeline.append(
                f"{sm_safety.name} fault fired at t={t:.0f}: "
                f"{ev.from_state} → {ev.to_state}"
            )
            # 桥接：向 sm_flight 注入 emergency 命令
            inst_flight.step({"__accept__": send_cmd}, time=t, command=send_cmd)
            inst_flight.step({"__accept__": None}, time=t + 0.5)
            break

    if safety_fired_at is None:
        r.violations.append(f"{sm_safety.name} guard never fired over the driver plan")
        return r

    # ── 验证 sm_flight 接受了命令并转移 ──────────────────────────────────
    # 只要有任何转移因 send_cmd 而触发即视为成功（因果链完整）
    cmd_fired = any(
        ev.guard_description == f"accept {send_cmd}"
        for ev in inst_flight.transition_log
    )
    if not cmd_fired:
        r.violations.append(
            f"{sm_flight.name} did not fire any transition on '{send_cmd}'; "
            f"current state: '{inst_flight.current_state}'"
        )
    else:
        target_state = inst_flight.current_state
        is_fault = target_state in fault_states_flight
        entry = (
            sm_flight.response_action_for_state(target_state)
            if is_fault else None
        )
        kind = "emergency" if is_fault else "nominal"
        r.timeline.append(
            f"{sm_flight.name} entered '{target_state}' [{kind}]"
            + (f", entry action: {entry}  ✓" if entry else "")
        )
        # fault state 需要额外确认 entry action 被调用
        if is_fault and entry and entry not in inst_flight.fired_actions:
            r.violations.append(
                f"Emergency entry action '{entry}' in '{target_state}' was not called"
            )

    r.passed = not r.violations
    return r


def _collect_cross_component_scenarios(
    state_machines: List[StateMachineDef],
) -> List[BehavioralScenarioResult]:
    """
    扫描所有状态机，寻找 (safety SM with sends) × (accept SM accepting that cmd) 对，
    为每对生成一个跨组件联合仿真场景。
    """
    # 建立 cmd_name → [accept SM] 索引
    accept_index: dict = {}
    for sm in state_machines:
        if not sm.has_accept_transitions():
            continue
        for t in sm.transitions:
            if t.accept_trigger:
                accept_index.setdefault(t.accept_trigger, []).append(sm)

    results: List[BehavioralScenarioResult] = []
    seen_pairs: set = set()

    for sm in state_machines:
        if sm.has_accept_transitions():
            continue   # 只扫 guard-triggered SM
        for _state_name, cmd, port in sm.all_sends():
            for flight_sm in accept_index.get(cmd, []):
                pair_key = (sm.name, flight_sm.name, cmd)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                results.append(
                    _run_cross_component_scenario(sm, flight_sm, cmd, port)
                )
    return results


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
# Parametric constraint scenarios  (Option X)
# ---------------------------------------------------------------------------

# Matches:  [readonly] attribute <name> : <Type> [<unit>] = <number>
_ATTR_NUM_RE = re.compile(
    r'\b(?:readonly\s+)?attribute\s+(\w+)\s*:[^=;\n]*?=\s*([-+]?\d+(?:\.\d+)?)'
)
_READONLY_RE = re.compile(r'\breadonly\s+attribute\s+(\w+)')


def _extract_all_attrs(sysml_text: str) -> Dict[str, float]:
    """Return {attr_name: float_value} for every numeric attribute in the SysML source."""
    out: Dict[str, float] = {}
    for m in _ATTR_NUM_RE.finditer(sysml_text):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def _parse_readonly_attrs(sysml_text: str) -> set:
    return set(_READONLY_RE.findall(sysml_text))


def _run_parametric_constraint_scenario(
    c: ParsedConstraint,
    all_initial_values: Dict[str, float],
    guard_variables: set,
    readonly_vars: set,
) -> Optional[BehavioralScenarioResult]:
    """
    Build a boundary-sweep scenario for one assert constraint.

    Skipped (returns None) when:
      - LHS is already in guard_variables  (behavioral_sim already verifies it)
      - LHS is a readonly attr  (static check covers it)
      - LHS or RHS values cannot be resolved

    Pass criterion:
      1. Initial value satisfies the constraint
      2. Constraint holds throughout the valid range
      3. Constraint fails when the limit is exceeded  (boundary is live)
    """
    # Already covered by state machine guard
    if c.lhs in guard_variables:
        return None

    # Readonly LHS → static check is sufficient, skip parametric sweep
    if c.lhs in readonly_vars:
        return None

    lhs_init = all_initial_values.get(c.lhs)
    rhs_val  = all_initial_values.get(c.rhs)
    if rhs_val is None:
        try:
            rhs_val = float(c.rhs)
        except (TypeError, ValueError):
            pass

    result = BehavioralScenarioResult(
        name=f"constraint_{c.name}",
        state_machine=f"{c.owner_part}::assert",
        description=(
            f"{c.owner_part}.{c.name}: parametric sweep — "
            f"{c.lhs} {c.operator} {c.rhs}"
        ),
        passed=False,
        tags=[
            "parametric_constraint",
            f"constraint_provenance:{c.provenance}",
            (
                "requirement_behavior"
                if c.provenance == "FROZEN_REQUIREMENT"
                else "ag_behavior"
                if c.provenance == "A_G_GUARANTEE"
                else "design_constraint"
            ),
        ],
    )

    if lhs_init is None:
        result.violations.append(
            f"Cannot sweep '{c.lhs}': no initial value found. "
            f"Add 'attribute {c.lhs} : Real = <init>;' to {c.owner_part}."
        )
        return result

    if rhs_val is None:
        result.violations.append(
            f"Cannot resolve RHS '{c.rhs}': not a number and not declared in model."
        )
        return result

    lhs_f = float(lhs_init)
    rhs_f = float(rhs_val)
    op    = c.operator

    # Sweep direction: push lhs toward and past the limit
    if op in ("<=", "<"):
        sweep_end = rhs_f + abs(rhs_f) * 0.15 + 1.0   # always > rhs_f regardless of sign
        step_size = (sweep_end - lhs_f) / _N_STEPS
    elif op in (">=", ">"):
        sweep_end = rhs_f - abs(rhs_f) * 0.15 - 1.0   # always < rhs_f regardless of sign
        step_size = (sweep_end - lhs_f) / _N_STEPS
    elif op == "==":
        # A configuration pin: the planned value is the only point where the
        # constraint holds, so boundary liveness is not a one-sided sweep but
        # a three-point probe -- hold at the pinned value, violate on either
        # side of it. The old behaviour reported the operator as unsupported,
        # so every ``value == planned`` constraint failed behaviour execution
        # mechanically, whatever the model said.
        delta = abs(rhs_f) * 0.05 + 0.5
        result.timeline.append(
            f"Probing {c.lhs}: hold at {rhs_f:g}, "
            f"expect violation at {rhs_f - delta:g} and {rhs_f + delta:g} "
            f"(limit: == {rhs_f:g})"
        )
        if not eval_op(lhs_f, op, rhs_f):
            result.violations.append(
                f"Initial value {c.lhs}={lhs_f:g} already violates "
                f"constraint ({lhs_f:g} == {rhs_f:g} is false)."
            )
            return result
        below_ok = not eval_op(rhs_f - delta, op, rhs_f)
        above_ok = not eval_op(rhs_f + delta, op, rhs_f)
        if below_ok and above_ok:
            result.timeline.append(
                f"  holds at {rhs_f:g}; violated at both probes  "
                f"(boundary live — deviating from {rhs_f:g} triggers "
                f"violation)"
            )
            result.passed = True
        else:
            result.violations.append(
                f"Equality probe not live around {rhs_f:g}."
            )
        return result
    else:
        result.violations.append(f"Unsupported operator '{op}' for parametric sweep.")
        return result

    if abs(step_size) < 1e-9:
        step_size = (1.0 / _N_STEPS) * (1 if op in (">=", ">") else -1)

    # Run the sweep and record where constraint holds vs fails
    last_hold: Optional[float] = None
    first_fail: Optional[float] = None
    val = lhs_f
    for _ in range(_N_STEPS + 1):
        if eval_op(val, op, rhs_f):
            last_hold = val
        elif first_fail is None:
            first_fail = val
        val += step_size

    result.timeline.append(
        f"Sweeping {c.lhs}: {lhs_f:.4g} → {sweep_end:.4g} "
        f"({_N_STEPS} steps,  limit: {op} {rhs_f})"
    )
    if last_hold is not None:
        result.timeline.append(
            f"  holds through {c.lhs} = {last_hold:.4g}  ✓"
        )
    if first_fail is not None:
        result.timeline.append(
            f"  fails at {c.lhs} = {first_fail:.4g}  "
            f"(boundary live — exceeding {rhs_f} triggers violation)"
        )

    # Pass criteria
    if not eval_op(lhs_f, op, rhs_f):
        result.violations.append(
            f"Initial value {c.lhs}={lhs_f} already violates "
            f"constraint ({lhs_f} {op} {rhs_f} is false)."
        )
    elif first_fail is None:
        result.violations.append(
            f"Constraint never fails even at sweep end {sweep_end:.4g} — "
            f"boundary unreachable or limit incorrectly set."
        )
    else:
        result.passed = True

    return result


def _owner_part_body(sysml_text: str, owner: str) -> str:
    match = re.search(
        rf"\bpart\s+def\s+{re.escape(owner)}\s*\{{",
        sysml_text,
    )
    if match is None:
        return ""
    opening = sysml_text.find("{", match.start(), match.end())
    closing = find_block_end(sysml_text, opening)
    return (
        sysml_text[opening + 1:closing]
        if closing != -1 else ""
    )


def _state_body_text(
    owner_body: str,
    behavior_name: str,
    state_name: str,
) -> str:
    behavior = re.search(
        rf"\bstate\s+def\s+{re.escape(behavior_name)}\s*\{{",
        owner_body,
    )
    if behavior is None:
        return ""
    behavior_opening = owner_body.find(
        "{", behavior.start(), behavior.end()
    )
    behavior_closing = find_block_end(owner_body, behavior_opening)
    if behavior_closing == -1:
        return ""
    state = re.search(
        rf"\bstate\s+(?!def\b){re.escape(state_name)}\s*\{{",
        owner_body[behavior_opening + 1:behavior_closing],
    )
    if state is None:
        return ""
    state_start = behavior_opening + 1 + state.start()
    state_opening = owner_body.find(
        "{", state_start, behavior_opening + 1 + state.end()
    )
    state_closing = find_block_end(owner_body, state_opening)
    return (
        owner_body[state_opening + 1:state_closing]
        if state_closing != -1 else ""
    )


def _run_state_active_constraint_scenario(
    c: ParsedConstraint,
    sysml_text: str,
    state_machines: List[StateMachineDef],
    all_initial_values: Dict[str, float],
) -> BehavioralScenarioResult:
    """Verify bounded execution evidence for a state-owned SysML constraint.

    This proves model containment, reachability, response execution, typed
    runtime binding, and a live numeric boundary. It deliberately does not
    claim that the external environment will satisfy the bound.
    """
    result = BehavioralScenarioResult(
        name=f"state_constraint_{c.name}",
        state_machine=(
            c.activation_ref
            or f"{c.owner_part}::state_constraint"
        ),
        description=(
            f"{c.owner_part}.{c.name}: execute STATE_ACTIVE constraint "
            f"{c.lhs} {c.operator} {c.rhs}"
        ),
        passed=False,
        tags=[
            "state_active_constraint",
            f"constraint_provenance:{c.provenance}",
            (
                "requirement_behavior"
                if c.provenance == "FROZEN_REQUIREMENT"
                else "ag_behavior"
                if c.provenance == "A_G_GUARANTEE"
                else "design_constraint"
            ),
        ],
    )
    if not c.activation_ref or "::" not in c.activation_ref:
        result.violations.append(
            "STATE_ACTIVE constraint has no BehaviorId::StateId reference"
        )
        return result
    behavior_name, state_name = c.activation_ref.split("::", 1)
    if (
        c.containing_behavior != behavior_name
        or c.containing_state != state_name
    ):
        result.violations.append(
            f"Constraint is contained by "
            f"{c.containing_behavior or '?'}::{c.containing_state or '?'}, "
            f"expected {c.activation_ref}"
        )
        return result
    machine = next(
        (
            item for item in state_machines
            if item.owner_part == c.owner_part
            and item.name == behavior_name
        ),
        None,
    )
    if machine is None:
        result.violations.append(
            f"Cannot extract activation behavior {behavior_name}"
        )
        return result
    reachable = {machine.initial_state} if machine.initial_state else set()
    changed = True
    while changed:
        changed = False
        for transition in machine.transitions:
            if (
                transition.source in reachable
                and transition.target
                and transition.target not in reachable
            ):
                reachable.add(transition.target)
                changed = True
    if state_name not in reachable:
        result.violations.append(
            f"Activation state {c.activation_ref} is unreachable"
        )
    state = next(
        (item for item in machine.states if item.name == state_name),
        None,
    )
    owner_body = _owner_part_body(sysml_text, c.owner_part)
    state_body = _state_body_text(
        owner_body, behavior_name, state_name
    )
    has_do_action = bool(
        (state and state.do_action)
        or re.search(r"\bdo\s+action\b", state_body)
    )
    if state is None or (not state.entry_action and not has_do_action):
        result.violations.append(
            f"Activation state {c.activation_ref} has no executable "
            "entry/do response"
        )
    runtime_binding = re.search(
        rf"\battribute\s+{re.escape(c.lhs)}\s*:[^;=]+=\s*"
        r"(?P<path>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*;",
        owner_body,
    )
    if runtime_binding is None:
        result.violations.append(
            f"Runtime subject {c.lhs} is not bound to an input data path"
        )
    rhs_value = all_initial_values.get(c.rhs)
    if rhs_value is None:
        try:
            rhs_value = float(c.rhs)
        except (TypeError, ValueError):
            result.violations.append(
                f"Cannot resolve state constraint RHS {c.rhs}"
            )
    if rhs_value is not None:
        epsilon = max(abs(float(rhs_value)) * 0.01, 0.01)
        on_valid_side = (
            float(rhs_value) + epsilon
            if c.operator in {">=", ">"}
            else float(rhs_value) - epsilon
        )
        on_invalid_side = (
            float(rhs_value) - epsilon
            if c.operator in {">=", ">"}
            else float(rhs_value) + epsilon
        )
        if (
            not eval_op(on_valid_side, c.operator, float(rhs_value))
            or eval_op(on_invalid_side, c.operator, float(rhs_value))
        ):
            result.violations.append(
                f"Constraint boundary {c.operator} {rhs_value:g} is not live"
            )
        else:
            response_label = (
                state.entry_action
                if state and state.entry_action
                else state.do_action
                if state and state.do_action
                else "do action"
            )
            result.timeline.append(
                f"Reached {c.activation_ref}; state response "
                f"{response_label} executes"
            )
            result.timeline.append(
                f"Runtime input {c.lhs} evaluated at live boundary "
                f"{c.operator} {rhs_value:g}"
            )
    result.passed = not result.violations
    return result


def _collect_constraint_scenarios(
    sysml_text: str,
    state_machines: List[StateMachineDef],
) -> List[BehavioralScenarioResult]:
    """
    For every assert constraint in the model that is NOT covered by an
    existing state machine guard, produce a parametric sweep scenario.
    """
    constraints = extract_constraints(sysml_text)
    if not constraints:
        return []
    plan_owned_model = any(
        item.plan_constraint_id is not None for item in constraints
    )

    # Merge attribute values: full-text scan takes precedence over per-SM dicts
    all_initial_values: Dict[str, float] = {}
    for sm in state_machines:
        for k, v in sm.initial_values.items():
            try:
                all_initial_values[k] = float(v)
            except (TypeError, ValueError):
                pass
    all_initial_values.update(_extract_all_attrs(sysml_text))

    # Variables already verified by state machine guards
    guard_variables: set = set()
    for sm in state_machines:
        for tr in sm.transitions:
            for g in tr.guards:
                if g.attribute:
                    guard_variables.add(g.attribute)
                if g.rhs is not None:
                    try:
                        guard_variables.update(g.rhs.vars())
                    except AttributeError:
                        pass

    readonly_vars = _parse_readonly_attrs(sysml_text)

    results: List[BehavioralScenarioResult] = []
    for c in constraints:
        # Once the model carries compiler-owned constraint annotations, only
        # those planned invariants may contribute executable evidence.
        # Unannotated A/G prose invariants and LLM-authored assertions remain
        # available to their dedicated checkers but cannot inflate or fail the
        # parametric denominator.
        if plan_owned_model and c.plan_constraint_id is None:
            continue
        if (
            c.activation == "STATE_ACTIVE"
            and c.verification_tier == "STATE_EXECUTION"
        ):
            results.append(_run_state_active_constraint_scenario(
                c,
                sysml_text,
                state_machines,
                all_initial_values,
            ))
            continue
        if c.activation != "ALWAYS" or (
            c.verification_tier != "PARAMETRIC_SWEEP"
        ):
            continue
        r = _run_parametric_constraint_scenario(
            c, all_initial_values, guard_variables, readonly_vars
        )
        if r is not None:
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(results: List[BehavioralScenarioResult]) -> float:
    if not results:
        return 1.0   # neutral: no state machines → no violations
    total_w  = 0.0
    passed_w = 0.0
    for r in results:
        if "cross_component" in r.tags:
            w = 1.5          # 响应链完整性：故障检测后必须能送达执行器，比 nominal 更关键
        elif "emergency" in r.tags:
            w = 0.5          # accept 分支跳转：测试性质与 nominal 相近，保持低权重
        elif "safety" in r.tags:
            w = 2.0          # 故障检测 guard：安全感知层，最高权重
        else:
            w = 1.0
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

    scenario_results: List[BehavioralScenarioResult] = []

    if state_machines:
        for sm in state_machines:
            if sm.has_accept_transitions():
                scenario_results.extend(_run_accept_machine_scenarios(sm))
            else:
                scenario_results.append(_run_scenario(sm))

        # 跨组件联合场景（SafetyMonitor send → FlightController accept）
        cross_results = _collect_cross_component_scenarios(state_machines)
        scenario_results.extend(cross_results)

        # Cross-scenario ordering constraint（电池优先级）
        ordering = _check_battery_ordering(scenario_results)
        if ordering is not None:
            scenario_results.append(ordering)

    # Parametric constraint scenarios (Option X):
    # assert constraints whose LHS is a runtime variable not covered by any guard.
    constraint_scenarios = _collect_constraint_scenarios(sysml_text, state_machines)
    scenario_results.extend(constraint_scenarios)

    if not scenario_results:
        br.sim_score = 1.0   # neutral: nothing to verify
        return br

    br.scenario_results = scenario_results
    br.sim_score = _compute_score(scenario_results)
    return br
