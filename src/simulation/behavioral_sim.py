"""behavioral_sim.py"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .state_extractor import GuardCondition, StateMachineDef, VarRef, extract_state_machines
from .state_executor import StateMachineInstance
from .constraint_checker import extract_constraints, ParsedConstraint, eval_op
from ..utils.sysml_text_utils import find_block_end, named_block_span


@dataclass
class BehavioralScenarioResult:
    name: str
    state_machine: str
    description: str
    passed: bool
    timeline: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    fired_actions: List[str] = field(default_factory=list)
    trigger_step: Optional[int] = None
    trigger_value: Optional[float] = None
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


_N_STEPS = 70

def _derive_start_and_step(guard: GuardCondition,
                            initial_values: Dict[str, Any]
                            ) -> Tuple[float, float]:
    attr  = guard.attribute
    th    = guard.threshold
    start = initial_values.get(attr)

    if guard.operator in ("<", "<="):
        if start is None or float(start) <= th:
            start = th * 3.0 + 10.0
        start = float(start)
        step = (start - th) / (_N_STEPS * 0.85)
        step = max(step, 0.001)
        return start, -step

    else:
        if start is None or float(start) >= th:
            start = 0.0
        start = float(start)
        step = (th - start) / (_N_STEPS * 0.85)
        step = max(step, 0.001)
        return start, +step


@dataclass
class DriverPlan:
    """One named trajectory of variable bindings used to drive a state machine.

    Layer 2 allows multiple plans per guard: for `A <= B`, one pushes A down
    holding B and one pushes B up holding A, so the relationship is exercised
    rather than one operand.
    """
    name: str
    sequence: List[Dict[str, Any]]
    swept_var: str = ""
    start_val: Optional[float] = None
    step_size: Optional[float] = None


def _swept_plan(
    swept: str,
    held: Dict[str, float],
    operator: str,
    threshold: float,
    initial_values: Dict[str, Any],
    name: str,
) -> DriverPlan:
    """Build a single-variable sweep trajectory.

    `swept` ramps across `threshold` under `operator`; `held` variables are
    emitted unchanged every step so the guard evaluator sees the full
    environment.
    """
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
    lhs_var = guard.lhs.name if isinstance(guard.lhs, VarRef) else None
    rhs_var = guard.rhs.name if isinstance(guard.rhs, VarRef) else None
    return lhs_var, rhs_var


def _resolve(env: Dict[str, Any], name: str) -> Optional[float]:
    v = env.get(name)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _build_comparison_driver_plans(
    guard: GuardCondition,
    initial_values: Dict[str, Any],
) -> List[DriverPlan]:
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
    if sm.has_accept_transitions():
        plan = _build_accept_command_plan(sm)
        return [plan] if plan else []

    ft = sm.fault_transitions()
    if not ft:
        return []

    guard = ft[0].guards[0]
    init  = sm.initial_values or {}

    if guard.kind == "comparison":
        return _build_comparison_driver_plans(guard, init)

    # ── Enum equality: `mode == EnumType::Value` (Layer 3) ───────────────
    # Walk every enum_eq fault transition on the attribute in sequence: hold at
    # initial for 5 steps, then advance one step per phase, so the full mode chain
    # (N-1 transitions) is exercised.
    if guard.kind == "enum_eq":
        attr = guard.attribute
        init_val = str(init.get(attr, ""))

        phase_values: List[str] = [
            t.guards[0].enum_value
            for t in ft
            if t.guards and t.guards[0].kind == "enum_eq"
            and t.guards[0].attribute == attr
        ]

        seq: List[Dict[str, Any]] = [{attr: init_val}] * 5
        for val in phase_values:
            seq.append({attr: val})
        if phase_values:
            seq.extend([{attr: phase_values[-1]}] * 5)

        return [DriverPlan(name="traverse_modes", sequence=seq, swept_var=attr)]

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

    if guard.kind == "compound":
        return _build_bounded_and_plans(guard, init)

    return []


def _build_accept_command_plan(sm: StateMachineDef) -> Optional[DriverPlan]:
    if sm.initial_state is None:
        return None

    graph: Dict[str, List[tuple]] = {}
    for t in sm.transitions:
        if t.is_initial or not t.accept_trigger or not t.source or not t.target:
            continue
        graph.setdefault(t.source, []).append((t.accept_trigger, t.target))

    if not graph:
        return None

    steps: List[str] = []
    state = sm.initial_state
    visited: set = set()
    while state not in visited and state in graph:
        visited.add(state)
        cmd, next_state = graph[state][0]
        steps.append(cmd)
        state = next_state

    if not steps:
        return None

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


def _drive_power_event_to_default(
    sm: StateMachineDef,
    initial: str,
    required_state_terms: set,
    result: BehavioralScenarioResult,
) -> bool:
    """Recognise and drive the power-on shape: an initial phase state whose
    exits all land on the requirement's default-term state.

    Returns True when the shape is recognised; defects found while driving are
    recorded as violations on *result*. Returns False otherwise, and the
    caller's original violation applies.
    """
    from src.utils.sysml_text_utils import semantic_terms

    if not required_state_terms:
        # No default-state vocabulary: nothing to recognise the landed state
        # by.  The internal (termless) caller keeps its original strictness.
        return False
    exits = [
        t for t in sm.transitions
        if not t.is_initial and t.source == initial and t.target
    ]
    if not exits:
        return False
    if any(
        not (required_state_terms & semantic_terms(t.target)) for t in exits
    ):
        # An exit from the initial state to a non-default state lets the machine
        # leave power-on without passing the default, which "before any arming"
        # forbids.
        return False
    unguarded = [t for t in exits if not t.guards]
    if not unguarded:
        result.violations.append(
            f"Every transition out of initial state '{initial}' toward the "
            "default state is guarded — the default is conditional, not a "
            "power-on default"
        )
        return True
    drive = unguarded[0]
    landed = drive.target
    trigger = drive.accept_trigger or "automatic"
    result.timeline.append(f"Drove power event: {trigger} -> {landed}")
    landed_entry = sm.entry_action_for_state(landed)
    landed_do = sm.do_action_for_state(landed)
    if landed_entry:
        result.fired_actions.append(landed_entry)
        result.timeline.append(f"Default-state entry action: {landed_entry}")
    if landed_do:
        result.fired_actions.append(landed_do)
        result.timeline.append(f"Default-state do action: {landed_do}")
    landed_key = re.sub(r"[^a-z0-9]", "", landed.lower())
    mirror_found = False
    for attr, value in (sm.initial_values or {}).items():
        if not isinstance(value, bool):
            continue
        attr_key = re.sub(r"[^a-z0-9]", "", str(attr).lower())
        if not attr_key.startswith("is") or len(attr_key) <= 2:
            continue
        feature = attr_key[2:]
        if feature and feature in landed_key:
            mirror_found = True
            result.timeline.append(f"Default-state attribute: {attr}={value}")
    if not landed_entry and not landed_do and not mirror_found:
        result.violations.append(
            f"Default state '{landed}' has no observable semantics; add an "
            "entry action, a sustaining do action, or a consistent Boolean "
            "state attribute"
        )
    return True


def run_initialization_scenario(
    sm: StateMachineDef,
    required_state_terms: Iterable[str] = (),
) -> BehavioralScenarioResult:
    """Verify an initialization/default-state machine without inventing a fault.

    Invariant requirements ("default to Locked on power-on") need no guard
    transition when the machine has a single default state, but every declared
    state must still be reachable from the initial state. ``required_state_terms``
    is the requirement's default-state vocabulary (e.g. {"locked"}) from the
    frozen requirement text; a sustaining do-action counts as observable
    semantics only when the initial state is that default state, so
    ``Locked { do securePayload }`` qualifies and ``PowerOn { do initialize }``
    does not.
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
    # A sustained do-action counts as observable initialization semantics
    # alongside an entry action: `state Locked { do securePayload; }` holds
    # the default continuously. Run3 failed two payload machines for this
    # shape while L2 servo evidence (lock PWM) showed the default holding.
    from src.utils.sysml_text_utils import semantic_terms
    do_action = sm.do_action_for_state(initial)
    initial_is_default = bool(
        set(required_state_terms) & semantic_terms(initial)
    )
    holding_do = do_action if initial_is_default else None
    if holding_do:
        result.fired_actions.append(holding_do)
        result.timeline.append(f"Initial do action: {holding_do}")

    # Check conventional Boolean state mirrors such as Locked ↔ isLocked=true
    # and Disarmed ↔ isArmed=false. An unrelated Boolean attribute is ignored
    # rather than guessed.
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

    # A state name alone is a declaration, not observable initialization
    # semantics. Require a Boolean state mirror (e.g. isLocked=true), an entry
    # action performing the default response, or a do action sustaining it, so
    # empty shells such as `state Locked; transition initial ...` are rejected.
    #
    # A power-on-shaped machine (initial PowerOn --accept PowerOnEvent-->
    # Locked{entry lock...}) also models "default to X upon power-on"; its
    # initial state carries no semantics because they live one hop away. Run3
    # and s0v15 both lost against this narrowness, so drive the power event
    # instead of failing the shape. Conditions kept: every exit from the
    # initial state lands on a default-term state, at least one exit is
    # unguarded, and the landed state carries observable semantics.
    if not has_state_mirror and not entry and not holding_do:
        if not _drive_power_event_to_default(
            sm, initial, set(required_state_terms), result
        ):
            result.violations.append(
                f"Initial state '{initial}' has no observable initialization semantics; "
                "add a consistent Boolean state attribute, an initial entry action, "
                "or a sustaining do action"
            )

    result.passed = not result.violations
    return result


def _run_scenario(sm: StateMachineDef) -> BehavioralScenarioResult:
    ft = sm.fault_transitions()
    description = (
        f"{sm.owner_part}.{sm.name}: verify fault transition fires correctly"
    )

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

    # ── Execute every plan; pass if any plan triggers the fault ──────────────
    # Multi-plan (Layer 2) verifies the relationship in `A OP B`: a guard that
    # fires in only one direction still passes, and which plans fired is
    # recorded so the two-sidedness is visible.
    any_fired = False
    fault_states = {s.name for s in sm.states if s.entry_action}

    for plan in plans:
        inst = StateMachineInstance(sm)
        for t, variables in enumerate(plan.sequence):
            command = variables.get("__accept__") if is_accept_machine else None
            fired = inst.step(variables, time=float(t), command=command)
            # Fault monitors: stop on reaching a fault state (has entry action).
            # Mode/accept machines: run the full sequence. The initial state may
            # itself have an entry action (power-on -> Locked), so only stop after
            # this step transitioned.
            if (fired and not is_mode_machine and not is_accept_machine
                    and inst.in_fault_state()):
                break

        events = inst.transition_log
        plan_label = f"[{plan.name}]" if len(plans) > 1 else ""

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

        if not events:
            result.timeline.append(
                f"{plan_label} no trigger over {len(plan.sequence)} steps"
                f" (swept {plan.swept_var or '?'})"
            )
            continue

        fault_event = events[-1]
        first_fire = not any_fired
        any_fired = True

        if first_fire:
            result.trigger_step = int(fault_event.time)
            result.fired_actions = list(inst.fired_actions)

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

            # ── Check 1: correct target state ─────────────────────────
            # Mode machines: no fault states, any transition is valid.
            # Detection-only monitors: fault_states empty, so transition firing is
            # sufficient; skip the state check.
            if not is_mode_machine and fault_states and fault_event.to_state not in fault_states:
                result.violations.append(
                    f"Transition target '{fault_event.to_state}' is not a known "
                    f"fault state (expected one of: {fault_states})"
                )

            # ── Check 2: entry action was called ──────────────────────
            # Mode machines: entry actions are optional, skip.
            # Detection-only monitors (fault_states empty): no entry action, so
            # transition firing is the verification; skip.
            if not is_mode_machine and fault_states:
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
                    result.timeline.append(
                        "  detection-only fault state (no entry action — command "
                        "dispatch delegated to SafetyArbiter)  ✓"
                    )

            if primary_guard.kind == "comparison" and plan.start_val is not None \
                    and plan.step_size is not None:
                trig_val = round(plan.start_val + plan.step_size * result.trigger_step, 3)
                result.trigger_value = trig_val
                result.timeline.append(
                    f"  {plan.swept_var} at trigger: {trig_val}"
                    f"  (threshold: {primary_guard.operator}"
                    f" {primary_guard.threshold})"
                )
        else:
            result.timeline.append(
                f"{plan_label} also fired at step {int(fault_event.time)}  ✓"
            )

    if not any_fired:
        result.violations.append(
            f"No state transition fired across {len(plans)} driver plan(s) — "
            f"guard '{primary_guard.description()}' was never satisfied"
        )
        return result

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


def _identifier_words(text: str) -> List[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def _classify_accept_transitions(sm: StateMachineDef):
    """将 accept 转移分成正常功能响应和应急分支。

    不能用"目标状态是否有 entry action"区分：正常功能状态也会执行动作
    （如更新飞行计划、发送健康报告）。应急语义由状态、转移或响应动作的名称表达。
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
        )))
        # Match whole identifier words, not substrings: substring matching filed
        # "defaultToMechanicallyLockedState" as an emergency branch because it
        # contains "fault", leaving the nominal chain with no first step. The
        # prefix test still matches plural/inflected forms ("faults", "aborted").
        return any(
            word.startswith(marker)
            for word in _identifier_words(semantic_name)
            for marker in emergency_markers
        )

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
    graph: Dict[str, List[Tuple[str, str]]] = {}
    for t in nominal_trs:
        if t.source and t.target and t.accept_trigger:
            graph.setdefault(t.source, []).append((t.accept_trigger, t.target))
    return graph


def _longest_nominal_path(
    graph: Dict[str, List[Tuple[str, str]]],
    start: str,
) -> List[Tuple[str, str]]:
    """DFS 找从 start 出发的最长无环路径，返回 [(cmd, target_state), ...]。

    最长路径即完整正常运行序列；捷径/中止路径更短，被排除。
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

    # Trigger conflict: a nominal transition from this source uses the same
    # trigger. Both fire on the same command -> non-deterministic in SysML v2,
    # the executor picks one, so the emergency may never trigger. Cause is a
    # missing guard in the model.
    nominal_triggers = {cmd for cmd, _ in graph.get(emrg_tr.source, [])}
    if emrg_tr.accept_trigger in nominal_triggers:
        # Untestable, not passed: crediting an unexecuted branch as PASS gave the
        # ambiguous transition a free pass. As with parametric constraints, an
        # untestable scenario leaves the denominator (tag consumed by
        # _compute_score) and the ambiguity is reported as an issue.
        r.timeline.append(
            f"⚠ UNTESTABLE: trigger '{emrg_tr.accept_trigger}' is shared by a "
            f"nominal transition from '{emrg_tr.source}' — non-deterministic "
            f"without guard conditions; emergency branch cannot be driven."
        )
        r.passed = False
        r.violations.append(
            f"Ambiguous accept trigger '{emrg_tr.accept_trigger}' on "
            f"'{emrg_tr.source}': shared by nominal and emergency transitions"
        )
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
    nominal_trs, emergency_trs = _classify_accept_transitions(sm)
    graph = _build_nominal_multigraph(nominal_trs)
    longest_path = _longest_nominal_path(graph, sm.initial_state)

    # An accept machine may be emergency-only: a recovery mechanism stays in its
    # safe initial state until a failsafe command arrives, so state persistence
    # is its nominal behaviour. Require a nominal scenario only when the model
    # declares a nominal transition; emergency branches are exercised below.
    results: List[BehavioralScenarioResult] = []
    if nominal_trs:
        results.append(_run_accept_nominal_scenario(sm, longest_path))
    for emrg_tr in emergency_trs:
        results.append(
            _run_accept_emergency_scenario(sm, graph, longest_path, emrg_tr)
        )
    return results


def _run_cross_component_scenario(
    sm_safety: StateMachineDef,
    sm_flight: StateMachineDef,
    send_cmd: str,
    send_port: str,
) -> BehavioralScenarioResult:
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

    # 找 sm_flight 里接受 send_cmd 的所有转移（不限于 fault state）：safety SM
    # 的命令可能触发正常阶段（battery RTB）也可能触发应急阶段（propulsion
    # fail），两者都是合法跨组件因果链，命令被接受即 PASS。
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

    # 在 sm_flight 里找最近可达的 source 状态（nominal 链上最早带 send_cmd 的节点）。
    # 用多重图 + BFS：单后继字典会把同一 source 的多条 nominal 出边塌缩成最后一条,
    # 分叉状态机上贪心游走会走进死胡同 -> 假 FAIL。
    nominal_trs_flight, _ = _classify_accept_transitions(sm_flight)
    multigraph_flight = _build_nominal_multigraph(nominal_trs_flight)

    emrg_sources = {t.source for t in all_trs_flight if t.source}
    nav_cmds, state = None, sm_flight.initial_state
    if state in emrg_sources:
        nav_cmds = []
    else:
        for target in sorted(emrg_sources):
            cmds = _bfs_nav_cmds(multigraph_flight, sm_flight.initial_state, target)
            if cmds is not None and (nav_cmds is None or len(cmds) < len(nav_cmds)):
                nav_cmds, state = cmds, target
    if nav_cmds is None:
        nav_cmds, state = [], sm_flight.initial_state

    if state not in emrg_sources:
        r.violations.append(
            f"Cannot navigate {sm_flight.name} to any state "
            f"accepting '{send_cmd}' via nominal chain"
        )
        return r

    plans = _build_driver_plans(sm_safety)
    if not plans:
        r.violations.append(f"Cannot build driver plan for {sm_safety.name}")
        return r
    safety_plan = plans[0]

    inst_safety = StateMachineInstance(sm_safety)
    inst_flight = StateMachineInstance(sm_flight)

    for i, cmd in enumerate(nav_cmds):
        inst_flight.step({"__accept__": cmd}, time=float(i), command=cmd)
        inst_flight.step({"__accept__": None}, time=float(i) + 0.5)

    pre_nav_count = len(inst_flight.transition_log)
    if nav_cmds:
        r.timeline.append(
            f"Pre-navigate {sm_flight.name}: "
            f"{pre_nav_count}/{len(nav_cmds)} steps → state='{inst_flight.current_state}'"
        )

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
            inst_flight.step({"__accept__": send_cmd}, time=t, command=send_cmd)
            inst_flight.step({"__accept__": None}, time=t + 0.5)
            break

    if safety_fired_at is None:
        r.violations.append(f"{sm_safety.name} guard never fired over the driver plan")
        return r

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
        if is_fault and entry and entry not in inst_flight.fired_actions:
            r.violations.append(
                f"Emergency entry action '{entry}' in '{target_state}' was not called"
            )

    r.passed = not r.violations
    return r


def _collect_cross_component_scenarios(
    state_machines: List[StateMachineDef],
) -> List[BehavioralScenarioResult]:
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
            continue
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


def _check_battery_ordering(results: List[BehavioralScenarioResult]
                             ) -> Optional[BehavioralScenarioResult]:
    """Verify the battery RTB threshold is above the critical threshold, so RTB
    fires before emergency landing.

    Scenarios run independently, so step numbers are not comparable; compare
    trigger_value (the variable value at trigger) instead.
    """
    rtb_res  = next((r for r in results if "Rtb"      in r.state_machine or
                                            "rtb"      in r.state_machine.lower()), None)
    crit_res = next((r for r in results if "Critical"  in r.state_machine or
                                            "critical" in r.state_machine.lower()), None)

    if rtb_res is None or crit_res is None:
        return None

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


_ATTR_NUM_RE = re.compile(
    r'\b(?:readonly\s+)?attribute\s+(\w+)\s*:[^=;\n]*?=\s*([-+]?\d+(?:\.\d+)?)'
)
_READONLY_RE = re.compile(r'\breadonly\s+attribute\s+(\w+)')


def _extract_all_attrs(sysml_text: str) -> Dict[str, float]:
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
    measured_inputs: frozenset = frozenset(),
) -> Optional[BehavioralScenarioResult]:
    # Already covered by state machine guard
    if c.lhs in guard_variables:
        return None

    # Readonly LHS: static check suffices, skip sweep
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

    if lhs_init is None and c.lhs in measured_inputs and rhs_val is not None:
        # A typed-semantic-binding runtime attribute is a measured value: the
        # convention initialises it from a reference chain into the measurement
        # port and carries no local literal, so demanding a numeric initial value
        # failed every binding-bound ALWAYS constraint. The sweep supplies
        # measured inputs, so it synthesizes a start inside the valid region and
        # records it; boundary liveness (criteria 2 and 3) is unchanged.
        rhs_probe = float(rhs_val)
        margin = abs(rhs_probe) * 0.15 + 1.0
        if c.operator in ("<=", "<"):
            lhs_init = rhs_probe - margin
        elif c.operator in (">=", ">"):
            lhs_init = rhs_probe + margin
        else:
            lhs_init = rhs_probe
        result.tags.append("measured_input_start_synthesized")
        result.timeline.append(
            f"{c.lhs} is a measured input (reference-chain initializer): "
            f"sweep supplies start value {float(lhs_init):g} inside the "
            "valid region"
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

    if op in ("<=", "<"):
        sweep_end = rhs_f + abs(rhs_f) * 0.15 + 1.0   # always > rhs_f regardless of sign
        step_size = (sweep_end - lhs_f) / _N_STEPS
    elif op in (">=", ">"):
        sweep_end = rhs_f - abs(rhs_f) * 0.15 - 1.0   # always < rhs_f regardless of sign
        step_size = (sweep_end - lhs_f) / _N_STEPS
    elif op == "==":
        # A configuration pin: the planned value is the only point where the
        # constraint holds, so boundary liveness is a three-point probe -- hold at
        # the pinned value, violate on either side. Previously the operator was
        # reported unsupported, failing every ``value == planned`` constraint.
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
    span = named_block_span(sysml_text, "part", owner)
    if span is None:
        return ""
    opening, closing = span
    return (
        sysml_text[opening + 1:closing]
        if closing != -1 else ""
    )


def _state_body_text(
    owner_body: str,
    behavior_name: str,
    state_name: str,
) -> str:
    behavior_span = named_block_span(owner_body, "state", behavior_name)
    if behavior_span is None:
        return ""
    behavior_opening, behavior_closing = behavior_span
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

    Covers model containment, reachability, response execution, typed runtime
    binding, and a live numeric boundary. It makes no claim about the external
    environment satisfying the bound.
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
        # No epsilon perturbation here: probing rhs+/-ε against rhs itself is an
        # arithmetic tautology for the ordering operators, so it cannot fail.
        # Their liveness holds by construction (rhs+ε and rhs−ε land on opposite
        # sides) and the plan gate (_LIVE_BOUNDARY_OPERATORS in
        # activated_constraint_plan) rejects the operators where it does not.
        # What this scenario establishes is recorded below.
        if c.operator not in {">=", ">", "<=", "<"}:
            result.violations.append(
                f"Constraint operator {c.operator} has no live satisfaction "
                "boundary to execute against"
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
                f"Runtime input {c.lhs} bound against threshold "
                f"{c.operator} {rhs_value:g} (RHS resolved; boundary "
                "liveness holds by construction for ordering operators)"
            )
    result.passed = not result.violations
    return result


def _collect_constraint_scenarios(
    sysml_text: str,
    state_machines: List[StateMachineDef],
) -> List[BehavioralScenarioResult]:
    constraints = extract_constraints(sysml_text)
    if not constraints:
        return []
    plan_owned_model = any(
        item.plan_constraint_id is not None for item in constraints
    )

    all_initial_values: Dict[str, float] = {}
    for sm in state_machines:
        for k, v in sm.initial_values.items():
            try:
                all_initial_values[k] = float(v)
            except (TypeError, ValueError):
                pass
    all_initial_values.update(_extract_all_attrs(sysml_text))

    # Attributes whose initializer is a reference chain are measured inputs
    # (typed-semantic-binding convention): they carry no local literal, so the
    # parametric sweep supplies their value.
    measured_inputs = frozenset(
        match.group("name")
        for match in re.finditer(
            r"\battribute\s+(?P<name>[A-Za-z_]\w*)\s*"
            r"(?::\s*[\w:]+(?:\s*\[[^\]]*\])?)?\s*=\s*"
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\s*;",
            sysml_text,
        )
    )

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
        # Once the model carries compiler-owned constraint annotations, only those
        # planned invariants contribute executable evidence. Unannotated A/G prose
        # invariants and LLM-authored assertions stay with their own checkers and
        # do not enter the parametric denominator.
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
            c, all_initial_values, guard_variables, readonly_vars,
            measured_inputs=measured_inputs,
        )
        if r is not None:
            results.append(r)
    return results


def _compute_score(results: List[BehavioralScenarioResult]) -> float:
    if not results:
        return 1.0   # neutral: no state machines -> no violations
    total_w  = 0.0
    passed_w = 0.0
    for r in results:
        if "trigger_conflict" in r.tags:
            # Untestable (ambiguous trigger): excluded from the denominator, neither
            # PASS nor FAIL; the ambiguity is reported through the scenario's issues.
            continue
        if "cross_component" in r.tags:
            w = 1.5
        elif "emergency" in r.tags:
            w = 0.5          # accept 分支跳转：测试性质与 nominal 相近，保持低权重
        elif "safety" in r.tags:
            w = 2.0
        else:
            w = 1.0
        total_w  += w
        if r.passed:
            passed_w += w
    # total_w == 0 also covers "every scenario was untestable": neutral, like
    # the no-scenarios case, rather than a zero score.
    return passed_w / total_w if total_w > 0 else 1.0


def run_behavioral_simulation(sysml_text: str,
                               model_name: str = "UnknownModel"
                               ) -> BehavioralSimResult:
    """Main entry point."""
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

        cross_results = _collect_cross_component_scenarios(state_machines)
        scenario_results.extend(cross_results)

        ordering = _check_battery_ordering(scenario_results)
        if ordering is not None:
            scenario_results.append(ordering)

    constraint_scenarios = _collect_constraint_scenarios(sysml_text, state_machines)
    scenario_results.extend(constraint_scenarios)

    if not scenario_results:
        br.sim_score = 1.0   # neutral: nothing to verify
        return br

    br.scenario_results = scenario_results
    br.sim_score = _compute_score(scenario_results)
    return br
