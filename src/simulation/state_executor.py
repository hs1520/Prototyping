"""
state_executor.py

单个状态机实例的执行引擎。

用法:
    inst = StateMachineInstance(sm_def)
    for t, variables in enumerate(test_sequence):
        fired = inst.step(variables, time=float(t))
    print(inst.transition_log)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state_extractor import GuardCondition, StateMachineDef


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------

@dataclass
class TransitionEvent:
    time: float
    from_state: str
    to_state: str
    transition_name: Optional[str]
    entry_action: Optional[str]
    guard_description: str
    variables_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        action_str = f"  → entry: {self.entry_action}" if self.entry_action else ""
        return (
            f"[t={self.time:.0f}]  {self.from_state} → {self.to_state}"
            f"  ({self.guard_description}){action_str}"
        )


# ---------------------------------------------------------------------------
# State machine instance
# ---------------------------------------------------------------------------

class StateMachineInstance:
    """
    Executes a StateMachineDef step-by-step given a sequence of variable dicts.

    State is reset between scenario runs via reset().
    """

    def __init__(self, sm: StateMachineDef) -> None:
        self.sm = sm
        self.current_state: Optional[str] = sm.initial_state
        self.transition_log: List[TransitionEvent] = []
        self.fired_actions: List[str] = []

    def reset(self) -> None:
        self.current_state = self.sm.initial_state
        self.transition_log.clear()
        self.fired_actions.clear()

    # ------------------------------------------------------------------ #
    #  Single time step                                                   #
    # ------------------------------------------------------------------ #

    def step(self, variables: Dict[str, Any], time: float = 0.0,
             command: Optional[str] = None) -> bool:
        """
        Evaluate all outgoing transitions from the current state and fire the
        first eligible one.

        Eligibility rules:
          • Accept-triggered transition  → fires when *command* matches
            ``tr.accept_trigger``; optional guards must also hold.
          • Guard-only transition        → fires when all guards hold
            (existing behaviour, *command* is ignored).

        Returns True if a transition fired this step.
        """
        if self.current_state is None:
            return False

        for tr in self.sm.transitions:
            if tr.is_initial:
                continue
            if tr.source != self.current_state:
                continue

            if tr.accept_trigger:
                # Accept-triggered: command must match
                if tr.accept_trigger != command:
                    continue
                # Optional guard on top of accept
                if tr.guards and not all(
                    self._eval_guard(g, variables) for g in tr.guards
                ):
                    continue
                self._fire_transition(tr, time, variables,
                                      guard_desc=f"accept {tr.accept_trigger}")
                return True

            # Guard-only transition (original behaviour)
            if not tr.guards:
                continue
            if all(self._eval_guard(g, variables) for g in tr.guards):
                guard_desc = " AND ".join(g.description() for g in tr.guards)
                self._fire_transition(tr, time, variables, guard_desc=guard_desc)
                return True

        return False

    def _fire_transition(
        self,
        tr,
        time: float,
        variables: Dict[str, Any],
        guard_desc: str = "",
    ) -> None:
        """Record a transition firing and advance current_state."""
        old_state = self.current_state
        self.current_state = tr.target

        entry = (
            self.sm.response_action_for_state(tr.target)
            if tr.target else None
        )
        if entry:
            self.fired_actions.append(entry)

        snap_attrs = (
            tr.guards[0].involved_attributes()
            if tr.guards else []
        )
        event = TransitionEvent(
            time=time,
            from_state=old_state,
            to_state=tr.target or "?",
            transition_name=tr.name,
            entry_action=entry,
            guard_description=guard_desc,
            variables_snapshot={k: variables[k] for k in snap_attrs if k in variables},
        )
        self.transition_log.append(event)

    def in_fault_state(self) -> bool:
        """True if the current state has an entry action (i.e. a fault state)."""
        if self.current_state is None:
            return False
        return self.sm.entry_action_for_state(self.current_state) is not None

    # ------------------------------------------------------------------ #
    #  Guard evaluation                                                   #
    # ------------------------------------------------------------------ #

    def _eval_guard(self, guard: GuardCondition, variables: Dict[str, Any]) -> bool:
        # Layer 1: when the guard carries full expression trees, evaluate them
        # against the COMPLETE environment (driven variables overlaid on the
        # owner part's initial values), so a variable/arithmetic RHS such as
        # `batteryCharge <= returnEnergyRequired` resolves correctly.
        if guard.kind == "comparison" and guard.lhs is not None and guard.rhs is not None:
            env: Dict[str, Any] = {**(self.sm.initial_values or {}), **variables}
            return guard.eval(env)

        if guard.kind == "comparison":
            val = variables.get(guard.attribute)
            if val is None:
                return False
            try:
                v = float(val)
            except (TypeError, ValueError):
                return False
            op = guard.operator
            th = guard.threshold
            if op == "<":  return v <  th
            if op == "<=": return v <= th
            if op == ">":  return v >  th
            if op == ">=": return v >= th
            if op == "==": return v == th
            if op == "!=": return v != th
            return False

        if guard.kind == "bool_true":
            return bool(variables.get(guard.attribute, False))

        if guard.kind == "bool_false":
            return not bool(variables.get(guard.attribute, False))

        if guard.kind == "enum_eq":
            env: Dict[str, Any] = {**(self.sm.initial_values or {}), **variables}
            current = env.get(guard.attribute)
            if current is None:
                return False
            return str(current) == guard.enum_value

        if guard.kind == "compound":
            if guard.compound_op == "and":
                return all(self._eval_guard(op, variables) for op in guard.operands)
            if guard.compound_op == "or":
                return any(self._eval_guard(op, variables) for op in guard.operands)

        return False
