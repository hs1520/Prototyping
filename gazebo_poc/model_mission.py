"""Let the generated model own the decision, and the harness only actuate it.

Why this exists
---------------
Every payload and parachute result in the 2026-08-30 authoritative run carried
the same caveat, written by the harness about itself:

    "the coordinate condition was evaluated by the harness, not by generated
     mission logic"
    "this exercises trajectory/position/actuator coupling but not generated
     mission-logic ownership of the trigger"

That caveat is the difference between "Gazebo can separate a joint" and "the
model we generated releases the payload". A reviewer's first question about any
of those numbers is which of the two was demonstrated, and the honest answer was
the first.

What this changes
-----------------
The generated model is an event-driven machine: every behaviour is
``accept <Event> -> <State> { entry action ... }``. It does not compute the
delivery condition itself — it consumes an event that something else produces.
So the faithful split is:

  * the harness derives events from telemetry (it is the event source the model
    declares) and executes whatever action the model fires;
  * the generated state machine owns the transition, the arbitration between
    competing behaviours, and the action.

:class:`ModelDrivenMission` is that boundary. ``offer()`` hands an event to the
generated machines and returns the actions THEY fired. When the model fires
nothing, the harness does nothing — which is what makes a suppressed release
(an abort condition holding the payload locked) a real observation rather than
a harness policy.

Honesty boundary
----------------
This does not make the model's *timing* physical: the machines step in Python,
so an event-to-action interval measured here is plumbing, not a system latency.
What it establishes is ownership — which artefact decided — plus the arbitration
between behaviours. Physical timing still comes from the Gazebo observation of
the actuation that follows.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class ModelAction(str, Enum):
    """Action definitions that the physical Gazebo adapter can execute."""

    RELEASE_PAYLOAD = "actuateRelease"
    DEPLOY_PARACHUTE = "deployParachute"


@dataclass(frozen=True)
class ModelDecision:
    """One transition the generated model fired, and the action it entered."""

    time: float
    owner_part: str
    machine: str
    event: str
    from_state: Optional[str]
    to_state: Optional[str]
    action: Optional[str]
    action_definition: Optional[str]

    def as_dict(self) -> dict:
        return {
            "time": round(self.time, 6),
            "owner_part": self.owner_part,
            "machine": self.machine,
            "event": self.event,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action": self.action,
            "action_definition": self.action_definition,
            "decided_by": "generated model",
        }


@dataclass
class ModelDrivenMission:
    """Generated state machines driven by events, owning their own actions."""

    model_text: str
    machines: Dict[str, Any] = field(default_factory=dict)
    decisions: List[ModelDecision] = field(default_factory=list)
    offered: List[Tuple[float, str]] = field(default_factory=list)
    _digest: str = ""

    def __post_init__(self) -> None:
        from src.simulation.state_extractor import extract_state_machines
        from src.simulation.state_executor import StateMachineInstance

        self._digest = hashlib.sha256(
            (self.model_text or "").encode("utf-8")
        ).hexdigest()
        for sm in extract_state_machines(self.model_text or ""):
            # Every machine is loaded, including purely guard-driven ones.
            # offer() steps them all, and a guard transition fires on the
            # variables it is given whether or not the event names it — so a
            # guard-only machine IS drivable. Skipping them hid the one that
            # matters most: SafetyArbiter expresses REQ-SAFE-005's precedence
            # entirely in guards ("... and not propulsionCriticalFailure"), and
            # while it was filtered out the precedence check saw no competing
            # response to take precedence over, and returned inconclusive
            # forever.
            self.machines[f"{sm.owner_part}.{sm.name}"] = StateMachineInstance(sm)

    # -- introspection ----------------------------------------------------

    def accepted_events(self) -> Tuple[str, ...]:
        """Every event name the loaded machines can consume."""
        events = {
            transition.accept_trigger
            for instance in self.machines.values()
            for transition in instance.sm.transitions
            if transition.accept_trigger
        }
        return tuple(sorted(events))

    def handles(self, event: str) -> bool:
        return event in self.accepted_events()

    def state_of(self, machine: str) -> Optional[str]:
        instance = self.machines.get(machine)
        return None if instance is None else instance.current_state

    def action_definitions_for_machine(self, machine: str) -> Tuple[str, ...]:
        """Executable response definitions declared by a named state machine."""
        actions = {
            action_definition
            for instance in self.machines.values()
            if instance.sm.name == machine
            for state in instance.sm.states
            for action_definition in (
                state.entry_action_def,
                state.do_action_def,
            )
            if action_definition is not None
        }
        return tuple(sorted(actions))

    # -- driving ----------------------------------------------------------

    def offer(self, event: str, *, time: float,
              variables: Optional[Mapping[str, Any]] = None
              ) -> Tuple[ModelDecision, ...]:
        """Offer one event to every loaded machine; return what the model fired.

        An empty result means the generated logic declined to act. The caller
        must then do nothing — that is the whole point of asking it.
        """
        self.offered.append((float(time), str(event)))
        env = dict(variables or {})
        fired: List[ModelDecision] = []
        for name, instance in self.machines.items():
            before = instance.current_state
            if not instance.step(env, time=float(time), command=event):
                continue
            target = instance.current_state
            decision = ModelDecision(
                time=float(time),
                owner_part=instance.sm.owner_part,
                machine=instance.sm.name,
                event=event,
                from_state=before,
                to_state=target,
                action=(instance.sm.response_action_for_state(target)
                        if target else None),
                action_definition=(
                    instance.sm.response_action_definition_for_state(target)
                    if target else None
                ),
            )
            fired.append(decision)
            self.decisions.append(decision)
        return tuple(fired)

    def actions_for(self, decisions: Sequence[ModelDecision]) -> Tuple[str, ...]:
        return tuple(d.action for d in decisions if d.action)

    def performed(
        self, decisions: Sequence[ModelDecision], action: ModelAction,
    ) -> bool:
        """Whether this transition set invoked the exact executable action."""
        return any(
            decision.action_definition == action.value
            for decision in decisions
        )

    # -- evidence ---------------------------------------------------------

    def provenance(self) -> dict:
        """What was executed, so the evidence can name it."""
        return {
            "model_sha256": self._digest,
            "machines": sorted(self.machines),
            "accepted_events": list(self.accepted_events()),
            "decision_owner": "generated model state machines",
            "harness_role": (
                "derives events from telemetry and actuates the action the "
                "model fires; it does not decide whether to act"
            ),
        }

    def decision_log(self) -> List[dict]:
        return [d.as_dict() for d in self.decisions]
