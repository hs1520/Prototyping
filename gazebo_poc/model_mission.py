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
    #: Boolean guard flags raised by the events offered so far, and the record
    #: of which event raised each — evidence has to be able to say the model
    #: was actually told the condition held.
    conditions: Dict[str, Any] = field(default_factory=dict)
    latched: List[Tuple[float, str, str]] = field(default_factory=list)
    #: (time, requested, resolved, how) — a harness that quietly renames the
    #: events it offers is its own hazard, so every rename is recorded and the
    #: evidence can name it.
    resolutions: List[Tuple[float, str, str, str]] = field(default_factory=list)
    #: (time, requested, why) — the offer established NOTHING: no declared
    #: event covered it and no boolean guard flag matched it. The model was
    #: never asked, so a scenario resting on it proves nothing.
    unresolved: List[Tuple[float, str, str]] = field(default_factory=list)
    #: (time, requested, flags) — no event matched, but the offer raised guard
    #: flags, so the condition WAS established. The model expresses this one as
    #: a standing boolean rather than an event; that is a modelling choice, not
    #: a gap.
    condition_only: List[Tuple[float, str, Tuple[str, ...]]] = field(
        default_factory=list)
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

    def resolve_event(self, event: str) -> Tuple[Optional[str], str]:
        """Map a scenario's canonical event onto the name THIS model declares.

        The harness held three hard-coded event spellings. A generated model is
        free to name its own events, and run3's did: it accepts
        ``DeliveryCoordinateConditionSatisfied`` where the harness offered
        ``DeliveryCoordinateSatisfied``. Nothing fired, and the resulting
        evidence would have read as "the generated logic declined to release" —
        a model defect that was really a harness spelling.

        Matching is on meaning, not on string equality: the scenario's words
        must all appear in the declared event's words. That admits a model that
        says more than the scenario (``...ConditionSatisfied``,
        ``...SubsystemFailure``) and rejects one that says something else. The
        most specific match wins, and a tie is refused rather than guessed —
        picking arbitrarily between two candidate events would silently decide
        which requirement the run exercised.
        """
        from src.prototyping.verification_obligations import semantic_terms

        declared = self.accepted_events()
        if event in declared:
            return event, "declared verbatim"
        wanted = semantic_terms(event)
        if not wanted:
            return None, "the scenario event carries no semantic terms"
        candidates = [
            name for name in declared if wanted <= semantic_terms(name)
        ]
        if not candidates:
            return None, (
                f"no declared event covers {sorted(wanted)}; the model declares "
                f"{list(declared)}"
            )
        ranked = sorted(candidates, key=lambda n: (len(semantic_terms(n)), n))
        best = semantic_terms(ranked[0])
        tied = [n for n in ranked if len(semantic_terms(n)) == len(best)]
        if len(tied) > 1:
            return None, f"ambiguous: {tied} all cover {sorted(wanted)}"
        return ranked[0], f"resolved from {event} by semantic match"

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

    def boolean_guard_attributes(self) -> Tuple[str, ...]:
        """Boolean flags the loaded guards read, whatever the model calls them."""
        def walk(guard) -> List[str]:
            if guard.kind in {"bool_true", "bool_false"}:
                return [guard.attribute]
            return [name for operand in guard.operands for name in walk(operand)]

        return tuple(sorted({
            name
            for instance in self.machines.values()
            for transition in instance.sm.transitions
            for guard in transition.guards
            for name in walk(guard)
            if name
        }))

    def _latched_by(self, event: str) -> Tuple[str, ...]:
        """Boolean guard flags this event's own name says it raises.

        A guard is only a guard if something sets the flag it reads. An unset
        flag evaluates FALSE, so a correctly guarded model behaves exactly like
        an unguarded one — measured: with no variables bound, a
        ``not deliveryAbortActive`` guard still fired the release. That failure
        looks like a model defect and is a harness defect, so the flag has to be
        bound from something.

        It is bound by matching the offered event's words against the words in
        the flags the MODEL declared, never against a name this harness holds:
        REQ-SAFE-006 must be checkable on a model that calls the flag whatever
        it likes. Only boolean flags are latched — writing True into a numeric
        attribute would corrupt an unrelated comparison guard.
        """
        from src.prototyping.verification_obligations import semantic_terms

        offered = semantic_terms(event)
        if not offered:
            return ()
        # Every word the EVENT uses must appear in the flag's name. A shared
        # word is not enough: "DeliveryCoordinateSatisfied" and
        # "deliveryAbortActive" share "delivery", and latching on that raised
        # the abort flag from the delivery event itself — the guard then
        # inhibited the ordinary release too, turning a fix into a new false
        # negative. Subset says the event names the condition rather than
        # merely touching the same subject.
        return tuple(
            name for name in self.boolean_guard_attributes()
            if offered <= semantic_terms(name)
        )

    def guards_reaching_action(self, action_definition: str) -> Tuple[str, ...]:
        """Guard descriptions on every transition that reaches an action.

        Empty means the action fires unconditionally. "The model has no guard"
        and "the harness never raised the flag the guard reads" produce the
        same behaviour and are different findings, so a verdict on inhibition
        has to be able to tell them apart.

        The action name is a MODEL identity. A harness that queries this with
        its own canonical spelling gets () for a model that names the action
        differently — indistinguishable from "unconditional" — which is how a
        guarded run3 was reported as carrying no inhibition logic. Verdicts
        should prefer :meth:`guards_for_event`, whose identity comes from the
        model via event resolution.
        """
        return tuple(
            guard.description()
            for instance in self.machines.values()
            for transition in instance.sm.transitions
            if not transition.is_initial
            and instance.sm.response_action_definition_for_state(
                transition.target) == action_definition
            for guard in transition.guards
        )

    def guards_for_event(self, event: str) -> Optional[Tuple[str, ...]]:
        """Guards on every transition that fires in response to *event* —
        identity by causal role, no action names involved.

        The scenario's canonical event is resolved onto the model's own
        declaration first (layer 1); the transitions accepting the resolved
        event ARE the response the requirement is about, whatever the model
        called the actions they run. Returns:

        - ``None``  — the event did not resolve: the question could not be
          put to this model, so the caller must report "not measured",
          never "unguarded";
        - ``()``    — resolved, and the responding transitions carry no
          guard: a real finding about the model;
        - guards    — resolved and guarded.
        """
        resolved, _how = self.resolve_event(event)
        if resolved is None:
            return None
        return tuple(
            guard.description()
            for instance in self.machines.values()
            for transition in instance.sm.transitions
            if not transition.is_initial
            and transition.accept_trigger == resolved
            for guard in transition.guards
        )

    def unlatched_boolean_attributes(self) -> Tuple[str, ...]:
        """Boolean guard flags no offered event ever raised.

        These read FALSE, so the model behaves as if unguarded on them. That is
        indistinguishable from a model with no guard at all, which is why a
        verdict that rests on inhibition has to check this and report
        INCONCLUSIVE rather than blame the model for a condition it was never
        told about.
        """
        return tuple(
            name for name in self.boolean_guard_attributes()
            if name not in self.conditions
        )

    # -- driving ----------------------------------------------------------

    def offer(self, event: str, *, time: float,
              variables: Optional[Mapping[str, Any]] = None
              ) -> Tuple[ModelDecision, ...]:
        """Offer one event to every loaded machine; return what the model fired.

        An empty result means the generated logic declined to act. The caller
        must then do nothing — that is the whole point of asking it.
        """
        requested = str(event)
        resolved, how = self.resolve_event(requested)
        if resolved is not None and resolved != requested:
            self.resolutions.append((float(time), requested, resolved, how))
        event = resolved or requested
        self.offered.append((float(time), str(event)))
        # Conditions latch: "whenever an abort is active" is a standing state,
        # not an instant, so a flag raised by an earlier event is still true
        # when the next one is offered. An explicit caller value always wins.
        raised = self._latched_by(requested) or self._latched_by(event)
        for name in raised:
            self.conditions[name] = True
            self.latched.append((float(time), str(event), name))
        # A model may express a condition as a standing boolean read by guards
        # rather than as an event — run3's delivery abort has no event at all.
        # So "no event matched" does NOT mean the model was never told: the
        # latched flag told it. Only an offer that resolved to nothing AND
        # raised nothing established nothing, and that is the one a verdict
        # must never rest on. Recording those together would make an exercised
        # scenario read like an unasked one.
        if resolved is None:
            if raised:
                self.condition_only.append(
                    (float(time), requested, tuple(raised)))
            else:
                self.unresolved.append((float(time), requested, how))
        env = dict(self.conditions)
        env.update(variables or {})
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
