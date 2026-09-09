"""Let the generated model own the mission decision; the harness only actuates it.

The generated model is an event-driven machine (``accept <Event> -> <State> { entry action }``)
and does not compute the delivery condition itself, so the split is: the harness derives events
from telemetry and executes whatever action the model fires, while the generated state machine
owns the transition, the arbitration between competing behaviours, and the action.
:class:`ModelDrivenMission` is that boundary - ``offer()`` hands an event to the machines and
returns the actions they fired, so a suppressed release (an abort holding the payload locked) is
an observation rather than a harness policy. The machines step in Python, so an event-to-action
interval measured here is plumbing, not system latency; what it establishes is which artefact
decided. Physical timing still comes from the Gazebo observation of the actuation.
"""
from __future__ import annotations

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
    # Boolean guard flags raised by the events offered so far, plus which event
    # raised each, so the evidence can say the model was told the condition held.
    conditions: Dict[str, Any] = field(default_factory=dict)
    latched: List[Tuple[float, str, str]] = field(default_factory=list)
    # (time, requested, resolved, how) - every event rename is recorded so the
    # evidence can name it.
    resolutions: List[Tuple[float, str, str, str]] = field(default_factory=list)
    # (time, requested, why) - the offer established nothing: no declared event
    # covered it and no guard flag matched it, so the model was never asked.
    unresolved: List[Tuple[float, str, str]] = field(default_factory=list)
    # (time, requested, flags) - no event matched but the offer raised guard flags,
    # so the condition was established: the model expresses it as a standing
    # boolean rather than an event.
    condition_only: List[Tuple[float, str, Tuple[str, ...]]] = field(
        default_factory=list)
    # (adapter_constant, model_action) - actuation identities accepted by causal
    # role rather than spelling; the rename is recorded.
    action_resolutions: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        from src.simulation.state_extractor import extract_state_machines
        from src.simulation.state_executor import StateMachineInstance

        for sm in extract_state_machines(self.model_text or ""):
            # Load every machine, including guard-only ones: offer() steps them all and a
            # guard transition fires on the variables it is given whether or not the event
            # names it. Filtering them out dropped SafetyArbiter, which expresses
            # REQ-SAFE-005's precedence entirely in guards, so the precedence check saw no
            # competing response and returned inconclusive.
            self.machines[f"{sm.owner_part}.{sm.name}"] = StateMachineInstance(sm)

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
        """Map a scenario's canonical event onto the name this model declares.

        A generated model names its own events: run3 accepts
        ``DeliveryCoordinateConditionSatisfied`` where the harness offers
        ``DeliveryCoordinateSatisfied``, and against hard-coded spellings nothing
        fired. Matching is therefore on meaning - all of the scenario's words must
        appear in the declared event's words, which admits a model that says more
        (``...ConditionSatisfied``) and rejects one that says something else. The
        most specific match wins; a tie is refused rather than guessed.
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

        An unset flag evaluates false, so a guarded model behaves like an unguarded
        one: with no variables bound, a ``not deliveryAbortActive`` guard still fired
        the release. Flags are bound by matching the offered event's words against the
        words in the flags the model declared, not against a name this harness holds,
        so REQ-SAFE-006 stays checkable whatever the model calls the flag. Only
        boolean flags are latched - writing True into a numeric attribute would
        corrupt a comparison guard.
        """
        from src.prototyping.verification_obligations import semantic_terms

        offered = semantic_terms(event)
        if not offered:
            return ()
        # Every word of the event must appear in the flag's name; a shared word is not
        # enough. "DeliveryCoordinateSatisfied" and "deliveryAbortActive" share
        # "delivery", and latching on that raised the abort flag from the delivery
        # event itself, inhibiting the ordinary release.
        return tuple(
            name for name in self.boolean_guard_attributes()
            if offered <= semantic_terms(name)
        )

    def guards_reaching_action(self, action_definition: str) -> Tuple[str, ...]:
        """Guard descriptions on every transition that reaches an action.

        Empty means the action fires unconditionally. The action name is a model
        identity, so querying with the harness's own spelling returns () for a model
        that names the action differently - indistinguishable from unconditional,
        which is how a guarded run3 was reported as carrying no inhibition logic.
        Prefer :meth:`guards_for_event`, whose identity comes from the model via
        event resolution.
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
        """Guards on every transition that fires in response to *event*; identity by
        causal role, no action names involved.

        The canonical event is resolved onto the model's own declaration first, and
        the transitions accepting the resolved event are the response the requirement
        is about, whatever the model called their actions. Returns:

        - ``None``  - the event did not resolve: the question could not be
          put to this model, so the caller reports "not measured",
          not "unguarded";
        - ``()``    - resolved, and the responding transitions carry no
          guard: a finding about the model;
        - guards    - resolved and guarded.
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

        These read false, so the model behaves as if unguarded on them and is
        indistinguishable from a model with no guard. A verdict resting on inhibition
        checks this and reports inconclusive rather than blaming the model.
        """
        return tuple(
            name for name in self.boolean_guard_attributes()
            if name not in self.conditions
        )

    def offer(self, event: str, *, time: float,
              variables: Optional[Mapping[str, Any]] = None
              ) -> Tuple[ModelDecision, ...]:
        """Offer one event to every loaded machine; return what the model fired.

        An empty result means the generated logic declined to act, and the caller then
        does nothing.
        """
        requested = str(event)
        resolved, how = self.resolve_event(requested)
        if resolved is not None and resolved != requested:
            self.resolutions.append((float(time), requested, resolved, how))
        event = resolved or requested
        self.offered.append((float(time), str(event)))
        # Conditions latch: "whenever an abort is active" is a standing state, so a
        # flag raised by an earlier event is still true at the next offer. An explicit
        # caller value wins.
        raised = self._latched_by(requested) or self._latched_by(event)
        for name in raised:
            self.conditions[name] = True
            self.latched.append((float(time), str(event), name))
        # A model may express a condition as a standing boolean read by guards rather
        # than as an event, so "no event matched" does not mean the model was never
        # told - the latched flag told it. Only an offer that resolved to nothing and
        # raised nothing established nothing; the two are recorded separately.
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

    def performed(
        self, decisions: Sequence[ModelDecision], action: ModelAction,
    ) -> bool:
        """Whether this transition set invoked the executable action.

        Verbatim match first, then identity by causal role: the decisions are the
        model's response to one offered event, so when they invoke exactly one
        distinct action, that action is the response whatever the model named it
        (run3 fires ``releasePayload`` where the adapter constant says
        ``actuateRelease``, and the literal comparison stopped the harness actuating
        at all). Causal role alone would accept ``lockPayload`` as a release, so the
        fallback carries the same semantic gate as resolve_event: the fired action
        must share a term with the adapter's action name (release↔release,
        deploy/parachute↔parachute). Two distinct fired actions are refused rather
        than guessed, and every causal-role acceptance is recorded in
        ``action_resolutions``.
        """
        from src.utils.sysml_text_utils import semantic_terms

        if any(
            decision.action_definition == action.value
            for decision in decisions
        ):
            return True
        fired = {
            decision.action_definition
            for decision in decisions
            if decision.action_definition
        }
        if len(fired) == 1:
            (resolved,) = fired
            if semantic_terms(resolved) & semantic_terms(action.value):
                self.action_resolutions.append((action.value, resolved))
                return True
        return False

    def provenance(self) -> dict:
        """What was executed, so the evidence can name it."""
        return {
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
