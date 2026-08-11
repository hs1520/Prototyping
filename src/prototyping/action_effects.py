"""Typed, checkable effects a planned response action is required to have.

The generation prompts define a behavioural implementation as a reachable state
whose entry action names a response, plus the transition that triggers it; the
action definition itself is allowed to stay empty, and 152 of the 155 archived
pilot definitions are.  Nothing downstream reads an action body except
``state_extractor.all_sends``, so an action that does nothing is indistinguishable
from one that does the right thing.

``ACTION_EFFECTS_V1`` is the bounded profile that closes that gap without
pretending to interpret arbitrary behaviour: a planned response action must send
one declared event out of one declared port, and that event must reach a
consumer that accepts it.  Numeric computation, continuous control and physical
effect stay with constraints, SITL and Gazebo; this profile does not claim them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from ..utils.sysml_text_utils import find_block_end


#: The only effect kind V1 can check end to end.
SEND_EVENT = "SEND_EVENT"

#: A response the plan deliberately does not claim: an external actuator, a
#: physical effect, or anything whose realisation is not a discrete send.
EXTERNAL_OR_UNSUPPORTED = "EXTERNAL_OR_UNSUPPORTED"

EFFECT_KINDS = (SEND_EVENT, EXTERNAL_OR_UNSUPPORTED)

#: Audit only: record what the model does, change no verdict.
LEGACY_AUDIT = "LEGACY_AUDIT"
#: Require the full chain for every planned response action.
ENFORCE_V1 = "ENFORCE_V1"
#: Do not analyse at all.
OFF = "OFF"

PROFILES = (OFF, LEGACY_AUDIT, ENFORCE_V1)

PROFILE_VERSION = "1.0"


@dataclass(frozen=True)
class PlannedActionEffect:
    """One requirement's response action, bound to identities rather than names.

    Every field is an exact element identity. The audit resolves each one against
    the committed model; nothing is matched by substring, and a response owned by
    another part never satisfies this requirement.
    """
    requirement_id: str
    owner_def: str
    action_def: str
    usage_label: str
    effect_kind: str
    event_type: str
    sender_port: str
    consumer_owner_def: str
    consumer_behavior: str
    accept_transition: str
    target_state: str
    #: The producing side's own identity. Without it the check can only ask
    #: whether *some* state invokes the action, which an unrelated state in an
    #: unrelated machine would satisfy.
    owner_behavior: str = ""
    response_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "owner_def": self.owner_def,
            "owner_behavior": self.owner_behavior,
            "response_state": self.response_state,
            "action_def": self.action_def,
            "usage_label": self.usage_label,
            "effect": {
                "kind": self.effect_kind,
                "event_type": self.event_type,
                "sender_port": self.sender_port,
            },
            "consumer": {
                "owner_def": self.consumer_owner_def,
                "behavior": self.consumer_behavior,
                "accept_transition": self.accept_transition,
                "target_state": self.target_state,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedActionEffect":
        effect = dict(value.get("effect") or {})
        consumer = dict(value.get("consumer") or {})
        return cls(
            requirement_id=str(value.get("requirement_id") or ""),
            owner_def=str(value.get("owner_def") or ""),
            owner_behavior=str(value.get("owner_behavior") or ""),
            response_state=str(value.get("response_state") or ""),
            action_def=str(value.get("action_def") or ""),
            usage_label=str(value.get("usage_label") or ""),
            effect_kind=str(effect.get("kind") or ""),
            event_type=str(effect.get("event_type") or ""),
            sender_port=str(effect.get("sender_port") or ""),
            consumer_owner_def=str(consumer.get("owner_def") or ""),
            consumer_behavior=str(consumer.get("behavior") or ""),
            accept_transition=str(consumer.get("accept_transition") or ""),
            target_state=str(consumer.get("target_state") or ""),
        )

    def is_complete_identity(self) -> bool:
        """Whether every identity the audit needs is actually named.

        A plan entry missing any of these cannot fail closed usefully: the audit
        would report a chain broken at a step the plan never specified.
        """
        if self.effect_kind == EXTERNAL_OR_UNSUPPORTED:
            return bool(
                self.requirement_id and self.owner_def and self.action_def
            )
        return all((
            self.requirement_id, self.owner_def, self.owner_behavior,
            self.response_state, self.action_def,
            self.usage_label, self.event_type, self.sender_port,
            self.consumer_owner_def, self.consumer_behavior,
            self.accept_transition, self.target_state,
        )) and self.effect_kind == SEND_EVENT


def parse_action_effects(
    payload: Any,
) -> Tuple[PlannedActionEffect, ...]:
    """Read the ``action_effects`` array of a schema-10 generation plan."""
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return ()
    return tuple(
        PlannedActionEffect.from_dict(item)
        for item in payload
        if isinstance(item, Mapping)
    )


# ---------------------------------------------------------------------------
# Deriving a chain from the plans that already exist, and writing it into the
# committed model.  Both are deterministic: no LLM call, and nothing is invented
# that the A/G chain and the architecture plan do not already name.
# ---------------------------------------------------------------------------

_OWNER_SCOPE_RE = re.compile(r"\bpart\s+def\s+([A-Za-z_]\w*)\s*\{")


def signal_for_guarantee(guarantee: str) -> str:
    """`parachuteDeploymentCommand` -> `ParachuteDeploymentCommandSignal`.

    The convention the A/G chain library already follows: every in-chain trigger
    signal is its producing component's guarantee, capitalised, plus `Signal`.
    A trigger with no in-chain producer is an environment input — a sensor
    report, an operator command, power-on — and is deliberately not derived.
    """
    if not guarantee:
        return ""
    return guarantee[0].upper() + guarantee[1:] + "Signal"


def derive_action_effects(
    chain: Any,
    components: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]] = (),
) -> Tuple[PlannedActionEffect, ...]:
    """Join an A/G chain with the architecture plan into checkable effects.

    The chain names the producing component, its response action, and the signal
    each consumer accepts; the architecture plan names the ports. Neither alone
    is enough, and together they need no interpretation.
    """
    by_name = {str(item.get("name")): item for item in components}
    produced = {
        signal_for_guarantee(component.guarantee): component
        for component in chain.components
    }
    effects: list[PlannedActionEffect] = []
    for consumer in chain.components:
        signal = consumer.trigger_signal
        producer = produced.get(signal) if signal else None
        if producer is None or producer.owner_def == consumer.owner_def:
            continue                      # environment input, not a response
        port = _sender_port(
            producer.owner_def, consumer.owner_def, by_name, connections
        )
        action = producer.response_action
        effects.append(PlannedActionEffect(
            requirement_id=chain.source_requirement,
            owner_def=producer.owner_def,
            owner_behavior=producer.behavior,
            response_state=_response_state(chain, producer),
            action_def=action,
            usage_label="on" + action[0].upper() + action[1:],
            effect_kind=SEND_EVENT,
            event_type=signal,
            sender_port=port,
            consumer_owner_def=consumer.owner_def,
            consumer_behavior=consumer.behavior,
            accept_transition=f"accept{signal}",
            target_state=consumer.response_state,
        ))
    return tuple(effects)


def _response_state(chain: Any, producer: Any) -> str:
    """The state the response is actually emitted into.

    The arbitration behaviour takes its state name from the priority block, not
    from the component's own `response_state`; asking the emitter keeps one
    answer rather than two.
    """
    from .ag_emitter import arbitration_response_state

    if getattr(producer, "behavior", None) == "SafetyResponseArbitration":
        emitted = arbitration_response_state(chain)
        if emitted:
            return emitted
    return producer.response_state


def _sender_port(
    producer: str,
    consumer: str,
    by_name: Mapping[str, Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
) -> str:
    for link in connections:
        if (str(link.get("source_component")) == producer
                and str(link.get("target_component")) == consumer):
            return str(link.get("source_port") or "")
    inbound = {
        str(port.get("name"))
        for port in by_name.get(consumer, {}).get("ports", ())
        if str(port.get("direction")) in ("in", "inout")
    }
    for port in by_name.get(producer, {}).get("ports", ()):
        if (str(port.get("direction")) in ("out", "inout")
                and str(port.get("name")) in inbound):
            return str(port.get("name"))
    return ""


def _owner_span(model_text: str, owner_def: str) -> Optional[Tuple[int, int]]:
    for match in _OWNER_SCOPE_RE.finditer(model_text):
        if match.group(1) != owner_def:
            continue
        opening = model_text.find("{", match.start())
        closing = find_block_end(model_text, opening)
        if closing != -1:
            return opening, closing
    return None


def materialize_action_effects(
    model_text: str, effects: Sequence[PlannedActionEffect]
) -> Tuple[str, int]:
    """Give each planned response the body and the type edge the plan names.

    Confined to the owner's own block, and confined to definitions the plan
    already names: an action the plan does not carry is never touched, and a
    body that already holds the planned send is left alone so a second pass is
    byte-identical.
    """
    text = str(model_text or "")
    written = 0
    for effect in effects:
        if effect.effect_kind != SEND_EVENT or not effect.is_complete_identity():
            continue
        span = _owner_span(text, effect.owner_def)
        if span is None:
            continue
        opening, closing = span
        body = text[opening:closing]
        updated = _write_send_body(body, effect)
        updated = _write_typed_usage(updated, effect)
        if updated != body:
            text = text[:opening] + updated + text[closing:]
            written += 1
    return text, written


def _write_send_body(owner_body: str, effect: PlannedActionEffect) -> str:
    pattern = re.compile(
        rf"\baction\s+def\s+{re.escape(effect.action_def)}\s*\{{"
    )
    match = pattern.search(owner_body)
    if match is None:
        return owner_body
    opening = owner_body.find("{", match.start())
    closing = find_block_end(owner_body, opening)
    if closing == -1:
        return owner_body
    current = owner_body[opening + 1:closing]
    send = f"send {effect.event_type}() to {effect.sender_port};"
    if send in current:
        return owner_body                       # already materialised
    if current.strip():
        return owner_body                       # a body we did not write: leave it
    return (
        owner_body[:opening + 1]
        + f" {send} "
        + owner_body[closing:]
    )


def _write_typed_usage(owner_body: str, effect: PlannedActionEffect) -> str:
    typed = (
        f"entry action {effect.usage_label} : {effect.action_def};"
    )
    if typed in owner_body:
        return owner_body
    bare = re.compile(
        rf"\bentry\s+action\s+{re.escape(effect.action_def)}\s*;"
    )
    return bare.sub(typed, owner_body, count=1)
