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

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "owner_def": self.owner_def,
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
            self.requirement_id, self.owner_def, self.action_def,
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
