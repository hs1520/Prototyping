"""Typed, checkable effects a planned response action must have.

The prompts define a behavioural implementation as a reachable state whose
entry action names a response; the action body may stay empty, and 152 of 155
archived pilot definitions are. ``ACTION_EFFECTS_V1`` closes that gap: a
planned response action sends one declared event out of one declared port,
and that event reaches a consumer that accepts it. Numeric computation,
continuous control and physical effect stay with constraints, SITL and Gazebo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


SEND_EVENT = "SEND_EVENT"

# A response the plan does not claim: an external actuator, a physical
# effect, or anything whose realisation is not a discrete send.
EXTERNAL_OR_UNSUPPORTED = "EXTERNAL_OR_UNSUPPORTED"

LEGACY_AUDIT = "LEGACY_AUDIT"
ENFORCE_V1 = "ENFORCE_V1"
OFF = "OFF"

PROFILES = (OFF, LEGACY_AUDIT, ENFORCE_V1)

PROFILE_VERSION = "1.0"


@dataclass(frozen=True)
class PlannedActionEffect:
    """One requirement's response action, bound to element identities.

    Each field is resolved against the committed model, not matched by substring,
    so a response owned by another part does not satisfy this requirement.
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
    # The producing side's own identity. Without it the check only asks whether
    # some state invokes the action, which an unrelated machine satisfies.
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
        """Whether every identity the audit needs is named.

        A plan entry missing one cannot fail closed usefully: the audit would report
        a chain broken at a step the plan never specified.
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
