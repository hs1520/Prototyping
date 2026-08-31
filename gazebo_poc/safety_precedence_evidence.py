"""Requirement Evidence for safety-response precedence under simultaneous hazards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SafetyPrecedenceEvidence:
    status: str
    winner_fired: bool
    competing_actions_fired: tuple[str, ...]
    declared_competing_actions: tuple[str, ...]
    description: str


def evaluate_safety_precedence(
    *,
    fired_action_definitions: Iterable[str],
    winner_action_definition: str,
    competing_action_definitions: Iterable[str],
) -> SafetyPrecedenceEvidence:
    fired = tuple(fired_action_definitions)
    competitors = tuple(competing_action_definitions)
    winner_fired = winner_action_definition in fired
    competing_fired = tuple(action for action in competitors if action in fired)
    if not competitors:
        status = "inconclusive"
    elif winner_fired and not competing_fired:
        status = "verified"
    else:
        status = "failed"
    return SafetyPrecedenceEvidence(
        status=status,
        winner_fired=winner_fired,
        competing_actions_fired=competing_fired,
        declared_competing_actions=competitors,
        description=(
            f"winner {winner_action_definition} fired={winner_fired}; declared "
            f"competing responses={list(competitors)}; competing responses "
            f"fired={list(competing_fired)}"
        ),
    )
