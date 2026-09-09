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
    control_actions_fired: tuple[str, ...]
    description: str


def evaluate_safety_precedence(
    *,
    fired_action_definitions: Iterable[str],
    winner_action_definition: str,
    competing_action_definitions: Iterable[str],
    control_fired_action_definitions: Iterable[str] | None = None,
) -> SafetyPrecedenceEvidence:
    """Precedence needs a control run; silence alone proves nothing.

    "No competing response fired" is evidence only if those responses would have
    fired, so the control run - the same hazard variables with the winning condition
    withheld - is what separates an inert arbiter from a correctly arbitrating one;
    ``control_fired_action_definitions`` is what fired in it. The competing set comes
    from the arbiter's full action table, which also contains the winning action, so
    the winner is excluded: leaving it in scored a successful parachute deployment
    as a competing response.
    """
    fired = tuple(fired_action_definitions)
    competitors = tuple(
        action for action in competing_action_definitions
        if action != winner_action_definition
    )
    winner_fired = winner_action_definition in fired
    competing_fired = tuple(action for action in competitors if action in fired)
    control = tuple(control_fired_action_definitions or ())
    control_competitors = tuple(action for action in competitors if action in control)

    if competing_fired or (competitors and not winner_fired):
        # A competitor that fired violates precedence with or without a control, and so
        # does a winner that never fired. Only the negative conclusion, "nothing
        # competed", needs the control.
        status = "failed"
        why = (
            f"competing responses {list(competing_fired)} fired alongside the "
            "winner" if competing_fired else
            "the winning response did not fire at all"
        )
    elif not competitors:
        status = "inconclusive"
        why = "no competing safety response is declared, so there is nothing to take precedence over"
    elif control_fired_action_definitions is None:
        status = "inconclusive"
        why = (
            "no control run was made, so a suppressed competitor cannot be told "
            "apart from one that was never going to fire"
        )
    elif not control_competitors:
        status = "inconclusive"
        why = (
            "the control run fired no competing response either, so this "
            "scenario does not put the winner in competition with anything"
        )
    else:
        status = "verified"
        why = (
            f"the control run fired {list(control_competitors)} from the same "
            "hazard state with nothing withheld but the winning condition; "
            "with it present those responses were suppressed and only the "
            "winner fired"
        )
    return SafetyPrecedenceEvidence(
        status=status,
        winner_fired=winner_fired,
        competing_actions_fired=competing_fired,
        declared_competing_actions=competitors,
        control_actions_fired=control_competitors,
        description=(
            f"winner {winner_action_definition} fired={winner_fired}; declared "
            f"competing responses={list(competitors)}; competing responses "
            f"fired={list(competing_fired)}; control run fired="
            f"{list(control_competitors)} — {why}"
        ),
    )
