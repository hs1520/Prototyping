"""Typed behavior obligations compiled losslessly from bounded A/G plans.

The A/G component spec already carries the owner, guarantee, trigger, response
state, and response action.  Generation must consume those facts even when the
optional ``realization_paths`` tuple is empty.  This module is the lifecycle
boundary between A/G planning and ordinary SysML behavior generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from .ag_emitter import AGChainSpec, AGComponentSpec


STATE_MACHINE = "STATE_MACHINE"
INVARIANT = "INVARIANT"
REALIZATION_KINDS = (STATE_MACHINE, INVARIANT)


@dataclass(frozen=True)
class TransitionObligation:
    source: str
    trigger: str
    target: str
    action: str
    guard: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "trigger": self.trigger,
            "target": self.target,
            "action": self.action,
            "guard": self.guard,
        }


@dataclass(frozen=True)
class BehaviorObligation:
    requirement_id: str
    contract_id: str
    owner_def: str
    owner_usage: str
    realization_kind: str
    stable_behavior_id: str
    assumptions: Tuple[str, ...]
    guarantees: Tuple[str, ...]
    initial_state: str | None = None
    transitions: Tuple[TransitionObligation, ...] = ()
    invariant_expression: str | None = None
    pattern: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "contract_id": self.contract_id,
            "owner_def": self.owner_def,
            "owner_usage": self.owner_usage,
            "realization_kind": self.realization_kind,
            "stable_behavior_id": self.stable_behavior_id,
            "assumptions": list(self.assumptions),
            "guarantees": list(self.guarantees),
            "initial_state": self.initial_state,
            "transitions": [item.to_dict() for item in self.transitions],
            "invariant_expression": self.invariant_expression,
            "pattern": self.pattern,
        }


@dataclass(frozen=True)
class BehaviorObligationPlan:
    obligations: Tuple[BehaviorObligation, ...]
    schema_version: str = "1.0"

    @property
    def status(self) -> str:
        if not self.obligations:
            return "INCOMPLETE"
        return "PASS" if all(_valid(item) for item in self.obligations) else "INVALID"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_role": "A_G_BEHAVIOR_OBLIGATION_PLAN",
            "status": self.status,
            "obligations": [item.to_dict() for item in self.obligations],
        }

    def render_for_prompt(self) -> str:
        lines = [
            "TYPED A/G BEHAVIOR OBLIGATIONS (mandatory; use exact stable IDs):"
        ]
        for item in self.obligations:
            lines.append(
                f"- {item.contract_id} -> {item.owner_def} "
                f"[{item.realization_kind}]"
            )
            if item.realization_kind == STATE_MACHINE:
                lines.append(
                    f"  state def {item.stable_behavior_id}; "
                    f"initial={item.initial_state}"
                )
                for transition in item.transitions:
                    guard = (
                        f"; guard={transition.guard}"
                        if transition.guard else ""
                    )
                    lines.append(
                        f"  {transition.source} --{transition.trigger}--> "
                        f"{transition.target}; entry action="
                        f"{transition.action}{guard}"
                    )
            else:
                lines.append(
                    f"  assert constraint {item.stable_behavior_id} "
                    f"{{ {item.invariant_expression} }}"
                )
            lines.append(
                "  establishes: " + ", ".join(item.guarantees)
            )
        return "\n".join(lines)


def _valid(item: BehaviorObligation) -> bool:
    if (
        not item.requirement_id
        or not item.contract_id
        or not item.owner_def
        or not item.owner_usage
        or not item.stable_behavior_id
        or not item.guarantees
        or item.realization_kind not in REALIZATION_KINDS
    ):
        return False
    if item.realization_kind == STATE_MACHINE:
        return bool(item.initial_state and item.transitions)
    return bool(item.invariant_expression)


def _invariant_expression(component: AGComponentSpec) -> str:
    guarantee = " and ".join(component.guarantees)
    assumptions = " and ".join(
        item.concept for item in component.assumptions
    )
    if not assumptions:
        return guarantee
    return f"not ({assumptions}) or ({guarantee})"


def _compile_component(
    chain: AGChainSpec, component: AGComponentSpec
) -> BehaviorObligation:
    paths = tuple(component.realization_paths)
    invariant = (
        component.trigger_signal is None
        and (
            not paths
            or all(
                path.trigger is None and path.source == path.target
                for path in paths
            )
        )
    )
    assumptions = tuple(item.concept for item in component.assumptions)
    if invariant:
        return BehaviorObligation(
            requirement_id=chain.source_requirement,
            contract_id=component.name,
            owner_def=component.owner_def,
            owner_usage=component.owner_usage,
            realization_kind=INVARIANT,
            stable_behavior_id=component.behavior,
            assumptions=assumptions,
            guarantees=component.guarantees,
            invariant_expression=_invariant_expression(component),
            pattern=chain.pattern,
        )

    transitions = (
        tuple(
            TransitionObligation(
                source=path.source,
                trigger=str(path.trigger or "continuous"),
                target=path.target,
                action=path.action,
                guard=path.guard,
            )
            for path in paths
        )
        if paths else (
            TransitionObligation(
                source=component.initial_state,
                trigger=str(component.trigger_signal or "continuous"),
                target=component.response_state,
                action=component.response_action,
            ),
        )
    )
    return BehaviorObligation(
        requirement_id=chain.source_requirement,
        contract_id=component.name,
        owner_def=component.owner_def,
        owner_usage=component.owner_usage,
        realization_kind=STATE_MACHINE,
        stable_behavior_id=component.behavior,
        assumptions=assumptions,
        guarantees=component.guarantees,
        initial_state=component.initial_state,
        transitions=transitions,
        pattern=chain.pattern,
    )


def compile_behavior_obligation_plan(
    specs: Iterable[AGChainSpec],
) -> BehaviorObligationPlan:
    return BehaviorObligationPlan(tuple(
        _compile_component(chain, component)
        for chain in specs
        for component in chain.components
    ))
