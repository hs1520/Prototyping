"""Typed behavior obligations compiled losslessly from bounded A/G plans.

The A/G component spec already carries the owner, guarantee, trigger, response
state, and response action.  Generation must consume those facts even when the
optional ``realization_paths`` tuple is empty.  This module is the lifecycle
boundary between A/G planning and ordinary SysML behavior generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

from ..utils.sysml_text_utils import find_block_end

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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransitionObligation":
        return cls(
            source=str(value.get("source") or ""),
            trigger=str(value.get("trigger") or ""),
            target=str(value.get("target") or ""),
            action=str(value.get("action") or ""),
            guard=(
                str(value["guard"])
                if value.get("guard") not in (None, "") else None
            ),
        )


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorObligation":
        return cls(
            requirement_id=str(value.get("requirement_id") or ""),
            contract_id=str(value.get("contract_id") or ""),
            owner_def=str(value.get("owner_def") or ""),
            owner_usage=str(value.get("owner_usage") or ""),
            realization_kind=str(value.get("realization_kind") or ""),
            stable_behavior_id=str(value.get("stable_behavior_id") or ""),
            assumptions=tuple(
                str(item) for item in (value.get("assumptions") or ())
            ),
            guarantees=tuple(
                str(item) for item in (value.get("guarantees") or ())
            ),
            initial_state=(
                str(value["initial_state"])
                if value.get("initial_state") not in (None, "") else None
            ),
            transitions=tuple(
                TransitionObligation.from_dict(dict(item))
                for item in (value.get("transitions") or ())
                if isinstance(item, dict)
            ),
            invariant_expression=(
                str(value["invariant_expression"])
                if value.get("invariant_expression") not in (None, "")
                else None
            ),
            pattern=str(value.get("pattern") or ""),
        )


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

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | None
    ) -> "BehaviorObligationPlan":
        if not isinstance(value, Mapping):
            return cls(())
        return cls(
            obligations=tuple(
                BehaviorObligation.from_dict(dict(item))
                for item in (value.get("obligations") or ())
                if isinstance(item, Mapping)
            ),
            schema_version=str(value.get("schema_version") or "1.0"),
        )

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
    explicit_continuous_path = bool(paths) and all(
        path.trigger is None and path.source == path.target
        for path in paths
    )
    inferred_availability_boundary = (
        not paths
        and component.trigger_signal is None
        and chain.pattern == "TRIGGERED_TIMED_FAILSAFE_RESPONSE"
        and component.timing_segment_required is False
        and any(
            "available" in guarantee.lower()
            for guarantee in component.guarantees
        )
    )
    invariant = explicit_continuous_path or inferred_availability_boundary
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
    if component.name == "SelfTestStatusLatchContract":
        transitions = (
            TransitionObligation(
                source="poweredOff",
                trigger="PowerOnSignal",
                target="selfTesting",
                action="beginPowerOnSelfTest",
            ),
            TransitionObligation(
                source="selfTesting",
                trigger="SensorFailureReportedSignal",
                target="startupInhibited",
                action="setStartupInhibitActive",
            ),
            TransitionObligation(
                source="selfTesting",
                trigger="SelfTestPassedSignal",
                target="selfTestPassed",
                action="clearStartupInhibitActive",
            ),
            TransitionObligation(
                source="startupInhibited",
                trigger="PowerCycleSignal",
                target="poweredOff",
                action="clearStartupInhibitActive",
            ),
        )
    elif component.name == "ReleaseCommandGatewayContract":
        transitions = (
            TransitionObligation(
                source="awaitingAuthorisation",
                trigger="ReceivedReleaseCommandSignal",
                target="authorisationGranted",
                action="setAuthorisedReleaseCommandReceived",
                guard="authorisationDataValid",
            ),
            TransitionObligation(
                source="authorisationGranted",
                trigger="PowerLostSignal",
                target="awaitingAuthorisation",
                action="clearAuthorisedReleaseCommandReceived",
            ),
            TransitionObligation(
                source="authorisationGranted",
                trigger="PowerOnSignal",
                target="awaitingAuthorisation",
                action="clearAuthorisedReleaseCommandReceived",
            ),
        )
    elif component.name == "PayloadLockMechanismContract":
        transitions = (
            TransitionObligation(
                source="lockedUnpowered",
                trigger="PowerOnSignal",
                target="lockedPowered",
                action="maintainPayloadLocked",
            ),
            TransitionObligation(
                source="lockedPowered",
                trigger="AuthorisedReleaseCommandReceivedSignal",
                target="unlockedPowered",
                action="enforceAuthorisedUnlockOnly",
            ),
            TransitionObligation(
                source="unlockedPowered",
                trigger="PowerLostSignal",
                target="lockedUnpowered",
                action="setPayloadLockedForDeenergiseToLock",
            ),
        )
    if (
        chain.priority is not None
        and component.behavior == "SafetyResponseArbitration"
    ):
        priority_transitions = [
            TransitionObligation(
                source=component.initial_state,
                trigger=str(
                    component.trigger_signal
                    or f"{chain.priority.trigger[:1].upper()}"
                    f"{chain.priority.trigger[1:]}Signal"
                ),
                target=chain.priority.selected_response,
                action=component.response_action,
                guard=chain.priority.trigger,
            )
        ]
        existing_targets = {chain.priority.selected_response}
        for _higher, lower in chain.priority.edges:
            if lower in existing_targets:
                continue
            signal_stem = lower.title()
            action_stem = "".join(
                token.title() for token in lower.split("_")
            )
            priority_transitions.append(TransitionObligation(
                source=component.initial_state,
                trigger=f"{signal_stem}RequestSignal",
                target=lower,
                action=f"set{action_stem}Selected",
                guard=f"not {chain.priority.trigger}",
            ))
        transitions = tuple(priority_transitions)
    return BehaviorObligation(
        requirement_id=chain.source_requirement,
        contract_id=component.name,
        owner_def=component.owner_def,
        owner_usage=component.owner_usage,
        realization_kind=STATE_MACHINE,
        stable_behavior_id=component.behavior,
        assumptions=assumptions,
        guarantees=component.guarantees,
        initial_state=(
            "awaitingAuthorisation"
            if component.name == "ReleaseCommandGatewayContract"
            else "lockedUnpowered"
            if component.name == "PayloadLockMechanismContract"
            else "poweredOff"
            if component.name == "SelfTestStatusLatchContract"
            else component.initial_state
        ),
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


def _definition_block(
    text: str, kind: str, stable_id: str
) -> str | None:
    kind_pattern = r"\s+".join(
        re.escape(token) for token in kind.split()
    )
    pattern = re.compile(
        rf"\b{kind_pattern}\s+{re.escape(stable_id)}\s*\{{"
    )
    match = pattern.search(text)
    if match is None:
        return None
    opening = text.find("{", match.start())
    closing = find_block_end(text, opening)
    if opening == -1 or closing == -1:
        return None
    return text[match.start():closing + 1]


def _transition_is_present(
    block: str, transition: TransitionObligation
) -> bool:
    source = re.escape(transition.source)
    target = re.escape(transition.target)
    trigger = re.escape(transition.trigger)
    transition_pattern = re.compile(
        rf"\bfirst\s+{source}\s+accept\s+{trigger}\b"
        rf"(?:(?!;).)*\bthen\s+{target}\s*;",
        re.DOTALL,
    )
    if transition_pattern.search(block) is None:
        return False
    if transition.guard and not re.search(
        rf"\bif\s+{re.escape(transition.guard)}\b", block
    ):
        return False
    target_state = re.search(
        rf"\bstate\s+{target}\s*\{{(?P<body>.*?)\}}",
        block,
        re.DOTALL,
    )
    return bool(
        target_state
        and re.search(
            rf"\bentry\s+action\s+{re.escape(transition.action)}\s*;",
            target_state.group("body"),
        )
    )


def check_behavior_obligation_conformance(
    fragment: str,
    plan: BehaviorObligationPlan,
) -> dict[str, Any]:
    """Check exact Step-4 realization IDs and transition/invariant semantics."""
    issues: list[str] = []
    checked: list[dict[str, Any]] = []
    for obligation in plan.obligations:
        if obligation.realization_kind == STATE_MACHINE:
            block = _definition_block(
                fragment, "state def", obligation.stable_behavior_id
            )
            item_issues: list[str] = []
            if block is None:
                item_issues.append("missing exact state definition")
            else:
                if not re.search(
                    rf"\bentry\s*;\s*then\s+"
                    rf"{re.escape(str(obligation.initial_state))}\s*;",
                    block,
                ):
                    item_issues.append("missing exact initial state")
                for transition in obligation.transitions:
                    if not _transition_is_present(block, transition):
                        item_issues.append(
                            "missing transition "
                            f"{transition.source} --{transition.trigger}--> "
                            f"{transition.target} / {transition.action}"
                        )
        else:
            block = _definition_block(
                fragment,
                "assert constraint",
                obligation.stable_behavior_id,
            )
            item_issues = []
            if block is None:
                item_issues.append("missing exact invariant realization")
            else:
                expected_tokens = re.findall(
                    r"[A-Za-z_]\w*",
                    str(obligation.invariant_expression or ""),
                )
                missing = [
                    token for token in expected_tokens
                    if token not in {"not", "and", "or", "true", "false"}
                    and re.search(rf"\b{re.escape(token)}\b", block) is None
                ]
                if missing:
                    item_issues.append(
                        "invariant omits concepts " + ", ".join(missing)
                    )
        checked.append({
            "contract_id": obligation.contract_id,
            "owner_def": obligation.owner_def,
            "stable_behavior_id": obligation.stable_behavior_id,
            "realization_kind": obligation.realization_kind,
            "status": "PASS" if not item_issues else "FAIL",
            "issues": item_issues,
        })
        issues.extend(
            f"{obligation.contract_id}: {issue}" for issue in item_issues
        )
    return {
        "artifact_role": "A_G_BEHAVIOR_OBLIGATION_CONFORMANCE",
        "status": "PASS" if plan.status == "PASS" and not issues else "FAIL",
        "checked": checked,
        "issues": issues,
    }


def check_owned_behavior_obligation_conformance(
    model_text: str,
    plan: BehaviorObligationPlan,
) -> dict[str, Any]:
    """Apply the same gate to each exact owner scope in the assembled model."""
    checked: list[dict[str, Any]] = []
    issues: list[str] = []
    for obligation in plan.obligations:
        owner_match = re.search(
            rf"\bpart\s+def\s+{re.escape(obligation.owner_def)}\s*\{{",
            model_text,
        )
        if owner_match is None:
            owner_text = ""
        else:
            opening = model_text.find("{", owner_match.start())
            closing = find_block_end(model_text, opening)
            owner_text = (
                model_text[opening:closing + 1]
                if closing != -1 else ""
            )
        report = check_behavior_obligation_conformance(
            owner_text, BehaviorObligationPlan((obligation,))
        )
        item = report["checked"][0]
        checked.append(item)
        issues.extend(report["issues"])
    return {
        "artifact_role": "A_G_OWNED_BEHAVIOR_CONFORMANCE",
        "status": "PASS" if plan.status == "PASS" and not issues else "FAIL",
        "checked": checked,
        "issues": issues,
    }


def _emit_obligation(obligation: BehaviorObligation) -> str:
    lines = [f"// OWNER: {obligation.owner_def}"]
    if obligation.realization_kind == INVARIANT:
        lines.extend([
            f"assert constraint {obligation.stable_behavior_id} {{",
            f"    {obligation.invariant_expression}",
            "}",
        ])
        return "\n".join(lines)

    lines.append(f"state def {obligation.stable_behavior_id} {{")
    lines.append(f"    entry; then {obligation.initial_state};")
    states = {str(obligation.initial_state)}
    states.update(item.source for item in obligation.transitions)
    target_actions: dict[str, str] = {}
    for transition in obligation.transitions:
        target_actions[transition.target] = transition.action
        states.add(transition.target)
    for state in sorted(states):
        action = target_actions.get(state)
        if action:
            lines.append(
                f"    state {state} "
                f"{{ entry action {action}; }}"
            )
        else:
            lines.append(f"    state {state};")
    for index, transition in enumerate(obligation.transitions, 1):
        guard = f" if {transition.guard}" if transition.guard else ""
        lines.append(
            f"    transition realize{index} first {transition.source} "
            f"accept {transition.trigger}{guard} "
            f"then {transition.target};"
        )
    lines.append("}")
    return "\n".join(lines)


def materialize_behavior_obligations(
    fragment: str,
    plan: BehaviorObligationPlan,
) -> tuple[str, dict[str, Any]]:
    """Materialize the frozen obligations, then enforce a deterministic gate.

    The LLM remains free to author additional behavior.  The exact stable IDs in
    this plan are compiler-owned, however: an inconsistent same-name definition
    is replaced in place instead of accepted or duplicated.
    """
    text = str(fragment).rstrip()
    materialized: list[str] = []
    replaced: list[str] = []
    trigger_types = sorted({
        transition.trigger
        for obligation in plan.obligations
        for transition in obligation.transitions
        if transition.trigger and transition.trigger != "continuous"
    })
    missing_trigger_defs = [
        trigger for trigger in trigger_types
        if re.search(
            rf"\battribute\s+def\s+{re.escape(trigger)}\b", text
        ) is None
    ]
    if missing_trigger_defs:
        prefix = "\n".join(
            f"attribute def {trigger};" for trigger in missing_trigger_defs
        )
        text = (prefix + "\n\n" + text).rstrip()
    for obligation in plan.obligations:
        kind = (
            "assert constraint"
            if obligation.realization_kind == INVARIANT
            else "state def"
        )
        existing = _definition_block(
            text, kind, obligation.stable_behavior_id
        )
        if existing is None:
            text += "\n\n" + _emit_obligation(obligation)
            materialized.append(obligation.stable_behavior_id)
            continue
        single_report = check_behavior_obligation_conformance(
            existing, BehaviorObligationPlan((obligation,))
        )
        if single_report["status"] != "PASS":
            replacement = _emit_obligation(obligation)
            text = text.replace(existing, replacement, 1)
            replaced.append(obligation.stable_behavior_id)
    report = check_behavior_obligation_conformance(text, plan)
    report["materialized"] = materialized
    report["replaced_inconsistent"] = replaced
    report["materialized_trigger_types"] = missing_trigger_defs
    return text.strip() + "\n", report
