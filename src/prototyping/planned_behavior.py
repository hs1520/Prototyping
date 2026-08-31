"""Plan-first SysML v2 behavior identities and deterministic serialization.

This module deliberately models only the behavior slice needed by a
``STATE_ACTIVE`` constraint: one owning part definition, one state definition,
its states, and the transitions that make the activation state reachable.  It
is not a general state-machine DSL.  The committed SysML remains the semantic
authority, while this IR prevents an LLM serialization step from silently
renaming plan-owned members.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ..utils.req_id import normalise_req_id, source_requirements_by_id
from ..utils.digest import sha256_text
from ..utils.sysml_text_utils import IDENTIFIER_RE, find_block_end, named_block_span
from .event_symbols import (
    PlannedEventSymbol,
    collect_planned_event_symbols,
    materialize_planned_event_symbols,
)


_TRIGGER_KINDS = {"ACCEPT", "GUARD"}
_STATE_ROLES = {"INITIAL", "NORMAL", "RESPONSE", "FAULT"}


@dataclass(frozen=True)
class PlannedState:
    state_id: str
    role: str = "NORMAL"
    entry_action: str | None = None
    do_action: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedState":
        action = value.get("entry_action")
        do_action = value.get("do_action")
        return cls(
            state_id=str(
                value.get("state_id") or value.get("id") or ""
            ).strip(),
            role=str(value.get("role") or "NORMAL").strip().upper(),
            entry_action=(
                str(action).strip() if action not in (None, "") else None
            ),
            do_action=(
                str(do_action).strip()
                if do_action not in (None, "") else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedTransition:
    transition_id: str
    source: str
    target: str
    trigger_kind: str
    trigger: str
    #: A guard that composes WITH an accept trigger, rather than replacing it.
    #: An inhibition is exactly this shape: the event still arrives, and the
    #: transition must not fire while the inhibiting condition holds. With only
    #: the either/or trigger_kind, "release on arrival" and "release on arrival
    #: unless aborted" are the same plan — which is how REQ_SAFE_006 came out
    #: as an unguarded transition that separated the payload during an abort.
    guard: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedTransition":
        return cls(
            transition_id=str(
                value.get("transition_id") or value.get("id") or ""
            ).strip(),
            source=str(value.get("source") or "").strip(),
            target=str(value.get("target") or "").strip(),
            trigger_kind=str(
                value.get("trigger_kind") or "ACCEPT"
            ).strip().upper(),
            trigger=str(value.get("trigger") or "").strip(),
            guard=str(value.get("guard") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedBehavior:
    owner: str
    behavior_id: str
    initial_state: str
    states: tuple[PlannedState, ...]
    transitions: tuple[PlannedTransition, ...]
    provenance: str = "FROZEN_REQUIREMENT"
    source_requirement_id: str | None = None
    source_digest: str | None = None

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        requirements: Sequence[str] = (),
    ) -> "PlannedBehavior":
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            provenance = {}
        req_id = str(
            provenance.get("requirement_id")
            or value.get("source_requirement_id")
            or ""
        ).strip()
        req_id = normalise_req_id(req_id) if req_id else None
        source = source_requirements_by_id(requirements).get(req_id or "")
        archived_digest = (
            provenance.get("source_digest")
            or value.get("source_digest")
        )
        return cls(
            owner=str(value.get("owner") or "").strip(),
            behavior_id=str(
                value.get("behavior_id") or value.get("name") or ""
            ).strip(),
            initial_state=str(value.get("initial_state") or "").strip(),
            states=tuple(
                PlannedState.from_dict(item)
                for item in (value.get("states") or ())
                if isinstance(item, Mapping)
            ),
            transitions=tuple(
                PlannedTransition.from_dict(item)
                for item in (value.get("transitions") or ())
                if isinstance(item, Mapping)
            ),
            provenance=str(
                provenance.get("kind")
                or value.get("provenance_kind")
                or "FROZEN_REQUIREMENT"
            ).strip().upper(),
            source_requirement_id=req_id,
            source_digest=(
                sha256_text(source)
                if source else (
                    str(archived_digest).strip()
                    if archived_digest not in (None, "") else None
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "behavior_id": self.behavior_id,
            "initial_state": self.initial_state,
            "states": [item.to_dict() for item in self.states],
            "transitions": [item.to_dict() for item in self.transitions],
            "provenance": {
                "kind": self.provenance,
                "requirement_id": self.source_requirement_id,
                "source_digest": self.source_digest,
            },
        }


def _plan_inhibition_issues(
    behaviors: Sequence[PlannedBehavior],
    sources: Mapping[str, str],
) -> list[str]:
    """An inhibition requirement must reach the plan as a guard — wherever
    the departure actually lives.

    REQ_SAFE_006 — "maintain the payload in the mechanically locked state
    whenever a delivery-abort condition is active, regardless of geographic
    proximity" — was planned as an unguarded ``Locked --accept
    DeliveryCoordinateSatisfied--> Releasing``. Every structural check passed
    and the generated vehicle separated its payload while an abort was
    active, reproduced independently at three tiers.

    An earlier version of this gate was scoped per-behaviour against that
    behaviour's OWN source requirement. Measured against a real plan it fired
    zero times: the inhibition requirement's behaviour was a transitionless
    corner machine ("DefaultToMechanicallyLockedStateBehavior"), while the
    departure itself (``Initial --accept DeliveryWaypointReached-->
    ReleasingPayload``) lived one behaviour over, traced to a different
    requirement. Surgical scoping was the defect, so the gate is now anchored
    on the whole plan:

    - the held state, wherever it is planned, may not be left without a guard
      naming the condition;
    - the departure itself (the safe state's exit vocabulary applied to the
      held object), wherever it is planned, needs the same guard at its
      boundary;
    - a plan that expresses neither cannot express the inhibition at all,
      which is a plan defect — not a silent skip;
    - "shall not transition to X while C" guards every boundary entry into X.

    Checks read the requirement's parsed intent rather than any spelling, so
    a model that renames its states still conforms.
    """
    from .verification_obligations import (
        RequirementIntentKind,
        held_state_exit_terms,
        parse_requirement_intent,
        semantic_terms,
    )

    issues: list[str] = []
    for req_id, requirement_text in sources.items():
        intent = parse_requirement_intent(requirement_text)
        if (
            intent.kind is not RequirementIntentKind.INHIBITION
            or not intent.condition_terms
        ):
            continue
        condition_label = "/".join(sorted(intent.condition_terms))

        def _guard_names_condition(transition: PlannedTransition) -> bool:
            guard = transition.guard or (
                transition.trigger
                if transition.trigger_kind == "GUARD" else ""
            )
            return bool(semantic_terms(guard) & intent.condition_terms)

        if intent.forbidden_state_terms:
            # Negative shape: every boundary entry into the forbidden state
            # needs the guard; an interior move is decided at its boundary.
            for index, behavior in enumerate(behaviors):
                for transition in behavior.transitions:
                    if not (
                        semantic_terms(transition.target)
                        & intent.forbidden_state_terms
                    ):
                        continue
                    if (
                        semantic_terms(transition.source)
                        & intent.forbidden_state_terms
                    ):
                        continue
                    if _guard_names_condition(transition):
                        continue
                    issues.append(
                        f"behaviors[{index}] transition "
                        f"{transition.transition_id} enters "
                        f"{transition.target} unconditionally, but {req_id} "
                        f"forbids that state whenever {condition_label} is "
                        "active; the transition needs a guard naming that "
                        "condition"
                    )
            continue

        if not intent.required_state_terms:
            continue

        flagged: set[tuple[int, str]] = set()
        held_anchor = False
        for index, behavior in enumerate(behaviors):
            held_states = {
                state.state_id for state in behavior.states
                if semantic_terms(state.state_id)
                & intent.required_state_terms
            }
            if held_states:
                held_anchor = True
            for transition in behavior.transitions:
                if transition.source not in held_states:
                    continue
                if _guard_names_condition(transition):
                    continue
                flagged.add((index, transition.transition_id))
                issues.append(
                    f"behaviors[{index}] transition "
                    f"{transition.transition_id} leaves "
                    f"{transition.source} unconditionally, but {req_id} "
                    f"requires that state to be held whenever "
                    f"{condition_label} is active; the transition needs a "
                    "guard naming that condition"
                )

        exit_terms = held_state_exit_terms(intent.required_state_terms)
        exit_anchor = False
        if exit_terms and intent.held_object_terms:
            for index, behavior in enumerate(behaviors):
                scope_terms = (
                    semantic_terms(behavior.owner)
                    | semantic_terms(behavior.behavior_id)
                )
                action_vocabulary = {
                    state.state_id: " ".join(
                        item
                        for item in (state.entry_action, state.do_action)
                        if item
                    )
                    for state in behavior.states
                }
                for transition in behavior.transitions:
                    target_terms = semantic_terms(
                        transition.target
                    ) | semantic_terms(
                        action_vocabulary.get(transition.target, "")
                    )
                    if not (target_terms & exit_terms):
                        continue
                    if not (
                        (target_terms | scope_terms)
                        & intent.held_object_terms
                    ):
                        continue
                    # Only the source STATE NAME evidences a prior departure
                    # (ReleasingPayload -> Released is interior). Its actions
                    # do not: run 2026-08-31 planned `Initial` with
                    # entry_action initializeRelease — preparation, not
                    # departure — and counting it hid the one boundary the
                    # guard belongs on.
                    if semantic_terms(transition.source) & exit_terms:
                        continue
                    exit_anchor = True
                    if (index, transition.transition_id) in flagged:
                        continue
                    if _guard_names_condition(transition):
                        continue
                    issues.append(
                        f"behaviors[{index}] transition "
                        f"{transition.transition_id} "
                        f"({behavior.owner}::{behavior.behavior_id}) departs "
                        f"the "
                        f"{'/'.join(sorted(intent.required_state_terms))} "
                        f"state via {transition.target} unconditionally, but "
                        f"{req_id} requires it held whenever "
                        f"{condition_label} is active; the transition needs "
                        "a guard naming that condition"
                    )

        if not held_anchor and not exit_anchor:
            issues.append(
                f"{req_id} requires "
                f"{'/'.join(sorted(intent.required_state_terms))} to be held "
                f"whenever {condition_label} is active, but no planned state "
                "matches the held state and no planned transition names its "
                "departure; the plan cannot express this inhibition"
            )
    return issues


def validate_planned_behaviors(
    behaviors: Sequence[PlannedBehavior],
    *,
    component_names: set[str],
    requirements: Sequence[str],
    state_active_constraints: Sequence[Any],
    component_port_names: Mapping[str, set[str]] | None = None,
    require_executable_responses: bool = True,
) -> list[str]:
    """Validate referential integrity before any SysML text is generated."""
    issues: list[str] = []
    sources = source_requirements_by_id(requirements)
    seen_behaviors: set[tuple[str, str]] = set()
    globally_named: set[str] = set()
    ports_by_owner = component_port_names or {}
    planned_actions_by_owner: dict[str, set[str]] = {}
    for behavior in behaviors:
        for state in behavior.states:
            for action in (state.entry_action, state.do_action):
                if action:
                    planned_actions_by_owner.setdefault(
                        behavior.owner, set()
                    ).add(action)

    for index, behavior in enumerate(behaviors):
        prefix = f"behaviors[{index}]"
        key = (behavior.owner, behavior.behavior_id)
        if key in seen_behaviors:
            issues.append(
                f"duplicate planned behavior "
                f"{behavior.owner}::{behavior.behavior_id}"
            )
        seen_behaviors.add(key)
        if behavior.behavior_id in globally_named:
            issues.append(
                f"planned behavior id {behavior.behavior_id} must be unique"
            )
        globally_named.add(behavior.behavior_id)
        # A behaviour id that equals a planned action name in the SAME owner
        # materializes as a state def and an action def sharing one name in
        # one scope — the exact collision the terminal
        # USER_NAMESPACE_INTEGRITY gate rejects (measured on draw 4c39e7ba:
        # WaypointModificationBehavior declared as both). Reject it while the
        # plan is still repairable.
        if behavior.behavior_id in planned_actions_by_owner.get(
            behavior.owner, set()
        ):
            issues.append(
                f"{prefix}.behavior_id '{behavior.behavior_id}' is also a "
                f"planned action name in {behavior.owner}; a state def and "
                "an action def may not share one name in the owner scope — "
                "rename one of them"
            )
        if behavior.owner not in component_names:
            issues.append(f"{prefix}.owner is not a planned component")
        if not IDENTIFIER_RE.fullmatch(behavior.behavior_id):
            issues.append(f"{prefix}.behavior_id is not a SysML identifier")
        if not IDENTIFIER_RE.fullmatch(behavior.initial_state):
            issues.append(f"{prefix}.initial_state is not a SysML identifier")

        state_ids: list[str] = []
        for state_index, state in enumerate(behavior.states):
            state_prefix = f"{prefix}.states[{state_index}]"
            if not IDENTIFIER_RE.fullmatch(state.state_id):
                issues.append(
                    f"{state_prefix}.state_id is not a SysML identifier"
                )
            if state.role not in _STATE_ROLES:
                issues.append(f"{state_prefix}.role is unsupported")
            if (
                state.entry_action is not None
                and not IDENTIFIER_RE.fullmatch(state.entry_action)
            ):
                issues.append(
                    f"{state_prefix}.entry_action is not a SysML identifier"
                )
            if (
                state.do_action is not None
                and not IDENTIFIER_RE.fullmatch(state.do_action)
            ):
                issues.append(
                    f"{state_prefix}.do_action is not a SysML identifier"
                )
            if (
                require_executable_responses
                and
                state.role in {"RESPONSE", "FAULT"}
                and not (state.entry_action or state.do_action)
            ):
                issues.append(
                    f"{state_prefix} role {state.role} requires an "
                    "executable entry_action or do_action"
                )
            state_ids.append(state.state_id)
        if len(set(state_ids)) != len(state_ids):
            issues.append(f"{prefix} state ids must be unique")
        if behavior.initial_state not in set(state_ids):
            issues.append(
                f"{prefix}.initial_state is not a declared state"
            )
        initial_roles = [
            item.state_id
            for item in behavior.states
            if item.role == "INITIAL"
        ]
        if initial_roles != [behavior.initial_state]:
            issues.append(
                f"{prefix} must mark exactly {behavior.initial_state} "
                "with role INITIAL"
            )

        transition_ids: set[str] = set()
        edges: list[tuple[str, str]] = []
        for transition_index, transition in enumerate(behavior.transitions):
            transition_prefix = (
                f"{prefix}.transitions[{transition_index}]"
            )
            if not IDENTIFIER_RE.fullmatch(transition.transition_id):
                issues.append(
                    f"{transition_prefix}.transition_id is not a SysML "
                    "identifier"
                )
            if transition.transition_id in transition_ids:
                issues.append(
                    f"{prefix} transition ids must be unique"
                )
            transition_ids.add(transition.transition_id)
            if transition.source not in set(state_ids):
                issues.append(
                    f"{transition_prefix}.source is not a declared state"
                )
            if transition.target not in set(state_ids):
                issues.append(
                    f"{transition_prefix}.target is not a declared state"
                )
            if transition.trigger_kind not in _TRIGGER_KINDS:
                issues.append(
                    f"{transition_prefix}.trigger_kind is unsupported"
                )
            if (
                transition.trigger_kind == "ACCEPT"
                and not IDENTIFIER_RE.fullmatch(transition.trigger)
            ):
                issues.append(
                    f"{transition_prefix}.trigger must name a SysML event "
                    "definition"
                )
            if (
                transition.trigger_kind == "ACCEPT"
                and transition.trigger in ports_by_owner.get(
                    behavior.owner, set()
                )
            ):
                issues.append(
                    f"{transition_prefix}.trigger must name an event item "
                    "classifier, not an owner port usage"
                )
            if (
                transition.trigger_kind == "GUARD"
                and (
                    not transition.trigger
                    or any(token in transition.trigger for token in "{};")
                )
            ):
                issues.append(
                    f"{transition_prefix}.trigger is not a safe guard "
                    "expression"
                )
            if transition.guard and any(
                token in transition.guard for token in "{};"
            ):
                issues.append(
                    f"{transition_prefix}.guard is not a safe guard expression"
                )
            if transition.guard and transition.trigger_kind != "ACCEPT":
                issues.append(
                    f"{transition_prefix}.guard composes with an accept "
                    "trigger; a GUARD transition carries its condition in "
                    "trigger"
                )
            edges.append((transition.source, transition.target))

        reachable = {behavior.initial_state}
        changed = True
        while changed:
            changed = False
            for source, target in edges:
                if source in reachable and target not in reachable:
                    reachable.add(target)
                    changed = True
        for state in behavior.states:
            if (
                state.role in {"RESPONSE", "FAULT"}
                and state.state_id not in reachable
            ):
                issues.append(
                    f"{prefix} state {state.state_id} is unreachable from "
                    f"{behavior.initial_state}"
                )

        if behavior.provenance not in {
            "FROZEN_REQUIREMENT",
            "A_G_GUARANTEE",
            "DESIGN_DECISION",
        }:
            issues.append(f"{prefix}.provenance.kind is unsupported")
        if (
            behavior.provenance == "FROZEN_REQUIREMENT"
            and requirements
            and behavior.source_requirement_id not in sources
        ):
            issues.append(
                f"{prefix} has no matching frozen source requirement"
            )
    issues.extend(_plan_inhibition_issues(behaviors, sources))

    planned_refs = {
        (behavior.owner, f"{behavior.behavior_id}::{state.state_id}"): (
            behavior,
            state,
        )
        for behavior in behaviors
        for state in behavior.states
    }
    for constraint in state_active_constraints:
        key = (constraint.owner, constraint.activation_ref)
        if key not in planned_refs:
            issues.append(
                f"{constraint.owner}.{constraint.constraint_id} activation "
                f"{constraint.activation_ref} has no typed behavior/state "
                "declaration"
            )
            continue
        behavior, state = planned_refs[key]
        if (
            require_executable_responses
            and not (state.entry_action or state.do_action)
        ):
            issues.append(
                f"{constraint.owner}.{constraint.constraint_id} activation "
                f"{constraint.activation_ref} has no executable "
                "entry_action or do_action"
            )
        if (
            constraint.source_requirement_id
            and behavior.source_requirement_id
            != constraint.source_requirement_id
        ):
            issues.append(
                f"{constraint.owner}.{constraint.constraint_id} and "
                f"{behavior.behavior_id} do not share requirement provenance"
            )
    return issues


def _definition_spans(
    text: str,
    kind: str,
    name: str,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pattern = re.compile(
        rf"\b{kind}\s+{re.escape(name)}\s*(?P<tail>[;{{])"
    )
    for match in pattern.finditer(text):
        if match.group("tail") == ";":
            spans.append((match.start(), match.end()))
            continue
        opening = text.find("{", match.start(), match.end())
        closing = find_block_end(text, opening)
        if closing != -1:
            spans.append((match.start(), closing + 1))
    return spans


def _remove_definitions(text: str, kind: str, names: set[str]) -> str:
    spans = [
        span
        for name in names
        for span in _definition_spans(text, kind, name)
    ]
    result = text
    for start, end in sorted(spans, reverse=True):
        line_start = result.rfind("\n", 0, start) + 1
        owner_start = result.rfind("// OWNER:", 0, line_start)
        if owner_start != -1:
            between = result[owner_start:line_start]
            if "\n" not in between.rstrip("\n"):
                line_start = owner_start
        line_end = result.find("\n", end)
        if line_end == -1:
            line_end = end
        result = result[:line_start] + result[line_end + 1:]
    return result


def emit_planned_behavior(behavior: PlannedBehavior) -> str:
    """Serialize one validated plan using canonical SysML v2 state syntax."""
    lines = [
        f"// OWNER: {behavior.owner}",
        f"state def {behavior.behavior_id} {{",
        f"    entry; then {behavior.initial_state};",
        "",
    ]
    for state in behavior.states:
        if state.entry_action or state.do_action:
            lines.extend([
                f"    state {state.state_id} {{",
            ])
            # Named, typed usages (`entry action onX : act;`) — the bare
            # spelling (`entry action act;`) declares a NESTED member named
            # `act` that shadows the part-level `action def act`, and the
            # emitter + declaration injector then mass-produce shadow pairs
            # by construction (measured: 59 of 61 warnings across the three
            # failed 2026-08-30 draws). The archived QUALIFIED models use
            # the typed spelling throughout.
            if state.entry_action:
                lines.append(
                    f"        entry action on{state.state_id} : "
                    f"{state.entry_action};"
                )
            if state.do_action:
                lines.append(
                    f"        do action run{state.state_id} : "
                    f"{state.do_action};"
                )
            lines.append("    }")
        else:
            lines.append(f"    state {state.state_id};")
    if behavior.transitions:
        lines.append("")
    for transition in behavior.transitions:
        trigger = (
            f"accept {transition.trigger}"
            if transition.trigger_kind == "ACCEPT"
            else f"if {transition.trigger}"
        )
        lines.extend([
            f"    transition {transition.transition_id}",
            f"        first {transition.source}",
            f"        {trigger}",
        ])
        if transition.guard and transition.trigger_kind == "ACCEPT":
            lines.append(f"        if {transition.guard}")
        lines.append(f"        then {transition.target};")
    lines.append("}")
    return "\n".join(lines)


def _planned_declarations(
    behaviors: Sequence[PlannedBehavior],
) -> str:
    actions = sorted({
        action
        for behavior in behaviors
        for state in behavior.states
        for action in (state.entry_action, state.do_action)
        if action
    })
    blocks: list[str] = []
    for action in actions:
        owner = next(
            behavior.owner
            for behavior in behaviors
            for state in behavior.states
            if action in {state.entry_action, state.do_action}
        )
        blocks.append(
            f"// OWNER: {owner}\naction def {action} {{}}"
        )
    blocks.extend(emit_planned_behavior(item) for item in behaviors)
    return "\n\n".join(blocks)


def materialize_planned_behaviors(
    fragment: str,
    behaviors: Sequence[PlannedBehavior],
    *,
    event_symbols: Sequence[PlannedEventSymbol] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Make the typed plan the sole writer of its behavior definitions."""
    if not behaviors:
        return str(fragment), {
            "artifact_role": "PLANNED_BEHAVIOR_CONFORMANCE",
            "status": "NOT_APPLICABLE",
            "checked": [],
            "issues": [],
            "materialized": [],
        }
    behavior_names = {item.behavior_id for item in behaviors}
    action_names = {
        action
        for behavior in behaviors
        for state in behavior.states
        for action in (state.entry_action, state.do_action)
        if action
    }
    text = _remove_definitions(
        str(fragment), r"state\s+def", behavior_names
    )
    text = _remove_definitions(text, r"action\s+def", set(action_names))
    symbols = tuple(
        event_symbols
        if event_symbols is not None
        else collect_planned_event_symbols(behaviors)
    )
    text, event_report = materialize_planned_event_symbols(text, symbols)
    text = text.rstrip()
    if text:
        text += "\n\n"
    text += _planned_declarations(behaviors)
    report = check_planned_behavior_conformance(
        text, behaviors, owned=False
    )
    report["materialized"] = [
        f"{item.owner}::{item.behavior_id}" for item in behaviors
    ]
    report["event_symbol_conformance"] = event_report
    if event_report["status"] == "FAIL":
        report["status"] = "FAIL"
        report["issues"].extend(event_report["issues"])
    return text.strip() + "\n", report


def _owner_body(text: str, owner: str) -> tuple[int, int] | None:
    span = named_block_span(text, "part", owner)
    return (span[0] + 1, span[1]) if span else None


def _behavior_block(
    text: str,
    behavior: PlannedBehavior,
    *,
    owned: bool,
) -> str | None:
    search_text = text
    if owned:
        span = _owner_body(text, behavior.owner)
        if span is None:
            return None
        search_text = text[span[0]:span[1]]
    spans = _definition_spans(
        search_text, r"state\s+def", behavior.behavior_id
    )
    if len(spans) != 1:
        return None
    start, end = spans[0]
    return search_text[start:end]


def _state_body(block: str, state_id: str) -> str | None:
    match = re.search(
        rf"\bstate\s+(?!def\b){re.escape(state_id)}\s*(?P<tail>[;{{])",
        block,
    )
    if match is None:
        return None
    if match.group("tail") == ";":
        return ""
    opening = block.find("{", match.start(), match.end())
    closing = find_block_end(block, opening)
    return block[opening + 1:closing] if closing != -1 else None


def check_planned_behavior_conformance(
    model_text: str,
    behaviors: Sequence[PlannedBehavior],
    *,
    owned: bool,
) -> dict[str, Any]:
    """Check exact owner/behavior/state identities and executable topology."""
    checked: list[dict[str, Any]] = []
    all_issues: list[str] = []
    for behavior in behaviors:
        issues: list[str] = []
        block = _behavior_block(model_text, behavior, owned=owned)
        qualified = f"{behavior.owner}::{behavior.behavior_id}"
        if block is None:
            issues.append(f"{qualified} is absent or duplicated")
        else:
            observed_states = set(re.findall(
                r"\bstate\s+(?!def\b)([A-Za-z_]\w*)\b", block
            ))
            for state in behavior.states:
                if state.state_id not in observed_states:
                    issues.append(
                        f"{qualified}::{state.state_id} is absent"
                    )
                    continue
                body = _state_body(block, state.state_id)
                if body is None:
                    issues.append(
                        f"{qualified}::{state.state_id} cannot be parsed"
                    )
                elif (
                    state.entry_action
                    and re.search(
                        rf"\bentry\s+action(?:\s+\w+\s*:\s*)?\s*"
                        rf"{re.escape(state.entry_action)}\s*;",
                        body,
                    ) is None
                ):
                    issues.append(
                        f"{qualified}::{state.state_id} does not invoke "
                        f"{state.entry_action}"
                    )
                if (
                    body is not None
                    and state.do_action
                    and re.search(
                        rf"\bdo\s+action(?:\s+\w+\s*:\s*)?\s*"
                        rf"{re.escape(state.do_action)}\s*;",
                        body,
                    ) is None
                ):
                    issues.append(
                        f"{qualified}::{state.state_id} does not invoke "
                        f"{state.do_action}"
                    )
            initial = re.search(
                r"\bentry\s*;\s*then\s+"
                r"([A-Za-z_]\w*)\s*;",
                block,
            )
            if initial is None or initial.group(1) != behavior.initial_state:
                issues.append(
                    f"{qualified} initial state is not "
                    f"{behavior.initial_state}"
                )
            edges: list[tuple[str, str]] = []
            for transition in behavior.transitions:
                match = re.search(
                    rf"\btransition\s+"
                    rf"{re.escape(transition.transition_id)}\b"
                    rf"(?P<body>.*?);",
                    block,
                    flags=re.DOTALL,
                )
                if match is None:
                    issues.append(
                        f"{qualified} transition "
                        f"{transition.transition_id} is absent"
                    )
                    continue
                transition_text = match.group("body")
                source = re.search(
                    r"\bfirst\s+([A-Za-z_]\w*)\b", transition_text
                )
                target = re.search(
                    r"\bthen\s+([A-Za-z_]\w*)\b", transition_text
                )
                if (
                    source is None
                    or source.group(1) != transition.source
                    or target is None
                    or target.group(1) != transition.target
                ):
                    issues.append(
                        f"{qualified} transition "
                        f"{transition.transition_id} changed endpoints"
                    )
                expected_trigger = (
                    rf"\baccept\s+{re.escape(transition.trigger)}\b"
                    if transition.trigger_kind == "ACCEPT"
                    else rf"\bif\s+{re.escape(transition.trigger)}\b"
                )
                if re.search(expected_trigger, transition_text) is None:
                    issues.append(
                        f"{qualified} transition "
                        f"{transition.transition_id} changed trigger"
                    )
                # A guard that the writer dropped leaves a transition that
                # fires unconditionally — a change that reads as harmless
                # because everything still parses and every state is still
                # reachable. It is the whole of REQ_SAFE_006's defect.
                if transition.guard and re.search(
                    rf"\bif\s+{re.escape(transition.guard)}",
                    transition_text,
                ) is None:
                    issues.append(
                        f"{qualified} transition "
                        f"{transition.transition_id} dropped its guard "
                        f"'{transition.guard}'"
                    )
                edges.append((transition.source, transition.target))
            reachable = {behavior.initial_state}
            changed = True
            while changed:
                changed = False
                for source, target in edges:
                    if source in reachable and target not in reachable:
                        reachable.add(target)
                        changed = True
            for state in behavior.states:
                if (
                    state.role in {"RESPONSE", "FAULT"}
                    and state.state_id not in reachable
                ):
                    issues.append(
                        f"{qualified}::{state.state_id} is unreachable"
                    )
        checked.append({
            "owner": behavior.owner,
            "behavior_id": behavior.behavior_id,
            "status": "PASS" if not issues else "FAIL",
            "issues": issues,
        })
        all_issues.extend(issues)
    return {
        "artifact_role": (
            "OWNED_PLANNED_BEHAVIOR_CONFORMANCE"
            if owned else "PLANNED_BEHAVIOR_CONFORMANCE"
        ),
        "status": "PASS" if not all_issues else "FAIL",
        "checked": checked,
        "issues": all_issues,
    }


def materialize_owned_planned_behaviors(
    model_text: str,
    behaviors: Sequence[PlannedBehavior],
    *,
    event_symbols: Sequence[PlannedEventSymbol] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compile exact plan-owned behavior definitions into their owner parts."""
    symbols = tuple(
        event_symbols
        if event_symbols is not None
        else collect_planned_event_symbols(behaviors)
    )
    text, event_report = materialize_planned_event_symbols(
        str(model_text), symbols
    )

    materialized: list[str] = []
    reverted: list[dict[str, str]] = []
    for behavior in behaviors:
        owner_span = _owner_body(text, behavior.owner)
        if owner_span is None:
            continue
        owner_text = text[owner_span[0]:owner_span[1]]
        action_declarations = []
        for state in behavior.states:
            for action in (state.entry_action, state.do_action):
                # Any DECLARATION spelling blocks re-declaration: `action def
                # X`, a typed usage `action X : T`, or a bare owned usage
                # `action X;` / `action X {`. (References inside state bodies
                # — `entry action u : X;` — do not match: `X` there follows a
                # colon, not the `action` keyword.)
                if (
                    action
                    and re.search(
                        rf"\baction\s+(?:def\s+)?"
                        rf"{re.escape(action)}\s*[:;{{]",
                        owner_text,
                    ) is None
                ):
                    action_declarations.append(
                        f"        action def {action} {{}}\n"
                    )
        existing = _definition_spans(
            owner_text, r"state\s+def", behavior.behavior_id
        )
        if not existing:
            # A part-level BODIED usage spelling (`state Name { ... }`) is the
            # same declaration in the extractor-invisible form; leaving it and
            # injecting a def beside it produced the measured double
            # declaration (a same-name sibling pair is exactly what syside
            # flags). Replace it with the plan-blessed def instead — but only
            # a TOP-LEVEL usage: a nested occurrence belongs to another scope
            # and replacing it there would plant the def in the wrong place
            # (the fingerprint guard below covers that shape by reverting).
            existing = [
                (start, end)
                for start, end in _definition_spans(
                    text=owner_text, kind=r"state", name=behavior.behavior_id
                )
                if owner_text[start:end].rstrip().endswith("}")
                and "exhibit" not in owner_text[max(0, start - 24):start]
                and owner_text[:start].count("{")
                == owner_text[:start].count("}")
            ]
        block = "\n".join(
            "        " + line if line else ""
            for line in emit_planned_behavior(behavior).splitlines()[1:]
        )
        insertion = (
            "\n"
            + "".join(action_declarations)
            + block
            + "\n    "
        )
        before_text = text
        before_fp = _shadow_fingerprint(text)
        if existing:
            start, end = existing[0]
            absolute_start = owner_span[0] + start
            absolute_end = owner_span[0] + end
            text = (
                text[:absolute_start]
                + insertion.strip("\n")
                + text[absolute_end:]
            )
        else:
            closing = owner_span[1]
            text = text[:closing] + insertion + text[closing:]
        after_fp = _shadow_fingerprint(text)
        if after_fp > before_fp:
            # Injection must never create a duplicate/shadow the model did
            # not already have — the surgical-pass discipline applied to our
            # own writers. Reverting leaves the behaviour absent, which the
            # conformance report below surfaces for the in-loop repair.
            text = before_text
            reverted.append({
                "behavior": f"{behavior.owner}::{behavior.behavior_id}",
                "reason": (
                    "materialization reverted: it would add "
                    f"{after_fp[0] - before_fp[0]} duplicate member(s) and "
                    f"{after_fp[1] - before_fp[1]} shadowing warning(s)"
                ),
            })
            continue
        materialized.append(
            f"{behavior.owner}::{behavior.behavior_id}"
        )

    report = check_planned_behavior_conformance(
        text, behaviors, owned=True
    )
    report["event_symbol_conformance"] = event_report
    if event_report["status"] == "FAIL":
        report["status"] = "FAIL"
        report["issues"].extend(event_report["issues"])
    report["materialized"] = materialized
    if reverted:
        report["reverted"] = reverted
        report["issues"].extend(item["reason"] for item in reverted)
    return text, report


def _shadow_fingerprint(model_text: str) -> tuple[int, int]:
    """(duplicate members, shadowing warnings) — the injection-safety metric.

    Every pipeline writer that adds named declarations compares this before
    and after each piece; a strictly worse fingerprint means the piece
    manufactured a namespace defect the model did not have (measured on the
    2026-08-30 draws: 59 of 61 shadowing warnings were injector-adjacent).
    """
    from ..simulation.syntax_checker import check_syntax
    from .namespace_integrity import check_user_namespace_integrity

    try:
        duplicates = len(
            check_user_namespace_integrity(model_text)["duplicate_members"]
        )
    except Exception:
        duplicates = 0
    try:
        warnings = sum(
            1 for w in check_syntax(model_text).warnings
            if "shadows" in str(w.get("message", ""))
        )
    except Exception:
        warnings = 0
    return duplicates, warnings
