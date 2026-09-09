"""Read-only audit of what the committed model's actions do.

The behavioural simulator credits a response on the action label alone
(``behavioral_sim`` check 2), so a model passes with every action definition
empty, as the archived pilot does. This module changes no verdict: it reports
each action definition's realisation state and, for every send, whether the
event reaches a consumer that accepts it. ``ENFORCE_V1`` turns that evidence
into a gate; ``LEGACY_AUDIT`` only records it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .action_effects import (
    ENFORCE_V1,
    EXTERNAL_OR_UNSUPPORTED,
    LEGACY_AUDIT,
    OFF,
    PROFILE_VERSION,
    PlannedActionEffect,
)
from ..utils.sysml_text_utils import find_block_end
from ..simulation.extractor import extract_behavioral_graph
from ..simulation.state_extractor import extract_state_machines

ARTIFACT_ROLE = "TERMINAL_ACTION_SEMANTICS_AUDIT"

REALISED_BY_BODY = "REALISED_BY_BODY"
REALISED_BY_TYPED_REF = "REALISED_BY_TYPED_REF"
LABEL_ONLY = "LABEL_ONLY"
ORPHAN = "ORPHAN"

# `action def X` with word boundaries; a plain substring search also matches
# `entry action defaultToLockedState;` and inflated the denominator by three
# in the archived pilot.
_ACTION_DEF_RE = re.compile(r"\baction\s+def\s+([A-Za-z_]\w*)")
_TYPED_USAGE_RE = re.compile(
    r"\b(?:entry|do|exit)\s+action\s+([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)"
)
_BARE_USAGE_RE = re.compile(
    r"\b(?:entry|do|exit)\s+action\s+([A-Za-z_]\w*)\s*;"
)
_OWNER_SCOPE_RE = re.compile(r"\b(?:part\s+def|package)\s+([A-Za-z_]\w*)\s*\{")
# `part actuator : PayloadActuator;` - the connection graph is keyed by part
# usage, a plan names the part definition. Without this map a plan naming
# `PayloadActuator` is reported as unrouted.
_PART_USAGE_RE = re.compile(
    r"\bpart\s+(?!def\b)([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*[;{]"
)

UNRESOLVED_ACTION = "UNRESOLVED_ACTION"
UNSUPPORTED_BODY = "UNSUPPORTED_BODY"
BARE_INVOCATION = "BARE_INVOCATION"
NO_CONNECT_PATH = "NO_CONNECT_PATH"
NO_ACCEPT_TRANSITION = "NO_ACCEPT_TRANSITION"
PLAN_IDENTITY_INCOMPLETE = "PLAN_IDENTITY_INCOMPLETE"


@dataclass
class ActionRecord:
    name: str
    owner_scope: Optional[str]
    realisation: str
    body_empty: bool
    typed_by: Tuple[str, ...] = ()
    named_by_bare: Tuple[str, ...] = ()
    sends: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_def": self.name,
            "owner_scope": self.owner_scope,
            "realisation": self.realisation,
            "body_empty": self.body_empty,
            "typed_by": list(self.typed_by),
            "named_by_bare": list(self.named_by_bare),
            "sends": [
                {"event_type": event, "sender_port": port}
                for event, port in self.sends
            ],
        }


@dataclass
class SendRecord:
    owner_behavior: str
    owner_part: Optional[str]
    state: str
    action_def: Optional[str]
    event_type: str
    sender_port: str
    routed_to: Tuple[str, ...] = ()
    accepted_by: Tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if not self.accepted_by:
            return NO_ACCEPT_TRANSITION
        if not self.routed_to:
            return NO_CONNECT_PATH
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_behavior": self.owner_behavior,
            "owner_part": self.owner_part,
            "state": self.state,
            "action_def": self.action_def,
            "event_type": self.event_type,
            "sender_port": self.sender_port,
            "routed_to": list(self.routed_to),
            "accepted_by": list(self.accepted_by),
            "status": self.status,
        }


@dataclass
class ActionSemanticsReport:
    profile: str
    status: str = "ADVISORY"
    summary: Dict[str, int] = field(default_factory=dict)
    actions: List[ActionRecord] = field(default_factory=list)
    sends: List[SendRecord] = field(default_factory=list)
    requirement_chains: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_role": ARTIFACT_ROLE,
            "profile": self.profile,
            "profile_version": PROFILE_VERSION,
            "status": self.status,
            "summary": dict(self.summary),
            "actions": [item.to_dict() for item in self.actions],
            "sends": [item.to_dict() for item in self.sends],
            "requirement_chains": list(self.requirement_chains),
            "issues": list(self.issues),
        }


def _owner_scopes(text: str) -> List[Tuple[int, int, str]]:
    scopes: List[Tuple[int, int, str]] = []
    for match in _OWNER_SCOPE_RE.finditer(text):
        opening = text.find("{", match.start())
        if opening == -1:
            continue
        closing = find_block_end(text, opening)
        if closing != -1:
            scopes.append((opening, closing, match.group(1)))
    return scopes


def _innermost_scope(
    scopes: Sequence[Tuple[int, int, str]], position: int
) -> Optional[str]:
    enclosing = [s for s in scopes if s[0] < position < s[1]]
    if not enclosing:
        return None
    return min(enclosing, key=lambda s: s[1] - s[0])[2]


def _body_is_empty(text: str, definition_start: int) -> bool:
    """Brace matching is used for this and nothing else.

    Whether a definition has content is lexical; every type relation below comes
    from syside.
    """
    opening = text.find("{", definition_start)
    if opening == -1:
        return True
    closing = find_block_end(text, opening)
    if closing == -1:
        return True
    return text[opening + 1:closing].strip() == ""


def _reachable_parts(model_text: str, part_name: str, port_name: str) -> Tuple[str, ...]:
    """Parts an out/inout port reaches over declared connections.

    A SysML connection is directed by its end ports, not by the connect
    statement, so the walk is undirected and direction is tested at the ends:
    sender out/inout, consumer in/inout.
    """
    graph = extract_behavioral_graph(model_text)
    if not graph.ports:
        return ()

    def _matches_sender(node) -> bool:
        return (
            node.port_name == port_name
            and node.direction in ("out", "inout")
            and (part_name is None or node.part_name.lower() == part_name.lower())
        )

    starts = [pid for pid, node in graph.ports.items() if _matches_sender(node)]
    if not starts:
        # Owner is a part definition, ports are keyed by part usage; fall back to the
        # port name alone so a naming mismatch is not reported as a routing failure.
        starts = [
            pid for pid, node in graph.ports.items()
            if node.port_name == port_name and node.direction in ("out", "inout")
        ]
    if not starts:
        return ()

    adjacency: Dict[str, List[str]] = {}
    for edge in graph.connections:
        adjacency.setdefault(edge.source, []).append(edge.target)
        adjacency.setdefault(edge.target, []).append(edge.source)

    seen = set(starts)
    frontier = list(starts)
    consumers: List[str] = []
    while frontier:
        current = frontier.pop()
        for neighbour in adjacency.get(current, ()):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            frontier.append(neighbour)
            node = graph.ports.get(neighbour)
            if node is None or node.direction not in ("in", "inout"):
                continue
            if node.part_name not in consumers:
                consumers.append(node.part_name)
    return tuple(consumers)


def analyze_action_semantics(
    model_text: str,
    action_effect_plan: Optional[Sequence[PlannedActionEffect]] = None,
    profile: str = LEGACY_AUDIT,
) -> ActionSemanticsReport:
    """Describe how every action definition in *model_text* is realised.

    With no ``action_effect_plan`` there are no planned chains to check and the
    report says so; chains are not reconstructed by name, since inferring an
    action's requirement from its spelling is the defect this audit measures.
    """
    text = model_text or ""
    report = ActionSemanticsReport(profile=profile)
    if profile == OFF or not text.strip():
        report.summary = _empty_summary()
        return report

    scopes = _owner_scopes(text)
    typed_targets: Dict[str, List[str]] = {}
    for match in _TYPED_USAGE_RE.finditer(text):
        typed_targets.setdefault(match.group(2), []).append(match.group(1))
    bare_names: Dict[str, int] = {}
    for match in _BARE_USAGE_RE.finditer(text):
        bare_names[match.group(1)] = bare_names.get(match.group(1), 0) + 1

    # Sends come from syside, not the text: reaching one means following the
    # usage's type edge into the definition body, which a bare usage lacks.
    machines = extract_state_machines(text)
    # Where each definition is invoked through a type edge. The usage label is a
    # local name and is not part of the key - three runs of one configuration
    # spelled the same invocation three ways.
    invocations: Dict[str, List[Tuple[str, str, str]]] = {}
    sends_by_action: Dict[str, List[Tuple[str, str]]] = {}
    send_records: List[SendRecord] = []
    accept_triggers: Dict[str, List[str]] = {}
    for machine in machines:
        for transition in machine.transitions:
            if transition.accept_trigger:
                accept_triggers.setdefault(
                    transition.accept_trigger, []
                ).append(machine.name)
        for state in machine.states:
            definition = state.entry_action_def or state.do_action_def
            if definition:
                invocations.setdefault(definition, []).append(
                    (machine.owner_part or "", machine.name, state.name)
                )
            for event, port in state.sends:
                if definition:
                    sends_by_action.setdefault(definition, []).append(
                        (event, port)
                    )
                send_records.append(SendRecord(
                    owner_behavior=machine.name,
                    owner_part=machine.owner_part,
                    state=state.name,
                    action_def=definition,
                    event_type=event,
                    sender_port=port,
                ))

    for record in send_records:
        record.accepted_by = tuple(accept_triggers.get(record.event_type, ()))
        record.routed_to = _reachable_parts(
            text, record.owner_part, record.sender_port
        )

    for match in _ACTION_DEF_RE.finditer(text):
        name = match.group(1)
        empty = _body_is_empty(text, match.end())
        typed = tuple(typed_targets.get(name, ()))
        if not empty:
            realisation = REALISED_BY_BODY
        elif typed:
            realisation = REALISED_BY_TYPED_REF
        elif name in bare_names:
            realisation = LABEL_ONLY
        else:
            realisation = ORPHAN
        report.actions.append(ActionRecord(
            name=name,
            owner_scope=_innermost_scope(scopes, match.start()),
            realisation=realisation,
            body_empty=empty,
            typed_by=typed,
            named_by_bare=(name,) if name in bare_names else (),
            sends=tuple(sends_by_action.get(name, ())),
        ))

    report.sends = send_records
    plan = tuple(action_effect_plan or ())
    report.requirement_chains = _check_planned_chains(
        plan, report, invocations, text
    )
    report.summary = _summarise(report, plan, bare_names, typed_targets)
    report.issues = _collect_issues(report, plan)
    report.status = _status_for(profile, report)
    return report


def _empty_summary() -> Dict[str, int]:
    return {
        "definitions_total": 0,
        "empty_definitions": 0,
        "planned_response_actions": 0,
        "typed_invocations": 0,
        "bare_invocations": 0,
        "supported_effects": 0,
        "unmatched_sends": 0,
        "unrouted_sends": 0,
        "complete_requirement_chains": 0,
    }


def _summarise(
    report: ActionSemanticsReport,
    plan: Sequence[PlannedActionEffect],
    bare_names: Dict[str, int],
    typed_targets: Dict[str, List[str]],
) -> Dict[str, int]:
    return {
        "definitions_total": len(report.actions),
        "empty_definitions": sum(1 for a in report.actions if a.body_empty),
        "planned_response_actions": len(plan),
        "typed_invocations": sum(len(v) for v in typed_targets.values()),
        "bare_invocations": sum(bare_names.values()),
        "supported_effects": sum(1 for a in report.actions if a.sends),
        # Counted independently of `status`: a send can be both unrouted and
        # unaccepted, and counting only the first understates the routing gap.
        "unmatched_sends": sum(1 for s in report.sends if not s.accepted_by),
        "unrouted_sends": sum(1 for s in report.sends if not s.routed_to),
        "complete_requirement_chains": sum(
            1 for c in report.requirement_chains if c["status"] == "PASS"
        ),
    }


def _check_planned_chains(
    plan: Sequence[PlannedActionEffect],
    report: ActionSemanticsReport,
    invocations: Dict[str, List[Tuple[str, str, str]]],
    text: str,
) -> List[Dict[str, Any]]:
    chains: List[Dict[str, Any]] = []
    by_name = {item.name: item for item in report.actions}
    for effect in plan:
        failures: List[str] = []
        evidence: Dict[str, Any] = {}
        if not effect.is_complete_identity():
            failures.append(PLAN_IDENTITY_INCOMPLETE)
        elif effect.effect_kind == EXTERNAL_OR_UNSUPPORTED:
            chains.append({
                "requirement_id": effect.requirement_id,
                "action_def": effect.action_def,
                "status": EXTERNAL_OR_UNSUPPORTED,
                "failures": [],
            })
            continue
        record = by_name.get(effect.action_def)
        if record is None:
            failures.append(UNRESOLVED_ACTION)
        else:
            if record.owner_scope and record.owner_scope != effect.owner_def:
                failures.append(UNRESOLVED_ACTION)
            if record.body_empty:
                failures.append(UNSUPPORTED_BODY)
            # The planned response state invokes this definition through a type edge.
            # The usage label is a local name and is not checked: three runs of one
            # configuration spelled the same invocation `onParachute`,
            # `...SelectedAction` and `onSet...`. Requiring the planned spelling would
            # reject correct models; accepting any state at all would let an unrelated
            # machine discharge the obligation.
            observed = tuple(invocations.get(effect.action_def, ()))
            if not any(
                owner == effect.owner_def
                and behaviour == effect.owner_behavior
                and state == effect.response_state
                for owner, behaviour, state in observed
            ):
                failures.append(BARE_INVOCATION)
                # Recorded because a report carrying only the code cannot distinguish a
                # missing type edge from a renamed state; the first failure needed a
                # separate reading of the model to tell which of the three did not match.
                evidence = {
                    "expected": {
                        "owner_def": effect.owner_def,
                        "owner_behavior": effect.owner_behavior,
                        "response_state": effect.response_state,
                    },
                    "observed": [
                        {
                            "owner_def": owner,
                            "owner_behavior": behaviour,
                            "response_state": state,
                        }
                        for owner, behaviour, state in observed
                    ],
                }
            if (effect.event_type, effect.sender_port) not in record.sends:
                failures.append(UNSUPPORTED_BODY)
        match = next(
            (
                s for s in report.sends
                if s.action_def == effect.action_def
                and s.event_type == effect.event_type
            ),
            None,
        )
        if match is None:
            failures.append(UNSUPPORTED_BODY)
        else:
            reached = set(match.routed_to)
            reached.update(
                definition for usage, definition in _PART_USAGE_RE.findall(text)
                if usage in match.routed_to
            )
            if effect.consumer_owner_def not in reached:
                failures.append(NO_CONNECT_PATH)
            if effect.consumer_behavior not in match.accepted_by:
                failures.append(NO_ACCEPT_TRANSITION)
        chains.append({
            "requirement_id": effect.requirement_id,
            "action_def": effect.action_def,
            "status": "PASS" if not failures else "FAIL",
            "failures": sorted(set(failures)),
            **({"invocation": evidence} if evidence else {}),
        })
    return chains


def _collect_issues(
    report: ActionSemanticsReport, plan: Sequence[PlannedActionEffect]
) -> List[str]:
    issues: List[str] = []
    for chain in report.requirement_chains:
        for failure in chain["failures"]:
            issues.append(
                f"{chain['requirement_id']}: {chain['action_def']} {failure}"
            )
    for record in report.sends:
        if record.status != "PASS":
            issues.append(
                f"{record.owner_behavior}.{record.state}: "
                f"{record.event_type} via {record.sender_port} {record.status}"
            )
    return issues


def _status_for(profile: str, report: ActionSemanticsReport) -> str:
    if profile != ENFORCE_V1:
        return "ADVISORY"
    if not report.requirement_chains:
        return "FAIL"
    return (
        "PASS"
        if all(c["status"] != "FAIL" for c in report.requirement_chains)
        else "FAIL"
    )
