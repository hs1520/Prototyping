"""Requirement → model-identity bindings for the verification harness.

The harness held a hard-coded vocabulary at four independent layers (accept
event names, action-def names, sent-command spellings), so a structurally
correct model that named things its own way was scored as defective — run3,
the first model to close REQ-SAFE-006, scored worse than two earlier models
that carried the actual defect but used the harness's spellings.

Every one of those identities already exists as validated data: the frozen
generation plan records, per requirement, the owning component, the behavior,
the accept-event symbol, the response state and its entry/do actions, the
transition guards, and (for causal-path realizations) the port route the
response travels. The plan is validated against the requirement texts at
freeze time and materialised into the model by a sole writer. Harness-held
canonical names are therefore duplicated derived data — the same defect class
as a `behavior_kind` field that contradicts its own `behaviors[]`.

This module is the single resolution layer both harness sides consume:

- **PLAN** bindings (this module): authoritative, built from the archived
  plan payload (a plain dict — no prototyping imports, this package sits
  below it in the dependency order).
- **SEMANTIC** resolution (``ModelDrivenMission.resolve_event`` and
  causal-role lookups): the fallback for models that have no plan.
- A binding that cannot be built is reported as such; a resolution failure
  must surface as "not measured", never as a model defect. The one inversion:
  a PLAN-bound name that is absent from the model text IS a model defect —
  materialisation guarantees presence, so absence means the model diverged
  from its own plan.

Judgement stays strict: bindings supply IDENTITY only. Whether the bound
transition carries the required guard, or the bound action sends the required
command, is still judged by the criteria — a resolver must never substitute
an element that would pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..utils.req_id import normalise_req_id

#: Response-shaped state roles: the states a requirement's trigger drives the
#: machine INTO, whose entry/do actions are the requirement's response.
_RESPONSE_ROLES = ("RESPONSE", "FAULT")


@dataclass(frozen=True)
class RequirementBinding:
    """The model identities that realize one requirement, per its own plan."""

    requirement_id: str
    provenance: str = "PLAN"
    owner: str = ""
    behavior: str = ""
    #: ACCEPT symbols on transitions into response states.
    trigger_events: Tuple[str, ...] = ()
    #: GUARD-kind trigger expressions on transitions into response states.
    trigger_conditions: Tuple[str, ...] = ()
    response_states: Tuple[str, ...] = ()
    #: entry/do action names of the response states — the actions the
    #: harness previously looked up under its own spellings.
    response_actions: Tuple[str, ...] = ()
    #: (transition_id, guard) for every transition into a response state,
    #: guard verbatim ("" = unconditional). Identity only — judging the
    #: guard's adequacy stays with the criteria.
    transition_guards: Tuple[Tuple[str, str], ...] = ()
    #: (source_component, source_port, target_component, target_port) hops of
    #: the realization's causal path — the route the response's command
    #: actually travels (layer-4 identity).
    route: Tuple[Tuple[str, str, str, str], ...] = ()
    issues: Tuple[str, ...] = ()

    @property
    def route_ports(self) -> Tuple[str, ...]:
        seen: list[str] = []
        for hop in self.route:
            for port in (hop[1], hop[3]):
                if port and port not in seen:
                    seen.append(port)
        return tuple(seen)


def plan_bindings(
    plan_payload: Optional[Mapping[str, Any]],
) -> Dict[str, RequirementBinding]:
    """{requirement_id: binding} from an archived plan payload dict.

    Accepts the raw ``whole_model_generation_plan`` dict as stored in run
    reports and model metadata. Returns {} when no plan is available — the
    caller then falls back to semantic resolution, never to a verdict.
    """
    if not isinstance(plan_payload, Mapping):
        return {}

    drafts: Dict[str, dict] = {}

    def draft(req_id: str) -> dict:
        return drafts.setdefault(req_id, {
            "owner": "", "behavior": "",
            "trigger_events": [], "trigger_conditions": [],
            "response_states": [], "response_actions": [],
            "transition_guards": [], "route": [], "issues": [],
        })

    for behavior in plan_payload.get("behaviors") or ():
        if not isinstance(behavior, Mapping):
            continue
        provenance = behavior.get("provenance")
        req_raw = (
            provenance.get("requirement_id")
            if isinstance(provenance, Mapping) else None
        ) or behavior.get("source_requirement_id")
        if not req_raw:
            continue
        entry = draft(normalise_req_id(str(req_raw)))
        entry["owner"] = str(behavior.get("owner") or entry["owner"])
        entry["behavior"] = str(
            behavior.get("behavior_id") or entry["behavior"]
        )
        states = {
            str(state.get("state_id")): state
            for state in behavior.get("states") or ()
            if isinstance(state, Mapping)
        }
        response_ids = {
            state_id for state_id, state in states.items()
            if str(state.get("role", "")).upper() in _RESPONSE_ROLES
        }
        for state_id in sorted(response_ids):
            entry["response_states"].append(state_id)
            state = states[state_id]
            for action in (state.get("entry_action"), state.get("do_action")):
                if action:
                    entry["response_actions"].append(str(action))
        for transition in behavior.get("transitions") or ():
            if not isinstance(transition, Mapping):
                continue
            # Identity is recorded for EVERY transition: an inhibition guard
            # can sit on a transition into a NORMAL-role state (run3's
            # PayloadLockBehavior guards Locked→Unlocked), and the criteria —
            # not this resolver — decide which transitions matter.
            kind = str(transition.get("trigger_kind", "")).upper()
            trigger = str(transition.get("trigger") or "")
            if kind == "ACCEPT" and trigger:
                entry["trigger_events"].append(trigger)
            elif trigger:
                entry["trigger_conditions"].append(trigger)
            entry["transition_guards"].append((
                str(transition.get("transition_id") or ""),
                str(transition.get("guard") or ""),
            ))

    for realization in plan_payload.get("requirement_realizations") or ():
        if not isinstance(realization, Mapping):
            continue
        req_raw = realization.get("requirement_id")
        if not req_raw:
            continue
        entry = draft(normalise_req_id(str(req_raw)))
        if realization.get("owner_component") and not entry["owner"]:
            entry["owner"] = str(realization["owner_component"])
        if realization.get("behavior_name") and not entry["behavior"]:
            entry["behavior"] = str(realization["behavior_name"])
        for hop in realization.get("connection_path") or ():
            if not isinstance(hop, Mapping):
                continue
            entry["route"].append((
                str(hop.get("source_component") or ""),
                str(hop.get("source_port") or ""),
                str(hop.get("target_component") or ""),
                str(hop.get("target_port") or ""),
            ))

    def _unique(items: Sequence) -> tuple:
        return tuple(dict.fromkeys(items))

    return {
        req_id: RequirementBinding(
            requirement_id=req_id,
            owner=entry["owner"],
            behavior=entry["behavior"],
            trigger_events=_unique(entry["trigger_events"]),
            trigger_conditions=_unique(entry["trigger_conditions"]),
            response_states=_unique(entry["response_states"]),
            response_actions=_unique(entry["response_actions"]),
            transition_guards=_unique(entry["transition_guards"]),
            route=_unique(entry["route"]),
            issues=tuple(entry["issues"]),
        )
        for req_id, entry in sorted(drafts.items())
    }


def binding_for(
    bindings: Mapping[str, RequirementBinding],
    requirement_id: str,
) -> Optional[RequirementBinding]:
    return bindings.get(normalise_req_id(str(requirement_id)))


#: Tokens that appear in guard expressions without being identities.
_GUARD_EXPR_STOP = frozenset({"not", "and", "or", "true", "false", "if"})

_IDENTIFIER_RE = __import__("re").compile(r"[A-Za-z_]\w*")


def identity_tokens(binding: RequirementBinding) -> frozenset:
    """Lowercased identifier spellings that ARE this requirement's model
    elements per its plan: accept-event names, and the flag tokens read by
    its transition guards / guard-kind triggers. Route ports are deliberately
    EXCLUDED — ports are plentiful and pooling them was measured to hand
    unrelated requirements a match (FUNC_004 drifted into the L2 suite)."""
    tokens = {event.lower() for event in binding.trigger_events}
    for expression in (
        *binding.trigger_conditions,
        *(guard for _tid, guard in binding.transition_guards),
    ):
        tokens.update(
            token.lower()
            for token in _IDENTIFIER_RE.findall(expression)
            if token.lower() not in _GUARD_EXPR_STOP
        )
    return frozenset(tokens)
