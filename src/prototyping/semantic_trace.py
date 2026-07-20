"""Contract-first semantic trace construction for generated SysML behavior."""
from __future__ import annotations

import re
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .contract_types import (
    ContractBundle,
    INCOMPLETE,
    READY,
    UNSUPPORTED,
    RequirementContract,
    RequirementObligation,
    contract_bundle_from_dict,
    normalise_req_id,
)
from .failure_routing import Diagnostic
from .platform_semantics import (
    action_matches,
    command_matches,
    normalize_symbol,
    platform_binding,
)
from .safety_patterns import PatternBinding, pattern_by_id


TRACE_SCHEMA_VERSION = "1.1"
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class TraceLink:
    req_id: str
    obligation_id: str
    link_kind: str
    expected_concept: str
    observed_element: str | None
    status: str
    evidence: tuple[str, ...] = ()


@dataclass
class RequirementTrace:
    req_id: str
    contract_status: str
    pattern_ids: list[str] = field(default_factory=list)
    links: list[TraceLink] = field(default_factory=list)
    conformance: str = OUT_OF_SCOPE
    findings: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "contract_status": self.contract_status,
            "pattern_ids": list(self.pattern_ids),
            "links": [asdict(link) for link in self.links],
            "conformance": self.conformance,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class SemanticTraceReport:
    schema_version: str
    model_name: str
    model_digest: str
    traces: tuple[RequirementTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trace in self.traces:
            counts[trace.conformance] = counts.get(trace.conformance, 0) + 1
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "traces": [trace.to_dict() for trace in self.traces],
            "counts": counts,
        }

    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return tuple(item for trace in self.traces for item in trace.findings)


_TRIGGER_ALIASES: dict[str, tuple[str, ...]] = {
    "critical_propulsion_failure": ("propulsion", "engine", "motor", "thrust", "criticalfailure"),
    "single_motor_failure": ("singlemotorfailure", "motorinoperative", "motorout"),
    "delivery_abort_condition": ("deliveryabort", "abortcondition", "payloadabort"),
    "sensor_self_test_failure": ("sensorselftest", "selftestfailed", "sensorfailed", "prearm"),
    "gcs_link_absent": ("commloss", "gcsloss", "linkloss", "heartbeat", "uplinkabsent"),
    "battery_state_of_charge": ("batterysoc", "batterycharge", "stateofcharge", "lowbattery"),
    "valid_waypoint_modification_command": ("validwaypoint", "waypointmodification", "modifywaypoint"),
    "automated_landing_completed": ("landingcompleted", "automatedlandingcompleted", "postlanding"),
    "delivery_waypoint_proximity": ("waypointdistance", "deliverywaypoint", "positionwithindelivery"),
    "delivery_coordinate_condition_satisfied": (
        "deliverycoordinatecondition", "coordinateconditionsatisfied",
        "deliveryconditionmet",
    ),
    "contingency_condition": ("contingencycondition", "contingency"),
    "power_on": ("poweron", "startup", "startupphase"),
    "obstacle_detected": ("obstacledistance", "obstacledetected", "collisionthreat"),
}

_RESPONSE_FALLBACK: dict[str, tuple[str, ...]] = {
    "avoid_obstacle": ("avoidobstacle", "avoidancemanoeuvre", "collisionavoidance"),
}


def _diag(
    req_id: str,
    obligation_id: str,
    code: str,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    affected: Iterable[str] = (),
) -> Diagnostic:
    return Diagnostic(
        diagnostic_id=f"{req_id}:{obligation_id}:{code}",
        req_id=req_id,
        source_stage="semantic_trace",
        finding_code=code,
        expected=dict(expected),
        observed=dict(observed),
        evidence_refs=(obligation_id,),
        affected_elements=tuple(dict.fromkeys(affected)),
    )


def _satisfy_map(model: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for part in getattr(model, "part_definitions", ()):
        for satisfy in getattr(part, "satisfy_relationships", ()):
            target = getattr(getattr(satisfy, "target", None), "name", "")
            if target:
                result.setdefault(normalise_req_id(target), []).append(part.name)
    return result


def _transition_signature(transition: Any) -> tuple[str, ...]:
    symbols: list[str] = []
    if getattr(transition, "accept_trigger", None):
        symbols.append(transition.accept_trigger)
    for guard in getattr(transition, "guards", ()):
        attrs = getattr(guard, "involved_attributes", lambda: [])()
        symbols.extend(attrs)
        symbols.append(getattr(guard, "description", lambda: "")())
    return tuple(symbols)


def _trigger_matches(obligation: RequirementObligation, transition: Any) -> bool:
    if obligation.trigger is None:
        return False
    aliases = tuple(normalize_symbol(item) for item in _TRIGGER_ALIASES.get(
        obligation.trigger.concept, (obligation.trigger.concept,)
    ))
    observed = tuple(normalize_symbol(item) for item in _transition_signature(transition))
    return any(alias and alias in symbol for alias in aliases for symbol in observed)


def _reachable_states(sm: Any) -> set[str]:
    initial = getattr(sm, "initial_state", None)
    if not initial:
        return set()
    reached = {initial}
    changed = True
    while changed:
        changed = False
        for transition in getattr(sm, "transitions", ()):
            source = getattr(transition, "source", None)
            target = getattr(transition, "target", None)
            if source in reached and target and target not in reached:
                reached.add(target)
                changed = True
    return reached


def _state(sm: Any, name: str | None) -> Any | None:
    return next((item for item in sm.states if item.name == name), None)


def _response_matches(obligation: RequirementObligation, state: Any) -> tuple[bool, bool]:
    if obligation.response is None or state is None:
        return False, False
    symbols = tuple(filter(None, (
        getattr(state, "entry_action", None),
        getattr(state, "entry_action_def", None),
    )))
    sends = tuple(command for command, _port in getattr(state, "sends", ()))
    binding = platform_binding(obligation.response.concept)
    if binding is not None:
        return action_matches(binding, *symbols), command_matches(binding, *sends)
    aliases = tuple(normalize_symbol(item) for item in _RESPONSE_FALLBACK.get(
        obligation.response.concept, (obligation.response.concept,)
    ))
    observed = tuple(normalize_symbol(item) for item in symbols)
    action_ok = any(alias and alias in symbol for alias in aliases for symbol in observed)
    return action_ok, True


def _response_timing_tokens(obligation: RequirementObligation) -> tuple[str, ...]:
    """Return response-specific tokens used to scope timing evidence.

    Timing is deliberately not accepted merely because *some* latency constraint
    exists in the model.  At least one response-specific token must occur in the
    constraint name/body so evidence for (for example) waypoint latency cannot
    close a parachute obligation.
    """
    if obligation.response is None:
        return ()
    raw = obligation.response.concept
    tokens = [normalize_symbol(raw)]
    tokens.extend(
        normalize_symbol(token)
        for token in raw.split("_")
        if len(token) >= 4
        and token not in {"deploy", "prevent", "maintain", "controlled"}
    )
    binding = platform_binding(raw)
    if binding is not None:
        tokens.extend(normalize_symbol(alias) for alias in binding.action_aliases)
    return tuple(dict.fromkeys(token for token in tokens if token))


def _timing_represented(model_text: str, obligation: RequirementObligation) -> bool:
    criterion = obligation.criterion
    if criterion is None or criterion.metric != "response_latency":
        return True
    expected = float(criterion.value)
    comparator = re.escape(criterion.comparator or "<=")
    response_tokens = _response_timing_tokens(obligation)
    attribute_values: dict[str, float] = {}
    for match in re.finditer(
        r"\battribute\s+(?P<name>\w+)\s*:[^;=]+="
        r"\s*(?P<value>[+-]?\d+(?:\.\d+)?)\b[^;]*;",
        model_text,
        re.IGNORECASE,
    ):
        attribute_values[match.group("name")] = float(match.group("value"))

    for match in re.finditer(
        r"\bassert\s+constraint\s+(?P<name>\w+)\s*\{(?P<body>[^{}]*)\}",
        model_text,
        re.IGNORECASE | re.DOTALL,
    ):
        body = match.group("body")
        scoped = normalize_symbol(match.group("name") + " " + body)
        if response_tokens and not any(token in scoped for token in response_tokens):
            continue
        if not re.search(comparator, body):
            continue
        literal_values = [
            float(item)
            for item in re.findall(r"(?<![A-Za-z_])[+-]?\d+(?:\.\d+)?", body)
        ]
        identifiers = re.findall(r"\b[A-Za-z_]\w*\b", body)
        bound_values = literal_values + [
            attribute_values[name]
            for name in identifiers
            if name in attribute_values
        ]
        tolerance = max(abs(expected) * 1e-9, 1e-9)
        if any(abs(value - expected) <= tolerance for value in bound_values):
            return True
    return False


def _trace_locked_default(
    trace: RequirementTrace,
    contract: RequirementContract,
    obligation: RequirementObligation,
    owners: list[str],
    state_machines: list[Any],
) -> None:
    """Trace a power-on default as an initial-state invariant, not an event edge."""
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "requirement_to_owner",
        "satisfying owner", owners[0] if owners else None,
        PASS if owners else FAIL, tuple(owners),
    ))
    if not owners:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "MISSING_REQUIREMENT_OWNER",
            {"satisfy": contract.req_id}, {},
        ))
        return

    machines = [sm for sm in state_machines if sm.owner_part in owners]
    binding = platform_binding("lock_payload")
    candidates: list[tuple[Any, Any]] = []
    for machine in machines:
        initial = _state(machine, machine.initial_state)
        if initial is not None:
            candidates.append((machine, initial))
    matched: tuple[Any, Any] | None = None
    for machine, initial in candidates:
        state_symbol = normalize_symbol(initial.name)
        action_ok = binding is not None and action_matches(
            binding, initial.entry_action or "", initial.entry_action_def or ""
        )
        if action_ok or any(
            token in state_symbol for token in ("locked", "secured", "held")
        ):
            matched = (machine, initial)
            break
    observed = tuple(f"{machine.name}:{state.name}" for machine, state in candidates)
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "owner_to_initial_safe_state",
        "payload locked at power-on",
        f"{matched[0].name}:{matched[1].name}" if matched else None,
        PASS if matched else FAIL, observed,
    ))
    oracle_ok = bool(
        binding is not None
        and binding.observation_concept
        == obligation.verification_intent.observation_concept
    )
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "initial_state_to_oracle",
        obligation.verification_intent.observation_concept,
        binding.observation_concept if binding and matched else None,
        PASS if matched and oracle_ok else FAIL,
        (binding.semantic_tag,) if binding else (),
    ))
    if not matched:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "INVARIANT_NOT_REPRESENTED",
            {"initial_state": "payload_locked"},
            {"initial_states": observed}, tuple(owners),
        ))
    elif not oracle_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "ORACLE_SEMANTIC_MISMATCH",
            {"observation_concept": obligation.verification_intent.observation_concept},
            {
                "binding_observation_concept": (
                    binding.observation_concept if binding else None
                ),
                "binding": binding.semantic_tag if binding else None,
            },
            tuple(owners),
        ))


def _lock_like_state(state: Any) -> bool:
    binding = platform_binding("lock_payload")
    symbol = normalize_symbol(getattr(state, "name", ""))
    return bool(
        any(token in symbol for token in ("locked", "secured", "held"))
        or (
            binding is not None
            and action_matches(
                binding,
                getattr(state, "entry_action", "") or "",
                getattr(state, "entry_action_def", "") or "",
            )
        )
    )


def _release_like_state(state: Any) -> bool:
    binding = platform_binding("release_payload")
    symbol = normalize_symbol(getattr(state, "name", ""))
    return bool(
        any(token in symbol for token in ("released", "unlocked", "open"))
        or (
            binding is not None
            and action_matches(
                binding,
                getattr(state, "entry_action", "") or "",
                getattr(state, "entry_action_def", "") or "",
            )
        )
    )


def _trace_locked_during_abort(
    trace: RequirementTrace,
    contract: RequirementContract,
    obligation: RequirementObligation,
    owners: list[str],
    state_machines: list[Any],
) -> None:
    """Bounded invariant: abort-active executions cannot enter release states.

    The MVP state extractor has no temporal model for an arbitrary condition
    changing while a state persists.  We therefore prove the representable
    boundary: the owner starts locked and every reachable edge into a release
    state explicitly requires the abort variable to be false.  This avoids the
    previous false positive that treated the inhibited release edge itself as
    the abort response.
    """
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "requirement_to_owner",
        "satisfying owner", owners[0] if owners else None,
        PASS if owners else FAIL, tuple(owners),
    ))
    if not owners:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "MISSING_REQUIREMENT_OWNER",
            {"satisfy": contract.req_id}, {},
        ))
        return

    machines = [sm for sm in state_machines if sm.owner_part in owners]
    initial_candidates = [
        (sm, _state(sm, sm.initial_state)) for sm in machines if sm.initial_state
    ]
    initial_candidates = [
        (sm, state) for sm, state in initial_candidates if state is not None
    ]
    safe_initial = next(
        ((sm, state) for sm, state in initial_candidates if _lock_like_state(state)),
        None,
    )
    observed_initial = tuple(
        f"{sm.name}:{state.name}" for sm, state in initial_candidates
    )
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "owner_to_initial_safe_state",
        "payload locked before any release",
        f"{safe_initial[0].name}:{safe_initial[1].name}" if safe_initial else None,
        PASS if safe_initial else FAIL, observed_initial,
    ))
    if safe_initial is None:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "INVARIANT_NOT_REPRESENTED",
            {"initial_state": "payload_locked"},
            {"initial_states": observed_initial}, tuple(owners),
        ))

    abort_variable = (
        obligation.trigger.variable if obligation.trigger is not None else None
    ) or "deliveryAbortConditionActive"
    release_edges: list[tuple[Any, Any]] = []
    unguarded_edges: list[tuple[Any, Any]] = []
    for machine in machines:
        reachable = _reachable_states(machine)
        for transition in machine.transitions:
            if transition.is_initial or transition.source not in reachable:
                continue
            target_state = _state(machine, transition.target)
            if target_state is None or not _release_like_state(target_state):
                continue
            release_edges.append((machine, transition))
            if not any(
                _guard_requires_false(guard, abort_variable)
                for guard in transition.guards
            ):
                unguarded_edges.append((machine, transition))
    evidence = tuple(
        f"{sm.name}:{transition.name or 'unnamed_transition'}"
        for sm, transition in release_edges
    )
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "release_gate_invariant",
        f"every release edge requires {abort_variable} == false",
        ", ".join(evidence) if evidence else "no reachable release edge",
        PASS if not unguarded_edges else FAIL, evidence,
    ))
    if unguarded_edges:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id,
            "GUARD_SEMANTIC_MISMATCH",
            {"all_release_edges_require": f"{abort_variable} == false"},
            {"unguarded_release_edges": [
                f"{sm.name}:{transition.name or 'unnamed_transition'}"
                for sm, transition in unguarded_edges
            ]},
            tuple(sm.name for sm, _transition in unguarded_edges),
        ))
    binding = platform_binding("lock_payload")
    invariant_ok = bool(safe_initial) and not unguarded_edges
    oracle_ok = bool(
        binding is not None
        and binding.observation_concept
        == obligation.verification_intent.observation_concept
    )
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "invariant_to_oracle",
        obligation.verification_intent.observation_concept,
        binding.observation_concept if binding else None,
        PASS if invariant_ok and oracle_ok else FAIL,
        (binding.semantic_tag,) if binding else (),
    ))
    if invariant_ok and not oracle_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "ORACLE_SEMANTIC_MISMATCH",
            {"observation_concept": obligation.verification_intent.observation_concept},
            {
                "binding_observation_concept": (
                    binding.observation_concept if binding else None
                ),
                "binding": binding.semantic_tag if binding else None,
            },
            tuple(owners),
        ))


def _guard_requires_false(guard: Any, variable: str) -> bool:
    if getattr(guard, "kind", "") == "compound":
        implications = [
            _guard_requires_false(item, variable) for item in guard.operands
        ]
        # A conjunction implies ``variable == false`` when any conjunct does;
        # a disjunction implies it only when every alternative does. Treat an
        # unknown operator conservatively so it cannot make an invariant pass.
        if getattr(guard, "compound_op", "") == "and":
            return any(implications)
        if getattr(guard, "compound_op", "") == "or":
            return bool(implications) and all(implications)
        return False
    return (
        getattr(guard, "kind", "") == "bool_false"
        and normalize_symbol(getattr(guard, "attribute", ""))
        == normalize_symbol(variable)
    )


def _startup_failure_can_reach_unsafe(
    machine: Any, start: str | None, trigger_variable: str | None
) -> bool:
    """Bounded graph check: a latched startup failure must not reach armed/airborne."""
    if not start:
        return False
    pending = [start]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        symbol = normalize_symbol(current)
        if symbol not in {"disarmed", "unarmed"} and (
            symbol in {"armed", "airborne", "flightactive"}
            or symbol.startswith("armed")
            or symbol.startswith("airborne")
        ):
            return True
        for transition in machine.transitions:
            if transition.source != current or not transition.target:
                continue
            if trigger_variable and any(
                _guard_requires_false(guard, trigger_variable)
                for guard in transition.guards
            ):
                continue
            pending.append(transition.target)
    return False


def _audit_pattern_invariants(
    trace: RequirementTrace,
    contract: RequirementContract,
    binding: PatternBinding,
    owners: list[str],
    state_machines: list[Any],
) -> None:
    """Evaluate the bounded MVP invariant rules that are not generic trace links."""
    pattern = pattern_by_id(binding.pattern_id)
    obligation = contract.obligation(binding.obligation_id)
    if pattern is None or obligation is None:
        return
    if binding.pattern_id != "StartupInhibit" or obligation.trigger is None:
        return

    machines = [sm for sm in state_machines if sm.owner_part in owners]
    candidates = [
        (sm, transition)
        for sm in machines for transition in sm.transitions
        if not transition.is_initial and _trigger_matches(obligation, transition)
    ]
    unsafe = any(
        _startup_failure_can_reach_unsafe(
            machine, transition.target, obligation.trigger.variable
        )
        for machine, transition in candidates
    )
    status = PASS if candidates and not unsafe else FAIL
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "pattern_invariant",
        "FAILED_STARTUP_CANNOT_ARM",
        "bounded failure-path reachability check" if candidates else None,
        status, tuple(machine.name for machine, _transition in candidates),
    ))
    if unsafe:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "INVARIANT_NOT_REPRESENTED",
            {"failed_startup_can_reach_armed_or_airborne": False},
            {"failed_startup_can_reach_armed_or_airborne": True},
            tuple(machine.name for machine, _transition in candidates),
        ))


def _trigger_threshold_matches(
    obligation: RequirementObligation, transition: Any
) -> tuple[bool, tuple[dict[str, Any], ...]]:
    trigger = obligation.trigger
    if trigger is None or trigger.value is None or trigger.comparator is None:
        return True, ()
    observations: list[dict[str, Any]] = []
    def collect(guard: Any) -> None:
        if getattr(guard, "kind", "") == "compound":
            for operand in getattr(guard, "operands", ()):
                collect(operand)
        elif getattr(guard, "kind", "") == "comparison":
            observations.append({
                "attribute": getattr(guard, "attribute", ""),
                "comparator": getattr(guard, "operator", ""),
                "value": getattr(guard, "threshold", None),
            })
    for guard in getattr(transition, "guards", ()):
        collect(guard)
    tolerance = max(abs(trigger.value) * 1e-9, 1e-9)
    match = any(
        item["comparator"] == trigger.comparator
        and item["value"] is not None
        and abs(float(item["value"]) - trigger.value) <= tolerance
        for item in observations
    )
    return match, tuple(observations)


def _guard_requires_abort_inactive(guard: Any) -> bool:
    """True when this guard can only pass while the abort flag is false.

    An ``or`` branch that omits the abort condition is not sufficient: the
    other branch could authorise the response while an abort is active.
    """
    kind = getattr(guard, "kind", "")
    if kind == "compound":
        operands = tuple(getattr(guard, "operands", ()))
        if getattr(guard, "compound_op", "") == "or":
            return bool(operands) and all(
                _guard_requires_abort_inactive(item) for item in operands
            )
        return any(_guard_requires_abort_inactive(item) for item in operands)
    return (
        kind == "bool_false"
        and "abort" in normalize_symbol(getattr(guard, "attribute", ""))
    )


def _trigger_qualifiers_match(
    model_text: str, obligation: RequirementObligation, transition: Any
) -> bool:
    trigger = obligation.trigger
    if trigger is None or not trigger.qualifiers:
        return True
    if "delivery_abort_inactive" not in trigger.qualifiers:
        return True
    # Structured guards are authoritative and independent of transition naming;
    # the textual scan below only recovers guard forms the extractor cannot
    # parse, and that fallback still requires a named transition statement.
    if any(
        _guard_requires_abort_inactive(guard)
        for guard in getattr(transition, "guards", ())
    ):
        return True
    transition_name = getattr(transition, "name", None)
    if not transition_name:
        return False
    match = re.search(
        rf"\btransition\s+{re.escape(transition_name)}\b(?P<body>[^;{{}}]*)",
        model_text,
        re.IGNORECASE | re.DOTALL,
    )
    body = match.group("body").lower() if match else ""
    return bool(re.search(
        r"\bnot\s+\w*(?:delivery)?\w*abort\w*|"
        r"\w*(?:delivery)?\w*abort\w*\s*==\s*false",
        body,
    ))


def _trace_obligation(
    trace: RequirementTrace,
    contract: RequirementContract,
    obligation: RequirementObligation,
    owners: list[str],
    state_machines: list[Any],
    model_text: str,
) -> None:
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "requirement_to_owner",
        "satisfying owner", owners[0] if owners else None,
        PASS if owners else FAIL, tuple(owners),
    ))
    if not owners:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "MISSING_REQUIREMENT_OWNER",
            {"satisfy": contract.req_id}, {},
        ))
        return

    machines = [sm for sm in state_machines if sm.owner_part in owners]
    candidates = [
        (sm, transition)
        for sm in machines for transition in sm.transitions
        if not transition.is_initial and _trigger_matches(obligation, transition)
    ]
    observed_triggers = tuple(
        symbol for sm in machines for transition in sm.transitions
        for symbol in _transition_signature(transition)
    )
    expected_trigger = obligation.trigger.concept if obligation.trigger else "state_invariant"
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "owner_to_trigger",
        expected_trigger,
        ", ".join(observed_triggers) or None,
        PASS if candidates else FAIL,
        tuple(sm.name for sm in machines),
    ))
    if not candidates:
        code = "MISSING_TRANSITION" if machines else "MISSING_TRIGGER"
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, code,
            {"trigger": expected_trigger},
            {"state_machines": [sm.name for sm in machines], "triggers": observed_triggers},
            owners,
        ))
        return

    # Choose the first deterministic candidate whose target action is correct;
    # otherwise retain the first candidate so the mismatch remains localized.
    chosen_sm, chosen_transition = candidates[0]
    for sm, transition in candidates:
        state = _state(sm, transition.target)
        if _response_matches(obligation, state)[0]:
            chosen_sm, chosen_transition = sm, transition
            break
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "trigger_to_transition",
        expected_trigger, chosen_transition.name or "unnamed_transition", PASS,
        (chosen_sm.name,),
    ))
    target = chosen_transition.target
    target_state = _state(chosen_sm, target)
    reachable = target in _reachable_states(chosen_sm)
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "transition_to_reachable_state",
        "reachable response state", target, PASS if reachable else FAIL,
        (chosen_sm.name, chosen_transition.name or "unnamed_transition"),
    ))
    if not reachable:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "UNREACHABLE_RESPONSE_STATE",
            {"reachable": True}, {"target": target}, (chosen_sm.name,),
        ))

    threshold_ok, threshold_observed = _trigger_threshold_matches(
        obligation, chosen_transition
    )
    qualifiers_ok = _trigger_qualifiers_match(
        model_text, obligation, chosen_transition
    )
    if not threshold_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "WRONG_THRESHOLD_OR_UNIT",
            {
                "comparator": obligation.trigger.comparator,
                "value": obligation.trigger.value,
                "unit": obligation.trigger.unit,
            },
            {"guards": threshold_observed},
            (chosen_sm.name, chosen_transition.name or "unnamed_transition"),
        ))
    if not qualifiers_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "GUARD_SEMANTIC_MISMATCH",
            {"qualifiers": list(obligation.trigger.qualifiers)},
            {"transition": chosen_transition.name},
            (chosen_sm.name, chosen_transition.name or "unnamed_transition"),
        ))

    action_ok, command_ok = _response_matches(obligation, target_state)
    response = obligation.response.concept if obligation.response else "none"
    action_name = None
    sends: tuple[str, ...] = ()
    if target_state is not None:
        action_name = target_state.entry_action_def or target_state.entry_action
        sends = tuple(command for command, _port in target_state.sends)
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "state_to_action",
        response, action_name, PASS if action_ok else FAIL,
        sends,
    ))
    binding = platform_binding(response)
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id,
        "action_to_platform_binding",
        binding.semantic_tag if binding else response,
        ", ".join(sends) if sends else action_name,
        PASS if action_ok and command_ok else FAIL,
        tuple(binding.command_aliases) if binding else (),
    ))
    if target_state is None or not action_name:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "MISSING_REQUIRED_ACTION",
            {"response": response}, {"target_state": target}, (chosen_sm.name,),
        ))
    elif not action_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "ACTION_SEMANTIC_MISMATCH",
            {"response": response}, {"action": action_name, "sends": sends},
            (chosen_sm.name, action_name),
        ))
    elif not command_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id,
            "ACTION_PLATFORM_BINDING_MISMATCH",
            {
                "response": response,
                "commands": list(binding.command_aliases) if binding else [],
            },
            {"action": action_name, "sends": sends},
            (chosen_sm.name, action_name),
        ))

    timing_ok = _timing_represented(model_text, obligation)
    if obligation.criterion and obligation.criterion.metric == "response_latency":
        trace.links.append(TraceLink(
            contract.req_id, obligation.obligation_id, "action_to_timing_constraint",
            f"{obligation.criterion.comparator} {obligation.criterion.value:g} {obligation.criterion.unit}",
            "timing constraint" if timing_ok else None,
            PASS if timing_ok else FAIL,
        ))
        if not timing_ok:
            trace.findings.append(_diag(
                contract.req_id, obligation.obligation_id, "MISSING_TIMING_CONSTRAINT",
                {"criterion": asdict(obligation.criterion)}, {},
                (chosen_sm.name,),
            ))

    oracle_ok = binding is None or (
        binding.observation_concept
        == obligation.verification_intent.observation_concept
    )
    trace.links.append(TraceLink(
        contract.req_id, obligation.obligation_id, "action_to_oracle",
        obligation.verification_intent.observation_concept,
        binding.observation_concept if binding else obligation.verification_intent.observation_concept,
        PASS if oracle_ok else FAIL,
        (binding.semantic_tag,) if binding else (),
    ))
    if not oracle_ok:
        trace.findings.append(_diag(
            contract.req_id, obligation.obligation_id, "ORACLE_SEMANTIC_MISMATCH",
            {"observation_concept": obligation.verification_intent.observation_concept},
            {
                "binding_observation_concept": binding.observation_concept,
                "binding": binding.semantic_tag,
            },
            (chosen_sm.name,),
        ))


def build_semantic_trace(
    model_text: str,
    bundle: ContractBundle | Mapping[str, Any] | None,
    *,
    pattern_bindings: Iterable[PatternBinding] = (),
    model_name: str = "model",
) -> SemanticTraceReport:
    """Build complete traces for supported contracts without deriving intent from the model."""
    from ..simulation.state_extractor import extract_state_machines
    from ..sysml.lite_model import build_lite_model

    typed = contract_bundle_from_dict(bundle)
    model = build_lite_model(model_text, model_name=model_name)
    owners = _satisfy_map(model)
    machines = extract_state_machines(model_text)
    bindings_by_req: dict[str, list[PatternBinding]] = {}
    for binding in pattern_bindings:
        bindings_by_req.setdefault(binding.req_id, []).append(binding)

    traces: list[RequirementTrace] = []
    for contract in typed.contracts:
        trace = RequirementTrace(
            req_id=contract.req_id,
            contract_status=contract.completeness,
            pattern_ids=list(dict.fromkeys(
                binding.pattern_id for binding in bindings_by_req.get(contract.req_id, ())
            )),
        )
        if contract.completeness == UNSUPPORTED:
            trace.conformance = OUT_OF_SCOPE
        elif contract.completeness == INCOMPLETE:
            trace.conformance = BLOCKED
        elif contract.kinds == ("obstacle_avoidance",) or all(
            obligation.response is not None
            and obligation.response.concept == "maintain_controlled_flight"
            for obligation in contract.obligations
        ):
            # Obstacle behavior is traced through the geometry/trajectory oracle;
            # it must not be forced into a state-machine topology.
            owner_names = owners.get(contract.req_id, [])
            trace.links.append(TraceLink(
                contract.req_id, contract.obligations[0].obligation_id,
                "requirement_to_owner", "satisfying owner",
                owner_names[0] if owner_names else None,
                PASS if owner_names else FAIL, tuple(owner_names),
            ))
            if not owner_names:
                trace.findings.append(_diag(
                    contract.req_id, contract.obligations[0].obligation_id,
                    "MISSING_REQUIREMENT_OWNER", {"satisfy": contract.req_id}, {},
                ))
            observation = contract.obligations[0].verification_intent.observation_concept
            trace.links.append(TraceLink(
                contract.req_id, contract.obligations[0].obligation_id,
                "contract_to_oracle", observation,
                observation,
                PASS,
            ))
            trace.conformance = PASS if not trace.findings else FAIL
        else:
            for obligation in contract.obligations:
                if (
                    obligation.trigger is not None
                    and obligation.trigger.concept == "power_on"
                    and obligation.response is not None
                    and obligation.response.concept == "lock_payload"
                ):
                    _trace_locked_default(
                        trace, contract, obligation,
                        owners.get(contract.req_id, []), machines,
                    )
                elif (
                    obligation.kind == "state_invariant"
                    and obligation.trigger is not None
                    and obligation.trigger.concept == "delivery_abort_condition"
                    and obligation.response is not None
                    and obligation.response.concept == "lock_payload"
                ):
                    _trace_locked_during_abort(
                        trace, contract, obligation,
                        owners.get(contract.req_id, []), machines,
                    )
                else:
                    _trace_obligation(
                        trace, contract, obligation,
                        owners.get(contract.req_id, []), machines, model_text,
                    )

        # Record selected pattern rules as auditable links; trace findings remain
        # the independent semantic verdict, avoiding pattern self-validation.
        for binding in bindings_by_req.get(contract.req_id, ()):
            pattern = pattern_by_id(binding.pattern_id)
            trace.links.append(TraceLink(
                contract.req_id, binding.obligation_id, "pattern_selection",
                binding.pattern_id, pattern.version if pattern else None,
                PASS if pattern else FAIL, (binding.selection_reason,),
            ))
            _audit_pattern_invariants(
                trace, contract, binding,
                owners.get(contract.req_id, []), machines,
            )
        if contract.completeness == READY:
            trace.conformance = PASS if not trace.findings else FAIL
        traces.append(trace)
    return SemanticTraceReport(
        TRACE_SCHEMA_VERSION,
        model_name,
        hashlib.sha256((model_text or "").encode("utf-8")).hexdigest(),
        tuple(traces),
    )
