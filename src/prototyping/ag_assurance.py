"""Increment-3 safety-pattern conformance and typed A/G failure routing.

This layer consumes only the graph extracted from committed SysML and the
runtime check report. It does not load evaluator gold and it does not assert a
formal Assume/Guarantee proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any, Dict, Iterable, Tuple

from .ag_contracts import AGDiagnostic, AGGraph, AGReport
from .ag_profile import (
    CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION,
    CODE_COMPONENT_GUARANTEE_NONATOMIC,
    CODE_CONTRACT_INCOMPLETE,
    CODE_CONTRACT_UNSUPPORTED,
    CODE_DECOMPOSITION_INSUFFICIENT,
    CODE_DECOMPOSITION_MISSING,
    CODE_DISCHARGE_EDGE_MISSING,
    CODE_GUARANTEE_MULTIPLE_OWNERS,
    CODE_GUARANTEE_NO_OWNER,
    CODE_INVARIANT_SEMANTICS_INVALID,
    CODE_INVARIANT_SEMANTICS_MISSING,
    CODE_OBSERVATION_MISSING,
    CODE_PATTERN_DECLARATION_INCONSISTENT,
    CODE_PATTERN_TOPOLOGY_INCOMPLETE,
    CODE_PRIORITY_TOPOLOGY_INCOMPLETE,
    CODE_PRIORITY_TOPOLOGY_MISSING,
    CODE_REALIZATION_ACTION_MISSING,
    CODE_REALIZATION_MISSING,
    CODE_REALIZATION_TRIGGER_MISSING,
    CODE_REALIZATION_UNREACHABLE,
    CODE_SOURCE_PROVENANCE_MISSING,
    CODE_SYSTEM_OBSERVATION_BINDING_MISSING,
    CODE_TIMING_BUDGET_EXCEEDED,
    CODE_TIMING_BUDGET_MISSING,
    CODE_UNIT_INCOMPATIBLE,
    LOCKED_UNTIL_RELEASE_PATTERN,
    STARTUP_INHIBIT_PATTERN,
    THRESHOLD_PATTERN,
    TIMED_PATTERN,
)
from .ag_convention import (
    PRIORITY_MODEL_WIRING,
    PRIORITY_OBLIGATIONS,
)


class FailureClass(str, Enum):
    CONTRACT_INCOMPLETENESS = "CONTRACT_INCOMPLETENESS"
    INTEGRATION_DECOMPOSITION_GAP = "INTEGRATION_DECOMPOSITION_GAP"
    MODEL_SEMANTIC_FAULT = "MODEL_SEMANTIC_FAULT"
    ARCHITECTURE_DESIGN_ISSUE = "ARCHITECTURE_DESIGN_ISSUE"
    VERIFIER_LIMITATION = "VERIFIER_LIMITATION"


class FailureRoute(str, Enum):
    CLARIFICATION_OR_BLOCKED = "CLARIFICATION_OR_BLOCKED"
    UPSTREAM_INTEGRATION_REPAIR = "UPSTREAM_INTEGRATION_REPAIR"
    DEPENDENCY_CLOSED_SURGICAL_REPAIR = "DEPENDENCY_CLOSED_SURGICAL_REPAIR"
    ARCHITECTURE_ANALYSIS_OR_DSE = "ARCHITECTURE_ANALYSIS_OR_DSE"
    PARTIAL_OR_UNASSIGNED = "PARTIAL_OR_UNASSIGNED"


_TIMED_FAILSAFE = TIMED_PATTERN
_THRESHOLD_TRIGGERED = THRESHOLD_PATTERN
_STARTUP_INHIBIT = STARTUP_INHIBIT_PATTERN
_LOCKED_UNTIL_RELEASE = LOCKED_UNTIL_RELEASE_PATTERN


def _token(value: Any) -> str:
    return "".join(
        character for character in str(value or "").lower()
        if character.isalnum()
    )


@dataclass(frozen=True)
class PatternCase:
    contract: str
    pattern: str
    trigger_present: bool
    reachable_response: bool
    entry_action_present: bool
    timing_criterion_present: bool
    invariant_preserved: bool
    default_safe_present: bool = True

    @property
    def status(self) -> str:
        core = all((
            self.trigger_present,
            self.reachable_response,
            self.entry_action_present,
            self.invariant_preserved,
        ))
        if not core:
            return "FAIL"
        # A triggered timed failsafe additionally requires a timing criterion.
        if self.pattern == _TIMED_FAILSAFE:
            return "PASS" if self.timing_criterion_present else "FAIL"
        # A threshold-triggered response is the same trigger→response shape with
        # no deadline, so the core is the whole obligation: requiring a timing
        # criterion here would re-impose the deadline the pattern does without,
        # and it carries none of the invariant patterns' extra duties.
        if self.pattern == _THRESHOLD_TRIGGERED:
            return "PASS"
        # A locked-until-authorised-release chain is a Boolean invariant with an
        # extra obligation the others do not carry: the power-on default state
        # must be the safe (locked) state, distinct from the guarded release
        # state — a model that powers on already released fails here even though
        # it satisfies the untimed core.
        if self.pattern == _LOCKED_UNTIL_RELEASE:
            return "PASS" if self.default_safe_present else "FAIL"
        # STARTUP_INHIBIT (and any other untimed Boolean invariant): the failed
        # self-test ↛ armed invariant has no timing obligation.
        return "PASS"


def check_safety_pattern_conformance(
    graph: AGGraph, report: AGReport
) -> Dict[str, Any]:
    """Check the bounded safety-pattern topology for the selected chain.

    The chain declares its student-selected safety pattern in the committed model (the
    ``safety_pattern=`` annotation on the system contract, the sole authority);
    that declaration is used because three patterns — one timed, two untimed —
    cannot be told apart by topology alone. For models that predate the
    declaration the pattern falls back to structural inference (a timing budget ⇒
    ``TRIGGERED_TIMED_FAILSAFE_RESPONSE``; otherwise ``STARTUP_INHIBIT``).
    """
    declared = None
    if graph.system is not None:
        declared = getattr(graph.system, "declared_pattern", None)
    by_contract = {
        item["contract"]: item for item in report.realization_links
    }
    behaviors = {item.name: item for item in graph.behaviors}
    cases = []
    for component in graph.components:
        realization = by_contract.get(component.name, {})
        reachable = list(realization.get("reachable_states") or ())
        actions = list(realization.get("response_actions") or ())
        response_states = list(realization.get("response_states") or ())
        initial_state = realization.get("initial_state")
        behavior = behaviors.get(str(realization.get("behavior") or ""))
        timed = component.timing_budget is not None
        pattern = declared or (_TIMED_FAILSAFE if timed else _STARTUP_INHIBIT)
        availability_invariant = component.timing_segment_required is False
        continuous_invariant = (
            realization.get("realization_kind") == "INVARIANT"
            and realization.get("status") == "PASS"
            and realization.get("continuous_guarantee") is True
        )
        # Default-safe: the power-on (initial) state is present and is not one of
        # the guarded response states — the locked default is genuinely distinct
        # from the released state it guards.
        if pattern == _LOCKED_UNTIL_RELEASE and (
            "PayloadLockMechanism" in component.name
        ):
            initial_action = (
                behavior.entry_actions.get(str(initial_state))
                if behavior is not None and initial_state is not None
                else None
            )
            unlock_transitions = [
                transition for transition in (behavior.transitions if behavior else ())
                if "unlocked" in str(transition.target).lower()
            ]
            authorised_unlock_only = bool(unlock_transitions) and all(
                "authorisedreleasecommandreceived" in _token(transition.trigger)
                for transition in unlock_transitions
            )
            power_loss_relocks = any(
                "unlocked" in str(transition.source).lower()
                and "locked" in str(transition.target).lower()
                and "unlocked" not in str(transition.target).lower()
                and "powerlost" in _token(transition.trigger)
                for transition in (behavior.transitions if behavior else ())
            )
            default_safe_present = (
                bool(initial_state)
                and "locked" in str(initial_state).lower()
                and "unlocked" not in str(initial_state).lower()
                and "payloadlocked" in _token(initial_action)
                and any("unlocked" in str(state).lower() for state in reachable)
                and authorised_unlock_only
                and power_loss_relocks
            )
        else:
            default_safe_present = continuous_invariant or (
                bool(initial_state) and initial_state not in response_states
            )
        pattern_invariant = (
            continuous_invariant
            or (bool(realization.get("trigger_ok")) and bool(actions))
        )
        if (
            pattern == _LOCKED_UNTIL_RELEASE
            and "ReleaseCommandGateway" in component.name
        ):
            transitions = tuple(behavior.transitions if behavior else ())
            authorised_sets = [
                transition for transition in transitions
                if transition.target == "authorisationGranted"
            ]
            guarded_authorised_set = bool(authorised_sets) and all(
                "receivedreleasecommand" in _token(transition.trigger)
                and _token(transition.guard) == "authorisationdatavalid"
                for transition in authorised_sets
            )
            latch_exits = [
                transition for transition in transitions
                if transition.source == "authorisationGranted"
            ]
            latch_clears_on_cycle_or_relock = (
                bool(latch_exits)
                and {
                    _token(transition.trigger) for transition in latch_exits
                } == {"powerlostsignal", "poweronsignal"}
                and all(
                    transition.target == "awaitingAuthorisation"
                    for transition in latch_exits
                )
            )
            pattern_invariant = (
                pattern_invariant
                and guarded_authorised_set
                and latch_clears_on_cycle_or_relock
            )
        if (
            pattern == _STARTUP_INHIBIT
            and "SelfTestStatusLatch" in component.name
        ):
            transitions = tuple(behavior.transitions if behavior else ())
            failure_latches = any(
                transition.source == "selfTesting"
                and transition.target == "startupInhibited"
                and "sensorfailurereported" in _token(transition.trigger)
                for transition in transitions
            )
            inhibited_exits = [
                transition for transition in transitions
                if transition.source == "startupInhibited"
            ]
            reset_only_on_power_cycle = bool(inhibited_exits) and all(
                transition.target == "poweredOff"
                and "powercycle" in _token(transition.trigger)
                for transition in inhibited_exits
            )
            pattern_invariant = (
                pattern_invariant
                and initial_state == "poweredOff"
                and failure_latches
                and reset_only_on_power_cycle
            )
        if (
            pattern == _LOCKED_UNTIL_RELEASE
            and "PayloadLockMechanism" in component.name
        ):
            pattern_invariant = pattern_invariant and default_safe_present
        case = PatternCase(
            contract=component.name,
            pattern=pattern,
            trigger_present=(
                continuous_invariant
                or bool(realization.get("trigger_ok"))
            ),
            # A non-timed availability invariant is established in its initial
            # state and deliberately has no activation transition or additive
            # timing segment. Other components still need a genuine path.
            reachable_response=(
                True
                if continuous_invariant
                else bool(reachable)
                if availability_invariant
                else len(reachable) >= 2
            ),
            entry_action_present=continuous_invariant or bool(actions),
            timing_criterion_present=(
                timed or component.timing_segment_required is False
            ),
            # In the bounded profile the invariant is that no response PASS is
            # possible without a reachable trigger and response entry action.
            invariant_preserved=pattern_invariant,
            default_safe_present=default_safe_present,
        )
        cases.append({**asdict(case), "status": case.status})
    checker_profile_failed = any(
        diagnostic.code in {
            CODE_PATTERN_TOPOLOGY_INCOMPLETE,
            CODE_PRIORITY_TOPOLOGY_MISSING,
            CODE_PRIORITY_TOPOLOGY_INCOMPLETE,
            CODE_INVARIANT_SEMANTICS_MISSING,
            CODE_INVARIANT_SEMANTICS_INVALID,
        }
        for diagnostic in report.diagnostics
    )
    verdict = (
        "PASS"
        if cases
        and all(c["status"] == "PASS" for c in cases)
        and not checker_profile_failed
        else "FAIL"
    )
    return {
        "schema_version": "1.0",
        "artifact_role": "INTERVENTION_PATTERN_CONFORMANCE",
        "producing_stage": "R2_PATTERN_CONFORMANCE",
        "measurement_boundary": "INTERVENTION",
        "source_model_revision": report.revision,
        "source_model_digest": report.model_digest,
        "selected_scope": report.source_requirement,
        "formal_proof": False,
        "verdict": verdict,
        "cases": cases,
    }


_CONTRACT_CODES = {
    CODE_CONTRACT_INCOMPLETE, CODE_CONTRACT_UNSUPPORTED,
    CODE_COMPONENT_GUARANTEE_NONATOMIC,
    CODE_SYSTEM_OBSERVATION_BINDING_MISSING,
    CODE_SOURCE_PROVENANCE_MISSING,
    CODE_PATTERN_DECLARATION_INCONSISTENT,
    CODE_PRIORITY_TOPOLOGY_MISSING,
    CODE_PRIORITY_TOPOLOGY_INCOMPLETE,
    CODE_INVARIANT_SEMANTICS_MISSING,
    CODE_INVARIANT_SEMANTICS_INVALID,
}
_INTEGRATION_CODES = {
    CODE_GUARANTEE_NO_OWNER, CODE_GUARANTEE_MULTIPLE_OWNERS,
    CODE_DECOMPOSITION_MISSING, CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION, CODE_DISCHARGE_EDGE_MISSING,
    CODE_DECOMPOSITION_INSUFFICIENT,
}
_MODEL_CODES = {
    CODE_REALIZATION_MISSING, CODE_REALIZATION_UNREACHABLE,
    CODE_REALIZATION_TRIGGER_MISSING, CODE_REALIZATION_ACTION_MISSING,
    CODE_PATTERN_TOPOLOGY_INCOMPLETE,
}
_DESIGN_CODES = {
    CODE_UNIT_INCOMPATIBLE, CODE_TIMING_BUDGET_EXCEEDED,
    CODE_TIMING_BUDGET_MISSING,
}
_VERIFIER_CODES = {CODE_OBSERVATION_MISSING}
_PRIORITY_INCOMPLETE = CODE_PRIORITY_TOPOLOGY_INCOMPLETE
_PRIORITY_OBLIGATION_BY_ID = {
    item.obligation_id: item for item in PRIORITY_OBLIGATIONS
}


def _classification(
    code: str,
    *,
    priority_obligation: str | None = None,
) -> Tuple[FailureClass, FailureRoute, bool]:
    if code == _PRIORITY_INCOMPLETE and priority_obligation is not None:
        obligation = _PRIORITY_OBLIGATION_BY_ID.get(priority_obligation)
        if (
            obligation is not None
            and obligation.failure_scope == PRIORITY_MODEL_WIRING
        ):
            return (
                FailureClass.MODEL_SEMANTIC_FAULT,
                FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR,
                True,
            )
        # Unknown and input/contract-valued obligations fail closed. In
        # particular, response vocabulary and precedence facts may not be guessed
        # by a local behavior repair.
        return (
            FailureClass.CONTRACT_INCOMPLETENESS,
            FailureRoute.CLARIFICATION_OR_BLOCKED,
            False,
        )
    if code in _MODEL_CODES:
        return (
            FailureClass.MODEL_SEMANTIC_FAULT,
            FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR,
            True,
        )
    if code in _INTEGRATION_CODES:
        return (
            FailureClass.INTEGRATION_DECOMPOSITION_GAP,
            FailureRoute.UPSTREAM_INTEGRATION_REPAIR,
            False,
        )
    if code in _DESIGN_CODES:
        return (
            FailureClass.ARCHITECTURE_DESIGN_ISSUE,
            FailureRoute.ARCHITECTURE_ANALYSIS_OR_DSE,
            False,
        )
    if code in _VERIFIER_CODES:
        return (
            FailureClass.VERIFIER_LIMITATION,
            FailureRoute.PARTIAL_OR_UNASSIGNED,
            False,
        )
    return (
        FailureClass.CONTRACT_INCOMPLETENESS,
        FailureRoute.CLARIFICATION_OR_BLOCKED,
        False,
    )


def _priority_obligation_ids(diagnostic: AGDiagnostic) -> Tuple[str, ...]:
    """Return the named priority failures, structured first, text as fallback.

    Current checker output carries a typed provenance list. The message fallback
    keeps routing fail-closed and useful for an archived/hand-authored diagnostic
    produced before that field existed.
    """
    values = diagnostic.provenance.get("unsatisfied_obligations", ())
    if isinstance(values, (list, tuple)):
        structured = tuple(
            dict.fromkeys(str(item).strip() for item in values if str(item).strip())
        )
        if structured:
            return structured
    match = re.search(r"\bunsatisfied\s*:\s*([^;]+)", diagnostic.message)
    if not match:
        return ()
    return tuple(dict.fromkeys(
        item.strip() for item in match.group(1).split(",") if item.strip()
    ))


def _priority_failure_message(
    diagnostic: AGDiagnostic, obligation_id: str
) -> str:
    obligation = _PRIORITY_OBLIGATION_BY_ID.get(obligation_id)
    if obligation is None:
        return (
            f"Priority topology obligation `{obligation_id}` is unsatisfied, "
            "but the obligation is unknown to the frozen convention; fail closed "
            "for clarification."
        )
    return (
        f"Priority topology obligation `{obligation_id}` is unsatisfied. "
        f"Required convention: {obligation.authoring_rule}"
    )


def route_failure_diagnostics(
    diagnostics: Iterable[AGDiagnostic],
    *,
    source_requirement: str | None,
    realization_links: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    behavior_by_contract = {
        str(item.get("contract")): item.get("behavior")
        for item in realization_links
    }
    failures = []
    for diagnostic in diagnostics:
        priority_ids = (
            _priority_obligation_ids(diagnostic)
            if diagnostic.code == _PRIORITY_INCOMPLETE
            else ()
        )
        # One aggregate checker diagnostic may contain both unrepairable
        # response vocabulary and repairable behavior wiring. Routing it as one
        # unit necessarily gives one side the wrong treatment, so publish one
        # typed failure per named obligation. An old aggregate with no names
        # remains one fail-closed contract failure.
        units: Tuple[str | None, ...] = priority_ids or (None,)
        for priority_id in units:
            failure_class, route, authorised = _classification(
                diagnostic.code,
                priority_obligation=priority_id,
            )
            affected = [item for item in (
                diagnostic.contract,
                behavior_by_contract.get(str(diagnostic.contract)),
            ) if item]
            obligation_elements = diagnostic.provenance.get(
                "obligation_affected_elements", {}
            )
            specific_affected: list[str] = []
            if (
                priority_id is not None
                and isinstance(obligation_elements, dict)
            ):
                specific_affected = [
                    str(item)
                    for item in obligation_elements.get(priority_id, ())
                    if item
                ]
                affected.extend(specific_affected)
            elif (
                diagnostic.code == "PATTERN_TOPOLOGY_INCOMPLETE"
                and isinstance(obligation_elements, dict)
            ):
                specific_affected = list(dict.fromkeys(
                    str(item)
                    for values in obligation_elements.values()
                    if isinstance(values, (list, tuple))
                    for item in values
                    if item
                ))
                affected.extend(specific_affected)
            routing_basis = (
                "NAMED_PRIORITY_OBLIGATION"
                if priority_id is not None
                else "DIAGNOSTIC_CODE"
            )
            behavior_target = (
                behavior_by_contract.get(str(diagnostic.contract))
                or (specific_affected[0] if specific_affected else None)
            )
            if authorised and priority_id is not None and not specific_affected:
                # A behavior-wiring obligation is repairable only when the
                # checker can bind it to an existing state definition. Otherwise
                # a "surgical" task would have to invent a behavior, rewrite a
                # contract, or guess which component owns the missing topology.
                failure_class = FailureClass.CONTRACT_INCOMPLETENESS
                route = FailureRoute.CLARIFICATION_OR_BLOCKED
                authorised = False
                routing_basis = (
                    "NAMED_PRIORITY_OBLIGATION_WITHOUT_BEHAVIOR_TARGET"
                )
            if (
                authorised
                and priority_id is None
                and not behavior_target
            ):
                # A route called "surgical behavior repair" needs an existing
                # state definition to replace. REALIZATION_MISSING and aggregate
                # pattern failures may have none; authorizing them would require
                # adding a definition, which the merge policy deliberately
                # forbids.
                failure_class = FailureClass.CONTRACT_INCOMPLETENESS
                route = FailureRoute.CLARIFICATION_OR_BLOCKED
                authorised = False
                routing_basis = "MODEL_FAULT_WITHOUT_BEHAVIOR_TARGET"
            compatible_signals = diagnostic.provenance.get(
                "scoped_repair_compatible_signals"
            )
            if (
                authorised
                and diagnostic.code in {
                    "REALIZATION_TRIGGER_MISSING",
                    "REALIZATION_UNREACHABLE",
                }
                and (
                    not isinstance(compatible_signals, (list, tuple))
                    or not compatible_signals
                )
            ):
                # The scoped merge may replace an existing state definition but
                # may not invent a package-level attribute definition. A missing
                # compatible signal therefore crosses the bounded edit boundary:
                # attempting it only spends the one repair budget on a patch the
                # merge gate is guaranteed to reject.
                failure_class = FailureClass.CONTRACT_INCOMPLETENESS
                route = FailureRoute.CLARIFICATION_OR_BLOCKED
                authorised = False
                routing_basis = (
                    "REALIZATION_WITHOUT_DECLARED_COMPATIBLE_SIGNAL"
                )
            affected = list(dict.fromkeys(affected))
            provenance = {
                **dict(diagnostic.provenance),
                **({
                    "aggregate_diagnostic_message": diagnostic.message,
                    "routed_priority_obligation": priority_id,
                } if priority_id is not None else {}),
            }
            failures.append({
                "failure_id": f"failure-{len(failures) + 1:04d}",
                "diagnostic_code": diagnostic.code,
                "message": (
                    _priority_failure_message(diagnostic, priority_id)
                    + (
                        " No existing behavior target was identified, so bounded "
                        "surgical repair is not authorized."
                        if routing_basis.endswith("WITHOUT_BEHAVIOR_TARGET")
                        else ""
                    )
                    if priority_id is not None else (
                        diagnostic.message
                        + (
                            " No compatible event signal is declared in the "
                            "package, so the bounded behavior-only repair cannot "
                            "introduce one."
                            if routing_basis
                            == "REALIZATION_WITHOUT_DECLARED_COMPATIBLE_SIGNAL"
                            else ""
                        )
                        + (
                            " No existing behavior target was identified, so "
                            "bounded surgical repair is not authorized."
                            if routing_basis
                            == "MODEL_FAULT_WITHOUT_BEHAVIOR_TARGET"
                            else ""
                        )
                    )
                ),
                "source_requirement": source_requirement,
                "contract": diagnostic.contract,
                "subject": diagnostic.subject,
                "classification": failure_class.value,
                "route": route.value,
                "repair_authorized": authorised,
                "affected_elements": affected,
                "priority_obligation": priority_id,
                "routing_basis": routing_basis,
                "provenance": provenance,
            })
    return {
        "schema_version": "1.0",
        "artifact_role": "INTERVENTION_FAILURE_ROUTING",
        "producing_stage": "R2_FAILURE_ROUTING",
        "measurement_boundary": "INTERVENTION",
        "failures": failures,
    }
