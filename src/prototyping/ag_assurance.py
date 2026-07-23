"""Increment-3 safety-pattern conformance and typed A/G failure routing.

This layer consumes only the graph extracted from committed SysML and the
runtime check report. It does not load evaluator gold and it does not assert a
formal Assume/Guarantee proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Tuple

from .ag_contracts import AGDiagnostic, AGGraph, AGReport


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


_TIMED_FAILSAFE = "TRIGGERED_TIMED_FAILSAFE_RESPONSE"
_STARTUP_INHIBIT = "STARTUP_INHIBIT"
_LOCKED_UNTIL_RELEASE = "LOCKED_UNTIL_AUTHORISED_RELEASE"


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
            default_safe_present = bool(initial_state) and (
                initial_state not in response_states
            )
        pattern_invariant = (
            bool(realization.get("trigger_ok")) and bool(actions)
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
            trigger_present=bool(realization.get("trigger_ok")),
            # A non-timed availability invariant is established in its initial
            # state and deliberately has no activation transition or additive
            # timing segment. Other components still need a genuine path.
            reachable_response=(
                bool(reachable)
                if availability_invariant
                else len(reachable) >= 2
            ),
            entry_action_present=bool(actions),
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
            "PATTERN_TOPOLOGY_INCOMPLETE",
            "PRIORITY_TOPOLOGY_MISSING",
            "PRIORITY_TOPOLOGY_INCOMPLETE",
            "INVARIANT_SEMANTICS_MISSING",
            "INVARIANT_SEMANTICS_INVALID",
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
    "CONTRACT_INCOMPLETE", "CONTRACT_UNSUPPORTED",
    "SOURCE_PROVENANCE_MISSING",
    "PATTERN_DECLARATION_INCONSISTENT",
    "PRIORITY_TOPOLOGY_MISSING",
    "PRIORITY_TOPOLOGY_INCOMPLETE",
    "INVARIANT_SEMANTICS_MISSING",
    "INVARIANT_SEMANTICS_INVALID",
}
_INTEGRATION_CODES = {
    "GUARANTEE_NO_OWNER", "GUARANTEE_MULTIPLE_OWNERS",
    "DECOMPOSITION_MISSING", "ASSUMPTION_UNDISCHARGED",
    "CIRCULAR_ASSUMPTION", "DISCHARGE_EDGE_MISSING",
    "DECOMPOSITION_INSUFFICIENT",
}
_MODEL_CODES = {
    "REALIZATION_MISSING", "REALIZATION_UNREACHABLE",
    "REALIZATION_TRIGGER_MISSING", "REALIZATION_ACTION_MISSING",
    "PATTERN_NONCONFORMANT", "PATTERN_TOPOLOGY_INCOMPLETE",
}
_DESIGN_CODES = {
    "UNIT_INCOMPATIBLE", "TIMING_BUDGET_EXCEEDED", "TIMING_BUDGET_MISSING",
}
_VERIFIER_CODES = {"OBSERVATION_MISSING"}


def _classification(code: str) -> Tuple[FailureClass, FailureRoute, bool]:
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
    for index, diagnostic in enumerate(
        diagnostics, 1
    ):
        failure_class, route, authorised = _classification(diagnostic.code)
        affected = [item for item in (
            diagnostic.contract,
            behavior_by_contract.get(str(diagnostic.contract)),
        ) if item]
        failures.append({
            "failure_id": f"failure-{index:04d}",
            "diagnostic_code": diagnostic.code,
            "message": diagnostic.message,
            "source_requirement": source_requirement,
            "contract": diagnostic.contract,
            "subject": diagnostic.subject,
            "classification": failure_class.value,
            "route": route.value,
            "repair_authorized": authorised,
            "affected_elements": affected,
            "provenance": dict(diagnostic.provenance),
        })
    return {
        "schema_version": "1.0",
        "artifact_role": "INTERVENTION_FAILURE_ROUTING",
        "producing_stage": "R2_FAILURE_ROUTING",
        "measurement_boundary": "INTERVENTION",
        "failures": failures,
    }
