"""Fail-closed, dependency-closed surgical repair for routed A/G failures."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .ag_assurance import FailureRoute, check_safety_pattern_conformance
from .ag_contracts import check_ag_graph
from .ag_extractor import extract_ag_graphs
from .blackboard import Blackboard, RecordType, TaskStatus
from .context_builder import ContextBuilder
from .task_session import SessionStatus, TaskSessionRegistry
from ..utils.sysml_text_utils import find_block_end, named_block_span, named_def_pattern
from ..utils.tokens import estimate_tokens
from ..simulation.surgical_refiner import (
    SurgicalAudit,
    attempt_surgical_refinement,
    build_dependency_closed_context,
)


@dataclass(frozen=True)
class AGRepairDecision:
    failure_id: str
    status: str
    reason: str
    base_model_revision: int
    base_model_digest: str
    committed_model_revision: int | None
    target_diagnostic_removed: bool
    regression_free: bool
    whole_model_fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _publish_decision(
    board: Blackboard,
    decision: AGRepairDecision,
    *,
    task_id: str | None = None,
    session_id: str | None = None,
    details: dict[str, Any] | None = None,
):
    payload = {
        **decision.to_dict(),
        "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
        "measurement_boundary": "INTERVENTION",
        **(details or {}),
    }
    return board.publish(
        RecordType.RESULT,
        "repair.decision",
        "AGRepairGate",
        payload,
        task_id=task_id,
        session_id=session_id,
    )


class _CapturingChat:
    """Archive the exact surgical-repair chat turn without provider sessions."""
    def __init__(self, llm: Any, session: Any, board: Blackboard):
        self._llm = llm
        self._session = session
        self._board = board

    def _append(self, role: str, content: str, *, token_count: int = 0) -> None:
        self._session.append(
            role,
            content,
            token_count=token_count,
            model_revision=self._board.current_revision,
            model_digest=self._board.current_model.model_digest,
            board_sequence=self._board.event_sequence,
        )

    def chat(self, prompt: str, *, system_prompt: str = "", **kwargs: Any) -> str:
        if system_prompt:
            self._append("system", system_prompt)
        self._append("user", prompt, token_count=estimate_tokens(prompt))
        response = str(self._llm.chat(
            prompt, system_prompt=system_prompt, **kwargs
        ))
        self._append(
            "assistant", response, token_count=estimate_tokens(response)
        )
        return response


def _behavior_tokens(model_text: str, behavior: str) -> set[tuple[str, str]]:
    """Named behavior content that an A/G repair is never allowed to shed."""
    span = named_block_span(model_text, "state", behavior)
    if span is None:
        return set()
    body = model_text[span[0] + 1:span[1]]
    tokens = {
        (kind, item)
        for kind, pattern in (
            ("state", r"\bstate\s+(\w+)"),
            # The name is optional; on `transition first idle ...` the old
            # pattern captured the keyword `first` as a bogus identity token,
            # so a renamed-to-unnamed rewrite read as an identity change.
            ("transition", r"\btransition\s+(?!first\b)(\w+)"),
            # Skip the optional usage label so the bare
            # `entry action deployParachute;` and the typed
            # `entry action onParachute : deployParachute;` yield the same
            # token — otherwise a repair that only respelled it reads as a loss.
            ("entry_action", r"\bentry\s+action\s+(?:\w+\s*:\s*)?(\w+)"),
            ("trigger", r"\baccept\s+(\w+)"),
        )
        for item in re.findall(pattern, body)
    }
    return tokens


_EVENT_DEF_RE = re.compile(
    r"^\s*(?:action\s+def\s+\w+\s*\{\s*\}|"
    r"(?:attribute|item)\s+def\s+\w+\s*;)\s*$",
    re.M,
)


def _ag_context_supplement(model_text: str, contract: str) -> str:
    """The A/G facts a reference-closure slice structurally cannot contain.

    `build_dependency_closed_context` closes over symbols the sliced elements
    REFERENCE. For an omission fault that is exactly the wrong direction: the
    element to restore is absent, so nothing references it and the closure cannot
    reach it. Measured on a real committed model — delete one transition and the
    slice keeps the injured state machine but loses
    `item def ParachuteDeploymentCommandSignal;`, the declaration the fix has
    to name. An agent that cannot see it either invents a signal name (an
    undeclared reference — the failure class that cost the authored mode every
    seed) or guesses from the diagnostic text.

    So the two A/G facts the diagnostic implies are added deterministically: the
    contract being realized, whose assumptions name the trigger concept and whose
    guarantee fixes the `set<Concept>` action name, and the package's declared
    event signals. Both are small and neither widens the EDIT scope — the slice is
    prompt context, and the accept gates are unchanged.
    """
    additions: list[str] = []
    match = re.search(
        rf"\brequirement def {re.escape(contract)}\s*\{{", model_text
    ) if contract else None
    if match:
        end = find_block_end(model_text, model_text.index("{", match.start()))
        if end != -1:
            additions.append(model_text[match.start():end + 1])
    signals = sorted({
        item.strip() for item in _EVENT_DEF_RE.findall(model_text)
    })
    if signals:
        additions.append("\n".join(signals))
    if not additions:
        return ""
    return (
        "\n\n// A/G context the reference closure cannot reach for an omission "
        "fault (read-only):\n" + "\n\n".join(additions)
    )


def _repair_feedback(diagnostic_code: str) -> str:
    """What the repair must and must not do — including the rules the GATE enforces.

    The feedback used to state only "repair the routed realization; these elements
    are immutable". Two rules the merge gate enforces were left unsaid, and a
    measured run died on both at once: the model added a new action definition
    (`addition_out_of_scope:action:DeployBallisticRecoveryParachute`) instead of
    restoring the conventional `set<GuaranteeConcept>` entry action inside the
    existing state. Ninth instance in this project of a gate demanding something
    the generator was never told, so the convention is rendered from
    `ag_convention` rather than restated here — one statement, one place.
    """
    from .ag_convention import DIAGNOSTIC_OBLIGATIONS

    published = {
        item.obligation_id: item.authoring_rule for item in DIAGNOSTIC_OBLIGATIONS
    }
    rule = published.get(diagnostic_code, "")
    return (
        "Repair only the routed behavior realization. Requirement definitions, "
        "contract constraints, thresholds, units, satisfy/dependency links, and "
        "unrelated elements are immutable. No whole-model fallback is allowed.\n"
        "Restore or correct EXISTING elements only: adding a new definition "
        "(action, state def, part def, attribute def) is out of scope and the "
        "merge gate rejects it. The element you need already exists or its name "
        "follows the convention below."
        + (f"\nThe convention for this defect: {rule}" if rule else "")
    )


def _diagnostic_obligations(diagnostic: Any) -> set[str]:
    """Named sub-obligations carried by an aggregate checker diagnostic."""
    provenance = getattr(diagnostic, "provenance", {}) or {}
    values = provenance.get("unsatisfied_obligations", ())
    if isinstance(values, (list, tuple)):
        result = {
            str(item).strip() for item in values if str(item).strip()
        }
        if result:
            return result
    message = str(getattr(diagnostic, "message", "") or "")
    match = re.search(r"\bunsatisfied\s*:\s*([^;]+)", message)
    return {
        item.strip() for item in match.group(1).split(",") if item.strip()
    } if match else set()


def _matching_diagnostic_obligations(
    report: Any,
    target: tuple[Any, Any, Any],
) -> set[str]:
    """Named obligations currently carried by one aggregate diagnostic."""
    code, contract, subject = target
    obligations: set[str] = set()
    for diagnostic in report.errors():
        if (
            diagnostic.code,
            diagnostic.contract,
            diagnostic.subject,
        ) == (code, contract, subject):
            obligations.update(_diagnostic_obligations(diagnostic))
    return obligations


def _extract_routed_ag_graph(
    model_text: str,
    source_requirement: str,
    *,
    revision: int | None = None,
    model_digest: str | None = None,
):
    """Select exactly the routed chain; never pool a multi-chain repair gate."""
    normalized = source_requirement.upper().replace("-", "_")
    matches = [
        graph
        for graph in extract_ag_graphs(
            model_text,
            revision=revision,
            model_digest=model_digest,
        )
        if (
            graph.system is not None
            and str(graph.system.source_requirement or "").upper().replace(
                "-", "_"
            ) == normalized
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"repair gate requires exactly one A/G graph for "
            f"{source_requirement}, found {len(matches)}"
        )
    return matches[0]


def attempt_dependency_closed_ag_repair(
    *,
    llm: Any,
    board: Blackboard,
    context_builder: ContextBuilder,
    sessions: TaskSessionRegistry,
    failure_record_id: str,
    analysis_record_id: str,
) -> AGRepairDecision:
    """Attempt one authorised model-semantic repair; never rewrites the full model."""
    failure_record = board.record(failure_record_id)
    failure = dict(failure_record.payload)
    route = failure.get("route")
    if (
        route != FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR.value
        or not failure.get("repair_authorized")
    ):
        raise ValueError("failure route does not authorize model-semantic repair")
    source_requirement = str(failure.get("source_requirement") or "")
    if not source_requirement:
        raise ValueError("repair requires source-requirement provenance")
    issue = (
        f"{source_requirement} {failure.get('contract') or ''} "
        + " ".join(str(x) for x in failure.get("affected_elements", ()))
        + f": {failure.get('message') or failure.get('diagnostic_code')}"
    )
    routed = next((
        item for item in reversed(board.records(topic="repair.routed"))
        if item.payload.get("failure_record_id") == failure_record_id
    ), None)
    task = (
        board.task(str(routed.payload["repair_task_id"]))
        if routed is not None
        else board.create_task(
            "A_G_SURGICAL_REPAIR", "RepairAgent",
            required_topics=("analysis.ag_trace", "diagnostic.failure"),
        )
    )
    context_slice = build_dependency_closed_context(
        board.current_model.model_text,
        [issue],
        allowed_req_ids={source_requirement},
    )
    if context_slice is None:
        decision = AGRepairDecision(
            str(failure.get("failure_id")), "BLOCKED",
            "dependency_closed_context_unresolved",
            board.current_revision, board.current_model.model_digest,
            None, False, False,
        )
        result = _publish_decision(
            board,
            decision,
            task_id=task.task_id,
        )
        board.transition_task(
            task.task_id,
            TaskStatus.BLOCKED,
            producer="AGRepairController",
            result_record_ids=(result.record_id,),
        )
        return decision
    board.transition_task(task.task_id, TaskStatus.ACTIVE)
    supplement = _ag_context_supplement(
        board.current_model.model_text, str(failure.get("contract") or "")
    )
    envelope = context_builder.build(
        task_id=task.task_id,
        agent_role="RepairAgent",
        objective=f"Remove {failure.get('diagnostic_code')} without semantic regression",
        allowed_operation="DEPENDENCY_CLOSED_SCOPED_MODEL_PATCH",
        model_context=context_slice.text + supplement,
        diagnostic_record_ids=(analysis_record_id, failure_record_id),
        included_record_ids=(analysis_record_id, failure_record_id),
        protected_elements=(
            source_requirement, "thresholds", "units", "requirement_defs",
            "unrelated_model_elements",
        ),
    )
    session = sessions.open(
        task_id=task.task_id,
        agent_role="RepairAgent",
        base_model_revision=task.base_model_revision,
        base_model_digest=task.base_model_digest,
        context_envelope_id=envelope.envelope_id,
        max_turns=2,
        max_tokens=120000,
    )
    before_graph = _extract_routed_ag_graph(
        board.current_model.model_text,
        source_requirement,
        revision=board.current_revision,
        model_digest=board.current_model.model_digest,
    )
    before = check_ag_graph(before_graph)
    before_ids = {(d.code, d.contract, d.subject) for d in before.errors()}
    audit = SurgicalAudit()
    try:
        outcome = attempt_surgical_refinement(
            _CapturingChat(llm, session, board),
            board.current_model.model_text,
            [issue],
            feedback=_repair_feedback(str(failure.get("diagnostic_code") or "")),
            audit=audit,
            context_slice=context_slice,
        )
    except Exception as exc:
        decision = AGRepairDecision(
            str(failure.get("failure_id")), "REJECTED",
            f"repair_execution_failed:{type(exc).__name__}",
            task.base_model_revision, task.base_model_digest,
            None, False, False,
        )
        result = _publish_decision(
            board,
            decision,
            task_id=task.task_id,
            session_id=session.session_id,
            details={
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
            },
        )
        board.transition_task(
            task.task_id,
            TaskStatus.REJECTED,
            producer="RepairAgent",
            result_record_ids=(result.record_id,),
        )
        session.close(
            SessionStatus.REJECTED,
            output_record_ids=(result.record_id,),
        )
        return decision
    if outcome is None:
        decision = AGRepairDecision(
            str(failure.get("failure_id")), "REJECTED", "surgical_merge_rejected",
            task.base_model_revision, task.base_model_digest, None, False, False,
        )
        result = _publish_decision(
            board,
            decision,
            task_id=task.task_id,
            session_id=session.session_id,
            details={
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
                # WHICH gate refused, not just that one did. Without this a
                # REJECTED decision is unactionable: an out-of-scope edit, an
                # unparseable patch and an over-strict gate all looked identical
                # in `repair_decisions.json`. Same defect as the lumped
                # PRIORITY_TOPOLOGY_INCOMPLETE diagnostic, in the repair artifact.
                "audit": {
                    "llm_invoked": audit.llm_invoked,
                    "response_count": audit.response_count,
                    "rejection_reasons": list(audit.rejection_reasons),
                    "context_mode": audit.context_mode,
                    "context_line_count": audit.context_line_count,
                    "full_model_line_count": audit.full_model_line_count,
                },
            },
        )
        board.transition_task(
            task.task_id,
            TaskStatus.REJECTED,
            producer="RepairAgent",
            result_record_ids=(result.record_id,),
        )
        session.close(
            SessionStatus.REJECTED,
            output_record_ids=(result.record_id,),
        )
        return decision

    after_graph = _extract_routed_ag_graph(
        outcome.merged_text,
        source_requirement,
    )
    after = check_ag_graph(after_graph)
    after_ids = {(d.code, d.contract, d.subject) for d in after.errors()}
    target = (
        failure.get("diagnostic_code"), failure.get("contract"), failure.get("subject")
    )
    priority_obligation = str(
        failure.get("priority_obligation") or ""
    ).strip() or None
    if priority_obligation is not None:
        # PRIORITY_TOPOLOGY_INCOMPLETE is deliberately one checker diagnostic for
        # comparability, but routing is per named obligation. A correct scoped
        # repair may remove its wiring obligation while an unrepairable response-
        # vocabulary obligation keeps the aggregate code alive. Judge the actual
        # routed target, not the container diagnostic.
        before_obligations = _matching_diagnostic_obligations(before, target)
        after_obligations = _matching_diagnostic_obligations(after, target)
        target_removed = priority_obligation not in after_obligations
        # Regression means an obligation appeared that was not present before.
        # Keeping the routed target is already reported by target_removed=False;
        # counting that same unchanged target as a new regression made the audit
        # claim two different failures for one fact.
        new_priority_obligations = after_obligations - before_obligations
        permitted_after = before_ids
    else:
        target_removed = target not in after_ids
        new_priority_obligations = set()
        permitted_after = before_ids - {target}
    # Every realizing state def named by the routed failure, identified by BEING a
    # state def rather than by ending in "Behavior". The suffix filter silently
    # exempted the one state def the emitter names differently
    # (`SafetyResponseArbitration`), so a repair could delete another state's entry
    # action from it and `behavior_preserved` stayed vacuously true. A measured run
    # did exactly that; only the pattern-conformance gate caught it, which is luck,
    # not design. Same defect class as a checker keyed on names instead of
    # declarations.
    affected_behaviors = [
        str(item) for item in failure.get("affected_elements", ())
        if named_def_pattern("state", str(item)).search(
            board.current_model.model_text
        )
    ]
    behavior_preserved = all(
        _behavior_tokens(board.current_model.model_text, behavior)
        <= _behavior_tokens(outcome.merged_text, behavior)
        for behavior in affected_behaviors
    )
    new_diagnostic_ids = after_ids - permitted_after
    regression_free = (
        not new_diagnostic_ids
        and not new_priority_obligations
        and behavior_preserved
    )
    # A SCOPED repair is judged for regression, not for finishing the chain. The
    # gate demanded `pattern verdict == PASS` outright, so a repair could clear the
    # one diagnostic it was routed and still be refused for an unrelated profile
    # failure it was never told about — measured: the routed task named only
    # REALIZATION_ACTION_MISSING, the patch removed it, and the refusal came from
    # PRIORITY_TOPOLOGY_INCOMPLETE, which was already there before the attempt. That
    # is the same asymmetry as a gate enforcing an unstated rule, and it makes a
    # bounded repair unacceptable no matter what it does. The condition is now "no
    # worse than before": breaking conformance (PASS -> FAIL) is still refused, and
    # the run verdict still reports the chain as failing, honestly.
    pattern = check_safety_pattern_conformance(after_graph, after)
    pattern_before = check_safety_pattern_conformance(before_graph, before)
    pattern_regressed = (
        pattern["verdict"] != "PASS" and pattern_before["verdict"] == "PASS"
    )
    if not target_removed or not regression_free or pattern_regressed:
        decision = AGRepairDecision(
            str(failure.get("failure_id")), "REJECTED",
            "target_not_removed_or_regression",
            task.base_model_revision, task.base_model_digest, None,
            target_removed, regression_free,
        )
        result = _publish_decision(
            board,
            decision,
            task_id=task.task_id,
            session_id=session.session_id,
            details={
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
                # which of the three gate conditions failed, and what changed.
                # "target_not_removed_or_regression" names three possibilities at
                # once; a reader of repair_decisions.json could not tell whether
                # the patch missed the target, introduced a new defect, or broke
                # pattern conformance.
                "gate": {
                    "target": [item for item in target],
                    "target_obligation": priority_obligation,
                    "target_removed": target_removed,
                    "regression_free": regression_free,
                    "behavior_preserved": behavior_preserved,
                    "pattern_verdict": pattern["verdict"],
                    "new_diagnostics": sorted(
                        [
                            *(
                                f"{code}:{contract}"
                                for code, contract, _subject in new_diagnostic_ids
                            ),
                            *(
                                f"{failure.get('diagnostic_code')}:"
                                f"{failure.get('contract')}:"
                                f"obligation={obligation}"
                                for obligation in new_priority_obligations
                            ),
                        ]
                    ),
                    "remaining_diagnostics": sorted(
                        f"{code}:{contract}" for code, contract, _subject in after_ids
                    ),
                },
            },
        )
        board.transition_task(
            task.task_id,
            TaskStatus.REJECTED,
            producer="AGRepairGate",
            result_record_ids=(result.record_id,),
        )
        session.close(
            SessionStatus.REJECTED,
            output_record_ids=(result.record_id,),
        )
        return decision

    session.assert_current(
        board.current_revision, board.current_model.model_digest
    )
    committed = board.commit_model(
        outcome.merged_text,
        base_revision=task.base_model_revision,
        base_digest=task.base_model_digest,
        producer="RepairAgent",
        task_id=task.task_id,
        session_id=session.session_id,
    )
    decision = AGRepairDecision(
        str(failure.get("failure_id")), "ACCEPTED", "all_gates_passed",
        task.base_model_revision, task.base_model_digest, committed.revision,
        True, True,
    )
    result = _publish_decision(
        board,
        decision,
        task_id=task.task_id,
        session_id=session.session_id,
        details={
            "context_envelope_digest": envelope.envelope_digest,
            "transcript_digest": session.transcript_digest,
        },
    )
    session.close(
        SessionStatus.COMPLETED,
        output_record_ids=(result.record_id,),
    )
    sessions.stale_after_commit(committed.revision, committed.model_digest)
    board.transition_task(
        task.task_id, TaskStatus.COMPLETED, producer="AGRepairGate",
        result_record_ids=(result.record_id,),
    )
    return decision
