"""Fail-closed, dependency-closed surgical repair for routed A/G failures."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .ag_assurance import FailureRoute, check_safety_pattern_conformance
from .ag_contracts import check_ag_graph
from .ag_extractor import extract_ag_graph
from .blackboard import Blackboard, RecordType, TaskStatus
from .context_builder import ContextBuilder
from .task_session import SessionStatus, TaskSessionRegistry
from ..agents.surgical_refiner import (
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
        return dict(self.__dict__)


class _CapturingChat:
    """Archive the exact surgical-repair chat turn without provider sessions."""
    def __init__(self, llm: Any, session: Any):
        self._llm = llm
        self._session = session

    def chat(self, prompt: str, *, system_prompt: str = "", **kwargs: Any) -> str:
        if system_prompt:
            self._session.append("system", system_prompt)
        self._session.append("user", prompt, token_count=max(1, len(prompt) // 4))
        response = str(self._llm.chat(
            prompt, system_prompt=system_prompt, **kwargs
        ))
        self._session.append(
            "assistant", response, token_count=max(1, len(response) // 4)
        )
        return response


def _behavior_tokens(model_text: str, behavior: str) -> set[tuple[str, str]]:
    """Named behavior content that an A/G repair is never allowed to shed."""
    from ..utils.sysml_text_utils import find_block_end

    match = re.search(rf"\bstate\s+def\s+{re.escape(behavior)}\s*\{{", model_text)
    if match is None:
        return set()
    brace = model_text.find("{", match.start())
    end = find_block_end(model_text, brace)
    body = model_text[brace + 1:end] if end != -1 else ""
    tokens = {
        (kind, item)
        for kind, pattern in (
            ("state", r"\bstate\s+(\w+)"),
            ("transition", r"\btransition\s+(\w+)"),
            ("entry_action", r"\bentry\s+action\s+(\w+)"),
            ("trigger", r"\baccept\s+(\w+)"),
        )
        for item in re.findall(pattern, body)
    }
    return tokens


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
    context_slice = build_dependency_closed_context(
        board.current_model.model_text,
        [issue],
        allowed_req_ids={source_requirement},
    )
    if context_slice is None:
        return AGRepairDecision(
            str(failure.get("failure_id")), "BLOCKED",
            "dependency_closed_context_unresolved",
            board.current_revision, board.current_model.model_digest,
            None, False, False,
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
    board.transition_task(task.task_id, TaskStatus.ACTIVE)
    envelope = context_builder.build(
        task_id=task.task_id,
        agent_role="RepairAgent",
        objective=f"Remove {failure.get('diagnostic_code')} without semantic regression",
        allowed_operation="DEPENDENCY_CLOSED_SCOPED_MODEL_PATCH",
        model_context=context_slice.text,
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
        max_tokens=30000,
    )
    before_graph = extract_ag_graph(
        board.current_model.model_text,
        revision=board.current_revision,
        model_digest=board.current_model.model_digest,
    )
    before = check_ag_graph(before_graph)
    before_ids = {(d.code, d.contract, d.subject) for d in before.errors()}
    audit = SurgicalAudit()
    outcome = attempt_surgical_refinement(
        _CapturingChat(llm, session),
        board.current_model.model_text,
        [issue],
        feedback=(
            "Repair only the routed behavior realization. Requirement definitions, "
            "contract constraints, thresholds, units, satisfy/dependency links, and "
            "unrelated elements are immutable. No whole-model fallback is allowed."
        ),
        audit=audit,
        context_slice=context_slice,
    )
    if outcome is None:
        board.transition_task(task.task_id, TaskStatus.REJECTED, producer="RepairAgent")
        session.close(SessionStatus.REJECTED)
        return AGRepairDecision(
            str(failure.get("failure_id")), "REJECTED", "surgical_merge_rejected",
            task.base_model_revision, task.base_model_digest, None, False, False,
        )

    after_graph = extract_ag_graph(outcome.merged_text)
    after = check_ag_graph(after_graph)
    after_ids = {(d.code, d.contract, d.subject) for d in after.errors()}
    target = (
        failure.get("diagnostic_code"), failure.get("contract"), failure.get("subject")
    )
    target_removed = target not in after_ids
    affected_behaviors = [
        str(item) for item in failure.get("affected_elements", ())
        if str(item).endswith("Behavior")
    ]
    behavior_preserved = all(
        _behavior_tokens(board.current_model.model_text, behavior)
        <= _behavior_tokens(outcome.merged_text, behavior)
        for behavior in affected_behaviors
    )
    regression_free = (
        not (after_ids - (before_ids - {target})) and behavior_preserved
    )
    pattern = check_safety_pattern_conformance(after_graph, after)
    if not target_removed or not regression_free or pattern["verdict"] != "PASS":
        board.transition_task(task.task_id, TaskStatus.REJECTED, producer="AGRepairGate")
        session.close(SessionStatus.REJECTED)
        return AGRepairDecision(
            str(failure.get("failure_id")), "REJECTED",
            "target_not_removed_or_regression",
            task.base_model_revision, task.base_model_digest, None,
            target_removed, regression_free,
        )

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
    session.close(SessionStatus.COMPLETED)
    result = board.publish(
        RecordType.RESULT, "repair.decision", "AGRepairGate",
        {
            "failure_id": failure.get("failure_id"),
            "status": "ACCEPTED",
            "target_diagnostic_removed": True,
            "regression_free": True,
            "whole_model_fallback_used": False,
            "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
            "measurement_boundary": "INTERVENTION",
            "context_envelope_digest": envelope.envelope_digest,
            "transcript_digest": session.transcript_digest,
        },
        task_id=task.task_id,
        session_id=session.session_id,
    )
    board.transition_task(
        task.task_id, TaskStatus.COMPLETED, producer="AGRepairGate",
        result_record_ids=(result.record_id,),
    )
    return AGRepairDecision(
        str(failure.get("failure_id")), "ACCEPTED", "all_gates_passed",
        task.base_model_revision, task.base_model_digest, committed.revision,
        True, True,
    )
