"""Deterministic, revision-pinned context construction for Agent tasks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .blackboard import Blackboard, RecordType


_EVALUATOR_ONLY_ROLES = {
    "EVALUATOR_GOLD",
    "BLIND_FAILURE_REVIEW_PACKET",
    "BLIND_FAILURE_LABEL",
    "FAILURE_TAXONOMY",
    "FROZEN_FAILURE_TAXONOMY",
    "POSTHOC_HUMAN_GOLD_EVALUATION",
    "POSTHOC_EVALUATION_READINESS_MANIFEST",
}
_EVALUATOR_ONLY_KEYS = {
    "gold",
    "human_gold",
    "evaluator_gold",
    "blind_label",
    "blind_labels",
}


def _contains_evaluator_only_material(value: Any) -> bool:
    """Recursively reject evaluator-only artifacts hidden inside board payloads."""
    if isinstance(value, Mapping):
        role = str(value.get("artifact_role", "")).strip().upper()
        if role in _EVALUATOR_ONLY_ROLES:
            return True
        for key, item in value.items():
            if str(key).strip().lower() in _EVALUATOR_ONLY_KEYS:
                return True
            if _contains_evaluator_only_material(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_evaluator_only_material(item) for item in value)
    return False


@dataclass(frozen=True)
class ContextEnvelope:
    envelope_id: str
    task_id: str
    agent_role: str
    objective: str
    allowed_operation: str
    model_revision: int
    model_digest: str
    source_requirements: tuple[str, ...] = ()
    model_context: str = ""
    diagnostic_record_ids: tuple[str, ...] = ()
    evidence_record_ids: tuple[str, ...] = ()
    protected_elements: tuple[str, ...] = ()
    previous_attempt_record_ids: tuple[str, ...] = ()
    included_record_ids: tuple[str, ...] = ()
    omitted_items: tuple[str, ...] = ()
    token_budget: int = 12000
    truncated: bool = False
    estimated_tokens: int = 0
    context_item_provenance: tuple[Mapping[str, Any], ...] = ()
    record_context: tuple[Mapping[str, Any], ...] = ()
    envelope_digest: str = ""

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_content:
            result.pop("model_context", None)
            result.pop("source_requirements", None)
        return result

    def render_for_prompt(self) -> str:
        requirements = "\n".join(f"- {item}" for item in self.source_requirements)
        protected = ", ".join(self.protected_elements) or "(none declared)"
        model_section = self.model_context or "(no committed model yet)"
        typed_records = json.dumps(
            list(self.record_context), ensure_ascii=False, sort_keys=True, indent=2,
            default=str,
        )
        return (
            "BLACKBOARD CONTEXT ENVELOPE (revision-pinned; not evaluator gold)\n"
            f"task_id: {self.task_id}\n"
            f"agent_role: {self.agent_role}\n"
            f"objective: {self.objective}\n"
            f"allowed_operation: {self.allowed_operation}\n"
            f"model_revision: {self.model_revision}\n"
            f"model_digest: {self.model_digest}\n"
            f"protected_elements: {protected}\n"
            "source_requirements:\n"
            f"{requirements or '- (none)'}\n"
            "typed_blackboard_records:\n"
            f"{typed_records}\n"
            "relevant_model_context:\n"
            f"```sysml\n{model_section}\n```\n"
            f"context_envelope_digest: {self.envelope_digest}"
        )


@dataclass(frozen=True)
class ContextRequirement:
    """One context category a role's task cannot be done without.

    ``failure_mode`` records what actually happens when it is removed, which is
    the point: a category whose absence RAISES is safe, and one whose absence
    silently degrades the result is the dangerous kind — the task still reports
    success on a worse answer.
    """

    category: str
    failure_mode: str  # "raises" | "empty_result" | "silent_degradation"
    evidence: str
    present: Callable[[Mapping[str, Any]], bool]


def _has_source_requirements(envelope: Mapping[str, Any]) -> bool:
    return bool(envelope.get("source_requirements"))


def _has_model_slice(envelope: Mapping[str, Any]) -> bool:
    return bool(str(envelope.get("model_context") or "").strip())


def _slice_carries_source_text(envelope: Mapping[str, Any]) -> bool:
    return "doc /*" in str(envelope.get("model_context") or "")


#: What each role's task demonstrably cannot be done without (§18-Q1).
#:
#: Derived by ABLATION, not by judgement, because the archive offered nothing to
#: learn from: across every archived revised run every task is COMPLETED except
#: ten that are BLOCKED by design (§11 routes an integration gap to BLOCKED), so
#: there is no observed context failure to generalise from. "Required" is therefore
#: given an operational meaning — remove the category and the task fails, returns
#: nothing, or silently returns a worse answer — and each entry below cites the
#: measured effect. `tests/test_option2_context_policy.py` re-runs every ablation,
#: so an entry cannot stay in the policy once it stops being load-bearing.
#:
#: Deliberately NOT listed: categories the envelope carries for provenance and
#: protection rather than for the task. VerificationAgent's `source_requirements`
#: is the clear case — the planner reads the committed slice and never touches it,
#: so calling it required would inflate the denominator of §13's
#: required-context-coverage metric with an item no task can fail on.
REQUIRED_CONTEXT_BY_ROLE: Mapping[str, Tuple[ContextRequirement, ...]] = {
    "DesignAgent": (
        ContextRequirement(
            "authoritative_source_requirements",
            "raises",
            "build_design_context without the authoritative SOURCE record raises "
            "'required typed publications are missing from context'",
            _has_source_requirements,
        ),
    ),
    "VerificationAgent": (
        ContextRequirement(
            "committed_requirement_slice",
            "empty_result",
            "plan_verification over an empty slice plans 0 of 6 requirements",
            _has_model_slice,
        ),
        ContextRequirement(
            "requirement_source_text",
            "silent_degradation",
            "with the `doc` text stripped the planner still reports 6 planned "
            "requirements, but every method collapses to `inspection` — the "
            "tier histogram goes from {behavioral:2, analysis:2, inspection:2} to "
            "{inspection:6}. It does not fail; it silently plans the wrong methods",
            _slice_carries_source_text,
        ),
    ),
}


def required_context_categories(role: str) -> Tuple[ContextRequirement, ...]:
    return REQUIRED_CONTEXT_BY_ROLE.get(str(role), ())


def context_coverage(envelope: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Which of this role's required categories the envelope actually carries.

    Returns None for a role with no declared policy: no policy means no
    denominator, and reporting 1.0 for "nothing was required" would be a vacuous
    pass — the same error as scoring an arm 0.00 for an intervention it never had.
    """
    requirements = required_context_categories(envelope.get("agent_role") or "")
    if not requirements:
        return None
    missing = [item.category for item in requirements if not item.present(envelope)]
    return {
        "required": len(requirements),
        "present": len(requirements) - len(missing),
        "missing": missing,
    }


class ContextBuilder:
    """Build context from ordinary board records only; gold has no input API."""

    def __init__(self, board: Blackboard):
        self.board = board
        self._sequence = 0
        self._envelopes: list[ContextEnvelope] = []

    @staticmethod
    def _digest(fields: Mapping[str, Any]) -> str:
        raw = json.dumps(
            dict(fields), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def build(
        self,
        *,
        task_id: str,
        agent_role: str,
        objective: str,
        allowed_operation: str,
        source_requirements: Iterable[str] = (),
        model_context: Optional[str] = None,
        protected_elements: Iterable[str] = (),
        diagnostic_record_ids: Iterable[str] = (),
        evidence_record_ids: Iterable[str] = (),
        previous_attempt_record_ids: Iterable[str] = (),
        included_record_ids: Iterable[str] = (),
        omitted_items: Iterable[str] = (),
        token_budget: int = 12000,
        truncated: bool = False,
        allow_deterministic_truncation: bool = False,
    ) -> ContextEnvelope:
        if int(token_budget) <= 0:
            raise ValueError("ContextEnvelope token_budget must be positive")
        omitted_values = tuple(str(item) for item in omitted_items)
        if truncated and not omitted_values:
            raise ValueError("truncated context must declare omitted_items")
        task = self.board.task(task_id)
        if task.agent_role != str(agent_role):
            raise ValueError(
                f"task role {task.agent_role!r} does not match {agent_role!r}"
            )
        current = self.board.current_model
        if task.base_model_revision != current.revision:
            raise ValueError("cannot build context for a stale task revision")
        included_ids = tuple(str(item) for item in included_record_ids)
        diagnostic_ids = tuple(str(item) for item in diagnostic_record_ids)
        evidence_ids = tuple(str(item) for item in evidence_record_ids)
        previous_ids = tuple(str(item) for item in previous_attempt_record_ids)
        reference_ids = tuple(dict.fromkeys(
            included_ids + diagnostic_ids + evidence_ids + previous_ids
        ))
        referenced_records = [self.board.record(item) for item in reference_ids]
        for record in referenced_records:
            if record.model_revision != current.revision:
                raise ValueError(
                    f"context record {record.record_id} is bound to stale "
                    f"revision {record.model_revision}"
                )
            if (
                record.topic.lower().startswith("gold.")
                or _contains_evaluator_only_material(record.payload)
            ):
                raise ValueError("evaluator gold cannot enter a ContextEnvelope")
        included_topics = {record.topic for record in referenced_records}
        missing_topics = set(task.required_topics) - included_topics
        if missing_topics:
            raise ValueError(
                "required typed publications are missing from context: "
                + ", ".join(sorted(missing_topics))
            )
        source_values = tuple(str(item) for item in source_requirements)
        if source_values:
            board_values: list[str] = []
            for record in referenced_records:
                if (
                    record.record_type is RecordType.SOURCE
                    and record.topic == "requirements.authoritative"
                ):
                    board_values.extend(
                        str(item)
                        for item in record.payload.get("requirements", ())
                    )
            if source_values != tuple(board_values):
                raise ValueError(
                    "source requirements must be reproduced exactly from "
                    "referenced Blackboard SOURCE records"
                )
        self._sequence += 1
        envelope_id = f"context-{self._sequence:06d}"
        content = current.model_text if model_context is None else str(model_context)
        provenance = tuple(
            {
                "kind": "blackboard_record",
                "record_id": record.record_id,
                "topic": record.topic,
                "record_type": record.record_type.value,
                "payload_digest": record.payload_digest,
                "model_revision": record.model_revision,
                "model_digest": record.model_digest,
            }
            for record in referenced_records
        ) + ({
            "kind": "committed_sysml_slice",
            "model_revision": current.revision,
            "model_digest": current.model_digest,
            "content_digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        },)
        record_context = tuple({
            "record_id": record.record_id,
            "record_type": record.record_type.value,
            "topic": record.topic,
            "producer": record.producer,
            "payload": dict(record.payload),
            "payload_digest": record.payload_digest,
        } for record in referenced_records)
        base = {
            "envelope_id": envelope_id,
            "task_id": task_id,
            "agent_role": str(agent_role),
            "objective": str(objective),
            "allowed_operation": str(allowed_operation),
            "model_revision": current.revision,
            "model_digest": current.model_digest,
            "source_requirements": source_values,
            "model_context": content,
            "diagnostic_record_ids": diagnostic_ids,
            "evidence_record_ids": evidence_ids,
            "protected_elements": tuple(str(item) for item in protected_elements),
            "previous_attempt_record_ids": previous_ids,
            "included_record_ids": included_ids,
            "omitted_items": omitted_values,
            "token_budget": int(token_budget),
            "truncated": bool(truncated),
            "estimated_tokens": 0,
            "context_item_provenance": provenance,
            "record_context": record_context,
        }
        envelope = ContextEnvelope(**base, envelope_digest=self._digest(base))
        estimated_tokens = max(1, len(envelope.render_for_prompt()) // 4)
        if estimated_tokens > envelope.token_budget:
            if allow_deterministic_truncation and content:
                # Required typed records and source requirements have priority.
                # Only the tail of the model slice is omitted, deterministically.
                overflow_chars = (estimated_tokens - envelope.token_budget) * 4
                keep = max(0, len(content) - overflow_chars - 256)
                omitted_values = omitted_values + ("model_context:tail",)
                base.update({
                    "model_context": content[:keep],
                    "omitted_items": omitted_values,
                    "truncated": True,
                    "context_item_provenance": provenance + ({
                        "kind": "truncation",
                        "policy": "REQUIRED_RECORDS_THEN_MODEL_PREFIX",
                        "omitted": "model_context:tail",
                    },),
                })
                envelope = ContextEnvelope(
                    **base, envelope_digest=self._digest(base)
                )
                estimated_tokens = max(
                    1, len(envelope.render_for_prompt()) // 4
                )
            if estimated_tokens > envelope.token_budget:
                raise ValueError(
                    "ContextEnvelope exceeds its token budget "
                    f"({estimated_tokens}>{envelope.token_budget}); build a "
                    "deterministically truncated envelope with omitted_items"
                )
        base["estimated_tokens"] = estimated_tokens
        envelope = ContextEnvelope(**base, envelope_digest=self._digest(base))
        self._envelopes.append(envelope)
        self.board.publish(
            RecordType.HISTORY,
            "context.created",
            "ContextBuilder",
            envelope.to_dict(),
            task_id=task_id,
        )
        return envelope

    def build_design_context(
        self,
        *,
        task_id: str,
        system_name: str,
        source_record_ids: Iterable[str],
        token_budget: int = 12000,
    ) -> ContextEnvelope:
        source_ids = tuple(str(item) for item in source_record_ids)
        requirements: list[str] = []
        for record_id in source_ids:
            record = self.board.record(record_id)
            if (
                record.topic.lower().startswith("gold.")
                or _contains_evaluator_only_material(record.payload)
            ):
                raise ValueError("evaluator gold cannot enter a ContextEnvelope")
            if record.topic != "requirements.authoritative":
                raise ValueError(
                    "design requirements must come from the authoritative "
                    "typed source publication"
                )
            requirements.extend(
                str(item) for item in record.payload.get("requirements", ())
            )
        return self.build(
            task_id=task_id,
            agent_role="DesignAgent",
            objective=f"Generate the initial SysML v2 model for {system_name}",
            allowed_operation="CREATE_INITIAL_MODEL",
            source_requirements=requirements,
            protected_elements=("requirement_ids", "requirement_source_text"),
            included_record_ids=source_ids,
            token_budget=token_budget,
        )

    def build_verification_context(
        self,
        *,
        task_id: str,
        source_record_ids: Iterable[str],
        token_budget: int = 12000,
    ) -> ContextEnvelope:
        """Context for the second handoff (DesignAgent -> VerificationAgent).

        The relevant model context is only the committed stakeholder requirement
        defs (the model design produced, sliced to what verification plans), so the
        envelope stays small and every requirement is present. The authoritative
        requirements are reproduced from a current-revision SOURCE record.
        """
        from .verification_planning import requirement_def_slice

        source_ids = tuple(str(item) for item in source_record_ids)
        requirements: list[str] = []
        for record_id in source_ids:
            record = self.board.record(record_id)
            if (
                record.topic.lower().startswith("gold.")
                or _contains_evaluator_only_material(record.payload)
            ):
                raise ValueError("evaluator gold cannot enter a ContextEnvelope")
            if record.topic != "requirements.authoritative":
                raise ValueError(
                    "verification requirements must come from the authoritative "
                    "typed source publication"
                )
            requirements.extend(
                str(item) for item in record.payload.get("requirements", ())
            )
        model_slice = requirement_def_slice(self.board.current_model.model_text)
        return self.build(
            task_id=task_id,
            agent_role="VerificationAgent",
            objective="Plan an IADT verification method for each committed requirement",
            allowed_operation="PRODUCE_VERIFICATION_PLAN",
            source_requirements=requirements,
            model_context=model_slice,
            protected_elements=("requirement_ids", "requirement_source_text"),
            included_record_ids=source_ids,
            token_budget=token_budget,
        )

    def build_repair_context(
        self,
        *,
        task_id: str,
        issues: list[str],
        repair_packet: Optional[Mapping[str, Any]] = None,
        allowed_req_ids: Optional[set[str]] = None,
        token_budget: int = 12000,
    ) -> ContextEnvelope:
        from ..agents.surgical_refiner import build_dependency_closed_context

        context = build_dependency_closed_context(
            self.board.current_model.model_text,
            issues,
            repair_packet=repair_packet,
            allowed_req_ids=allowed_req_ids,
        )
        if context is None:
            raise ValueError("dependency-closed repair context could not be built")
        return self.build(
            task_id=task_id,
            agent_role="RepairAgent",
            objective="Repair only the reported semantic/model defects",
            allowed_operation="SCOPED_MODEL_PATCH",
            model_context=context.text,
            protected_elements=("requirement_defs", "unrelated_model_elements"),
            token_budget=token_budget,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "envelopes": [
                item.to_dict(include_content=True) for item in self._envelopes
            ],
        }
