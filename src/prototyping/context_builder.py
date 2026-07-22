"""Deterministic, revision-pinned context construction for Agent tasks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

from .blackboard import Blackboard, RecordType


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
                or str(record.payload.get("artifact_role", "")).upper()
                == "EVALUATOR_GOLD"
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
                or str(record.payload.get("artifact_role", "")).upper()
                == "EVALUATOR_GOLD"
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
