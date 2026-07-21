"""Typed, revisioned shared workspace for the revised Option 2 MVP.

The committed SysML text is the semantic authority.  Other records are
revision-bound coordination, analysis, or evidence state and cannot mutate the
model except through :meth:`Blackboard.commit_model`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


def text_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def payload_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str,
    )
    return text_digest(encoded)


class BlackboardError(RuntimeError):
    pass


class StaleRevisionError(BlackboardError):
    pass


class InvalidTaskTransition(BlackboardError):
    pass


class RecordType(str, Enum):
    SOURCE = "SOURCE"
    MODEL = "MODEL"
    ANALYSIS = "ANALYSIS"
    CONTROL = "CONTROL"
    EVIDENCE = "EVIDENCE"
    HISTORY = "HISTORY"
    RESULT = "RESULT"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


_TASK_TRANSITIONS = {
    TaskStatus.PENDING: {TaskStatus.ACTIVE, TaskStatus.BLOCKED, TaskStatus.REJECTED},
    TaskStatus.ACTIVE: {
        TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.REJECTED
    },
    TaskStatus.COMPLETED: set(),
    TaskStatus.BLOCKED: set(),
    TaskStatus.REJECTED: set(),
}


@dataclass(frozen=True)
class ModelRevision:
    revision: int
    model_text: str
    model_digest: str
    parent_revision: Optional[int]
    producer: str
    task_id: Optional[str] = None

    def audit_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_text:
            result.pop("model_text", None)
        return result


@dataclass(frozen=True)
class BlackboardRecord:
    record_id: str
    sequence: int
    record_type: RecordType
    topic: str
    producer: str
    payload: Mapping[str, Any]
    payload_digest: str
    model_revision: int
    model_digest: str
    task_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["record_type"] = self.record_type.value
        result["payload"] = dict(self.payload)
        return result


@dataclass
class BlackboardTask:
    task_id: str
    kind: str
    agent_role: str
    base_model_revision: int
    base_model_digest: str
    required_topics: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.PENDING
    result_record_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


class Blackboard:
    """In-memory MVP board with deterministic IDs and serializable snapshots."""

    def __init__(self, workspace_id: str, initial_model_text: str = ""):
        self.workspace_id = str(workspace_id)
        self._record_sequence = 0
        self._task_sequence = 0
        initial_text = str(initial_model_text)
        self._revisions: list[ModelRevision] = [
            ModelRevision(
                revision=0,
                model_text=initial_text,
                model_digest=text_digest(initial_text),
                parent_revision=None,
                producer="workspace",
            )
        ]
        self._records: list[BlackboardRecord] = []
        self._tasks: dict[str, BlackboardTask] = {}

    @property
    def current_model(self) -> ModelRevision:
        return self._revisions[-1]

    @property
    def current_revision(self) -> int:
        return self.current_model.revision

    def publish(
        self,
        record_type: RecordType,
        topic: str,
        producer: str,
        payload: Mapping[str, Any],
        *,
        model_revision: Optional[int] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        allow_stale: bool = False,
    ) -> BlackboardRecord:
        revision = self.current_revision if model_revision is None else int(model_revision)
        if revision < 0 or revision >= len(self._revisions):
            raise StaleRevisionError(f"unknown model revision: {revision}")
        if not allow_stale and revision != self.current_revision:
            raise StaleRevisionError(
                f"record revision {revision} is stale; current={self.current_revision}"
            )
        model = self._revisions[revision]
        self._record_sequence += 1
        clean_payload = dict(payload)
        record = BlackboardRecord(
            record_id=f"record-{self._record_sequence:06d}",
            sequence=self._record_sequence,
            record_type=RecordType(record_type),
            topic=str(topic),
            producer=str(producer),
            payload=clean_payload,
            payload_digest=payload_digest(clean_payload),
            model_revision=revision,
            model_digest=model.model_digest,
            task_id=task_id,
            session_id=session_id,
        )
        self._records.append(record)
        return record

    def create_task(
        self,
        kind: str,
        agent_role: str,
        *,
        required_topics: Iterable[str] = (),
    ) -> BlackboardTask:
        self._task_sequence += 1
        task = BlackboardTask(
            task_id=f"task-{self._task_sequence:06d}",
            kind=str(kind),
            agent_role=str(agent_role),
            base_model_revision=self.current_revision,
            base_model_digest=self.current_model.model_digest,
            required_topics=tuple(str(item) for item in required_topics),
        )
        self._tasks[task.task_id] = task
        self.publish(
            RecordType.CONTROL,
            "task.created",
            "controller",
            task.to_dict(),
            task_id=task.task_id,
        )
        return task

    def transition_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        producer: str = "controller",
        result_record_ids: Iterable[str] = (),
    ) -> BlackboardTask:
        task = self._tasks[task_id]
        target = TaskStatus(status)
        if target not in _TASK_TRANSITIONS[task.status]:
            raise InvalidTaskTransition(f"{task.status.value} -> {target.value}")
        task.status = target
        task.result_record_ids.extend(str(item) for item in result_record_ids)
        self.publish(
            RecordType.CONTROL,
            "task.transition",
            producer,
            task.to_dict(),
            task_id=task.task_id,
        )
        return task

    def rebase_task(
        self,
        task_id: str,
        *,
        producer: str = "controller",
    ) -> BlackboardTask:
        task = self._tasks[task_id]
        if task.status is not TaskStatus.ACTIVE:
            raise InvalidTaskTransition(
                f"only ACTIVE tasks can rebase; status={task.status.value}"
            )
        task.base_model_revision = self.current_revision
        task.base_model_digest = self.current_model.model_digest
        self.publish(
            RecordType.CONTROL,
            "task.rebased",
            producer,
            task.to_dict(),
            task_id=task.task_id,
        )
        return task

    def commit_model(
        self,
        model_text: str,
        *,
        base_revision: int,
        producer: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ModelRevision:
        if int(base_revision) != self.current_revision:
            raise StaleRevisionError(
                f"patch base revision {base_revision} is stale; "
                f"current={self.current_revision}"
            )
        text = str(model_text)
        revision = ModelRevision(
            revision=self.current_revision + 1,
            model_text=text,
            model_digest=text_digest(text),
            parent_revision=self.current_revision,
            producer=str(producer),
            task_id=task_id,
        )
        self._revisions.append(revision)
        self.publish(
            RecordType.MODEL,
            "model.committed",
            producer,
            revision.audit_dict(),
            model_revision=revision.revision,
            task_id=task_id,
            session_id=session_id,
        )
        return revision

    def records(
        self,
        *,
        record_type: Optional[RecordType] = None,
        topic: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> tuple[BlackboardRecord, ...]:
        result = self._records
        if record_type is not None:
            value = RecordType(record_type)
            result = [item for item in result if item.record_type is value]
        if topic is not None:
            result = [item for item in result if item.topic == topic]
        if task_id is not None:
            result = [item for item in result if item.task_id == task_id]
        return tuple(result)

    def task(self, task_id: str) -> BlackboardTask:
        return self._tasks[task_id]

    def record(self, record_id: str) -> BlackboardRecord:
        for record in self._records:
            if record.record_id == record_id:
                return record
        raise KeyError(record_id)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "workspace_id": self.workspace_id,
            "semantic_authority": "COMMITTED_SYSML_MODEL",
            "current_model": self.current_model.audit_dict(),
            "model_revisions": [item.audit_dict() for item in self._revisions],
            "tasks": [self._tasks[key].to_dict() for key in sorted(self._tasks)],
            "records": [item.to_dict() for item in self._records],
        }
