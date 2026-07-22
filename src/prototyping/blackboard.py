"""Typed, revisioned shared workspace for the revised Option 2 MVP.

The committed SysML text is the semantic authority.  Other records are
revision-bound coordination, analysis, or evidence state and cannot mutate the
model except through :meth:`Blackboard.commit_model`.
"""
from __future__ import annotations

import hashlib
import json
import re
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


class ProtectedModelElementError(BlackboardError):
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
        self._protected_requirement_defs: dict[str, str] = {}
        self._check_and_extend_protection(initial_text)

    @staticmethod
    def _requirement_identities(model_text: str) -> dict[str, str]:
        """Exact requirement-definition bodies, including thresholds and units."""
        from ..utils.sysml_text_utils import find_block_end

        result: dict[str, str] = {}
        pattern = re.compile(r"\brequirement\s+def\s+([A-Za-z_]\w*)")
        for match in pattern.finditer(model_text or ""):
            brace = (model_text or "").find("{", match.end())
            if brace == -1:
                continue
            end = find_block_end(model_text, brace)
            if end != -1:
                result[match.group(1)] = model_text[match.start():end + 1].strip()
        return result

    @staticmethod
    def _element_index(model_text: str) -> list[dict[str, Any]]:
        """Deterministic current-revision definition index for scoped queries."""
        from ..utils.sysml_text_utils import find_block_end

        index: list[dict[str, Any]] = []
        pattern = re.compile(
            r"\b(part|requirement|state|action|verification|constraint|port|item)"
            r"\s+def\s+([A-Za-z_]\w*)"
        )
        for match in pattern.finditer(model_text or ""):
            brace = (model_text or "").find("{", match.end())
            semi = (model_text or "").find(";", match.end())
            if brace != -1 and (semi == -1 or brace < semi):
                end = find_block_end(model_text, brace)
                if end == -1:
                    continue
                end += 1
            elif semi != -1:
                end = semi + 1
            else:
                continue
            source = model_text[match.start():end]
            index.append({
                "element_id": match.group(2),
                "kind": f"{match.group(1)} def",
                "span": {"start": match.start(), "end": end},
                "source_digest": text_digest(source),
            })
        return index

    def _check_and_extend_protection(self, model_text: str) -> None:
        identities = self._requirement_identities(model_text)
        changed = {
            name for name, original in self._protected_requirement_defs.items()
            if identities.get(name) != original
        }
        if changed:
            raise ProtectedModelElementError(
                "commit would change or remove protected requirement/contract "
                "definitions: " + ", ".join(sorted(changed))
            )
        # Once a stakeholder requirement or approved A/G requirement definition
        # appears in a committed revision it becomes immutable. Exact block
        # identity protects source text, comparators, thresholds, units, assume/
        # require constraints, and acceptance criteria together.
        for name, body in identities.items():
            self._protected_requirement_defs.setdefault(name, body)

    @staticmethod
    def _normalise_source_text(value: str) -> str:
        return " ".join(str(value).split())

    def _validate_authoritative_sources(self, model_text: str) -> None:
        """Require exact stakeholder text inside the committed requirement def."""
        identities = self._requirement_identities(model_text)
        for record in self._records:
            if (
                record.record_type is not RecordType.SOURCE
                or record.topic != "requirements.authoritative"
            ):
                continue
            for raw in record.payload.get("requirements", ()):
                match = re.match(
                    r"\s*(REQ[-_][A-Za-z0-9]+[-_]\d+)\s*:\s*(.*)",
                    str(raw), re.DOTALL,
                )
                if not match:
                    raise ProtectedModelElementError(
                        f"authoritative requirement has no supported ID: {raw!r}"
                    )
                req_id = match.group(1).upper().replace("-", "_")
                block = identities.get(req_id)
                if block is None:
                    raise ProtectedModelElementError(
                        f"committed model is missing authoritative {req_id}"
                    )
                doc = re.search(r"\bdoc\s*/\*(.*?)\*/", block, re.DOTALL)
                expected = self._normalise_source_text(match.group(2))
                actual = self._normalise_source_text(doc.group(1) if doc else "")
                if actual != expected:
                    raise ProtectedModelElementError(
                        f"committed {req_id} source text differs from the "
                        "authoritative Blackboard publication"
                    )

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
        base_digest: str,
        producer: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ModelRevision:
        if int(base_revision) != self.current_revision:
            raise StaleRevisionError(
                f"patch base revision {base_revision} is stale; "
                f"current={self.current_revision}"
            )
        if str(base_digest) != self.current_model.model_digest:
            raise StaleRevisionError(
                "patch base digest does not match the committed model at "
                f"revision {self.current_revision}"
            )
        text = str(model_text)
        self._validate_authoritative_sources(text)
        self._check_and_extend_protection(text)
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
            "model_element_index": self._element_index(
                self.current_model.model_text
            ),
            "protected_requirement_digests": {
                name: text_digest(body)
                for name, body in sorted(self._protected_requirement_defs.items())
            },
            "tasks": [self._tasks[key].to_dict() for key in sorted(self._tasks)],
            "records": [item.to_dict() for item in self._records],
        }
