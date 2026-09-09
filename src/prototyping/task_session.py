"""Application-owned short-lived task sessions for revised Option 2."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional


class SessionError(RuntimeError):
    pass


class SessionStatus(str, Enum):
    OPEN = "OPEN"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    EXPIRED = "EXPIRED"
    STALE = "STALE"


_TERMINAL = {
    SessionStatus.COMPLETED,
    SessionStatus.REJECTED,
    SessionStatus.BLOCKED,
    SessionStatus.EXPIRED,
    SessionStatus.STALE,
}


@dataclass(frozen=True)
class SessionMessage:
    sequence: int
    role: str
    content: str
    token_count: int = 0
    model_revision: Optional[int] = None
    model_digest: Optional[str] = None
    board_sequence: Optional[int] = None


@dataclass
class TaskSession:
    session_id: str
    task_id: str
    agent_role: str
    base_model_revision: int
    base_model_digest: str
    context_envelope_ids: list[str] = field(default_factory=list)
    max_turns: int = 12
    max_tokens: int = 600000
    status: SessionStatus = SessionStatus.OPEN
    messages: list[SessionMessage] = field(default_factory=list)
    output_record_ids: list[str] = field(default_factory=list)
    rebased_from_session_id: Optional[str] = None

    @property
    def used_tokens(self) -> int:
        return sum(item.token_count for item in self.messages)

    @property
    def assistant_turns(self) -> int:
        return sum(item.role == "assistant" for item in self.messages)

    def append(
        self,
        role: str,
        content: str,
        *,
        token_count: int = 0,
        model_revision: Optional[int] = None,
        model_digest: Optional[str] = None,
        board_sequence: Optional[int] = None,
    ) -> None:
        if self.status is not SessionStatus.OPEN:
            raise SessionError(f"session is not open: {self.status.value}")
        if self.assistant_turns >= self.max_turns:
            self.status = SessionStatus.EXPIRED
            raise SessionError("session turn budget exhausted")
        if self.used_tokens + int(token_count) > self.max_tokens:
            self.status = SessionStatus.EXPIRED
            raise SessionError("session token budget exhausted")
        self.messages.append(SessionMessage(
            sequence=len(self.messages) + 1,
            role=str(role),
            content=str(content),
            token_count=max(int(token_count), 0),
            model_revision=(
                int(model_revision) if model_revision is not None else None
            ),
            model_digest=(
                str(model_digest) if model_digest is not None else None
            ),
            board_sequence=(
                int(board_sequence) if board_sequence is not None else None
            ),
        ))

    def assert_current(self, revision: int, digest: str) -> None:
        if (
            int(revision) != self.base_model_revision
            or str(digest) != self.base_model_digest
        ):
            self.status = SessionStatus.STALE
            raise SessionError("session base model revision is stale")

    def close(
        self,
        status: SessionStatus,
        *,
        output_record_ids: Iterable[str] = (),
    ) -> None:
        target = SessionStatus(status)
        if target is SessionStatus.OPEN:
            raise SessionError("close status cannot be OPEN")
        if self.status in _TERMINAL and self.status is not target:
            raise SessionError(f"session already terminal: {self.status.value}")
        self.status = target
        self.output_record_ids.extend(str(item) for item in output_record_ids)

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "base_model_revision": self.base_model_revision,
            "base_model_digest": self.base_model_digest,
            "context_envelope_ids": list(self.context_envelope_ids),
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "used_tokens": self.used_tokens,
            "assistant_turns": self.assistant_turns,
            "status": self.status.value,
            "output_record_ids": list(self.output_record_ids),
            "rebased_from_session_id": self.rebased_from_session_id,
        }
        if include_messages:
            result["messages"] = [asdict(item) for item in self.messages]
        return result


class TaskSessionRegistry:
    def __init__(self):
        self._sequence = 0
        self._sessions: dict[str, TaskSession] = {}
        self._task_roles: dict[str, str] = {}

    def open(
        self,
        *,
        task_id: str,
        agent_role: str,
        base_model_revision: int,
        base_model_digest: str,
        context_envelope_id: str,
        max_turns: int = 12,
        max_tokens: int = 600000,
        rebased_from_session_id: Optional[str] = None,
    ) -> TaskSession:
        task_key = str(task_id)
        role = str(agent_role)
        existing_role = self._task_roles.get(task_key)
        if existing_role is not None and existing_role != role:
            raise SessionError(
                f"task {task_key} belongs to {existing_role}, not {role}"
            )
        for existing in self._sessions.values():
            if (
                existing.task_id == task_id
                and existing.status is SessionStatus.OPEN
            ):
                raise SessionError(f"task already has an open session: {task_id}")
        self._sequence += 1
        session = TaskSession(
            session_id=f"session-{self._sequence:06d}",
            task_id=str(task_id),
            agent_role=str(agent_role),
            base_model_revision=int(base_model_revision),
            base_model_digest=str(base_model_digest),
            context_envelope_ids=[str(context_envelope_id)],
            max_turns=int(max_turns),
            max_tokens=int(max_tokens),
            rebased_from_session_id=rebased_from_session_id,
        )
        self._sessions[session.session_id] = session
        self._task_roles[task_key] = role
        return session

    def rebase(
        self,
        session_id: str,
        *,
        base_model_revision: int,
        base_model_digest: str,
        context_envelope_id: str,
    ) -> TaskSession:
        previous = self.get(session_id)
        if previous.status is not SessionStatus.STALE:
            raise SessionError(
                "only a STALE session can be rebased into a fresh session"
            )
        return self.open(
            task_id=previous.task_id,
            agent_role=previous.agent_role,
            base_model_revision=base_model_revision,
            base_model_digest=base_model_digest,
            context_envelope_id=context_envelope_id,
            max_turns=previous.max_turns,
            max_tokens=previous.max_tokens,
            rebased_from_session_id=previous.session_id,
        )

    def get(self, session_id: str) -> TaskSession:
        return self._sessions[session_id]

    def stale_after_commit(
        self,
        revision: int,
        digest: str,
        *,
        except_session_id: Optional[str] = None,
    ) -> tuple[str, ...]:
        stale = []
        for session in self._sessions.values():
            if session.status is not SessionStatus.OPEN:
                continue
            if session.session_id == except_session_id:
                continue
            if (
                session.base_model_revision != int(revision)
                or session.base_model_digest != str(digest)
            ):
                session.status = SessionStatus.STALE
                stale.append(session.session_id)
        return tuple(stale)

    def snapshot(self, *, include_messages: bool = False) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "sessions": [
                self._sessions[key].to_dict(include_messages=include_messages)
                for key in sorted(self._sessions)
            ],
        }
