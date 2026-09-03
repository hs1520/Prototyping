"""Event-driven Blackboard control component (Hayes-Roth control separation).

In a blackboard architecture the control is a first-class component, separate from
the knowledge sources and the shared workspace: it inspects the board and
opportunistically activates whichever knowledge source can now contribute. Each
downstream knowledge source registers the typed topics it needs; the Controller
activates a source when the board satisfies its preconditions and it has not
already run, to a fixpoint (activating one source may satisfy the next).

Adding a knowledge source is then registration rather than new orchestrator wiring,
and each firing is recorded on the board as a ``control.activation`` record.

Bounded scope (design §15): the Controller drives post-design board-mediated
knowledge sources, including verification planning and R2 semantic assurance. The
DesignAgent generation step stays orchestrator-driven because it interleaves with
LLM generation rather than being a pure board-triggered activation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Tuple

from .blackboard import Blackboard, RecordType


class AgentRole(str, Enum):
    """Who a knowledge source acts as.

    A closed set rather than free text: the role is written onto every
    ``control.activation`` record and so into the run artefacts, where a typo in a free
    string is discoverable only by reading the evidence. A name that is not a member
    fails here, at registration.
    """

    ORCHESTRATOR = "Orchestrator"
    DESIGN = "DesignAgent"
    VERIFICATION = "VerificationAgent"
    ASSURANCE = "AssuranceAgent"
    AG_PLANNING = "AGPlanningAgent"
    REPAIR = "RepairAgent"


@dataclass(frozen=True)
class KnowledgeSource:
    """A downstream knowledge source with typed board preconditions."""

    name: str
    agent_role: AgentRole
    precondition_topics: Tuple[str, ...]
    activate: Callable[[], Any]
    output_topics: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Accept the string form so existing call sites and archived fixtures
        # keep working, but reject anything outside the closed set before it
        # reaches the artefacts.
        try:
            role = AgentRole(self.agent_role)
        except ValueError:
            raise ValueError(
                f"knowledge source {self.name!r} declares unknown agent role "
                f"{self.agent_role!r}; expected one of "
                + ", ".join(sorted(item.value for item in AgentRole))
            ) from None
        object.__setattr__(self, "agent_role", role)


class BlackboardController:
    """Opportunistic, event-driven activation of registered knowledge sources."""

    def __init__(self, board: Blackboard):
        self.board = board
        self._sources: List[KnowledgeSource] = []
        self._activated: set[str] = set()
        self.activation_log: List[dict] = []

    def register(self, source: KnowledgeSource) -> "BlackboardController":
        if any(existing.name == source.name for existing in self._sources):
            raise ValueError(
                f"knowledge source {source.name!r} is already registered"
            )
        self._sources.append(source)
        return self

    def _published_topics(self) -> set:
        """Topics available as preconditions right now.

        A stale publication does not activate work against a newer model revision, but only
        records asserting something about the model go stale. A process fact such as "the
        requirements phase finished" survives a later revision, and expiring it would break
        any chain spanning a commit: a source needing two topics published under different
        revisions could never activate, leaving opportunistic activation unavailable in the
        pipelines that revise their model.

        The record carries the distinction (``revision_bound``), so the publisher makes it
        rather than this method. Revision-bound records expire on the revision they were
        published against; unbound ones persist.
        """
        current_revision = self.board.current_revision
        current_digest = self.board.current_model.model_digest
        return {
            record.topic
            for record in self.board.records()
            if not getattr(record, "revision_bound", True)
            or (
                record.model_revision == current_revision
                and record.model_digest == current_digest
            )
        }

    def _topics_at_any_revision(self) -> set:
        return {record.topic for record in self.board.records()}

    def _hidden_topics(self) -> set:
        """Topics that exist on the board but are invisible at the current revision.

        A non-empty result means a model was committed onto this board and the facts
        published before it were not reaffirmed. Detected from the board's own contents
        rather than from a `commit_model` call, so it holds however the commit was
        spelled.
        """
        return self._topics_at_any_revision() - self._published_topics()

    def activatable(self) -> List[KnowledgeSource]:
        """Registered sources not yet run whose every precondition is on the board."""
        published = self._published_topics()
        return [
            source
            for source in self._sources
            if source.name not in self._activated
            and all(topic in published for topic in source.precondition_topics)
        ]

    def run(self, *, allow_partial: bool = False) -> List[dict]:
        """Activate every activatable source to a fixpoint.

        Deterministic: sources fire in registration order, each at most once. A firing
        publishes a typed ``control.activation`` record so the control decisions are
        auditable on the board, then the agenda is re-evaluated so a source unblocked by
        that firing runs on the next pass.

        Completeness is the default. The output-topic check catches a source that ran and
        failed to publish what it declared, but not one that never became activatable,
        since an early fixpoint is indistinguishable from finishing; leaving that to the
        caller reported partial runs as successes. ``run`` raises when a registered source
        never fires, naming it and the topics it is still waiting on.
        ``allow_partial=True`` is for callers that deliberately register more sources than
        the board will satisfy, and has to be stated rather than got by omission.
        """
        agenda: List[dict] = []
        progressed = True
        while progressed:
            progressed = False
            for source in self.activatable():
                activation_index = len(self.activation_log)
                try:
                    result = source.activate()
                    missing_outputs = (
                        set(source.output_topics) - self._published_topics()
                    )
                    if missing_outputs:
                        raise RuntimeError(
                            f"knowledge source {source.name!r} did not publish "
                            "declared output topic(s): "
                            + ", ".join(sorted(missing_outputs))
                        )
                except Exception as exc:
                    self._activated.add(source.name)
                    record = self.board.publish(
                        RecordType.CONTROL,
                        "control.activation",
                        "BlackboardController",
                        {
                            "knowledge_source": source.name,
                            "agent_role": source.agent_role.value,
                            "precondition_topics": list(
                                source.precondition_topics
                            ),
                            "output_topics": list(source.output_topics),
                            "activation_index": activation_index,
                            "status": "FAILED",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    entry = {
                        "knowledge_source": source.name,
                        "agent_role": source.agent_role.value,
                        "precondition_topics": list(
                            source.precondition_topics
                        ),
                        "output_topics": list(source.output_topics),
                        "activation_record_id": record.record_id,
                        "status": "FAILED",
                    }
                    self.activation_log.append(entry)
                    raise
                self._activated.add(source.name)
                record = self.board.publish(
                    RecordType.CONTROL,
                    "control.activation",
                    "BlackboardController",
                    {
                        "knowledge_source": source.name,
                        "agent_role": source.agent_role.value,
                        "precondition_topics": list(source.precondition_topics),
                        "output_topics": list(source.output_topics),
                        "activation_index": activation_index,
                        "status": "COMPLETED",
                    },
                )
                entry = {
                    "knowledge_source": source.name,
                    "agent_role": source.agent_role.value,
                    "precondition_topics": list(source.precondition_topics),
                    "output_topics": list(source.output_topics),
                    "activation_record_id": record.record_id,
                    "status": "COMPLETED",
                }
                self.activation_log.append(entry)
                agenda.append({**entry, "result": result})
                progressed = True
        if not allow_partial:
            stalled = {
                source.name
                for source in self._sources
                if source.name not in self._activated
            }
            if stalled:
                published = self._published_topics()
                hidden = self._hidden_topics()
                parts = []
                for source in self._sources:
                    if source.name not in stalled:
                        continue
                    awaited = sorted(
                        topic
                        for topic in source.precondition_topics
                        if topic not in published
                    )
                    # Distinguish a topic nobody produced from one that was produced
                    # and then hidden by a model commit on this board.
                    described = ", ".join(
                        f"{topic} (published earlier, hidden by a model "
                        "revision on this board and never reaffirmed)"
                        if topic in hidden
                        else topic
                        for topic in awaited
                    )
                    parts.append(f"{source.name} awaiting {described}")
                raise RuntimeError(
                    "control reached a fixpoint with knowledge sources that "
                    f"never activated: {'; '.join(parts)}"
                )
        return agenda

    def agenda(self) -> dict:
        """The auditable control record: what was registered and what fired."""
        published = self._published_topics()
        return {
            "schema_version": "1.0",
            "artifact_role": "BLACKBOARD_CONTROL_AGENDA",
            "producing_stage": "EVENT_DRIVEN_CONTROL",
            "measurement_boundary": "COORDINATION",
            "registered_knowledge_sources": [
                {
                    "name": source.name,
                    "agent_role": source.agent_role.value,
                    "precondition_topics": list(source.precondition_topics),
                    "output_topics": list(source.output_topics),
                    "activated": source.name in self._activated,
                    "preconditions_met": all(
                        topic in published
                        for topic in source.precondition_topics
                    ),
                }
                for source in self._sources
            ],
            "activations": list(self.activation_log),
        }
