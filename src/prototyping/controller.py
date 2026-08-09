"""Event-driven Blackboard control component (Hayes-Roth control separation).

In a blackboard architecture the *control* is a first-class component, separate
from the knowledge sources and the shared workspace: it inspects the board and
opportunistically activates whichever knowledge source can now contribute. This
module makes that control explicit. Each downstream knowledge source registers the
typed topics it needs; the Controller activates a source exactly when the board
satisfies its preconditions and it has not already run, to a fixpoint (activating
one source may satisfy the next).

The pay-off is that adding a knowledge source becomes *registration*, not new
bespoke wiring in the orchestrator — control lives in one place and is recorded on
the board (a ``control.activation`` record per firing) as evidence.

Bounded scope (design §15): the Controller drives post-design board-mediated
knowledge sources, including verification planning and R2 semantic assurance.
The DesignAgent generation step stays orchestrator-driven because it is
interleaved with LLM generation and is not a pure board-triggered activation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from .blackboard import Blackboard, RecordType


@dataclass(frozen=True)
class KnowledgeSource:
    """A downstream knowledge source with typed board preconditions."""

    name: str
    agent_role: str
    precondition_topics: Tuple[str, ...]
    activate: Callable[[], Any]
    output_topics: Tuple[str, ...] = ()


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
        # A stale publication must never activate work against a newer model
        # revision. Upstream facts that remain valid are explicitly reaffirmed
        # on the current revision by the orchestrator/knowledge source.
        #
        # Note what this costs, because it is not obvious and it bounds the
        # architecture. A source whose preconditions were published under
        # *different* revisions can never activate, since only one revision is
        # ever visible here. Chains therefore have to be either revision-flat or
        # strictly sequential, each source depending on the topic its immediate
        # predecessor just published. The generation pipeline is the second kind,
        # and it stays viable only because its board never commits a model — see
        # the invariant recorded at the `pipeline-runtime` board's construction.
        # If that ever changes, `run(require_all=True)` reports it as a stall and
        # `_hidden_topics` says which topics the revision change took away, so
        # the diagnosis does not depend on recognising how the commit was
        # written.
        return {
            record.topic
            for record in self.board.records()
            if record.model_revision == self.board.current_revision
            and record.model_digest == self.board.current_model.model_digest
        }

    def _topics_at_any_revision(self) -> set:
        """Every topic ever published on this board, ignoring model revision."""
        return {record.topic for record in self.board.records()}

    def _hidden_topics(self) -> set:
        """Topics that exist on the board but are invisible at the current revision.

        A non-empty result means a model was committed onto this board and the
        facts published before it were not reaffirmed afterwards. That is the
        failure mode `_published_topics` creates, and it is detected here from
        the board's own contents rather than by looking for a `commit_model`
        call, so it holds however the commit was spelled.
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

    def run(self, *, require_all: bool = False) -> List[dict]:
        """Activate every activatable source to a fixpoint.

        Deterministic: sources fire in registration order; each fires at most once.
        A firing publishes a typed ``control.activation`` record so the control
        decisions are auditable on the board, then the agenda is re-evaluated so a
        source unblocked by that firing runs on the next pass.

        ``require_all`` closes a gap the output-topic check leaves open. That
        check catches a source that ran and failed to publish what it declared;
        it cannot catch a source that never became activatable at all, because
        reaching a fixpoint early is indistinguishable from finishing. The
        agenda records the difference in ``activated``/``preconditions_met``, but
        a caller that does not read those fields would take a short run for a
        complete one. Pass ``require_all=True`` where every registered source is
        meant to fire — the generation pipeline does — and a stalled chain raises
        instead of returning a partial result. It stays off by default so that
        callers registering a subset, and the unit tests that assert an unmet
        precondition makes ``run()`` a no-op, keep their semantics.
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
                            "agent_role": source.agent_role,
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
                        "agent_role": source.agent_role,
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
                        "agent_role": source.agent_role,
                        "precondition_topics": list(source.precondition_topics),
                        "output_topics": list(source.output_topics),
                        "activation_index": activation_index,
                        "status": "COMPLETED",
                    },
                )
                entry = {
                    "knowledge_source": source.name,
                    "agent_role": source.agent_role,
                    "precondition_topics": list(source.precondition_topics),
                    "output_topics": list(source.output_topics),
                    "activation_record_id": record.record_id,
                    "status": "COMPLETED",
                }
                self.activation_log.append(entry)
                agenda.append({**entry, "result": result})
                progressed = True
        if require_all:
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
                    # Distinguish a topic nobody ever produced from one that was
                    # produced and then hidden by a model commit on this board.
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
                    "agent_role": source.agent_role,
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
