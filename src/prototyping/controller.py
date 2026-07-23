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

Bounded scope (design §15): the Controller drives the post-design board-mediated
handoffs (verification planning today; extensible). The DesignAgent generation
step stays orchestrator-driven because it is interleaved with LLM generation and
is not a pure board-triggered activation.
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
        return {record.topic for record in self.board.records()}

    def activatable(self) -> List[KnowledgeSource]:
        """Registered sources not yet run whose every precondition is on the board."""
        published = self._published_topics()
        return [
            source
            for source in self._sources
            if source.name not in self._activated
            and all(topic in published for topic in source.precondition_topics)
        ]

    def run(self) -> List[dict]:
        """Activate every activatable source to a fixpoint.

        Deterministic: sources fire in registration order; each fires at most once.
        A firing publishes a typed ``control.activation`` record so the control
        decisions are auditable on the board, then the agenda is re-evaluated so a
        source unblocked by that firing runs on the next pass.
        """
        agenda: List[dict] = []
        progressed = True
        while progressed:
            progressed = False
            for source in self.activatable():
                result = source.activate()
                self._activated.add(source.name)
                record = self.board.publish(
                    RecordType.CONTROL,
                    "control.activation",
                    "BlackboardController",
                    {
                        "knowledge_source": source.name,
                        "agent_role": source.agent_role,
                        "precondition_topics": list(source.precondition_topics),
                        "activation_index": len(self.activation_log),
                    },
                )
                entry = {
                    "knowledge_source": source.name,
                    "agent_role": source.agent_role,
                    "precondition_topics": list(source.precondition_topics),
                    "activation_record_id": record.record_id,
                }
                self.activation_log.append(entry)
                agenda.append({**entry, "result": result})
                progressed = True
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
