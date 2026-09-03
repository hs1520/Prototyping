from __future__ import annotations

import pytest

from src.prototyping.blackboard import Blackboard, RecordType
from src.prototyping.controller import (
    AgentRole,
    BlackboardController,
    KnowledgeSource,
)


def _board() -> Blackboard:
    return Blackboard("Drone")


def _publish_topic(board: Blackboard, topic: str) -> None:
    board.publish(RecordType.CONTROL, topic, "Test", {})


def test_source_needs_preconditions():
    board = _board()
    fired: list[str] = []
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="verification", agent_role="VerificationAgent",
        precondition_topics=("agent.design.result",),
        activate=lambda: fired.append("verification"),
    ))

    assert controller.activatable() == []
    assert controller.run(allow_partial=True) == []
    assert fired == []

    _publish_topic(board, "agent.design.result")
    assert [s.name for s in controller.activatable()] == ["verification"]
    agenda = controller.run()
    assert fired == ["verification"]
    assert [a["knowledge_source"] for a in agenda] == ["verification"]


def test_source_fires_once():
    board = _board()
    _publish_topic(board, "start")
    fired: list[str] = []
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="ks", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=("start",),
        activate=lambda: fired.append("ks"),
    ))
    controller.run()
    controller.run()
    assert fired == ["ks"]


def test_firing_unblocks_next_source():
    board = _board()
    _publish_topic(board, "start")
    order: list[str] = []

    def fire_a():
        order.append("a")
        _publish_topic(board, "mid")

    def fire_b():
        order.append("b")

    controller = BlackboardController(board)
    # register B first to prove activation follows preconditions, not registration
    controller.register(KnowledgeSource(
        name="b", agent_role=AgentRole.DESIGN, precondition_topics=("mid",), activate=fire_b,
    ))
    controller.register(KnowledgeSource(
        name="a", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=("start",), activate=fire_a,
    ))
    controller.run()
    assert order == ["a", "b"]


def test_activation_recorded_on_board():
    board = _board()
    _publish_topic(board, "agent.design.result")
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="verification", agent_role="VerificationAgent",
        precondition_topics=("agent.design.result",), activate=lambda: "plan",
    ))
    agenda = controller.run()
    activations = board.records(topic="control.activation")
    assert len(activations) == 1
    assert activations[0].payload["knowledge_source"] == "verification"
    assert activations[0].producer == "BlackboardController"
    assert agenda[0]["result"] == "plan"


def test_duplicate_registration_rejected():
    controller = BlackboardController(_board())
    ks = KnowledgeSource(
        name="dup", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=(), activate=lambda: None,
    )
    controller.register(ks)
    with pytest.raises(ValueError, match="already registered"):
        controller.register(ks)


def test_agenda_reports_without_firing():
    board = _board()
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="verification", agent_role="VerificationAgent",
        precondition_topics=("agent.design.result",), activate=lambda: None,
    ))
    agenda = controller.agenda()
    assert agenda["artifact_role"] == "BLACKBOARD_CONTROL_AGENDA"
    ks = agenda["registered_knowledge_sources"][0]
    assert ks["activated"] is False
    assert ks["preconditions_met"] is False


def test_stale_topic_cannot_activate():
    board = _board()
    _publish_topic(board, "ready")
    board.commit_model(
        "package Current {}",
        base_revision=board.current_revision,
        base_digest=board.current_model.model_digest,
        producer="test",
    )
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="current_only",
        agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("ready",),
        activate=lambda: None,
    ))
    assert controller.activatable() == []


def test_output_topic_enforced():
    board = _board()
    _publish_topic(board, "ready")
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="broken",
        agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("ready",),
        activate=lambda: None,
        output_topics=("result.required",),
    ))
    with pytest.raises(RuntimeError, match="did not publish"):
        controller.run()
    activation = board.records(topic="control.activation")
    assert len(activation) == 1
    assert activation[0].payload["status"] == "FAILED"
    assert controller.agenda()["activations"][0]["status"] == "FAILED"


def test_allow_partial_returns_short():
    # Completeness is the default, so a caller registering more sources than the
    # board satisfies has to say so.
    board = _board()
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="never",
        agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("absent",),
        activate=lambda: None,
    ))
    assert controller.run(allow_partial=True) == []


def test_stalled_chain_errors():
    # The output-topic check cannot see this: `blocked` never ran, so nothing
    # failed to publish -- control reached a fixpoint early.
    board = _board()
    _publish_topic(board, "start")
    fired: list[str] = []
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="runs", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=("start",),
        activate=lambda: fired.append("runs"),
    ))
    controller.register(KnowledgeSource(
        name="blocked", agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("never.published",),
        activate=lambda: fired.append("blocked"),
    ))
    with pytest.raises(RuntimeError, match="never activated"):
        controller.run()
    assert fired == ["runs"]


def test_stall_error_names_topic():
    board = _board()
    _publish_topic(board, "start")
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="blocked", agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("start", "missing.topic"),
        activate=lambda: None,
    ))
    with pytest.raises(RuntimeError) as excinfo:
        controller.run()
    message = str(excinfo.value)
    assert "blocked awaiting missing.topic" in message
    assert "start" not in message.split("awaiting", 1)[1]


def test_commit_hides_earlier_topics():
    """Detected from the board rather than from source text.

    `_published_topics` only sees the current revision, so committing a model hides
    topics published before it unless something reaffirms them. The commit here goes
    through an alias that no textual search for `commit_model` would find, and the
    controller still diagnoses it.
    """
    board = _board()
    _publish_topic(board, "start")
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="downstream", agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("start",),
        activate=lambda: None,
    ))
    assert [s.name for s in controller.activatable()] == ["downstream"]

    alias = board
    alias.commit_model(
        "part def P;",
        base_revision=board.current_revision,
        base_digest=board.current_model.model_digest,
        producer="Test",
    )

    assert "start" in controller._topics_at_any_revision()
    assert "start" in controller._hidden_topics()
    assert controller.activatable() == []

    with pytest.raises(RuntimeError) as excinfo:
        controller.run()
    message = str(excinfo.value)
    assert "downstream awaiting start" in message
    assert "hidden by a model revision" in message


def test_unproduced_topic_not_hidden():
    board = _board()
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="blocked", agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("never.produced",),
        activate=lambda: None,
    ))
    with pytest.raises(RuntimeError) as excinfo:
        controller.run()
    message = str(excinfo.value)
    assert "blocked awaiting never.produced" in message
    assert "hidden by a model revision" not in message


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown agent role"):
        KnowledgeSource(
            name="typo", agent_role="AssurenceAgent",
            precondition_topics=(), activate=lambda: None,
        )


def test_role_string_normalised():
    source = KnowledgeSource(
        name="ok", agent_role="VerificationAgent",
        precondition_topics=(), activate=lambda: None,
    )
    assert source.agent_role is AgentRole.VERIFICATION


def test_process_fact_survives_commit():
    """Staleness is a property the publisher declares.

    Every record used to expire with the revision, so a chain spanning a commit
    could not proceed. The record now says whether it asserts something about the
    model.
    """
    board = _board()
    board.publish(
        RecordType.RESULT, "phase.done", "Test", {}, revision_bound=False,
    )
    board.publish(RecordType.ANALYSIS, "analysis.of.model", "Test", {})
    controller = BlackboardController(board)

    board.commit_model(
        "part def P;",
        base_revision=board.current_revision,
        base_digest=board.current_model.model_digest,
        producer="Test",
    )

    published = controller._published_topics()
    assert "phase.done" in published
    assert "analysis.of.model" not in published


def test_chain_spans_commit():
    board = _board()
    board.publish(
        RecordType.SOURCE, "start", "Test", {}, revision_bound=False,
    )
    fired: list[str] = []
    controller = BlackboardController(board)

    def commit_then_publish():
        board.commit_model(
            "part def P;",
            base_revision=board.current_revision,
            base_digest=board.current_model.model_digest,
            producer="Test",
        )
        board.publish(
            RecordType.RESULT, "mid", "Test", {}, revision_bound=False,
        )
        fired.append("first")

    controller.register(KnowledgeSource(
        name="first", agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("start",), activate=commit_then_publish,
        output_topics=("mid",),
    ))
    controller.register(KnowledgeSource(
        name="second", agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("start", "mid"),
        activate=lambda: fired.append("second"),
    ))
    controller.run()
    assert fired == ["first", "second"]
