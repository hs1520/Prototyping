"""Event-driven Blackboard control component (Hayes-Roth control separation).

The Controller activates a knowledge source exactly when the board satisfies its
typed preconditions and it has not already run — control is explicit and
opportunistic, not implicit in a caller's fixed sequence. Registering a source
with its preconditions is enough to schedule it, and each firing is recorded on
the board as a ``control.activation``.
"""
from __future__ import annotations

import pytest

from src.prototyping.blackboard import Blackboard, RecordType
from src.prototyping.controller import BlackboardController, KnowledgeSource


def _board() -> Blackboard:
    return Blackboard("Drone")


def _publish_topic(board: Blackboard, topic: str) -> None:
    board.publish(RecordType.CONTROL, topic, "Test", {})


def test_a_source_activates_only_when_its_preconditions_are_on_the_board():
    board = _board()
    fired: list[str] = []
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="verification", agent_role="VerificationAgent",
        precondition_topics=("agent.design.result",),
        activate=lambda: fired.append("verification"),
    ))

    # precondition absent -> not activatable, run() is a no-op
    assert controller.activatable() == []
    assert controller.run() == []
    assert fired == []

    # publish the precondition -> the source becomes activatable and fires
    _publish_topic(board, "agent.design.result")
    assert [s.name for s in controller.activatable()] == ["verification"]
    agenda = controller.run()
    assert fired == ["verification"]
    assert [a["knowledge_source"] for a in agenda] == ["verification"]


def test_each_source_fires_at_most_once():
    board = _board()
    _publish_topic(board, "start")
    fired: list[str] = []
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="ks", agent_role="Role", precondition_topics=("start",),
        activate=lambda: fired.append("ks"),
    ))
    controller.run()
    controller.run()  # already activated -> no second firing
    assert fired == ["ks"]


def test_control_reaches_a_fixpoint_a_firing_can_unblock_the_next_source():
    board = _board()
    _publish_topic(board, "start")
    order: list[str] = []

    def fire_a():
        order.append("a")
        _publish_topic(board, "mid")  # unblocks B

    def fire_b():
        order.append("b")

    controller = BlackboardController(board)
    # register B first to prove activation follows preconditions, not registration
    controller.register(KnowledgeSource(
        name="b", agent_role="B", precondition_topics=("mid",), activate=fire_b,
    ))
    controller.register(KnowledgeSource(
        name="a", agent_role="A", precondition_topics=("start",), activate=fire_a,
    ))
    controller.run()
    # A fires (start present), publishes mid, then B fires on the next pass
    assert order == ["a", "b"]


def test_every_activation_is_recorded_on_the_board():
    board = _board()
    _publish_topic(board, "agent.design.result")
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="verification", agent_role="VerificationAgent",
        precondition_topics=("agent.design.result",), activate=lambda: "plan",
    ))
    agenda = controller.run()
    # the control decision is auditable on the board
    activations = board.records(topic="control.activation")
    assert len(activations) == 1
    assert activations[0].payload["knowledge_source"] == "verification"
    assert activations[0].producer == "BlackboardController"
    # the returned agenda carries the source's own result
    assert agenda[0]["result"] == "plan"


def test_duplicate_registration_is_rejected():
    controller = BlackboardController(_board())
    ks = KnowledgeSource(
        name="dup", agent_role="R", precondition_topics=(), activate=lambda: None,
    )
    controller.register(ks)
    with pytest.raises(ValueError, match="already registered"):
        controller.register(ks)


def test_agenda_reports_registration_and_readiness_without_firing():
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
    assert ks["preconditions_met"] is False  # precondition not yet on the board


def test_stale_topic_cannot_activate_a_current_revision_source():
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
        agent_role="Role",
        precondition_topics=("ready",),
        activate=lambda: None,
    ))
    assert controller.activatable() == []


def test_declared_output_topic_is_enforced_and_failure_is_audited():
    board = _board()
    _publish_topic(board, "ready")
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="broken",
        agent_role="Role",
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
