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
from src.prototyping.controller import (
    AgentRole,
    BlackboardController,
    KnowledgeSource,
)


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
    assert controller.run(allow_partial=True) == []
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
        name="ks", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=("start",),
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
        name="b", agent_role=AgentRole.DESIGN, precondition_topics=("mid",), activate=fire_b,
    ))
    controller.register(KnowledgeSource(
        name="a", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=("start",), activate=fire_a,
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
        name="dup", agent_role=AgentRole.ORCHESTRATOR, precondition_topics=(), activate=lambda: None,
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
        agent_role=AgentRole.ORCHESTRATOR,
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


def test_allow_partial_lets_a_short_run_return_instead_of_raising():
    # Completeness is the default, so a caller that means to register more
    # sources than the board satisfies must say so.
    board = _board()
    controller = BlackboardController(board)
    controller.register(KnowledgeSource(
        name="never",
        agent_role=AgentRole.ORCHESTRATOR,
        precondition_topics=("absent",),
        activate=lambda: None,
    ))
    assert controller.run(allow_partial=True) == []


def test_a_stalled_chain_is_an_error_by_default():
    # The output-topic check cannot see this case: `blocked` never ran at all,
    # so nothing failed to publish -- control simply reached a fixpoint early.
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
    # The source that could run still did; the error is about completeness.
    assert fired == ["runs"]


def test_stall_error_names_the_topic_each_stalled_source_is_waiting_on():
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
    # The precondition that *was* satisfied is not reported as outstanding.
    assert "start" not in message.split("awaiting", 1)[1]


def test_a_model_commit_hides_earlier_topics_and_the_stall_says_so():
    """The real hazard, detected from the board rather than from source text.

    `_published_topics` only sees the current model revision. Committing a model
    advances it, so topics published before the commit stop being visible unless
    something reaffirms them. This test performs the commit through an alias, so
    no textual search for `commit_model` on a named attribute would find it; the
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

    alias = board                      # the commit is not written as board.commit_model
    alias.commit_model(
        "part def P;",
        base_revision=board.current_revision,
        base_digest=board.current_model.model_digest,
        producer="Test",
    )

    # The topic still exists on the board, but not at the current revision.
    assert "start" in controller._topics_at_any_revision()
    assert "start" in controller._hidden_topics()
    assert controller.activatable() == []

    with pytest.raises(RuntimeError) as excinfo:
        controller.run()
    message = str(excinfo.value)
    assert "downstream awaiting start" in message
    assert "hidden by a model revision" in message


def test_a_topic_nobody_produced_is_not_reported_as_hidden():
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


def test_an_unknown_agent_role_is_rejected_at_registration_not_in_the_artefacts():
    with pytest.raises(ValueError, match="unknown agent role"):
        KnowledgeSource(
            name="typo", agent_role="AssurenceAgent",
            precondition_topics=(), activate=lambda: None,
        )


def test_a_known_role_given_as_a_string_is_normalised_to_the_enum():
    source = KnowledgeSource(
        name="ok", agent_role="VerificationAgent",
        precondition_topics=(), activate=lambda: None,
    )
    assert source.agent_role is AgentRole.VERIFICATION


def test_a_process_fact_survives_a_model_commit_and_a_model_fact_does_not():
    """The root-cause fix: staleness is a property the publisher declares.

    Before this, every record expired with the revision, so a chain spanning a
    commit could not proceed and opportunistic activation was unavailable to any
    pipeline that revises its model. Now the record says whether it asserts
    something about the model.
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
    assert "phase.done" in published        # process fact: still true
    assert "analysis.of.model" not in published  # model fact: now stale


def test_a_chain_spanning_a_model_commit_now_completes():
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
        # Two preconditions published under *different* model revisions -- the
        # case that could never activate before.
        precondition_topics=("start", "mid"),
        activate=lambda: fired.append("second"),
    ))
    controller.run()
    assert fired == ["first", "second"]
