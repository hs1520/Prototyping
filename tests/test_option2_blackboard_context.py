from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.orchestrator import Orchestrator
from src.prototyping.blackboard import (
    Blackboard,
    RecordType,
    StaleRevisionError,
    TaskStatus,
)
from src.prototyping.context_builder import ContextBuilder
from src.prototyping.experiment_arms import (
    LEGACY_EXPERIMENT_NAMESPACE,
    REVISED_EXPERIMENT_NAMESPACE,
    RevisedExperimentArm,
)
from src.prototyping.robustness import RobustnessOptions
from src.prototyping.task_session import (
    SessionError,
    SessionStatus,
    TaskSessionRegistry,
)
from src.sysml.lite_model import build_lite_model


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("LLM should not be called")


_REQ = "REQ-SAFE-005: Critical propulsion failure shall deploy a parachute."
_MODEL = """package Drone {
    requirement def REQ_SAFE_005 { doc /* deploy parachute */ }
    part def SafetyMonitor {
        satisfy requirement REQ_SAFE_005;
        attribute propulsionCriticalFailure : Boolean = false;
    }
}"""


def test_legacy_and_revised_experiment_namespaces_cannot_be_confused():
    assert LEGACY_EXPERIMENT_NAMESPACE != REVISED_EXPERIMENT_NAMESPACE
    assert RobustnessOptions.b2().as_metadata()["configuration"] == "B2"
    assert (
        RobustnessOptions.b2().as_metadata()["experiment_namespace"]
        == LEGACY_EXPERIMENT_NAMESPACE
    )
    assert RevisedExperimentArm.parse("R1") is RevisedExperimentArm.BLACKBOARD_CONTEXT
    assert RevisedExperimentArm.parse("R1-LONG") is RevisedExperimentArm.LONG_SESSION_DIAGNOSTIC


def test_unimplemented_revised_arm_and_mixed_legacy_intervention_fail_closed():
    with pytest.raises(NotImplementedError):
        Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    with pytest.raises(ValueError, match="cannot be mixed"):
        Orchestrator(
            _NoCallLLM(),
            robustness_options=RobustnessOptions.b1(),
            revised_experiment_arm="R1-BBCTX",
        )


def test_blackboard_is_revision_bound_and_sysml_is_the_only_commit_authority():
    board = Blackboard("Drone")
    source = board.publish(
        RecordType.SOURCE,
        "requirements.authoritative",
        "RequirementsAgent",
        {"requirements": [_REQ]},
    )
    assert source.model_revision == 0
    committed = board.commit_model(_MODEL, base_revision=0, producer="DesignAgent")
    assert committed.revision == 1
    assert board.snapshot()["semantic_authority"] == "COMMITTED_SYSML_MODEL"
    with pytest.raises(StaleRevisionError):
        board.commit_model(_MODEL, base_revision=0, producer="StaleAgent")
    with pytest.raises(StaleRevisionError):
        board.publish(
            RecordType.ANALYSIS,
            "stale.analysis",
            "Checker",
            {},
            model_revision=-1,
        )


def test_context_builder_is_role_checked_revision_pinned_and_has_no_gold_input():
    board = Blackboard("Drone")
    source = board.publish(
        RecordType.SOURCE,
        "requirements.authoritative",
        "RequirementsAgent",
        {"requirements": [_REQ]},
    )
    task = board.create_task(
        "INITIAL_MODEL_GENERATION",
        "DesignAgent",
        required_topics=("requirements.authoritative",),
    )
    board.transition_task(task.task_id, TaskStatus.ACTIVE)
    builder = ContextBuilder(board)
    envelope = builder.build_design_context(
        task_id=task.task_id,
        system_name="Drone",
        source_record_ids=(source.record_id,),
    )
    assert envelope.model_revision == 0
    assert envelope.model_digest == board.current_model.model_digest
    assert "not evaluator gold" in envelope.render_for_prompt()
    assert "gold" not in ContextBuilder.build.__code__.co_varnames
    raw_task = board.create_task("RAW_SOURCE", "DesignAgent")
    with pytest.raises(ValueError, match="SOURCE records"):
        builder.build(
            task_id=raw_task.task_id,
            agent_role="DesignAgent",
            objective="inject raw semantic input",
            allowed_operation="CREATE_INITIAL_MODEL",
            source_requirements=[_REQ],
        )
    small_task = board.create_task("SMALL_CONTEXT", "DesignAgent")
    with pytest.raises(ValueError, match="exceeds its token budget"):
        builder.build(
            task_id=small_task.task_id,
            agent_role="DesignAgent",
            objective="too large for the declared budget",
            allowed_operation="READ_ONLY",
            model_context="x" * 1000,
            token_budget=10,
        )
    with pytest.raises(ValueError, match="does not match"):
        builder.build(
            task_id=task.task_id,
            agent_role="RepairAgent",
            objective="wrong role",
            allowed_operation="PATCH",
        )
    missing_task = board.create_task(
        "SECOND_DESIGN",
        "DesignAgent",
        required_topics=("requirements.authoritative",),
    )
    with pytest.raises(ValueError, match="required typed publications"):
        builder.build_design_context(
            task_id=missing_task.task_id,
            system_name="Drone",
            source_record_ids=(),
        )
    gold = board.publish(
        RecordType.EVIDENCE,
        "gold.semantic",
        "Evaluator",
        {"artifact_role": "EVALUATOR_GOLD"},
    )
    gold_task = board.create_task("GOLD_LEAK", "DesignAgent")
    with pytest.raises(ValueError, match="gold cannot enter"):
        builder.build_design_context(
            task_id=gold_task.task_id,
            system_name="Drone",
            source_record_ids=(gold.record_id,),
        )
    board.commit_model(_MODEL, base_revision=0, producer="DesignAgent")
    with pytest.raises(ValueError, match="stale task"):
        builder.build_design_context(
            task_id=task.task_id,
            system_name="Drone",
            source_record_ids=(source.record_id,),
        )


def test_task_session_has_one_task_role_revision_and_bounded_lifetime():
    registry = TaskSessionRegistry()
    session = registry.open(
        task_id="task-1",
        agent_role="DesignAgent",
        base_model_revision=0,
        base_model_digest="abc",
        context_envelope_id="context-1",
        max_turns=1,
        max_tokens=10,
    )
    session.append("user", "input", token_count=2)
    session.append("assistant", "output", token_count=2)
    with pytest.raises(SessionError, match="turn budget"):
        session.append("user", "another", token_count=1)
    assert session.status is SessionStatus.EXPIRED
    assert session.transcript_digest
    session.close(SessionStatus.EXPIRED)
    with pytest.raises(SessionError, match="belongs to DesignAgent"):
        registry.open(
            task_id="task-1",
            agent_role="RepairAgent",
            base_model_revision=0,
            base_model_digest="abc",
            context_envelope_id="context-2",
        )


def test_model_commit_stales_other_open_sessions():
    board = Blackboard("Drone")
    registry = TaskSessionRegistry()
    task = board.create_task("VERIFY", "VerificationAgent")
    board.transition_task(task.task_id, TaskStatus.ACTIVE)
    session = registry.open(
        task_id=task.task_id,
        agent_role="VerificationAgent",
        base_model_revision=0,
        base_model_digest=board.current_model.model_digest,
        context_envelope_id="context-2",
    )
    committed = board.commit_model(_MODEL, base_revision=0, producer="DesignAgent")
    stale = registry.stale_after_commit(
        committed.revision, committed.model_digest
    )
    assert stale == (session.session_id,)
    assert session.status is SessionStatus.STALE
    rebased_task = board.rebase_task(task.task_id)
    builder = ContextBuilder(board)
    envelope = builder.build(
        task_id=task.task_id,
        agent_role="VerificationAgent",
        objective="verify current model",
        allowed_operation="READ_ONLY_VERIFY",
    )
    replacement = registry.rebase(
        session.session_id,
        base_model_revision=rebased_task.base_model_revision,
        base_model_digest=rebased_task.base_model_digest,
        context_envelope_id=envelope.envelope_id,
    )
    assert replacement.status is SessionStatus.OPEN
    assert replacement.rebased_from_session_id == session.session_id


def test_r1_handoff_is_recorded_and_commits_the_agent_model():
    orchestrator = Orchestrator(
        _NoCallLLM(), revised_experiment_arm="R1-BBCTX"
    )
    orchestrator.last_requirement_input = {
        "mode": "frozen",
        "requirement_set_digest": "req-digest",
    }
    orchestrator._prepare_design_handoff("Drone", [_REQ])
    envelope = orchestrator._active_design_handoff["envelope"]
    assert envelope.agent_role == "DesignAgent"
    model = build_lite_model(_MODEL, model_name="Drone")
    orchestrator._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}),
        model,
    )
    artifacts = orchestrator._build_collaboration_artifacts()
    assert artifacts["revised_experiment"]["configuration"] == "R1-BBCTX"
    collaboration = artifacts["collaboration"]
    assert collaboration["blackboard"]["current_model"]["revision"] == 1
    assert collaboration["blackboard"]["tasks"][0]["status"] == "COMPLETED"
    assert collaboration["task_sessions"]["sessions"][0]["status"] == "COMPLETED"
    assert collaboration["task_sessions"]["sessions"][0]["agent_role"] == "DesignAgent"
    assert collaboration["contexts"]["envelopes"][0]["source_requirements"] == (_REQ,)


def test_r1_rejects_and_closes_the_task_when_design_agent_raises():
    orchestrator = Orchestrator(
        _NoCallLLM(), revised_experiment_arm="R1-BBCTX"
    )
    orchestrator._prepare_design_handoff("Drone", [_REQ])

    def fail(_task):
        raise RuntimeError("generation failed")

    orchestrator.design_agent.run = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        orchestrator._generate_initial_design("Drone", [_REQ])
    snapshot = orchestrator.blackboard.snapshot()
    assert snapshot["tasks"][0]["status"] == "REJECTED"
    sessions = orchestrator.task_sessions.snapshot()["sessions"]
    assert sessions[0]["status"] == "REJECTED"
