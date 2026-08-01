from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.orchestrator import Orchestrator
from src.simulation.surgical_refiner import _find_def_span
from src.prototyping.blackboard import (
    Blackboard,
    RecordType,
    ProtectedModelElementError,
    StaleRevisionError,
    TaskStatus,
)
from src.prototyping.context_builder import ContextBuilder
from src.prototyping.experiment_arms import (
    LEGACY_EXPERIMENT_NAMESPACE,
    REVISED_EXPERIMENT_NAMESPACE,
    RevisedExperimentArm,
)
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
    requirement def REQ_SAFE_005 { doc /* Critical propulsion failure shall deploy a parachute. */ }
    part def SafetyMonitor {
        satisfy requirement REQ_SAFE_005;
        attribute propulsionCriticalFailure : Boolean = false;
    }
}"""


def test_legacy_and_revised_experiment_namespaces_cannot_be_confused():
    assert LEGACY_EXPERIMENT_NAMESPACE != REVISED_EXPERIMENT_NAMESPACE
    assert RevisedExperimentArm.parse("R1") is RevisedExperimentArm.BLACKBOARD_CONTEXT
    assert RevisedExperimentArm.parse("R1-LONG") is RevisedExperimentArm.LONG_SESSION_DIAGNOSTIC
    for legacy_label in ("B0", "B1", "B2"):
        with pytest.raises(ValueError, match="unknown revised experiment arm"):
            RevisedExperimentArm.parse(legacy_label)


def test_unimplemented_revised_arm_fails_closed():
    # R1-LONG is still reserved-but-not-implemented and must fail closed.
    with pytest.raises(NotImplementedError):
        Orchestrator(_NoCallLLM(), revised_experiment_arm="R1-LONG")


def test_r2_arm_is_runnable_but_not_evaluation_ready():
    # R2-BBAG runs its A/G intervention (Increment 2) but is not gold-poolable
    # until the independent evaluator lands (§13/§15).
    assert RevisedExperimentArm.SEMANTIC_ASSURANCE.implemented is True
    assert RevisedExperimentArm.SEMANTIC_ASSURANCE.evaluation_ready is False
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    assert orch.revised_experiment_arm is RevisedExperimentArm.SEMANTIC_ASSURANCE


def test_blackboard_is_revision_bound_and_sysml_is_the_only_commit_authority():
    board = Blackboard("Drone")
    source = board.publish(
        RecordType.SOURCE,
        "requirements.authoritative",
        "RequirementsAgent",
        {"requirements": [_REQ]},
    )
    assert source.model_revision == 0
    committed = board.commit_model(
        _MODEL, base_revision=0, base_digest=board.current_model.model_digest,
        producer="DesignAgent"
    )
    assert committed.revision == 1
    assert board.snapshot()["semantic_authority"] == "COMMITTED_SYSML_MODEL"
    assert any(
        item["element_id"] == "REQ_SAFE_005"
        for item in board.snapshot()["model_element_index"]
    )
    with pytest.raises(StaleRevisionError):
        board.commit_model(
            _MODEL, base_revision=0, base_digest=board.current_model.model_digest,
            producer="StaleAgent"
        )
    with pytest.raises(StaleRevisionError, match="base digest"):
        board.commit_model(
            _MODEL,
            base_revision=board.current_revision,
            base_digest="not-the-current-digest",
            producer="StaleDigestAgent",
        )
    with pytest.raises(ProtectedModelElementError):
        board.commit_model(
            _MODEL.replace("deploy a parachute", "deploy a parachute within 9 seconds"),
            base_revision=board.current_revision,
            base_digest=board.current_model.model_digest,
            producer="ThresholdWeakener",
        )
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
    assert envelope.estimated_tokens > 0
    assert envelope.record_context[0]["record_id"] == source.record_id
    assert envelope.context_item_provenance[0]["payload_digest"] == source.payload_digest
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
    truncated_task = board.create_task("TRUNCATED_CONTEXT", "DesignAgent")
    truncated = builder.build(
        task_id=truncated_task.task_id,
        agent_role="DesignAgent",
        objective="bounded model view",
        allowed_operation="READ_ONLY",
        model_context="x" * 4000,
        token_budget=300,
        allow_deterministic_truncation=True,
    )
    assert truncated.truncated is True
    assert "model_context:tail" in truncated.omitted_items
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
    nested_gold = board.publish(
        RecordType.EVIDENCE,
        "review.material",
        "Evaluator",
        {"wrapper": {"artifact_role": "EVALUATOR_GOLD"}},
    )
    nested_task = board.create_task("NESTED_GOLD_LEAK", "DesignAgent")
    board.transition_task(nested_task.task_id, TaskStatus.ACTIVE)
    with pytest.raises(ValueError, match="gold cannot enter"):
        builder.build(
            task_id=nested_task.task_id,
            agent_role="DesignAgent",
            objective="reject nested evaluator material",
            allowed_operation="READ_ONLY",
            included_record_ids=(nested_gold.record_id,),
        )
    blind_packet = board.publish(
        RecordType.EVIDENCE,
        "review.packet",
        "Evaluator",
        {"artifact_role": "BLIND_FAILURE_REVIEW_PACKET"},
    )
    blind_task = board.create_task("BLIND_PACKET_LEAK", "DesignAgent")
    board.transition_task(blind_task.task_id, TaskStatus.ACTIVE)
    with pytest.raises(ValueError, match="gold cannot enter"):
        builder.build(
            task_id=blind_task.task_id,
            agent_role="DesignAgent",
            objective="reject evaluator-only blind packet",
            allowed_operation="READ_ONLY",
            included_record_ids=(blind_packet.record_id,),
        )
    board.commit_model(
        _MODEL, base_revision=0, base_digest=board.current_model.model_digest,
        producer="DesignAgent"
    )
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
    committed = board.commit_model(
        _MODEL, base_revision=0, base_digest=board.current_model.model_digest,
        producer="DesignAgent"
    )
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
    result_record = next(
        item for item in collaboration["blackboard"]["records"]
        if item["topic"] == "agent.design.result"
    )
    assert result_record["payload"]["context_envelope_digest"]
    assert result_record["payload"]["transcript_digest"]
    assert result_record["payload"]["accepted_status"] == "ACCEPTED"


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


# --- R2-BBAG A/G wiring (Increment 2, step 1: orchestrator integration) -------

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_emitter import emit_ag_package

_MINI_AG = (
    "package Source { requirement def REQ_SAFE_005 { doc /* source */ } }\n"
    + emit_ag_package(REQ_SAFE_005_CHAIN)
)


def _r2_orchestrator_with_committed_model(
    model_text, arm="R2-BBAG", llm=None
):
    orch = Orchestrator(llm or _NoCallLLM(), revised_experiment_arm=arm)
    orch.blackboard = Blackboard("MiniAG")
    orch.context_builder = ContextBuilder(orch.blackboard)
    orch.task_sessions = TaskSessionRegistry()
    orch.blackboard.commit_model(
        model_text,
        base_revision=orch.blackboard.current_revision,
        base_digest=orch.blackboard.current_model.model_digest,
        producer="test",
    )
    return orch


def test_r2_wiring_extracts_checks_and_publishes_ag_trace():
    orch = _r2_orchestrator_with_committed_model(_MINI_AG)
    arts = orch._build_collaboration_artifacts(_MINI_AG)

    graph = arts["ag_contract_graph"]
    assert graph["verdict"] == "PASS"
    assert graph["source_model_revision"] == orch.blackboard.current_revision
    assert graph["source_model_digest"] == orch.blackboard.current_model.model_digest
    assert arts["revised_experiment"]["evaluation_ready"] is False

    # a typed ANALYSIS record was published to the blackboard at current revision
    ag = [r for r in orch.blackboard.snapshot()["records"]
          if r["topic"] == "analysis.ag_trace"]
    assert len(ag) == 1
    assert ag[0]["record_type"] == "ANALYSIS"
    assert ag[0]["payload"]["verdict"] == "PASS"
    assert ag[0]["payload"]["evaluation_ready"] is False
    assert ag[0]["model_revision"] == orch.blackboard.current_revision
    with pytest.raises(ValueError, match="does not match the committed"):
        orch._build_collaboration_artifacts(_MINI_AG + "\n// uncommitted")


def test_r2_wiring_is_honest_when_model_has_no_ag_contracts():
    # The committed model is the sole authority: a model without A/G contracts
    # yields an INCOMPLETE trace, never a fabricated PASS (§6.2).
    orch = _r2_orchestrator_with_committed_model(_MODEL)
    graph = orch._build_collaboration_artifacts(_MODEL)["ag_contract_graph"]
    assert graph["verdict"] == "INCOMPLETE"


class _OneShotRepairLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        return self.response


def test_r2_controller_executes_one_routed_repair_and_rechecks_terminal_revision():
    broken = _MINI_AG.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    span = _find_def_span(_MINI_AG, "state", "RecoverySystemBehavior")
    assert span is not None
    fixed_behavior = _MINI_AG[span[0]:span[1]]
    llm = _OneShotRepairLLM(f"```sysml\n{fixed_behavior}\n```")
    orch = _r2_orchestrator_with_committed_model(broken, llm=llm)

    arts = orch._build_collaboration_artifacts(broken)

    assert llm.calls == 1
    assert orch.blackboard.current_revision == 2
    assert arts["_terminal_model_sysml"] == orch.blackboard.current_model.model_text
    assert arts["ag_contract_graph"]["verdict"] == "PASS"
    assert arts["pattern_conformance_report"]["verdict"] == "PASS"
    assert len(arts["failure_diagnostics"]["analysis_history"]) == 2
    assert any(
        item["status"] == "ACCEPTED"
        for item in arts["repair_decisions"]["decisions"]
    )
    sessions = orch.task_sessions.snapshot(include_messages=True)["sessions"]
    assert sessions[-1]["status"] == "COMPLETED"
    assert sessions[-1]["messages"]


def test_r2_controller_records_rejected_repair_without_changing_revision():
    broken = _MINI_AG.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    llm = _OneShotRepairLLM("```sysml\nstate def Unrelated { }\n```")
    orch = _r2_orchestrator_with_committed_model(broken, llm=llm)

    arts = orch._build_collaboration_artifacts(broken)

    # First named obligation, its bounded feedback retry, then the next named
    # repairable obligation in the fixed run-level budget.
    assert llm.calls == 3
    assert orch.blackboard.current_revision == 1
    assert arts["ag_contract_graph"]["verdict"] != "PASS"
    assert any(
        item["status"] == "REJECTED"
        for item in arts["repair_decisions"]["decisions"]
    )
    tasks = orch.blackboard.snapshot()["tasks"]
    assert any(
        item["kind"] == "A_G_SURGICAL_REPAIR"
        and item["status"] == "REJECTED"
        for item in tasks
    )
    assert not any(
        item["kind"] == "A_G_SURGICAL_REPAIR"
        and item["status"] == "PENDING"
        for item in tasks
    )


def test_r2_controller_blocks_unsupported_upstream_decomposition_repair():
    broken = _MINI_AG.replace(
        "    dependency dischargeParachuteDeploymentCommand__to__RecoverySystemContract "
        "from SafetyResponseArbiterContract to RecoverySystemContract;\n",
        "",
    )
    orch = _r2_orchestrator_with_committed_model(broken)

    arts = orch._build_collaboration_artifacts(broken)

    assert arts["ag_contract_graph"]["verdict"] != "PASS"
    blocked = [
        item for item in arts["repair_decisions"]["decisions"]
        if item["status"] == "BLOCKED"
        and "upstream_decomposition_repair" in item["reason"]
    ]
    assert blocked
    tasks = orch.blackboard.snapshot()["tasks"]
    assert any(
        item["kind"] == "A_G_UPSTREAM_INTEGRATION_REPAIR"
        and item["status"] == "BLOCKED"
        for item in tasks
    )


def test_r1_arm_does_not_emit_an_ag_trace():
    orch = _r2_orchestrator_with_committed_model(_MINI_AG, arm="R1-BBCTX")
    arts = orch._build_collaboration_artifacts(_MINI_AG)
    assert "ag_contract_graph" not in arts


def test_the_derived_ag_graph_cannot_become_model_authority():
    """§14: the derived JSON view is never generation or repair authority.

    Two halves, because only the pair closes it. The positive rule (SysML is the
    sole commit authority) was already pinned; what was not is the NEGATIVE one —
    that serialising the graph and feeding it back cannot install a fact the
    committed SysML does not contain. A JSON-supplied A/G fact is exactly the
    "repair by editing the export" failure §6.2 forbids.
    """
    import json

    from src.prototyping.ag_contracts import check_ag_graph
    from src.prototyping.ag_extractor import extract_ag_graph
    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package

    model = (
        "package Drone {\n"
        "    requirement def REQ_SAFE_005 { doc /* Critical propulsion failure "
        "shall deploy a parachute. */ }\n"
        "}\n\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    )
    view = check_ag_graph(extract_ag_graph(model)).to_dict()
    assert view["verdict"] == "PASS"

    # 1. the derived view cannot be committed as the model: it carries none of the
    #    authoritative requirement text the commit gate demands
    board = Blackboard("Drone")
    board.publish(
        RecordType.SOURCE, "requirements.authoritative", "RequirementsAgent",
        {"requirements": [_REQ]},
    )
    with pytest.raises(ProtectedModelElementError):
        board.commit_model(
            json.dumps(view), base_revision=0,
            base_digest=board.current_model.model_digest, producer="RepairAgent",
        )

    # 2. a fact injected into the derived view does not survive re-extraction:
    #    extraction reads the committed SysML, so the JSON cannot add an edge
    tampered = json.loads(json.dumps(view))
    tampered["graph"]["allocations"].append({
        "contract": "InventedContract", "guarantee": "inventedGuarantee",
        "owner": "inventedPart",
    })
    tampered["verdict"] = "PASS"
    reextracted = check_ag_graph(extract_ag_graph(model)).to_dict()
    assert reextracted["graph"]["allocations"] == view["graph"]["allocations"]
    assert not any(
        item.get("contract") == "InventedContract"
        for item in reextracted["graph"]["allocations"]
    )
    # and the checker's own input type is an extracted graph, never a parsed view
    with pytest.raises((AttributeError, TypeError)):
        check_ag_graph(tampered)
