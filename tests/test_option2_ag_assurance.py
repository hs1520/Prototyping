from __future__ import annotations

import pytest

from src.agents.surgical_refiner import _find_def_span
from src.prototyping.ag_assurance import (
    FailureClass,
    FailureRoute,
    check_safety_pattern_conformance,
    route_failure_diagnostics,
)
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_repair import attempt_dependency_closed_ag_repair
from src.prototyping.blackboard import Blackboard, RecordType
from src.prototyping.context_builder import ContextBuilder
from src.prototyping.task_session import TaskSessionRegistry


GOOD = (
    "package Source { requirement def REQ_SAFE_005 { doc /* source */ } }\n"
    + emit_ag_package(REQ_SAFE_005_CHAIN)
)


def _check(text: str):
    graph = extract_ag_graph(text, revision=1)
    return graph, check_ag_graph(graph)


def test_pattern_pass_requires_real_topology_not_selection_alone():
    graph, report = _check(GOOD)
    pattern = check_safety_pattern_conformance(graph, report)
    assert report.verdict == "PASS"
    assert pattern["verdict"] == "PASS"
    assert len(pattern["cases"]) == 3
    assert pattern["formal_proof"] is False

    broken = GOOD.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    graph, report = _check(broken)
    assert "REALIZATION_ACTION_MISSING" in {d.code for d in report.errors()}
    assert check_safety_pattern_conformance(graph, report)["verdict"] == "FAIL"


def test_typed_failure_routing_distinguishes_model_integration_and_verifier():
    broken_behavior = GOOD.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    graph, report = _check(broken_behavior)
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    model_failure = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "REALIZATION_ACTION_MISSING"
    )
    assert model_failure["classification"] == FailureClass.MODEL_SEMANTIC_FAULT.value
    assert model_failure["route"] == FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR.value
    assert model_failure["repair_authorized"] is True

    missing_edge = GOOD.replace(
        "    dependency dischargeParachuteDeploymentCommand__to__RecoverySystemContract "
        "from SafetyResponseArbiterContract to RecoverySystemContract;\n",
        "",
    )
    _graph, report = _check(missing_edge)
    routed = route_failure_diagnostics(
        report.diagnostics, source_requirement=report.source_requirement
    )
    integration = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "DISCHARGE_EDGE_MISSING"
    )
    assert integration["classification"] == FailureClass.INTEGRATION_DECOMPOSITION_GAP.value
    assert integration["repair_authorized"] is False

    no_observer = GOOD.replace(
        "    dependency observeSystemParachuteContract from SystemParachuteContract to ParachuteDeploymentVerification;\n",
        "",
    )
    _graph, report = _check(no_observer)
    routed = route_failure_diagnostics(
        report.diagnostics, source_requirement=report.source_requirement
    )
    verifier = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "OBSERVATION_MISSING"
    )
    assert verifier["classification"] == FailureClass.VERIFIER_LIMITATION.value
    assert verifier["repair_authorized"] is False


class _RepairLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def chat(self, _prompt: str, **_kwargs) -> str:
        self.calls += 1
        return self.response


def _definition(text: str, kind: str, name: str) -> str:
    span = _find_def_span(text, kind, name)
    assert span is not None
    return text[span[0]:span[1]]


def test_board_mediated_dependency_closed_repair_accepts_only_targeted_fix():
    broken = GOOD.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    graph, report = _check(broken)
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "REALIZATION_ACTION_MISSING"
    )

    board = Blackboard("Drone")
    board.commit_model(
        broken,
        base_revision=0,
        base_digest=board.current_model.model_digest,
        producer="test",
    )
    analysis = board.publish(
        RecordType.ANALYSIS, "analysis.ag_trace", "AGChecker",
        {"diagnostics": [d.as_dict() for d in report.diagnostics]},
    )
    failure_record = board.publish(
        RecordType.ANALYSIS, "diagnostic.failure", "AGFailureRouter", failure,
    )
    good_behavior = _definition(GOOD, "state", "RecoverySystemBehavior")
    llm = _RepairLLM(f"```sysml\n{good_behavior}\n```")
    decision = attempt_dependency_closed_ag_repair(
        llm=llm,
        board=board,
        context_builder=ContextBuilder(board),
        sessions=TaskSessionRegistry(),
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )
    assert decision.status == "ACCEPTED"
    assert decision.target_diagnostic_removed is True
    assert decision.regression_free is True
    assert decision.whole_model_fallback_used is False
    assert llm.calls == 1
    assert "requirement def REQ_SAFE_005" in board.current_model.model_text
    assert check_ag_graph(extract_ag_graph(board.current_model.model_text)).verdict == "PASS"


@pytest.mark.parametrize(
    "malicious_block",
    [
        lambda: (
            "requirement def REQ_SAFE_005 "
            "{ doc /* rewritten source */ }"
        ),
        lambda: _definition(
            GOOD, "requirement", "RecoverySystemContract"
        ).replace("0.35 [SI::s]", "0.34 [SI::s]"),
        lambda: _definition(
            GOOD, "requirement", "RecoverySystemContract"
        ).replace("[SI::s]", "[SI::ms]"),
        lambda: _definition(
            GOOD, "state", "SafetyResponseArbitration"
        ).replace(
            "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand",
            "issueDifferentCommand",
        ),
    ],
)
def test_ag_repair_rejects_source_threshold_unit_and_unrelated_edits(
    malicious_block
):
    broken = GOOD.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    graph, report = _check(broken)
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "REALIZATION_ACTION_MISSING"
    )
    board = Blackboard("Drone")
    board.commit_model(
        broken,
        base_revision=0,
        base_digest=board.current_model.model_digest,
        producer="test",
    )
    analysis = board.publish(
        RecordType.ANALYSIS,
        "analysis.ag_trace",
        "AGChecker",
        {"diagnostics": [item.as_dict() for item in report.diagnostics]},
    )
    failure_record = board.publish(
        RecordType.ANALYSIS,
        "diagnostic.failure",
        "AGFailureRouter",
        failure,
    )
    decision = attempt_dependency_closed_ag_repair(
        llm=_RepairLLM(f"```sysml\n{malicious_block()}\n```"),
        board=board,
        context_builder=ContextBuilder(board),
        sessions=TaskSessionRegistry(),
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )
    assert decision.status == "REJECTED"
    assert decision.whole_model_fallback_used is False
    assert board.current_revision == 1
    assert board.current_model.model_text == broken


def test_a_rejected_repair_records_which_gate_refused_and_why():
    """A REJECTED repair decision must be actionable.

    `repair_decisions.json` recorded only that a repair was rejected, with the
    reason `surgical_merge_rejected` or `target_not_removed_or_regression` — the
    latter naming three possibilities at once. An out-of-scope edit, a patch that
    missed its target, a new defect and a broken pattern all looked identical. The
    same defect as the lumped PRIORITY_TOPOLOGY_INCOMPLETE diagnostic, in the
    repair artifact instead of the checker.

    Measured on a real provider run: `addition_out_of_scope:action:...` — the
    model added a new action definition instead of restoring the existing entry
    action. That is a fact about the prompt, and it was invisible before.
    """
    broken = GOOD.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    graph, report = _check(broken)
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "REALIZATION_ACTION_MISSING"
    )

    board = Blackboard("Drone")
    board.commit_model(
        broken, base_revision=0,
        base_digest=board.current_model.model_digest, producer="test",
    )
    analysis = board.publish(
        RecordType.ANALYSIS, "analysis.ag_trace", "AGChecker", {"diagnostics": []},
    )
    failure_record = board.publish(
        RecordType.ANALYSIS, "diagnostic.failure", "AGFailureRouter", failure,
    )

    # a patch that merges but leaves the target diagnostic in place
    unchanged = _definition(broken, "state", "RecoverySystemBehavior")
    decision = attempt_dependency_closed_ag_repair(
        llm=_RepairLLM(f"```sysml\n{unchanged}\n```"),
        board=board,
        context_builder=ContextBuilder(board),
        sessions=TaskSessionRegistry(),
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )
    assert decision.status == "REJECTED"

    published = next(
        item.payload for item in reversed(board.records(topic="repair.decision"))
    )
    detail = published.get("gate") or published.get("audit")
    assert detail, f"a rejection must say which gate refused: {published}"
    if "gate" in published:
        gate = published["gate"]
        assert gate["target"][0] == "REALIZATION_ACTION_MISSING"
        assert gate["target_removed"] is False
        assert "remaining_diagnostics" in gate
    else:
        assert published["audit"]["rejection_reasons"]


def test_the_repair_prompt_states_the_rules_its_gate_enforces():
    """Ninth instance of the recurring defect: the merge gate refuses an added
    definition and requires the conventional action name, and the repair feedback
    said neither. A measured run was rejected with
    `addition_out_of_scope:action:DeployBallisticRecoveryParachute` — the model
    invented an action rather than restoring the existing one."""
    from src.prototyping.ag_repair import _repair_feedback

    feedback = _repair_feedback("REALIZATION_ACTION_MISSING")
    assert "adding a new definition" in feedback
    assert "out of scope" in feedback
    # single-sourced from ag_convention, not restated here
    assert "prefixed with `set`" in feedback
    # and an unknown code still gets the scope rules, just no convention line
    assert "adding a new definition" in _repair_feedback("SOMETHING_ELSE")
