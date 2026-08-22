from __future__ import annotations

import pytest

from src.simulation.surgical_refiner import _find_def_span
from src.prototyping.ag_assurance import (
    FailureClass,
    FailureRoute,
    check_safety_pattern_conformance,
    route_failure_diagnostics,
)
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import AGDiagnostic, check_ag_graph
from src.prototyping.ag_convention import (
    PRIORITY_INPUT_OR_CONTRACT,
    PRIORITY_MODEL_WIRING,
    PRIORITY_OBLIGATIONS,
)
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


def test_nonatomic_component_guarantee_is_blocked_contract_work():
    routed = route_failure_diagnostics(
        (
            AGDiagnostic(
                "COMPONENT_GUARANTEE_NONATOMIC",
                "component guarantee is compound",
                contract="RecoveryContract",
                subject="g_response",
            ),
        ),
        source_requirement="REQ_SAFE_005",
    )
    failure = routed["failures"][0]
    assert failure["classification"] == (
        FailureClass.CONTRACT_INCOMPLETENESS.value
    )
    assert failure["route"] == FailureRoute.CLARIFICATION_OR_BLOCKED.value
    assert failure["repair_authorized"] is False


def test_itemized_pattern_topology_routes_to_its_existing_behavior():
    diagnostic = AGDiagnostic(
        "PATTERN_TOPOLOGY_INCOMPLETE",
        "startup topology incomplete; unsatisfied: reset_state_present",
        contract="SystemContract",
        subject="STARTUP_INHIBIT",
        provenance={
            "unsatisfied_obligations": ["reset_state_present"],
            "obligation_affected_elements": {
                "reset_state_present": ["LatchBehavior"]
            },
        },
    )
    routed = route_failure_diagnostics(
        (diagnostic,),
        source_requirement="REQ_SAFE_004",
        realization_links=(),
    )
    failure = routed["failures"][0]
    assert failure["route"] == (
        FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR.value
    )
    assert failure["repair_authorized"] is True
    assert "LatchBehavior" in failure["affected_elements"]


def test_model_fault_without_existing_behavior_target_fails_closed():
    """The merge policy replaces existing definitions; it cannot satisfy a
    REALIZATION_MISSING fault by inventing the absent state definition.
    """
    routed = route_failure_diagnostics(
        (
            AGDiagnostic(
                "REALIZATION_MISSING",
                "ComponentContract has no realization",
                contract="ComponentContract",
            ),
        ),
        source_requirement="REQ_SAFE_005",
        realization_links=(),
    )
    failure = routed["failures"][0]
    assert failure["repair_authorized"] is False
    assert failure["route"] == FailureRoute.CLARIFICATION_OR_BLOCKED.value
    assert failure["routing_basis"] == "MODEL_FAULT_WITHOUT_BEHAVIOR_TARGET"


def test_every_priority_obligation_has_the_reviewed_routing_scope():
    """Moving an obligation across the repair boundary must fail loudly.

    The aggregate checker code is not a routing unit. These are the individual
    facts it reports. Only topology that can be restored inside an existing state
    definition is authorised; changing the response vocabulary, contract facts,
    or verification boundary remains blocked.
    """
    expected_wiring = {
        "selection_guarded_by_trigger",
        "competing_transitions_guarded",
        "selected_transition_reachable",
        "selection_action_connected",
        "recovery_power_available_at_boundary",
        "deployment_action_connected",
    }
    expected_blocked = {
        "response_member_provenance",
        "response_set_members",
        "precedence_edges",
        "single_highest_response",
        "selected_response",
        "trigger_concept",
        "trigger_matches_timing_origin",
        "arbiter_guarantees",
        "observation_connected",
    }
    by_scope = {
        PRIORITY_MODEL_WIRING: {
            item.obligation_id
            for item in PRIORITY_OBLIGATIONS
            if item.failure_scope == PRIORITY_MODEL_WIRING
        },
        PRIORITY_INPUT_OR_CONTRACT: {
            item.obligation_id
            for item in PRIORITY_OBLIGATIONS
            if item.failure_scope == PRIORITY_INPUT_OR_CONTRACT
        },
    }
    assert by_scope[PRIORITY_MODEL_WIRING] == expected_wiring
    assert by_scope[PRIORITY_INPUT_OR_CONTRACT] == expected_blocked
    assert set.union(*by_scope.values()) == {
        item.obligation_id for item in PRIORITY_OBLIGATIONS
    }


def test_priority_routing_splits_wiring_from_input_semantics():
    diagnostic = AGDiagnostic(
        "PRIORITY_TOPOLOGY_INCOMPLETE",
        "priority topology is incomplete; unsatisfied: response_set_members, "
        "selection_action_connected",
        contract="SystemParachuteContract",
        subject="criticalPropulsionFailureDetected",
        provenance={
            "unsatisfied_obligations": [
                "response_set_members",
                "selection_action_connected",
            ],
            "obligation_affected_elements": {
                "selection_action_connected": ["SafetyResponseArbitration"],
            },
        },
    )
    routed = route_failure_diagnostics(
        [diagnostic],
        source_requirement="REQ_SAFE_005",
    )
    assert len(routed["failures"]) == 2
    by_obligation = {
        item["priority_obligation"]: item for item in routed["failures"]
    }

    blocked = by_obligation["response_set_members"]
    assert blocked["classification"] == FailureClass.CONTRACT_INCOMPLETENESS.value
    assert blocked["route"] == FailureRoute.CLARIFICATION_OR_BLOCKED.value
    assert blocked["repair_authorized"] is False

    repairable = by_obligation["selection_action_connected"]
    assert repairable["classification"] == FailureClass.MODEL_SEMANTIC_FAULT.value
    assert (
        repairable["route"]
        == FailureRoute.DEPENDENCY_CLOSED_SURGICAL_REPAIR.value
    )
    assert repairable["repair_authorized"] is True
    assert "SafetyResponseArbitration" in repairable["affected_elements"]
    assert "selection_action_connected" in repairable["message"]
    assert repairable["routing_basis"] == "NAMED_PRIORITY_OBLIGATION"


def test_unknown_priority_obligation_fails_closed():
    routed = route_failure_diagnostics(
        [AGDiagnostic(
            "PRIORITY_TOPOLOGY_INCOMPLETE",
            "unsatisfied: future_obligation",
            contract="SystemParachuteContract",
            provenance={"unsatisfied_obligations": ["future_obligation"]},
        )],
        source_requirement="REQ_SAFE_005",
    )
    failure = routed["failures"][0]
    assert failure["priority_obligation"] == "future_obligation"
    assert failure["repair_authorized"] is False
    assert failure["route"] == FailureRoute.CLARIFICATION_OR_BLOCKED.value


def test_wiring_obligation_without_an_existing_behavior_target_fails_closed():
    routed = route_failure_diagnostics(
        [AGDiagnostic(
            "PRIORITY_TOPOLOGY_INCOMPLETE",
            "unsatisfied: recovery_power_available_at_boundary",
            contract="SystemParachuteContract",
            provenance={
                "unsatisfied_obligations": [
                    "recovery_power_available_at_boundary"
                ],
                "obligation_affected_elements": {
                    "recovery_power_available_at_boundary": [],
                },
            },
        )],
        source_requirement="REQ_SAFE_005",
    )
    failure = routed["failures"][0]

    assert failure["repair_authorized"] is False
    assert failure["classification"] == FailureClass.CONTRACT_INCOMPLETENESS.value
    assert failure["route"] == FailureRoute.CLARIFICATION_OR_BLOCKED.value
    assert failure["routing_basis"] == (
        "NAMED_PRIORITY_OBLIGATION_WITHOUT_BEHAVIOR_TARGET"
    )


def test_checker_binds_repairable_priority_obligation_to_its_state_def():
    injured = GOOD.replace(
        "entry action "
        "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand;",
        "",
    )
    _graph, report = _check(injured)
    diagnostic = next(
        item for item in report.errors()
        if item.code == "PRIORITY_TOPOLOGY_INCOMPLETE"
    )
    assert "selection_action_connected" in diagnostic.provenance[
        "unsatisfied_obligations"
    ]
    assert "SafetyResponseArbitration" in diagnostic.provenance[
        "obligation_affected_elements"
    ]["selection_action_connected"]


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


def test_priority_wiring_repair_can_succeed_while_input_obligation_stays_blocked():
    """The routed sub-obligation, not its aggregate diagnostic, is the target."""
    precedence = (
        "        require constraint "
        "precedence_PARACHUTE_DEPLOYMENT_over_LOW_BATTERY_RETURN_TO_BASE "
        "{ not criticalPropulsionFailureDetected or selectedResponse != "
        "FLIGHT_RESPONSES_V1::LOW_BATTERY_RETURN_TO_BASE }\n"
    )
    action = (
        "entry action "
        "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand;"
    )
    injured = GOOD.replace(precedence, "").replace(action, "")
    graph, report = _check(injured)
    priority = next(
        item for item in report.errors()
        if item.code == "PRIORITY_TOPOLOGY_INCOMPLETE"
    )
    assert {
        "precedence_edges", "selection_action_connected"
    } <= set(priority.provenance["unsatisfied_obligations"])
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"]
        if item.get("priority_obligation") == "selection_action_connected"
    )
    assert failure["repair_authorized"] is True

    board = Blackboard("Drone")
    board.commit_model(
        injured,
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
    repaired_behavior = _definition(
        GOOD, "state", "SafetyResponseArbitration"
    )
    decision = attempt_dependency_closed_ag_repair(
        llm=_RepairLLM(f"```sysml\n{repaired_behavior}\n```"),
        board=board,
        context_builder=ContextBuilder(board),
        sessions=TaskSessionRegistry(),
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )
    assert decision.status == "ACCEPTED", decision.reason
    assert decision.target_diagnostic_removed is True
    after = check_ag_graph(extract_ag_graph(board.current_model.model_text))
    remaining = next(
        item for item in after.errors()
        if item.code == "PRIORITY_TOPOLOGY_INCOMPLETE"
    )
    assert "selection_action_connected" not in remaining.provenance[
        "unsatisfied_obligations"
    ]
    assert "precedence_edges" in remaining.provenance[
        "unsatisfied_obligations"
    ]


def test_priority_repair_rejects_a_new_obligation_hidden_under_the_same_code():
    """Replacing one named defect with another is a regression, not a repair."""
    action = (
        "entry action "
        "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand;"
    )
    injured = GOOD.replace(action, "")
    graph, report = _check(injured)
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"]
        if item.get("priority_obligation") == "selection_action_connected"
    )

    board = Blackboard("Drone")
    board.commit_model(
        injured,
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
    regressing_behavior = _definition(
        GOOD, "state", "SafetyResponseArbitration"
    ).replace(
        "if not criticalPropulsionFailureDetected then "
        "CONTROLLED_BATTERY_LANDING;",
        "if criticalPropulsionFailureDetected then "
        "CONTROLLED_BATTERY_LANDING;",
    )
    decision = attempt_dependency_closed_ag_repair(
        llm=_RepairLLM(f"```sysml\n{regressing_behavior}\n```"),
        board=board,
        context_builder=ContextBuilder(board),
        sessions=TaskSessionRegistry(),
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )
    assert decision.status == "REJECTED"
    assert decision.target_diagnostic_removed is True
    assert decision.regression_free is False
    published = next(
        item.payload for item in reversed(board.records(topic="repair.decision"))
    )
    assert any(
        "obligation=competing_transitions_guarded" in item
        for item in published["gate"]["new_diagnostics"]
    )
    assert board.current_revision == 1


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
    from src.simulation.surgical_refiner import SURGICAL_SYSTEM_PROMPT
    assert "never translate it" in SURGICAL_SYSTEM_PROMPT
    assert "entry; then <state>;" in SURGICAL_SYSTEM_PROMPT


def test_preservation_guards_every_realizing_state_def_not_just_named_behaviors():
    """The preservation gate selected affected elements by the suffix "Behavior".

    The emitter names one state def differently — `SafetyResponseArbitration` — so
    that one was silently exempt from token preservation: a repair could shed a
    transition or an entry action from it and `behavior_preserved` stayed vacuously
    true. Selection is now by BEING a state def in the committed model.

    Staged so the loss is real: the patch restores the routed target and deletes a
    transition, which is a preserved token.
    """
    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package

    model = (
        "package Drone {\n    requirement def REQ_SAFE_005 { doc /* deploy the "
        "parachute within 0.5 seconds */ }\n}\n\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN)
    )
    arbitration = _definition(model, "state", "SafetyResponseArbitration")
    assert "SafetyResponseArbitration" in arbitration
    assert not arbitration.split("{")[0].strip().endswith("Behavior"), (
        "this test only means something while the emitter names this state def "
        "without the suffix the old filter required"
    )

    action = (
        "entry action setParachuteResponseSelectedAndIssue"
        "ParachuteDeploymentCommand;"
    )
    graph, report = _check(model.replace(action, ""))
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        (item for item in routed["failures"]
         if item.get("repair_authorized")
         and "SafetyResponseArbitration" in (item.get("affected_elements") or ())),
        None,
    )
    if failure is None:
        pytest.skip("this injury no longer routes through the arbitration state def")

    board = Blackboard("Drone")
    board.commit_model(
        model.replace(action, ""), base_revision=0,
        base_digest=board.current_model.model_digest, producer="test",
    )
    analysis = board.publish(
        RecordType.ANALYSIS, "analysis.ag_trace", "AGChecker", {"diagnostics": []},
    )
    failure_record = board.publish(
        RecordType.ANALYSIS, "diagnostic.failure", "AGFailureRouter", failure,
    )
    # restores the target action, but sheds a transition from the same state def
    import re as _re

    transition = _re.search(
        r"\n\s*transition select\w+ [^;]+;", arbitration
    )
    assert transition, "the reference arbitration must carry the selection transition"
    lossy = arbitration.replace(transition.group(0), "")
    decision = attempt_dependency_closed_ag_repair(
        llm=_RepairLLM(f"```sysml\n{lossy}\n```"),
        board=board,
        context_builder=ContextBuilder(board),
        sessions=TaskSessionRegistry(),
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )
    assert decision.status == "REJECTED", decision.reason
    assert decision.regression_free is False, (
        "a shed transition must be caught by preservation on a state def the old "
        "suffix filter exempted"
    )


def test_a_scoped_repair_is_judged_for_regression_not_for_finishing_the_chain():
    """The accept gate demanded `pattern verdict == PASS` outright.

    Measured: the routed task named only REALIZATION_ACTION_MISSING, the patch
    removed it, and the refusal came from PRIORITY_TOPOLOGY_INCOMPLETE — which was
    already present before the attempt. A bounded repair was therefore
    unacceptable no matter what it did. The condition is now "no worse than
    before"; breaking conformance is still refused, and the run verdict still
    reports the chain as failing.
    """
    from src.prototyping.ag_assurance import check_safety_pattern_conformance
    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package
    from src.prototyping.ag_extractor import extract_ag_graph

    model = (
        "package Drone {\n    requirement def REQ_SAFE_005 { doc /* deploy the "
        "parachute within 0.5 seconds */ }\n}\n\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN)
    )
    action = (
        "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand"
    )
    injured = model.replace(f"entry action {action};", "")
    graph = extract_ag_graph(injured)
    report = check_ag_graph(graph)
    codes = {item.code for item in report.errors()}
    assert {"REALIZATION_ACTION_MISSING", "PRIORITY_TOPOLOGY_INCOMPLETE"} <= codes
    # pattern conformance already fails BEFORE any repair, on a profile diagnostic
    # the routed task does not name
    assert check_safety_pattern_conformance(graph, report)["verdict"] == "FAIL"
