"""A/G-aware generation (Stage 2-3) — emitter, chain library, R2 integration.

The selected decomposition candidate renders to valid SysML and round-trips to a checker
PASS, and under R2-BBAG the orchestrator merges it so the committed model carries
the contracts and the A/G trace is non-empty. R0/R1 models never carry them.
"""
from __future__ import annotations

import pytest

from src.agents.orchestrator import Orchestrator
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN, select_ag_chains
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package, merge_ag_contracts
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.blackboard import Blackboard
from src.prototyping.context_builder import ContextBuilder
from src.prototyping.task_session import TaskSessionRegistry
from src.simulation.syntax_checker import check_syntax


class _NoCallLLM:
    def complete(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("LLM should not be called")


_BASE_MODEL = (
    "package Drone {\n"
    "    requirement def REQ_SAFE_005 { doc /* The system shall deploy the "
    "parachute within 0.5 s. */ }\n"
    "    part def SafetyMonitor {}\n}"
)
_REQS = ["REQ-SAFE-005: The system shall deploy the parachute within 0.5 s."]


def test_emitted_chain_passes_the_syside_gate():
    result = check_syntax(emit_ag_package(REQ_SAFE_005_CHAIN))
    assert result.has_errors is False
    assert result.score == 1.0


def test_emitted_chain_round_trips_to_a_checker_pass():
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    report = check_ag_graph(extract_ag_graph(sysml, revision=1))
    assert report.verdict == "PASS", [d.code for d in report.diagnostics]
    assert report.timing["sum"] == 0.45
    assert report.timing["ok"] is True
    assert len(report.allocations) == 3
    prediction = report.to_dict()["graph"]
    assert prediction["timing"]["origin"] == "criticalPropulsionFailureDetected"
    assert prediction["priority"]["arbitration_topology"][
        "parachute_transition_reachable"
    ] is True

    unguarded = sysml.replace(
        "if criticalPropulsionFailureDetected then parachuteDeploymentSelected",
        "then parachuteDeploymentSelected",
    )
    unguarded_prediction = check_ag_graph(
        extract_ag_graph(unguarded, revision=1)
    ).to_dict()["graph"]
    assert unguarded_prediction["priority"]["arbitration_topology"][
        "parachute_transition_reachable"
    ] is False


def test_runtime_checker_fails_closed_when_priority_semantics_are_removed():
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    without_priority = sysml.replace(
        "requirement def SafetyResponsePriorityContract",
        "requirement def RemovedPriorityContract",
    )
    report = check_ag_graph(extract_ag_graph(without_priority, revision=1))
    assert report.verdict == "FAIL"
    assert "PRIORITY_TOPOLOGY_MISSING" in {
        diagnostic.code for diagnostic in report.diagnostics
    }

    missing_selection_action = sysml.replace(
        "state parachuteDeploymentSelected "
        "{ entry action issueParachuteDeploymentCommand; }",
        "state parachuteDeploymentSelected;",
    )
    report = check_ag_graph(
        extract_ag_graph(missing_selection_action, revision=1)
    )
    assert report.verdict == "FAIL"
    assert "PRIORITY_TOPOLOGY_INCOMPLETE" in {
        diagnostic.code for diagnostic in report.diagnostics
    }


@pytest.mark.parametrize(
    ("before", "after", "expected_code"),
    [
        (
            "state def SafetyResponseArbitration",
            "state def RemovedSafetyResponseArbitration",
            "PRIORITY_TOPOLOGY_MISSING",
        ),
        (
            "if not criticalPropulsionFailureDetected then "
            "CONTROLLED_BATTERY_LANDING;",
            "then CONTROLLED_BATTERY_LANDING;",
            "PRIORITY_TOPOLOGY_INCOMPLETE",
        ),
        (
            "transition selectParachute first awaitingResponse "
            "accept CriticalPropulsionFailureDetectedSignal "
            "if criticalPropulsionFailureDetected "
            "then parachuteDeploymentSelected;",
            "",
            "PRIORITY_TOPOLOGY_INCOMPLETE",
        ),
        (
            "state deployed { entry action setParachuteDeployed; }",
            "state deployed;",
            "PRIORITY_TOPOLOGY_INCOMPLETE",
        ),
        (
            "dependency observeSystemParachuteContract "
            "from SystemParachuteContract "
            "to ParachuteDeploymentVerification;",
            "",
            "PRIORITY_TOPOLOGY_INCOMPLETE",
        ),
    ],
)
def test_safe005_checker_rejects_each_missing_priority_topology_fact(
    before, after, expected_code
):
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    assert before in sysml
    report = check_ag_graph(
        extract_ag_graph(sysml.replace(before, after), revision=1)
    )
    assert report.verdict == "FAIL"
    assert expected_code in {diagnostic.code for diagnostic in report.diagnostics}


def test_merge_preserves_the_base_model_and_stays_valid():
    merged = merge_ag_contracts(_BASE_MODEL, [REQ_SAFE_005_CHAIN])
    assert "part def SafetyMonitor" in merged  # base preserved verbatim
    assert "requirement def SystemParachuteContract" in merged
    assert check_syntax(merged).has_errors is False
    graph = extract_ag_graph(merged, revision=1)
    assert graph.system is not None
    assert len(graph.components) == 3


def test_select_ag_chains_matches_only_present_source_requirements():
    assert select_ag_chains(_REQS) == (REQ_SAFE_005_CHAIN,)
    assert select_ag_chains(["REQ-FUNC-001: unrelated"]) == ()
    assert select_ag_chains([]) == ()


def test_r2_orchestrator_merges_ag_layer_and_emits_non_empty_pass_trace():
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    merged = orch._apply_ag_contract_layer(_BASE_MODEL, _REQS)
    assert "SystemParachuteContract" in merged

    orch.blackboard = Blackboard("Drone")
    orch.context_builder = ContextBuilder(orch.blackboard)
    orch.task_sessions = TaskSessionRegistry()
    orch.blackboard.commit_model(
        merged,
        base_revision=orch.blackboard.current_revision,
        base_digest=orch.blackboard.current_model.model_digest,
        producer="test",
    )
    graph = orch._build_collaboration_artifacts(merged)["ag_contract_graph"]
    assert graph["verdict"] == "PASS"
    assert len(graph["graph"]["allocations"]) == 3
    assert len(graph["graph"]["discharge_edges"]) == 5


def test_r1_and_r0_never_apply_the_ag_contract_layer():
    r1 = Orchestrator(_NoCallLLM(), revised_experiment_arm="R1-BBCTX")
    assert r1._apply_ag_contract_layer(_BASE_MODEL, _REQS) == _BASE_MODEL
    r0 = Orchestrator(_NoCallLLM())  # no revised arm
    assert r0._apply_ag_contract_layer(_BASE_MODEL, _REQS) == _BASE_MODEL


# --- End-to-end R2-BBAG orchestrator seam (the sequence generate() runs) -------
# MockLLM cannot synthesise valid multi-step SysML, so a full mock-driven
# generate() is not viable; this drives the real R2 integration methods that
# generate() calls (prepare -> finalize handoff -> A/G layer -> commit ->
# collaboration artifacts) with a realistic committed model, deterministically.

def test_r2_end_to_end_orchestrator_seam_produces_full_evidence_chain():
    from types import SimpleNamespace

    from src.prototyping.requirement_inputs import build_frozen_requirement_set
    from src.sysml.lite_model import build_lite_model
    from src.utils.sysml_text_utils import get_sysml_text

    reqs = [
        "REQ-SAFE-005: The system shall deploy the parachute within 0.5 seconds "
        "of a critical propulsion failure. [SEV:Catastrophic]",
        "REQ-FUNC-001: The system shall detect obstacles within 15 m.",
    ]
    artifact = build_frozen_requirement_set(reqs, name="r2-e2e", source="test")

    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    orch.last_requirement_input = {
        "mode": "frozen",
        "requirement_set_digest": artifact.get("requirement_set_digest"),
    }

    # Phase 2 seam: real blackboard/context/session handoff, then a committed model.
    orch._prepare_design_handoff("DeliveryUAV", reqs)
    design_model = build_lite_model(
        "package DeliveryUAV {\n"
        "    requirement def REQ_SAFE_005 { doc /* The system shall deploy the "
        "parachute within 0.5 seconds of a critical propulsion failure. "
        "[SEV:Catastrophic] */ }\n"
        "    requirement def REQ_FUNC_001 { doc /* The system shall detect "
        "obstacles within 15 m. */ }\n"
        "    part def SafetyMonitor { satisfy requirement REQ_SAFE_005; }\n"
        "    part def RecoverySystem {}\n"
        "}",
        model_name="DeliveryUAV",
    )
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}),
        design_model,
    )

    # generate()/explore() tail seam.
    final_sysml = get_sysml_text(design_model)
    final_sysml = orch._apply_ag_contract_layer(final_sysml, reqs)
    orch._commit_terminal_model(final_sysml, producer="smoke")
    artifacts = orch._build_collaboration_artifacts(final_sysml)

    # A/G layer merged into the committed authority model.
    assert "requirement def SystemParachuteContract" in final_sysml
    assert orch.blackboard.current_model.model_text == final_sysml

    # Non-empty PASS A/G trace attached to the run artifacts.
    graph = artifacts["ag_contract_graph"]
    assert graph["verdict"] == "PASS"
    assert len(graph["graph"]["allocations"]) == 3
    assert graph["source_model_revision"] == orch.blackboard.current_revision

    # Honest experiment metadata: runnable R2, not gold-poolable.
    assert artifacts["revised_experiment"]["configuration"] == "R2-BBAG"
    assert artifacts["revised_experiment"]["evaluation_ready"] is False

    # The full typed record chain is on the board: source -> design result ->
    # model revisions -> A/G analysis, all under the SysML authority.
    records = artifacts["collaboration"]["blackboard"]["records"]
    topics = {r["topic"] for r in records}
    assert "requirements.authoritative" in topics
    assert "agent.design.result" in topics
    assert "analysis.ag_trace" in topics
    ag_rec = next(r for r in records if r["topic"] == "analysis.ag_trace")
    assert ag_rec["record_type"] == "ANALYSIS"
    assert ag_rec["payload"]["verdict"] == "PASS"
    assert ag_rec["model_revision"] == orch.blackboard.current_revision
    sessions = artifacts["collaboration"]["task_sessions"]["sessions"]
    assert sessions[0]["status"] == "COMPLETED"
