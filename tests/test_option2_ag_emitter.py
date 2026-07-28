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
    emitted = emit_ag_package(REQ_SAFE_005_CHAIN)
    result = check_syntax(
        emitted,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert result.has_errors is False
    assert result.score == 1.0
    assert "private import ScalarValues::*;" in emitted
    assert "private import ISQ::*;" in emitted
    assert "private import SI::*;" in emitted
    assert "attribute maxLatency : DurationValue = 0.5 [s];" in emitted


def test_r2_accepts_real_generated_base_shape_but_keeps_emitter_raw_gate():
    """The R2 gate must not reclassify R0/R1 stdlib-loader false positives.

    Real LLM models commonly use official ``transition initial`` and SI unit
    syntax without explicitly importing every standard namespace.  The shared
    generated-model gate allowlists only those known Syside diagnostics, while
    the deterministic A/G package remains subject to the unfiltered raw gate.
    """
    generated_shape = """
package Drone {
    requirement def REQ_SAFE_005 {
        doc /* The system shall deploy the parachute within 0.5 s. */
    }
    part def SafetyMonitor {
        attribute deploymentTime : Real = 0.5 [s];
        state def Monitor {
            state nominal;
            state failed;
            transition initial then nominal;
            transition detect first nominal then failed;
        }
    }
}
"""
    raw = check_syntax(
        generated_shape,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert raw.has_errors is True

    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    merged = orch._apply_ag_contract_layer(generated_shape, _REQS)

    shared_gate = check_syntax(
        merged,
        fail_closed=True,
        filter_stdlib_diagnostics=True,
    )
    assert shared_gate.has_errors is False
    assert shared_gate.score == 1.0
    assert "requirement def SystemParachuteContract" in merged


def test_r2_raw_gate_rejects_a_broken_deterministic_package(monkeypatch):
    import src.prototyping.ag_emitter as emitter

    monkeypatch.setattr(
        emitter,
        "emit_ag_package",
        lambda _spec: "package Broken { requirement def Missing {",
    )
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")

    with pytest.raises(
        RuntimeError, match="A/G package failed the raw syntax gate"
    ) as caught:
        orch._apply_ag_contract_layer(_BASE_MODEL, _REQS)
    message = str(caught.value)
    assert "diagnostics:" in message
    assert "parser L" in message


def test_a_warning_on_the_base_model_does_not_fail_the_arm_closed(monkeypatch):
    """A warning is not a syntax failure.

    The base-model gate rejected on `score != 1.0`, so a single warning — 0.05 of
    score, zero parser errors, zero sema errors — was enough to fail a whole arm
    closed. A measured seed lost its R2 evidence to it, with the self-contradicting
    diagnostic "failed the shared syntax gate: ✓ no syntax errors (score=0.950)".
    The same defect was removed from the A/G authoring loop once already.
    """
    import src.agents.orchestrator as orchestrator_module
    from src.simulation.syntax_checker import SyntaxCheckResult

    real = orchestrator_module.check_syntax

    def _warned(text, **kwargs):
        result = real(text, **kwargs)
        # only the BASE model earns the warning; the emitted package is left alone
        # so this pins the base-model gate and nothing else
        if result.has_errors or text != _BASE_MODEL:
            return result
        return SyntaxCheckResult(
            has_errors=False,
            parser_errors=[],
            sema_errors=[],
            warnings=[{"line": 1, "col": 1, "message": "style", "code": ""}],
            score=0.95,
        )

    monkeypatch.setattr(orchestrator_module, "check_syntax", _warned)
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")

    merged = orch._apply_ag_contract_layer(_BASE_MODEL, _REQS)
    assert "package REQ_SAFE_005_AG" in merged


def test_emitted_chain_round_trips_to_a_checker_pass():
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    report = check_ag_graph(extract_ag_graph(sysml, revision=1))
    assert report.verdict == "PASS", [d.code for d in report.diagnostics]
    assert report.timing["sum"] == 0.45
    assert report.timing["ok"] is True
    assert len(report.allocations) == 4
    assert {
        item["guarantee"] for item in report.allocations
        if item["contract"] == "SafetyResponseArbiterContract"
    } == {"parachuteDeploymentCommand", "parachuteResponseSelected"}
    prediction = report.to_dict()["graph"]
    assert prediction["timing"]["origin"] == "criticalPropulsionFailureDetected"
    assert {
        item["response"]
        for item in prediction["priority"]["member_provenance"]
    } == set(REQ_SAFE_005_CHAIN.priority.members)
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


def test_priority_topology_is_structural_not_bound_to_reviewed_element_names():
    """Equivalent authored names must not become a checker false positive."""
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    authored_names = (
        sysml
        .replace(
            "CriticalPropulsionFailureDetectedSignal",
            "criticalPropulsionFailureDetectedSignal",
        )
        .replace("parachuteDeploymentSelected", "ParachuteDeployment")
        .replace(
            "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand",
            "setparachuteResponseSelected",
        )
        .replace(
            "then CONTROLLED_BATTERY_LANDING;",
            "then ControlledBatteryLanding;",
        )
        .replace(
            "state CONTROLLED_BATTERY_LANDING;",
            "state ControlledBatteryLanding;",
        )
    )
    topology = extract_ag_graph(authored_names).priority[
        "arbitration_topology"
    ]

    assert topology["parachute_transition_reachable"] is True
    assert topology["selection_action_connected"] is True
    assert {
        item["response"]
        for item in topology["competing_transition_guards"]
    } >= {"CONTROLLED_BATTERY_LANDING"}


def test_wiring_is_checked_independently_of_an_incomplete_response_vocabulary():
    """A blocked enum-vocabulary fault must not manufacture a wiring fault."""
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    bare_enum_literals = sysml
    for member in REQ_SAFE_005_CHAIN.priority.members:
        bare_enum_literals = bare_enum_literals.replace(
            f"enum {member};", f"{member};"
        )
    graph = extract_ag_graph(bare_enum_literals)
    topology = graph.priority["arbitration_topology"]
    report = check_ag_graph(graph)
    priority = next(
        item for item in report.errors()
        if item.code == "PRIORITY_TOPOLOGY_INCOMPLETE"
    )
    unsatisfied = set(priority.provenance["unsatisfied_obligations"])

    assert graph.priority["members"] == []
    assert topology["parachute_transition_reachable"] is True
    assert topology["selection_action_connected"] is True
    assert "response_set_members" in unsatisfied
    assert "selected_transition_reachable" not in unsatisfied
    assert "selection_action_connected" not in unsatisfied


def test_priority_extractor_holds_no_reviewed_transition_or_action_names():
    import inspect
    from src.prototyping import ag_extractor

    source = inspect.getsource(ag_extractor._extract_priority).lower()
    for reviewed_name in (
        "criticalpropulsionfailuredetectedsignal",
        "parachutedeploymentselected",
        "parachutedeploymentcommand",
    ):
        assert reviewed_name not in source


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
        "{ entry action "
        "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand; }",
        "state parachuteDeploymentSelected;",
    )
    report = check_ag_graph(
        extract_ag_graph(missing_selection_action, revision=1)
    )
    assert report.verdict == "FAIL"
    assert "PRIORITY_TOPOLOGY_INCOMPLETE" in {
        diagnostic.code for diagnostic in report.diagnostics
    }


def test_current_checker_requires_priority_provenance_but_archive_replay_can_opt_out():
    sysml = _BASE_MODEL + "\n" + emit_ag_package(REQ_SAFE_005_CHAIN)
    without_provenance = "\n".join(
        line for line in sysml.splitlines()
        if "response_member=" not in line
    )
    graph = extract_ag_graph(without_provenance, revision=1)
    strict = check_ag_graph(graph)
    assert strict.verdict == "FAIL"
    assert any(
        "response_member_provenance"
        in diagnostic.provenance.get("unsatisfied_obligations", ())
        for diagnostic in strict.diagnostics
    )
    historical = check_ag_graph(
        graph, require_priority_member_provenance=False
    )
    assert historical.verdict == "PASS"


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
        (
            "require constraint g_parachuteResponseSelected "
            "{ parachuteResponseSelected }",
            "",
            "PRIORITY_TOPOLOGY_INCOMPLETE",
        ),
        (
            "state recoveryPowerAvailable "
            "{ entry action setRecoveryActuationPowerAvailable; }",
            "state recoveryPowerAvailable;",
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
    assert len(graph["graph"]["allocations"]) == 4
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
    assert len(graph["graph"]["allocations"]) == 4
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
