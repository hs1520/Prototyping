"""Multi-chain R2-BBAG: a run that co-selects two student-approved chains.

REQ_SAFE_004 (startup inhibit) and REQ_SAFE_005 (timed failsafe) each emit their
own A/G package, so the committed model holds two system contracts. The trace
checks each chain on its own graph and aggregates; pooling both into one graph
makes the decomposition root ambiguous (``system=None``).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_extractor import extract_ag_graph, extract_ag_graphs
from src.prototyping.ag_chains import REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.run_artifacts import write_revised_run_artifacts
from src.simulation.syntax_checker import check_syntax
from src.utils.sysml_text_utils import get_sysml_text
from src.sysml.lite_model import build_lite_model


class _NoCallLLM:
    def complete(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("LLM should not be called")


_REQS = [
    "REQ-SAFE-004: do not arm if a sensor fails the power-on self-test.",
    "REQ-SAFE-005: deploy the parachute within 0.5 s of a critical propulsion "
    "failure.",
]

# The committed requirement-def doc reproduces the authoritative stakeholder
# text verbatim (Blackboard protects source text), so each doc mirrors _REQS.
_BASE_MODEL = (
    "package DeliveryUAV { "
    "requirement def REQ_SAFE_004 { doc /* do not arm if a sensor fails the "
    "power-on self-test. */ } "
    "requirement def REQ_SAFE_005 { doc /* deploy the parachute within 0.5 s of a "
    "critical propulsion failure. */ } "
    "part def SafetyMonitor {} }"
)


def _two_chain_model() -> str:
    return (
        _BASE_MODEL + "\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN) + "\n"
        + emit_ag_package(REQ_SAFE_004_CHAIN)
    )


def test_two_contracts_checked_per_chain():
    model = _two_chain_model()
    # The pooled single-graph extraction cannot disambiguate two roots.
    pooled = check_ag_graph(extract_ag_graph(model))
    assert pooled.verdict != "PASS"

    graphs = extract_ag_graphs(model, revision=1)
    assert len(graphs) == 2
    by_source = {}
    for graph in graphs:
        report = check_ag_graph(graph)
        by_source[report.source_requirement] = report.verdict
    assert by_source == {"REQ_SAFE_005": "PASS", "REQ_SAFE_004": "PASS"}


def _r2_two_chain_run():
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", _REQS)
    model = build_lite_model(_BASE_MODEL, model_name="DeliveryUAV")
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}), model
    )
    final_sysml = orch._apply_ag_contract_layer(get_sysml_text(model), _REQS)
    orch._commit_terminal_model(final_sysml, producer="test")
    artifacts = orch._build_collaboration_artifacts(final_sysml)
    return orch, artifacts, {
        **artifacts,
        "model_sysml": final_sysml,
        "llm_usage": {"calls": 3, "total_tokens": 4200, "elapsed_seconds": 5.0},
    }


def test_orchestrator_aggregates_two_chains():
    orch, artifacts, _ = _r2_two_chain_run()
    graph = artifacts["ag_contract_graph"]
    assert graph["multi_chain"] is True
    assert graph["chain_count"] == 2
    assert graph["verdict"] == "PASS"
    assert set(graph["source_requirements"]) == {"REQ_SAFE_005", "REQ_SAFE_004"}
    assert {c["source_requirement"] for c in graph["chains"]} == {
        "REQ_SAFE_005", "REQ_SAFE_004",
    }
    assert artifacts["pattern_conformance_report"]["verdict"] == "PASS"
    assert artifacts["revised_experiment"]["runtime_assurance_status"] == "PASS"


def test_each_chain_publishes_record():
    orch, _artifacts, _ = _r2_two_chain_run()
    traces = [
        r for r in orch.blackboard.records(topic="analysis.ag_trace")
    ]
    assert len(traces) == 2
    assert all(t.payload["verdict"] == "PASS" for t in traces)


def test_per_chain_graph_artifacts(tmp_path):
    _orch, _artifacts, result = _r2_two_chain_run()
    written = write_revised_run_artifacts(result, tmp_path)
    names = set(written)
    assert "ag_contract_graph" in names
    assert "ag_contract_graph.REQ_SAFE_004" in names
    assert "ag_contract_graph.REQ_SAFE_005" in names
    per_chain = json.loads(
        (tmp_path / "ag_contract_graph.REQ_SAFE_004.json").read_text()
    )
    assert per_chain["source_requirement"] == "REQ_SAFE_004"
    assert per_chain["verdict"] == "PASS"
    manifest = json.loads(
        (tmp_path / "ag_replay_manifest.json").read_text()
    )
    assert manifest["artifact_role"] == "A_G_REPLAY_BUNDLE_MANIFEST"
    assert {
        item["source_requirement"] for item in manifest["bundles"]
    } == {"REQ_SAFE_004", "REQ_SAFE_005"}
    for item in manifest["bundles"]:
        bundle = tmp_path / item["path"]
        assert bundle.read_text() == result["model_sysml"]
        gate = check_syntax(
            bundle.read_text(),
            fail_closed=True,
            filter_stdlib_diagnostics=False,
        )
        assert not gate.has_errors
        assert not gate.warnings


class _FixingLLM:
    def __init__(self, replacement: str):
        self._replacement = replacement
        self.calls = 0

    def chat(self, _prompt, system_prompt="", **_kwargs):
        self.calls += 1
        return f"```sysml\n{self._replacement}\n```"

    def complete(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("repair uses chat()")


class _SequenceLLM:
    def __init__(self, replacements):
        self._replacements = iter(replacements)
        self.calls = 0
        self.prompts = []

    def chat(self, prompt, system_prompt="", **_kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return f"```sysml\n{next(self._replacements)}\n```"

    def complete(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("repair uses chat()")


def _injured_two_chain_model() -> tuple[str, str]:
    import re

    model = _two_chain_model()
    match = re.search(
        r"state (\w+) \{ entry action (setArmingTransitionInhibited); \}", model
    )
    assert match, "expected REQ_SAFE_004's arming response action"
    healthy = match.group(0)
    return model.replace(healthy, f"state {match.group(1)};"), healthy


def test_repair_runs_in_multichain():
    """§15 Increment 3's exit gate, previously unreachable in every pilot.

    Repair was disabled whenever several chains co-existed, so every archived repair
    decision read `multi_chain_auto_repair_out_of_scope`. Repair is now per chain as
    a fixpoint: an accepted repair commits a revision that invalidates every other
    chain's graph, so each round re-extracts and re-checks all chains.
    """
    from src.simulation.surgical_refiner import _find_def_span

    injured, _healthy_state = _injured_two_chain_model()
    span = _find_def_span(_two_chain_model(), "state", "ArmingAuthorityBehavior")
    assert span is not None
    llm = _FixingLLM(_two_chain_model()[span[0]:span[1]])

    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG")
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", _REQS)
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}),
        build_lite_model(_BASE_MODEL, model_name="DeliveryUAV"),
    )
    orch._commit_terminal_model(injured, producer="test")
    artifacts = orch._build_collaboration_artifacts(injured)

    decisions = artifacts["repair_decisions"]["decisions"]
    assert not any(
        item.get("reason") == "multi_chain_auto_repair_out_of_scope"
        for item in decisions
    ), "the multi-chain exemption must be gone"
    accepted = [item for item in decisions if item["status"] == "ACCEPTED"]
    assert accepted, f"the repair must be attempted and accepted: {decisions}"
    assert llm.calls >= 1

    assert accepted[0]["committed_model_revision"] == orch.blackboard.current_revision
    history = artifacts["failure_diagnostics"]["analysis_history"]
    assert {item["analysis_round"] for item in history} == {0, 1}
    round_one = [item for item in history if item["analysis_round"] == 1]
    assert len(round_one) == 2, "both chains re-checked, not only the repaired one"
    assert all(
        item["source_model_revision"] == orch.blackboard.current_revision
        for item in round_one
    ), "a re-check must run on the committed revision, not the superseded one"

    graph = artifacts["ag_contract_graph"]
    assert graph["source_model_revision"] == orch.blackboard.current_revision
    assert graph["verdict"] == "PASS", "the repaired chain now passes"


def test_repair_extracts_routed_chain():
    from src.prototyping.ag_repair import _extract_routed_ag_graph

    model = _two_chain_model()
    safe_004 = _extract_routed_ag_graph(model, "REQ_SAFE_004")
    safe_005 = _extract_routed_ag_graph(model, "REQ_SAFE_005")

    assert safe_004.system.source_requirement == "REQ_SAFE_004"
    assert safe_005.system.source_requirement == "REQ_SAFE_005"
    assert safe_004.system.name != safe_005.system.name


def test_repair_budget_fixed_and_audited():
    injured, _healthy = _injured_two_chain_model()
    llm = _FixingLLM("state def ArmingAuthorityBehavior { entry; then idle; }")

    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG")
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", _REQS)
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}),
        build_lite_model(_BASE_MODEL, model_name="DeliveryUAV"),
    )
    orch._commit_terminal_model(injured, producer="test")
    artifacts = orch._build_collaboration_artifacts(injured)

    decisions = artifacts["repair_decisions"]["decisions"]
    assert sum(
        1 for item in decisions
        if item["status"] in {"ACCEPTED", "REJECTED"}
    ) <= orch.maximum_ag_repair_attempts


def test_rejected_chain_no_suppression():
    from src.simulation.surgical_refiner import _find_def_span

    healthy = _two_chain_model()
    injured, _ = _injured_two_chain_model()
    injured = injured.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed;",
    )
    recovery_span = _find_def_span(
        injured, "state", "RecoverySystemBehavior"
    )
    arming_span = _find_def_span(
        healthy, "state", "ArmingAuthorityBehavior"
    )
    assert recovery_span is not None and arming_span is not None
    llm = _SequenceLLM([
        injured[recovery_span[0]:recovery_span[1]],
        injured[recovery_span[0]:recovery_span[1]],
        healthy[arming_span[0]:arming_span[1]],
    ])
    orch = Orchestrator(
        llm,
        revised_experiment_arm="R2-BBAG",
        maximum_ag_repair_attempts=3,
    )
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", _REQS)
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}),
        build_lite_model(_BASE_MODEL, model_name="DeliveryUAV"),
    )
    orch._commit_terminal_model(injured, producer="test")
    artifacts = orch._build_collaboration_artifacts(injured)

    decisions = artifacts["repair_decisions"]["decisions"]
    assert any(item["status"] == "REJECTED" for item in decisions)
    assert any(item["status"] == "ACCEPTED" for item in decisions)
    assert llm.calls == 3
    assert "Previous bounded repair was rejected" in llm.prompts[1]
    assert "target_removed=False" in llm.prompts[1]
    assert all(
        task["status"] != "PENDING"
        for task in artifacts["collaboration"]["blackboard"]["tasks"]
        if task["kind"] == "A_G_SURGICAL_REPAIR"
    )
