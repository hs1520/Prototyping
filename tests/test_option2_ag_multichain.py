"""Multi-chain R2-BBAG: a run that co-selects two student-approved chains.

The drone requirement set carries both REQ_SAFE_004 (startup inhibit) and
REQ_SAFE_005 (timed failsafe). Each is an independent A/G decomposition emitted
as its own package, so the committed model holds two system contracts. The A/G
trace must check each chain on its own graph and aggregate — pooling both system
contracts into one graph makes the decomposition root ambiguous (``system=None``)
and was a real regression once the second chain was added.
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

# The committed requirement-def doc must reproduce the authoritative stakeholder
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


def test_two_system_contracts_are_extracted_and_checked_per_chain():
    model = _two_chain_model()
    # The pooled single-graph extraction cannot disambiguate two roots.
    pooled = check_ag_graph(extract_ag_graph(model))
    assert pooled.verdict != "PASS"  # ambiguous system contract, honest failure

    # Per-chain extraction resolves each system independently and both PASS.
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


def test_orchestrator_aggregates_two_chains_into_one_passing_run():
    orch, artifacts, _ = _r2_two_chain_run()
    graph = artifacts["ag_contract_graph"]
    assert graph["multi_chain"] is True
    assert graph["chain_count"] == 2
    assert graph["verdict"] == "PASS"  # conjunction of both chains
    assert set(graph["source_requirements"]) == {"REQ_SAFE_005", "REQ_SAFE_004"}
    # each chain keeps its own full A/G graph for post-hoc evaluation
    assert {c["source_requirement"] for c in graph["chains"]} == {
        "REQ_SAFE_005", "REQ_SAFE_004",
    }
    assert artifacts["pattern_conformance_report"]["verdict"] == "PASS"
    assert artifacts["revised_experiment"]["runtime_assurance_status"] == "PASS"


def test_each_chain_publishes_its_own_analysis_record():
    orch, _artifacts, _ = _r2_two_chain_run()
    traces = [
        r for r in orch.blackboard.records(topic="analysis.ag_trace")
    ]
    # one independent assurance case per source requirement
    assert len(traces) == 2
    assert all(t.payload["verdict"] == "PASS" for t in traces)


def test_multichain_run_writes_per_chain_graph_artifacts(tmp_path):
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
