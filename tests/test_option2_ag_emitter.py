"""A/G-aware generation (Stage 2-3) — emitter, chain library, R2 integration.

The reviewed decomposition renders to valid SysML that round-trips to a checker
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


_BASE_MODEL = "package Drone {\n    part def SafetyMonitor {}\n}"
_REQS = ["REQ-SAFE-005: The system shall deploy the parachute within 0.5 s."]


def test_emitted_chain_passes_the_syside_gate():
    result = check_syntax(emit_ag_package(REQ_SAFE_005_CHAIN))
    assert result.has_errors is False
    assert result.score == 1.0


def test_emitted_chain_round_trips_to_a_checker_pass():
    sysml = emit_ag_package(REQ_SAFE_005_CHAIN)
    report = check_ag_graph(extract_ag_graph(sysml, revision=1))
    assert report.verdict == "PASS", [d.code for d in report.diagnostics]
    assert report.timing["sum"] == 0.5
    assert report.timing["ok"] is True
    assert len(report.allocations) == 3


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
        merged, base_revision=orch.blackboard.current_revision, producer="test"
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
