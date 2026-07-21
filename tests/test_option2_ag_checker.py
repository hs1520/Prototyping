"""R2-BBAG bounded Assume-Guarantee checker — Increment 2 tests.

Covers the worked REQ_SAFE_005 chain (design §7), the validated SysML convention
(§6.2), and the required-test invariants (§16): one owner per component guarantee,
assumption discharge, incompatible units/timing detection, and honest failure
localisation (a genuine gap is not mislabelled circular).
"""
from __future__ import annotations

from src.prototyping.ag_contracts import (
    AGEdge,
    AGGraph,
    Assumption,
    Contract,
    Guarantee,
    READY,
    INCOMPLETE,
    check_ag_graph,
    CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION,
    CODE_GUARANTEE_NO_OWNER,
    CODE_TIMING_BUDGET_EXCEEDED,
    CODE_UNIT_INCOMPATIBLE,
    CODE_DECOMPOSITION_INSUFFICIENT,
)
from src.prototyping.ag_extractor import extract_ag_graph
from src.simulation.syntax_checker import check_syntax


# REQ_SAFE_005 — critical propulsion failure → parachute deployment (design §7).
REQ_SAFE_005_SYSML = """package ParachuteAG {
    requirement def SystemParachuteContract {
        attribute airborne : Boolean;
        attribute criticalPropulsionFailure : Boolean;
        attribute parachuteDeployed : Boolean;
        attribute deploymentLatency : Real;
        attribute maxLatency : Real = 0.5;
        assume constraint env_airborne { airborne }
        assume constraint trig_failure { criticalPropulsionFailure }
        require constraint g_latency { deploymentLatency <= maxLatency }
        require constraint g_observed { parachuteDeployed }
    }
    requirement def PropulsionMonitorContract {
        attribute failureSensingAvailable : Boolean;
        attribute criticalFailureEvent : Boolean;
        attribute latencyBudget : Real = 0.05;
        assume constraint env_sensing { failureSensingAvailable }
        require constraint g_event { criticalFailureEvent }
    }
    requirement def SafetyMonitorContract {
        attribute airborne : Boolean;
        attribute criticalFailureEvent : Boolean;
        attribute parachuteCommand : Boolean;
        attribute latencyBudget : Real = 0.10;
        assume constraint a_airborne { airborne }
        assume constraint a_event { criticalFailureEvent }
        require constraint g_command { parachuteCommand }
    }
    requirement def RecoverySystemContract {
        attribute parachuteCommand : Boolean;
        attribute actuatorPower : Boolean;
        attribute parachuteDeployed : Boolean;
        attribute latencyBudget : Real = 0.35;
        assume constraint a_command { parachuteCommand }
        assume constraint env_power { actuatorPower }
        require constraint g_deployed { parachuteDeployed }
    }
    dependency decomposeProp from SystemParachuteContract to PropulsionMonitorContract;
    dependency decomposeSafety from SystemParachuteContract to SafetyMonitorContract;
    dependency decomposeRecovery from SystemParachuteContract to RecoverySystemContract;
}"""


def _codes(report):
    return {d.code for d in report.diagnostics}


def test_convention_passes_the_syside_gate():
    # §6.2: the bounded A/G convention must be valid SysML, not invented keywords.
    result = check_syntax(REQ_SAFE_005_SYSML)
    assert result.has_errors is False
    assert result.score == 1.0


def test_req_safe_005_chain_extracts_and_passes():
    graph = extract_ag_graph(REQ_SAFE_005_SYSML, revision=7)
    assert graph.system is not None
    assert graph.system.name == "SystemParachuteContract"
    assert graph.system.observation == "parachuteDeployed"
    assert {c.name for c in graph.components} == {
        "PropulsionMonitorContract", "SafetyMonitorContract", "RecoverySystemContract"
    }

    report = check_ag_graph(graph)
    assert report.verdict == "PASS", _codes(report)
    assert report.system_completeness == READY
    assert set(report.component_completeness.values()) == {READY}
    # additive timing: 0.05 + 0.10 + 0.35 == 0.5 == deadline
    assert report.timing["sum"] == 0.5
    assert report.timing["ok"] is True
    # every non-environment assumption is discharged; environment ones are marked
    assert report.discharge["SafetyMonitorContract.criticalFailureEvent"] == "discharged"
    assert report.discharge["RecoverySystemContract.actuatorPower"] == "environment"


def test_exactly_one_owner_per_component_guarantee():
    # §16: every component guarantee has exactly one responsible owner.
    graph = extract_ag_graph(REQ_SAFE_005_SYSML)
    owners = {c.name: 0 for c in graph.components}
    for e in graph.edges:
        if e.kind == "decomposes":
            owners[e.dst] += 1
    assert set(owners.values()) == {1}
    assert CODE_GUARANTEE_NO_OWNER not in _codes(check_ag_graph(graph))


def test_additive_timing_budget_exceeded_is_detected():
    over = REQ_SAFE_005_SYSML.replace(
        "attribute latencyBudget : Real = 0.35;",
        "attribute latencyBudget : Real = 0.40;",
    )
    report = check_ag_graph(extract_ag_graph(over))
    assert report.verdict == "FAIL"
    assert CODE_TIMING_BUDGET_EXCEEDED in _codes(report)
    assert report.timing["sum"] == 0.55


def test_undischarged_assumption_is_localised_and_not_mislabelled_circular():
    # Remove the upstream guarantee that produces criticalFailureEvent.
    broken = REQ_SAFE_005_SYSML.replace(
        "require constraint g_event { criticalFailureEvent }", ""
    )
    report = check_ag_graph(extract_ag_graph(broken))
    assert report.verdict == "FAIL"
    assert CODE_ASSUMPTION_UNDISCHARGED in _codes(report)
    # A cascade behind an upstream gap must not be reported as a cycle (§16).
    assert CODE_CIRCULAR_ASSUMPTION not in _codes(report)
    assert report.discharge["SafetyMonitorContract.criticalFailureEvent"] == "undischarged"
    assert report.component_completeness["PropulsionMonitorContract"] == INCOMPLETE


def test_missing_decomposition_owner_is_flagged():
    orphaned = REQ_SAFE_005_SYSML.replace(
        "    dependency decomposeRecovery from SystemParachuteContract to RecoverySystemContract;\n",
        "",
    )
    report = check_ag_graph(extract_ag_graph(orphaned))
    assert report.verdict == "FAIL"
    assert CODE_GUARANTEE_NO_OWNER in _codes(report)


def test_genuine_circular_assumption_is_detected():
    # A needs B's guarantee and B needs A's guarantee; neither is environment.
    a = Contract(
        name="A", role="component",
        assumptions=(Assumption(concept="b_sig", expr="b_sig", kind="boolean"),),
        guarantees=(Guarantee(concept="a_sig", expr="a_sig", kind="boolean"),),
        element_id="A",
    )
    b = Contract(
        name="B", role="component",
        assumptions=(Assumption(concept="a_sig", expr="a_sig", kind="boolean"),),
        guarantees=(Guarantee(concept="b_sig", expr="b_sig", kind="boolean"),),
        element_id="B",
    )
    system = Contract(
        name="Sys", role="system",
        assumptions=(Assumption(concept="start", expr="start", kind="boolean",
                                is_environment=True, constraint_name="env_start"),),
        guarantees=(Guarantee(concept="a_sig", expr="a_sig", kind="boolean"),),
        observation="a_sig", element_id="Sys",
    )
    graph = AGGraph(
        system=system, components=(a, b),
        edges=(AGEdge("decomposes", "Sys", "A"), AGEdge("decomposes", "Sys", "B")),
        revision=1,
    )
    report = check_ag_graph(graph)
    assert report.verdict == "FAIL"
    assert CODE_CIRCULAR_ASSUMPTION in _codes(report)


def test_incompatible_timing_units_are_detected():
    system = Contract(
        name="Sys", role="system", timing_budget=0.5, timing_unit="s",
        assumptions=(Assumption("x", "x", "boolean", is_environment=True,
                                constraint_name="env_x"),),
        guarantees=(Guarantee("y", "y", "boolean"),),
        observation="y", element_id="Sys",
    )
    comp = Contract(
        name="C", role="component", timing_budget=100.0, timing_unit="ms",
        assumptions=(Assumption("x", "x", "boolean", is_environment=True,
                                constraint_name="env_x"),),
        guarantees=(Guarantee("y", "y", "boolean"),),
        element_id="C",
    )
    graph = AGGraph(system=system, components=(comp,),
                    edges=(AGEdge("decomposes", "Sys", "C"),), revision=2)
    report = check_ag_graph(graph)
    assert CODE_UNIT_INCOMPATIBLE in _codes(report)
    assert report.verdict == "FAIL"


def test_report_cites_source_revision_digest_and_is_regenerable():
    graph = extract_ag_graph(REQ_SAFE_005_SYSML, revision=42, model_digest="deadbeef")
    report = check_ag_graph(graph)
    d = report.to_dict()
    assert d["source_model_revision"] == 42
    assert d["source_model_digest"] == "deadbeef"
    assert d["checker_version"] == report.checker_version
    assert d["artifact_role"] == "POSTHOC_A_G_TRACE"
    # deterministic: same input → identical derived view
    assert check_ag_graph(extract_ag_graph(REQ_SAFE_005_SYSML, revision=42,
                                           model_digest="deadbeef")).to_dict() == d


def test_decomposition_insufficient_when_observation_unproduced():
    # System observes 'z' but no component produces it.
    system = Contract(
        name="Sys", role="system", observation="z",
        assumptions=(Assumption("start", "start", "boolean", is_environment=True,
                                constraint_name="env_start"),),
        guarantees=(Guarantee("z", "z", "boolean"),), element_id="Sys",
    )
    comp = Contract(
        name="C", role="component",
        assumptions=(Assumption("start", "start", "boolean", is_environment=True,
                                constraint_name="env_start"),),
        guarantees=(Guarantee("other", "other", "boolean"),), element_id="C",
    )
    graph = AGGraph(system=system, components=(comp,),
                    edges=(AGEdge("decomposes", "Sys", "C"),), revision=1)
    report = check_ag_graph(graph)
    assert CODE_DECOMPOSITION_INSUFFICIENT in _codes(report)
