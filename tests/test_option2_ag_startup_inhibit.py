"""Second reviewed A/G chain: REQ_SAFE_004 startup self-test -> arming inhibit.

Exercises a structurally different property KIND (a Boolean unreachability
invariant, not a timed chain) and a second safety pattern (STARTUP_INHIBIT),
which is the strongest available generality evidence for the bounded A/G method.
"""
from __future__ import annotations

from src.prototyping.ag_assurance import (
    check_safety_pattern_conformance,
    route_failure_diagnostics,
)
from src.prototyping.ag_chains import (
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_005_CHAIN,
    select_ag_chains,
)
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph
from src.simulation.syntax_checker import check_syntax

_SRC = (
    "The system shall not transition to the armed or airborne state if any "
    "onboard sensor reports a failure during the power-on self-test sequence."
)


def _model() -> str:
    # The stakeholder requirement def supplies the source provenance the checker
    # requires; the A/G package carries the reviewed decomposition + behavior.
    return (
        "package Src { requirement def REQ_SAFE_004 { doc /* " + _SRC + " */ } }\n"
        + emit_ag_package(REQ_SAFE_004_CHAIN)
    )


def test_startup_inhibit_chain_is_valid_sysml_and_passes_the_ag_trace():
    model = _model()
    syntax = check_syntax(model)
    assert syntax.has_errors is False
    assert syntax.score == 1.0

    report = check_ag_graph(extract_ag_graph(model, revision=1))
    assert report.verdict == "PASS", [d.code for d in report.diagnostics]
    # it is an invariant, not a timed chain: no timing composition applies
    assert report.timing["ok"] is None
    assert len(report.allocations) == 2
    assert len(report.discharge_edges) == 4


def test_startup_inhibit_pattern_conformance_passes_without_a_timing_criterion():
    report = check_ag_graph(extract_ag_graph(_model(), revision=1))
    pattern = check_safety_pattern_conformance(
        extract_ag_graph(_model(), revision=1), report
    )
    assert pattern["verdict"] == "PASS"
    assert {c["pattern"] for c in pattern["cases"]} == {"STARTUP_INHIBIT"}
    # the whole point: a startup invariant conforms with no timing obligation
    assert all(c["timing_criterion_present"] is False for c in pattern["cases"])
    assert all(c["status"] == "PASS" for c in pattern["cases"])


def test_the_two_chains_use_two_distinct_patterns():
    assert REQ_SAFE_005_CHAIN.pattern == "TRIGGERED_TIMED_FAILSAFE_RESPONSE"
    assert REQ_SAFE_004_CHAIN.pattern == "STARTUP_INHIBIT"
    # both are selected when both requirements are present in a run
    chosen = select_ag_chains([
        "REQ-SAFE-005: deploy parachute", "REQ-SAFE-004: prevent arming",
    ])
    assert {c.source_requirement for c in chosen} == {"REQ_SAFE_005", "REQ_SAFE_004"}


def test_startup_inhibit_pattern_needs_real_topology_not_a_label():
    # Strip the realizing behavior: pattern conformance must FAIL (a pattern is
    # never a PASS by selection alone, §16), and routing flags the model faults.
    model = _model()
    # remove every realization edge so no behavior is reachable
    broken = "\n".join(
        line for line in model.splitlines()
        if "dependency realize" not in line
    )
    report = check_ag_graph(extract_ag_graph(broken, revision=1))
    pattern = check_safety_pattern_conformance(
        extract_ag_graph(broken, revision=1), report
    )
    assert pattern["verdict"] == "FAIL"
    routes = route_failure_diagnostics(
        report.diagnostics, source_requirement="REQ_SAFE_004",
        realization_links=report.realization_links,
    )
    assert len(routes["failures"]) >= 1
