"""Pre-generation A/G checks must not fabricate terminal implementation."""
from __future__ import annotations

import pytest

from src.agents.orchestrator import Orchestrator
from src.prototyping.ag_chains import select_ag_chains
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_planning import (
    PLANNING_CONSISTENCY,
    check_ag_planning_graph,
    emit_ag_planning_package,
)
from src.simulation.syntax_checker import check_syntax


_REQUIREMENTS = [
    "REQ-SAFE-004: inhibit arming and flight after a failed power-on self-test.",
    "REQ-SAFE-005: deploy the parachute within 0.5 seconds after a critical "
    "propulsion failure.",
    "REQ-SAFE-008: keep the payload mechanically locked until authorised.",
]


@pytest.mark.parametrize("spec", select_ag_chains(_REQUIREMENTS))
def test_planning_package_has_contracts_but_no_shadow_implementation(spec):
    package = emit_ag_planning_package(spec)

    assert f"requirement def {spec.system_contract}" in package
    assert "part def " not in package
    assert "satisfy requirement" not in package
    assert "state def " not in package
    assert "dependency realize" not in package


@pytest.mark.parametrize("spec", select_ag_chains(_REQUIREMENTS))
def test_planning_profile_passes_without_claiming_terminal_realization(spec):
    inputs = Orchestrator._requirement_planning_model(_REQUIREMENTS)
    candidate = inputs + "\n" + emit_ag_planning_package(spec)
    graph = extract_ag_graph(candidate)

    planning = check_ag_planning_graph(graph)
    terminal = check_ag_graph(graph)
    syntax = check_syntax(
        candidate,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )

    assert not syntax.has_errors
    assert planning.profile == PLANNING_CONSISTENCY
    assert planning.verdict == "PASS", [
        item.as_dict() for item in planning.diagnostics
    ]
    assert terminal.verdict != "PASS"


def test_planning_profile_still_rejects_missing_decomposition():
    spec = select_ag_chains(_REQUIREMENTS)[0]
    inputs = Orchestrator._requirement_planning_model(_REQUIREMENTS)
    package = emit_ag_planning_package(spec)
    package = package.replace(
        f"dependency decompose{spec.components[0].name} "
        f"from {spec.system_contract} to {spec.components[0].name};",
        "",
    )

    report = check_ag_planning_graph(
        extract_ag_graph(inputs + "\n" + package)
    )

    assert report.verdict == "FAIL"
    assert any(
        item.code == "DECOMPOSITION_MISSING"
        for item in report.diagnostics
    )
