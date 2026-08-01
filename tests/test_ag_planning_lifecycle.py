"""Pre-generation A/G checks must not fabricate terminal implementation."""
from __future__ import annotations

import hashlib

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

_PLANNING_BYTES = {
    "REQ_SAFE_004": (
        3699,
        "8ab4c89d5cd1b870ef993e523b02334db849e01d2f81db01743f4e9f7911153a",
    ),
    "REQ_SAFE_005": (
        5629,
        "c6bef6f8f7e46b41b7d56cbebfea84635ba41d5cd3f61c1f733ad1051e822cc8",
    ),
    "REQ_SAFE_008": (
        2881,
        "59dd5dc376be8a6dfd21df43b1390fa4fa5990c8e162099a87406ceb613e07de",
    ),
}


@pytest.mark.parametrize("spec", select_ag_chains(_REQUIREMENTS))
def test_direct_planning_emission_preserves_historical_bytes(spec):
    payload = emit_ag_planning_package(spec).encode("utf-8")
    assert (len(payload), hashlib.sha256(payload).hexdigest()) == (
        _PLANNING_BYTES[spec.source_requirement]
    )


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
