"""Planning-only A/G artifacts and their bounded consistency gate.

Planning runs before a concrete system model exists, so it commits contract
decomposition decisions but claims no component owner or realizing behavior. This
module derives a planning-only SysML package from the deterministic emitter and
validates only the obligations meaningful at that lifecycle stage.

The terminal model stays the semantic authority; planning packages are generation
inputs and are not merged into the committed model verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from .ag_contracts import (
    AGDiagnostic,
    AGGraph,
    CODE_DECOMPOSITION_MISSING,
    CODE_SOURCE_PROVENANCE_MISSING,
    INCOMPLETE,
    UNSUPPORTED,
    _check_discharge,
    _check_sufficiency,
    _check_timing,
    _classify_completeness,
)
from .ag_emitter import AGChainSpec, emit_ag_package
from ..sysml.writer import EmissionMode


PLANNING_CONSISTENCY = "PLANNING_CONSISTENCY"


def emit_ag_planning_package(spec: AGChainSpec) -> str:
    """Render contract decisions without owner or realization claims."""
    # A planning artifact names the behavior obligation in the typed AGChainSpec,
    # but no concrete state definition exists until the main model is generated,
    # so realization edges would assert an absent behavior. The terminal binder
    # recreates those edges against qualified main-model paths.
    return emit_ag_package(spec, mode=EmissionMode.CONTRACTS_ONLY)

@dataclass(frozen=True)
class AGPlanningReport:
    verdict: str
    system_completeness: str
    component_completeness: Mapping[str, str]
    diagnostics: Tuple[AGDiagnostic, ...]
    timing: Mapping[str, Any]
    discharge: Mapping[str, str]
    profile: str = PLANNING_CONSISTENCY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_role": "A_G_PLANNING_CONSISTENCY",
            "profile": self.profile,
            "verdict": self.verdict,
            "system_completeness": self.system_completeness,
            "component_completeness": dict(self.component_completeness),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "timing": dict(self.timing),
            "discharge": dict(self.discharge),
        }


def check_ag_planning_graph(
    graph: AGGraph,
    *,
    aliases: Mapping[str, str] | None = None,
) -> AGPlanningReport:
    """Check obligations decidable before owner/behavior generation.

    Excludes ownership, behavior realization and executable pattern topology: those are
    terminal obligations, and treating them as pre-generation facts produced the former
    shadow model.
    """
    alias_map = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (aliases or {}).items()
    }
    diagnostics: List[AGDiagnostic] = list(graph.parse_diagnostics)

    if (
        graph.system
        and graph.system.source_requirement
        and graph.system.source_requirement not in graph.source_requirement_ids
    ):
        diagnostics.append(AGDiagnostic(
            CODE_SOURCE_PROVENANCE_MISSING,
            f"authoritative source requirement "
            f"{graph.system.source_requirement} is absent from planning inputs",
            contract=graph.system.name,
            subject=graph.system.source_requirement,
        ))

    system_completeness = UNSUPPORTED
    component_completeness: Dict[str, str] = {}
    if graph.system is not None:
        system_completeness, system_diags = _classify_completeness(
            graph.system,
            owner_count=None,
        )
        diagnostics.extend(system_diags)
    for component in graph.components:
        state, component_diags = _classify_completeness(
            component,
            owner_count=None,
        )
        component_completeness[component.name] = state
        diagnostics.extend(component_diags)
        decomposition_count = sum(
            edge.kind == "decomposes" and edge.dst == component.name
            for edge in graph.edges
        )
        if decomposition_count != 1:
            diagnostics.append(AGDiagnostic(
                CODE_DECOMPOSITION_MISSING,
                f"{component.name} must have exactly one planning decomposition "
                f"edge; found {decomposition_count}",
                contract=component.name,
            ))

    discharge, _edges, discharge_diags = _check_discharge(graph, alias_map)
    diagnostics.extend(discharge_diags)
    timing, timing_diags = _check_timing(graph)
    diagnostics.extend(timing_diags)
    diagnostics.extend(
        _check_sufficiency(graph, discharge_diags, alias_map)
    )

    incomplete = (
        system_completeness in {INCOMPLETE, UNSUPPORTED}
        or any(
            value in {INCOMPLETE, UNSUPPORTED}
            for value in component_completeness.values()
        )
    )
    if any(item.severity == "error" for item in diagnostics):
        verdict = "FAIL"
    elif incomplete:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    return AGPlanningReport(
        verdict=verdict,
        system_completeness=system_completeness,
        component_completeness=component_completeness,
        diagnostics=tuple(diagnostics),
        timing=timing,
        discharge=discharge,
    )
