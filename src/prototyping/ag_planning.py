"""Planning-only A/G artifacts and their bounded consistency gate.

Pre-generation planning happens before a concrete system model exists.  It may
therefore commit contract decomposition decisions, but it must not claim that a
component owner or realizing behavior already exists.  This module derives a
planning-only SysML package from the deterministic emitter and validates only
the obligations that are meaningful at that lifecycle stage.

The terminal model remains the semantic authority.  Planning packages are
generation inputs and must never be merged into the committed model verbatim.
"""
from __future__ import annotations

import re
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
from .ag_emitter import AGChainSpec, ag_event_signals, emit_ag_package
from ..utils.sysml_text_utils import find_block_end


PLANNING_CONSISTENCY = "PLANNING_CONSISTENCY"


def _remove_named_block(text: str, keyword: str, name: str) -> str:
    pattern = re.compile(
        rf"\b{re.escape(keyword)}\s+{re.escape(name)}\s*\{{"
    )
    result = str(text)
    while True:
        match = pattern.search(result)
        if match is None:
            return result
        brace = result.find("{", match.start())
        end = find_block_end(result, brace)
        if end == -1:
            return result
        result = result[:match.start()] + result[end + 1:]


def strip_ag_implementation(
    package_text: str, spec: AGChainSpec
) -> str:
    """Remove implementation claims from an emitted or authored A/G package."""
    text = str(package_text)

    # A planning artifact may describe the behavior obligation by name in the
    # typed AGChainSpec, but no concrete state definition exists until the main
    # model has been generated.
    for behavior in dict.fromkeys(
        component.behavior for component in spec.components
    ):
        text = _remove_named_block(text, "state def", behavior)

    for component in spec.components:
        text = re.sub(
            rf"(?m)^\s*part\s+def\s+"
            rf"{re.escape(component.owner_def)}\s*;\s*\n?",
            "",
            text,
        )
        text = re.sub(
            rf"(?m)^\s*part\s+{re.escape(component.owner_usage)}\s*:\s*"
            rf"{re.escape(component.owner_def)}\s*;\s*\n?",
            "",
            text,
        )
        text = re.sub(
            rf"(?m)^\s*satisfy\s+requirement\s+\w+\s*:\s*"
            rf"{re.escape(component.name)}\s+by\s+"
            rf"{re.escape(component.owner_usage)}\s*;\s*\n?",
            "",
            text,
        )

    # Realization edges would assert that the removed behavior exists.  The
    # terminal binder recreates these edges against qualified main-model paths.
    text = re.sub(
        r"(?m)^\s*dependency\s+realize\w+\s+from\s+\w+\s+to\s+\w+\s*;\s*\n?",
        "",
        text,
    )
    return text


def emit_ag_planning_package(spec: AGChainSpec) -> str:
    """Render contract decisions without fictional owners or realizations."""
    return strip_ag_implementation(emit_ag_package(spec), spec)


def strip_ag_local_event_definitions(
    package_text: str,
    spec: AGChainSpec,
) -> str:
    """Remove standalone event types before terminal canonical imports.

    A terminal A/G package imports the exact system event classifiers and must
    not retain package-local classifiers with merely equal simple names.
    """
    text = str(package_text)
    for event_name in ag_event_signals(spec):
        text = re.sub(
            rf"(?m)^\s*item\s+def\s+"
            rf"{re.escape(event_name)}\s*;\s*\n?",
            "",
            text,
        )
    return text


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

    This deliberately excludes ownership, behavior realization, and executable
    pattern topology.  Those are terminal obligations and treating them as
    pre-generation facts is the root cause of the former shadow model.
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
