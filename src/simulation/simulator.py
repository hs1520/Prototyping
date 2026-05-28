"""
simulator.py

ScenarioSimulator: runs reachability checks on an ExecutionGraph (nx.DiGraph)
for each Scenario, returning structured ScenarioResult objects.

Reachability strategy (three-tier):
  Tier 1 — Port-level path: find a directed path through the full graph
            (ports + parts + actions).  Most precise.
  Tier 2 — Part-level shortcut: check if a "via_*" edge exists between parts.
            Faster and tolerates unresolved port names.
  Tier 3 — Undirected fallback: check undirected connectivity for warning-level
            results (connected but direction may be wrong).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

try:
    import networkx as nx
    _HAS_NX = True
except ModuleNotFoundError:
    _HAS_NX = False

from .scenarios import Scenario
from .exec_graph import shortest_path, reachable_from, all_simple_paths


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    tags: List[str]
    reachable: bool                    # True = all target_nodes reachable
    path: List[str]                    # representative path (first target)
    missing_nodes: List[str]           # required_nodes NOT on any path
    unreachable_targets: List[str]     # target_nodes that have no path
    issues: List[str]
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.reachable and not self.missing_nodes


class ScenarioSimulator:
    """
    Run scenario reachability checks against a compiled networkx DiGraph.
    """

    def __init__(self, G: "nx.DiGraph") -> None:
        if not _HAS_NX:
            raise ImportError("networkx is required for simulation")
        self.G = G

    # ── Public interface ────────────────────────────────────────────────────

    def run(self, scenario: Scenario) -> ScenarioResult:
        issues: List[str] = []
        warnings: List[str] = []
        unreachable_targets: List[str] = []
        representative_path: List[str] = []
        missing_required: List[str] = []

        # Resolve entry nodes (fall back to first matching part if exact id missing)
        entry_nodes = self._resolve_nodes(scenario.entry_nodes, label="entry")
        if not entry_nodes:
            return ScenarioResult(
                scenario_name=scenario.name,
                description=scenario.description,
                tags=scenario.tags,
                reachable=False,
                path=[],
                missing_nodes=list(scenario.required_nodes),
                unreachable_targets=list(scenario.target_nodes),
                issues=[f"None of the entry nodes exist in graph: {scenario.entry_nodes}"],
            )

        target_nodes = self._resolve_nodes(scenario.target_nodes, label="target")
        if not target_nodes:
            return ScenarioResult(
                scenario_name=scenario.name,
                description=scenario.description,
                tags=scenario.tags,
                reachable=False,
                path=[],
                missing_nodes=[],
                unreachable_targets=list(scenario.target_nodes),
                issues=[f"None of the target nodes exist in graph: {scenario.target_nodes}"],
            )

        # ── Check each target is reachable from some entry ──────────────────
        for tgt in target_nodes:
            reached = False
            best_path: Optional[List[str]] = None
            for src in entry_nodes:
                p = shortest_path(self.G, src, tgt)
                if p:
                    reached = True
                    if best_path is None or len(p) < len(best_path):
                        best_path = p
            if reached and best_path:
                if not representative_path:
                    representative_path = best_path
            else:
                unreachable_targets.append(tgt)
                # Tier-3: undirected fallback
                if self._undirected_connected(entry_nodes, tgt):
                    warnings.append(
                        f"'{tgt}' is connected to entry nodes but signal direction may be wrong"
                    )
                else:
                    issues.append(f"No path to target '{tgt}' from {entry_nodes}")

        # ── Check required intermediate nodes ───────────────────────────────
        for req in scenario.required_nodes:
            req_resolved = self._resolve_node(req)
            if req_resolved is None:
                missing_required.append(req)
                issues.append(f"Required node '{req}' not found in graph")
                continue
            # req must be reachable from some entry AND must reach some target
            from_entry = any(
                nx.has_path(self.G, src, req_resolved)
                for src in entry_nodes
                if src in self.G
            )
            to_target = any(
                nx.has_path(self.G, req_resolved, tgt)
                for tgt in target_nodes
                if tgt in self.G
            )
            if not (from_entry and to_target):
                missing_required.append(req)
                issues.append(
                    f"Required node '{req}' is not on any entry→target path "
                    f"(reachable_from_entry={from_entry}, reaches_target={to_target})"
                )

        overall_reachable = len(unreachable_targets) == 0

        return ScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            tags=scenario.tags,
            reachable=overall_reachable,
            path=representative_path,
            missing_nodes=missing_required,
            unreachable_targets=unreachable_targets,
            issues=issues,
            warnings=warnings,
        )

    def run_all(self, scenarios: List[Scenario]) -> List[ScenarioResult]:
        return [self.run(s) for s in scenarios]

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_node(self, name: str) -> Optional[str]:
        """Return graph node id for *name*, or None if not found."""
        if name in self.G:
            return name
        lower = name.lower()
        for n in self.G.nodes:
            if n.lower() == lower:
                return n
        return None

    def _resolve_nodes(self, names: List[str], label: str = "") -> List[str]:
        resolved = []
        for n in names:
            r = self._resolve_node(n)
            if r:
                resolved.append(r)
        return resolved

    def _undirected_connected(self, sources: List[str], target: str) -> bool:
        if not _HAS_NX:
            return False
        UG = self.G.to_undirected()
        tgt = self._resolve_node(target)
        if tgt is None:
            return False
        for src in sources:
            s = self._resolve_node(src)
            if s and nx.has_path(UG, s, tgt):
                return True
        return False
