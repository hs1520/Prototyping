"""
validator.py

SimulationValidator: top-level orchestrator that:
  1. Parses SysML v2 text via syside native API (extract_behavioral_graph)
  2. Extracts BehavioralGraph (extractor.py)
  3. Builds networkx ExecutionGraph (exec_graph.py)
  4. Selects / auto-detects scenarios (scenarios.py)
  5. Runs ScenarioSimulator (simulator.py)
  6. Returns SimulationResult

SimulationResult is designed to integrate cleanly alongside the existing
EvaluationResult from src/dse/evaluator.py — it produces:
  - reachability_score: float 0–1 suitable as a new evaluation dimension
  - per-scenario pass/fail
  - aggregated issues and recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .extractor import extract_behavioral_graph, BehavioralGraph
from .exec_graph import build_exec_graph
from .scenarios import (
    Scenario,
    classify_parts_by_role,
    select_scenarios,
)
from .simulator import ScenarioSimulator, ScenarioResult
from .behavioral_sim import run_behavioral_simulation, BehavioralSimResult

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    model_name: str
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    reachability_score: float = 0.0   # 0–1, structural connectivity score
    # Primary structural evidence when a typed generation plan is active.
    # The role-derived scenario score above remains an advisory diagnostic.
    requirement_reachability_score: Optional[float] = None
    requirement_scenarios_passed: int = 0
    requirement_scenarios_total: int = 0
    structural_obligation_report: Optional[dict] = None
    behavioral_result: Optional["BehavioralSimResult"] = None  # state machine execution
    parse_errors: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Graph-level statistics
    num_parts: int = 0
    num_ports: int = 0
    num_connections: int = 0
    num_actions: int = 0
    isolated_parts: List[str] = field(default_factory=list)  # parts with no connect statements
    #: Unconnected parts whose part def the model itself declares passive
    #: (`// PLAN-PASSIVE` marker). Not defects: a structural body exchanges
    #: nothing by plan, so these are excluded from the isolation penalty and
    #: from refinement's wire-it-in feedback, and listed here for the audit.
    passive_unconnected_parts: List[str] = field(default_factory=list)
    role_assignments: Dict[str, List[str]] = field(default_factory=dict)
    weakly_connected_components: List[List[str]] = field(default_factory=list)
    role_scenario_definitions: List[Dict[str, Any]] = field(
        default_factory=list
    )

    def passed_scenarios(self) -> List[ScenarioResult]:
        return [r for r in self.scenario_results if r.passed]

    def failed_scenarios(self) -> List[ScenarioResult]:
        return [r for r in self.scenario_results if not r.passed]

    def advisory_structural_evidence(self) -> Dict[str, Any]:
        """Return auditable evidence without promoting heuristics to requirements."""
        results = {
            item.scenario_name: item for item in self.scenario_results
        }
        scenarios: List[Dict[str, Any]] = []
        definitions = self.role_scenario_definitions or [
            {
                "name": item.scenario_name,
                "description": item.description,
                "tags": list(item.tags),
                "provenance": "LEGACY_ADVISORY",
            }
            for item in self.scenario_results
        ]
        for definition in definitions:
            outcome = results.get(str(definition.get("name") or ""))
            scenarios.append({
                **definition,
                "status": (
                    "PASS"
                    if outcome is not None and outcome.passed
                    else "FAIL"
                ),
                "path": list(outcome.path) if outcome is not None else [],
                "issues": (
                    list(outcome.issues) if outcome is not None else [
                        "scenario was selected but no outcome was recorded"
                    ]
                ),
                "warnings": (
                    list(outcome.warnings) if outcome is not None else []
                ),
            })
        return {
            "evidence_kind": "UNTRACED_ADVISORY",
            "qualification_effect": "NONE",
            "role_assignments": {
                role: list(parts)
                for role, parts in sorted(self.role_assignments.items())
            },
            "weakly_connected_components": [
                list(component)
                for component in self.weakly_connected_components
            ],
            "scenarios": scenarios,
        }

    @property
    def behavioral_score(self) -> float:
        if self.behavioral_result is None:
            return 1.0
        return self.behavioral_result.sim_score

    @property
    def combined_score(self) -> float:
        """60% structural connectivity + 40% behavioral simulation."""
        structural = (
            self.requirement_reachability_score
            if self.requirement_reachability_score is not None
            else self.reachability_score
        )
        # Gate on "any behavioral scenario ran", not on the state-machine count:
        # constraint-sweep scenarios are collected even for models with zero
        # state defs, and their failures must not be discarded from the score.
        if self.behavioral_result is None or not self.behavioral_result.scenario_results:
            return structural
        return 0.6 * structural + 0.4 * self.behavioral_score

    def summary(self) -> str:
        total = len(self.scenario_results)
        passed = len(self.passed_scenarios())
        lines = [
            f"=== Simulation Result: {self.model_name} ===",
            (
                f"  Requirement Structural Score:  "
                f"{self.requirement_reachability_score:.3f}"
                f"  ({self.requirement_scenarios_passed}/"
                f"{self.requirement_scenarios_total})"
                if self.requirement_reachability_score is not None
                else f"  Structural Score:  {self.reachability_score:.3f}"
            ),
            f"  Advisory Role-Scenario Score:  {self.reachability_score:.3f}",
        ]
        if self.isolated_parts:
            lines.append(f"  Isolated parts:  {', '.join(self.isolated_parts)}"
                         f"  ({len(self.isolated_parts)} part(s) with no connections)")
        if self.passive_unconnected_parts:
            lines.append(
                "  Declared passive (unconnected by plan):  "
                f"{', '.join(self.passive_unconnected_parts)}"
            )
        if self.behavioral_result and self.behavioral_result.scenario_results:
            lines.append(f"  Behavioral Score:  {self.behavioral_score:.3f}"
                         f"  ({self.behavioral_result.passed_count()}/"
                         f"{len(self.behavioral_result.scenario_results)} SM scenarios)")
            lines.append(f"  Combined Score:    {self.combined_score:.3f}")
        lines += [
            f"  Scenarios: {passed}/{total} passed",
            f"  Graph:     {self.num_parts} parts, {self.num_ports} ports, "
            f"{self.num_connections} connections, {self.num_actions} actions",
        ]
        if self.parse_errors:
            lines.append(f"  Parse errors: {len(self.parse_errors)}")
        if self.failed_scenarios():
            lines.append("  Failed scenarios:")
            for r in self.failed_scenarios():
                lines.append(f"    ✗ {r.scenario_name}: {'; '.join(r.issues[:2])}")
        if self.passed_scenarios():
            lines.append("  Passed scenarios:")
            for r in self.passed_scenarios():
                path_str = " → ".join(r.path) if r.path else "(direct)"
                lines.append(f"    ✓ {r.scenario_name}: {path_str[:80]}")
        if self.behavioral_result and self.behavioral_result.scenario_results:
            lines.append("")
            lines += self.behavioral_result.summary_lines()
        if self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    • {rec}")
        return "\n".join(lines)


class SimulationValidator:
    """
    Validates a SysML v2 model text by structural reachability simulation.

    Usage:
        validator = SimulationValidator()
        result = validator.validate(sysml_text, model_name="DroneSystem")
        print(result.summary())
        print(f"Score: {result.reachability_score:.2f}")
    """

    def __init__(
        self,
        predefined_scenarios: Optional[List[Scenario]] = None,
        use_drone_scenarios: bool = True,
    ) -> None:
        self._predefined = predefined_scenarios
        self._use_drone = use_drone_scenarios

    def validate(
        self,
        sysml_text: str,
        model_name: str = "GeneratedModel",
        extra_scenarios: Optional[List[Scenario]] = None,
    ) -> SimulationResult:
        result = SimulationResult(model_name=model_name)

        # ── Step 1: Extract behavioral graph (syside native) ───────────────
        # extract_behavioral_graph parses the text internally via syside;
        # syntax errors are handled upstream by the orchestrator's syntax gate.
        try:
            bg = extract_behavioral_graph(
                sysml_text,
                root_package=model_name,
            )
        except Exception as e:
            result.issues.append(f"Behavioral extraction failed: {e}")
            log.error("Extraction error: %s", e)
            return result

        result.num_parts      = len(bg.parts)
        result.num_ports      = len(bg.ports)
        result.num_connections = len(bg.connections)
        result.num_actions    = len(bg.actions)
        result.role_assignments = classify_parts_by_role(bg)
        result.weakly_connected_components = _part_components(bg)

        if result.num_parts == 0:
            result.issues.append("No part definitions found — cannot simulate")
            result.reachability_score = 0.0
            return result

        # ── Step 3: Build execution graph ───────────────────────────────────
        try:
            G = build_exec_graph(bg)
        except Exception as e:
            result.issues.append(f"Graph build failed: {e}")
            log.error("Graph build error: %s", e)
            return result

        # ── Step 3b: Detect isolated parts ──────────────────────────────────
        connected_in_graph: set = (
            {c.source.split(".")[0] for c in bg.connections}
            | {c.target.split(".")[0] for c in bg.connections}
        )
        unconnected = [p for p in bg.parts if p not in connected_in_graph]
        # A declared-passive structural body exchanges nothing BY PLAN
        # (// PLAN-PASSIVE marker; scenario selection already exempts it in
        # scenarios.py). Charging it the isolation penalty docked a perfect
        # run 10% per passive body, and the isolated-parts feedback told
        # refinement to wire it in — pushing the LLM against the plan's own
        # passivity discipline.
        result.passive_unconnected_parts = [
            p for p in unconnected
            if bg.parts[p].def_name in bg.passive_defs
        ]
        result.isolated_parts = [
            p for p in unconnected
            if bg.parts[p].def_name not in bg.passive_defs
        ]
        if result.isolated_parts:
            result.issues.append(
                f"ISOLATED PARTS — no connect statements found for: "
                f"{', '.join(result.isolated_parts)}. "
                f"These parts are architecturally dead: they cannot receive or "
                f"propagate signals. Add connect statements to wire them in."
            )

        # ── Step 4: Select scenarios ─────────────────────────────────────────
        # Use model-driven auto-detection (classifies by usage name + def name).
        # Predefined scenarios are still accepted when callers pass them in.
        candidates: Optional[List[Scenario]] = self._predefined
        scenarios = select_scenarios(bg, predefined=candidates)
        scenario_provenance = (
            "PREDEFINED_ADVISORY"
            if candidates is not None else "AUTO_ROLE_HEURISTIC"
        )
        result.role_scenario_definitions = [
            {
                "name": item.name,
                "description": item.description,
                "entry_nodes": list(item.entry_nodes),
                "target_nodes": list(item.target_nodes),
                "required_nodes": list(item.required_nodes),
                "tags": list(item.tags),
                "provenance": scenario_provenance,
            }
            for item in scenarios
        ]
        if extra_scenarios:
            scenarios.extend(extra_scenarios)
            result.role_scenario_definitions.extend(
                {
                    "name": item.name,
                    "description": item.description,
                    "entry_nodes": list(item.entry_nodes),
                    "target_nodes": list(item.target_nodes),
                    "required_nodes": list(item.required_nodes),
                    "tags": list(item.tags),
                    "provenance": "CALLER_ADVISORY",
                }
                for item in extra_scenarios
            )

        if not scenarios:
            result.issues.append("No applicable scenarios found — check part names")
            result.reachability_score = 0.5  # partial credit: model parsed OK
            return result

        # ── Step 5: Run simulation ──────────────────────────────────────────
        sim = ScenarioSimulator(G)
        scenario_results = sim.run_all(scenarios)
        result.scenario_results = scenario_results

        # ── Step 6: Compute score ───────────────────────────────────────────
        result.reachability_score = _compute_score(
            scenario_results, result.isolated_parts, len(bg.parts)
        )

        # ── Step 7: Generate recommendations ───────────────────────────────
        result.issues = _collect_issues(scenario_results)
        result.recommendations = _generate_recommendations(bg, scenario_results)

        # ── Step 8: Behavioral simulation (state machine execution) ──────────
        try:
            result.behavioral_result = run_behavioral_simulation(
                sysml_text, model_name=model_name
            )
            for v in result.behavioral_result.all_violations():
                result.issues.append(f"[BEHAVIORAL] {v}")
        except Exception as e:
            log.warning("Behavioral simulation failed (non-fatal): %s", e)

        return result


def _part_components(bg: BehavioralGraph) -> List[List[str]]:
    """Compute deterministic part-only weak components for audit evidence."""
    adjacency: Dict[str, set[str]] = {
        part_name: set() for part_name in bg.parts
    }
    for connection in bg.connections:
        source = connection.source.split(".", 1)[0]
        target = connection.target.split(".", 1)[0]
        if source not in adjacency or target not in adjacency:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    remaining = set(adjacency)
    components: List[List[str]] = []
    while remaining:
        seed = min(remaining)
        frontier = [seed]
        component: set[str] = set()
        while frontier:
            node = frontier.pop()
            if node in component:
                continue
            component.add(node)
            frontier.extend(sorted(adjacency[node] - component))
        remaining -= component
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


# ---------------------------------------------------------------------------
# Scoring and recommendation helpers
# ---------------------------------------------------------------------------

def _compute_score(
    results: List[ScenarioResult],
    isolated_parts: List[str],
    total_parts: int,
) -> float:
    """
    Compute a 0–1 reachability score from scenario results.

    Weights:
      - safety-tagged scenarios: weight 2.0
      - emergency-tagged: weight 2.0
      - nominal: weight 1.0

    Isolation penalty: each isolated part reduces the base score
    proportionally (up to -50%).  Isolated parts are genuine design
    defects — a part with no connections contributes nothing to the
    system and cannot satisfy any operational scenario.
    """
    if not results:
        return 0.0

    total_weight = 0.0
    weighted_pass = 0.0

    for r in results:
        w = 2.0 if ("safety" in r.tags or "emergency" in r.tags) else 1.0
        total_weight += w
        if r.passed:
            weighted_pass += w
        elif r.reachable and not r.missing_nodes:
            # reached target but has warnings — partial credit
            weighted_pass += w * 0.5

    base_score = weighted_pass / total_weight if total_weight > 0 else 0.0

    if isolated_parts and total_parts > 0:
        isolation_ratio = len(isolated_parts) / total_parts
        # Each isolated part penalises up to 50% of the base score
        penalty = min(0.5, isolation_ratio)
        base_score = base_score * (1.0 - penalty)

    return base_score


def _collect_issues(results: List[ScenarioResult]) -> List[str]:
    issues = []
    for r in results:
        for issue in r.issues:
            issues.append(f"[{r.scenario_name}] {issue}")
    return issues


def _generate_recommendations(
    bg: BehavioralGraph, results: List[ScenarioResult]
) -> List[str]:
    recs = []

    failed = [r for r in results if not r.passed]
    if not failed:
        return ["All scenarios passed — model connectivity is structurally sound"]

    # Identify parts with no connections (reported as issues upstream; skip duplicate)
    # Parts with only in-ports (potential dead ends)
    for pname, pnode in bg.parts.items():
        dirs = [bg.ports[pid].direction for pid in pnode.port_ids if pid in bg.ports]
        if dirs and all(d == "in" for d in dirs):
            recs.append(
                f"'{pname}' has only input ports — ensure it has at least one output port "
                "or the model cannot propagate signals through it"
            )

    # Specific failed scenario recommendations
    for r in failed:
        if "emergency" in r.tags or "safety" in r.tags:
            recs.append(
                f"Safety scenario '{r.scenario_name}' failed — "
                "missing safety signal path could be a critical design flaw"
            )
        elif "power" in r.tags:
            recs.append(
                f"Power scenario '{r.scenario_name}' failed — "
                "verify that power ports are connected and directions are correct"
            )

    return recs
