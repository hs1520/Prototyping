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
from typing import Dict, List, Optional

from .extractor import extract_behavioral_graph, BehavioralGraph, PartNode
from .exec_graph import build_exec_graph
from .scenarios import Scenario, DRONE_SCENARIOS, select_scenarios
from .simulator import ScenarioSimulator, ScenarioResult
from .behavioral_sim import run_behavioral_simulation, BehavioralSimResult

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    model_name: str
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    reachability_score: float = 0.0   # 0–1, structural connectivity score
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

    def passed_scenarios(self) -> List[ScenarioResult]:
        return [r for r in self.scenario_results if r.passed]

    def failed_scenarios(self) -> List[ScenarioResult]:
        return [r for r in self.scenario_results if not r.passed]

    @property
    def behavioral_score(self) -> float:
        if self.behavioral_result is None:
            return 1.0
        return self.behavioral_result.sim_score

    @property
    def combined_score(self) -> float:
        """60% structural connectivity + 40% behavioral simulation."""
        if self.behavioral_result is None or self.behavioral_result.extracted_sm_count == 0:
            return self.reachability_score
        return 0.6 * self.reachability_score + 0.4 * self.behavioral_score

    def summary(self) -> str:
        total = len(self.scenario_results)
        passed = len(self.passed_scenarios())
        lines = [
            f"=== Simulation Result: {self.model_name} ===",
            f"  Structural Score:  {self.reachability_score:.3f}",
        ]
        if self.isolated_parts:
            lines.append(f"  Isolated parts:  {', '.join(self.isolated_parts)}"
                         f"  ({len(self.isolated_parts)} part(s) with no connections)")
        if self.behavioral_result and self.behavioral_result.extracted_sm_count > 0:
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
        if self.behavioral_result and self.behavioral_result.extracted_sm_count > 0:
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
        result.isolated_parts = [
            p for p in bg.parts if p not in connected_in_graph
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
        if extra_scenarios:
            scenarios.extend(extra_scenarios)

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
    connected_src = {c.source.split(".")[0] for c in bg.connections}
    connected_tgt = {c.target.split(".")[0] for c in bg.connections}
    isolated = [p for p in bg.parts if p not in connected_src and p not in connected_tgt]

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
