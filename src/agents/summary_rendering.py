"""Shared completion-summary lines for generation and exploration."""
from __future__ import annotations

from typing import Any


def simulation_summary_lines(simulation: Any) -> list[str]:
    if simulation.requirement_reachability_score is not None:
        return [
            "  Requirement reachability: "
            f"{simulation.requirement_reachability_score:.3f} "
            f"({simulation.requirement_scenarios_passed}/"
            f"{simulation.requirement_scenarios_total} frozen paths)",
            "  Advisory role scenarios: "
            f"{simulation.reachability_score:.3f} "
            f"({len(simulation.passed_scenarios())}/"
            f"{len(simulation.scenario_results)} scenarios)",
        ]
    return [
        "  Simulation reachability:  "
        f"{simulation.reachability_score:.3f} "
        f"({len(simulation.passed_scenarios())}/"
        f"{len(simulation.scenario_results)} scenarios)"
    ]


def runtime_footer_lines(model: Any, llm: Any, *, verbose: bool) -> list[str]:
    lines: list[str] = []
    warnings = (getattr(model, "metadata", None) or {}).get("sim_warnings", "")
    if warnings:
        lines.append("")
        lines.extend(f"  {line}" for line in warnings.splitlines())
    ledger = getattr(llm, "ledger", None)
    if ledger is not None and getattr(ledger, "calls", 0):
        lines.append(f"  LLM usage:        {ledger.summary()}")
    if verbose:
        from ..utils.suppressed import suppressed_summary

        summary = suppressed_summary()
        if summary:
            lines.append(f"  suppressed:       {summary}")
    return lines
