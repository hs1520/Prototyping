"""Non-degradation evidence for terminal A/G binding.

The A/G layer may add assurance relationships; it does not change the executable
architecture or turn a passing frozen causal obligation into a failure.
Role-derived scenarios stay diagnostic, and the comparison is set-based so an
aggregate score cannot hide a regression.
"""
from __future__ import annotations

from typing import Any


def _structural_passes(result: Any) -> set[str]:
    return {
        str(item.scenario_name)
        for item in (getattr(result, "scenario_results", ()) or ())
        if bool(getattr(item, "passed", False))
    }


def _behavioral_passes(result: Any) -> set[str]:
    behavioral = getattr(result, "behavioral_result", None)
    return {
        str(item.name)
        for item in (getattr(behavioral, "scenario_results", ()) or ())
        if bool(getattr(item, "passed", False))
    }


def _requirement_structural_passes(result: Any) -> tuple[set[str], bool]:
    report = getattr(result, "structural_obligation_report", None)
    if not isinstance(report, dict) or not report.get("total"):
        return set(), False
    return {
        str(item.get("obligation_id"))
        for item in (report.get("results") or ())
        if item.get("status") == "PASS"
    }, True


def build_ag_non_degradation_report(before: Any, after: Any) -> dict[str, Any]:
    """Compare the pre-binding and post-binding simulation evidence."""
    before_structural = _structural_passes(before)
    after_structural = _structural_passes(after)
    before_requirement, before_fixed = _requirement_structural_passes(before)
    after_requirement, after_fixed = _requirement_structural_passes(after)
    before_behavioral = _behavioral_passes(before)
    after_behavioral = _behavioral_passes(after)
    lost_structural = sorted(before_structural - after_structural)
    fixed_structural_evidence = before_fixed and after_fixed
    lost_requirement = sorted(before_requirement - after_requirement)
    lost_behavioral = sorted(before_behavioral - after_behavioral)

    topology_fields = (
        "num_parts",
        "num_ports",
        "num_connections",
        "num_actions",
    )
    topology_before = {
        name: int(getattr(before, name, 0) or 0) for name in topology_fields
    }
    topology_after = {
        name: int(getattr(after, name, 0) or 0) for name in topology_fields
    }
    before_behavior = getattr(before, "behavioral_result", None)
    after_behavior = getattr(after, "behavioral_result", None)
    state_machines_before = int(
        getattr(before_behavior, "extracted_sm_count", 0) or 0
    )
    state_machines_after = int(
        getattr(after_behavior, "extracted_sm_count", 0) or 0
    )
    topology_unchanged = topology_before == topology_after
    state_machine_count_unchanged = (
        state_machines_before == state_machines_after
    )
    passed = (
        not (
            lost_requirement
            if fixed_structural_evidence
            else lost_structural
        )
        and not lost_behavioral
        and topology_unchanged
        and state_machine_count_unchanged
    )
    return {
        "artifact_role": "A_G_NON_DEGRADATION",
        "status": "PASS" if passed else "FAIL",
        "structural_evidence_kind": (
            "FROZEN_REQUIREMENT_CAUSAL_PATHS"
            if fixed_structural_evidence
            else "ADAPTIVE_ROLE_SCENARIOS_LEGACY"
        ),
        "lost_requirement_structural_obligations": lost_requirement,
        "lost_structural_scenarios": lost_structural,
        "role_scenarios_advisory": fixed_structural_evidence,
        "lost_behavioral_scenarios": lost_behavioral,
        "structural_passed_before": len(before_structural),
        "structural_passed_after": len(after_structural),
        "behavioral_passed_before": len(before_behavioral),
        "behavioral_passed_after": len(after_behavioral),
        "topology_unchanged": topology_unchanged,
        "topology_before": topology_before,
        "topology_after": topology_after,
        "state_machine_count_unchanged": state_machine_count_unchanged,
        "state_machines_before": state_machines_before,
        "state_machines_after": state_machines_after,
    }
