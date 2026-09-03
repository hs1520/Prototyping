"""Derive one conservative research conclusion from same-run evidence.

Every claim carries an explicit scope and fidelity rather than rolling up green
labels: a Phase 8 datasheet closure supports only the bottom-up realization
claim, while SITL and Gazebo decide different claims.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

from ..realization.matcher import mapping_policy


SCHEMA_VERSION = 1
SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
INCOMPLETE = "INCOMPLETE"
PARTIAL = "PARTIAL"


def _claim(claim_id: str, status: str, statement: str, scope: str,
           evidence: Mapping[str, Any], limitation: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "status": status,
        "statement": statement,
        "scope": scope,
        "evidence": dict(evidence),
        "limitation": limitation,
    }


def _matrix_counts(matrix: Mapping[str, Any]) -> dict[str, int]:
    rows = list(matrix.get("rows") or [])
    counts = dict(sorted(Counter(str(r.get("status")) for r in rows).items()))
    declared = ((matrix.get("summary") or {}).get("by_status") or {})
    if declared != counts:
        raise ValueError(
            f"verification matrix summary is inconsistent with rows: {declared} != {counts}"
        )
    return counts


def _validate_phase8(run: Mapping[str, Any], realization: Mapping[str, Any]) -> tuple[str, list[dict]]:
    verdict = str(realization.get("verdict") or "UNKNOWN")
    closure_rows = [
        row for row in (realization.get("per_requirement") or [])
        if row.get("scope", "closure") == "closure"
    ]
    if verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}:
        chosen = realization.get("chosen") or {}
        recommended = run.get("recommended_design_inputs") or {}
        if not chosen:
            raise ValueError("Phase 8 CLOSED has no chosen physical realization")
        policy = realization.get("mapping_policy") or {}
        authoritative_policy = mapping_policy()
        if policy != authoritative_policy:
            raise ValueError("Phase 8 CLOSED design-mapping policy differs from the authoritative policy")
        drifts = {x.get("name"): x for x in (chosen.get("design_drift") or [])}
        required_drifts = {"rotor_radius_m", "battery_capacity_mah", "battery_cells"}
        if not required_drifts <= set(drifts):
            raise ValueError("Phase 8 CLOSED lacks complete cumulative design-drift evidence")
        for name in ("rotor_count", "battery_cells"):
            if chosen.get(name) != recommended.get(name):
                raise ValueError(f"Phase 8 CLOSED changes architecture invariant {name}")
        value_pairs = {
            "rotor_radius_m": (
                recommended.get("rotor_radius_m"), chosen.get("rotor_radius_m"),
                authoritative_policy["max_relative_drift"]["rotor_radius_m"],
            ),
            "battery_capacity_mah": (
                recommended.get("battery_capacity_mah"), chosen.get("battery_capacity_mah"),
                authoritative_policy["max_relative_drift"]["battery_capacity_mah"],
            ),
            "battery_cells": (
                recommended.get("battery_cells"), chosen.get("battery_cells"), 0.0,
            ),
        }
        for name, (expected, realized, limit) in value_pairs.items():
            if expected is None or realized is None:
                raise ValueError(f"Phase 8 CLOSED lacks mapping identity value {name}")
            expected_f, realized_f = float(expected), float(realized)
            relative = (realized_f - expected_f) / (abs(expected_f) if expected_f else 1.0)
            row = drifts[name]
            recorded = (row.get("expected"), row.get("realized"), row.get("relative_delta"), row.get("limit"))
            recomputed = (expected_f, realized_f, relative, float(limit))
            if any(a is None or not math.isclose(float(a), b, rel_tol=1e-12, abs_tol=1e-12)
                   for a, b in zip(recorded, recomputed)):
                raise ValueError(f"Phase 8 CLOSED has inconsistent cumulative drift data for {name}")
            within = abs(relative) <= float(limit) + 1e-12
            if row.get("within_limit") is not within:
                raise ValueError(f"Phase 8 CLOSED has inconsistent drift verdict for {name}")
            if not within:
                raise ValueError("Phase 8 CLOSED exceeds the design-mapping drift policy")
        if not closure_rows:
            raise ValueError("Phase 8 CLOSED has no closure-scope requirement evidence")
        if any(row.get("met") is not True for row in closure_rows):
            raise ValueError("Phase 8 CLOSED contradicts an unmet closure-scope requirement")
        return SUPPORTED, closure_rows
    if verdict in {"INFEASIBLE_REALIZATION", "NO_RECOMMENDABLE_DESIGN"}:
        return REFUTED, closure_rows
    return INCOMPLETE, closure_rows


def _validate_safety(sitl: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    l2 = list(sitl.get("safety_l2") or [])
    trace = list(sitl.get("traceability") or [])
    passed = sum(r.get("passed") is True for r in l2)
    blocked = sum(r.get("passed") is not True for r in trace)
    expected = (
        "PARTIAL" if blocked and l2 else
        "BLOCKED" if blocked else
        "NOT_RUN" if not l2 else
        "PASS" if passed == len(l2) else
        "FAIL"
    )
    actual = str((sitl.get("safety_verification") or {}).get("status") or "UNKNOWN")
    if actual != expected:
        raise ValueError(f"SITL safety status is inconsistent with executed evidence: {actual} != {expected}")
    claim_status = {
        "PASS": SUPPORTED,
        "FAIL": REFUTED,
        "PARTIAL": PARTIAL,
        "BLOCKED": INCOMPLETE,
        "NOT_RUN": INCOMPLETE,
    }[expected]
    return claim_status, {"l2_passed": passed, "l2_total": len(l2), "traceability_blocked": blocked}


def _validate_gazebo(gazebo: Mapping[str, Any]) -> str:
    overall = str(gazebo.get("status") or "UNKNOWN").upper()
    statuses = {str(x.get("status") or "").upper() for x in (gazebo.get("req_results") or [])}
    # ``overall`` is runner/model health; per-requirement statuses are
    # outcomes. They are orthogonal - an uncalibrated model can still produce an
    # exact actuator-observer FAIL - so that failure is kept even when the runner
    # health label is more specific than FAIL.
    if "FAIL" in statuses:
        return REFUTED
    if overall == "PASS" and (not statuses or statuses != {"PASS"}):
        raise ValueError("Gazebo PASS requires a non-empty all-PASS requirement result set")
    if overall in {"FAIL", "DYNAMICS_INFEASIBLE"}:
        return REFUTED
    if overall == "PASS":
        return SUPPORTED
    if overall == "PARTIAL":
        return PARTIAL
    return INCOMPLETE


def derive_research_conclusion(run: Mapping[str, Any], gazebo: Mapping[str, Any],
                               sitl: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sole machine-readable conclusion for an authoritative run."""
    realization = run.get("realization") or {}
    phase8_status, closure_rows = _validate_phase8(run, realization)
    flight_passed = (sitl.get("flight") or {}).get("passed")
    if flight_passed not in {True, False}:
        raise ValueError("SITL flight feasibility has no boolean result")
    flight_status = SUPPORTED if flight_passed else REFUTED
    safety_status, safety_counts = _validate_safety(sitl)
    gazebo_status = _validate_gazebo(gazebo)
    matrix_counts = _matrix_counts(matrix)
    total = len(matrix.get("rows") or [])
    if matrix_counts.get("failed", 0):
        matrix_status = REFUTED
    elif total and matrix_counts.get("verified", 0) == total:
        matrix_status = SUPPORTED
    else:
        matrix_status = INCOMPLETE

    claims = [
        _claim(
            "bottom_up_realization_closure", phase8_status,
            "The recommended architecture maps to catalog components and closes the Phase 8 endurance/mass requirements.",
            "Catalog compatibility plus manufacturer-datasheet endurance and mass analysis",
            {
                "verdict": realization.get("verdict"),
                "closure_requirement_count": len(closure_rows),
                "mapping_policy": realization.get("mapping_policy"),
                "design_drift": (realization.get("chosen") or {}).get("design_drift"),
            },
            "Does not establish flight endurance, safety behaviour, or all-requirement compliance.",
        ),
        _claim(
            "native_flight_feasibility", flight_status,
            "The parameterized design can arm, take off, and maintain the tested native-SITL flight condition.",
            "Native ArduPilot SITL arm/takeoff/hover scenario",
            {"flight_passed": flight_passed},
            "Native SITL is architecture-nondiscriminating for endurance and is not a physical flight test.",
        ),
        _claim(
            "executable_safety_subset", safety_status,
            "The executable and traceable L2 safety subset behaves as required in native SITL.",
            "Only generated L2 checks whose requirement-to-trigger traceability passed",
            safety_counts,
            "Unmapped or traceability-blocked safety requirements are not validated by passing executable cases.",
        ),
        _claim(
            "gazebo_dynamics_requirements", gazebo_status,
            "The implemented Gazebo dynamics checks satisfy their linked requirements.",
            "Same-run Gazebo high-fidelity dynamics and trajectory checks",
            {"gazebo_status": gazebo.get("status")},
            "Gazebo does not validate datasheet endurance; planned or suspended checks remain unevidenced.",
        ),
        _claim(
            "complete_requirement_verification", matrix_status,
            "Every extracted requirement is verified and met at an assigned evidence tier.",
            "Exact same-run requirement set and final verification matrix",
            {"total": total, "by_status": matrix_counts},
            "Any failed, partial, planned, blocked, unassigned, or out-of-scope row prevents this claim.",
        ),
    ]
    statuses = {c["status"] for c in claims}
    if phase8_status == REFUTED:
        overall = "NOT_SUPPORTED"
    elif REFUTED in statuses:
        overall = "MIXED"
    elif statuses == {SUPPORTED}:
        overall = "SUPPORTED"
    else:
        overall = "INCOMPLETE"

    supported = [c["claim_id"] for c in claims if c["status"] == SUPPORTED]
    refuted = [c["claim_id"] for c in claims if c["status"] == REFUTED]
    incomplete = [c["claim_id"] for c in claims if c["status"] in {INCOMPLETE, PARTIAL}]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": (run.get("artifact_provenance") or {}).get("run_id"),
        "overall": overall,
        "claims": claims,
        "supported_claim_ids": supported,
        "refuted_claim_ids": refuted,
        "incomplete_claim_ids": incomplete,
        "meeting_statement": (
            f"Overall evidence is {overall}. Supported claims: {', '.join(supported) or 'none'}; "
            f"refuted claims: {', '.join(refuted) or 'none'}; "
            f"incomplete/partial claims: {', '.join(incomplete) or 'none'}."
        ),
        "forbidden_overclaim": (
            "Do not state that the complete system or all requirements are validated unless "
            "overall=SUPPORTED and complete_requirement_verification=SUPPORTED."
        ),
    }


def to_markdown(conclusion: Mapping[str, Any]) -> str:
    lines = [
        "# Authoritative Research Conclusion", "",
        f"- Run ID: {conclusion.get('run_id')}",
        f"- Overall: **{conclusion.get('overall')}**", "",
        str(conclusion.get("meeting_statement") or ""), "",
        "| Claim | Status | Scope | Limitation |", "|---|---|---|---|",
    ]
    for claim in conclusion.get("claims") or []:
        lines.append(
            f"| {claim['claim_id']} | {claim['status']} | {claim['scope']} | {claim['limitation']} |"
        )
    lines.extend(["", f"> {conclusion.get('forbidden_overclaim')}"])
    return "\n".join(lines) + "\n"
