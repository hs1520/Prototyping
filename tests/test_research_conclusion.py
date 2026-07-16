from __future__ import annotations

import pytest

from src.prototyping.research_conclusion import derive_research_conclusion
from src.realization.matcher import mapping_policy


def _evidence():
    run = {
        "artifact_provenance": {"run_id": "r1"},
        "recommended_design_inputs": {
            "rotor_count": 4, "battery_cells": 6,
            "rotor_radius_m": 0.20, "battery_capacity_mah": 10000,
        },
        "realization": {
            "verdict": "CLOSED",
            "mapping_policy": mapping_policy(),
            "chosen": {
                "frame": "F", "rotor_count": 4, "battery_cells": 6,
                "rotor_radius_m": 0.20, "battery_capacity_mah": 10000,
                "design_drift": [
                    {"name": "rotor_radius_m", "expected": 0.20, "realized": 0.20,
                     "relative_delta": 0.0, "limit": 0.10, "within_limit": True},
                    {"name": "battery_capacity_mah", "expected": 10000, "realized": 10000,
                     "relative_delta": 0.0, "limit": 0.10, "within_limit": True},
                    {"name": "battery_cells", "expected": 6, "realized": 6,
                     "relative_delta": 0.0, "limit": 0.0, "within_limit": True},
                ],
            },
            "per_requirement": [
                {"req_id": "REQ-PERF-1", "scope": "closure", "met": True},
            ],
        },
    }
    gazebo = {
        "status": "PASS",
        "req_results": [{"req_id": "REQ-FUNC-1", "status": "PASS"}],
    }
    sitl = {
        "flight": {"passed": True},
        "traceability": [],
        "safety_l2": [{"req_id": "REQ-SAFE-1", "passed": True}],
        "safety_verification": {"status": "PASS"},
    }
    matrix = {
        "summary": {"by_status": {"verified": 1}},
        "rows": [{"req_id": "REQ-1", "status": "verified"}],
    }
    return run, gazebo, sitl, matrix


def _claims(result):
    return {claim["claim_id"]: claim for claim in result["claims"]}


def test_all_supported_requires_every_requirement_verified():
    result = derive_research_conclusion(*_evidence())

    assert result["overall"] == "SUPPORTED"
    assert all(c["status"] == "SUPPORTED" for c in result["claims"])


def test_local_passes_do_not_hide_a_gazebo_requirement_failure():
    run, gazebo, sitl, matrix = _evidence()
    gazebo["status"] = "FAIL"
    gazebo["req_results"][0]["status"] = "FAIL"
    matrix["summary"]["by_status"] = {"failed": 1}
    matrix["rows"][0]["status"] = "failed"

    result = derive_research_conclusion(run, gazebo, sitl, matrix)
    claims = _claims(result)

    assert result["overall"] == "MIXED"
    assert claims["bottom_up_realization_closure"]["status"] == "SUPPORTED"
    assert claims["native_flight_feasibility"]["status"] == "SUPPORTED"
    assert claims["gazebo_dynamics_requirements"]["status"] == "REFUTED"
    assert claims["complete_requirement_verification"]["status"] == "REFUTED"


def test_uncalibrated_gazebo_health_does_not_hide_an_exact_requirement_failure():
    run, gazebo, sitl, matrix = _evidence()
    gazebo["status"] = "GAZEBO_MODEL_UNCALIBRATED"
    gazebo["req_results"][0]["status"] = "FAIL"
    matrix["summary"]["by_status"] = {"failed": 1}
    matrix["rows"][0]["status"] = "failed"

    result = derive_research_conclusion(run, gazebo, sitl, matrix)

    assert result["overall"] == "MIXED"
    assert _claims(result)["gazebo_dynamics_requirements"]["status"] == "REFUTED"


def test_gaps_force_incomplete_even_when_executed_checks_pass():
    run, gazebo, sitl, matrix = _evidence()
    matrix["summary"]["by_status"] = {"unassigned": 1, "verified": 1}
    matrix["rows"].append({"req_id": "REQ-2", "status": "unassigned"})

    result = derive_research_conclusion(run, gazebo, sitl, matrix)

    assert result["overall"] == "INCOMPLETE"
    assert _claims(result)["complete_requirement_verification"]["status"] == "INCOMPLETE"


def test_closed_cannot_contradict_its_own_closure_rows():
    run, gazebo, sitl, matrix = _evidence()
    run["realization"]["per_requirement"][0]["met"] = False

    with pytest.raises(ValueError, match="CLOSED contradicts"):
        derive_research_conclusion(run, gazebo, sitl, matrix)


def test_closed_cannot_exceed_mapping_drift_policy():
    run, gazebo, sitl, matrix = _evidence()
    run["realization"]["chosen"]["rotor_radius_m"] = 0.24
    row = run["realization"]["chosen"]["design_drift"][0]
    row.update(realized=0.24, relative_delta=0.20, within_limit=False)

    with pytest.raises(ValueError, match="exceeds the design-mapping drift policy"):
        derive_research_conclusion(run, gazebo, sitl, matrix)


def test_closed_cannot_forge_a_within_limit_flag():
    run, gazebo, sitl, matrix = _evidence()
    run["realization"]["chosen"]["design_drift"][0]["relative_delta"] = 0.5

    with pytest.raises(ValueError, match="inconsistent cumulative drift data"):
        derive_research_conclusion(run, gazebo, sitl, matrix)


def test_closed_cannot_change_architecture_identity():
    run, gazebo, sitl, matrix = _evidence()
    run["realization"]["chosen"]["battery_cells"] = 4

    with pytest.raises(ValueError, match="architecture invariant battery_cells"):
        derive_research_conclusion(run, gazebo, sitl, matrix)


def test_declared_matrix_summary_must_equal_rows():
    run, gazebo, sitl, matrix = _evidence()
    matrix["summary"]["by_status"] = {"verified": 99}

    with pytest.raises(ValueError, match="summary is inconsistent"):
        derive_research_conclusion(run, gazebo, sitl, matrix)


def test_sitl_pass_cannot_hide_failed_executable_case():
    run, gazebo, sitl, matrix = _evidence()
    sitl["safety_l2"][0]["passed"] = False

    with pytest.raises(ValueError, match="SITL safety status is inconsistent"):
        derive_research_conclusion(run, gazebo, sitl, matrix)
