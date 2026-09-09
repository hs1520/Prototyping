from src.prototyping.verification_obligations import (
    CriterionSource,
    ObligationKind,
    compile_verification_obligations,
    evaluate_evidence,
)
from gazebo_poc.single_motor_out_evidence import evaluate_single_motor_out
import pytest


_REQ_TEXT = (
    "The system shall maintain controlled flight following the failure of "
    "a single propulsion unit."
)


def test_surface_criterion_gates():
    """REQ-SAFE-007 names no measurement method, so its observable surface - stable
    hover held in every run - is the verdict.

    The 5 deg RMS interpretation stays on the record but cannot fail a requirement
    that never stated it.
    """
    runs = [
        {"return_code": 0, "stable": True, "attitude_rms_deg": 1.59},
        {"return_code": 0, "stable": True, "attitude_rms_deg": 13.46},
        {"return_code": 0, "stable": True, "attitude_rms_deg": 19.56},
        {"return_code": 0, "stable": True, "attitude_rms_deg": 2.10},
        {"return_code": 0, "stable": True, "attitude_rms_deg": 15.00},
    ]

    claim = evaluate_single_motor_out(runs)
    obligations = compile_verification_obligations("REQ_SAFE_007", _REQ_TEXT)
    result = evaluate_evidence(obligations, [claim])[0]

    assert result.kind is ObligationKind.CONTROLLED_FLIGHT
    assert claim.status == "verified"
    assert result.status == "verified"
    assert claim.criterion.metric == "hover_stable"
    assert claim.criterion.source is CriterionSource.REQUIREMENT
    assert claim.criterion.accepted_for_requirement is True
    assert [(item.passed_runs, item.total_runs) for item in claim.sensitivity] == [
        (5, 5),
        (2, 5),
        (5, 5),
    ]
    assert "informational" in claim.sensitivity[1].interpretation


def test_unstable_flight_fails():
    """Worst case, not average: one departed flight fails the redundancy claim, and
    the surface criterion reports it as a failure rather than a partial.
    """
    runs = [
        {"return_code": 0, "stable": True, "attitude_rms_deg": 1.59},
        {"return_code": 1, "stable": False, "attitude_rms_deg": None},
    ]

    claim = evaluate_single_motor_out(runs)
    obligations = compile_verification_obligations("REQ_SAFE_007", _REQ_TEXT)
    result = evaluate_evidence(obligations, [claim])[0]

    assert claim.status == "failed"
    assert result.status == "failed"
    assert [(item.passed_runs, item.total_runs) for item in claim.sensitivity] == [
        (1, 2),
        (1, 2),
        (1, 2),
    ]


def test_no_stability_observation_fails():
    claim = evaluate_single_motor_out(
        [{"return_code": 0, "attitude_rms_deg": 1.59}]
    )
    assert claim.status == "failed"
    assert claim.sensitivity[0].passed_runs == 0


def test_empty_campaign_rejected():
    with pytest.raises(ValueError, match="at least one flight run"):
        evaluate_single_motor_out([])
