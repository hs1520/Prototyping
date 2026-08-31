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


def test_motor_out_surface_criterion_gates_and_attitude_stays_informational():
    """REQ-SAFE-007 names no measurement method, so its own observable surface
    — controlled flight maintained (stable hover held) in every run — is the
    verdict. The 5 deg RMS engineering interpretation stays on the record but
    cannot fail a requirement that never stated it."""
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
    # surface first, then the informational interpretation, then completion
    assert [(item.passed_runs, item.total_runs) for item in claim.sensitivity] == [
        (5, 5),
        (2, 5),
        (5, 5),
    ]
    assert "informational" in claim.sensitivity[1].interpretation


def test_motor_out_surface_criterion_still_fails_an_unstable_flight():
    """Worst case, not average: one departed flight fails the redundancy
    claim, and the accepted surface criterion makes that a real failure,
    not a demoted partial."""
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


def test_motor_out_runs_without_a_stability_observation_do_not_pass():
    """A run that never recorded the surface observable cannot meet it —
    absence of measurement is not evidence of controlled flight."""
    claim = evaluate_single_motor_out(
        [{"return_code": 0, "attitude_rms_deg": 1.59}]
    )
    assert claim.status == "failed"
    assert claim.sensitivity[0].passed_runs == 0


def test_motor_out_evidence_rejects_an_empty_campaign():
    with pytest.raises(ValueError, match="at least one flight run"):
        evaluate_single_motor_out([])
