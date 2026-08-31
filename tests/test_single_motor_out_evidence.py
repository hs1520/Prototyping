from src.prototyping.verification_obligations import (
    CriterionSource,
    ObligationKind,
    compile_verification_obligations,
    evaluate_evidence,
)
from gazebo_poc.single_motor_out_evidence import evaluate_single_motor_out
import pytest


def test_motor_out_evidence_preserves_criterion_source_and_sensitivity():
    runs = [
        {"return_code": 0, "attitude_rms_deg": 1.59},
        {"return_code": 0, "attitude_rms_deg": 13.46},
        {"return_code": 0, "attitude_rms_deg": 19.56},
        {"return_code": 0, "attitude_rms_deg": 2.10},
        {"return_code": 0, "attitude_rms_deg": 15.00},
    ]

    claim = evaluate_single_motor_out(runs)
    obligations = compile_verification_obligations(
        "REQ_SAFE_007",
        "The system shall maintain controlled flight following the failure of "
        "a single propulsion unit.",
    )
    result = evaluate_evidence(obligations, [claim])[0]

    assert result.kind is ObligationKind.CONTROLLED_FLIGHT
    assert result.status == "partial"
    assert claim.status == "failed"
    assert claim.criterion.source is CriterionSource.ENGINEERING_JUDGEMENT
    assert claim.criterion.threshold == 5.0
    assert claim.criterion.accepted_for_requirement is False
    assert [(item.passed_runs, item.total_runs) for item in claim.sensitivity] == [
        (2, 5),
        (5, 5),
    ]


def test_motor_out_evidence_rejects_an_empty_campaign():
    with pytest.raises(ValueError, match="at least one flight run"):
        evaluate_single_motor_out([])
