"""Requirement Evidence for the single-motor-out flight scenario."""
from __future__ import annotations

from typing import Mapping, Sequence

from src.prototyping.verification_obligations import (
    CriterionEvaluation,
    CriterionSource,
    EvidenceCapability,
    EvidenceClaim,
    VerificationCriterion,
)


CONTROLLED_FLIGHT_ATTITUDE_CRITERION = VerificationCriterion(
    metric="attitude_rms_deg",
    operator="<=",
    threshold=5.0,
    unit="deg",
    source=CriterionSource.ENGINEERING_JUDGEMENT,
    basis=(
        "separates attitude tracking with reduced margin from the recorded "
        "large-amplitude wobble"
    ),
    accepted_for_requirement=False,
)


def evaluate_single_motor_out(
    runs: Sequence[Mapping[str, object]],
) -> EvidenceClaim:
    """Interpret repeated raw flights without hiding the chosen definition."""
    if not runs:
        raise ValueError("single-motor-out evidence requires at least one flight run")
    total = len(runs)
    attitude_passes = sum(
        float(run["attitude_rms_deg"])
        <= CONTROLLED_FLIGHT_ATTITUDE_CRITERION.threshold
        for run in runs
    )
    completed = sum(int(run["return_code"]) == 0 for run in runs)
    raw_status = "verified" if attitude_passes == total else "failed"
    return EvidenceClaim(
        description=(
            f"{attitude_passes} of {total} one-motor-out flights met the "
            f"{CONTROLLED_FLIGHT_ATTITUDE_CRITERION.threshold:g} deg RMS "
            f"engineering interpretation; {completed} of {total} completed "
            "without scenario termination"
        ),
        status=raw_status,
        capabilities=frozenset({
            EvidenceCapability.CONTROLLED_FLIGHT_OBSERVED,
        }),
        criterion=CONTROLLED_FLIGHT_ATTITUDE_CRITERION,
        sensitivity=(
            CriterionEvaluation(
                "attitude RMS <= 5 deg",
                attitude_passes,
                total,
            ),
            CriterionEvaluation(
                "flight completed without scenario termination",
                completed,
                total,
            ),
        ),
    )
