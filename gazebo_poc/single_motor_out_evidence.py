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


# REQ-SAFE-007 says "maintain controlled flight" and names no measurement method,
# so it is judged on its own observable: the commanded hover held to scenario end
# in every run, worst case rather than average. Where the requirement states no
# bound, meeting its stated observable is the pass; a tighter bound of ours is
# reported, not gated on.
CONTROLLED_FLIGHT_SURFACE_CRITERION = VerificationCriterion(
    metric="hover_stable",
    operator="==",
    threshold=1.0,
    unit="bool",
    source=CriterionSource.REQUIREMENT,
    basis=(
        "'maintain controlled flight' with no stated measurement method; the "
        "observable surface is that the commanded hover was held to scenario "
        "end in every run"
    ),
    accepted_for_requirement=True,
)

# An attitude bound the requirement does not state. It separates tracking with
# reduced margin from the recorded large-amplitude wobble and is reported, but
# does not gate the verdict.
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
    stable_passes = sum(bool(run.get("stable")) for run in runs)
    attitude_passes = sum(
        run.get("attitude_rms_deg") is not None
        and float(run["attitude_rms_deg"])
        <= CONTROLLED_FLIGHT_ATTITUDE_CRITERION.threshold
        for run in runs
    )
    completed = sum(int(run["return_code"]) == 0 for run in runs)
    raw_status = "verified" if stable_passes == total else "failed"
    return EvidenceClaim(
        description=(
            f"{stable_passes} of {total} one-motor-out flights maintained "
            "controlled flight (commanded hover held to scenario end); "
            f"{attitude_passes} of {total} additionally met the "
            f"{CONTROLLED_FLIGHT_ATTITUDE_CRITERION.threshold:g} deg RMS "
            f"engineering interpretation; {completed} of {total} completed "
            "without scenario termination"
        ),
        status=raw_status,
        capabilities=frozenset({
            EvidenceCapability.CONTROLLED_FLIGHT_OBSERVED,
        }),
        criterion=CONTROLLED_FLIGHT_SURFACE_CRITERION,
        sensitivity=(
            CriterionEvaluation(
                "controlled flight maintained (stable hover held)",
                stable_passes,
                total,
            ),
            CriterionEvaluation(
                "attitude RMS <= 5 deg (informational, not in REQ-SAFE-007)",
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
