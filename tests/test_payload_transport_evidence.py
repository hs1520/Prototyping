"""Transport evidence must come from windows observed to be carrying.

REQ-FUNC-003's attitude bound applies while *transporting*. Attachment used to
be inferred from the run configuration — a flag set before takeoff — even though
the same flight releases the payload part-way through, so a cruise window
measured after separation would have counted as transport evidence.
"""
from __future__ import annotations

from gazebo_poc.payload_transport_evidence import (
    ATTACHED_MAX_DISTANCE_M,
    PayloadAttachment,
    TransportWindow,
    evaluate_payload_transport,
    observe_attachment,
)
from src.prototyping.verification_obligations import EvidenceCapability


def _window(label, distance, rms, observed=True):
    return TransportWindow(
        label=label,
        attachment=PayloadAttachment(observed=observed, distance_m=distance),
        attitude_rms_deg=rms,
    )


def test_attachment_is_a_distance_between_two_ground_truth_poses():
    attached = observe_attachment((10.0, 5.0, 10.0), (10.0, 5.0, 9.85))
    assert attached.observed and attached.attached
    assert attached.distance_m < ATTACHED_MAX_DISTANCE_M

    released = observe_attachment((10.0, 5.0, 10.0), (10.2, 5.1, 0.06))
    assert released.observed and not released.attached

    # a pose that could not be read is not an attached payload
    missing = observe_attachment((10.0, 5.0, 10.0), None)
    assert not missing.observed and not missing.attached


def test_windows_measured_after_release_are_excluded_not_counted():
    claim = evaluate_payload_transport(
        [
            _window("hover", 0.15, 0.008),
            _window("cruise@rc1220", 0.15, 0.026),
            # flown after separation — good attitude, but not transport evidence
            _window("cruise@rc1100", 9.4, 0.004),
        ],
        attitude_limit_deg=1.0,
    )

    assert claim.status == "verified"
    # the verdict rests on the loaded windows' worst, not the unloaded window's best
    assert "worst attitude RMS 0.026 deg at cruise@rc1220" in claim.description
    assert "1 window(s) excluded as observed unloaded" in claim.description
    assert "cruise@rc1100" in claim.description
    assert claim.sensitivity[0].passed_runs == 2
    assert claim.sensitivity[0].total_runs == 3


def test_a_run_that_never_observed_attachment_cannot_close_the_payload_clause():
    """Absence of an observation is not evidence of attachment."""
    claim = evaluate_payload_transport(
        [_window("cruise@rc1220", None, 0.026, observed=False)],
        attitude_limit_deg=1.0,
    )

    assert claim.status == "planned"
    assert "attachment was never observed" in claim.description
    assert EvidenceCapability.PAYLOAD_STATE_OBSERVED not in claim.capabilities
    # it still measured an attitude, and says so
    assert EvidenceCapability.ATTITUDE_RMS_MEASURED in claim.capabilities


def test_all_windows_unloaded_is_reported_as_no_transport_evidence():
    claim = evaluate_payload_transport(
        [_window("cruise@rc1220", 8.0, 0.02), _window("cruise@rc1100", 12.0, 0.03)],
        attitude_limit_deg=1.0,
    )

    assert claim.status == "planned"
    assert "every one of 2 measured windows was observed UNLOADED" in claim.description
    # attachment WAS observed here, so the payload state is known
    assert EvidenceCapability.PAYLOAD_STATE_OBSERVED in claim.capabilities


def test_a_loaded_window_over_the_bound_is_a_failure():
    claim = evaluate_payload_transport(
        [_window("hover", 0.15, 0.008), _window("cruise@rc1100", 0.15, 1.4)],
        attitude_limit_deg=1.0,
    )

    assert claim.status == "failed"
    assert "worst attitude RMS 1.400 deg" in claim.description
    assert claim.sensitivity[1].passed_runs == 1
    assert claim.sensitivity[1].total_runs == 2


def test_the_bound_is_the_requirements_own_so_it_may_close():
    claim = evaluate_payload_transport(
        [_window("hover", 0.15, 0.008)], attitude_limit_deg=1.0,
    )
    assert claim.criterion.accepted_for_requirement is True
    assert claim.criterion.source.name == "REQUIREMENT"
