"""Transport evidence must come from windows observed to be carrying.

REQ-FUNC-003's attitude bound applies while *transporting*. Attachment used to
be inferred from the run configuration — a flag set before takeoff — even though
the same flight releases the payload part-way through, so a cruise window
measured after separation would have counted as transport evidence.
"""
from __future__ import annotations

from gazebo_poc.payload_transport_evidence import (
    ATTACHED_MAX_VERTICAL_SEPARATION_M,
    PayloadAttachment,
    TransportWindow,
    evaluate_payload_transport,
    observe_attachment,
)
from src.prototyping.verification_obligations import EvidenceCapability


def _window(label, vertical_separation, rms, observed=True, distance=None):
    return TransportWindow(
        label=label,
        attachment=PayloadAttachment(
            observed=observed,
            distance_m=distance if distance is not None else vertical_separation,
            vertical_separation_m=vertical_separation,
        ),
        attitude_rms_deg=rms,
    )


def test_attachment_is_decided_on_vertical_separation_not_total_distance():
    """Measured: at 1656 m downrange an ATTACHED payload trailed 8.4 m
    horizontally while staying 0.09 m from the vehicle in altitude. A 3-D
    distance bound called that released."""
    trailing = observe_attachment(
        (-9.49371, 1656.59, 9.31783), (-9.52536, 1665.01, 9.22905))
    assert trailing.observed and trailing.attached
    assert trailing.distance_m > 8.0            # far apart horizontally
    assert trailing.vertical_separation_m < 0.1  # but flying together

    # a released payload is on the ground while the vehicle is at altitude
    released = observe_attachment((-6.2, 887.4, 9.32), (-2.4, 3.0, 0.10))
    assert released.observed and not released.attached
    assert released.vertical_separation_m > ATTACHED_MAX_VERTICAL_SEPARATION_M

    # a pose that could not be read is not an attached payload
    missing = observe_attachment((10.0, 5.0, 10.0), None)
    assert not missing.observed and not missing.attached


def test_windows_measured_after_release_are_excluded_not_counted():
    claim = evaluate_payload_transport(
        [
            _window("hover", 0.15, 0.008),
            _window("cruise@rc1220", 0.15, 0.026),
            # flown after separation — good attitude, but not transport evidence
            _window("cruise@rc1100", 9.4, 0.004, distance=880.0),
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
        [_window("cruise@rc1220", 8.0, 0.02), _window("cruise@rc1100", 9.2, 0.03)],
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


def test_a_reconstruction_that_drops_the_deciding_field_is_caught():
    """The deciding quantity is vertical separation. A rebuild that carries the
    distance but forgets it silently decides "not attached" — which looks like a
    conservative verdict and is really a lost field. It cost a full flight to
    notice, because everything simply reported UNLOADED."""
    complete = TransportWindow.from_dict({
        "label": "hover", "attitude_rms_deg": 0.01,
        "attachment": {"observed": True, "distance_m": 8.42,
                       "vertical_separation_m": 0.089},
    })
    assert complete.attachment.attached

    lossy = TransportWindow.from_dict({
        "label": "hover", "attitude_rms_deg": 0.01,
        "attachment": {"observed": True, "distance_m": 8.42},
    })
    assert not lossy.attachment.attached
    # and it must not masquerade as a measured verdict
    assert lossy.attachment.vertical_separation_m is None


def test_horizontal_lag_alone_never_decides_attachment():
    """Measured trailing distances of 4.6, 6.9 and 8.9 m all belonged to a
    payload that was still flying with the vehicle."""
    for lag in (4.6, 6.9, 8.9):
        window = _window("cruise", 0.09, 0.02, distance=lag)
        assert window.attachment.attached, f"{lag} m of lag is not a release"
