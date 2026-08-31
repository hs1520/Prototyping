from src.sitl.active_route_evidence import (
    RouteObservation,
    RouteObservationKind,
    evaluate_active_route_update,
)


def test_mission_storage_readback_does_not_prove_active_route_incorporation():
    evidence = evaluate_active_route_update(
        accepted_at_s=10.0,
        target_lat_e7=-353626220,
        target_lon_e7=1491658370,
        observations=[RouteObservation(
            observed_at_s=10.006,
            kind=RouteObservationKind.MISSION_STORAGE,
            lat_e7=-353626220,
            lon_e7=1491658370,
        )],
        max_latency_s=1.0,
    )

    assert evidence.status == "inconclusive"
    assert evidence.storage_revision_seen is True
    assert evidence.active_target_seen is False
    assert evidence.latency_s is None


def test_other_controller_targets_without_revision_are_a_failure():
    evidence = evaluate_active_route_update(
        accepted_at_s=10.0,
        target_lat_e7=-353626220,
        target_lon_e7=1491658370,
        observations=[RouteObservation(
            observed_at_s=10.2,
            kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
            lat_e7=-353620000,
            lon_e7=1491652000,
        )],
        max_latency_s=1.0,
    )

    assert evidence.status == "failed"
    assert evidence.active_target_seen is False


def test_controller_target_proves_when_the_active_route_adopted_revision():
    evidence = evaluate_active_route_update(
        accepted_at_s=10.0,
        target_lat_e7=-353626220,
        target_lon_e7=1491658370,
        observations=[
            RouteObservation(
                observed_at_s=10.004,
                kind=RouteObservationKind.MISSION_STORAGE,
                lat_e7=-353626220,
                lon_e7=1491658370,
            ),
            RouteObservation(
                observed_at_s=10.083,
                kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                lat_e7=-353626220,
                lon_e7=1491658370,
            ),
        ],
        max_latency_s=1.0,
    )

    assert evidence.status == "verified"
    assert evidence.storage_revision_seen is True
    assert evidence.active_target_seen is True
    assert evidence.latency_s == 0.083


def test_controller_target_after_deadline_is_a_requirement_failure():
    evidence = evaluate_active_route_update(
        accepted_at_s=10.0,
        target_lat_e7=-353626220,
        target_lon_e7=1491658370,
        observations=[RouteObservation(
            observed_at_s=11.2,
            kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
            lat_e7=-353626220,
            lon_e7=1491658370,
        )],
        max_latency_s=1.0,
    )

    assert evidence.status == "failed"
    assert evidence.latency_s == 1.2


def test_a_navigator_that_held_the_old_target_is_named_as_such():
    """Measured on the authoritative model: the revision reached mission storage
    in 12 ms and the navigation controller kept the old destination for the whole
    observation window. "No target at all" and "held the old one" are different
    failures, and only the second says the revision was ignored."""
    old_coord = (-353617620, 1491652370)
    revised = (-353617620, 1491662370)
    observations = [
        RouteObservation(observed_at_s=1.0 + i * 0.1,
                         kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                         lat_e7=old_coord[0], lon_e7=old_coord[1])
        for i in range(20)
    ]
    observations.append(RouteObservation(
        observed_at_s=3.0, kind=RouteObservationKind.MISSION_STORAGE,
        lat_e7=revised[0], lon_e7=revised[1]))

    evidence = evaluate_active_route_update(
        accepted_at_s=1.0, target_lat_e7=revised[0], target_lon_e7=revised[1],
        observations=observations, max_latency_s=1.0,
    )

    assert evidence.status == "failed"
    assert evidence.storage_revision_seen is True
    assert evidence.active_target_seen is False
    assert "sampled 20 times and held 1 distinct coordinate(s)" in evidence.description
    assert str(old_coord[1]) in evidence.description
    assert str(revised[1]) in evidence.description


def test_the_reported_target_is_never_bit_identical_to_the_commanded_one():
    """The 2026-08-31 false negative. POSITION_TARGET_GLOBAL_INT reports the
    controller's own target, converted to a NEU offset from the EKF origin and
    back; the run held the revised waypoint 0.58 m from the commanded integers
    while sitting 90 m from the one it replaced. Integer equality called that
    "never adopted" — a criterion only mission storage could ever satisfy, and
    storage is the observable this module exists to reject."""
    old = (-353617620, 1491652370)
    revised = (-353617620, 1491662370)
    reported = (-353617590, 1491662318)          # measured, not constructed

    observations = [
        RouteObservation(observed_at_s=0.5 + i * 0.1,
                         kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                         lat_e7=old[0], lon_e7=old[1])
        for i in range(5)
    ] + [
        RouteObservation(observed_at_s=1.0 + i * 0.1,
                         kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                         lat_e7=reported[0], lon_e7=reported[1])
        for i in range(51)
    ]

    evidence = evaluate_active_route_update(
        accepted_at_s=1.0, target_lat_e7=revised[0], target_lon_e7=revised[1],
        observations=observations, max_latency_s=1.0,
    )

    assert evidence.status == "verified"
    assert evidence.active_target_seen is True
    assert evidence.separation_m < 1.0            # 0.58 m from the revision
    assert evidence.baseline_separation_m > 80.0  # 90 m from what it replaced
    assert "unambiguous" in evidence.description


def test_a_revision_too_small_to_discriminate_is_inconclusive_not_verified():
    """Within tolerance of the new coordinate AND within tolerance of the old
    one is not evidence of anything: a navigator that ignored the revision
    reads identically. The scenario failed, not the vehicle, and the verdict
    must not launder that into a pass."""
    old = (-353617620, 1491652370)
    revised = (-353617620, 1491652380)            # 0.09 m away
    reported = (-353617620, 1491652374)

    evidence = evaluate_active_route_update(
        accepted_at_s=1.0, target_lat_e7=revised[0], target_lon_e7=revised[1],
        observations=[
            RouteObservation(observed_at_s=0.9,
                             kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                             lat_e7=old[0], lon_e7=old[1]),
            RouteObservation(observed_at_s=1.1,
                             kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                             lat_e7=reported[0], lon_e7=reported[1]),
        ],
        max_latency_s=1.0,
    )

    assert evidence.status == "inconclusive"
    assert "cannot tell adoption from standing still" in evidence.description


def test_an_exact_hit_needs_no_baseline_to_discriminate():
    """Approximate matching is what needs a baseline. An exact hit cannot be a
    near-miss of some other coordinate, so it stands on its own."""
    revised = (-353617620, 1491662370)
    evidence = evaluate_active_route_update(
        accepted_at_s=1.0, target_lat_e7=revised[0], target_lon_e7=revised[1],
        observations=[RouteObservation(
            observed_at_s=1.08,
            kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
            lat_e7=revised[0], lon_e7=revised[1])],
        max_latency_s=1.0,
    )
    assert evidence.status == "verified"
    assert evidence.baseline_separation_m is None


def test_a_target_near_the_revision_but_far_past_the_deadline_still_fails():
    """Loosening the coordinate test must not loosen the timing one."""
    old = (-353617620, 1491652370)
    revised = (-353617620, 1491662370)
    evidence = evaluate_active_route_update(
        accepted_at_s=1.0, target_lat_e7=revised[0], target_lon_e7=revised[1],
        observations=[
            RouteObservation(observed_at_s=0.9,
                             kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                             lat_e7=old[0], lon_e7=old[1]),
            RouteObservation(observed_at_s=3.4,
                             kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
                             lat_e7=-353617590, lon_e7=1491662318),
        ],
        max_latency_s=1.0,
    )
    assert evidence.status == "failed"
    assert evidence.latency_s == 2.4


def test_sampling_that_starts_at_acceptance_reports_an_upper_bound():
    """With no pre-revision sample, an adoption that had already happened is
    indistinguishable from one that happened at the first sample. Reporting
    that number as a measurement would overstate what was observed."""
    revised = (-353617620, 1491662370)
    evidence = evaluate_active_route_update(
        accepted_at_s=1.0, target_lat_e7=revised[0], target_lon_e7=revised[1],
        observations=[RouteObservation(
            observed_at_s=1.1,
            kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
            lat_e7=revised[0], lon_e7=revised[1])],
        max_latency_s=1.0,
    )
    assert evidence.status == "verified"
    assert evidence.latency_is_upper_bound is True
    assert "upper bound" in evidence.description


def test_a_commanded_pre_revision_coordinate_can_stand_in_for_a_baseline():
    """When the run did not sample before the revision, the commanded old
    coordinate still establishes that the two are far enough apart to tell
    apart — a weaker basis than an observed baseline, and enough for this."""
    evidence = evaluate_active_route_update(
        accepted_at_s=1.0,
        target_lat_e7=-353617620, target_lon_e7=1491662370,
        observations=[RouteObservation(
            observed_at_s=1.1,
            kind=RouteObservationKind.ACTIVE_CONTROLLER_TARGET,
            lat_e7=-353617590, lon_e7=1491662318)],
        max_latency_s=1.0,
        pre_revision_lat_e7=-353617620, pre_revision_lon_e7=1491652370,
    )
    assert evidence.status == "verified"
    assert evidence.baseline_separation_m > 80.0
