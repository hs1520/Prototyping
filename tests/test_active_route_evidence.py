from src.sitl.active_route_evidence import (
    RouteObservation,
    RouteObservationKind,
    evaluate_active_route_update,
)


def test_storage_readback_not_proof():
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


def test_wrong_target_fails():
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


def test_controller_target_proves_adoption():
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


def test_target_after_deadline_fails():
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


def test_held_old_target_named():
    """Measured on the authoritative model: the revision reached mission storage in
    12 ms while the navigation controller kept the old destination for the whole
    observation window. "No target at all" and "held the old one" are different
    failures; only the second says the revision was ignored.
    """
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


def test_reported_target_not_bit_exact():
    """The 2026-08-31 false negative. POSITION_TARGET_GLOBAL_INT reports the
    controller's own target through a NEU round-trip via the EKF origin; the run
    held the revised waypoint 0.58 m from the commanded integers and 90 m from the
    one it replaced. Integer equality called that "never adopted" - a criterion
    only mission storage could satisfy, and storage is the observable this module
    rejects.
    """
    old = (-353617620, 1491652370)
    revised = (-353617620, 1491662370)
    reported = (-353617590, 1491662318)

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
    assert evidence.separation_m < 1.0
    assert evidence.baseline_separation_m > 80.0
    assert "unambiguous" in evidence.description


def test_tiny_revision_inconclusive():
    """Within tolerance of both the new and the old coordinate is not evidence: a
    navigator that ignored the revision reads identically. The scenario failed,
    not the vehicle, so the verdict is not a pass.
    """
    old = (-353617620, 1491652370)
    revised = (-353617620, 1491652380)
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


def test_exact_hit_needs_no_baseline():
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


def test_late_near_target_fails():
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


def test_latency_upper_bound():
    """With no pre-revision sample, an adoption that already happened is
    indistinguishable from one at the first sample; reporting that number as a
    measurement would overstate the observation.
    """
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


def test_commanded_baseline_accepted():
    """Without a pre-revision sample, the commanded old coordinate still shows the
    two are far enough apart to distinguish - a weaker basis than an observed
    baseline, and enough here.
    """
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
