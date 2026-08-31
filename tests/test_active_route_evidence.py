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
