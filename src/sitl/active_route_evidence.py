"""Requirement Evidence for adopting a revised waypoint in the active route."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class RouteObservationKind(str, Enum):
    """The subsystem that exposed a waypoint coordinate."""

    MISSION_STORAGE = "mission_storage"
    ACTIVE_CONTROLLER_TARGET = "active_controller_target"


@dataclass(frozen=True)
class RouteObservation:
    observed_at_s: float
    kind: RouteObservationKind
    lat_e7: int
    lon_e7: int


@dataclass(frozen=True)
class ActiveRouteUpdateEvidence:
    status: str
    latency_s: float | None
    storage_revision_seen: bool
    active_target_seen: bool
    description: str


def evaluate_active_route_update(
    *,
    accepted_at_s: float,
    target_lat_e7: int,
    target_lon_e7: int,
    observations: Iterable[RouteObservation],
    max_latency_s: float,
) -> ActiveRouteUpdateEvidence:
    """Judge adoption from the controller target, never mission read-back."""
    observed = tuple(observations)
    matching = tuple(
        observation
        for observation in observed
        if observation.observed_at_s >= accepted_at_s
        and observation.lat_e7 == target_lat_e7
        and observation.lon_e7 == target_lon_e7
    )
    storage_seen = any(
        observation.kind is RouteObservationKind.MISSION_STORAGE
        for observation in matching
    )
    active = tuple(
        observation
        for observation in matching
        if observation.kind is RouteObservationKind.ACTIVE_CONTROLLER_TARGET
    )
    if not active:
        # "the navigator had no target" and "the navigator held the OLD target"
        # are different failures, and only the second says the revision was
        # ignored. Naming the coordinate it held says which one happened.
        controller_samples = tuple(
            observation for observation in observed
            if observation.kind is RouteObservationKind.ACTIVE_CONTROLLER_TARGET
        )
        controller_observer_available = bool(controller_samples)
        held = ""
        if controller_samples:
            last = controller_samples[-1]
            distinct = {(o.lat_e7, o.lon_e7) for o in controller_samples}
            held = (
                f" The navigation target was sampled {len(controller_samples)} times "
                f"and held {len(distinct)} distinct coordinate(s), last "
                f"({last.lat_e7}, {last.lon_e7}) against a revision to "
                f"({target_lat_e7}, {target_lon_e7})."
            )
        return ActiveRouteUpdateEvidence(
            status=("failed" if controller_observer_available else "inconclusive"),
            latency_s=None,
            storage_revision_seen=storage_seen,
            active_target_seen=False,
            description=(
                "mission storage carried the revision, but no active-controller "
                "target samples were available; active-route adoption is not "
                "observable with this telemetry channel"
                if storage_seen and not controller_observer_available else
                "the revised coordinate was present in mission storage but never "
                "appeared as the active navigation-controller target" + held
                if storage_seen else
                "the revised coordinate appeared in neither mission storage nor "
                "the active navigation-controller target" + held
            ),
        )
    adopted_at = min(observation.observed_at_s for observation in active)
    latency = round(adopted_at - accepted_at_s, 6)
    return ActiveRouteUpdateEvidence(
        status="verified" if latency <= max_latency_s else "failed",
        latency_s=latency,
        storage_revision_seen=storage_seen,
        active_target_seen=True,
        description=(
            f"the revised coordinate became the active navigation-controller "
            f"target {latency:.3f} s after MISSION_ACK (limit "
            f"{max_latency_s:.3f} s)"
        ),
    )
