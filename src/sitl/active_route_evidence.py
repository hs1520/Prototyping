"""Requirement Evidence for adopting a revised waypoint in the active route."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

# How far the navigation controller's reported target may sit from the
# commanded coordinate and still be that coordinate.
#
# POSITION_TARGET_GLOBAL_INT does not echo the mission item: ArduPilot converts
# it to a NEU offset from the EKF origin and back for the report, so the round
# trip carries rounding and origin error - the 2026-08-31 run held the revised
# waypoint at 0.58 m from the commanded coordinate. Integer equality, demanded
# here before, is satisfiable only by mission storage, which returns what was
# written.
ADOPTION_TOLERANCE_M = 2.0

# An approximate match means "adopted" only if the revision was far enough
# away to tell adoption from standing still. Below this multiple of the
# tolerance no observation can discriminate, so the verdict blames the
# scenario rather than the vehicle.
DISCRIMINATION_FACTOR = 4.0

_M_PER_DEG = 111_320.0


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
    separation_m: float | None = None
    baseline_separation_m: float | None = None
    latency_is_upper_bound: bool = False


def separation_m(lat_e7_a: int, lon_e7_a: int, lat_e7_b: int, lon_e7_b: int) -> float:
    """Ground distance between two 1e-7-degree coordinates."""
    lat_a, lat_b = lat_e7_a / 1e7, lat_e7_b / 1e7
    d_lat = (lat_a - lat_b) * _M_PER_DEG
    d_lon = (
        (lon_e7_a - lon_e7_b) / 1e7
        * _M_PER_DEG * math.cos(math.radians((lat_a + lat_b) / 2.0))
    )
    return math.hypot(d_lat, d_lon)


def evaluate_active_route_update(
    *,
    accepted_at_s: float,
    target_lat_e7: int,
    target_lon_e7: int,
    observations: Iterable[RouteObservation],
    max_latency_s: float,
    pre_revision_lat_e7: Optional[int] = None,
    pre_revision_lon_e7: Optional[int] = None,
) -> ActiveRouteUpdateEvidence:
    """Judge adoption from the controller target, not mission read-back."""
    observed = tuple(observations)

    def gap(observation: RouteObservation) -> float:
        return separation_m(
            observation.lat_e7, observation.lon_e7, target_lat_e7, target_lon_e7)

    exact = [
        item for item in observed
        if item.lat_e7 == target_lat_e7 and item.lon_e7 == target_lon_e7
    ]
    storage_seen = any(
        item.kind is RouteObservationKind.MISSION_STORAGE for item in exact
    )
    controller_samples = tuple(
        item for item in observed
        if item.kind is RouteObservationKind.ACTIVE_CONTROLLER_TARGET
    )
    after = tuple(
        item for item in controller_samples if item.observed_at_s >= accepted_at_s
    )
    # What the controller was steering to before the revision, measured through
    # the same transform as the samples after it. Comparing against the commanded
    # pre-revision item would mix a commanded coordinate with a reported one and
    # charge the transform error to the vehicle.
    before = tuple(
        item for item in controller_samples if item.observed_at_s < accepted_at_s
    )
    baseline_gap: Optional[float] = None
    if before:
        baseline_gap = min(gap(item) for item in before)
    elif pre_revision_lat_e7 is not None and pre_revision_lon_e7 is not None:
        baseline_gap = separation_m(
            pre_revision_lat_e7, pre_revision_lon_e7, target_lat_e7, target_lon_e7)

    # An exact hit needs no discrimination: it cannot be a near-miss of another
    # coordinate. An approximate one does, or "the target never moved" and "the
    # target arrived" are the same measurement.
    adopted = [item for item in after if gap(item) <= ADOPTION_TOLERANCE_M]
    exact_adopted = [
        item for item in after
        if item.lat_e7 == target_lat_e7 and item.lon_e7 == target_lon_e7
    ]
    discriminating = (
        baseline_gap is not None
        and baseline_gap >= DISCRIMINATION_FACTOR * ADOPTION_TOLERANCE_M
    )

    if adopted and not exact_adopted and not discriminating:
        held = adopted[0]
        return ActiveRouteUpdateEvidence(
            status="inconclusive",
            latency_s=None,
            storage_revision_seen=storage_seen,
            active_target_seen=True,
            separation_m=gap(held),
            baseline_separation_m=baseline_gap,
            description=(
                f"the navigation target sits {gap(held):.2f} m from the revised "
                f"coordinate, within the {ADOPTION_TOLERANCE_M:.1f} m adoption "
                "tolerance — but "
                + (
                    f"the revision moved the waypoint only "
                    f"{baseline_gap:.2f} m, so a target that never moved would "
                    "read the same; the scenario cannot tell adoption from "
                    "standing still"
                    if baseline_gap is not None else
                    "no pre-revision controller target was sampled, so there is "
                    "nothing to tell adoption from standing still"
                )
            ),
        )

    if not adopted:
        # "no target" and "held the old target" are different failures, and only the
        # second says the revision was ignored; naming the coordinate it held says
        # which one happened.
        held = ""
        if controller_samples:
            last = controller_samples[-1]
            distinct = {(o.lat_e7, o.lon_e7) for o in controller_samples}
            held = (
                f" The navigation target was sampled {len(controller_samples)} times "
                f"and held {len(distinct)} distinct coordinate(s), last "
                f"({last.lat_e7}, {last.lon_e7}) at {gap(last):.2f} m from the "
                f"revision to ({target_lat_e7}, {target_lon_e7})."
            )
        return ActiveRouteUpdateEvidence(
            status=("failed" if controller_samples else "inconclusive"),
            latency_s=None,
            storage_revision_seen=storage_seen,
            active_target_seen=False,
            baseline_separation_m=baseline_gap,
            description=(
                "mission storage carried the revision, but no active-controller "
                "target samples were available; active-route adoption is not "
                "observable with this telemetry channel"
                if storage_seen and not controller_samples else
                "the revised coordinate was present in mission storage but never "
                "appeared as the active navigation-controller target" + held
                if storage_seen else
                "the revised coordinate appeared in neither mission storage nor "
                "the active navigation-controller target" + held
            ),
        )

    first = min(adopted, key=lambda item: item.observed_at_s)
    latency = round(first.observed_at_s - accepted_at_s, 6)
    # Sampling that begins at acceptance cannot see an adoption that had already
    # happened: the first sample is then an upper bound, not a measurement.
    upper_bound = not before and first is min(
        after, key=lambda item: item.observed_at_s)
    return ActiveRouteUpdateEvidence(
        status="verified" if latency <= max_latency_s else "failed",
        latency_s=latency,
        storage_revision_seen=storage_seen,
        active_target_seen=True,
        separation_m=gap(first),
        baseline_separation_m=baseline_gap,
        latency_is_upper_bound=upper_bound,
        description=(
            f"the revised coordinate became the active navigation-controller "
            f"target {latency:.3f} s after MISSION_ACK (limit "
            f"{max_latency_s:.3f} s), reported {gap(first):.2f} m from the "
            f"commanded coordinate"
            + (f"; the controller had been steering {baseline_gap:.2f} m away "
               f"before the revision, so the move is unambiguous"
               if baseline_gap is not None else "")
            + ("; sampling began at acceptance, so this latency is an upper "
               "bound rather than a measurement" if upper_bound else "")
        ),
    )
