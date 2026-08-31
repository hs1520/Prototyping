"""Requirement Evidence for carrying a payload, judged on observed attachment.

REQ-FUNC-003 asks the vehicle to *transport* a payload while holding an
attitude bound. Whether it was transporting anything at the moment the attitude
was measured is a physical fact about the simulation, and it was previously
inferred from the run's configuration::

    cruise_sweep_payload_attached = bool(payload_release and payload_mass_kg > 0)

That is a statement about what the harness was asked to do, not about what the
vehicle was carrying. The delivery payload is released during the same flight,
so a cruise window measured after separation would have been counted as
transport evidence on the strength of a flag set before takeoff.

Attachment is therefore observed per window, from the Gazebo poses of the
vehicle and the payload model: a payload still on the detachable joint tracks
the airframe, a released one does not. Windows observed unloaded are reported
but excluded, and a window whose attachment could not be observed at all is
never assumed loaded — the absence of an observation is not evidence of one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.prototyping.verification_obligations import (
    CriterionEvaluation,
    CriterionSource,
    EvidenceCapability,
    EvidenceClaim,
    VerificationCriterion,
)

#: A payload on the detachable joint sits ~0.15 m under the airframe origin and
#: moves with it. This bound is loose enough for that offset plus simulation
#: jitter, and far tighter than the separation a released box reaches within
#: the settle window that follows any cruise point.
ATTACHED_MAX_DISTANCE_M = 2.0


@dataclass(frozen=True)
class PayloadAttachment:
    """What the poses said about the payload at one moment."""

    observed: bool
    distance_m: Optional[float] = None

    @property
    def attached(self) -> bool:
        return (
            self.observed
            and self.distance_m is not None
            and self.distance_m <= ATTACHED_MAX_DISTANCE_M
        )

    def as_dict(self) -> dict:
        return {
            "observed": self.observed,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 4),
            "attached": self.attached,
            "attached_max_distance_m": ATTACHED_MAX_DISTANCE_M,
        }


def observe_attachment(vehicle_xyz, payload_xyz) -> PayloadAttachment:
    """Attachment as a distance between two ground-truth poses."""
    if vehicle_xyz is None or payload_xyz is None:
        return PayloadAttachment(observed=False)
    return PayloadAttachment(observed=True, distance_m=math.dist(vehicle_xyz, payload_xyz))


@dataclass(frozen=True)
class TransportWindow:
    """One measurement window, and what the vehicle was carrying during it."""

    label: str
    attachment: PayloadAttachment
    attitude_rms_deg: Optional[float] = None
    speed_mps: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "TransportWindow":
        """Rebuild a window from a recorded run, so the report and the flight
        judge the same observation rather than two readings of it."""
        state = data.get("attachment") or {}
        return cls(
            label=str(data.get("label", "?")),
            attachment=PayloadAttachment(
                observed=bool(state.get("observed")),
                distance_m=state.get("distance_m"),
            ),
            attitude_rms_deg=data.get("attitude_rms_deg"),
            speed_mps=data.get("speed_mps"),
        )

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "attitude_rms_deg": (
                None if self.attitude_rms_deg is None
                else round(self.attitude_rms_deg, 4)
            ),
            "speed_mps": None if self.speed_mps is None else round(self.speed_mps, 3),
            "attachment": self.attachment.as_dict(),
        }


def evaluate_payload_transport(
    windows: Sequence[TransportWindow],
    *,
    attitude_limit_deg: float,
) -> EvidenceClaim:
    """Judge the transport attitude bound on windows observed to be carrying.

    The returned claim declares PAYLOAD_STATE_OBSERVED only when attachment was
    actually observed, so a run that never looked cannot close the payload
    clause on configuration alone.
    """
    measured = [
        window for window in windows
        if window.attitude_rms_deg is not None
    ]
    loaded = [window for window in measured if window.attachment.attached]
    unloaded = [window for window in measured if not window.attachment.attached]
    any_observation = any(window.attachment.observed for window in windows)

    criterion = VerificationCriterion(
        metric="attitude_rms_deg",
        operator="<=",
        threshold=float(attitude_limit_deg),
        unit="deg",
        source=CriterionSource.REQUIREMENT,
        basis="roll and pitch RMS bound stated by the requirement",
        accepted_for_requirement=True,
    )

    capabilities = {EvidenceCapability.ATTITUDE_RMS_MEASURED}
    if any_observation:
        capabilities.add(EvidenceCapability.PAYLOAD_STATE_OBSERVED)

    if not loaded:
        reason = (
            "attachment was never observed, so no window can be shown to have "
            "been carrying anything"
            if not any_observation else
            f"every one of {len(measured)} measured windows was observed "
            "UNLOADED, so none of them is transport evidence"
        )
        return EvidenceClaim(
            description=(
                f"payload transport not demonstrated: {reason}. "
                f"Windows: {_describe(windows)}"
            ),
            status="planned",
            capabilities=frozenset(capabilities),
            criterion=criterion,
            sensitivity=(
                CriterionEvaluation(
                    "measurement window observed carrying the payload",
                    len(loaded),
                    len(measured),
                ),
            ),
        )

    worst = max(loaded, key=lambda window: window.attitude_rms_deg)
    met = worst.attitude_rms_deg <= attitude_limit_deg
    excluded = (
        f"; {len(unloaded)} window(s) excluded as observed unloaded "
        f"({', '.join(window.label for window in unloaded)})"
        if unloaded else ""
    )
    return EvidenceClaim(
        description=(
            f"payload transport over {len(loaded)} window(s) observed carrying the "
            f"payload: worst attitude RMS {worst.attitude_rms_deg:.3f} deg at "
            f"{worst.label} against a {attitude_limit_deg:.1f} deg bound{excluded}. "
            f"Windows: {_describe(windows)}"
        ),
        status="verified" if met else "failed",
        capabilities=frozenset(capabilities),
        criterion=criterion,
        sensitivity=(
            CriterionEvaluation(
                "measurement window observed carrying the payload",
                len(loaded),
                len(measured),
            ),
            CriterionEvaluation(
                f"attitude RMS <= {attitude_limit_deg:g} deg while carrying",
                sum(w.attitude_rms_deg <= attitude_limit_deg for w in loaded),
                len(loaded),
            ),
        ),
    )


def _describe(windows: Sequence[TransportWindow]) -> str:
    if not windows:
        return "none"
    return "; ".join(
        f"{w.label} "
        + ("attached" if w.attachment.attached
           else "UNLOADED" if w.attachment.observed else "attachment not observed")
        + (f" at {w.attachment.distance_m:.2f} m" if w.attachment.distance_m is not None else "")
        + (f", RMS {w.attitude_rms_deg:.3f} deg" if w.attitude_rms_deg is not None else "")
        for w in windows
    )
