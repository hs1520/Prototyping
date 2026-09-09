"""Requirement Evidence for carrying a payload, judged on observed attachment.

REQ-FUNC-003 asks the vehicle to transport a payload while holding an attitude
bound. Whether it was carrying anything when the attitude was measured used to be
inferred from configuration (``payload_release and payload_mass_kg > 0``), which
says what the harness was asked to do; the delivery payload is released mid-flight,
so a window measured after separation counted as transport evidence. Attachment is
now observed per window from the Gazebo poses of vehicle and payload: windows
observed unloaded are reported but excluded, and a window whose attachment could
not be observed is not assumed loaded.
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

# Attachment is judged on vertical separation, not 3-D distance.
#
# Under a fast dash the detachable joint is compliant enough that an attached
# payload trails: at 1656 m downrange the vehicle sat at (..., 1656.59, 9.32) and
# its attached payload at (..., 1665.01, 9.23) - 8.9 m apart horizontally, 0.09 m
# vertically - and a 2 m 3-D bound called that released.
#
# A released payload falls, so vertical separation splits the two cleanly: 0.09 m
# while carrying, ~9 m once the box is down and the vehicle is still at altitude.
ATTACHED_MAX_VERTICAL_SEPARATION_M = 2.0

# Reporting only: the horizontal lag says something about joint compliance, not
# about whether the payload is aboard.
ATTACHED_MAX_DISTANCE_M = ATTACHED_MAX_VERTICAL_SEPARATION_M


@dataclass(frozen=True)
class PayloadAttachment:
    """What the poses said about the payload at one moment."""

    observed: bool
    distance_m: Optional[float] = None
    vertical_separation_m: Optional[float] = None

    @property
    def attached(self) -> bool:
        return (
            self.observed
            and self.vertical_separation_m is not None
            and self.vertical_separation_m <= ATTACHED_MAX_VERTICAL_SEPARATION_M
        )

    def as_dict(self) -> dict:
        return {
            "observed": self.observed,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 4),
            "vertical_separation_m": (
                None if self.vertical_separation_m is None
                else round(self.vertical_separation_m, 4)
            ),
            "attached": self.attached,
            "attached_max_vertical_separation_m": ATTACHED_MAX_VERTICAL_SEPARATION_M,
        }


def observe_attachment(vehicle_xyz, payload_xyz) -> PayloadAttachment:
    """Attachment from two ground-truth poses, decided on vertical separation."""
    if vehicle_xyz is None or payload_xyz is None:
        return PayloadAttachment(observed=False)
    return PayloadAttachment(
        observed=True,
        distance_m=math.dist(vehicle_xyz, payload_xyz),
        vertical_separation_m=abs(vehicle_xyz[2] - payload_xyz[2]),
    )


@dataclass(frozen=True)
class TransportWindow:
    """One measurement window, and what the vehicle was carrying during it."""

    label: str
    attachment: PayloadAttachment
    attitude_rms_deg: Optional[float] = None
    speed_mps: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "TransportWindow":
        """Rebuild a window from a recorded run, so the report and the flight judge the same
        observation.
        """
        state = data.get("attachment") or {}
        return cls(
            label=str(data.get("label", "?")),
            attachment=PayloadAttachment(
                observed=bool(state.get("observed")),
                distance_m=state.get("distance_m"),
                vertical_separation_m=state.get("vertical_separation_m"),
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

    The claim declares PAYLOAD_STATE_OBSERVED only when attachment was observed, so
    a run that never looked cannot close the payload clause on configuration alone.
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
        + (f" ({w.attachment.vertical_separation_m:.2f} m vertical, "
           f"{w.attachment.distance_m:.2f} m total)"
           if w.attachment.vertical_separation_m is not None else "")
        + (f", RMS {w.attitude_rms_deg:.3f} deg" if w.attitude_rms_deg is not None else "")
        for w in windows
    )
