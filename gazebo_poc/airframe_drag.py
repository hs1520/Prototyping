"""Geometry-derived parasitic drag for the generated airframe.

Why this exists
---------------
The generated SDF gives every rotor a ``LiftDrag`` plugin, but those produce
*rotor thrust*; the airframe body itself carried no aerodynamic drag at all.
A forward dash therefore had no terminal velocity — it simply accelerated for
as long as the dash window lasted, and the recorded "cruise speed" was a
function of how long we flew, not of the vehicle. The 2026-08-30 authoritative
run shows the consequence directly: ground speed *rose* from 14.2 m/s to
28.9 m/s after a 15 m/s headwind was injected, which no real headwind can do.

This module computes the equivalent flat-plate area ``f = sum(Cd_i * A_i)`` of
the airframe from the same geometry constants ``multirotor_sdf`` draws it
from, so the airframe that is drawn and the airframe the drag model sees
cannot diverge.

Honesty boundary
----------------
This is a bluff-body sum over the drawn primitives (hub cylinder, arm boxes,
payload box) with textbook drag coefficients. It is NOT CFD, it does not model
rotor-wake or body interference, and it assumes the frontal projection along
the body x-axis. It is a defensible parasitic-drag estimate for a multirotor
of this shape, and it is reported alongside every speed it influences.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .multirotor_sdf import (
    ARM_THICKNESS_M,
    HUB_HEIGHT_M,
    _motor_table,
    arm_length_m,
    arm_width_m,
    hub_radius_m,
)

# Textbook bluff-body drag coefficients (Re ~1e5, incompressible).
#: Circular cylinder, axis normal to the flow.
CD_HUB = 1.0
#: Rectangular-section arm, sharp edges.
CD_ARM = 1.2
#: Rectangular box (the slung payload).
CD_PAYLOAD = 1.05

#: The payload box drawn by ``templates/all_models/payload_box/model.sdf``.
#: ``run_flight._prepare_payload_model`` re-derives the box inertia from these
#: same numbers, so the drawn box and the box the drag model sees are one box.
PAYLOAD_BOX_SIZE_M: Tuple[float, float, float] = (0.12, 0.08, 0.06)

AIR_DENSITY = 1.2041


@dataclass(frozen=True)
class DragBreakdown:
    """Per-component frontal area and flat-plate contribution, in m^2."""

    hub_area_m2: float
    arms_area_m2: float
    payload_area_m2: float
    flat_plate_area_m2: float

    @property
    def frontal_area_m2(self) -> float:
        return self.hub_area_m2 + self.arms_area_m2 + self.payload_area_m2

    def as_dict(self) -> dict:
        return {
            "hub_area_m2": round(self.hub_area_m2, 6),
            "arms_area_m2": round(self.arms_area_m2, 6),
            "payload_area_m2": round(self.payload_area_m2, 6),
            "frontal_area_m2": round(self.frontal_area_m2, 6),
            "flat_plate_area_m2": round(self.flat_plate_area_m2, 6),
            "cd_hub": CD_HUB,
            "cd_arm": CD_ARM,
            "cd_payload": CD_PAYLOAD,
            "method": (
                "bluff-body sum over the drawn SDF primitives "
                "(hub cylinder + arm boxes + payload box), body-x projection"
            ),
        }


def _arm_frontal_area_m2(angles_deg: Sequence[float], arm_len: float,
                         arm_w: float) -> float:
    """Projected area of the arm boxes onto the y-z plane.

    ``multirotor_sdf`` places arm *i* at yaw ``atan2(-L sin(th), L cos(th))``,
    i.e. yaw ``-th``. A box of length ``arm_len`` and width ``arm_w`` at that
    yaw spans ``arm_len*|sin(th)| + arm_w*|cos(th)|`` across the flow.
    """
    total = 0.0
    for angle in angles_deg:
        th = math.radians(angle)
        span = arm_len * abs(math.sin(th)) + arm_w * abs(math.cos(th))
        total += ARM_THICKNESS_M * span
    return total


def drag_breakdown(rotor_count: int, rotor_radius_m: float,
                   payload_attached: bool = False) -> DragBreakdown:
    """Frontal areas and equivalent flat-plate area for the drawn airframe."""
    arm_len = arm_length_m(rotor_radius_m)
    arm_w = arm_width_m(rotor_radius_m)
    angles: List[float] = [angle for angle, _spin in _motor_table(rotor_count)]

    hub = 2.0 * hub_radius_m(arm_len) * HUB_HEIGHT_M
    arms = _arm_frontal_area_m2(angles, arm_len, arm_w)
    _, box_y, box_z = PAYLOAD_BOX_SIZE_M
    payload = (box_y * box_z) if payload_attached else 0.0

    flat_plate = CD_HUB * hub + CD_ARM * arms + CD_PAYLOAD * payload
    return DragBreakdown(hub, arms, payload, flat_plate)


def flat_plate_area_m2(rotor_count: int, rotor_radius_m: float,
                       payload_attached: bool = False) -> float:
    """Equivalent flat-plate area ``f`` (m^2) for the drawn airframe."""
    return drag_breakdown(rotor_count, rotor_radius_m, payload_attached).flat_plate_area_m2


def terminal_speed_mps(flat_plate_m2: float, mass_kg: float,
                       tilt_rad: float) -> float:
    """Steady forward airspeed where parasitic drag balances the tilted thrust.

    ``m g tan(theta) = 0.5 rho f V^2``. This is the prediction the measured
    dash is checked against; a dash that never reaches it has not cruised.
    """
    if flat_plate_m2 <= 0 or mass_kg <= 0 or tilt_rad <= 0:
        return 0.0
    horizontal_n = mass_kg * 9.81 * math.tan(tilt_rad)
    return math.sqrt(horizontal_n / (0.5 * AIR_DENSITY * flat_plate_m2))
