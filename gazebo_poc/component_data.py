"""Real, cited motor+propeller bench data — the fidelity source for power/endurance.

The analytic estimator (src/dse/physics_estimator.py) lumps hover efficiency into a single
figure-of-merit (FOM=0.62) + drive efficiency (ETA_DRIVE=0.75). This module replaces that
lumped guess with a SPECIFIC real motor+prop's published thrust→current curve, so endurance
is grounded in a datasheet operating point rather than an assumed efficiency.

Source: T-Motor Antigravity MN5008 KV340 with P18×6.1" CF propeller, 24 V (6S) bench test.
  https://store.tmotor.com/product/mn5008-kv340-motor-antigravity-type.html
Each row is PER MOTOR. This is static-thrust bench data (no ground effect / forward flight /
battery sag under pack-level load) — high-fidelity vs the lumped FOM, but NOT ground truth;
real flight carries a further ~10-20% (documented, not hidden).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class MotorPropPoint:
    throttle: float   # fraction 0..1
    thrust_g: float   # grams, per motor
    current_a: float  # amperes, per motor
    power_w: float    # watts, per motor


@dataclass(frozen=True)
class MotorProp:
    name: str
    source_url: str
    voltage_v: float
    cells: int
    curve: Tuple[MotorPropPoint, ...]

    def max_thrust_g(self) -> float:
        return max(p.thrust_g for p in self.curve)

    def interp_at_thrust(self, thrust_g: float) -> Tuple[float, float]:
        """Linear-interpolate (current_a, power_w) at a per-motor thrust (g). Clamps to the
        curve's endpoints; raises if thrust exceeds the motor's max (can't hover)."""
        pts: List[MotorPropPoint] = sorted(self.curve, key=lambda p: p.thrust_g)
        if thrust_g <= pts[0].thrust_g:
            return pts[0].current_a, pts[0].power_w
        if thrust_g > pts[-1].thrust_g:
            raise ValueError(
                f"required {thrust_g:.0f} g/motor exceeds {self.name} max "
                f"{pts[-1].thrust_g:.0f} g — motor can't hover this design")
        for a, b in zip(pts, pts[1:]):
            if a.thrust_g <= thrust_g <= b.thrust_g:
                f = (thrust_g - a.thrust_g) / (b.thrust_g - a.thrust_g)
                return (a.current_a + f * (b.current_a - a.current_a),
                        a.power_w + f * (b.power_w - a.power_w))
        return pts[-1].current_a, pts[-1].power_w


# T-Motor MN5008 KV340 + P18×6.1" CF, 24V/6S (per-motor bench rows).
MN5008_KV340_18x61 = MotorProp(
    name="T-Motor MN5008 KV340 + P18x6.1 (6S)",
    source_url="https://store.tmotor.com/product/mn5008-kv340-motor-antigravity-type.html",
    voltage_v=22.2, cells=6,
    curve=(
        MotorPropPoint(0.40,  996.0,  3.76,  88.0),
        MotorPropPoint(0.45, 1238.0,  5.12, 120.0),
        MotorPropPoint(0.50, 1541.0,  6.93, 162.0),
        MotorPropPoint(0.55, 1822.0,  8.95, 208.0),
        MotorPropPoint(0.60, 2120.0, 11.12, 258.0),
        MotorPropPoint(0.65, 2394.0, 13.46, 311.0),
        MotorPropPoint(0.75, 2966.0, 18.83, 432.0),
        MotorPropPoint(1.00, 4215.0, 33.50, 754.0),
    ),
)

CATALOG = {MN5008_KV340_18x61.name: MN5008_KV340_18x61}
