"""Real component catalog for bottom-up realization.

Every value in DEFAULT_CATALOG must be traceable to a manufacturer page via
source_url and retrieved. Tests use synthetic catalogs instead of relying on
catalog coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class MotorPropPoint:
    throttle: float
    thrust_g: float
    current_a: float
    power_w: float


@dataclass(frozen=True)
class MotorPropCombo:
    name: str
    source_url: str
    retrieved: str
    voltage_v: float
    cells: int
    motor_mass_g: float
    prop_mass_g: float
    prop_diameter_in: float
    curve: Tuple[MotorPropPoint, ...]
    price_usd: float | None = None

    def max_thrust_g(self) -> float:
        return max(p.thrust_g for p in self.curve)

    def interp_at_thrust(self, thrust_g: float) -> Tuple[float, float]:
        """Return (current_a, power_w) at per-motor thrust using linear interpolation."""
        pts: List[MotorPropPoint] = sorted(self.curve, key=lambda p: p.thrust_g)
        if thrust_g <= pts[0].thrust_g:
            return pts[0].current_a, pts[0].power_w
        if thrust_g > pts[-1].thrust_g:
            raise ValueError(
                f"required {thrust_g:.0f} g/motor exceeds {self.name} max "
                f"{pts[-1].thrust_g:.0f} g"
            )
        for a, b in zip(pts, pts[1:]):
            if a.thrust_g <= thrust_g <= b.thrust_g:
                f = (thrust_g - a.thrust_g) / (b.thrust_g - a.thrust_g)
                return (
                    a.current_a + f * (b.current_a - a.current_a),
                    a.power_w + f * (b.power_w - a.power_w),
                )
        return pts[-1].current_a, pts[-1].power_w

    def throttle_at_thrust(self, thrust_g: float) -> float:
        """Return throttle fraction at per-motor thrust using linear interpolation."""
        pts: List[MotorPropPoint] = sorted(self.curve, key=lambda p: p.thrust_g)
        if thrust_g <= pts[0].thrust_g:
            return pts[0].throttle
        if thrust_g >= pts[-1].thrust_g:
            return pts[-1].throttle
        for a, b in zip(pts, pts[1:]):
            if a.thrust_g <= thrust_g <= b.thrust_g:
                f = (thrust_g - a.thrust_g) / (b.thrust_g - a.thrust_g)
                return a.throttle + f * (b.throttle - a.throttle)
        return pts[-1].throttle


@dataclass(frozen=True)
class BatteryPack:
    name: str
    source_url: str
    retrieved: str
    capacity_mah: float
    cells: int
    mass_g: float
    c_rating: float
    price_usd: float | None = None


@dataclass(frozen=True)
class Frame:
    name: str
    source_url: str
    retrieved: str
    mass_g: float
    arms: int
    max_prop_in: float
    price_usd: float | None = None


@dataclass(frozen=True)
class ComponentCatalog:
    combos: Tuple[MotorPropCombo, ...]
    packs: Tuple[BatteryPack, ...]
    frames: Tuple[Frame, ...]


TMOTOR_MN5008_URL = "https://store.tmotor.com/product/mn5008-kv340-motor-antigravity-type.html"
TMOTOR_P18_URL = "https://store.tmotor.com/product/polish-carbon-fiber-18x6_1-prop.html"


# T-Motor MN5008 KV340 + P18x6.1 CF, 22.2-23.44V/6S per-motor bench rows.
# Manufacturer lines used:
# - MN5008 page: KV340 motor weight 135g; P18x6.1 test curve; price 89.99.
# - P18x6.1 page: model 18x6.1; single-blade integrated weight 31.5g.
MN5008_KV340_18x61 = MotorPropCombo(
    name="T-Motor MN5008 KV340 + P18x6.1 (6S)",
    source_url=TMOTOR_MN5008_URL,
    retrieved="2026-07-03",
    voltage_v=22.2,
    cells=6,
    motor_mass_g=135.0,
    prop_mass_g=31.5,
    prop_diameter_in=18.0,
    price_usd=89.99 + 82.90 / 2.0,
    curve=(
        MotorPropPoint(0.40, 996.0, 3.76, 88.0),
        MotorPropPoint(0.45, 1238.0, 5.12, 120.0),
        MotorPropPoint(0.50, 1541.0, 6.93, 162.0),
        MotorPropPoint(0.55, 1822.0, 8.95, 208.0),
        MotorPropPoint(0.60, 2120.0, 11.12, 258.0),
        MotorPropPoint(0.65, 2394.0, 13.46, 311.0),
        MotorPropPoint(0.75, 2966.0, 18.83, 432.0),
        MotorPropPoint(1.00, 4215.0, 33.50, 754.0),
    ),
)


DEFAULT_CATALOG = ComponentCatalog(
    combos=(MN5008_KV340_18x61,),
    packs=(),
    frames=(),
)


CATALOG = {MN5008_KV340_18x61.name: MN5008_KV340_18x61}

