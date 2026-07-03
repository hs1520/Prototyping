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
    source_url: str          # motor product page (bench curve + motor mass)
    retrieved: str
    voltage_v: float
    cells: int
    motor_mass_g: float
    prop_mass_g: float
    prop_diameter_in: float
    curve: Tuple[MotorPropPoint, ...]
    price_usd: float | None = None
    prop_source_url: str = ""  # prop product page (prop mass/diameter), when distinct

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
    prop_source_url=TMOTOR_P18_URL,
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


TMOTOR_MN4006_URL = "https://store.tmotor.com/product/mn4006-kv380-motor-antigravity-type.html"
TMOTOR_P16_URL = "https://store.tmotor.com/product/polish-carbon-fiber-16x5_4-prop.html"

# T-Motor Antigravity MN4006 KV380 + P16x5.4 CF, bench table published at 24V (6S).
# Manufacturer lines used (retrieved 2026-07-03):
# - MN4006 page: bench rows below (24V, 16x5.4 CF prop); motor weight 68g incl cable.
# - P16x5.4 page: single-blade integrated propeller weight 25±1.5g.
# price_usd=None: the store listing price ($149.90) is ambiguous between single motor
# and 2PCS/SET across T-Motor pages — omitted rather than guessed (§7 rule).
MN4006_KV380_16x54 = MotorPropCombo(
    name="T-Motor MN4006 KV380 + P16x5.4 (6S)",
    source_url=TMOTOR_MN4006_URL,
    prop_source_url=TMOTOR_P16_URL,
    retrieved="2026-07-03",
    voltage_v=24.0,
    cells=6,
    motor_mass_g=68.0,
    prop_mass_g=25.0,
    prop_diameter_in=16.0,
    price_usd=None,
    curve=(
        MotorPropPoint(0.50, 928.0, 3.70, 89.0),
        MotorPropPoint(0.55, 1096.0, 4.80, 115.0),
        MotorPropPoint(0.60, 1258.0, 5.90, 142.0),
        MotorPropPoint(0.65, 1427.0, 7.20, 173.0),
        MotorPropPoint(0.75, 1740.0, 10.00, 240.0),
        MotorPropPoint(0.85, 1970.0, 12.90, 310.0),
        MotorPropPoint(1.00, 2309.0, 17.50, 420.0),
    ),
)


TMOTOR_MN3508_URL = "https://store.tmotor.com/product/mn3508-motor-navigator-type.html"
TMOTOR_P15_URL = "https://store.tmotor.com/product/polish-carbon-fiber-15x5-prop.html"

# T-Motor Navigator MN3508 KV380 + P15x5 CF. Manufacturer lines used
# (retrieved 2026-07-03, verified against store.tmotor.com):
# - MN3508 page: KV380 motor weight 103g incl cable; the 15x5CF bench table is
#   published ONLY at 14.8V (4S). The page's 22.2V tables use 12x4 / 13x4.4 props,
#   NOT 15x5 — so there is NO 6S/15x5 combo here (a 6S variant would require
#   fabricating a bench curve, forbidden by §7). Hence 4S-only.
# - P15x5 page: model 15x5; single-blade integrated propeller weight 21±1.5g.
# price_usd = single motor listing ($69.90) + half of 2PCS/PAIR P15x5 listing ($55.90/2).
MN3508_KV380_15x5_4S = MotorPropCombo(
    name="T-Motor MN3508 KV380 + P15x5 (4S)",
    source_url=TMOTOR_MN3508_URL,
    prop_source_url=TMOTOR_P15_URL,
    retrieved="2026-07-03",
    voltage_v=14.8,
    cells=4,
    motor_mass_g=103.0,
    prop_mass_g=21.0,
    prop_diameter_in=15.0,
    price_usd=69.90 + 55.90 / 2.0,
    curve=(
        MotorPropPoint(0.50, 430.0, 1.6, 23.68),
        MotorPropPoint(0.65, 670.0, 3.4, 50.32),
        MotorPropPoint(0.75, 820.0, 5.0, 74.0),
        MotorPropPoint(0.85, 1000.0, 6.4, 94.72),
        MotorPropPoint(1.00, 1100.0, 7.5, 111.0),
    ),
)


# ── Battery packs — all 6S (cells_match with the 6S combos), Gens Ace/Tattu official
# product pages (genstattu.com), net weights as published, retrieved 2026-07-03.
# price_usd=None: page prices not captured at collection time (mass axis is default).
TATTU_PACKS = (
    BatteryPack(
        name="Tattu G-Tech 8000mAh 6S 25C",
        source_url="https://genstattu.com/tattu-8000mah-22-2v-25c-6s1p-lipo-battery-pack-with-xt60-plug.html",
        retrieved="2026-07-03",
        capacity_mah=8000.0, cells=6, mass_g=1160.0, c_rating=25.0,
    ),
    BatteryPack(
        name="Tattu Plus 10000mAh 6S 25C",
        source_url="https://genstattu.com/tattu-plus-22-2v-25c-6s-liPo-battery-10000-mah-with-as150-xt150-plug.html",
        retrieved="2026-07-03",
        capacity_mah=10000.0, cells=6, mass_g=1517.0, c_rating=25.0,
    ),
    BatteryPack(
        name="Tattu Plus 12000mAh 6S 15C",
        source_url="https://genstattu.com/tattu-plus-15c-12000mah-6s1p-as150-xt150-plug-lipo-battery.html",
        retrieved="2026-07-03",
        capacity_mah=12000.0, cells=6, mass_g=1670.0, c_rating=15.0,
    ),
    BatteryPack(
        name="Tattu Plus 16000mAh 6S 15C",
        source_url="https://genstattu.com/tattu-plus-16000mah-6s-15c-22-2v-lipo-battery-pack-with-xt90s/",
        retrieved="2026-07-03",
        capacity_mah=16000.0, cells=6, mass_g=1932.0, c_rating=15.0,
    ),
    BatteryPack(
        name="Tattu Plus 22000mAh 6S 25C",
        source_url="https://genstattu.com/tattu-plus-25c-22000mah-6s1p-xt90-smart-lipo-battery.html",
        retrieved="2026-07-03",
        capacity_mah=22000.0, cells=6, mass_g=2650.0, c_rating=25.0,
    ),
)

TATTU_4S_PACKS = (
    BatteryPack(
        name="Tattu R-Line V5 1300mAh 4S 150C",
        source_url=(
            "https://genstattu.com/"
            "tattu-r-line-version-5-0-1300mah-4s-14-8v-150c-lipo-battery-pack-with-xt60-plug/"
        ),
        retrieved="2026-07-03",
        capacity_mah=1300.0,
        cells=4,
        mass_g=156.0,
        c_rating=150.0,
        price_usd=29.99,
    ),
    BatteryPack(
        name="Tattu R-Line V5 850mAh 4S 150C",
        source_url=(
            "https://genstattu.com/"
            "tattu-r-line-version-5-0-850mah-4s-150c-14-8v-lipo-battery-pack-with-xt30u-f-plug/"
        ),
        retrieved="2026-07-03",
        capacity_mah=850.0,
        cells=4,
        mass_g=100.0,
        c_rating=150.0,
        price_usd=22.99,
    ),
    BatteryPack(
        name="Tattu 650mAh 4S 95C LiHV",
        source_url="https://genstattu.com/tattu-650mah-4s-15-2v-95c-lipo-battery-long-pack-with-xt30-plug/",
        retrieved="2026-07-03",
        capacity_mah=650.0,
        cells=4,
        mass_g=60.0,
        c_rating=95.0,
        price_usd=16.49,
    ),
)


# ── Frames.
# PROVENANCE NOTE (§7 flagged): the Tarot official site is not reliably reachable, so
# both frame entries cite the largest authorized distributor pages that reproduce the
# manufacturer spec sheet (net weight / wheelbase / prop range). This is a deliberate,
# documented relaxation of the manufacturer-first rule — review before publication.
# Holybro S500/X650 were REJECTED: only ARF/kit-with-motors weights are published,
# never the bare-frame mass this catalog's mass model requires.
TAROT_FRAMES = (
    Frame(
        # Tarot X6 TL6X001 umbrella-folding hexa: wheelbase 960mm, 18in props,
        # net weight 2.0kg (incl. electronic retractable landing gear), MTOW 12kg.
        name="Tarot X6 TL6X001 (hexa 960mm)",
        source_url="https://www.foxtechfpv.com/tarot-x6-hexacopter-frame-p-1945.html",
        retrieved="2026-07-03",
        mass_g=2000.0, arms=6, max_prop_in=18.0,
    ),
    Frame(
        # Tarot 650 Sport TL65S01 foldable quad: wheelbase 600mm, 12-15in props,
        # net weight 750g (incl. electric retractable landing skid).
        name="Tarot 650 Sport TL65S01 (quad 600mm)",
        source_url="https://www.arrishobby.com/products/tarot-650-sport-quadcopter-tl65s01-with-electric-retractable-landing-skid",
        retrieved="2026-07-03",
        mass_g=750.0, arms=4, max_prop_in=15.0,
    ),
)


DEFAULT_CATALOG = ComponentCatalog(
    combos=(MN5008_KV340_18x61, MN4006_KV380_16x54, MN3508_KV380_15x5_4S),
    packs=TATTU_PACKS + TATTU_4S_PACKS,
    frames=TAROT_FRAMES,
)


CATALOG = {c.name: c for c in DEFAULT_CATALOG.combos}
