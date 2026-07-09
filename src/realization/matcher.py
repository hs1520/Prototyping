"""Match top-down DSE designs to nearest real component realizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..dse.physics_estimator import DesignInputs
from .bottom_up import RealizedDesign, RealizedMetrics, realized_metrics, realize_from_design
from .catalog import ComponentCatalog, DEFAULT_CATALOG

HOVER_THROTTLE_MAX = 0.65
TWR_MIN = 1.5


@dataclass(frozen=True)
class InterfaceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DesignDrift:
    name: str
    expected: float
    realized: float
    delta: float
    relative_delta: float


@dataclass(frozen=True)
class RealizedCandidate:
    rd: RealizedDesign
    metrics: RealizedMetrics
    distance: float
    checks: Tuple[InterfaceCheck, ...]
    design_drift: Tuple[DesignDrift, ...]


def per_rotor_radius_m(combo) -> float:
    return combo.prop_diameter_in * 0.0254 / 2.0


def _max_current_per_motor(combo) -> float:
    return max(p.current_a for p in combo.curve)


def _distance(design: DesignInputs, rd: RealizedDesign) -> float:
    radius_den = design.rotor_radius_m or 1.0
    cap_den = design.battery_capacity_mah or 1.0
    cells_den = max(design.battery_cells, 1)
    return (
        abs(per_rotor_radius_m(rd.combo) - design.rotor_radius_m) / radius_den
        + abs(rd.pack.capacity_mah - design.battery_capacity_mah) / cap_den
        + abs(rd.pack.cells - design.battery_cells) / cells_den
    )


def _design_drift(design: DesignInputs, rd: RealizedDesign) -> Tuple[DesignDrift, ...]:
    radius_expected = design.rotor_radius_m
    radius_realized = per_rotor_radius_m(rd.combo)
    cap_expected = design.battery_capacity_mah
    cap_realized = rd.pack.capacity_mah
    cells_expected = float(design.battery_cells)
    cells_realized = float(rd.pack.cells)

    def drift(name: str, expected: float, realized: float) -> DesignDrift:
        denom = abs(expected) if expected else 1.0
        return DesignDrift(
            name=name,
            expected=expected,
            realized=realized,
            delta=realized - expected,
            relative_delta=(realized - expected) / denom,
        )

    return (
        drift("rotor_radius_m", radius_expected, radius_realized),
        drift("battery_capacity_mah", cap_expected, cap_realized),
        drift("battery_cells", cells_expected, cells_realized),
    )


def evaluate_combination(design: DesignInputs, requirements: List[str],
                         combo, pack, frame, cost_axis: str = "mass") -> RealizedCandidate:
    checks = [
        InterfaceCheck(
            "cells_match",
            pack.cells == combo.cells,
            f"pack {pack.cells}S vs combo {combo.cells}S",
        ),
        InterfaceCheck(
            "arms_match",
            frame.arms == design.rotor_count,
            f"frame arms {frame.arms} vs design rotors {design.rotor_count}",
        ),
        InterfaceCheck(
            "prop_fits",
            combo.prop_diameter_in <= frame.max_prop_in,
            f"prop {combo.prop_diameter_in:g}in vs frame max {frame.max_prop_in:g}in",
        ),
    ]
    rd = realize_from_design(design, requirements, combo, pack, frame)
    try:
        metrics = realized_metrics(rd, design.cruise_speed_mps, cost_axis=cost_axis)
        checks.extend([
            InterfaceCheck(
                "can_hover",
                True,
                f"hover thrust {metrics.hover_thrust_per_motor_g:.1f}g within curve",
            ),
            InterfaceCheck(
                "hover_throttle",
                metrics.hover_throttle <= HOVER_THROTTLE_MAX,
                f"hover throttle {metrics.hover_throttle:.3f} <= {HOVER_THROTTLE_MAX}",
            ),
            InterfaceCheck(
                "twr",
                metrics.twr_max >= TWR_MIN,
                f"TWR {metrics.twr_max:.3f} >= {TWR_MIN}",
            ),
        ])
    except ValueError as exc:
        metrics = RealizedMetrics(
            total_mass_kg=0.0,
            hover_thrust_per_motor_g=0.0,
            hover_current_per_motor_a=0.0,
            hover_throttle=1.0,
            total_hover_current_a=0.0,
            twr_max=0.0,
            endurance_min=0.0,
            range_m=0.0,
            cost=0.0,
            notes=(str(exc),),
        )
        checks.extend([
            InterfaceCheck("can_hover", False, str(exc)),
            InterfaceCheck("hover_throttle", False, "not evaluated because can_hover failed"),
            InterfaceCheck("twr", False, "not evaluated because can_hover failed"),
        ])
    capacity_a = pack.capacity_mah / 1000.0
    required_a = frame.arms * _max_current_per_motor(combo)
    checks.append(InterfaceCheck(
        "c_rating",
        capacity_a * pack.c_rating >= required_a,
        f"{capacity_a * pack.c_rating:.1f}A available vs {required_a:.1f}A required",
    ))
    return RealizedCandidate(
        rd=rd,
        metrics=metrics,
        distance=_distance(design, rd),
        checks=tuple(checks),
        design_drift=_design_drift(design, rd),
    )


def all_combinations(design: DesignInputs, requirements: List[str],
                     catalog: ComponentCatalog = DEFAULT_CATALOG,
                     cost_axis: str = "mass") -> List[RealizedCandidate]:
    out = []
    for combo in catalog.combos:
        for pack in catalog.packs:
            for frame in catalog.frames:
                out.append(evaluate_combination(design, requirements, combo, pack, frame, cost_axis))
    return out


def match(design: DesignInputs, requirements: List[str],
          catalog: ComponentCatalog = DEFAULT_CATALOG,
          cost_axis: str = "mass") -> List[RealizedCandidate]:
    candidates = [c for c in all_combinations(design, requirements, catalog, cost_axis)
                  if all(ch.passed for ch in c.checks)]
    return sorted(candidates, key=lambda c: (
        c.distance,
        c.metrics.cost,
        c.rd.combo.name,
        c.rd.pack.name,
        c.rd.frame.name,
    ))
