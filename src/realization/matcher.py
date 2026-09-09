"""Match top-down DSE designs to nearest real component realizations."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, List, Mapping, Tuple

from ..dse.physics_estimator import DesignInputs
from .bottom_up import RealizedDesign, RealizedMetrics, realized_metrics, realize_from_design
from .catalog import (
    ComponentCatalog,
    DEFAULT_CATALOG,
    IntegrationBundle,
    NO_INTEGRATION_BUNDLE,
)

HOVER_THROTTLE_MAX = 0.65
TWR_MIN = 1.5
MAPPING_POLICY_VERSION = 1
ROTOR_RADIUS_MAX_RELATIVE_DRIFT = 0.10
BATTERY_CAPACITY_MAX_RELATIVE_DRIFT = 0.10
_DRIFT_EPSILON = 1e-12


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
    limit: float
    within_limit: bool


@dataclass(frozen=True)
class RealizedCandidate:
    rd: RealizedDesign
    metrics: RealizedMetrics
    distance: float
    checks: Tuple[InterfaceCheck, ...]
    design_drift: Tuple[DesignDrift, ...]
    source_design: DesignInputs


def mapping_policy() -> dict:
    """Machine-readable identity boundary for DSE -> catalog realization."""
    return {
        "schema_version": MAPPING_POLICY_VERSION,
        "architecture_invariants": {
            "rotor_count": "exact",
            "battery_cells": "exact",
        },
        "max_relative_drift": {
            "rotor_radius_m": ROTOR_RADIUS_MAX_RELATIVE_DRIFT,
            "battery_capacity_mah": BATTERY_CAPACITY_MAX_RELATIVE_DRIFT,
        },
        "reference": "all mapping and resize drift is measured from the original DSE recommendation",
    }


def catalog_design_domain(catalog: ComponentCatalog = DEFAULT_CATALOG) -> dict:
    """Return the evidence-backed discrete architecture domain in the catalog."""
    supported_cells = sorted(
        {combo.cells for combo in catalog.combos}
        & {pack.cells for pack in catalog.packs}
    )
    architectures = set()
    for frame in catalog.frames:
        for combo in catalog.combos:
            radius = per_rotor_radius_m(combo)
            if combo.cells not in supported_cells or combo.prop_diameter_in > frame.max_prop_in:
                continue
            architectures.add((frame.arms, radius, combo.cells))
    return {
        "rotor_counts": sorted({x[0] for x in architectures}),
        "rotor_radius_m": sorted({x[1] for x in architectures}),
        "battery_cells": supported_cells,
        "battery_capacity_mah_by_cells": {
            str(cells): sorted({p.capacity_mah for p in catalog.packs if p.cells == cells})
            for cells in supported_cells
        },
        "architectures": [
            {"rotor_count": n, "rotor_radius_m": r, "battery_cells": cells}
            for n, r, cells in sorted(architectures)
        ],
    }


def variant_design_is_catalog_admissible(
    design: Mapping[str, Any], catalog: ComponentCatalog = DEFAULT_CATALOG,
) -> bool:
    """Validate the catalog-controlled fields present in a partial variant design."""
    controlled = {k: design[k] for k in (
        "rotor_count", "rotor_radius_m", "battery_cells"
    ) if k in design}
    if not controlled:
        return True
    for arch in catalog_design_domain(catalog)["architectures"]:
        if "rotor_count" in controlled and int(controlled["rotor_count"]) != arch["rotor_count"]:
            continue
        if "battery_cells" in controlled and int(controlled["battery_cells"]) != arch["battery_cells"]:
            continue
        if "rotor_radius_m" in controlled:
            expected = float(controlled["rotor_radius_m"])
            if expected <= 0:
                continue
            relative = abs(arch["rotor_radius_m"] - expected) / expected
            if relative > ROTOR_RADIUS_MAX_RELATIVE_DRIFT + _DRIFT_EPSILON:
                continue
        return True
    return False


def catalog_capacity_options(
    design: DesignInputs, requirements: List[str],
    catalog: ComponentCatalog = DEFAULT_CATALOG,
) -> List[float]:
    """Real pack capacities for which the complete design has a valid mapping."""
    options = []
    for capacity in sorted({
        p.capacity_mah for p in catalog.packs if p.cells == design.battery_cells
    }):
        trial = replace(design, battery_capacity_mah=capacity)
        if match(trial, requirements, catalog):
            options.append(float(capacity))
    return options


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

    limits = {
        "rotor_radius_m": ROTOR_RADIUS_MAX_RELATIVE_DRIFT,
        "battery_capacity_mah": BATTERY_CAPACITY_MAX_RELATIVE_DRIFT,
        "battery_cells": 0.0,
    }

    def drift(name: str, expected: float, realized: float) -> DesignDrift:
        denom = abs(expected) if expected else 1.0
        relative = (realized - expected) / denom
        limit = limits[name]
        return DesignDrift(
            name=name,
            expected=expected,
            realized=realized,
            delta=realized - expected,
            relative_delta=relative,
            limit=limit,
            within_limit=abs(relative) <= limit + _DRIFT_EPSILON,
        )

    return (
        drift("rotor_radius_m", radius_expected, radius_realized),
        drift("battery_capacity_mah", cap_expected, cap_realized),
        drift("battery_cells", cells_expected, cells_realized),
    )


def evaluate_combination(design: DesignInputs, requirements: List[str],
                         combo, pack, frame, cost_axis: str = "mass",
                         integration_bundle: IntegrationBundle = NO_INTEGRATION_BUNDLE,
                         ) -> RealizedCandidate:
    checks = [
        InterfaceCheck(
            "design_cells_match",
            pack.cells == design.battery_cells,
            f"realization {pack.cells}S vs recommended design {design.battery_cells}S",
        ),
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
        InterfaceCheck(
            "integration_architecture",
            not integration_bundle.compatible_rotor_counts
            or frame.arms in integration_bundle.compatible_rotor_counts,
            f"integration bundle supports rotors "
            f"{integration_bundle.compatible_rotor_counts or 'any'} vs frame arms {frame.arms}",
        ),
    ]
    rd = realize_from_design(
        design, requirements, combo, pack, frame, integration_bundle
    )
    drifts = _design_drift(design, rd)
    by_name = {item.name: item for item in drifts}
    checks.extend([
        InterfaceCheck(
            "mapping_rotor_radius",
            by_name["rotor_radius_m"].within_limit,
            f"relative drift {by_name['rotor_radius_m'].relative_delta:+.3f} "
            f"within ±{ROTOR_RADIUS_MAX_RELATIVE_DRIFT:.2f}",
        ),
        InterfaceCheck(
            "mapping_battery_capacity",
            by_name["battery_capacity_mah"].within_limit,
            f"relative drift {by_name['battery_capacity_mah'].relative_delta:+.3f} "
            f"within ±{BATTERY_CAPACITY_MAX_RELATIVE_DRIFT:.2f}",
        ),
    ])
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
            pack_voltage_v=pack.operating_voltage_v(),
            voltage_ratio=pack.operating_voltage_v() / combo.voltage_v,
            derated_max_thrust_per_motor_g=0.0,
            notes=(str(exc),),
        )
        checks.extend([
            InterfaceCheck("can_hover", False, str(exc)),
            InterfaceCheck("hover_throttle", False, "not evaluated because can_hover failed"),
            InterfaceCheck("twr", False, "not evaluated because can_hover failed"),
        ])
    capacity_a = pack.capacity_mah / 1000.0
    # Convert the curve's maximum electrical operating point to pack current.
    # A lower-nominal-voltage pack cannot be credited with the curve-voltage
    # current figure unchanged.
    current_voltage_factor = max(1.0, combo.voltage_v / pack.operating_voltage_v())
    required_a = frame.arms * _max_current_per_motor(combo) * current_voltage_factor
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
        design_drift=drifts,
        source_design=design,
    )


def all_combinations(design: DesignInputs, requirements: List[str],
                     catalog: ComponentCatalog = DEFAULT_CATALOG,
                     cost_axis: str = "mass") -> List[RealizedCandidate]:
    out = []
    integration_bundles = catalog.integration_bundles or (NO_INTEGRATION_BUNDLE,)
    for combo in catalog.combos:
        for pack in catalog.packs:
            for frame in catalog.frames:
                for bundle in integration_bundles:
                    out.append(evaluate_combination(
                        design,
                        requirements,
                        combo,
                        pack,
                        frame,
                        cost_axis,
                        bundle,
                    ))
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
        c.rd.integration_bundle.name,
    ))
