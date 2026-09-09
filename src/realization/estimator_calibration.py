"""F1: calibrate the lumped estimator against the manufacturer-datasheet layer.

The generic constants (FOM=0.62, ENERGY_DENSITY=150 Wh/kg,
BASE_FRAME_KG/ROTOR_MASS_COEF) came out biased against the datasheet tier
(~36% low on endurance, experiment B; catalog-snap ties made the n=3 Pareto
rank check uninformative). The catalog grid is used as the calibration set
instead of DSE designs: every feasible comboxpackxframe combination carries
both a datasheet truth (real masses + bench hover current) and an estimator
prediction for the same configuration, giving n in the tens, no snap
degeneracy, and a fit/validate split (fit on the grid, validate on the DSE
Pareto front via rank_preservation).

Fitted effective parameters (all physical, no black-box regression):
  * fom_eff             = median(P_ideal / P_datasheet_electrical) - absorbs
                          motor+ESC efficiency into one hover system efficiency
                          (applied as fom=fom_eff, eta_drive=1.0);
  * energy_density      = median over packs of capacity*cells*3.7V / mass;
  * base_frame_kg,
    rotor_mass_coef     = least-squares dry-mass fit (frame + N.(motor+prop))
                          vs total disk area over comboxframe pairs.

Scope: the orchestrator applies calibration only around the search
(run_variation_dse), so the injected SysML calc defs keep the documented
textbook constants and Phase 8's estimator_value column keeps the
uncalibrated lumped values, leaving the lumped-vs-datasheet contrast visible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

from ..dse.calibration import kendall_tau, spearman_rho
from ..dse.physics_estimator import (
    AVIONICS_POWER_W,
    CELL_V,
    DesignInputs,
    G,
    RHO,
    USABLE,
    calibrated,
    endurance_min,
)
from .catalog import ComponentCatalog, DEFAULT_CATALOG


@dataclass(frozen=True)
class GridPoint:
    """One feasible catalog configuration with datasheet truth + estimator inputs."""
    label: str
    design: DesignInputs
    real_mass_kg: float
    ds_endurance_min: float
    disk_area_m2: float
    # Current-based propulsion power at nominal pack voltage, the same arithmetic
    # the datasheet endurance uses (Ah/I), so the fom fit and the endurance figure
    # share one power definition (the bench power column can differ from V*I).
    p_prop_current_w: float


@dataclass(frozen=True)
class EstimatorCalibration:
    fom_eff: float
    energy_density_wh_kg: float
    base_frame_kg: float
    rotor_mass_coef: float
    n_points: int
    endurance_mape_before: float
    endurance_mape_after: float
    notes: Tuple[str, ...] = ()

    def overrides(self) -> Dict[str, float]:
        """Kwargs for physics_estimator.set_calibration/calibrated.

        fom_eff absorbs drive efficiency, so eta_drive is pinned to 1.0.
        """
        return {
            "fom": self.fom_eff,
            "eta_drive": 1.0,
            "energy_density_wh_kg": self.energy_density_wh_kg,
            "base_frame_kg": self.base_frame_kg,
            "rotor_mass_coef": self.rotor_mass_coef,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "fom_eff": self.fom_eff,
            "energy_density_wh_kg": self.energy_density_wh_kg,
            "base_frame_kg": self.base_frame_kg,
            "rotor_mass_coef": self.rotor_mass_coef,
            "n_points": self.n_points,
            "endurance_mape_before": self.endurance_mape_before,
            "endurance_mape_after": self.endurance_mape_after,
            "notes": list(self.notes),
        }


def calibration_grid(catalog: ComponentCatalog = DEFAULT_CATALOG,
                     include_clamped: bool = True) -> List[GridPoint]:
    """Every feasible comboxpackxframe configuration (payload-free).

    ``include_clamped=False`` keeps only configurations whose hover thrust lies
    inside the published bench curve: below the lowest bench row the current is
    clamped, so the datasheet value is pessimistic and would skew a fit.
    """
    points: List[GridPoint] = []
    for combo in catalog.combos:
        radius_m = combo.prop_diameter_in * 0.0254 / 2.0
        curve_min_g = min(pt.thrust_g for pt in combo.curve)
        for frame in catalog.frames:
            if combo.prop_diameter_in > frame.max_prop_in:
                continue
            dry_kg = (frame.mass_g
                      + frame.arms * (combo.motor_mass_g + combo.prop_mass_g)) / 1000.0
            for pack in catalog.packs:
                if pack.cells != combo.cells:
                    continue
                mass_kg = dry_kg + pack.mass_g / 1000.0
                per_motor_g = mass_kg * 1000.0 / frame.arms
                if not include_clamped and per_motor_g < curve_min_g:
                    continue
                try:
                    current_a, power_w = combo.interp_at_thrust(per_motor_g)
                except ValueError:
                    continue
                del power_w  # bench power can differ from V*I; endurance uses I
                area = frame.arms * math.pi * radius_m ** 2
                # Known approximation, unchanged since 2026-08-26: CELL_V rates Li-ion packs
                # at LiPo 3.7 V/cell. The fitted calibration values cited by the dissertation
                # (0.829 / 15.2%) were produced with this constant, so correcting it here
                # would diverge from the archived claims. Use pack.operating_voltage_v() on a
                # refit; forward_flight_check already does.
                v_nom = pack.cells * CELL_V
                total_a = current_a * frame.arms + AVIONICS_POWER_W / v_nom
                ds_endurance = (pack.capacity_mah / 1000.0 * USABLE) / total_a * 60.0
                points.append(GridPoint(
                    label=f"{combo.name} | {pack.name} | {frame.name}",
                    design=DesignInputs(0.0, pack.capacity_mah, pack.cells,
                                        frame.arms, radius_m, 0.0),
                    real_mass_kg=mass_kg,
                    ds_endurance_min=ds_endurance,
                    disk_area_m2=area,
                    p_prop_current_w=v_nom * current_a * frame.arms,
                ))
    return points


def _dry_mass_fit(catalog: ComponentCatalog) -> Tuple[float, float, str]:
    """Bounded least-squares (base_kg, coef) for dry mass = base + coef . disk_area.

    An unconstrained line through this small grid can go unphysical (negative
    intercept, since heavy frames pair with big props), so base is grid-searched
    over a physical band and coef is the conditional non-negative slope.
    """
    xs, ys = [], []
    for combo in catalog.combos:
        radius_m = combo.prop_diameter_in * 0.0254 / 2.0
        for frame in catalog.frames:
            if combo.prop_diameter_in > frame.max_prop_in:
                continue
            xs.append(frame.arms * math.pi * radius_m ** 2)
            ys.append((frame.mass_g
                       + frame.arms * (combo.motor_mass_g + combo.prop_mass_g)) / 1000.0)
    n = len(xs)
    if n < 2 or max(xs) == min(xs):
        return 0.0, 0.0, f"dry-mass fit degenerate (n={n}); keeping default constants"
    sxx = sum(x * x for x in xs)
    best: Tuple[float, float, float] | None = None
    base_hi = min(ys)  # base cannot exceed the lightest dry build
    steps = 29
    for i in range(steps):
        base = 0.1 + (base_hi - 0.1) * i / (steps - 1)
        coef = max(0.0, sum((y - base) * x for x, y in zip(xs, ys)) / sxx)
        sse = sum((base + coef * x - y) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, base, coef)
    _, base, coef = best
    if coef <= 0:
        return 0.0, 0.0, f"dry-mass fit unphysical (coef={coef:.3f}); keeping default constants"
    return base, coef, ""


def _mape(points: List[GridPoint]) -> float:
    errs = [abs(endurance_min(p.design) - p.ds_endurance_min) / p.ds_endurance_min
            for p in points if p.ds_endurance_min > 0]
    return sum(errs) / len(errs) if errs else float("nan")


def fit_from_catalog(catalog: ComponentCatalog = DEFAULT_CATALOG) -> EstimatorCalibration:
    """Fit effective estimator constants on the catalog grid (deterministic)."""
    from ..dse.physics_estimator import (
        BASE_FRAME_KG, ENERGY_DENSITY_WH_KG, ETA_DRIVE, FOM, ROTOR_MASS_COEF,
    )

    points = calibration_grid(catalog, include_clamped=False)
    n_all = len(calibration_grid(catalog, include_clamped=True))
    if len(points) < 3:
        raise ValueError(f"calibration needs ≥3 feasible catalog configurations, got {len(points)}")
    notes: List[str] = []
    if n_all > len(points):
        notes.append(f"excluded {n_all - len(points)} clamped configurations "
                     "(hover below the lowest published bench row — no manufacturer data)")

    density = median(
        (pack.capacity_mah / 1000.0) * pack.cells * CELL_V / (pack.mass_g / 1000.0)
        for pack in catalog.packs
    )
    base, coef, mass_note = _dry_mass_fit(catalog)
    if mass_note:
        notes.append(mass_note)
        base, coef = BASE_FRAME_KG, ROTOR_MASS_COEF

    # Stage 2: fom_eff is fitted so the estimator-side power (ideal power at the
    # calibrated mass model) matches the current-based datasheet propulsion power.
    # Fitting at the estimator's own mass rather than the real mass lets fom_eff
    # absorb the residual mass-model error instead of leaking it into the
    # endurance prediction.
    def _m_est(pt: GridPoint) -> float:
        battery = (pt.design.battery_capacity_mah / 1000.0
                   * pt.design.battery_cells * CELL_V) / density
        return base + coef * pt.disk_area_m2 + battery

    fom_eff = median(
        ((_m_est(pt) * G) ** 1.5 / math.sqrt(2.0 * RHO * pt.disk_area_m2))
        / pt.p_prop_current_w
        for pt in points
    )

    defaults = {"fom": FOM, "eta_drive": ETA_DRIVE,
                "energy_density_wh_kg": ENERGY_DENSITY_WH_KG,
                "base_frame_kg": BASE_FRAME_KG, "rotor_mass_coef": ROTOR_MASS_COEF}
    with calibrated(**defaults):
        mape_before = _mape(points)
    cal = EstimatorCalibration(
        fom_eff=fom_eff,
        energy_density_wh_kg=density,
        base_frame_kg=base,
        rotor_mass_coef=coef,
        n_points=len(points),
        endurance_mape_before=mape_before,
        endurance_mape_after=float("nan"),
        notes=tuple(notes),
    )
    with calibrated(**cal.overrides()):
        mape_after = _mape(points)
    return EstimatorCalibration(
        fom_eff=cal.fom_eff,
        energy_density_wh_kg=cal.energy_density_wh_kg,
        base_frame_kg=cal.base_frame_kg,
        rotor_mass_coef=cal.rotor_mass_coef,
        n_points=cal.n_points,
        endurance_mape_before=mape_before,
        endurance_mape_after=mape_after,
        notes=cal.notes,
    )


def catalog_rank_check(catalog: ComponentCatalog = DEFAULT_CATALOG,
                       fit: Optional[EstimatorCalibration] = None) -> Dict[str, float]:
    """Estimator-vs-datasheet endurance rank correlation on the catalog grid.

    L↔M rank evidence with usable n and no catalog-snap ties (each point is a
    distinct configuration); Pareto-front rank_preservation remains the
    out-of-sample check.
    """
    points = calibration_grid(catalog, include_clamped=False)
    measured = [p.ds_endurance_min for p in points]
    out: Dict[str, float] = {"n": float(len(points))}
    if len(points) < 3:
        return out
    from ..dse.physics_estimator import (
        BASE_FRAME_KG, ENERGY_DENSITY_WH_KG, ETA_DRIVE, FOM, ROTOR_MASS_COEF,
    )
    with calibrated(fom=FOM, eta_drive=ETA_DRIVE,
                    energy_density_wh_kg=ENERGY_DENSITY_WH_KG,
                    base_frame_kg=BASE_FRAME_KG, rotor_mass_coef=ROTOR_MASS_COEF):
        before = [endurance_min(p.design) for p in points]
    out["spearman_before"] = spearman_rho(before, measured)
    out["kendall_before"] = kendall_tau(before, measured)
    if fit is not None:
        with calibrated(**fit.overrides()):
            after = [endurance_min(p.design) for p in points]
        out["spearman_after"] = spearman_rho(after, measured)
        out["kendall_after"] = kendall_tau(after, measured)
    return out
