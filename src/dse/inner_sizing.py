"""Inner-layer continuous sizing for variation DSE (the BO half of the bilevel).

For a fixed outer architecture (frame / rotor / payload), Bayesian-optimize
battery capacity to the cheapest size that still meets the endurance requirement.
Endurance satisfaction saturates at the target while all-up mass grows with
capacity (battery self-weight), so `satisfaction − mass_weight*mass` has an
interior optimum - the smallest battery that meets endurance. This is the inner
axis SITL calibrates (endurance-vs-capacity, Spearman=1.0).
"""
from __future__ import annotations

from typing import Dict, Iterable

from .inner_bo import BayesianOptimizer
from .physics_estimator import DesignInputs, endurance_min, total_mass_kg

# The realization catalog contains 24Ah and 30Ah 4S UAV packs; a 22Ah cap on the
# continuous fallback would put that domain out of reach whenever discrete catalog
# sizing is unavailable.
CAPACITY_BOUNDS = (3000.0, 30000.0)


def optimize_capacity(
    arch: Dict[str, float],
    target_endurance_min: float,
    bounds=CAPACITY_BOUNDS,
    mass_weight: float = 0.05,
    n_init: int = 4,
    n_iter: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    """Inner BO over battery capacity for a fixed architecture."""
    def objective(cap: float) -> float:
        di = DesignInputs(battery_capacity_mah=cap, **arch)
        e = endurance_min(di)
        sat = min(1.0, e / target_endurance_min) if target_endurance_min > 0 else 1.0
        return sat - mass_weight * total_mass_kg(di)

    res = BayesianOptimizer(
        bounds=bounds, objective=objective, n_init=n_init, n_iter=n_iter, random_seed=seed
    ).optimize()
    cap = res.best_x
    di = DesignInputs(battery_capacity_mah=cap, **arch)
    return {
        "capacity_mah": round(cap, 1),
        "endurance_min": round(endurance_min(di), 2),
        "total_mass_kg": round(total_mass_kg(di), 3),
        "bo_evals": res.n_evals,
    }


def optimize_discrete_capacity(
    arch: Dict[str, float], target_endurance_min: float,
    capacities_mah: Iterable[float], mass_weight: float = 0.05,
) -> Dict[str, float]:
    """Select a real catalog capacity using the same objective as continuous BO."""
    candidates = sorted({float(x) for x in capacities_mah if float(x) > 0})
    if not candidates:
        raise ValueError("at least one catalog capacity is required")

    def score(cap: float) -> tuple[float, float]:
        di = DesignInputs(battery_capacity_mah=cap, **arch)
        endurance = endurance_min(di)
        sat = min(1.0, endurance / target_endurance_min) if target_endurance_min > 0 else 1.0
        return sat - mass_weight * total_mass_kg(di), -cap

    cap = max(candidates, key=score)
    di = DesignInputs(battery_capacity_mah=cap, **arch)
    return {
        "capacity_mah": round(cap, 1),
        "endurance_min": round(endurance_min(di), 2),
        "total_mass_kg": round(total_mass_kg(di), 3),
        "bo_evals": len(candidates),
        "sizing_mode": "catalog_discrete",
    }
