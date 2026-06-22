"""Inner-layer continuous sizing for variation DSE (the BO half of the bilevel).

Given a FIXED outer architecture (frame / rotor / payload, chosen by the outer
MO-MCTS), Bayesian-optimize the continuous BATTERY CAPACITY to the most economical
size that still meets the endurance requirement.

Non-degenerate by construction: endurance satisfaction SATURATES at the target
(capped), while all-up mass GROWS with capacity (emergent battery self-weight). So
the inner objective `satisfaction − mass_weight·mass` has an INTERNAL optimum — the
smallest battery that meets endurance — not a boundary one. This is the inner axis
that SITL actually calibrates (endurance-vs-capacity, Spearman=1.0).
"""
from __future__ import annotations

from typing import Dict

from .inner_bo import BayesianOptimizer
from .physics_estimator import DesignInputs, endurance_min, total_mass_kg

CAPACITY_BOUNDS = (3000.0, 22000.0)  # mAh, plausible multirotor pack range


def optimize_capacity(
    arch: Dict[str, float],
    target_endurance_min: float,
    bounds=CAPACITY_BOUNDS,
    mass_weight: float = 0.05,
    n_init: int = 4,
    n_iter: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    """Inner BO over battery capacity for a fixed architecture.

    arch: the non-capacity design inputs (payload_mass_kg, battery_cells, rotor_count,
    rotor_radius_m, cruise_speed_mps). Returns the chosen capacity + its emergent metrics.
    """
    def objective(cap: float) -> float:
        di = DesignInputs(battery_capacity_mah=cap, **arch)
        e = endurance_min(di)
        sat = min(1.0, e / target_endurance_min) if target_endurance_min > 0 else 1.0
        # maximize: meet endurance, then prefer the lightest (cheapest) pack
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
