"""Discrete battery re-sizing on real catalog packs."""
from __future__ import annotations

from typing import List, Optional

from .catalog import ComponentCatalog
from .closure_types import requirement_verdicts_met
from .matcher import RealizedCandidate, evaluate_combination


def resize_on_real_packs(candidate: RealizedCandidate, requirements: List[str],
                         catalog: ComponentCatalog,
                         cost_axis: str = "mass") -> Optional[RealizedCandidate]:
    viable = []
    design_like = _design_like(candidate)
    for pack in catalog.packs:
        trial = evaluate_combination(
            design_like,
            requirements,
            candidate.rd.combo,
            pack,
            candidate.rd.frame,
            cost_axis,
        )
        if not all(ch.passed for ch in trial.checks):
            continue
        if requirement_verdicts_met(design_like, trial.metrics, requirements):
            viable.append(trial)
    if not viable:
        return None
    return sorted(viable, key=lambda c: (
        c.metrics.total_mass_kg,
        c.rd.pack.capacity_mah,
        c.rd.pack.name,
    ))[0]


def _design_like(candidate: RealizedCandidate):
    from ..dse.physics_estimator import DesignInputs

    rd = candidate.rd
    return DesignInputs(
        payload_mass_kg=rd.delivery_payload_kg + rd.equipment_mass_kg,
        battery_capacity_mah=rd.pack.capacity_mah,
        battery_cells=rd.pack.cells,
        rotor_count=rd.rotor_count,
        rotor_radius_m=rd.combo.prop_diameter_in * 0.0254 / 2.0,
        cruise_speed_mps=(
            candidate.metrics.range_m / (candidate.metrics.endurance_min * 60.0)
            if candidate.metrics.endurance_min > 0 else 0.0
        ),
    )

