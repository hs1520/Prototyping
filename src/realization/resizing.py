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
    # Preserve design identity across the whole mapping chain.  Using the first
    # snapped catalog candidate as the next reference would reset drift to zero
    # and permit arbitrarily large cumulative resize changes.
    source_design = candidate.source_design
    for pack in catalog.packs:
        trial = evaluate_combination(
            source_design,
            requirements,
            candidate.rd.combo,
            pack,
            candidate.rd.frame,
            cost_axis,
            candidate.rd.integration_bundle,
        )
        if not all(ch.passed for ch in trial.checks):
            continue
        if requirement_verdicts_met(source_design, trial.metrics, requirements):
            viable.append(trial)
    if not viable:
        return None
    return sorted(viable, key=lambda c: (
        c.metrics.total_mass_kg,
        c.rd.pack.capacity_mah,
        c.rd.pack.name,
    ))[0]
