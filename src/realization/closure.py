"""Meet-in-the-middle closure between DSE recommendation and real components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..dse.physics_estimator import DesignInputs, endurance_min
from ..sitl.dse_calibration import calibrate_ranking
from .catalog import ComponentCatalog, DEFAULT_CATALOG
from .closure_types import RequirementVerdict, requirement_verdicts
from .matcher import InterfaceCheck, RealizedCandidate, all_combinations, match
from .resizing import resize_on_real_packs


@dataclass(frozen=True)
class ClosureReport:
    verdict: str
    chosen: Optional[RealizedCandidate]
    per_requirement: Tuple[RequirementVerdict, ...]
    failed_checks: Tuple[InterfaceCheck, ...]
    rank_preservation: Dict[str, float]
    resize_note: str
    notes: Tuple[str, ...]


def close_the_loop(design: DesignInputs,
                   pareto_designs: List[Tuple[DesignInputs, dict]],
                   requirements: List[str],
                   catalog: ComponentCatalog = DEFAULT_CATALOG,
                   cost_axis: str = "mass") -> ClosureReport:
    notes: List[str] = []
    ranked = match(design, requirements, catalog, cost_axis)
    rank = _rank_preservation(pareto_designs, requirements, catalog, cost_axis, notes)
    if not ranked:
        failed = _nearest_failed_checks(design, requirements, catalog, cost_axis)
        return ClosureReport(
            verdict="INFEASIBLE_REALIZATION",
            chosen=None,
            per_requirement=(),
            failed_checks=failed,
            rank_preservation=rank,
            resize_note="",
            notes=tuple(notes),
        )
    chosen = ranked[0]
    per_req = requirement_verdicts(design, chosen.metrics, requirements)
    if per_req and all(v.met for v in per_req):
        return ClosureReport("CLOSED", chosen, per_req, (), rank, "", tuple(notes))
    resized = resize_on_real_packs(chosen, requirements, catalog, cost_axis)
    if resized is not None:
        resized_req = requirement_verdicts(_design_for_candidate(design, resized), resized.metrics, requirements)
        note = f"battery resized from {chosen.rd.pack.name} to {resized.rd.pack.name}"
        return ClosureReport("CLOSED_AFTER_RESIZE", resized, resized_req, (), rank, note, tuple(notes))
    failed = tuple(ch for ch in chosen.checks if not ch.passed)
    if not failed:
        failed = tuple(
            InterfaceCheck(v.req_id, False,
                           f"{v.family}: realized {v.realized_value:.3f} vs target {v.target:.3f}")
            for v in per_req if not v.met
        )
    return ClosureReport(
        "INFEASIBLE_REALIZATION",
        chosen,
        per_req,
        failed,
        rank,
        "",
        tuple(notes),
    )


def _nearest_failed_checks(design, requirements, catalog, cost_axis) -> Tuple[InterfaceCheck, ...]:
    combos = all_combinations(design, requirements, catalog, cost_axis)
    if not combos:
        missing = []
        if not catalog.combos:
            missing.append(InterfaceCheck("catalog_combos", False, "no motor-prop combos in catalog"))
        if not catalog.packs:
            missing.append(InterfaceCheck("catalog_packs", False, "no battery packs in catalog"))
        if not catalog.frames:
            missing.append(InterfaceCheck("catalog_frames", False, "no frames in catalog"))
        return tuple(missing)
    best = sorted(combos, key=lambda c: (sum(not ch.passed for ch in c.checks), c.distance))[0]
    return tuple(ch for ch in best.checks if not ch.passed)


def _rank_preservation(pareto_designs, requirements, catalog, cost_axis, notes) -> Dict[str, float]:
    labels, predicted, measured = [], [], []
    for i, item in enumerate(pareto_designs or []):
        di = item[0]
        ms = match(di, requirements, catalog, cost_axis)
        if not ms:
            notes.append(f"rank preservation skipped pareto[{i}] with no feasible realization")
            continue
        labels.append(f"pareto[{i}]")
        predicted.append(endurance_min(di))
        measured.append(ms[0].metrics.endurance_min)
    if len(labels) < 3:
        notes.append("rank preservation skipped: fewer than 3 realized Pareto candidates")
        return {"n": float(len(labels))}
    cal = calibrate_ranking(labels, predicted, measured)
    return {
        "spearman": cal.spearman,
        "kendall": cal.kendall,
        "top1_match": 1.0 if cal.top1_match else 0.0,
        "n": float(len(labels)),
    }


def _design_for_candidate(original: DesignInputs, candidate: RealizedCandidate) -> DesignInputs:
    return DesignInputs(
        payload_mass_kg=original.payload_mass_kg,
        battery_capacity_mah=candidate.rd.pack.capacity_mah,
        battery_cells=candidate.rd.pack.cells,
        rotor_count=candidate.rd.rotor_count,
        rotor_radius_m=original.rotor_radius_m,
        cruise_speed_mps=original.cruise_speed_mps,
    )

