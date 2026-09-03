"""Meet-in-the-middle closure between DSE recommendation and real components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..dse.physics_estimator import DesignInputs, endurance_min
from ..sitl.dse_calibration import calibrate_ranking
from .catalog import ComponentCatalog, DEFAULT_CATALOG
from .closure_types import (
    RequirementVerdict,
    closure_scope,
    forward_flight_scope,
    requirement_verdicts,
)
from .forward_flight_check import forward_flight_verdicts
from .matcher import InterfaceCheck, RealizedCandidate, all_combinations, match
from .resizing import resize_on_real_packs


@dataclass(frozen=True)
class ClosureReport:
    verdict: str
    chosen: Optional[RealizedCandidate]
    per_requirement: Tuple[RequirementVerdict, ...]
    failed_checks: Tuple[InterfaceCheck, ...]
    rank_preservation: Dict[str, object]
    resize_note: str
    notes: Tuple[str, ...]
    forward_flight_ok: Optional[bool] = None


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
            forward_flight_ok=None,
        )
    chosen = ranked[0]
    ff = forward_flight_verdicts(chosen.rd, requirements)
    per_req = requirement_verdicts(design, chosen.metrics, requirements, ff)
    forward_ok = _forward_flight_ok(per_req)
    closure_req = _closure_scope_verdicts(per_req)
    if closure_req and all(v.met for v in closure_req):
        return ClosureReport("CLOSED", chosen, per_req, (), rank, "", tuple(notes), forward_ok)
    resized = resize_on_real_packs(chosen, requirements, catalog, cost_axis)
    if resized is not None:
        resized_ff = forward_flight_verdicts(resized.rd, requirements)
        resized_req = requirement_verdicts(
            _design_for_candidate(design, resized),
            resized.metrics,
            requirements,
            resized_ff,
        )
        resized_forward_ok = _forward_flight_ok(resized_req)
        note = f"battery resized from {chosen.rd.pack.name} to {resized.rd.pack.name}"
        return ClosureReport(
            "CLOSED_AFTER_RESIZE",
            resized,
            resized_req,
            (),
            rank,
            note,
            tuple(notes),
            resized_forward_ok,
        )
    # chosen came from match(), whose candidates pass every interface check, so
    # a resize-failed INFEASIBLE is attributed to the unmet closure requirements.
    failed = tuple(
        InterfaceCheck(v.req_id, False,
                       f"{v.family}: realized {v.realized_value:.3f} vs target {v.target:.3f}")
        for v in closure_req if not v.met
    )
    if not failed and not closure_req:
        failed = (InterfaceCheck(
            "closure_scope",
            False,
            "no endurance/mass requirement in datasheet-closure scope",
        ),)
    return ClosureReport(
        "INFEASIBLE_REALIZATION",
        chosen,
        per_req,
        failed,
        rank,
        "",
        tuple(notes),
        forward_ok,
    )


def _closure_scope_verdicts(per_req: Tuple[RequirementVerdict, ...]) -> Tuple[RequirementVerdict, ...]:
    return tuple(v for v in per_req if closure_scope(v))


def _forward_flight_ok(per_req: Tuple[RequirementVerdict, ...]) -> Optional[bool]:
    scoped = [v for v in per_req if forward_flight_scope(v)]
    return all(v.met for v in scoped) if scoped else None


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


def _rank_preservation(pareto_designs, requirements, catalog, cost_axis, notes) -> Dict[str, object]:
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
    # Persist the raw series either way: a bare correlation number cannot be
    # diagnosed afterwards, since catalog-snap ties look like a rank inversion
    # (the ambiguity the 2026-07 run hit).
    details = {
        "n": float(len(labels)),
        "labels": list(labels),
        "predicted": [float(p) for p in predicted],
        "measured": [float(m) for m in measured],
    }
    if len(labels) < 3:
        notes.append("rank preservation skipped: fewer than 3 realized Pareto candidates")
        return details
    if len(set(measured)) < len(measured):
        notes.append("rank preservation: measured endurances contain ties — multiple "
                     "Pareto designs snap to the same catalog realization, so the "
                     "correlation reflects catalog granularity, not estimator fidelity")
    cal = calibrate_ranking(labels, predicted, measured)
    return {
        **details,
        "spearman": cal.spearman,
        "kendall": cal.kendall,
        "top1_match": 1.0 if cal.top1_match else 0.0,
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
