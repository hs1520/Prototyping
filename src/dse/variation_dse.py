"""DSE over LLM-declared SysML v2 variation points (the "replace" path).

The explored space is the set of ``variation`` points the LLM declared in the
model (admitted only if they carry a rationale + linked requirement). Each point
becomes a dynamic operator; the outer MO-MCTS explores combinations; each
candidate is *resolved* into a concrete model and scored by the REAL
DesignEvaluator (full models, so the real evaluator is appropriate here).

Returns a Pareto front over (design_quality, simplicity) + a recommended concrete
model. Falls back to None when the model declares no admissible variation points.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import find_block_end, get_sysml_text
from .domain_objective import (
    architecture_design,
    architecture_objectives,
    design_arch_inputs,
    endurance_target,
    max_rated_payload,
    normalize_variation_ownership,
    strip_inner_loop_attrs,
    objective_names,
    objectives_from_design,
    requirement_targets,
)
from .inner_sizing import optimize_capacity
from .physics_estimator import DesignInputs
from .weighting import recommend as weighted_recommend
from .mo_mcts import MultiObjectiveMCTS, Objectives, State
from .quality_eval import evaluate_design_quality
from .variation_parser import (
    VariationPoint,
    admitted,
    parse_variation_points,
    port_safe_split,
    resolve_model,
)

_QUALITY_DIMS = (
    "requirement_coverage", "structural_completeness", "safety_assurance", "interface_quality",
)


@dataclass
class VariationOperator:
    """Dynamic MO-MCTS operator wrapping one LLM-declared variation point."""
    vp: VariationPoint

    @property
    def point_id(self) -> str:
        return self.vp.point_id

    @property
    def variants(self) -> List[str]:
        return self.vp.variant_names

    def feasible(self, variant: str, ctx, state=None) -> bool:
        return variant in self.vp.variant_names

    def resolve(self, variant: str) -> str:  # not used in-search (resolution is collective)
        return f"{self.vp.kind} {self.vp.point_id} : {self.vp.type_of(variant)};"


@dataclass
class VariationDSEResult:
    recommended_choices: Dict[str, str]
    concrete_model: str
    pareto_front: List[Tuple[State, Objectives]]
    admitted_points: List[str]
    rejected_points: List[str]
    real_quality: Dict[str, float]
    evaluated: int = 0  # number of distinct architectures scored during the search
    recommended_capacity_mah: Optional[float] = None  # inner-BO chosen battery capacity
    notes: List[str] = field(default_factory=list)
    # Pareto front resolved to concrete design inputs (for the SysML trade study);
    # recommended_design is the front member the recommendation picked.
    pareto_designs: List[Tuple["DesignInputs", Objectives]] = field(default_factory=list)
    recommended_design: Optional["DesignInputs"] = None
    # Per-alternative variant→implementation bindings ({point_id: impl_type_name}), index-
    # aligned with pareto_designs, so the trade study can FORMALLY bind each alternative to
    # the variant definitions it's composed of (object-level traceability, not a comment).
    pareto_bindings: List[Dict[str, str]] = field(default_factory=list)
    recommended_bindings: Dict[str, str] = field(default_factory=dict)


def _design_quality(dims: Dict[str, float]) -> float:
    vals = [dims.get(d, 0.0) for d in _QUALITY_DIMS]
    return sum(vals) / len(vals)


def _recommendation_weights(names, requirements) -> Dict[str, float]:
    """Objective weights for picking the recommended design: each perf family weighted
    by how many requirement targets it carries (requirement emphasis), other objectives
    (cost, generic dims) a baseline 1. Normalised. Objective + traceable to requirements
    — not a lexicographic guess."""
    counts: Dict[str, float] = {}
    for fts in requirement_targets(requirements).values():
        for fam, _ in fts:
            counts[fam + "_sat"] = counts.get(fam + "_sat", 0.0) + 1.0
    w = {n: counts.get(n, 1.0) for n in names}
    total = sum(w.values()) or 1.0
    return {n: v / total for n, v in w.items()}


def _write_back_capacity(model_text: str, capacity_mah: float) -> str:
    """Inject the inner-BO battery capacity into the resolved model so downstream
    (SITL params, reports) reads it. Prefer the part def declaring batteryCells (the
    power variant); else fall back to the first specialised variant part def."""
    ins = f" attribute batteryCapacityMah : Real = {capacity_mah};"
    fallback_end = None
    for m in re.finditer(r"\bpart\s+def\s+\w+\s*(?::>[^{]*)?\{", model_text):
        brace = model_text.index("{", m.start())
        end = find_block_end(model_text, brace)
        if end == -1:
            continue
        header = model_text[m.start():brace]
        body = model_text[brace + 1:end]
        if "batteryCapacityMah" in body:
            continue  # already present
        if "batteryCells" in body:
            return model_text[:end] + ins + model_text[end:]   # power variant — best home
        if ":>" in header and fallback_end is None:
            fallback_end = end                                  # first specialised variant
    if fallback_end is not None:
        return model_text[:fallback_end] + ins + model_text[fallback_end:]
    return model_text


def run_variation_dse(
    model,
    requirements: Optional[List[str]] = None,
    iterations: Optional[int] = None,
    random_seed: Optional[int] = 0,
) -> Optional[VariationDSEResult]:
    requirements = requirements or []
    base_text = get_sysml_text(model)
    if not base_text:
        return None
    points = parse_variation_points(base_text)
    ok, bad = admitted(points)
    notes: List[str] = []
    if bad:
        notes.append(
            f"{len(bad)} variation point(s) rejected (no rationale/linked requirement): "
            + ", ".join(p.point_id for p in bad)
        )
    # resolve-safety: only explore points whose variants share a port interface,
    # so binding a variant never breaks the host's connects.
    ok, unsafe = port_safe_split(ok, base_text)
    if unsafe:
        notes.append(
            f"{len(unsafe)} variation point(s) rejected (variants lack a shared port "
            f"interface → resolve would break connects): "
            + ", ".join(p.point_id for p in unsafe)
        )
    if not ok:
        return None  # no admissible, resolve-safe LLM-declared variation space

    # Deduplicate design-field ownership: if two variation points parametrise the same
    # physical quantity (e.g. propulsion AND airframe both declaring rotorCount), the
    # search would explore incoherent combos (hexa propulsion + octo airframe). Keep each
    # field on its canonical owner, strip it from the others (notes record what changed).
    base_text, norm_notes = normalize_variation_ownership(base_text, ok)
    notes.extend(norm_notes)
    # Strip inner-loop variables (e.g. batteryCapacityMah) from variants: capacity is the
    # inner-BO variable, not a discrete variant choice — pinning it per variant makes the
    # interface inconsistent, misleads, and interferes with capacity write-back.
    base_text, strip_notes = strip_inner_loop_attrs(base_text, ok)
    notes.extend(strip_notes)

    operators = [VariationOperator(p) for p in ok]
    # Budget scales with the total number of variant choices (points × variants per
    # point), so wider models and denser variant sets don't get under-explored.
    if iterations is None:
        total_choices = sum(len(op.variants) for op in operators)
        iterations = max(60, 30 * total_choices)
    cache: Dict[Tuple, Objectives] = {}
    inner_cap: Dict[Tuple, float] = {}   # per-architecture inner-BO battery capacity

    # Domain-aware objective (requirement-target driven) when the requirements
    # supply measurable targets matching variant attributes; else fall back to the
    # generic design-quality / simplicity dims.
    domain_names = objective_names(requirements)
    use_domain = len(domain_names) > 1  # at least one perf family + cost_efficiency
    names = domain_names if use_domain else ["design_quality", "simplicity"]
    endurance_tgt = endurance_target(requirements) if use_domain else 0.0
    # REQ_PERF_002 mandates endurance "at the maximum rated payload": evaluate every
    # design at that worst-case load (not the 0.5 kg default) so sizing/feasibility are
    # honest. 0 → no payload requirement → keep the resolved/default payload.
    rated_payload = max_rated_payload(requirements) if use_domain else 0.0

    def _at_rated_payload(arch: Dict[str, float]) -> Dict[str, float]:
        if rated_payload > 0:
            arch = {**arch, "payload_mass_kg": rated_payload}
        return arch

    def objective_fn(state: State, ctx) -> Objectives:
        key = tuple(sorted(state.items()))
        if key not in cache:
            if use_domain:
                # BILEVEL: outer = discrete architecture (this state); inner BO tunes the
                # continuous battery capacity to the cheapest pack meeting the endurance
                # target, then we score the inner-optimized design.
                di0 = architecture_design(ok, dict(state), base_text)
                arch = _at_rated_payload(design_arch_inputs(di0))
                if endurance_tgt > 0:
                    cap = optimize_capacity(arch, endurance_tgt, seed=random_seed or 0)["capacity_mah"]
                else:
                    cap = di0.battery_capacity_mah
                di = DesignInputs(battery_capacity_mah=cap, **arch)
                cache[key] = objectives_from_design(di, ok, dict(state), requirements)
                inner_cap[key] = cap
            else:
                concrete = resolve_model(base_text, ok, dict(state))
                dims = evaluate_design_quality(concrete)
                n_parts = len(re.findall(r"\bpart\s+\w+\s*:", concrete))
                cache[key] = {
                    "design_quality": _design_quality(dims),
                    "simplicity": 1.0 / (1.0 + n_parts / 10.0),
                }
        return cache[key]

    mcts = MultiObjectiveMCTS(operators, objective_fn, object(), names, [0.0] * len(names),
                             random_seed=random_seed)
    front = mcts.search(iterations=iterations)

    # recommend objectively: per-objective weights from requirement emphasis (how many
    # targets each family carries; cost a baseline), then Chebyshev (max-min balance) —
    # avoids the arbitrary lexicographic bias toward the first objective. (Not SAFE-
    # severity weighting: variation objectives are perf families, not failure severities.)
    rec_weights = _recommendation_weights(names, requirements)
    # FEASIBILITY GATE: hard "at least" perf requirements (e.g. endurance) cap their
    # satisfaction at 1.0, so a design meets them iff every perf _sat is ~1.0. Some
    # architectures can't reach the target even at the inner-BO capacity bound — they
    # must not be RECOMMENDED (they still stay on the front / in the trade study to show
    # the trade-off). Recommend among feasible designs; if none, fall back + flag it.
    # tolerance: the inner BO sizes the CHEAPEST pack meeting the target, so a feasible
    # design lands marginally under (e.g. 24.97/25 = 0.999); genuinely infeasible designs
    # miss by a wide margin (a tiny-rotor craft sits at ~0.54), so 0.98 cleanly separates.
    perf_objs = [n for n in names if n.endswith("_sat")]
    feasible = [m for m in front.members
                if all(m[1].get(n, 0.0) >= 0.98 for n in perf_objs)]
    pool = feasible if feasible else front.members
    if perf_objs and not feasible:
        notes.append("no explored design meets all hard performance requirements at the "
                     "capacity bounds; recommending the closest (INFEASIBLE).")
    rec_state, _ = weighted_recommend(pool, rec_weights, method="chebyshev")
    concrete = resolve_model(base_text, ok, dict(rec_state))
    rec_cap = inner_cap.get(tuple(sorted(rec_state.items())))
    if rec_cap is not None:
        wb = _write_back_capacity(concrete, rec_cap)
        if not check_syntax(wb).has_errors:
            concrete = wb   # inner-optimized capacity now lives in the model

    # Resolve each Pareto member to concrete design inputs (inner-optimized capacity) so
    # the recommendation can be presented as a SysML trade study over real alternatives.
    def _resolve_di(state: State) -> DesignInputs:
        di0 = architecture_design(ok, dict(state), base_text)
        cap = inner_cap.get(tuple(sorted(state.items())), di0.battery_capacity_mah)
        return DesignInputs(battery_capacity_mah=cap, **_at_rated_payload(design_arch_inputs(di0)))

    def _bindings(state: State) -> Dict[str, str]:
        """{point_id: chosen variant's impl type name} — the variant defs this design uses."""
        s = dict(state)
        return {p.point_id: p.type_of(s[p.point_id])
                for p in ok if p.point_id in s and p.type_of(s[p.point_id])}

    pareto_designs: List[Tuple[DesignInputs, Objectives]] = []
    pareto_bindings: List[Dict[str, str]] = []
    rec_design: Optional[DesignInputs] = None
    rec_bindings: Dict[str, str] = {}
    if use_domain:
        seen = set()
        for state, objs in front.members:
            di = _resolve_di(state)
            sig = (di.battery_capacity_mah, di.battery_cells, di.rotor_count,
                   di.rotor_radius_m, di.payload_mass_kg, di.cruise_speed_mps)
            if sig in seen:
                continue
            seen.add(sig)
            pareto_designs.append((di, objs))
            pareto_bindings.append(_bindings(state))
        rec_design = _resolve_di(rec_state)
        rec_bindings = _bindings(rec_state)
    return VariationDSEResult(
        recommended_choices=dict(rec_state),
        concrete_model=concrete,
        pareto_front=front.members,
        admitted_points=[p.point_id for p in ok],
        rejected_points=[p.point_id for p in bad],
        real_quality=evaluate_design_quality(concrete),
        evaluated=len(cache),
        recommended_capacity_mah=rec_cap,
        notes=notes,
        pareto_designs=pareto_designs,
        recommended_design=rec_design,
        pareto_bindings=pareto_bindings,
        recommended_bindings=rec_bindings,
    )
