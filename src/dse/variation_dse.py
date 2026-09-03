"""DSE over LLM-declared SysML v2 variation points (the "replace" path).

The explored space is the ``variation`` points the LLM declared in the model,
admitted only if they carry a rationale + linked requirement. Each point becomes
a dynamic operator, the outer MO-MCTS explores combinations, and each candidate
is resolved into a concrete model and scored by the DesignEvaluator (full models,
so the real evaluator applies here). Returns a Pareto front over (design_quality,
simplicity) + a recommended concrete model, or None when the model declares no
admissible variation points.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.suppressed import record_suppressed
from ..utils.sysml_text_utils import get_sysml_text, named_block_span
from .analysis_emitter import _inject_attr_into_type
from .domain_objective import (
    DESIGN_DEFAULTS,
    architecture_design,
    design_arch_inputs,
    endurance_target,
    evaluation_overrides,
    mass_limit,
    normalize_variation_space,
    objective_names,
    objectives_from_design,
    requirement_targets,
    variant_design_inputs,
)
from .inner_sizing import optimize_capacity, optimize_discrete_capacity
from .physics_estimator import DesignInputs, total_mass_kg
from .weighting import recommend as weighted_recommend, sensitivity
from .mo_mcts import MultiObjectiveMCTS, Objectives, State, dominates
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

    def resolve(self, variant: str) -> str:
        return f"{self.vp.kind} {self.vp.point_id} : {self.vp.type_of(variant)};"


@dataclass
class VariationDSEResult:
    recommended_choices: Dict[str, str]
    concrete_model: str
    pareto_front: List[Tuple[State, Objectives]]
    admitted_points: List[str]
    rejected_points: List[str]
    real_quality: Dict[str, float]
    evaluated: int = 0
    # The unconstrained estimator Pareto is diagnostic only. ``pareto_front`` above
    # is the official front rebuilt *after* all hard constraints are applied.
    exploratory_pareto_front: List[Tuple[State, Objectives]] = field(default_factory=list)
    search_space_size: Optional[int] = None
    coverage_mode: str = "mcts"
    estimator_feasible_count: int = 0
    mapping_compliant_count: Optional[int] = None
    phase8_closable_count: Optional[int] = None
    constraint_feasible_count: int = 0
    recommended_capacity_mah: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    pareto_designs: List[Tuple["DesignInputs", Objectives]] = field(default_factory=list)
    recommended_design: Optional["DesignInputs"] = None
    recommended_realizable: Optional[bool] = None
    realizable_front_count: Optional[int] = None
    recommended_by: Optional[str] = None
    recommended_estimator_feasible: Optional[bool] = None
    # Per-alternative variant->implementation bindings ({point_id: impl_type_name}),
    # index-aligned with pareto_designs, so the trade study binds each alternative to
    # the variant definitions it is composed of (object-level traceability).
    pareto_bindings: List[Dict[str, str]] = field(default_factory=list)
    recommended_bindings: Dict[str, str] = field(default_factory=dict)
    recommendation_status: str = "RECOMMENDED"
    recommendable_front_count: Optional[int] = None
    exploratory_choices: Dict[str, str] = field(default_factory=dict)
    exploratory_design: Optional["DesignInputs"] = None
    # Weight-simplex sensitivity of the weight-stage pick over the official front
    # (docs/DSE_REDESIGN.md §三-A). Diagnostic only: when recommended_by=="datasheet"
    # the datasheet rank decides the final selection and the weights only break ties.
    weight_sensitivity: Optional[Dict] = None


def _design_quality(dims: Dict[str, float]) -> float:
    vals = [dims.get(d, 0.0) for d in _QUALITY_DIMS]
    return sum(vals) / len(vals)


def _state_label(state: State) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(state.items()))


def _recommendation_weights(names, requirements) -> Dict[str, float]:
    """Objective weights for picking the recommended design: each perf family is
    weighted by how many requirement targets it carries, other objectives (cost,
    generic dims) get a baseline 1, then normalised. Traceable to the requirement set.
    """
    counts: Dict[str, float] = {}
    for fts in requirement_targets(requirements).values():
        for fam, _ in fts:
            counts[fam + "_sat"] = counts.get(fam + "_sat", 0.0) + 1.0
    w = {n: counts.get(n, 1.0) for n in names}
    total = sum(w.values()) or 1.0
    return {n: v / total for n, v in w.items()}


def _pareto_members(
    members: List[Tuple[State, Objectives]], names: List[str],
) -> List[Tuple[State, Objectives]]:
    """Rebuild a maximisation Pareto front from an arbitrary candidate set.

    Objective-equal but architecturally distinct states are retained: a later
    datasheet rank may distinguish them, and dropping one for being evaluated second
    would make the recommendation order-dependent.
    """
    unique: List[Tuple[State, Objectives]] = []
    seen = set()
    for state, objectives in members:
        key = tuple(sorted(state.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append((dict(state), dict(objectives)))

    def vector(objectives: Objectives) -> Tuple[float, ...]:
        return tuple(objectives[name] for name in names)

    return [
        member for member in unique
        if not any(
            other is not member and dominates(vector(other[1]), vector(member[1]))
            for other in unique
        )
    ]


def _write_back_capacity(
    model_text: str, capacity_mah: float, bound_types: Sequence[str] = (),
) -> str:
    """Inject the inner-BO battery capacity into the resolved model so downstream (SITL
    params, reports) reads it.

    The capacity is written into a bound variant's type def - the one declaring
    ``batteryCells`` (the power owner) first, else the first bound def without a
    capacity. A flat first-specialised-def heuristic landed it in the first retained
    Pareto alternative instead of the chosen one, so the archived model asserted a
    capacity for a design that was never selected. Without ``bound_types``
    (degenerate spaces with untyped variants) the text is returned unchanged.
    """
    ordered = list(dict.fromkeys(t for t in bound_types if t))
    fallback = None
    for type_name in ordered:
        span = named_block_span(model_text, "part", type_name)
        if span is None:
            # A bodiless bound def (`part def X :> Y;`) still owns the design, so materialise
            # a body; dropping the capacity would fly SITL at the 5000 mAh default rather than
            # the optimum.
            decl = re.search(
                rf"\bpart\s+def\s+{re.escape(type_name)}\b[^{{;\n]*;", model_text
            )
            if decl is not None:
                head = model_text[decl.start():decl.end() - 1]
                body = (
                    f"{head} {{ attribute batteryCapacityMah : Real = "
                    f"{float(capacity_mah)}; }}"
                )
                return (
                    model_text[:decl.start()] + body + model_text[decl.end():]
                )
            continue
        body = model_text[span[0] + 1:span[1]]
        if "batteryCapacityMah" in body:
            return model_text
        if "batteryCells" in body:
            injected, ok = _inject_attr_into_type(
                model_text, type_name, "batteryCapacityMah", float(capacity_mah)
            )
            return injected if ok else model_text
        if fallback is None:
            fallback = type_name
    if fallback is not None:
        injected, ok = _inject_attr_into_type(
            model_text, fallback, "batteryCapacityMah", float(capacity_mah)
        )
        return injected if ok else model_text
    return model_text


def run_variation_dse(
    model,
    requirements: Optional[List[str]] = None,
    iterations: Optional[int] = None,
    random_seed: Optional[int] = 0,
    realizability: Optional[Callable[[DesignInputs], bool]] = None,
    realization_rank: Optional[Callable[[DesignInputs], float]] = None,
    recommendability: Optional[Callable[[DesignInputs], bool]] = None,
    capacity_options: Optional[Callable[[DesignInputs], List[float]]] = None,
    exhaustive_limit: int = 4096,
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
    # so binding a variant cannot break the host's connects.
    ok, unsafe = port_safe_split(ok, base_text)
    if unsafe:
        notes.append(
            f"{len(unsafe)} variation point(s) rejected (variants lack a shared port "
            f"interface → resolve would break connects): "
            + ", ".join(p.point_id for p in unsafe)
        )
    if not ok:
        return None

    # Deduplicate design-field ownership: if two variation points parametrise the
    # same physical quantity (propulsion and airframe both declaring rotorCount),
    # the search would explore incoherent combos (hexa propulsion + octo airframe).
    # Keep each field on its canonical owner and strip it from the others; notes
    # record the changes. The same entry point also strips inner-loop variables,
    # so the LLM-declared variation space gets a uniform interface.
    base_text, norm_notes = normalize_variation_space(base_text, ok)
    notes.extend(norm_notes)

    operators = [VariationOperator(p) for p in ok]
    # Budget scales with the total number of variant choices (points x variants per
    # point), so wider models and denser variant sets don't get under-explored.
    if iterations is None:
        total_choices = sum(len(op.variants) for op in operators)
        iterations = max(60, 30 * total_choices)
    cache: Dict[Tuple, Objectives] = {}
    inner_cap: Dict[Tuple, float] = {}
    sizing_mode: Dict[Tuple, str] = {}

    domain_names = objective_names(requirements)
    use_domain = len(domain_names) > 1  # at least one perf family + cost_efficiency
    names = domain_names if use_domain else ["design_quality", "simplicity"]
    endurance_tgt = endurance_target(requirements) if use_domain else 0.0
    # All-up non-structural mass = delivery payload (the rated requirement load,
    # REQ_PERF_002 "at maximum rated payload") + Σ component masses (massKg) of the
    # chosen variants (sensor/gimbal/airframe/...). Without the sum the chosen
    # components never enter Endurance/MTOW and both come out optimistic. Assumes
    # variant massKg is equipment mass, separate from the requirement's delivery
    # payload, which holds for the current generation.
    _rated = evaluation_overrides(requirements).get("payload_mass_kg", 0.0) if use_domain else 0.0
    _delivery = _rated if _rated > 0 else DESIGN_DEFAULTS["payload_mass_kg"]

    def _added_mass(state: State) -> float:
        s = dict(state)
        comp = sum(variant_design_inputs(base_text, vp.type_of(s[vp.point_id])).get("payload_mass_kg", 0.0)
                   for vp in ok if vp.point_id in s)
        return _delivery + comp

    def _design_with_mass(state: State, di0: DesignInputs) -> Dict[str, float]:
        arch = design_arch_inputs(di0)
        arch["payload_mass_kg"] = _added_mass(state)
        return arch

    def objective_fn(state: State, ctx) -> Objectives:
        key = tuple(sorted(state.items()))
        if key not in cache:
            if use_domain:
                di0 = architecture_design(ok, dict(state), base_text)
                arch = _design_with_mass(state, di0)
                if endurance_tgt > 0:
                    seed_design = DesignInputs(
                        battery_capacity_mah=di0.battery_capacity_mah, **arch
                    )
                    options = []
                    if capacity_options is not None:
                        try:
                            options = list(capacity_options(seed_design) or [])
                        except Exception as exc:
                            record_suppressed("variation_dse.capacity_options", exc)
                    if options:
                        sized = optimize_discrete_capacity(arch, endurance_tgt, options)
                        sizing_mode[key] = "catalog_discrete"
                    else:
                        sized = optimize_capacity(arch, endurance_tgt, seed=random_seed or 0)
                        sizing_mode[key] = "exploratory_continuous"
                    cap = sized["capacity_mah"]
                else:
                    cap = di0.battery_capacity_mah
                    sizing_mode[key] = "declared"
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

    search_ctx = object()
    mcts = MultiObjectiveMCTS(
        operators, objective_fn, search_ctx, names, [0.0] * len(names),
        random_seed=random_seed,
    )
    raw_front = mcts.search(iterations=iterations)

    # The search archive holds only nondominated rollouts, while the hard constraints
    # are evaluated over all scored designs, so every terminal state is kept in
    # ``cache`` and finite catalog spaces are covered explicitly. Current generated
    # spaces are small (e.g. 10 catalog architectures x 2 equipment choices), so
    # exhaustive coverage is cheap to audit and does not rely on a stochastic rollout
    # sampling each one.
    search_space_size = math.prod(len(op.variants) for op in operators)
    coverage_mode = "mcts"
    if 0 < search_space_size <= exhaustive_limit:
        for choices in itertools.product(*(op.variants for op in operators)):
            state: State = {}
            legal = True
            for op, variant in zip(operators, choices):
                if not op.feasible(variant, search_ctx, state):
                    legal = False
                    break
                state[op.point_id] = variant
            if legal:
                objective_fn(state, search_ctx)
        coverage_mode = "exhaustive"
        notes.append(
            f"search coverage: exhaustively evaluated {len(cache)}/{search_space_size} "
            "legal outer configurations"
        )
    else:
        # For a large space keep MO-MCTS, but still force at least one complete evaluation
        # of every catalog architecture seed. Other points use their lightest declared
        # equipment choice as a deterministic baseline.
        seed_ops = [
            op for op in operators
            if "catalog architecture seed" in op.vp.rationale.lower()
        ]
        if seed_ops:
            baseline: State = {}
            for op in operators:
                baseline[op.point_id] = min(
                    op.variants,
                    key=lambda variant: variant_design_inputs(
                        base_text, op.vp.type_of(variant)
                    ).get("payload_mass_kg", 0.0),
                )
            seeded = 0
            for seed_op in seed_ops:
                for variant in seed_op.variants:
                    state = dict(baseline)
                    state[seed_op.point_id] = variant
                    objective_fn(state, search_ctx)
                    seeded += 1
            coverage_mode = "mcts+catalog-seed-stratified"
            notes.append(
                f"search coverage: large space ({search_space_size} configurations); "
                f"MO-MCTS plus {seeded} deterministic catalog-seed evaluations"
            )

    # Test seams and external search implementations may return archive members
    # without calling objective_fn. Hydrate those states so the constraint pipeline
    # still operates on a single evaluated-candidate representation.
    for state, objectives in raw_front.members:
        key = tuple(sorted(state.items()))
        if key not in cache:
            cache[key] = dict(objectives)
            if use_domain:
                di0 = architecture_design(ok, dict(state), base_text)
                inner_cap[key] = di0.battery_capacity_mah
                sizing_mode[key] = "search-reported"

    catalog_sized = sum(mode == "catalog_discrete" for mode in sizing_mode.values())
    exploratory_sized = sum(mode == "exploratory_continuous" for mode in sizing_mode.values())
    if capacity_options is not None:
        notes.append(
            f"inner sizing: {catalog_sized} architecture(s) used real catalog pack "
            f"capacities; {exploratory_sized} remained exploratory-continuous"
        )

    def _resolve_di(state: State) -> DesignInputs:
        di0 = architecture_design(ok, dict(state), base_text)
        cap = inner_cap.get(tuple(sorted(state.items())), di0.battery_capacity_mah)
        return DesignInputs(battery_capacity_mah=cap, **_design_with_mass(state, di0))

    evaluated_members = [
        (dict(key), dict(objectives))
        for key, objectives in sorted(cache.items(), key=lambda item: item[0])
    ]
    if not evaluated_members:
        return None

    # Keep the unconstrained estimator front as an exploratory diagnostic; it is not
    # the input to the mapping/closure filters.
    exploratory_front = _pareto_members(evaluated_members, names)

    # Hard-constraint pipeline, in order:
    #   all evaluated -> estimator feasible -> catalog mapping -> Phase 8 closure
    # The official Pareto front is rebuilt only after that, so this is constrained
    # multi-objective optimisation rather than a filter over an unconstrained front.
    perf_objs = [n for n in names if n.endswith("_sat")]
    perf_feasible = [
        member for member in evaluated_members
        if all(member[1].get(name, 0.0) >= 0.98 for name in perf_objs)
    ]
    _mass_req, mtow_limit = mass_limit(requirements)
    estimator_feasible = []
    for member in perf_feasible:
        if mtow_limit > 0 and total_mass_kg(_resolve_di(member[0])) > mtow_limit + 1e-9:
            continue
        estimator_feasible.append(member)
    if mtow_limit > 0:
        notes.append(
            f"estimator gate: {len(estimator_feasible)}/{len(perf_feasible)} "
            f"performance-feasible designs also satisfy MTOW <= {mtow_limit:g} kg"
        )

    # A raising predicate is not an engineering "no": count the failures and attribute
    # them, or a systematic code fault (renamed catalog field, division by zero in the
    # estimator) empties the feasible set and gets published as the verdict "no Pareto
    # member closes Phase 8". record_suppressed alone is invisible - no run artifact
    # writes it.
    gate_errors: Dict[str, int] = {"realizability": 0, "recommendability": 0}
    mapping_compliant = list(estimator_feasible)
    if realizability is not None:
        mapping_compliant = []
        for member in estimator_feasible:
            try:
                if bool(realizability(_resolve_di(member[0]))):
                    mapping_compliant.append(member)
            except Exception as exc:
                gate_errors["realizability"] += 1
                record_suppressed("variation_dse.realizability", exc)

    phase8_closable = list(mapping_compliant)
    if recommendability is not None:
        phase8_closable = []
        for member in mapping_compliant:
            try:
                if bool(recommendability(_resolve_di(member[0]))):
                    phase8_closable.append(member)
            except Exception as exc:
                gate_errors["recommendability"] += 1
                record_suppressed("variation_dse.recommendability", exc)

    if gate_errors["realizability"]:
        notes.append(
            f"GATE ERROR: realizability predicate raised on "
            f"{gate_errors['realizability']}/{len(estimator_feasible)} "
            "design(s) — those were dropped as infeasible; the constrained "
            "front may be incomplete for a CODE reason, not a design reason"
        )
    if gate_errors["recommendability"]:
        notes.append(
            f"GATE ERROR: recommendability predicate raised on "
            f"{gate_errors['recommendability']}/{len(mapping_compliant)} "
            "design(s) — those were dropped as unclosable; the constrained "
            "front may be incomplete for a CODE reason, not a design reason"
        )

    official_front = _pareto_members(phase8_closable, names)
    rec_weights = _recommendation_weights(names, requirements)
    exploratory_pool = [
        member for member in exploratory_front
        if member in estimator_feasible
    ] or exploratory_front
    exploratory_state, _ = weighted_recommend(
        exploratory_pool, rec_weights, method="chebyshev"
    )

    rec_state: Optional[State] = None
    recommendation_status = "NO_RECOMMENDABLE_DESIGN"
    recommended_realizable: Optional[bool] = (
        bool(official_front) if realizability is not None else None
    )
    recommended_by: Optional[str] = None
    if official_front:
        rec_state, _ = weighted_recommend(official_front, rec_weights, method="chebyshev")
        recommendation_status = "RECOMMENDED"
        if realizability is not None or recommendability is not None:
            recommended_by = "estimator-within-constrained-pareto"
        notes.append(
            f"official Pareto rebuilt from {len(phase8_closable)} constrained-feasible "
            f"design(s): {len(official_front)} nondominated alternative(s)"
        )
        if realization_rank is not None:
            ranked = []
            for member in official_front:
                try:
                    ranked.append((float(realization_rank(_resolve_di(member[0]))), member))
                except Exception as exc:
                    record_suppressed("variation_dse.realization_rank", exc)
            if ranked:
                best_rank = max(rank for rank, _ in ranked)
                tied = [member for rank, member in ranked if abs(rank - best_rank) <= 1e-9]
                rec_state, _ = weighted_recommend(tied, rec_weights, method="chebyshev")
                recommended_by = "datasheet"
                notes.append("recommendation selected by datasheet rank inside constrained Pareto")
                if len(tied) > 1:
                    notes.append("datasheet rank tie broken by estimator recommendation")
            else:
                notes.append(
                    "realization_rank failed for all constrained-Pareto members — "
                    "using estimator ranking inside the constrained front"
                )
    else:
        if not estimator_feasible:
            notes.append(
                "NO_RECOMMENDABLE_DESIGN: no evaluated design passes the estimator "
                "performance + MTOW gate"
            )
        elif realizability is not None and not mapping_compliant:
            if estimator_feasible and gate_errors["realizability"] == len(estimator_feasible):
                notes.append(
                    "NO_RECOMMENDABLE_DESIGN: the realizability predicate raised "
                    "on EVERY estimator-feasible design — this is a code/catalog "
                    "failure, NOT evidence that no design is mapping-compliant"
                )
            else:
                notes.append(
                    "NO_RECOMMENDABLE_DESIGN: estimator-feasible designs exist, but none "
                    "is catalog mapping-compliant"
                )
        elif recommendability is not None and not phase8_closable:
            if mapping_compliant and gate_errors["recommendability"] == len(mapping_compliant):
                notes.append(
                    "NO_RECOMMENDABLE_DESIGN: the recommendability predicate raised "
                    "on EVERY mapping-compliant design — this is a code failure, "
                    "NOT evidence that no design closes Phase 8"
                )
            else:
                notes.append(
                    "NO_RECOMMENDABLE_DESIGN: estimator-feasible, mapping-compliant designs "
                    "exist, but none closes Phase 8"
                )
        else:
            notes.append("NO_RECOMMENDABLE_DESIGN: constrained feasible set is empty")
        recommended_by = "none" if (realizability is not None or recommendability is not None) else None

    mapping_count = len(mapping_compliant) if realizability is not None else None
    phase8_count = len(phase8_closable) if recommendability is not None else None
    notes.append(
        "constraint pipeline: "
        f"evaluated={len(evaluated_members)} -> estimator={len(estimator_feasible)} -> "
        f"catalog={mapping_count if mapping_count is not None else 'not-applied'} -> "
        f"phase8={phase8_count if phase8_count is not None else 'not-applied'} -> "
        f"official_pareto={len(official_front)}"
    )
    recommended_estimator_feasible = True if rec_state is not None else None

    # Weight-simplex sensitivity over the official front, using the same scalarisation
    # as the recommender above (method="chebyshev"; sensitivity() defaults to
    # weighted_sum, which would report robustness of a pick this path never makes).
    # A 1-member front is computed too, so robustness == 100% is machine-readable
    # evidence rather than a skipped case.
    weight_sensitivity: Optional[Dict] = None
    if official_front:
        sens = sensitivity(
            official_front, rec_weights, label_fn=_state_label,
            n_samples=2000, method="chebyshev", random_seed=random_seed,
        )
        weight_sensitivity = {
            "nominal_label": sens.nominal_label,
            "nominal_robustness": sens.nominal_robustness,
            "selection_frequency": sens.selection_frequency,
            "n_samples": sens.n_samples,
            "method": "chebyshev",
            "nominal_matches_recommendation": (
                rec_state is not None
                and _state_label(rec_state) == sens.nominal_label
            ),
        }
        notes.append(
            f"weight-simplex sensitivity: nominal={sens.nominal_label} wins "
            f"{sens.nominal_robustness:.1%} of sampled weight space "
            f"({sens.n_samples} samples, chebyshev)"
        )

    def _bindings(state: State) -> Dict[str, str]:
        s = dict(state)
        return {p.point_id: p.type_of(s[p.point_id])
                for p in ok if p.point_id in s and p.type_of(s[p.point_id])}

    concrete = base_text if rec_state is None else resolve_model(base_text, ok, dict(rec_state))
    rec_cap = None if rec_state is None else inner_cap.get(tuple(sorted(rec_state.items())))
    if rec_cap is not None:
        wb = _write_back_capacity(
            concrete, rec_cap, list(_bindings(dict(rec_state)).values())
        )
        if not check_syntax(wb).has_errors:
            concrete = wb

    # Resolve each Pareto member to concrete design inputs (inner-optimized capacity)
    # so the recommendation is presented as a SysML trade study over alternatives.

    pareto_designs: List[Tuple[DesignInputs, Objectives]] = []
    pareto_bindings: List[Dict[str, str]] = []
    rec_design: Optional[DesignInputs] = None
    rec_bindings: Dict[str, str] = {}
    exploratory_design: Optional[DesignInputs] = None
    if use_domain:
        seen = set()
        for state, objs in official_front:
            di = _resolve_di(state)
            sig = (di.battery_capacity_mah, di.battery_cells, di.rotor_count,
                   di.rotor_radius_m, di.payload_mass_kg, di.cruise_speed_mps)
            if sig in seen:
                continue
            seen.add(sig)
            pareto_designs.append((di, objs))
            pareto_bindings.append(_bindings(state))
        exploratory_design = _resolve_di(exploratory_state)
        if rec_state is not None:
            rec_design = _resolve_di(rec_state)
            rec_bindings = _bindings(rec_state)
    return VariationDSEResult(
        recommended_choices=dict(rec_state) if rec_state is not None else {},
        concrete_model=concrete,
        pareto_front=official_front,
        admitted_points=[p.point_id for p in ok],
        rejected_points=[p.point_id for p in bad],
        real_quality=evaluate_design_quality(concrete),
        evaluated=len(cache),
        exploratory_pareto_front=exploratory_front,
        search_space_size=search_space_size,
        coverage_mode=coverage_mode,
        estimator_feasible_count=len(estimator_feasible),
        mapping_compliant_count=mapping_count,
        phase8_closable_count=phase8_count,
        constraint_feasible_count=len(phase8_closable),
        recommended_capacity_mah=rec_cap,
        notes=notes,
        pareto_designs=pareto_designs,
        recommended_design=rec_design,
        recommended_realizable=recommended_realizable,
        realizable_front_count=mapping_count,
        recommended_by=recommended_by,
        recommended_estimator_feasible=recommended_estimator_feasible,
        pareto_bindings=pareto_bindings,
        recommended_bindings=rec_bindings,
        recommendation_status=recommendation_status,
        recommendable_front_count=phase8_count,
        exploratory_choices=dict(exploratory_state),
        exploratory_design=exploratory_design,
        weight_sensitivity=weight_sensitivity,
    )
