"""Requirement-traceable weighting + weight-simplex sensitivity (pain point A).

The search is weight-free (it returns a Pareto front); weights enter only to
recommend one design from the front, replacing the hardcoded
0.40/0.30/0.15/0.15. ``derive_weights`` takes objective weights from the
requirement set's own composition (category counts x priority), so every weight
traces to the input. ``sensitivity`` samples the whole weight simplex (Dirichlet)
and reports how robust a recommendation is: the fraction of weight space in which
it wins, and how often each design would be selected. Pure-Python (no numpy).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Objectives = Dict[str, float]
State = Dict[str, str]
FrontMember = Tuple[State, Objectives]


def derive_weights(
    category_counts: Dict[str, int],
    objective_categories: Dict[str, Sequence[str]],
    priorities: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Weights per objective, proportional to the requirement mass behind it.

    ``category_counts``      e.g. {"SAFE": 5, "PERF": 3, "CONS": 2}
    ``objective_categories`` which requirement categories feed each objective,
                             e.g. {"reliability": ["SAFE", "PERF"],
                                   "cost_efficiency": ["CONS"]}
    ``priorities``           optional per-category priority multiplier (default 1).

    Traceable: a project with more SAFE requirements gets a higher reliability weight
    because of its requirements.
    """
    priorities = priorities or {}
    raw: Dict[str, float] = {}
    for obj, cats in objective_categories.items():
        raw[obj] = sum(
            category_counts.get(c, 0) * priorities.get(c, 1.0) for c in cats
        )
    return _normalise(raw, len(objective_categories))


# severity -> per-requirement importance (Minor ~ a baseline requirement of 1.0;
# nonlinear so one catastrophic dominates many minor ones). Policy, and itself a
# candidate sensitivity axis like the weight simplex.
from .requirements_profile import RequirementProfile, Severity  # noqa: E402

SEVERITY_IMPORTANCE: Dict[Severity, float] = {
    Severity.CATASTROPHIC: 10.0,
    Severity.HAZARDOUS: 6.0,
    Severity.MAJOR: 3.0,
    Severity.MINOR: 1.0,
    Severity.NO_EFFECT: 0.25,
}


def _safe_mass(profile: RequirementProfile, default_severity: Severity) -> float:
    mass = sum(SEVERITY_IMPORTANCE[s] for s in profile.safe_severities)
    mass += profile.unclassified_safe * SEVERITY_IMPORTANCE[default_severity]
    return mass


def _normalise(raw: Dict[str, float], n: int) -> Dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        return {o: 1.0 / n for o in raw}
    return {o: v / total for o, v in raw.items()}


def derive_weights_from_profile(
    profile: RequirementProfile,
    objective_categories: Dict[str, Sequence[str]],
    priorities: Optional[Dict[str, float]] = None,
    default_severity: Severity = Severity.MAJOR,
) -> Dict[str, float]:
    """Severity-weighted objective weights (the upgrade to count-based).

    The SAFE category contributes its severity mass (Σ importance per hazard
    severity) rather than its count, so one Catastrophic requirement outweighs many
    Minor ones; other categories contribute their count (baseline 1 each).
    Unclassified SAFE requirements use ``default_severity`` (flagged via
    ``profile.unclassified_safe``). Same severity input as the redundancy rule, but
    aggregated by SUM (overall safety burden) rather than MAX (worst hazard).
    """
    priorities = priorities or {}
    raw: Dict[str, float] = {}
    for obj, cats in objective_categories.items():
        mass = 0.0
        for c in cats:
            cat_mass = (
                _safe_mass(profile, default_severity)
                if c == "SAFE"
                else float(profile.count(c))
            )
            mass += cat_mass * priorities.get(c, 1.0)
        raw[obj] = mass
    return _normalise(raw, len(objective_categories))


def _weighted_sum(obj: Objectives, w: Dict[str, float]) -> float:
    return sum(w.get(k, 0.0) * v for k, v in obj.items())


def _chebyshev(obj: Objectives, w: Dict[str, float], ideal: Objectives) -> float:
    terms = [w.get(k, 0.0) * (ideal[k] - obj.get(k, 0.0)) for k in ideal]
    aug = 0.001 * sum(w.get(k, 0.0) * obj.get(k, 0.0) for k in ideal)
    return -max(terms) + aug


def recommend(
    front: Sequence[FrontMember],
    weights: Dict[str, float],
    method: str = "weighted_sum",
) -> FrontMember:
    """Pick one design from the Pareto front under a weight vector."""
    if not front:
        raise ValueError("empty front")
    if method == "weighted_sum":
        return max(front, key=lambda m: _weighted_sum(m[1], weights))
    if method == "chebyshev":
        names = front[0][1].keys()
        ideal = {k: max(m[1].get(k, 0.0) for m in front) for k in names}
        return max(front, key=lambda m: _chebyshev(m[1], weights, ideal))
    raise ValueError(f"unknown method {method!r}")


@dataclass
class SensitivityReport:
    objective_names: List[str]
    selection_frequency: Dict[str, float]
    nominal_label: str
    nominal_robustness: float
    n_samples: int
    labeler: object = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"weight-simplex sensitivity ({self.n_samples} samples over "
            f"{len(self.objective_names)} objectives):",
            f"  nominal recommendation = {self.nominal_label}  "
            f"robustness = {self.nominal_robustness:.1%} of weight space",
            "  selection frequency:",
        ]
        for lbl, f in sorted(self.selection_frequency.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {lbl:24s} {f:.1%}")
        return "\n".join(lines)


def _dirichlet(names: Sequence[str], rng: random.Random) -> Dict[str, float]:
    # Dirichlet(1,...,1) = normalise iid Exponential(1) -> uniform over the simplex
    g = {n: -math.log(rng.random()) for n in names}
    s = sum(g.values())
    return {n: v / s for n, v in g.items()}


def sensitivity(
    front: Sequence[FrontMember],
    nominal_weights: Dict[str, float],
    label_fn,
    n_samples: int = 2000,
    method: str = "weighted_sum",
    random_seed: Optional[int] = 0,
) -> SensitivityReport:
    """Sample the weight simplex and report recommendation robustness."""
    if not front:
        raise ValueError("empty front")
    names = list(front[0][1].keys())
    rng = random.Random(random_seed)

    counts: Dict[str, int] = {}
    for _ in range(n_samples):
        w = _dirichlet(names, rng)
        winner = recommend(front, w, method=method)
        lbl = label_fn(winner[0])
        counts[lbl] = counts.get(lbl, 0) + 1

    freq = {lbl: c / n_samples for lbl, c in counts.items()}
    nominal = label_fn(recommend(front, nominal_weights, method=method)[0])
    return SensitivityReport(
        objective_names=names,
        selection_frequency=freq,
        nominal_label=nominal,
        nominal_robustness=freq.get(nominal, 0.0),
        n_samples=n_samples,
    )
