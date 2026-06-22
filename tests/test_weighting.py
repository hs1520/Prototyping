"""Tests for requirement-traceable weighting + weight-simplex sensitivity (pain A)."""
from __future__ import annotations

import pytest

from src.dse.requirements_profile import RequirementProfile
from src.dse.weighting import (
    derive_weights,
    derive_weights_from_profile,
    recommend,
    sensitivity,
)

_OBJMAP = {"reliability": ["SAFE", "PERF"], "cost_efficiency": ["CONS"]}


def _req(cat, n, sev=None):
    tag = f" [SEV:{sev}]" if sev else ""
    return f"REQ-{cat}-{n:03d}: The system shall do thing {n}.{tag}"

_FRONT = [
    ({"arbitration": "single", "topology": "centralised"}, {"reliability": 0.85, "cost_efficiency": 1.00}),
    ({"arbitration": "dual", "topology": "centralised"}, {"reliability": 0.977, "cost_efficiency": 0.75}),
    ({"arbitration": "triple", "topology": "centralised"}, {"reliability": 0.997, "cost_efficiency": 0.50}),
]
_LABEL = lambda s: f"{s['arbitration']}+{s['topology']}"


# ── traceable weights ──────────────────────────────────────────────────────

def test_weights_track_requirement_composition():
    safety = derive_weights({"SAFE": 6, "PERF": 2, "CONS": 1}, _OBJMAP)
    cost = derive_weights({"SAFE": 1, "PERF": 1, "CONS": 6}, _OBJMAP)
    assert safety["reliability"] > safety["cost_efficiency"]
    assert cost["cost_efficiency"] > cost["reliability"]
    # more SAFE reqs => strictly higher reliability weight (traceable, not fiat)
    assert safety["reliability"] > cost["reliability"]


def test_weights_normalised_and_priority_applies():
    w = derive_weights({"SAFE": 2, "CONS": 2}, _OBJMAP)
    assert sum(w.values()) == pytest.approx(1.0)
    # priority on SAFE shifts mass to reliability
    wp = derive_weights({"SAFE": 2, "CONS": 2}, _OBJMAP, priorities={"SAFE": 3.0})
    assert wp["reliability"] > w["reliability"]


def test_no_requirements_falls_back_to_uniform():
    w = derive_weights({}, _OBJMAP)
    assert w["reliability"] == pytest.approx(0.5)
    assert w["cost_efficiency"] == pytest.approx(0.5)


# ── severity-weighted weights (the principled upgrade) ──────────────────────

_OBJMAP2 = {"reliability": ["SAFE"], "performance": ["PERF"]}


def test_severity_mass_outweighs_count_for_catastrophic():
    """1 Catastrophic SAFE + 4 PERF → reliability ≈ 10/(10+4) = 0.714,
    whereas count-based would give only 1/(1+4) = 0.2."""
    prof = RequirementProfile.from_requirements(
        [_req("SAFE", 1, "Catastrophic")] + [_req("PERF", i) for i in range(1, 5)]
    )
    w = derive_weights_from_profile(prof, _OBJMAP2)
    assert w["reliability"] == pytest.approx(10 / 14, abs=1e-3)
    # strictly higher than the count-based weight on the same requirements
    count_w = derive_weights({"SAFE": 1, "PERF": 4}, _OBJMAP2)
    assert w["reliability"] > count_w["reliability"]


def test_many_minor_safe_accumulate_but_stay_proportional():
    prof = RequirementProfile.from_requirements(
        [_req("SAFE", i, "Minor") for i in range(1, 31)] + [_req("PERF", i) for i in range(1, 5)]
    )
    w = derive_weights_from_profile(prof, _OBJMAP2)
    # 30 minor (mass 30) vs 4 perf → 30/34
    assert w["reliability"] == pytest.approx(30 / 34, abs=1e-3)


def test_one_catastrophic_outweighs_many_minor_in_safety_mass():
    # against the SAME competing performance mass, one catastrophic (10) outweighs
    # eight minor (8) → higher reliability weight
    perf = [_req("PERF", i) for i in range(1, 6)]  # fixed mass 5 in both
    cata = RequirementProfile.from_requirements([_req("SAFE", 1, "Catastrophic")] + perf)
    minors = RequirementProfile.from_requirements([_req("SAFE", i, "Minor") for i in range(1, 9)] + perf)
    objmap = {"reliability": ["SAFE"], "performance": ["PERF"]}
    w_cata = derive_weights_from_profile(cata, objmap)
    w_minor = derive_weights_from_profile(minors, objmap)
    assert w_cata["reliability"] == pytest.approx(10 / 15, abs=1e-3)   # 10/(10+5)
    assert w_minor["reliability"] == pytest.approx(8 / 13, abs=1e-3)   # 8/(8+5)
    assert w_cata["reliability"] > w_minor["reliability"]


def test_unclassified_safe_uses_default_importance():
    prof = RequirementProfile.from_requirements([_req("SAFE", 1)])  # no tag → MAJOR=3
    w = derive_weights_from_profile(prof, {"reliability": ["SAFE"], "perf": ["PERF"]})
    assert w["reliability"] == pytest.approx(1.0)  # only SAFE mass present


def test_severity_weights_sum_to_one():
    prof = RequirementProfile.from_requirements(
        [_req("SAFE", 1, "Hazardous"), _req("PERF", 1), _req("CONS", 1)]
    )
    w = derive_weights_from_profile(
        prof, {"reliability": ["SAFE", "PERF"], "cost_efficiency": ["CONS"]}
    )
    assert sum(w.values()) == pytest.approx(1.0)


# ── recommendation ─────────────────────────────────────────────────────────

def test_recommend_follows_weights():
    rel_heavy = recommend(_FRONT, {"reliability": 0.95, "cost_efficiency": 0.05})
    cost_heavy = recommend(_FRONT, {"reliability": 0.05, "cost_efficiency": 0.95})
    assert _LABEL(rel_heavy[0]) == "triple+centralised"
    assert _LABEL(cost_heavy[0]) == "single+centralised"


def test_chebyshev_reaches_concave_point_weighted_sum_misses():
    # B is non-dominated but below the A–C line → weighted sum never selects it
    front = [
        ({"id": "A"}, {"o1": 1.0, "o2": 0.0}),
        ({"id": "B"}, {"o1": 0.4, "o2": 0.4}),
        ({"id": "C"}, {"o1": 0.0, "o2": 1.0}),
    ]
    w = {"o1": 0.5, "o2": 0.5}
    assert recommend(front, w, method="weighted_sum")[0]["id"] != "B"
    assert recommend(front, w, method="chebyshev")[0]["id"] == "B"


# ── sensitivity ────────────────────────────────────────────────────────────

def test_sensitivity_frequencies_form_distribution():
    w = derive_weights({"SAFE": 6, "PERF": 2, "CONS": 1}, _OBJMAP)
    rep = sensitivity(_FRONT, w, _LABEL, n_samples=3000, random_seed=1)
    assert sum(rep.selection_frequency.values()) == pytest.approx(1.0)
    assert 0.0 <= rep.nominal_robustness <= 1.0
    # the cheap extreme (single) wins a large share of the simplex → robust region exists
    assert rep.selection_frequency["single+centralised"] > 0.4


def test_sensitivity_is_deterministic_under_seed():
    w = {"reliability": 0.5, "cost_efficiency": 0.5}
    a = sensitivity(_FRONT, w, _LABEL, n_samples=1000, random_seed=7)
    b = sensitivity(_FRONT, w, _LABEL, n_samples=1000, random_seed=7)
    assert a.selection_frequency == b.selection_frequency
