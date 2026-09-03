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


def test_weights_track_composition():
    safety = derive_weights({"SAFE": 6, "PERF": 2, "CONS": 1}, _OBJMAP)
    cost = derive_weights({"SAFE": 1, "PERF": 1, "CONS": 6}, _OBJMAP)
    assert safety["reliability"] > safety["cost_efficiency"]
    assert cost["cost_efficiency"] > cost["reliability"]
    assert safety["reliability"] > cost["reliability"]


def test_weights_normalised_priority():
    w = derive_weights({"SAFE": 2, "CONS": 2}, _OBJMAP)
    assert sum(w.values()) == pytest.approx(1.0)
    wp = derive_weights({"SAFE": 2, "CONS": 2}, _OBJMAP, priorities={"SAFE": 3.0})
    assert wp["reliability"] > w["reliability"]


def test_no_requirements_uniform():
    w = derive_weights({}, _OBJMAP)
    assert w["reliability"] == pytest.approx(0.5)
    assert w["cost_efficiency"] == pytest.approx(0.5)


_OBJMAP2 = {"reliability": ["SAFE"], "performance": ["PERF"]}


def test_severity_outweighs_count():
    prof = RequirementProfile.from_requirements(
        [_req("SAFE", 1, "Catastrophic")] + [_req("PERF", i) for i in range(1, 5)]
    )
    w = derive_weights_from_profile(prof, _OBJMAP2)
    assert w["reliability"] == pytest.approx(10 / 14, abs=1e-3)
    count_w = derive_weights({"SAFE": 1, "PERF": 4}, _OBJMAP2)
    assert w["reliability"] > count_w["reliability"]


def test_many_minor_stay_proportional():
    prof = RequirementProfile.from_requirements(
        [_req("SAFE", i, "Minor") for i in range(1, 31)] + [_req("PERF", i) for i in range(1, 5)]
    )
    w = derive_weights_from_profile(prof, _OBJMAP2)
    assert w["reliability"] == pytest.approx(30 / 34, abs=1e-3)


def test_catastrophic_outweighs_minors():
    perf = [_req("PERF", i) for i in range(1, 6)]
    cata = RequirementProfile.from_requirements([_req("SAFE", 1, "Catastrophic")] + perf)
    minors = RequirementProfile.from_requirements([_req("SAFE", i, "Minor") for i in range(1, 9)] + perf)
    objmap = {"reliability": ["SAFE"], "performance": ["PERF"]}
    w_cata = derive_weights_from_profile(cata, objmap)
    w_minor = derive_weights_from_profile(minors, objmap)
    assert w_cata["reliability"] == pytest.approx(10 / 15, abs=1e-3)
    assert w_minor["reliability"] == pytest.approx(8 / 13, abs=1e-3)
    assert w_cata["reliability"] > w_minor["reliability"]


def test_unclassified_default_importance():
    prof = RequirementProfile.from_requirements([_req("SAFE", 1)])
    w = derive_weights_from_profile(prof, {"reliability": ["SAFE"], "perf": ["PERF"]})
    assert w["reliability"] == pytest.approx(1.0)


def test_severity_weights_sum_to_one():
    prof = RequirementProfile.from_requirements(
        [_req("SAFE", 1, "Hazardous"), _req("PERF", 1), _req("CONS", 1)]
    )
    w = derive_weights_from_profile(
        prof, {"reliability": ["SAFE", "PERF"], "cost_efficiency": ["CONS"]}
    )
    assert sum(w.values()) == pytest.approx(1.0)


def test_recommend_follows_weights():
    rel_heavy = recommend(_FRONT, {"reliability": 0.95, "cost_efficiency": 0.05})
    cost_heavy = recommend(_FRONT, {"reliability": 0.05, "cost_efficiency": 0.95})
    assert _LABEL(rel_heavy[0]) == "triple+centralised"
    assert _LABEL(cost_heavy[0]) == "single+centralised"


def test_chebyshev_reaches_concave_point():
    # B is non-dominated but below the A–C line -> weighted sum never selects it
    front = [
        ({"id": "A"}, {"o1": 1.0, "o2": 0.0}),
        ({"id": "B"}, {"o1": 0.4, "o2": 0.4}),
        ({"id": "C"}, {"o1": 0.0, "o2": 1.0}),
    ]
    w = {"o1": 0.5, "o2": 0.5}
    assert recommend(front, w, method="weighted_sum")[0]["id"] != "B"
    assert recommend(front, w, method="chebyshev")[0]["id"] == "B"


def test_frequencies_form_distribution():
    w = derive_weights({"SAFE": 6, "PERF": 2, "CONS": 1}, _OBJMAP)
    rep = sensitivity(_FRONT, w, _LABEL, n_samples=3000, random_seed=1)
    assert sum(rep.selection_frequency.values()) == pytest.approx(1.0)
    assert 0.0 <= rep.nominal_robustness <= 1.0
    assert rep.selection_frequency["single+centralised"] > 0.4


def test_sensitivity_deterministic():
    w = {"reliability": 0.5, "cost_efficiency": 0.5}
    a = sensitivity(_FRONT, w, _LABEL, n_samples=1000, random_seed=7)
    b = sensitivity(_FRONT, w, _LABEL, n_samples=1000, random_seed=7)
    assert a.selection_frequency == b.selection_frequency
