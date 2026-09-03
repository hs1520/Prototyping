"""Weight-simplex sensitivity wired into the official variation-DSE path.

Per docs/DSE_REDESIGN.md §三-A the recommendation's weight-robustness is
measured. run_variation_dse calls sensitivity() with the same scalarisation its
recommender uses (method="chebyshev" explicitly, since sensitivity() defaults to
weighted_sum and the fallback bilevel path's call relies on that default), seeds
it from its own random_seed, and persists the report through pipeline state into
the run report dict. A 1-member front is computed and stored too, at robustness
100%.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.dse.physics_estimator import DesignInputs
from src.dse.variation_dse import VariationDSEResult, _state_label, run_variation_dse
from src.dse.weighting import recommend, sensitivity

# ── chebyshev-vs-weighted_sum discriminating construction ─────────────────
# Two Pareto members, three objectives, uniform weights: Y wins the weighted
# sum (larger total), X wins chebyshev (smaller worst weighted shortfall from
# the ideal point). Two members over two objectives cannot split the methods,
# hence the third objective.

_X = ({"vp": "X"}, {"a": 1.0, "b": 0.0, "c": 0.0})
_Y = ({"vp": "Y"}, {"a": 0.55, "b": 0.35, "c": 0.35})
_W = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_front_splits_scalarisations():
    assert recommend([_X, _Y], _W, method="chebyshev")[0] == _X[0]
    assert recommend([_X, _Y], _W, method="weighted_sum")[0] == _Y[0]


def test_nominal_follows_chebyshev():
    rep = sensitivity([_X, _Y], _W, label_fn=_state_label,
                      n_samples=64, method="chebyshev", random_seed=0)
    assert rep.nominal_label == _state_label(_X[0])
    # the default-method call (pipeline_adapter style) reports the other winner;
    # copying it into the official path is the regression
    rep_default = sensitivity([_X, _Y], _W, label_fn=_state_label,
                              n_samples=64, random_seed=0)
    assert rep_default.nominal_label == _state_label(_Y[0])


_MODEL = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def Octo4S :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 8.0;
        attribute batteryCells : Real = 4.0;
        attribute rotorRadiusM : Real = 0.1905;
    }
    part def Hexa6S :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 6.0;
        attribute batteryCells : Real = 6.0;
        attribute rotorRadiusM : Real = 0.2286;
    }
    part def Airframe {
        variation part liftArch : LiftIface {
            doc /* rationale: endurance vs catalogue availability; satisfies REQ-PERF-002 */
            variant part octo : Octo4S;
            variant part hexa : Hexa6S;
        }
    }
}"""

_REQS = ["REQ-PERF-002: endurance at least 20 minutes."]


def _model(text=_MODEL):
    return SimpleNamespace(metadata={"last_sysml_text": text})


def _front(self, iterations):
    return SimpleNamespace(members=[
        ({"liftArch": "octo"}, {"time_sat": 0.99, "cost_efficiency": 1.0}),
        ({"liftArch": "hexa"}, {"time_sat": 1.0, "cost_efficiency": 0.90}),
    ])


def test_official_path_chebyshev_seed(monkeypatch):
    monkeypatch.setattr("src.dse.variation_dse.MultiObjectiveMCTS.search", _front)
    captured = {}

    def spy(front, weights, **kwargs):
        captured["front_len"] = len(front)
        captured.update({k: v for k, v in kwargs.items() if k != "label_fn"})
        return sensitivity(front, weights, **kwargs)

    monkeypatch.setattr("src.dse.variation_dse.sensitivity", spy)
    res = run_variation_dse(
        _model(), requirements=_REQS,
        realizability=lambda di: True,
        random_seed=7,
        exhaustive_limit=0,
    )
    assert res is not None and len(res.pareto_front) == 2
    assert captured["front_len"] == 2
    assert captured["method"] == "chebyshev"
    assert captured["random_seed"] == 7
    assert captured["n_samples"] == 2000

    ws = res.weight_sensitivity
    assert ws is not None
    assert ws["method"] == "chebyshev"
    assert ws["n_samples"] == 2000
    assert sum(ws["selection_frequency"].values()) == pytest.approx(1.0)
    assert 0.0 < ws["nominal_robustness"] <= 1.0
    # nominal = chebyshev pick under the requirement weights (octo: worst weighted
    # shortfall 0.5*0.01 vs hexa's 0.5*0.10), and it is the final recommendation
    # here (no datasheet rank supplied)
    assert ws["nominal_label"] == "liftArch=octo"
    assert ws["nominal_matches_recommendation"] is True
    assert any("weight-simplex sensitivity" in n for n in res.notes)


def test_single_member_front_robust(monkeypatch):
    monkeypatch.setattr("src.dse.variation_dse.MultiObjectiveMCTS.search", _front)
    res = run_variation_dse(
        _model(), requirements=_REQS,
        realizability=lambda di: di.rotor_count == 6,
        exhaustive_limit=0,
    )
    assert res is not None and len(res.pareto_front) == 1
    ws = res.weight_sensitivity
    assert ws is not None
    assert ws["nominal_robustness"] == 1.0
    assert ws["selection_frequency"] == {ws["nominal_label"]: 1.0}
    assert ws["nominal_matches_recommendation"] is True


def test_datasheet_override_mismatch(monkeypatch):
    monkeypatch.setattr("src.dse.variation_dse.MultiObjectiveMCTS.search", _front)
    res = run_variation_dse(
        _model(), requirements=_REQS,
        realizability=lambda di: True,
        realization_rank=lambda di: 100.0 if di.rotor_count == 6 else 10.0,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_by == "datasheet"
    assert res.recommended_choices == {"liftArch": "hexa"}
    ws = res.weight_sensitivity
    # the weight-stage nominal is octo while the datasheet rank picked hexa; the
    # stored mismatch stops reports crediting the weights with a rank-driven
    # selection
    assert ws["nominal_label"] == "liftArch=octo"
    assert ws["nominal_matches_recommendation"] is False


def test_empty_front_stores_no_report(monkeypatch):
    monkeypatch.setattr("src.dse.variation_dse.MultiObjectiveMCTS.search", _front)
    res = run_variation_dse(
        _model(), requirements=_REQS,
        realizability=lambda di: False,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert res.weight_sensitivity is None


_SENTINEL = {
    "nominal_label": "liftArch=hexa",
    "nominal_robustness": 0.87,
    "selection_frequency": {"liftArch=hexa": 0.87, "liftArch=octo": 0.13},
    "n_samples": 2000,
    "method": "chebyshev",
    "nominal_matches_recommendation": True,
}


def test_report_persists_in_state(monkeypatch):
    from src.agents.orchestrator import Orchestrator
    from src.agents.pipeline_records import PipelineRuntimeState

    text = _MODEL

    class Model:
        name = "Drone"

        def __init__(self):
            self.metadata = {"last_sysml_text": text}

        def to_sysml_text(self):
            return self.metadata["last_sysml_text"]

    design = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)

    def fake_run_variation_dse(model, requirements=None, **kwargs):
        return VariationDSEResult(
            recommended_choices={"liftArch": "hexa"},
            concrete_model=text.replace("variation part liftArch", "part liftArch"),
            pareto_front=[({"liftArch": "hexa"}, {"time_sat": 1.0, "cost_efficiency": 1.0})],
            admitted_points=["liftArch"],
            rejected_points=[],
            real_quality={},
            pareto_designs=[(design, {"time_sat": 1.0, "cost_efficiency": 1.0})],
            recommended_design=design,
            recommended_realizable=True,
            realizable_front_count=1,
            recommended_by="datasheet",
            weight_sensitivity=dict(_SENTINEL),
        )

    monkeypatch.setattr(
        "src.dse.variation_dse.run_variation_dse", fake_run_variation_dse
    )
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    orch._explore_variations(Model(), _REQS, seed=0)
    assert orch.last_weight_sensitivity == _SENTINEL
    state = orch._runtime_board.latest_typed(
        "pipeline.runtime.state", PipelineRuntimeState
    )
    assert state.weight_sensitivity == _SENTINEL
