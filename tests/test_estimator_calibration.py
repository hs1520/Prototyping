"""F1 estimator calibration: catalog grid as the calibration set.

Pins: the physical fits (energy density, fom_eff), the clamp-exclusion honesty
rule, MAPE improvement on the trustworthy grid, calibration state hygiene
(explicit, restorable, legacy-identical when empty), and the search-only scope
in the orchestrator.
"""
from __future__ import annotations

import pytest

from src.dse.physics_estimator import (
    DesignInputs,
    calibrated,
    calibration_active,
    clear_calibration,
    endurance_min,
    set_calibration,
)
from src.realization.catalog import DEFAULT_CATALOG
from src.realization.estimator_calibration import (
    calibration_grid,
    catalog_rank_check,
    fit_from_catalog,
)

from .realization_fixtures import catalog, combo, frame, pack


def test_set_calibration_rejects_unknown_keys_and_context_restores():
    clear_calibration()
    with pytest.raises(ValueError):
        set_calibration(bogus_key=1.0)
    d = DesignInputs(1.0, 12000, 6, 4, 0.19, 0.0)
    base = endurance_min(d)
    with calibrated(fom=0.9, eta_drive=1.0):
        assert calibration_active() == {"fom": 0.9, "eta_drive": 1.0}
        assert endurance_min(d) > base  # better efficiency → longer endurance
    assert calibration_active() == {}
    assert endurance_min(d) == base  # bit-identical legacy behavior restored


def test_energy_density_fitted_from_pack_data():
    cat = catalog(packs=[
        pack("a", capacity=12000, mass=1200),   # 12·6·3.7/1.2 = 222 Wh/kg
        pack("b", capacity=8000, mass=800),     # 222 Wh/kg
        pack("c", capacity=16000, mass=1600),   # 222 Wh/kg
    ])
    cal = fit_from_catalog(cat)
    assert cal.energy_density_wh_kg == pytest.approx(222.0, rel=1e-3)


def test_fit_on_default_catalog_reduces_mape_and_excludes_clamped_points():
    cal = fit_from_catalog(DEFAULT_CATALOG)
    assert cal.n_points >= 10
    assert 0.3 < cal.fom_eff < 0.9
    assert cal.endurance_mape_after < cal.endurance_mape_before
    assert any("clamped" in n for n in cal.notes)
    # the fit set only contains interpolation-region points
    trusted = calibration_grid(DEFAULT_CATALOG, include_clamped=False)
    full = calibration_grid(DEFAULT_CATALOG, include_clamped=True)
    assert len(trusted) == cal.n_points < len(full)


def test_catalog_rank_check_has_usable_n_and_no_snap_ties():
    cal = fit_from_catalog(DEFAULT_CATALOG)
    rank = catalog_rank_check(DEFAULT_CATALOG, fit=cal)
    assert rank["n"] >= 10
    assert -1.0 <= rank["spearman_before"] <= 1.0
    # calibration is a near-monotone rescale — ordering must not degrade materially
    assert rank["spearman_after"] >= rank["spearman_before"] - 0.05


def test_experiment_b_hexa_moves_toward_datasheet_truth():
    from src.realization.closure import close_the_loop

    d = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: endurance at least 25 minutes."],
                         DEFAULT_CATALOG)
    ds = rep.chosen.metrics.endurance_min  # ≈29.9 min (experiment B)

    cal = fit_from_catalog(DEFAULT_CATALOG)
    clear_calibration()
    err_before = abs(endurance_min(d) - ds) / ds
    with calibrated(**cal.overrides()):
        err_after = abs(endurance_min(d) - ds) / ds
    assert err_after < err_before  # calibrated estimator is closer to datasheet
    assert err_after < 0.25


def test_orchestrator_applies_calibration_only_around_the_search(monkeypatch):
    from types import SimpleNamespace

    from src.agents.orchestrator import Orchestrator
    from src.dse.variation_dse import VariationDSEResult

    text = """package Drone {
        port def Sig;
        part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def Hexa6S :> LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def Octo4S :> LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def Airframe {
            variation part liftArch : LiftIface {
                doc /* rationale: catalogue-aware trade; satisfies REQ-PERF-002 */
                variant part hexa : Hexa6S;
                variant part octo : Octo4S;
            }
        }
    }"""
    model = SimpleNamespace(name="Drone", metadata={"last_sysml_text": text})
    design = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)
    captured = {}

    def fake_run_variation_dse(m, requirements=None, iterations=None, random_seed=0,
                               realizability=None, realization_rank=None):
        captured["during"] = calibration_active()
        return VariationDSEResult(
            recommended_choices={"liftArch": "hexa"},
            concrete_model=text.replace("variation part liftArch", "part liftArch"),
            pareto_front=[({"liftArch": "hexa"}, {"time_sat": 1.0, "cost_efficiency": 1.0})],
            admitted_points=["liftArch"],
            rejected_points=[],
            real_quality={},
            pareto_designs=[(design, {"time_sat": 1.0, "cost_efficiency": 1.0})],
            recommended_design=design,
        )

    monkeypatch.setattr("src.dse.variation_dse.run_variation_dse", fake_run_variation_dse)
    clear_calibration()
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    orch._explore_variations(model, ["REQ-PERF-002: endurance at least 20 minutes."], seed=0)

    assert captured["during"].get("fom", 0) > 0          # calibrated DURING the search
    assert captured["during"].get("eta_drive") == 1.0    # fom_eff absorbs drive efficiency
    assert calibration_active() == {}                    # restored AFTER the search
    assert orch.last_estimator_calibration is not None
    assert orch.last_estimator_calibration["n_points"] >= 10
    assert "rank_check" in orch.last_estimator_calibration

    # opt-out keeps the legacy estimator and records nothing
    orch2 = Orchestrator(llm=object(), use_variation_dse=True,
                         estimator_calibration=False)
    orch2._explore_variations(model, ["REQ-PERF-002: endurance at least 20 minutes."], seed=0)
    assert captured["during"] == {}
    assert orch2.last_estimator_calibration is None
