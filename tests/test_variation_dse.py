"""Tests for DSE over LLM-declared variation points (the replace path)."""
from __future__ import annotations

from types import SimpleNamespace

from src.dse.variation_dse import run_variation_dse
from src.simulation.syntax_checker import check_syntax

_MODEL = """package Drone {
    port def Sig;
    part def Motor;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def QuadRotor :> LiftIface { part m : Motor[4]; }
    part def HexaRotor :> LiftIface { part m : Motor[6]; }
    part def BatteryIface { out port power : Sig; }
    part def SingleBattery :> BatteryIface;
    part def DualBattery :> BatteryIface { part cell : Motor[2]; }
    part def Airframe {
        variation part propulsion : LiftIface {
            doc /* rationale: lift vs endurance; satisfies REQ-PERF-002 */
            variant part quad : QuadRotor;
            variant part hexa : HexaRotor;
        }
        variation part battery : BatteryIface {
            doc /* rationale: endurance vs mass; satisfies REQ-SAFE-001 */
            variant part single : SingleBattery;
            variant part dual : DualBattery;
        }
        variation part unjustified { variant part a : QuadRotor; variant part b : HexaRotor; }
    }
}"""


def _model(text=_MODEL):
    return SimpleNamespace(metadata={"last_sysml_text": text})


def test_explores_admitted_points_and_recommends_concrete_model():
    res = run_variation_dse(_model(), iterations=40, random_seed=1)
    assert res is not None
    assert set(res.admitted_points) == {"propulsion", "battery"}
    assert res.rejected_points == ["unjustified"]
    # a choice is made for every admitted point
    assert set(res.recommended_choices) == {"propulsion", "battery"}
    # the recommended concrete model is valid and has the variations resolved
    assert not check_syntax(res.concrete_model).has_errors
    assert "variation part propulsion" not in res.concrete_model


def test_pareto_front_non_empty_and_real_quality_attached():
    res = run_variation_dse(_model(), iterations=40, random_seed=1)
    assert len(res.pareto_front) >= 1
    assert "safety_assurance" in res.real_quality
    assert "mcts_fidelity" not in res.real_quality


def test_rejected_point_flagged_in_notes():
    res = run_variation_dse(_model(), iterations=20, random_seed=1)
    assert any("unjustified" in n for n in res.notes)


def test_no_admissible_variation_points_returns_none():
    # variation point without rationale/requirement → not admissible → None
    bare = """package P {
    part def A; part def B;
    part def Sub { variation part x { variant part a : A; variant part b : B; } }
}"""
    assert run_variation_dse(_model(bare), iterations=10) is None


def test_model_without_variation_returns_none():
    plain = "package P { part def A; part a : A; }"
    assert run_variation_dse(_model(plain), iterations=10) is None


def test_bilevel_inner_bo_sizes_battery_capacity():
    """Bilevel: outer search picks the discrete architecture; the INNER BO sizes the
    battery capacity (not variant-declared) to meet endurance, and writes it back."""
    m = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def QuadRotor :> LiftIface { attribute batteryCells : Real = 6.0; attribute rotorRadiusM : Real = 0.16; attribute cruiseSpeedMps : Real = 16.0; }
    part def VtolWing  :> LiftIface { attribute batteryCells : Real = 4.0; attribute rotorRadiusM : Real = 0.13; attribute cruiseSpeedMps : Real = 26.0; }
    part def Airframe {
        variation part liftArch : LiftIface {
            doc /* rationale: speed vs endurance; satisfies REQ-PERF-001, REQ-PERF-002 */
            variant part quad : QuadRotor;
            variant part vtol : VtolWing;
        }
    }
}"""
    reqs = ["REQ-PERF-001: cruise speed at least 20 m/s", "REQ-PERF-002: endurance at least 30 minutes"]
    res = run_variation_dse(_model(m), requirements=reqs, iterations=30, random_seed=1)
    assert res.recommended_capacity_mah is not None
    assert 3000 <= res.recommended_capacity_mah <= 22000   # inner BO chose within bounds
    assert "batteryCapacityMah" in res.concrete_model      # inner-optimized value written back


def test_capacity_write_back_prefers_cells_then_falls_back():
    from src.dse.variation_dse import _write_back_capacity
    # power variant (has batteryCells) → capacity written there
    with_cells = "package P { part def V :> B { attribute batteryCells : Real = 6.0; } }"
    assert "batteryCapacityMah : Real = 12000.0" in _write_back_capacity(with_cells, 12000.0)
    # no cells anywhere → fall back to the first specialised variant part def
    no_cells = "package P { part def V :> B { attribute rotorCount : Real = 6.0; } }"
    assert "batteryCapacityMah : Real = 9000.0" in _write_back_capacity(no_cells, 9000.0)


def test_recommendation_weights_track_requirement_emphasis():
    from src.dse.variation_dse import _recommendation_weights
    names = ["speed_sat", "time_sat", "cost_efficiency"]
    reqs = ["REQ-PERF-001: cruise speed at least 18 m/s",
            "REQ-PERF-002: endurance at least 30 minutes",
            "REQ-PERF-003: endurance at least 25 minutes"]  # time: 2 targets, speed: 1
    w = _recommendation_weights(names, reqs)
    assert abs(sum(w.values()) - 1.0) < 1e-9        # normalised
    assert w["time_sat"] > w["speed_sat"]           # more endurance targets → more weight
