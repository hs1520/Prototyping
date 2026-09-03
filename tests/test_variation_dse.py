from __future__ import annotations

from pathlib import Path
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


def test_admitted_points_concrete_model():
    res = run_variation_dse(_model(), iterations=40, random_seed=1)
    assert res is not None
    assert set(res.admitted_points) == {"propulsion", "battery"}
    assert res.rejected_points == ["unjustified"]
    assert set(res.recommended_choices) == {"propulsion", "battery"}
    assert not check_syntax(res.concrete_model).has_errors
    assert "variation part propulsion" not in res.concrete_model


def test_pareto_front_real_quality():
    res = run_variation_dse(_model(), iterations=40, random_seed=1)
    assert len(res.pareto_front) >= 1
    assert "safety_assurance" in res.real_quality
    assert "mcts_fidelity" not in res.real_quality


def test_rejected_point_flagged_in_notes():
    res = run_variation_dse(_model(), iterations=20, random_seed=1)
    assert any("unjustified" in n for n in res.notes)


def test_no_admissible_points_none():
    bare = """package P {
    part def A; part def B;
    part def Sub { variation part x { variant part a : A; variant part b : B; } }
}"""
    assert run_variation_dse(_model(bare), iterations=10) is None


def test_no_variation_returns_none():
    plain = "package P { part def A; part a : A; }"
    assert run_variation_dse(_model(plain), iterations=10) is None


def test_inner_bo_sizes_capacity():
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
    reqs = ["REQ-PERF-001: cruise speed at least 20 m/s", "REQ-PERF-002: endurance at least 20 minutes"]
    res = run_variation_dse(_model(m), requirements=reqs, iterations=30, random_seed=1)
    assert res.recommended_capacity_mah is not None
    assert 3000 <= res.recommended_capacity_mah <= 22000
    assert "batteryCapacityMah" in res.concrete_model


def test_write_back_bound_variant():
    from src.dse.variation_dse import _write_back_capacity
    # Two retained alternatives, only Chosen is bound. The capacity lands in Chosen
    # even though Loser declares batteryCells first; the old flat heuristic wrote it
    # into Loser, so the archived model asserted a capacity for an unselected design.
    m = (
        "package P {\n"
        "  part def Loser :> B { attribute batteryCells : Real = 4.0; }\n"
        "  part def Chosen :> B { attribute batteryCells : Real = 6.0; }\n"
        "  part power : Chosen;\n"
        "}"
    )
    out = _write_back_capacity(m, 12000.0, ["Chosen"])
    chosen_body = out.split("part def Chosen")[1].split("}")[0]
    loser_body = out.split("part def Loser")[1].split("}")[0]
    assert "batteryCapacityMah : Real = 12000.0" in chosen_body
    assert "batteryCapacityMah" not in loser_body
    m2 = (
        "package P {\n"
        "  part def OtherLoser :> B { attribute batteryCells : Real = 4.0; }\n"
        "  part def Frame :> B { attribute rotorCount : Real = 6.0; }\n"
        "  part frame : Frame;\n"
        "}"
    )
    out2 = _write_back_capacity(m2, 9000.0, ["Frame"])
    assert "batteryCapacityMah : Real = 9000.0" in out2.split("part def Frame")[1].split("}")[0]
    assert _write_back_capacity(m, 9000.0, []) == m


def test_resolved_model_is_committed():
    """Regression for the archived-model authority defect (run 33e359ca): retained
    Pareto alternatives and trade-study alt bindings do not leak into the committed
    design a reader resolves from the final model text.
    """
    from src.dse.domain_objective import committed_bindings, resolve_design_attributes

    m = (
        "package Drone {\n"
        "  part def PropulsionSystem;\n"
        "  part def Catalog_r4 :> PropulsionSystem {"
        " attribute rotorCount : Real = 4.0;"
        " attribute rotorRadiusM : Real = 0.1905;"
        " attribute batteryCells : Real = 4.0; }\n"
        "  part def Catalog_r6 :> PropulsionSystem {"
        " attribute rotorCount : Real = 6.0;"
        " attribute rotorRadiusM : Real = 0.2032;"
        " attribute batteryCells : Real = 6.0;"
        " attribute batteryCapacityMah : Real = 16000.0; }\n"
        "  part propulsionSystem : Catalog_r6;\n"
        "  analysis def DesignTradeStudy {\n"
        "    part alt0Design { part propulsionSystem : Catalog_r4; }\n"
        "    part alt1Design { part propulsionSystem : Catalog_r6; }\n"
        "  }\n"
        "}"
    )
    assert committed_bindings(m) == [("propulsionSystem", "Catalog_r6")]
    r = resolve_design_attributes(m)
    assert r.field_values["rotor_count"] == 6.0
    assert r.field_values["rotor_radius_m"] == 0.2032
    assert r.field_values["battery_cells"] == 6.0
    assert r.field_values["battery_capacity_mah"] == 16000.0


def test_weights_track_requirements():
    from src.dse.variation_dse import _recommendation_weights
    names = ["speed_sat", "time_sat", "cost_efficiency"]
    reqs = ["REQ-PERF-001: cruise speed at least 18 m/s",
            "REQ-PERF-002: endurance at least 30 minutes",
            "REQ-PERF-003: endurance at least 25 minutes"]
    w = _recommendation_weights(names, reqs)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["time_sat"] > w["speed_sat"]


_REALIZABILITY_MODEL = """package Drone {
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


def _front_for_realizability(self, iterations):
    return SimpleNamespace(members=[
        ({"liftArch": "octo"}, {"time_sat": 0.99, "cost_efficiency": 1.0}),
        ({"liftArch": "hexa"}, {"time_sat": 1.0, "cost_efficiency": 0.90}),
    ])


def _front_with_only_infeasible_realizable(self, iterations):
    return SimpleNamespace(members=[
        ({"liftArch": "octo"}, {"time_sat": 1.0, "cost_efficiency": 1.0}),
        ({"liftArch": "hexa"}, {"time_sat": 0.90, "cost_efficiency": 0.90}),
    ])


def test_prefers_realizable_subset(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: di.rotor_count == 6,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_choices == {"liftArch": "hexa"}
    assert res.recommended_design is not None
    assert res.recommended_design.rotor_count == 6
    assert res.recommended_realizable is True
    assert res.realizable_front_count == 1
    assert res.recommended_estimator_feasible is True
    assert any("official Pareto rebuilt from 1 constrained-feasible design" in n
               for n in res.notes)


def test_rank_reorders_realizable(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: True,
        realization_rank=lambda di: 100.0 if di.rotor_count == 6 else 10.0,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_choices == {"liftArch": "hexa"}
    assert res.recommended_design is not None
    assert res.recommended_design.rotor_count == 6
    assert res.recommended_realizable is True
    assert res.realizable_front_count == 2
    assert res.recommended_by == "datasheet"
    assert res.recommended_estimator_feasible is True
    assert any("datasheet rank inside constrained Pareto" in n for n in res.notes)


def test_rank_tie_uses_estimator(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: True,
        realization_rank=lambda di: 10.0,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_choices == {"liftArch": "octo"}
    assert res.recommended_by == "datasheet"
    assert any("rank tie broken by estimator" in n for n in res.notes)


def test_infeasible_member_no_bypass(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_with_only_infeasible_realizable,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: di.rotor_count == 6,
        realization_rank=lambda di: 100.0,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_choices == {}
    assert res.recommended_design is None
    assert res.recommended_realizable is False
    assert res.realizable_front_count == 0
    assert res.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert res.recommended_by == "none"
    assert res.recommended_estimator_feasible is None
    assert res.exploratory_choices == {"liftArch": "octo"}
    assert any("none is catalog mapping-compliant" in n for n in res.notes)


def test_no_realizable_no_recommendation(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    reqs = ["REQ-PERF-002: endurance at least 20 minutes."]
    baseline = run_variation_dse(
        _model(_REALIZABILITY_MODEL), requirements=reqs, exhaustive_limit=0
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=reqs,
        realizability=lambda di: False,
        exhaustive_limit=0,
    )
    assert res is not None and baseline is not None
    assert res.recommended_choices == {}
    assert res.recommended_design is None
    assert res.exploratory_choices == baseline.recommended_choices
    assert res.recommended_realizable is False
    assert res.realizable_front_count == 0
    assert res.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert res.recommended_by == "none"
    assert res.recommended_estimator_feasible is None
    assert any("none is catalog mapping-compliant" in n for n in res.notes)


def test_rank_no_unrealizable_fallback(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    reqs = ["REQ-PERF-002: endurance at least 20 minutes."]
    baseline = run_variation_dse(
        _model(_REALIZABILITY_MODEL), requirements=reqs, exhaustive_limit=0
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=reqs,
        realizability=lambda di: False,
        realization_rank=lambda di: 1.0,
        exhaustive_limit=0,
    )
    assert res is not None and baseline is not None
    assert res.recommended_choices == {}
    assert res.exploratory_choices == baseline.recommended_choices
    assert res.recommended_realizable is False
    assert res.realizable_front_count == 0
    assert res.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert res.recommended_by == "none"
    assert res.recommended_estimator_feasible is None


def test_phase8_gate_narrows_front(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: True,
        recommendability=lambda di: di.rotor_count == 6,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_choices == {"liftArch": "hexa"}
    assert res.realizable_front_count == 2
    assert res.recommendable_front_count == 1
    assert res.recommendation_status == "RECOMMENDED"


def test_no_phase8_not_recommendable(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: True,
        recommendability=lambda di: False,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert res.recommended_choices == {}
    assert res.realizable_front_count == 2
    assert res.recommendable_front_count == 0
    assert any("none closes Phase 8" in n for n in res.notes)


def test_constraints_before_pareto(monkeypatch):
    """A catalog/Phase8-feasible design dominated by an infeasible estimator winner
    reappears once constraints are applied to the complete evaluated set.
    """
    monkeypatch.setattr(
        "src.dse.variation_dse.objectives_from_design",
        lambda di, *_args, **_kwargs: (
            {"time_sat": 1.0, "cost_efficiency": 1.0}
            if di.rotor_count == 8 else
            {"time_sat": 1.0, "cost_efficiency": 0.8}
        ),
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        iterations=1,
        realizability=lambda di: di.rotor_count == 6,
        recommendability=lambda di: di.rotor_count == 6,
    )

    assert res is not None
    assert res.coverage_mode == "exhaustive"
    assert res.evaluated == res.search_space_size == 2
    assert [state for state, _ in res.exploratory_pareto_front] == [
        {"liftArch": "octo"}
    ]
    assert [state for state, _ in res.pareto_front] == [{"liftArch": "hexa"}]
    assert res.recommended_choices == {"liftArch": "hexa"}
    assert res.estimator_feasible_count == 2
    assert res.mapping_compliant_count == 1
    assert res.phase8_closable_count == 1
    assert res.constraint_feasible_count == 1
    assert any("evaluated=2 -> estimator=2 -> catalog=1 -> phase8=1" in n
               for n in res.notes)


def test_gate_includes_mtow():
    mapping_calls = []
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=[
            "REQ-PERF-002: endurance at least 20 minutes.",
            "REQ-CONS-001: MTOW shall be <= 1 kg.",
        ],
        realizability=lambda di: mapping_calls.append(di) or True,
        recommendability=lambda di: True,
        iterations=1,
    )

    assert res is not None
    assert res.estimator_feasible_count == 0
    assert mapping_calls == []
    assert res.pareto_front == []
    assert res.recommendation_status == "NO_RECOMMENDABLE_DESIGN"
    assert any("performance + MTOW gate" in n for n in res.notes)


def test_capacity_options_discrete_packs():
    capacities = [8000.0, 12000.0]
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=lambda di: True,
        recommendability=lambda di: True,
        capacity_options=lambda di: capacities,
        iterations=20,
    )
    assert res is not None
    assert res.pareto_designs
    assert all(d.battery_capacity_mah in capacities for d, _ in res.pareto_designs)
    assert any("used real catalog pack capacities" in n for n in res.notes)


def test_realizability_none_legacy(monkeypatch):
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_for_realizability,
    )
    res = run_variation_dse(
        _model(_REALIZABILITY_MODEL),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=None,
        exhaustive_limit=0,
    )
    assert res is not None
    assert res.recommended_choices == {"liftArch": "octo"}
    assert res.recommended_realizable is None
    assert res.realizable_front_count is None
    assert res.recommended_by is None
    assert res.recommended_estimator_feasible is True
    assert not any("realizable front members" in n or "front member realizable" in n
                   for n in res.notes)


def test_no_realization_import():
    dse_dir = Path(__file__).resolve().parents[1] / "src" / "dse"
    offenders = []
    for path in dse_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "src.realization" in text or "..realization" in text or " import realization" in text:
            offenders.append(path.name)
    assert offenders == []


def test_write_back_bodiless_def():
    from src.dse.variation_dse import _write_back_capacity
    m = "package P {\n  part def Frame_light :> Airframe;\n  part frame : Frame_light;\n}"
    out = _write_back_capacity(m, 12000.0, ["Frame_light"])
    assert "part def Frame_light :> Airframe { attribute batteryCapacityMah : Real = 12000.0; }" in out


def test_bindings_accept_multiplicity():
    from src.dse.domain_objective import committed_bindings
    m = "package P { part def R6 :> Base { attribute rotorCount : Real = 6.0; } part rotors : R6[1]; }"
    assert ("rotors", "R6") in committed_bindings(m)
