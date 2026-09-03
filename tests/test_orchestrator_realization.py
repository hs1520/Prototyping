from src.agents.orchestrator import Orchestrator, _public_realization
from src.agents.refinement import (
    ModelRevision,
    RefinedRevision,
    RefinementClosure,
)
from src.dse.design_space import DesignConfiguration, DesignSpace
from src.dse.physics_estimator import DesignInputs
from src.dse.variation_dse import VariationDSEResult


def test_public_artifact_hides_report():
    d = DesignInputs(2.5, 16000, 6, 4, 18 * 0.0254 / 2, 0.0)
    artifact = Orchestrator._realization_artifact(
        d,
        [(d, {})],
        ["REQ-PERF-002: flight endurance of at least 25 minutes."],
    )
    public = _public_realization(artifact)
    assert public["verdict"] == "INFEASIBLE_REALIZATION"
    assert "REALIZATION GAP" in public["summary"]
    assert "realization_model_sysml" in public
    assert "_report" not in public


def test_artifact_reports_deferred_scope():
    d = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)
    artifact = Orchestrator._realization_artifact(
        d,
        [(d, {})],
        [
            "REQ-PERF-002: endurance at least 25 minutes.",
            "REQ-CONS-003: maximum takeoff weight shall be below 25 kg.",
            "REQ-PERF-003: cruise speed at least 15 m/s.",
            "REQ-FUNC-001: operational range of at least 0.1 km.",
        ],
    )
    public = _public_realization(artifact)
    assert public["verdict"] in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert "datasheet closure families" in public["summary"]
    assert "forward_flight families" in public["summary"]
    assert public["forward_flight_ok"] is not None
    assert all("scope" in v for v in public["per_requirement"])
    assert {v["family"] for v in public["per_requirement"] if v["scope"] == "forward_flight"} == {
        "range",
        "speed",
    }


def test_artifact_persists_derating():
    d = DesignInputs(1.5, 30000, 4, 6, 17 * 0.0254 / 2, 0.0)
    artifact = Orchestrator._realization_artifact(
        d,
        [],
        [
            "REQ-PERF-002: minimum endurance of 25 minutes with maximum rated payload",
            "REQ-CONS-003: maximum take-off mass shall not exceed 8.0 kg",
        ],
    )
    public = _public_realization(artifact)
    chosen = public["chosen"]
    assert public["verdict"] == "CLOSED"
    assert chosen["integration_mass_g"] == 500.0
    assert chosen["pack_nominal_voltage_v"] == 14.4
    assert chosen["motor_curve_voltage_v"] == 14.8
    assert chosen["voltage_ratio"] < 1.0
    assert chosen["derated_max_thrust_per_motor_g"] < 3000.0
    assert "packNominalVoltageV" in public["realization_model_sysml"]
    assert "realizedIntegration" in public["realization_model_sysml"]


def test_constructor_accepts_inject_flag():
    orch = Orchestrator(llm=object(), realization_inject=True)
    assert orch.realization_inject is True


def test_phase8_returns_and_injects(monkeypatch):
    class Model:
        def __init__(self):
            self.metadata = {"last_sysml_text": "package Drone { part def A; }"}
            self.part_definitions = []

        def to_sysml_text(self):
            return self.metadata["last_sysml_text"]

        def get_summary(self):
            return {}

    class Sim:
        reachability_score = 1.0
        scenario_results = []

        def passed_scenarios(self):
            return []

        def failed_scenarios(self):
            return []

    design = DesignInputs(2.5, 16000, 6, 4, 18 * 0.0254 / 2, 0.0)

    def fake_explore_variations(self, model, requirements, seed):
        self.last_recommended_design = design
        self.last_pareto_designs = [(design, {})]
        self.last_recommended_bindings = {}
        ds = DesignSpace("fake")
        cfg = DesignConfiguration("best", parameters={"x": 1}, scores={"s": 1.0})
        ds.add_configuration(cfg)
        return ds, cfg, [cfg]

    def fake_refine(self, request):
        return RefinedRevision(
            revision=ModelRevision.capture(request.base.materialize()),
            score=1.0,
            requirements=request.requirements,
            dse_best_config=request.dse_best_config,
            _simulation=Sim(),
        )

    monkeypatch.setattr(Orchestrator, "_explore_variations", fake_explore_variations)
    monkeypatch.setattr(RefinementClosure, "refine", fake_refine)
    monkeypatch.setattr(Orchestrator, "_print_final_sim", lambda self, sim: None)
    # Keep Phase 9 explicit so this test documents its Phase 8-only scope.
    # high-fidelity runner, which would otherwise launch native SITL/Gazebo.
    orch = Orchestrator(llm=object(), use_variation_dse=True, realization_inject=True,
                        phase9_hifi=None)
    result = orch.explore({
        "model": Model(),
        "requirements": ["REQ-PERF-002: endurance at least 25 minutes."],
        "system_name": "Drone",
        "evaluation_history": [],
    })
    assert result["realization"]["verdict"] == "INFEASIBLE_REALIZATION"
    assert "UnrealizedDesign" in result["model_sysml"]
    # run report carries the weight-sensitivity slot (None here - the faked
    # exploration never ran the DSE), so the field reaches realization_run.json
    assert "weight_sensitivity" in result


def test_dse_receives_realizability(monkeypatch):
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

    class Model:
        name = "Drone"

        def __init__(self):
            self.metadata = {"last_sysml_text": text}

        def to_sysml_text(self):
            return self.metadata["last_sysml_text"]

    captured = {}
    design = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)

    def fake_run_variation_dse(model, requirements=None, iterations=None,
                               random_seed=0, realizability=None, realization_rank=None,
                               recommendability=None, capacity_options=None):
        captured["realizability"] = realizability
        captured["realization_rank"] = realization_rank
        captured["recommendability"] = recommendability
        captured["capacity_options"] = capacity_options
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
        )

    monkeypatch.setattr(
        "src.dse.variation_dse.run_variation_dse",
        fake_run_variation_dse,
    )
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    ds, best, front = orch._explore_variations(
        Model(),
        ["REQ-PERF-002: flight endurance of at least 25 minutes."],
        seed=0,
    )
    assert callable(captured["realizability"])
    assert captured["realizability"](design) is True
    assert callable(captured["realization_rank"])
    assert captured["realization_rank"](design) > 0.0
    assert callable(captured["recommendability"])
    assert captured["recommendability"](design) is True
    assert callable(captured["capacity_options"])
    assert captured["capacity_options"](design)
    assert best.parameters == {"liftArch": "hexa"}
    assert len(front) == 1
    assert ds.parameter_by_name("liftArch") is not None
    assert orch.last_recommended_realizable is True
    assert orch.last_realizable_front_count == 1
    assert orch.last_recommended_by == "datasheet"


def test_no_recommendation_restores_model(monkeypatch):
    base = """package Drone {
        port def Sig;
        part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def Airframe;
    }"""
    exploration = """package Drone {
        port def Sig;
        part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def A :> LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def B :> LiftIface { in port cmd : Sig; out port thrust : Sig; }
        part def Airframe {
            variation part liftArch : LiftIface {
                doc /* rationale: exploratory trade; satisfies REQ-PERF-002 */
                variant part a : A;
                variant part b : B;
            }
        }
    }"""

    class Model:
        name = "Drone"

        def __init__(self):
            self.metadata = {"last_sysml_text": base}

        def to_sysml_text(self):
            return self.metadata["last_sysml_text"]

    def introduce(self, model, requirements):  # noqa: ARG001
        model.metadata["last_sysml_text"] = exploration
        return model

    def no_recommendation(*args, **kwargs):
        return VariationDSEResult(
            recommended_choices={},
            concrete_model=exploration,
            pareto_front=[({"liftArch": "a"}, {"time_sat": 0.5, "cost_efficiency": 1.0})],
            admitted_points=["liftArch"],
            rejected_points=[],
            real_quality={},
            recommendation_status="NO_RECOMMENDABLE_DESIGN",
            exploratory_choices={"liftArch": "a"},
        )

    monkeypatch.setattr(Orchestrator, "_introduce_variations", introduce)
    monkeypatch.setattr("src.dse.variation_dse.run_variation_dse", no_recommendation)
    model = Model()
    orch = Orchestrator(
        llm=object(), use_variation_dse=True, estimator_calibration=False
    )

    _, best, _ = orch._explore_variations(
        model, ["REQ-PERF-002: endurance at least 20 minutes."], seed=0
    )

    assert best.name == "no_recommendable_design"
    assert best.parameters == {}
    assert model.metadata["last_sysml_text"] == base
    assert "variation part" not in model.metadata["last_sysml_text"]
