from src.agents.orchestrator import Orchestrator, _public_realization
from src.dse.design_space import DesignConfiguration, DesignSpace
from src.dse.physics_estimator import DesignInputs
from src.dse.variation_dse import VariationDSEResult


def test_realization_artifact_is_best_effort_and_public_shape_has_no_internal_report():
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


def test_orchestrator_constructor_accepts_realization_inject_flag():
    orch = Orchestrator(llm=object(), realization_inject=True)
    assert orch.realization_inject is True


def test_explore_phase8_returns_realization_and_can_inject(monkeypatch):
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

    def fake_refine(self, model, requirements, **kwargs):
        return model, 1.0, Sim()

    monkeypatch.setattr(Orchestrator, "_explore_variations", fake_explore_variations)
    monkeypatch.setattr(Orchestrator, "_iterative_refinement", fake_refine)
    monkeypatch.setattr(Orchestrator, "_print_final_sim", lambda self, sim: None)
    orch = Orchestrator(llm=object(), use_variation_dse=True, realization_inject=True)
    result = orch.explore({
        "model": Model(),
        "requirements": ["REQ-PERF-002: endurance at least 25 minutes."],
        "system_name": "Drone",
        "evaluation_history": [],
    })
    assert result["realization"]["verdict"] == "INFEASIBLE_REALIZATION"
    assert "UnrealizedDesign" in result["model_sysml"]


def test_variation_dse_receives_best_effort_realizability_predicate(monkeypatch):
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
                               random_seed=0, realizability=None):
        captured["realizability"] = realizability
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
    assert best.parameters == {"liftArch": "hexa"}
    assert len(front) == 1
    assert ds.parameter_by_name("liftArch") is not None
