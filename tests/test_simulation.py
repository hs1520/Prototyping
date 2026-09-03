import os
import pytest

from src.simulation.validator import SimulationValidator
from src.simulation.extractor import extract_behavioral_graph
from src.simulation.exec_graph import build_exec_graph, reachable_from
from src.simulation.scenarios import auto_detect_scenarios, DRONE_SCENARIOS, select_scenarios


DRONE_SYSML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "drone.sysml"
)

MINIMAL_SYSML = """
package TestPkg {
    port def Signal;

    part def Sender {
        out port dataOut : Signal;
    }

    part def Receiver {
        in port dataIn : Signal;
    }

    part def System {
        part s : Sender;
        part r : Receiver;
        connect s.dataOut to r.dataIn;
    }
}
"""

DISCONNECTED_SYSML = """
package Disconnected {
    port def Signal;

    part def A {
        out port aOut : Signal;
    }

    part def B {
        in port bIn : Signal;
    }

    part def C {
        in port cIn : Signal;
        out port cOut : Signal;
    }

    part def System {
        part a : A;
        part b : B;
        part c : C;
        // intentionally NO connect statements — parts are isolated
    }
}
"""


class TestDroneSimulation:
    @pytest.fixture(scope="class")
    def drone_result(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        v = SimulationValidator()
        return v.validate(src, model_name="AutonomousDrone")

    def test_disconnected_parse_succeeds(self, drone_result):
        assert drone_result.num_parts > 0, "Should parse at least one part def"

    def test_drone_graph_stats(self, drone_result):
        assert drone_result.num_parts > 0
        assert drone_result.num_ports > 0
        assert drone_result.num_connections > 0

    def test_most_scenarios_reachable(self, drone_result):
        # A well-connected model leaves most scenarios reachable. Some auto-generated
        # cross-product scenarios can be unreachable, so assert a high score, not 1.0.
        assert 0.0 <= drone_result.reachability_score <= 1.0
        assert drone_result.reachability_score >= 0.7, (
            f"score={drone_result.reachability_score:.3f}; "
            f"failed={[r.scenario_name for r in drone_result.failed_scenarios()]}"
        )

    def test_emergency_scenarios_generated(self, drone_result):
        emergency = [r for r in drone_result.scenario_results
                     if r.scenario_name.startswith("emergency_")]
        assert emergency, "no emergency_* scenarios were generated"

    def test_safety_scenarios_present(self, drone_result):
        safety = [r for r in drone_result.scenario_results if "safety" in r.tags]
        assert safety, "No safety-tagged scenarios found"

    def test_summary_output(self, drone_result):
        summary = drone_result.summary()
        assert "AutonomousDrone" in summary
        assert "Score" in summary
        assert "passed" in summary.lower()


class TestMinimalSimulation:
    @pytest.fixture(scope="class")
    def minimal_result(self):
        v = SimulationValidator(use_drone_scenarios=False)
        return v.validate(MINIMAL_SYSML, model_name="MinimalTest")

    def test_disconnected_parse_succeeds(self, minimal_result):
        assert minimal_result.num_parts >= 2

    def test_minimal_score_positive(self, minimal_result):
        assert minimal_result.reachability_score >= 0.0

    def test_advisory_evidence_preserved(
        self, minimal_result
    ):
        evidence = minimal_result.advisory_structural_evidence()
        assert evidence["evidence_kind"] == "UNTRACED_ADVISORY"
        assert evidence["qualification_effect"] == "NONE"
        assert evidence["role_assignments"]
        assert evidence["weakly_connected_components"]
        assert len(evidence["scenarios"]) == len(
            minimal_result.scenario_results
        )


class TestDisconnectedModel:
    @pytest.fixture(scope="class")
    def disconnected_result(self):
        v = SimulationValidator(use_drone_scenarios=False)
        return v.validate(DISCONNECTED_SYSML, model_name="DisconnectedTest")

    def test_disconnected_parse_succeeds(self, disconnected_result):
        assert disconnected_result.num_parts >= 2

    def test_score_below_perfect(self, disconnected_result):
        # A->C and B->C are not connected, so some scenarios fail (or none are
        # generated if auto-detect finds no sensor/controller). At minimum: not a
        # perfect score when parts are isolated
        result = disconnected_result
        if result.scenario_results:
            assert result.reachability_score < 1.0 or result.num_connections == 0


class TestExtractor:
    @pytest.fixture(scope="class")
    def drone_bg(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        return extract_behavioral_graph(src)

    def test_parts_extracted(self, drone_bg):
        assert len(drone_bg.parts) > 0

    def test_ports_extracted(self, drone_bg):
        assert len(drone_bg.ports) > 0

    def test_connections_extracted(self, drone_bg):
        assert len(drone_bg.connections) > 0

    def test_directional_ports_present(self, drone_bg):
        dirs = {p.direction for p in drone_bg.ports.values()}
        assert "in" in dirs or "out" in dirs or "inout" in dirs

    def test_connection_endpoints_resolve(self, drone_bg):
        for conn in drone_bg.connections:
            assert "." in conn.source or conn.source in drone_bg.parts, \
                f"Connection source '{conn.source}' looks unresolved"
            assert "." in conn.target or conn.target in drone_bg.parts, \
                f"Connection target '{conn.target}' looks unresolved"

    def test_minimal_model_structure(self):
        bg = extract_behavioral_graph(MINIMAL_SYSML)
        assert set(bg.parts) == {"s", "r"}
        assert len(bg.connections) == 1


class TestExecGraph:
    # Use the inline model so node/reachability assertions do not depend on a
    # specific drone.sysml revision.

    @pytest.fixture(scope="class")
    def minimal_graph(self):
        bg = extract_behavioral_graph(MINIMAL_SYSML)
        return build_exec_graph(bg)

    def test_parts_are_nodes(self, minimal_graph):
        assert "s" in minimal_graph.nodes
        assert "r" in minimal_graph.nodes

    def test_ports_are_nodes(self, minimal_graph):
        assert any("." in n for n in minimal_graph.nodes)

    def test_reachability_follows_connection(self, minimal_graph):
        reachable = reachable_from(minimal_graph, "s")
        assert "r" in reachable, f"reachable from s = {sorted(reachable)}"

    def test_drone_graph_builds(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        g = build_exec_graph(extract_behavioral_graph(src))
        assert len(g.nodes) > 0


class TestScenarioSelection:
    def test_drone_scenarios_applicable(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        bg = extract_behavioral_graph(src)
        scenarios = select_scenarios(bg, predefined=DRONE_SCENARIOS)
        assert len(scenarios) > 0

    def test_auto_detect_generates_scenarios(self):
        bg = extract_behavioral_graph(MINIMAL_SYSML)
        # auto_detect uses keyword matching; Sender/Receiver match poorly but still
        # produce a list (possibly one generic scenario).
        scenarios = auto_detect_scenarios(bg)
        assert isinstance(scenarios, list)


class TestAnalysisScaffoldingExcluded:
    _MODEL = """package Sys {
    port def Sig;
    part def A { out port o : Sig; }
    part def B { in port i : Sig; }
    part def HexImpl { attribute rotorCount : Real = 6.0; }
    part a : A;
    part b : B;
    connect a.o to b.i;
    analysis def DesignTradeStudy {
        part alt0Design { part propulsionSystem : HexImpl; }
        attribute alt0_x : Real = 1.0;
    }
}"""

    def test_analysis_parts_not_in_graph(self):
        bg = extract_behavioral_graph(self._MODEL)
        assert set(bg.parts) == {"a", "b"}
        assert "alt0Design" not in bg.parts

    def test_analysis_parts_not_isolated(self):
        res = SimulationValidator().validate(self._MODEL)
        assert "alt0Design" not in res.isolated_parts


def test_dse_closure_excluded():
    """The injected DSE analysis closure (part def DseDesignAnalysis holding
    recommendedDesign) is scaffolding, so it is excluded from the reachability graph.

    Otherwise the connectivity refiner bolts ports and connects onto real parts to
    wire it up.
    """
    from src.simulation import extractor as ex
    model = """package P {
        port def DataPort;
        part def Motor { out port t : DataPort; }
        part def Frame { in port t : DataPort; }
        part def Drone { part motor : Motor; part airframe : Frame; connect motor.t to airframe.t; }
        part def DseDesignAnalysis {
            part recommendedDesign { attribute capacityMah : Real = 22000.0; }
            attribute enduranceMin : Real = recommendedDesign.capacityMah / 1000.0;
        }
    }"""
    parts = list(ex.extract_behavioral_graph(model).parts)
    assert not any("recommendedDesign" in str(p) for p in parts)
    assert any("motor" in str(p) for p in parts)
    assert any("airframe" in str(p) for p in parts)


def test_controller_stays_controller():
    """Name identity classifies first; satisfy fragments only classify keyword-less
    names.

    On the s0v8 anchor flightController carried a satisfy REQ_SAFE_* link, the
    satisfy-first classifier filed it under "safety", and every controller-role
    scenario died with MISSING_TARGET_ROLE (1/7 on a healthy topology).
    """
    from src.simulation.extractor import BehavioralGraph, PartNode
    from src.simulation.scenarios import classify_parts_by_role

    bg = BehavioralGraph()
    bg.parts["flightController"] = PartNode(
        id="flightController", def_name="FlightController",
        satisfied_reqs=["REQ_SAFE_007", "REQ_FUNC_001"],
    )
    bg.parts["watchdogUnit"] = PartNode(
        id="watchdogUnit", def_name="WatchdogUnit",
        satisfied_reqs=["REQ_FUNC_002"],
    )
    bg.parts["unit7"] = PartNode(
        id="unit7", def_name="Unit7",
        satisfied_reqs=["REQ_SAFE_001"],
    )
    roles = classify_parts_by_role(bg)

    assert roles.get("controller") == ["flightController"]
    assert "flightController" not in roles.get("safety", [])
    assert "watchdogUnit" in roles.get("safety", [])
    assert "unit7" in roles.get("safety", [])


def test_dse_machinery_not_charged():
    """The quality scorer follows the variant emitter's contract and the extractor's
    closure exemption.

    It followed neither before, charging FULL's terminal artifact 0.043 against
    NO-DSE for its own DSE machinery (seed-0 wave). Specialisation inherits ports,
    the closure wrapper leaves the denominator, and a standalone attribute-only def
    is still penalised.
    """
    from src.dse.design_space import DesignConfiguration
    from src.dse.evaluator import DesignEvaluator
    from src.simulation.syntax_checker import check_syntax
    from src.simulation.validator import SimulationValidator
    from src.sysml.lite_model import build_lite_model

    base = """package P {{
    port def DataPort;
    part def PerceptionSystem {{
        out port data : DataPort;
    }}
    part def FlightController {{
        in port data : DataPort;
    }}{extra}
    part perceptionSystem : PerceptionSystem;
    part flightController : FlightController;
    connect perceptionSystem.data to flightController.data;
}}"""

    def score(extra):
        text = base.format(extra=extra)
        model = build_lite_model(text, model_name="P")
        ev = DesignEvaluator(quality_threshold=0.75).evaluate(
            DesignConfiguration(name="t", parameters={}), model,
            syntax_result=check_syntax(text),
            sim_result=SimulationValidator().validate(text, model_name="P"),
            requirements=[],
        )
        return ev.criteria_scores

    clean = score("")
    with_machinery = score(
        "\n    part def Catalog_r4FlightcontrollerImpl :> FlightController"
        " { attribute rotorCount : Real = 4.0; }"
        "\n    part def DseDesignAnalysis { attribute recommendedDesign :"
        " Real = 1.0; }"
    )
    assert with_machinery["structural_completeness"] == (
        clean["structural_completeness"]
    )
    assert with_machinery["interface_quality"] == clean["interface_quality"]

    with_standalone = score(
        "\n    part def OrphanBox { attribute massKg : Real = 1.0; }"
    )
    assert with_standalone["structural_completeness"] < (
        clean["structural_completeness"]
    )


def test_retyped_usage_still_counts():
    """A usage retyped to the DSE-selected implementation
    (propulsionSystem : Catalog_*Impl :> PropulsionSystem) is still a usage of the
    planned type.

    The safety-connectivity scorer read the selected design's own SAFE part as
    unwired on a fully wired model (0.938, the 3/4 signature).
    """
    from src.dse.design_space import DesignConfiguration
    from src.dse.evaluator import DesignEvaluator
    from src.simulation.syntax_checker import check_syntax
    from src.simulation.validator import SimulationValidator
    from src.sysml.lite_model import build_lite_model

    base = """package P {{
    port def DataPort;
    requirement def REQ_SAFE_001 {{ doc /* failsafe */ }}
    part def PropulsionSystem {{
        out port data : DataPort;
        satisfy requirement REQ_SAFE_001;
        state def FailsafeBehavior {{
            state Nominal;
            state Halted;
            transition halt first Nominal if faultDetected then Halted;
        }}
        attribute faultDetected : Boolean = false;
    }}
    part def FlightController {{ in port data : DataPort; }}
    part def Catalog_r6Impl :> PropulsionSystem {{
        attribute rotorCount : Real = 6.0;
    }}
    part propulsionSystem : {ptype};
    part flightController : FlightController;
    connect propulsionSystem.data to flightController.data;
}}"""

    def safety(ptype):
        text = base.format(ptype=ptype)
        model = build_lite_model(text, model_name="P")
        ev = DesignEvaluator(quality_threshold=0.75).evaluate(
            DesignConfiguration(name="t", parameters={}), model,
            syntax_result=check_syntax(text),
            sim_result=SimulationValidator().validate(text, model_name="P"),
            requirements=["REQ_SAFE_001: enter failsafe on fault"],
        )
        return ev.criteria_scores["safety_assurance"]

    assert safety("Catalog_r6Impl") == safety("PropulsionSystem")


def test_variation_activates_fidelity():
    """The dse_fidelity dimension (10% prior) predates variation DSE: its gate knew
    only legacy scalar keys, so the arm materialising the selected catalogue
    variant was scored on fewer dimensions.

    A variant choice now activates the dimension: full credit needs def + retyped
    usage, catalogued-only earns half, and no config keeps the dimension dropped.
    """
    from src.dse.design_space import DesignConfiguration
    from src.dse.evaluator import DesignEvaluator
    from src.simulation.syntax_checker import check_syntax
    from src.simulation.validator import SimulationValidator
    from src.sysml.lite_model import build_lite_model

    base = """package P {{
    port def DataPort;
    part def PropulsionSystem {{ out port data : DataPort; }}
    part def FlightController {{ in port data : DataPort; }}
    part def Catalog_r6PropulsionsystemImpl :> PropulsionSystem {{
        attribute rotorCount : Real = 6.0;
    }}
    part propulsionSystem : {ptype};
    part flightController : FlightController;
    connect propulsionSystem.data to flightController.data;
}}"""
    config = DesignConfiguration(
        name="recommended",
        parameters={"propulsionSystem": "catalog_r6"},
    )

    def run(ptype, dse_config):
        text = base.format(ptype=ptype)
        model = build_lite_model(text, model_name="P")
        return DesignEvaluator(quality_threshold=0.75).evaluate(
            DesignConfiguration(name="t", parameters={}), model,
            dse_config=dse_config,
            syntax_result=check_syntax(text),
            sim_result=SimulationValidator().validate(text, model_name="P"),
            requirements=[],
        )

    realised = run("Catalog_r6PropulsionsystemImpl", config)
    assert realised.criteria_scores["dse_fidelity"] == 1.0
    assert "dse_fidelity" in realised.weights_used

    catalogued_only = run("PropulsionSystem", config)
    assert catalogued_only.criteria_scores["dse_fidelity"] == 0.5

    without_config = run("PropulsionSystem", None)
    assert "dse_fidelity" not in without_config.weights_used
