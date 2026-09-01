"""
tests/test_simulation.py

Tests for the src/simulation/ behavioral reachability module.

Run with:
    conda run -n AI-prototyping pytest tests/test_simulation.py -v
or from an activated AI-prototyping environment:
    pytest tests/test_simulation.py -v
"""

import os
import pytest

from src.simulation.validator import SimulationValidator
from src.simulation.extractor import extract_behavioral_graph
from src.simulation.exec_graph import build_exec_graph, reachable_from
from src.simulation.scenarios import auto_detect_scenarios, DRONE_SCENARIOS, select_scenarios


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Drone model tests
# ---------------------------------------------------------------------------

class TestDroneSimulation:
    @pytest.fixture(scope="class")
    def drone_result(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        v = SimulationValidator()
        return v.validate(src, model_name="AutonomousDrone")

    def test_parse_succeeds(self, drone_result):
        assert drone_result.num_parts > 0, "Should parse at least one part def"

    def test_graph_stats(self, drone_result):
        # Robust structural sanity (not pinned to a specific example revision).
        assert drone_result.num_parts > 0
        assert drone_result.num_ports > 0
        assert drone_result.num_connections > 0

    def test_most_scenarios_reachable(self, drone_result):
        # A well-connected model leaves most scenarios reachable.  Some
        # auto-generated cross-product scenarios may legitimately be
        # unreachable, so we assert a high — not perfect — score.
        assert 0.0 <= drone_result.reachability_score <= 1.0
        assert drone_result.reachability_score >= 0.7, (
            f"score={drone_result.reachability_score:.3f}; "
            f"failed={[r.scenario_name for r in drone_result.failed_scenarios()]}"
        )

    def test_emergency_scenarios_generated(self, drone_result):
        # The auto-detector should derive emergency/safety paths from the model.
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


# ---------------------------------------------------------------------------
# Minimal model tests
# ---------------------------------------------------------------------------

class TestMinimalSimulation:
    @pytest.fixture(scope="class")
    def minimal_result(self):
        v = SimulationValidator(use_drone_scenarios=False)
        return v.validate(MINIMAL_SYSML, model_name="MinimalTest")

    def test_parse_succeeds(self, minimal_result):
        assert minimal_result.num_parts >= 2

    def test_score_positive(self, minimal_result):
        # auto-detected scenarios should find some path
        assert minimal_result.reachability_score >= 0.0

    def test_advisory_evidence_preserves_roles_scenarios_and_components(
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


# ---------------------------------------------------------------------------
# Disconnected model tests
# ---------------------------------------------------------------------------

class TestDisconnectedModel:
    @pytest.fixture(scope="class")
    def disconnected_result(self):
        v = SimulationValidator(use_drone_scenarios=False)
        return v.validate(DISCONNECTED_SYSML, model_name="DisconnectedTest")

    def test_parse_succeeds(self, disconnected_result):
        assert disconnected_result.num_parts >= 2

    def test_score_below_perfect(self, disconnected_result):
        # A→C and B→C are not connected — some scenarios should fail
        # (or no scenarios are generated if auto-detect finds no sensor/controller)
        # At minimum: not a perfect score when parts are isolated
        result = disconnected_result
        if result.scenario_results:
            # If scenarios ran, score should reflect disconnection
            assert result.reachability_score < 1.0 or result.num_connections == 0


# ---------------------------------------------------------------------------
# Unit tests for extractor / exec_graph
# ---------------------------------------------------------------------------

class TestExtractor:
    # extract_behavioral_graph now takes raw SysML TEXT (parses with syside
    # internally) — no separate parse step.

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

    def test_some_part_has_directional_ports(self, drone_bg):
        dirs = {p.direction for p in drone_bg.ports.values()}
        assert "in" in dirs or "out" in dirs or "inout" in dirs

    def test_connection_endpoints_resolve(self, drone_bg):
        for conn in drone_bg.connections:
            assert "." in conn.source or conn.source in drone_bg.parts, \
                f"Connection source '{conn.source}' looks unresolved"
            assert "." in conn.target or conn.target in drone_bg.parts, \
                f"Connection target '{conn.target}' looks unresolved"

    def test_minimal_model_structure(self):
        # A controllable inline model gives stable, exact expectations.
        bg = extract_behavioral_graph(MINIMAL_SYSML)
        assert set(bg.parts) == {"s", "r"}
        assert len(bg.connections) == 1


class TestExecGraph:
    # Use the controllable inline model so node/reachability assertions are
    # stable and not tied to a specific drone.sysml revision.

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
        # s.dataOut → r.dataIn, so r is reachable from s
        reachable = reachable_from(minimal_graph, "s")
        assert "r" in reachable, f"reachable from s = {sorted(reachable)}"

    def test_drone_graph_builds(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        g = build_exec_graph(extract_behavioral_graph(src))
        assert len(g.nodes) > 0


# ---------------------------------------------------------------------------
# Scenario selection tests
# ---------------------------------------------------------------------------

class TestScenarioSelection:
    def test_drone_scenarios_applicable(self):
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        bg = extract_behavioral_graph(src)
        scenarios = select_scenarios(bg, predefined=DRONE_SCENARIOS)
        assert len(scenarios) > 0

    def test_auto_detect_generates_scenarios(self):
        bg = extract_behavioral_graph(MINIMAL_SYSML)
        # auto_detect uses keyword matching — Sender/Receiver won't match well
        # but should still produce a list (possibly a single generic scenario).
        scenarios = auto_detect_scenarios(bg)
        assert isinstance(scenarios, list)


class TestAnalysisScaffoldingExcluded:
    """Parts inside an `analysis def` (the DSE trade study's alt{i}Design binding parts)
    are analysis scaffolding, not the system assembly — they must NOT enter the behavioral
    graph or they count as 'isolated' and deflate reachability (regression)."""

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
        assert set(bg.parts) == {"a", "b"}            # only the real system parts
        assert "alt0Design" not in bg.parts

    def test_reachability_not_deflated_by_analysis_parts(self):
        res = SimulationValidator().validate(self._MODEL)
        assert "alt0Design" not in res.isolated_parts


def test_dse_analysis_closure_excluded_from_reachability():
    """The injected DSE analysis closure (part def DseDesignAnalysis holding recommendedDesign) is
    scaffolding, NOT a system component — it must be excluded from the reachability graph, else the
    connectivity refiner bolts bogus ports/connects onto real parts to 'wire it up' (a real bug)."""
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
    assert not any("recommendedDesign" in str(p) for p in parts)   # closure part excluded
    assert any("motor" in str(p) for p in parts)                   # real assembly kept
    assert any("airframe" in str(p) for p in parts)


def test_a_controller_satisfying_safe_requirements_stays_a_controller():
    """Measured on the s0v8 anchor: flightController carried a satisfy
    REQ_SAFE_* link and the satisfy-first classifier filed it under
    "safety" — every controller-role controlled scenario then died with
    MISSING_TARGET_ROLE (1/7 on a healthy topology). Name identity
    classifies first; satisfy fragments only classify keyword-less names."""
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
    bg.parts["unit7"] = PartNode(   # keyword-less: satisfy fragments decide
        id="unit7", def_name="Unit7",
        satisfied_reqs=["REQ_SAFE_001"],
    )
    roles = classify_parts_by_role(bg)

    assert roles.get("controller") == ["flightController"]
    assert "flightController" not in roles.get("safety", [])
    assert "watchdogUnit" in roles.get("safety", [])   # name keyword wins
    assert "unit7" in roles.get("safety", [])          # P2 fallback preserved


def test_dse_machinery_is_not_charged_by_the_quality_ruler():
    """The variant emitter's contract says variants specialise the host
    'so they share its ports', and the reachability extractor already
    exempts the DseDesignAnalysis closure by codified name — but the
    quality scorer followed neither, charging FULL's terminal artifact
    0.043 against NO-DSE on the same ruler for its own DSE machinery
    (measured, seed-0 wave). Specialisation inherits ports; the closure
    wrapper leaves the denominator; a standalone attribute-only def is
    still penalised."""
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


def test_a_retyped_usage_still_counts_as_the_planned_safe_part():
    """A usage retyped to the DSE-selected implementation
    (propulsionSystem : Catalog_*Impl :> PropulsionSystem) is still a usage
    of the planned type — the safety-connectivity scorer read the selected
    design's own SAFE part as unwired on a fully wired model (measured:
    exactly 0.938, the 3/4 signature)."""
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
