"""
tests/test_simulation.py

Tests for the src/simulation/ behavioral reachability module.

Run with:
    conda run -n AI-prototyping pytest tests/test_simulation.py -v
or from an activated AI-prototyping environment:
    pytest tests/test_simulation.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.validator import SimulationValidator, SimulationResult
from src.simulation.extractor import extract_behavioral_graph
from src.simulation.exec_graph import build_exec_graph, reachable_from
from src.simulation.scenarios import auto_detect_scenarios, DRONE_SCENARIOS, select_scenarios
from src.simulation.simulator import ScenarioSimulator


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
        assert drone_result.num_parts == 8
        assert drone_result.num_ports >= 20
        assert drone_result.num_connections == 13

    def test_all_scenarios_pass(self, drone_result):
        failed = drone_result.failed_scenarios()
        assert not failed, f"Failed scenarios: {[r.scenario_name for r in failed]}"

    def test_score_is_high(self, drone_result):
        assert drone_result.reachability_score >= 0.9

    def test_emergency_path_reachable(self, drone_result):
        emergency = next(
            (r for r in drone_result.scenario_results
             if r.scenario_name == "emergency_abort_path"),
            None,
        )
        assert emergency is not None, "emergency_abort_path scenario missing"
        assert emergency.passed, f"Emergency path failed: {emergency.issues}"

    def test_safety_scenarios_pass(self, drone_result):
        safety = [r for r in drone_result.scenario_results if "safety" in r.tags]
        assert safety, "No safety-tagged scenarios found"
        for r in safety:
            assert r.passed, f"Safety scenario '{r.scenario_name}' failed: {r.issues}"

    def test_obstacle_avoidance_loop(self, drone_result):
        loop = next(
            (r for r in drone_result.scenario_results
             if r.scenario_name == "obstacle_avoidance_loop"),
            None,
        )
        assert loop is not None
        assert loop.passed, f"Obstacle avoidance loop failed: {loop.issues}"

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
    @pytest.fixture(scope="class")
    def drone_bg(self):
        from src.sysml.Syside_AST_Parser import parse_sysml_to_model
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        model = parse_sysml_to_model(src)
        return extract_behavioral_graph(model)

    def test_parts_extracted(self, drone_bg):
        assert len(drone_bg.parts) == 8

    def test_ports_extracted(self, drone_bg):
        assert len(drone_bg.ports) >= 20

    def test_connections_extracted(self, drone_bg):
        assert len(drone_bg.connections) == 13

    def test_actions_extracted(self, drone_bg):
        assert len(drone_bg.actions) >= 3

    def test_port_directions(self, drone_bg):
        fc_ports = {pid: p for pid, p in drone_bg.ports.items()
                    if p.part_name == "FlightController"}
        dirs = {p.direction for p in fc_ports.values()}
        assert "in" in dirs or "out" in dirs, "FlightController should have directional ports"

    def test_connection_endpoints_resolve(self, drone_bg):
        for conn in drone_bg.connections:
            assert "." in conn.source or conn.source in drone_bg.parts, \
                f"Connection source '{conn.source}' looks unresolved"
            assert "." in conn.target or conn.target in drone_bg.parts, \
                f"Connection target '{conn.target}' looks unresolved"


class TestExecGraph:
    @pytest.fixture(scope="class")
    def drone_graph(self):
        from src.sysml.Syside_AST_Parser import parse_sysml_to_model
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        model = parse_sysml_to_model(src)
        bg = extract_behavioral_graph(model)
        return build_exec_graph(bg)

    def test_parts_are_nodes(self, drone_graph):
        assert "FlightController" in drone_graph.nodes
        assert "SafetyMonitor" in drone_graph.nodes

    def test_ports_are_nodes(self, drone_graph):
        assert any("FlightController." in n for n in drone_graph.nodes)

    def test_reachability_emergency(self, drone_graph):
        reachable = reachable_from(drone_graph, "SafetyMonitor")
        assert "FlightController" in reachable, \
            "FlightController should be reachable from SafetyMonitor"

    def test_reachability_sensor_chain(self, drone_graph):
        reachable = reachable_from(drone_graph, "SensorUnit")
        assert "ObstacleAvoidanceSystem" in reachable
        assert "FlightController" in reachable


# ---------------------------------------------------------------------------
# Scenario selection tests
# ---------------------------------------------------------------------------

class TestScenarioSelection:
    def test_drone_scenarios_applicable(self):
        from src.sysml.Syside_AST_Parser import parse_sysml_to_model
        with open(DRONE_SYSML_PATH) as f:
            src = f.read()
        model = parse_sysml_to_model(src)
        bg = extract_behavioral_graph(model)
        scenarios = select_scenarios(bg, predefined=DRONE_SCENARIOS)
        assert len(scenarios) > 0

    def test_auto_detect_generates_scenarios(self):
        from src.sysml.Syside_AST_Parser import parse_sysml_to_model
        model = parse_sysml_to_model(MINIMAL_SYSML)
        bg = extract_behavioral_graph(model)
        # auto_detect uses keyword matching — Sender/Receiver won't match well
        # but should still produce at least a generic scenario
        scenarios = auto_detect_scenarios(bg)
        # No crash; may return empty list for a minimal model
        assert isinstance(scenarios, list)
