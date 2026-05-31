"""Tests for Design Space Exploration and MCTS."""

import math
import sys
from types import ModuleType
import pytest

# Stub heavy optional deps not installed in the test environment
for _name, _attrs in [
    ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
    ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
    ("syside", {}),
]:
    if _name not in sys.modules:
        _mod = ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod
from src.dse.design_space import (
    DesignConfiguration,
    DesignParameter,
    DesignSpace,
    ParameterType,
)
from src.dse.evaluator import DesignEvaluator, EvaluationResult
from src.dse.mcts import MCTSDesignExplorer, MCTSNode
from src.sysml.model import (
    ConnectionEnd,
    ConnectionUsage,
    ElementRef,
    FeatureDirection,
    PortUsage,
    PartDefinition,
    RequirementDefinition,
    SysMLModel,
)


class TestDesignParameter:
    def test_continuous_parameter(self):
        param = DesignParameter(
            name="frequency",
            param_type=ParameterType.CONTINUOUS,
            default_value=100.0,
            min_value=10.0,
            max_value=1000.0,
            unit="Hz",
        )
        assert param.is_valid(100.0)
        assert param.is_valid(10.0)
        assert param.is_valid(1000.0)
        assert not param.is_valid(5.0)   # below min
        assert not param.is_valid(2000.0)  # above max

    def test_categorical_parameter(self):
        param = DesignParameter(
            name="protocol",
            param_type=ParameterType.CATEGORICAL,
            default_value="CAN",
            choices=["CAN", "Ethernet", "SPI"],
        )
        assert param.is_valid("CAN")
        assert param.is_valid("SPI")
        assert not param.is_valid("USB")

    def test_boolean_parameter(self):
        param = DesignParameter(
            name="redundancy",
            param_type=ParameterType.BOOLEAN,
            default_value=False,
        )
        assert param.is_valid(True)
        assert param.is_valid(False)
        assert not param.is_valid("yes")

    def test_clamp(self):
        param = DesignParameter(
            name="speed",
            param_type=ParameterType.CONTINUOUS,
            default_value=50.0,
            min_value=0.0,
            max_value=100.0,
        )
        assert param.clamp(150.0) == 100.0
        assert param.clamp(-10.0) == 0.0
        assert param.clamp(50.0) == 50.0


class TestDesignConfiguration:
    def test_creation(self):
        config = DesignConfiguration(
            name="test_config",
            parameters={"redundancy": "dual", "frequency": 200.0},
        )
        assert config.name == "test_config"
        assert config.parameters["redundancy"] == "dual"

    def test_overall_score_empty(self):
        config = DesignConfiguration(name="test")
        assert config.overall_score == 0.0

    def test_overall_score(self):
        config = DesignConfiguration(
            name="test",
            scores={"completeness": 0.8, "performance": 0.6},
        )
        assert abs(config.overall_score - 0.7) < 1e-6

    def test_copy_with_changes(self):
        config = DesignConfiguration(
            name="original",
            parameters={"a": 1, "b": 2},
            scores={"s1": 0.5},
        )
        new_config = config.copy_with_changes({"a": 99})
        assert new_config.parameters["a"] == 99
        assert new_config.parameters["b"] == 2
        assert new_config.scores == {}  # scores reset
        assert new_config.parent_id == config.id
        assert new_config.id != config.id

    def test_dominates(self):
        config_a = DesignConfiguration(
            name="A",
            scores={"s1": 0.9, "s2": 0.8},
        )
        config_b = DesignConfiguration(
            name="B",
            scores={"s1": 0.7, "s2": 0.6},
        )
        assert config_a.dominates(config_b)
        assert not config_b.dominates(config_a)

    def test_no_dominance_with_tradeoffs(self):
        config_a = DesignConfiguration(name="A", scores={"s1": 0.9, "s2": 0.5})
        config_b = DesignConfiguration(name="B", scores={"s1": 0.5, "s2": 0.9})
        assert not config_a.dominates(config_b)
        assert not config_b.dominates(config_a)


class TestDesignSpace:
    @pytest.fixture
    def design_space(self):
        space = DesignSpace(name="TestSpace")
        space.add_parameter(DesignParameter(
            name="redundancy",
            param_type=ParameterType.CATEGORICAL,
            default_value="none",
            choices=["none", "dual", "triple"],
        ))
        space.add_parameter(DesignParameter(
            name="frequency",
            param_type=ParameterType.CONTINUOUS,
            default_value=100.0,
            min_value=10.0,
            max_value=1000.0,
        ))
        return space

    def test_get_default_configuration(self, design_space):
        config = design_space.get_default_configuration()
        assert config.parameters["redundancy"] == "none"
        assert config.parameters["frequency"] == 100.0

    def test_get_best_configuration_empty(self, design_space):
        assert design_space.get_best_configuration() is None

    def test_get_best_configuration(self, design_space):
        design_space.add_configuration(DesignConfiguration(
            name="c1", scores={"s": 0.5}
        ))
        design_space.add_configuration(DesignConfiguration(
            name="c2", scores={"s": 0.9}
        ))
        best = design_space.get_best_configuration()
        assert best.name == "c2"

    def test_pareto_front(self, design_space):
        c1 = DesignConfiguration(name="c1", scores={"a": 0.9, "b": 0.5})
        c2 = DesignConfiguration(name="c2", scores={"a": 0.5, "b": 0.9})
        c3 = DesignConfiguration(name="c3", scores={"a": 0.3, "b": 0.3})
        design_space.add_configuration(c1)
        design_space.add_configuration(c2)
        design_space.add_configuration(c3)
        
        pareto = design_space.get_pareto_front()
        pareto_names = [c.name for c in pareto]
        assert "c1" in pareto_names
        assert "c2" in pareto_names
        assert "c3" not in pareto_names

    def test_get_summary(self, design_space):
        design_space.add_configuration(DesignConfiguration(
            name="c1", scores={"s": 0.7}
        ))
        summary = design_space.get_summary()
        assert summary["parameters"] == 2
        assert summary["configurations_evaluated"] == 1


class TestMCTSDesignExplorer:
    @pytest.fixture
    def design_space(self):
        space = DesignSpace(name="MCTSTest")
        space.add_parameter(DesignParameter(
            name="protocol",
            param_type=ParameterType.CATEGORICAL,
            default_value="CAN",
            choices=["CAN", "Ethernet", "SPI"],
        ))
        space.add_parameter(DesignParameter(
            name="frequency",
            param_type=ParameterType.CONTINUOUS,
            default_value=100.0,
            min_value=10.0,
            max_value=500.0,
        ))
        space.add_parameter(DesignParameter(
            name="redundant",
            param_type=ParameterType.BOOLEAN,
            default_value=False,
        ))
        return space

    def test_search_runs(self, design_space):
        def evaluator(config):
            return {"performance": 0.7, "reliability": 0.6}

        explorer = MCTSDesignExplorer(
            design_space=design_space,
            evaluation_function=evaluator,
            random_seed=42,
        )
        best = explorer.search(num_iterations=10)
        assert best is not None
        assert isinstance(best.overall_score, float)

    def test_search_explores_configs(self, design_space):
        def evaluator(config):
            return {"score": config.parameters.get("frequency", 0) / 500.0}

        explorer = MCTSDesignExplorer(
            design_space=design_space,
            evaluation_function=evaluator,
            random_seed=42,
        )
        explorer.search(num_iterations=20)
        # Should explore more than 1 configuration
        assert len(design_space.configurations) > 1

    def test_mcts_tree_summary(self, design_space):
        def evaluator(config):
            return {"score": 0.5}

        explorer = MCTSDesignExplorer(
            design_space=design_space,
            evaluation_function=evaluator,
            random_seed=0,
        )
        explorer.search(num_iterations=15)
        summary = explorer.get_exploration_tree_summary()
        assert "total_nodes" in summary
        assert summary["total_nodes"] >= 1
        assert "best_score" in summary

    def test_ucb1_score(self):
        parent_config = DesignConfiguration(name="root")
        parent = MCTSNode(config=parent_config, visits=10, total_reward=5.0)
        child_config = DesignConfiguration(name="child")
        child = MCTSNode(
            config=child_config, parent=parent, visits=3, total_reward=2.0
        )
        score = child.ucb1_score(exploration_constant=math.sqrt(2))
        assert score > 0
        # UCB1 = exploitation + exploration
        # = 2/3 + sqrt(2) * sqrt(log(10)/3)
        exploitation = 2.0 / 3.0
        exploration = math.sqrt(2) * math.sqrt(math.log(10) / 3)
        expected = exploitation + exploration
        assert abs(score - expected) < 1e-6

    def test_unvisited_node_gets_infinite_ucb(self):
        parent_config = DesignConfiguration(name="root")
        parent = MCTSNode(config=parent_config, visits=5)
        child_config = DesignConfiguration(name="child")
        child = MCTSNode(config=child_config, parent=parent, visits=0)
        assert child.ucb1_score() == float("inf")


class TestDesignEvaluator:
    @pytest.fixture
    def evaluator(self):
        return DesignEvaluator()

    @pytest.fixture
    def simple_model(self):
        model = SysMLModel(name="TestSystem")

        sensor = PartDefinition(name="Sensor")
        sensor.add_port(PortUsage(name="dataOut", direction=FeatureDirection.OUT))

        ctrl = PartDefinition(name="Controller")
        ctrl.add_port(PortUsage(name="sensorIn", direction=FeatureDirection.IN))

        model.add_part_definition(sensor)
        model.add_part_definition(ctrl)
        model.add_requirement_definition(RequirementDefinition(name="R1", text="req"))

        conn = ConnectionUsage(name="c1")
        conn.add_end(ConnectionEnd(role_name="source", feature_ref=ElementRef(name="Sensor.dataOut", path="Sensor.dataOut")))
        conn.add_end(ConnectionEnd(role_name="target", feature_ref=ElementRef(name="Controller.sensorIn", path="Controller.sensorIn")))
        model.add_top_level_usage(conn)

        return model

    def test_evaluate_returns_result(self, evaluator, simple_model):
        config = DesignConfiguration(name="test")
        result = evaluator.evaluate(config, simple_model)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.weighted_total <= 1.0
        # A model with an unsatisfied requirement and attribute-less parts must
        # surface issues and recommendations (criterion names may evolve, so we
        # assert the evaluator flags problems rather than matching exact text).
        assert result.issues, "deficient model should produce issues"
        assert result.recommendations, "deficient model should produce recommendations"

    def test_evaluate_empty_model(self, evaluator):
        model = SysMLModel(name="Empty")
        config = DesignConfiguration(name="test")
        result = evaluator.evaluate(config, model)
        assert isinstance(result, EvaluationResult)

    def test_is_acceptable(self, evaluator, simple_model):
        config = DesignConfiguration(name="test")
        result = evaluator.evaluate(config, simple_model)
        # Just test the method is callable
        assert isinstance(result.is_acceptable(), bool)

    def test_simple_score(self, evaluator):
        config = DesignConfiguration(
            name="test",
            parameters={"a": 1, "b": "two", "c": True, "d": 1.0, "e": False},
        )
        scores = evaluator.simple_score(config)
        assert isinstance(scores, dict)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_default_criteria_count(self, evaluator):
        # 7 dimensions: requirement_satisfaction, mcts_fidelity,
        # structural_integrity, safety_assurance, interface_correctness,
        # syntactic_validity, behavioral_reachability.
        assert len(evaluator.criteria) == 7
