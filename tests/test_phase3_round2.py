"""Tests for Phase 3 round-2 optimisations:
  #1 weighted overall_score
  #4 exploration transparency print
  #5 MCTS early stopping (patience)
  #6 configurable random seed
  #2 Pareto front output
  #3 inter-parameter constraints
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from src.agents.orchestrator import Orchestrator
from src.dse.design_space import (
    DEFAULT_DIMENSION_WEIGHTS,
    DesignConfiguration,
    DesignParameter,
    DesignSpace,
    ParameterType,
)
from src.dse.mcts import MCTSDesignExplorer
from src.llm.interface import MockLLM
from src.sysml.model import PartDefinition, SysMLModel


# ─────────────────────────────────────────────────────────────────────────────
# #1 — Weighted overall_score
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedOverallScore:

    def test_safety_dominates_simplicity(self):
        """A safe-but-complex design must outscore an unsafe-but-simple one."""
        safe_complex = DesignConfiguration(
            name="safe_complex",
            scores={
                "safety_margin": 1.0,
                "perf_satisfaction": 0.5,
                "protocol_match": 0.5,
                "simplicity": 0.0,
            },
        )
        unsafe_simple = DesignConfiguration(
            name="unsafe_simple",
            scores={
                "safety_margin": 0.0,
                "perf_satisfaction": 0.5,
                "protocol_match": 0.5,
                "simplicity": 1.0,
            },
        )
        # Equal-weight average would tie at 0.5; weighted should not.
        assert safe_complex.overall_score > unsafe_simple.overall_score
        # Safety weight (0.40) > simplicity weight (0.15), so the gap is meaningful.
        assert safe_complex.overall_score - unsafe_simple.overall_score > 0.2

    def test_known_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6

    def test_weighted_score_matches_explicit_calculation(self):
        cfg = DesignConfiguration(
            name="calc",
            scores={
                "safety_margin": 1.0,
                "perf_satisfaction": 1.0,
                "protocol_match": 1.0,
                "simplicity": 1.0,
            },
        )
        # All scores 1.0 → weighted average = 1.0 regardless of weights
        assert abs(cfg.overall_score - 1.0) < 1e-6

    def test_unknown_dimensions_fall_back_to_equal_weights(self):
        # All-unknown keys should use equal-weight average
        cfg = DesignConfiguration(
            name="unknown",
            scores={"custom_a": 0.4, "custom_b": 0.6},
        )
        assert abs(cfg.overall_score - 0.5) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# #5 — Early stopping via patience
# ─────────────────────────────────────────────────────────────────────────────

class TestMCTSEarlyStopping:

    def _flat_evaluator(self, config):
        """Constant evaluator — every config scores the same."""
        return {"safety_margin": 0.5, "simplicity": 0.5}

    def _make_space(self) -> DesignSpace:
        space = DesignSpace(name="flat")
        space.add_parameter(DesignParameter(
            name="x", param_type=ParameterType.CATEGORICAL,
            default_value="a", choices=["a", "b", "c"],
        ))
        return space

    def test_early_stop_triggers_when_no_improvement(self):
        explorer = MCTSDesignExplorer(
            design_space=self._make_space(),
            evaluation_function=self._flat_evaluator,
            random_seed=1,
        )
        explorer.search(num_iterations=200, patience=5)
        assert explorer.early_stopped is True
        assert explorer.iterations_run < 200

    def test_no_early_stop_when_patience_is_none(self):
        explorer = MCTSDesignExplorer(
            design_space=self._make_space(),
            evaluation_function=self._flat_evaluator,
            random_seed=1,
        )
        explorer.search(num_iterations=20, patience=None)
        assert explorer.early_stopped is False
        assert explorer.iterations_run == 20

    def test_iterations_run_recorded(self):
        explorer = MCTSDesignExplorer(
            design_space=self._make_space(),
            evaluation_function=self._flat_evaluator,
            random_seed=1,
        )
        explorer.search(num_iterations=10, patience=None)
        assert explorer.iterations_run == 10


# ─────────────────────────────────────────────────────────────────────────────
# #6 — Configurable random seed
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurableSeed:

    def test_same_seed_produces_same_trajectory(self):
        orch = Orchestrator(llm=MockLLM())
        model = SysMLModel(name="Test")
        model.part_definitions.append(PartDefinition(name="P1"))
        reqs = ["REQ-PERF-001: The system shall update at 100 Hz."]

        ds_a, best_a, _ = orch._explore_design_space(
            model, mcts_iterations=20, requirements=reqs,
            random_seed=123, patience=None,
        )
        ds_b, best_b, _ = orch._explore_design_space(
            model, mcts_iterations=20, requirements=reqs,
            random_seed=123, patience=None,
        )
        # Same seed → identical winning parameters
        assert best_a.parameters == best_b.parameters

    def test_different_seeds_produce_different_exploration_orders(self):
        """Even if the final winner is the same (small space → quick convergence),
        the trajectory of explored configurations should differ across seeds."""
        orch = Orchestrator(llm=MockLLM())
        model = SysMLModel(name="Test")
        model.part_definitions.append(PartDefinition(name="P1"))
        reqs = [
            "REQ-PERF-001: The system shall update at 100 Hz.",
            "REQ-SAFE-001: The system shall halt on fault.",
            "REQ-INTF-001: The system shall use MAVLink.",
        ]

        trajectories = []
        for seed in (1, 99):
            ds, _, _ = orch._explore_design_space(
                model, mcts_iterations=20, requirements=reqs,
                random_seed=seed, patience=None,
            )
            # Use the order of explored configurations as the trajectory signature
            trajectory = tuple(
                tuple(sorted(c.parameters.items())) for c in ds.configurations
            )
            trajectories.append(trajectory)
        assert trajectories[0] != trajectories[1], \
            "Different seeds should produce different exploration trajectories"


# ─────────────────────────────────────────────────────────────────────────────
# #2 — Pareto front returned by _explore_design_space
# ─────────────────────────────────────────────────────────────────────────────

class TestParetoFrontOutput:

    def test_returns_three_tuple(self):
        orch = Orchestrator(llm=MockLLM())
        model = SysMLModel(name="Test")
        model.part_definitions.append(PartDefinition(name="P1"))
        result = orch._explore_design_space(
            model, mcts_iterations=15, requirements=[
                "REQ-PERF-001: The system shall update at 100 Hz.",
            ],
            random_seed=1, patience=None,
        )
        assert len(result) == 3
        ds, best, pareto = result
        assert isinstance(pareto, list)

    def test_pareto_front_sorted_by_overall_score(self):
        orch = Orchestrator(llm=MockLLM())
        model = SysMLModel(name="Test")
        model.part_definitions.append(PartDefinition(name="P1"))
        _, _, pareto = orch._explore_design_space(
            model, mcts_iterations=20,
            requirements=[
                "REQ-PERF-001: The system shall update at 100 Hz.",
                "REQ-SAFE-001: The system shall fail safely.",
            ],
            random_seed=42, patience=None,
        )
        scores = [c.overall_score for c in pareto]
        assert scores == sorted(scores, reverse=True)

    def test_best_config_is_on_pareto_front(self):
        orch = Orchestrator(llm=MockLLM())
        model = SysMLModel(name="Test")
        model.part_definitions.append(PartDefinition(name="P1"))
        _, best, pareto = orch._explore_design_space(
            model, mcts_iterations=20, requirements=[
                "REQ-PERF-001: The system shall update at 100 Hz.",
            ],
            random_seed=42, patience=None,
        )
        # Best config is non-dominated by definition
        assert any(c.id == best.id for c in pareto)


# ─────────────────────────────────────────────────────────────────────────────
# #3 — Inter-parameter constraints
# ─────────────────────────────────────────────────────────────────────────────

class TestInterParameterConstraints:

    def test_triple_redundancy_blocks_low_sensor_count(self):
        space = DesignSpace(name="test")
        Orchestrator._add_inter_parameter_constraints(space)
        # triple + 2 sensors → infeasible
        assert space.is_feasible({"redundancy_level": "triple", "num_sensors": 2}) is False
        # triple + 3 sensors → feasible
        assert space.is_feasible({"redundancy_level": "triple", "num_sensors": 3}) is True

    def test_dual_redundancy_blocks_single_sensor(self):
        space = DesignSpace(name="test")
        Orchestrator._add_inter_parameter_constraints(space)
        assert space.is_feasible({"redundancy_level": "dual", "num_sensors": 1}) is False
        assert space.is_feasible({"redundancy_level": "dual", "num_sensors": 2}) is True

    def test_centralised_caps_sensors(self):
        space = DesignSpace(name="test")
        Orchestrator._add_inter_parameter_constraints(space)
        assert space.is_feasible({
            "distributed_control": False, "num_sensors": 6, "redundancy_level": "none"
        }) is False
        assert space.is_feasible({
            "distributed_control": False, "num_sensors": 4, "redundancy_level": "none"
        }) is True
        # Distributed has no cap
        assert space.is_feasible({
            "distributed_control": True, "num_sensors": 6, "redundancy_level": "none"
        }) is True

    def test_no_constraints_means_always_feasible(self):
        space = DesignSpace(name="empty")
        assert space.is_feasible({"anything": "goes"}) is True

    def test_mcts_skips_infeasible_expansions(self):
        """MCTS must produce only feasible configurations."""
        space = DesignSpace(name="test")
        space.add_parameter(DesignParameter(
            name="redundancy_level", param_type=ParameterType.CATEGORICAL,
            default_value="none", choices=["none", "dual", "triple"],
        ))
        space.add_parameter(DesignParameter(
            name="num_sensors", param_type=ParameterType.DISCRETE,
            default_value=2, choices=[1, 2, 3, 4],
        ))
        Orchestrator._add_inter_parameter_constraints(space)

        explorer = MCTSDesignExplorer(
            design_space=space,
            evaluation_function=lambda c: {"safety_margin": 0.5, "simplicity": 0.5},
            random_seed=7,
        )
        explorer.search(num_iterations=40, patience=None)

        # Every recorded configuration must be feasible
        for cfg in space.configurations:
            assert space.is_feasible(cfg.parameters), (
                f"Infeasible config produced: {cfg.parameters}"
            )
