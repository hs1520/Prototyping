from __future__ import annotations

from src.agents.dse_injectors import build_dse_design_constraints
from src.dse.design_space import DesignConfiguration
from src.dse.evaluator import (
    DIMENSION_WEIGHTS,
    DesignEvaluator,
    EvaluationResult,
    derive_dimension_weights,
)

_SAFE_HEAVY = [
    "REQ-SAFE-001: failsafe on GPS loss. [SEV:Catastrophic]",
    "REQ-SAFE-002: parachute on motor failure. [SEV:Catastrophic]",
    "REQ-SAFE-003: geofence enforcement. [SEV:Hazardous]",
    "REQ-FUNC-001: navigate to waypoints.",
]
_INTF_HEAVY = [
    "REQ-INTF-001: MAVLink telemetry.",
    "REQ-INTF-002: smart grid API.",
    "REQ-INTF-003: fleet manager reporting.",
    "REQ-FUNC-001: do the thing.",
]


class TestDeriveDimensionWeights:
    def test_no_requirements_returns_prior(self):
        assert derive_dimension_weights(None) == DIMENSION_WEIGHTS
        assert derive_dimension_weights([]) == DIMENSION_WEIGHTS

    def test_unclassifiable_returns_prior(self):
        assert derive_dimension_weights(["the system shall work"]) == DIMENSION_WEIGHTS

    def test_safe_heavy_boosts_safety(self):
        w = derive_dimension_weights(_SAFE_HEAVY)
        assert w["safety_assurance"] > DIMENSION_WEIGHTS["safety_assurance"]
        assert w["safety_assurance"] > w["interface_quality"]

    def test_intf_heavy_boosts_interface(self):
        w = derive_dimension_weights(_INTF_HEAVY)
        assert w["interface_quality"] > DIMENSION_WEIGHTS["interface_quality"]

    def test_weights_sum_one_ratios_kept(self):
        w = derive_dimension_weights(_SAFE_HEAVY)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        ratio_prior = DIMENSION_WEIGHTS["syntactic_validity"] / DIMENSION_WEIGHTS["structural_completeness"]
        ratio_derived = w["syntactic_validity"] / w["structural_completeness"]
        assert abs(ratio_prior - ratio_derived) < 1e-9


class TestVerdictRobustness:
    def _result(self, scores, total):
        return EvaluationResult(
            configuration_name="t", criteria_scores=scores, weighted_total=total
        )

    def test_high_scores_fully_robust(self):
        ev = DesignEvaluator(quality_threshold=0.75)
        res = self._result({"a": 0.95, "b": 0.92, "c": 0.90}, 0.92)
        assert ev.verdict_robustness(res) == 1.0

    def test_borderline_scores_partly_robust(self):
        ev = DesignEvaluator(quality_threshold=0.75)
        res = self._result({"a": 0.95, "b": 0.95, "c": 0.30}, 0.78)
        rob = ev.verdict_robustness(res)
        assert 0.0 < rob < 1.0

    def test_deterministic_for_fixed_seed(self):
        ev = DesignEvaluator()
        res = self._result({"a": 0.9, "b": 0.5}, 0.76)
        assert ev.verdict_robustness(res) == ev.verdict_robustness(res)


class TestDseFidelityNA:
    def _evaluate(self, dse_config, requirements=None):
        from src.sysml.model import PartDefinition, SysMLModel

        model = SysMLModel(name="T")
        model.add_part_definition(PartDefinition(name="Controller"))
        ev = DesignEvaluator()
        return ev.evaluate(
            DesignConfiguration(name="c"), model,
            dse_config=dse_config, requirements=requirements,
        )

    def test_variation_scores_dse_fidelity(self):
        """Variation configs are scored on dse_fidelity instead of skipping the dimension.

        The dimension's semantics (selected decisions realised in the model) applies to
        variant choices, and an unrealised choice earns 0.0, so grading handles the
        inflation concern. Skipping it scored the FULL arm, which materialised its
        selected catalogue variant, on fewer dimensions than the arms doing less.
        """
        variation_cfg = DesignConfiguration(
            name="v", parameters={"propulsionSystem": "hexa", "powerSystem": "p6"}
        )
        result = self._evaluate(variation_cfg)
        assert "dse_fidelity" in result.weights_used
        assert result.criteria_scores["dse_fidelity"] == 0.0

    def test_catalog_config_keeps_dse_fidelity(self):
        catalog_cfg = DesignConfiguration(
            name="b", parameters={"redundancy_level": "triple", "num_sensors": 3}
        )
        result = self._evaluate(catalog_cfg)
        assert "dse_fidelity" in result.criteria_scores

    def test_weights_used_normalised(self):
        result = self._evaluate(None, requirements=_SAFE_HEAVY)
        assert result.weights_used
        assert abs(sum(result.weights_used.values()) - 1.0) < 1e-3


class TestConstraintsNoDanglingHeader:
    def test_variation_constraints_empty(self):
        cfg = DesignConfiguration(
            name="v", parameters={"propulsionSystem": "hexa", "sensorSuite": "lidar"}
        )
        assert build_dse_design_constraints(cfg) == ""

    def test_catalog_constraints_have_header(self):
        cfg = DesignConfiguration(
            name="b", parameters={"redundancy_level": "triple"}
        )
        text = build_dse_design_constraints(cfg)
        assert "DSE Architectural Decisions" in text
        assert "redundancy_level=triple" in text
