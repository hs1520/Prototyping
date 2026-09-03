from __future__ import annotations

from tests._dep_stubs import install_missing_dep_stubs

install_missing_dep_stubs()

from src.dse.design_space import (
    DEFAULT_DIMENSION_WEIGHTS,
    DesignConfiguration,
)


class TestWeightedOverallScore:
    def test_safety_dominates_simplicity(self):
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

    def test_score_matches_explicit_calc(self):
        cfg = DesignConfiguration(
            name="calc",
            scores={
                "safety_margin": 1.0,
                "perf_satisfaction": 1.0,
                "protocol_match": 1.0,
                "simplicity": 1.0,
            },
        )
        assert abs(cfg.overall_score - 1.0) < 1e-6

    def test_unknown_dimensions_equal_weights(self):
        cfg = DesignConfiguration(
            name="unknown",
            scores={"custom_a": 0.4, "custom_b": 0.6},
        )
        assert abs(cfg.overall_score - 0.5) < 1e-6
