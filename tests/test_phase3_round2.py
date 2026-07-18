"""Tests for the weighted DesignConfiguration.overall_score.

The scalar-MCTS features this file also used to cover (early stopping,
seeding, _explore_design_space, inter-parameter constraints) were removed
with the legacy scalar MCTS.
"""
from __future__ import annotations

import sys
from types import ModuleType

for _name, _attrs in [
    ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
    ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
    ("syside", {}),
]:
    if _name not in sys.modules:
        try:  # prefer the real package — a stub here poisons later test files
            __import__(_name)
            continue
        except ImportError:
            pass
        _mod = ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod

from src.dse.design_space import (
    DEFAULT_DIMENSION_WEIGHTS,
    DesignConfiguration,
)


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
