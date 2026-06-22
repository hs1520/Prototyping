"""Real design-quality scoring via the live DesignEvaluator (Item F, step 5).

The bilevel search is driven by cheap grounded objectives (fast, runs every node).
This module scores a design with the *real* multi-dimensional DesignEvaluator
(requirement coverage, structural completeness, behavioral verification, safety
assurance, interface quality, syntactic validity) — the genuine SysML design
quality. Per the multi-fidelity design it is applied to the FEW front / recommended
designs (re-parse + full evaluate is expensive and needs the complete model), not
to every search node, where it would be slow and degenerate on bare fragments.

``mcts_fidelity`` is dropped (the self-verification dimension the redesign removed).
"""
from __future__ import annotations

from typing import Dict, Optional

from ..sysml.lite_model import build_lite_model
from .design_space import DesignConfiguration
from .evaluator import DesignEvaluator


def evaluate_design_quality(
    sysml_text: str,
    evaluator: Optional[DesignEvaluator] = None,
    sim_result=None,
) -> Dict[str, float]:
    """Parse ``sysml_text`` into a model and return the real design-quality
    dimension scores (0..1). ``mcts_config=None`` drops mcts_fidelity."""
    model = build_lite_model(sysml_text, model_name="bilevel_candidate")
    if not hasattr(model, "metadata") or model.metadata is None:
        object.__setattr__(model, "metadata", {})
    model.metadata["last_sysml_text"] = sysml_text  # text-based scorers read this

    evaluator = evaluator or DesignEvaluator()
    result = evaluator.evaluate(
        DesignConfiguration(name="bilevel_candidate", parameters={}),
        model,
        mcts_config=None,        # drop mcts_fidelity (self-verification dim)
        sim_result=sim_result,
    )
    return dict(result.criteria_scores)
