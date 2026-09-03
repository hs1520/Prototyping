"""Design-quality scoring via the live DesignEvaluator (Item F, step 5).

The bilevel search runs on cheap grounded objectives; this module scores a design
with the full multi-dimensional DesignEvaluator (requirement coverage, structural
completeness, behavioral verification, safety assurance, interface quality,
syntactic validity). Re-parse + full evaluate is expensive and needs a complete
model, so per the multi-fidelity design it is applied to the few front /
recommended designs, not to every search node, where it would be slow and
degenerate on bare fragments. ``dse_fidelity`` is dropped (the self-verification
dimension the redesign removed).
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
    """Parse ``sysml_text`` into a model and return the real design-quality dimension scores (0..1)."""
    model = build_lite_model(sysml_text, model_name="bilevel_candidate")
    if not hasattr(model, "metadata") or model.metadata is None:
        object.__setattr__(model, "metadata", {})
    model.metadata["last_sysml_text"] = sysml_text  # text-based scorers read this

    evaluator = evaluator or DesignEvaluator()
    result = evaluator.evaluate(
        DesignConfiguration(name="bilevel_candidate", parameters={}),
        model,
        dse_config=None,        # drop dse_fidelity (self-verification dim)
        sim_result=sim_result,
    )
    return dict(result.criteria_scores)
