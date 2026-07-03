"""Design Space Exploration (DSE) package.

Search engines live in :mod:`mo_mcts` (multi-objective MCTS), :mod:`bilevel` /
:mod:`inner_bo` (bilevel coupling + inner Bayesian optimisation), and
:mod:`variation_dse` (LLM-declared variation points).  The legacy scalar
weighted-sum MCTS was removed.
"""

from .design_space import (
    DesignConfiguration,
    DesignParameter,
    DesignSpace,
    ParameterType,
)
from .evaluator import DesignEvaluator, EvaluationCriteria, EvaluationResult

__all__ = [
    "DesignConfiguration",
    "DesignEvaluator",
    "DesignParameter",
    "DesignSpace",
    "EvaluationCriteria",
    "EvaluationResult",
    "ParameterType",
]
