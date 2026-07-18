"""
Design Space Exploration (DSE) module.

Provides data structures for representing and navigating the design space
of cyber-physical systems in MBSE contexts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ParameterType(str, Enum):
    """Type of a design parameter."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


# Default dimension weights for `overall_score`.
# Safety dominates because SAFE constraint violations have physical consequences.
# `simplicity` is a cost proxy and ranks lowest among the four.
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "safety_margin":     0.40,
    "perf_satisfaction": 0.30,
    "protocol_match":    0.15,
    "simplicity":        0.15,
}

# Safety hard floor: a configuration whose safety_margin is below this threshold
# cannot win overall, regardless of how well it scores on other dimensions.
# This prevents MCTS from choosing redundancy=none for safety-critical systems.
SAFETY_HARD_FLOOR: float = 0.20   # below this value → overall score is capped
SAFETY_FLOOR_CAP: float  = 0.30   # the cap applied when below SAFETY_HARD_FLOOR


@dataclass
class DesignParameter:
    """A single design parameter with its range and type."""
    name: str
    param_type: ParameterType
    default_value: Any
    # For continuous parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # For discrete/categorical parameters
    choices: Optional[List[Any]] = None
    unit: str = ""
    description: str = ""

    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this parameter."""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
        elif self.param_type in (ParameterType.DISCRETE, ParameterType.CATEGORICAL):
            return self.choices is None or value in self.choices
        elif self.param_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        return True

    def clamp(self, value: Any) -> Any:
        """Clamp a value to the valid range."""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.min_value is not None:
                value = max(self.min_value, value)
            if self.max_value is not None:
                value = min(self.max_value, value)
        return value


@dataclass
class DesignConfiguration:
    """
    A specific configuration of design parameters — a point in design space.
    """
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])

    @property
    def overall_score(self) -> float:
        """Compute the weighted overall score.

        Uses DEFAULT_DIMENSION_WEIGHTS when score keys match known dimensions.
        Unknown keys (e.g. for legacy or custom scoring) fall back to equal weights.

        Safety hard floor: if safety_margin < SAFETY_HARD_FLOOR the overall score
        is capped at SAFETY_FLOOR_CAP so that unsafe configurations never win the
        MCTS search regardless of how well they score on other dimensions.
        """
        if not self.scores:
            return 0.0
        known = {k: v for k, v in self.scores.items() if k in DEFAULT_DIMENSION_WEIGHTS}
        unknown = {k: v for k, v in self.scores.items() if k not in DEFAULT_DIMENSION_WEIGHTS}
        if not known:
            # No recognised dimensions — fall back to equal-weight average
            return sum(unknown.values()) / len(unknown)
        weighted_sum = sum(v * DEFAULT_DIMENSION_WEIGHTS[k] for k, v in known.items())
        weight_total = sum(DEFAULT_DIMENSION_WEIGHTS[k] for k in known)
        if not unknown:
            result = weighted_sum / weight_total
        else:
            # If both known and unknown dims present, blend by equal weight to remaining mass
            remaining_weight = max(0.0, 1.0 - weight_total)
            if remaining_weight <= 0:
                result = weighted_sum / weight_total
            else:
                unknown_avg = sum(unknown.values()) / len(unknown)
                result = weighted_sum + unknown_avg * remaining_weight

        # Safety hard floor: cap overall score for unsafe configurations so MCTS
        # cannot select redundancy=none when safety_margin is critically low.
        safety = self.scores.get("safety_margin")
        if safety is not None and safety < SAFETY_HARD_FLOOR:
            result = min(result, SAFETY_FLOOR_CAP)

        return result

    def copy_with_changes(self, changes: Dict[str, Any]) -> DesignConfiguration:
        """Create a copy of this configuration with some parameter changes."""
        new_config = copy.deepcopy(self)
        new_config.parameters.update(changes)
        new_config.scores = {}
        new_config.parent_id = self.id
        new_config.id = __import__("uuid").uuid4().hex[:8]
        return new_config

    def dominates(self, other: DesignConfiguration) -> bool:
        """
        Return True if this configuration Pareto-dominates the other.
        (Better in at least one objective and not worse in any.)
        """
        if not self.scores or not other.scores:
            return False
        all_keys = set(self.scores) | set(other.scores)
        at_least_one_better = False
        for key in all_keys:
            self_val = self.scores.get(key, 0.0)
            other_val = other.scores.get(key, 0.0)
            if self_val < other_val:
                return False
            if self_val > other_val:
                at_least_one_better = True
        return at_least_one_better


@dataclass
class DesignSpace:
    """
    Defines the space of possible designs to explore.

    Combines parameter definitions with evaluated configurations.
    `constraints` is a list of predicates `fn(parameters_dict) -> bool` that
    return True when the configuration is feasible.
    """
    name: str
    parameters: List[DesignParameter] = field(default_factory=list)
    configurations: List[DesignConfiguration] = field(default_factory=list)
    objective_weights: Dict[str, float] = field(default_factory=dict)
    constraints: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)

    def add_parameter(self, param: DesignParameter) -> None:
        """Add a design parameter to the space."""
        self.parameters.append(param)

    def add_configuration(self, config: DesignConfiguration) -> None:
        """Add an evaluated configuration."""
        self.configurations.append(config)

    def get_default_configuration(self) -> DesignConfiguration:
        """Create a configuration with all default parameter values."""
        params = {p.name: p.default_value for p in self.parameters}
        return DesignConfiguration(
            name="default",
            parameters=params,
        )

    def get_best_configuration(self) -> Optional[DesignConfiguration]:
        """Return the configuration with the highest overall score."""
        if not self.configurations:
            return None
        return max(self.configurations, key=lambda c: c.overall_score)

    def get_pareto_front(self) -> List[DesignConfiguration]:
        """
        Compute the Pareto-optimal front from all evaluated configurations.
        """
        if not self.configurations:
            return []
        pareto: List[DesignConfiguration] = []
        for candidate in self.configurations:
            dominated = False
            for other in self.configurations:
                if other is not candidate and other.dominates(candidate):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)
        return pareto

    def parameter_by_name(self, name: str) -> Optional[DesignParameter]:
        """Find a design parameter by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the design space exploration results."""
        pareto = self.get_pareto_front()
        best = self.get_best_configuration()
        return {
            "name": self.name,
            "parameters": len(self.parameters),
            "configurations_evaluated": len(self.configurations),
            "pareto_front_size": len(pareto),
            "best_overall_score": best.overall_score if best else 0.0,
            "best_configuration": best.name if best else None,
        }
