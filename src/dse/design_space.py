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
        """Compute the weighted overall score."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

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
    """
    name: str
    parameters: List[DesignParameter] = field(default_factory=list)
    configurations: List[DesignConfiguration] = field(default_factory=list)
    objective_weights: Dict[str, float] = field(default_factory=dict)

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
