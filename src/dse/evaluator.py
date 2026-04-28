"""
Design evaluator module.

Provides functions for evaluating SysML v2 designs against requirements,
scoring them on multiple quality attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .design_space import DesignConfiguration
from ..sysml.model import SysMLModel


@dataclass
class EvaluationCriteria:
    """A single evaluation criterion with a weight and scoring function."""
    name: str
    weight: float
    description: str
    scoring_function: Optional[Callable[[DesignConfiguration, SysMLModel], float]] = None

    def score(self, config: DesignConfiguration, model: SysMLModel) -> float:
        """Compute the score for this criterion."""
        if self.scoring_function:
            return self.scoring_function(config, model)
        return 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating a design configuration."""
    configuration_name: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    weighted_total: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def is_acceptable(self, threshold: float = 0.6) -> bool:
        """Check if the design meets the minimum quality threshold."""
        return self.weighted_total >= threshold


class DesignEvaluator:
    """
    Evaluates SysML v2 design configurations against multiple criteria.

    Supports both rule-based and LLM-assisted evaluation.
    """

    def __init__(self):
        self.criteria: List[EvaluationCriteria] = []
        self._add_default_criteria()

    def _add_default_criteria(self) -> None:
        """Add default MBSE quality criteria."""
        self.criteria = [
            EvaluationCriteria(
                name="functional_completeness",
                weight=0.30,
                description="All functional requirements are addressed",
                scoring_function=self._score_functional_completeness,
            ),
            EvaluationCriteria(
                name="structural_quality",
                weight=0.25,
                description="Design structure is well-organized and modular",
                scoring_function=self._score_structural_quality,
            ),
            EvaluationCriteria(
                name="interface_consistency",
                weight=0.20,
                description="Ports and connections are consistent",
                scoring_function=self._score_interface_consistency,
            ),
            EvaluationCriteria(
                name="requirement_traceability",
                weight=0.25,
                description="Design elements are traceable to requirements",
                scoring_function=self._score_traceability,
            ),
        ]

    def add_criterion(self, criterion: EvaluationCriteria) -> None:
        """Add a custom evaluation criterion."""
        self.criteria.append(criterion)

    def evaluate(
        self,
        config: DesignConfiguration,
        model: SysMLModel,
    ) -> EvaluationResult:
        """Evaluate a design configuration against all criteria."""
        total_weight = sum(c.weight for c in self.criteria)
        result = EvaluationResult(configuration_name=config.name)
        weighted_sum = 0.0

        for criterion in self.criteria:
            score = criterion.score(config, model)
            result.criteria_scores[criterion.name] = score
            weighted_sum += score * criterion.weight

        result.weighted_total = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Generate issues and recommendations
        for criterion_name, score in result.criteria_scores.items():
            if score < 0.5:
                issue = self._build_issue_message(criterion_name, score)
                recommendation = self._build_recommendation(criterion_name)
                result.issues.append(issue)
                result.recommendations.append(recommendation)

        return result

    def evaluate_from_scores(
        self,
        config: DesignConfiguration,
        scores: Dict[str, float],
    ) -> EvaluationResult:
        """Evaluate using pre-computed scores (e.g., from LLM evaluation)."""
        total_weight = sum(c.weight for c in self.criteria)
        result = EvaluationResult(configuration_name=config.name)
        result.criteria_scores = scores

        weighted_sum = 0.0
        for criterion in self.criteria:
            score = scores.get(criterion.name, 0.0)
            weighted_sum += score * criterion.weight

        result.weighted_total = weighted_sum / total_weight if total_weight > 0 else 0.0
        return result

    def simple_score(self, config: DesignConfiguration) -> Dict[str, float]:
        """
        Simple scoring function for use with MCTS that doesn't require a SysMLModel.

        Returns scores based on configuration parameter heuristics.
        """
        scores: Dict[str, float] = {}

        # Score based on parameter completeness
        param_count = len(config.parameters)
        scores["functional_completeness"] = min(1.0, param_count / 5.0)

        # Structural quality based on parameter diversity
        if param_count > 0:
            unique_types = len(set(str(type(v).__name__) for v in config.parameters.values()))
            scores["structural_quality"] = min(1.0, unique_types / 3.0)
        else:
            scores["structural_quality"] = 0.0

        # Interface consistency - all boolean params True is penalized slightly
        bool_params = [v for v in config.parameters.values() if isinstance(v, bool)]
        if bool_params:
            true_ratio = sum(1 for v in bool_params if v) / len(bool_params)
            scores["interface_consistency"] = 0.7 + 0.3 * (1 - abs(true_ratio - 0.5) * 2)
        else:
            scores["interface_consistency"] = 0.75

        # Traceability based on parameter naming conventions
        scores["requirement_traceability"] = 0.70

        return scores

    # -------------------------------------------------------------------------
    # Default scoring functions
    # -------------------------------------------------------------------------

    def _count_connections(self, model: SysMLModel) -> int:
        """Count connection usages across top-level and part definitions."""
        seen_ids = set()
        count = 0

        for usage in getattr(model, "top_level_usages", []):
            if usage.__class__.__name__ != "ConnectionUsage":
                continue
            uid = getattr(usage, "id", None)
            if uid and uid in seen_ids:
                continue
            if uid:
                seen_ids.add(uid)
            count += 1

        for part in getattr(model, "part_definitions", []):
            for conn in getattr(part, "connection_usages", []):
                uid = getattr(conn, "id", None)
                if uid and uid in seen_ids:
                    continue
                if uid:
                    seen_ids.add(uid)
                count += 1

        return count

    def _score_functional_completeness(
        self, config: DesignConfiguration, model: SysMLModel
    ) -> float:
        """Score how well the model addresses functional requirements."""
        if not model.requirement_definitions:
            return 0.5  # No requirements to check
        if not model.part_definitions:
            return 0.0  # No design elements
        # Heuristic: ratio of requirements to design part definitions
        coverage = min(
            1.0,
            len(model.part_definitions) / max(1, len(model.requirement_definitions)),
        )
        return coverage

    def _score_structural_quality(
        self, config: DesignConfiguration, model: SysMLModel
    ) -> float:
        """Score the structural quality of the design."""
        if not model.part_definitions:
            return 0.0
        # Reward having multiple parts with ports (modular design)
        parts_with_ports = sum(1 for p in model.part_definitions if p.ports)
        modularity = parts_with_ports / len(model.part_definitions)
        # Reward having connectors (connected design)
        connectivity = min(1.0, self._count_connections(model) / max(1, len(model.part_definitions)))
        return 0.6 * modularity + 0.4 * connectivity

    def _score_interface_consistency(
        self, config: DesignConfiguration, model: SysMLModel
    ) -> float:
        """Score the consistency of interfaces (ports and connections)."""
        if not model.part_definitions:
            return 0.5
        total_ports = sum(len(p.ports) for p in model.part_definitions)
        connected_ports = self._count_connections(model) * 2
        if total_ports == 0:
            return 0.5
        return min(1.0, connected_ports / total_ports)

    def _score_traceability(
        self, config: DesignConfiguration, model: SysMLModel
    ) -> float:
        """Score the traceability from design to requirements."""
        if not model.requirement_definitions:
            return 0.5
        if not model.part_definitions:
            return 0.0
        # Check how many part definitions reference requirements
        parts_with_traces = sum(
            1 for p in model.part_definitions if getattr(p, "satisfy_relationships", [])
        )
        return parts_with_traces / len(model.part_definitions)

    @staticmethod
    def _build_issue_message(criterion_name: str, score: float) -> str:
        """Create an actionable issue message for the refinement loop."""
        return f"{criterion_name}: score={score:.2f} below refinement threshold"

    @staticmethod
    def _build_recommendation(criterion_name: str) -> str:
        """Create criterion-specific refinement guidance."""
        guidance = {
            "functional_completeness": "Add missing parts/actions that explicitly cover the uncovered requirements.",
            "structural_quality": "Refine the architecture into clearer modular parts and reduce monolithic structure.",
            "interface_consistency": "Align ports and connectors so all exposed interfaces are consistent and connected.",
            "requirement_traceability": "Add `satisfy` links from blocks to requirement IDs and preserve requirement allocation in the model.",
        }
        return guidance.get(
            criterion_name,
            f"Review the design elements related to '{criterion_name}' and refine the weakest parts.",
        )

