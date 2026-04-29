"""
Design evaluator module.

Provides functions for evaluating SysML v2 designs against requirements,
scoring them on multiple quality attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .design_space import DesignConfiguration
from ..sysml.model import FeatureDirection, SysMLModel


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
                description="All requirements have explicit satisfy links",
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
                weight=0.15,
                description="Fraction of requirements covered by satisfy links",
                scoring_function=self._score_traceability,
            ),
            EvaluationCriteria(
                name="safety_coverage",
                weight=0.10,
                description="SAFE requirements have corresponding fault-handling behavior",
                scoring_function=self._score_safety_coverage,
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

        # Generate specific, model-aware issues and recommendations
        specific_issues, specific_recs = self._diagnose_model(model)
        result.issues.extend(specific_issues)
        result.recommendations.extend(specific_recs)

        # Append criterion-level summaries for criteria with low scores
        for criterion_name, score in result.criteria_scores.items():
            if score < 0.5:
                result.issues.append(
                    f"{criterion_name}: score={score:.2f} — see specific issues above"
                )
                result.recommendations.append(self._build_recommendation(criterion_name))

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
        """Score the fraction of requirement definitions covered by satisfy links."""
        if not model.requirement_definitions:
            return 0.5  # No requirements to check
        if not model.part_definitions:
            return 0.0
        req_ids = {r.name for r in model.requirement_definitions}
        satisfied_ids = {
            sr.target.name
            for part in model.part_definitions
            for sr in part.satisfy_relationships
            if sr.target and sr.target.name
        }
        return len(satisfied_ids & req_ids) / len(req_ids)

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
        """Fraction of requirement definitions that have at least one incoming satisfy link."""
        if not model.requirement_definitions:
            return 0.5
        if not model.part_definitions:
            return 0.0
        req_ids = {r.name for r in model.requirement_definitions}
        satisfied_ids = {
            sr.target.name
            for part in model.part_definitions
            for sr in part.satisfy_relationships
            if sr.target and sr.target.name
        }
        return len(satisfied_ids & req_ids) / len(req_ids)

    def _score_safety_coverage(
        self, config: DesignConfiguration, model: SysMLModel
    ) -> float:
        """Score whether SAFE requirements have fault-handling actions or nested defs."""
        safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
        if not safe_reqs:
            return 1.0  # N/A — no penalty when no SAFE requirements exist

        _safety_keywords = {"emergency", "fault", "safe", "shutdown", "failsafe", "monitor"}

        def _has_safety_behavior(part) -> bool:
            for action in part.actions:
                if any(kw in action.name.lower() for kw in _safety_keywords):
                    return True
            for nd in part.nested_definitions:
                if any(kw in nd.name.lower() for kw in _safety_keywords):
                    return True
            return False

        parts_with_safety = sum(1 for p in model.part_definitions if _has_safety_behavior(p))
        # Score: proportional to whether *any* part has safety behavior
        # (binary check is more useful than a ratio here)
        return 1.0 if parts_with_safety > 0 else 0.0

    def _diagnose_model(self, model: SysMLModel) -> tuple:
        """
        Produce specific, model-aware issue and recommendation strings.

        Returns (issues: List[str], recommendations: List[str]).
        These are generated independently of score thresholds so the
        refinement loop gets actionable feedback regardless of overall score.
        """
        issues: List[str] = []
        recs: List[str] = []

        req_ids = {r.name for r in model.requirement_definitions}
        satisfied_ids = {
            sr.target.name
            for part in model.part_definitions
            for sr in part.satisfy_relationships
            if sr.target and sr.target.name
        }

        # 1. Untraced requirements
        untraced = sorted(req_ids - satisfied_ids)
        if untraced:
            issues.append(f"Untraced requirements: {', '.join(untraced)}")
            recs.append(
                "Add a `satisfy REQ_X_NNN by <PartName>;` statement for each untraced requirement."
            )

        # 2. Parts missing ports
        parts_no_ports = [p.name for p in model.part_definitions if not p.ports]
        if parts_no_ports:
            issues.append(f"Parts missing ports: {', '.join(parts_no_ports)}")
            recs.append("Add at least one port with direction (in/out/inout) to each part def.")

        # 3. Parts whose ports all lack direction
        parts_undirected = [
            p.name for p in model.part_definitions
            if p.ports and all(
                getattr(port, "direction", FeatureDirection.NONE) == FeatureDirection.NONE
                for port in p.ports
            )
        ]
        if parts_undirected:
            issues.append(f"Parts with undirected ports: {', '.join(parts_undirected)}")
            recs.append("Set direction (in / out / inout) on all port usages.")

        # 4. Parts missing numeric attributes
        parts_no_attrs = [
            p.name for p in model.part_definitions
            if not p.attributes or all(
                not getattr(a, "default_value", None) for a in p.attributes
            )
        ]
        if parts_no_attrs:
            issues.append(f"Parts missing numeric attributes: {', '.join(parts_no_attrs)}")
            recs.append(
                "Add at least one attribute with a numeric default value and SI unit to each part def."
            )

        # 5. SAFE requirements without fault-handling behavior
        safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
        if safe_reqs:
            _safety_kws = {"emergency", "fault", "safe", "shutdown", "failsafe", "monitor"}
            has_any_safety = any(
                any(kw in a.name.lower() for kw in _safety_kws for a in p.actions)
                or any(kw in nd.name.lower() for kw in _safety_kws for nd in p.nested_definitions)
                for p in model.part_definitions
            )
            if not has_any_safety:
                safe_ids = ", ".join(r.name for r in safe_reqs)
                issues.append(
                    f"SAFE requirement(s) {safe_ids} have no fault-handling "
                    f"actions or state defs in any part"
                )
                recs.append(
                    "For each SAFE requirement, add a state def with explicit fault-entry "
                    "transition and an emergency action def (e.g., emergencyStop, shutdownSafely)."
                )

        return issues, recs

    @staticmethod
    def _build_recommendation(criterion_name: str) -> str:
        """Create criterion-specific refinement guidance."""
        guidance = {
            "functional_completeness": (
                "Add satisfy links to cover all requirement IDs — "
                "every REQ_X_NNN must appear in exactly one satisfy statement."
            ),
            "structural_quality": (
                "Refine the architecture into clearly modular parts; "
                "each part def must have ≥ 1 directed port and ≥ 1 numeric attribute."
            ),
            "interface_consistency": (
                "Align ports and connectors: add connect statements to link "
                "all exposed in/out ports between components."
            ),
            "requirement_traceability": (
                "Add `satisfy` links from blocks to requirement IDs and preserve "
                "requirement allocation in the model."
            ),
            "safety_coverage": (
                "For each SAFE requirement, add a state def with a fault-entry "
                "transition and an emergency action def inside the responsible part."
            ),
        }
        return guidance.get(
            criterion_name,
            f"Review the design elements related to '{criterion_name}' and refine the weakest parts.",
        )

