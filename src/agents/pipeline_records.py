"""Typed board records shared by pipeline knowledge sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class DesignHandoffRecord:
    task_id: str
    envelope_id: str
    session_id: str
    captured_llm_calls: int = 0
    observer_error_count_before: int = 0
    observer_errors: list[str] = field(default_factory=list)
    status: str = "ACTIVE"


@dataclass
class AGPlanningHandoffRecord:
    task_id: str
    envelope_id: str
    session_id: str
    observer_id: Optional[int] = None
    status: str = "ACTIVE"


@dataclass(frozen=True)
class AGGenerationPlanRecord:
    plan: Mapping[str, Any]


@dataclass(frozen=True)
class ModelGenerationPlanRecord:
    plan: Mapping[str, Any]


@dataclass(frozen=True)
class PipelineRuntimeState:
    recommended_design: Any = None
    pareto_designs: list[Any] = field(default_factory=list)
    recommended_bindings: Dict[str, Any] = field(default_factory=dict)
    recommended_realizable: Any = None
    realizable_front_count: Any = None
    recommended_by: Any = None
    recommended_estimator_feasible: Any = None
    recommendation_status: Any = None
    recommendable_front_count: Any = None
    exploratory_design: Any = None
    exploratory_pareto_alternatives: list[Any] = field(default_factory=list)
    constraint_counts: Dict[str, Any] = field(default_factory=dict)
    search_coverage: Dict[str, Any] = field(default_factory=dict)
    variation_proposal_source: Any = None
    functional_closure: Any = None
    verification_anchor_attempts: list[Dict[str, Any]] = field(default_factory=list)
    requirement_semantic_analysis: Any = None
    requirement_input: Dict[str, Any] = field(default_factory=dict)
    ag_authoring_attempts: list[Dict[str, Any]] = field(default_factory=list)
    ag_binding_report: Any = None
    ag_non_degradation: Any = None
    action_semantics_audit: Any = None
    estimator_calibration: Any = None
    ag_generation_plan: Optional[AGGenerationPlanRecord] = None
    model_generation_plan: Optional[ModelGenerationPlanRecord] = None


@dataclass
class GenerationContext:
    system_name: str
    system_description: str
    additional_requirements: list[str]
    parse_strict: Any
    platform_profile: Any
    frozen_requirements: Any
    requirements: list[str] = field(default_factory=list)
    model: Any = None
    final_model: Any = None
    final_score: Any = None
    final_sim: Any = None
    pre_terminal_score: Any = None
    pre_ag_sysml: str = ""
    pre_ag_sim: Any = None
    final_sysml: str = ""
    generation_plan_conformance: Any = None
    collaboration_artifacts: Dict[str, Any] = field(default_factory=dict)
    terminal_consistency: Any = None
    structural_obligation_report: Any = None
    semantic_fidelity_report: Any = None
    model_qualification: Any = None
    verification_plan: Any = None
    assurance_artifacts: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
