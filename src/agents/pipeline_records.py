"""Typed board records shared by pipeline knowledge sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:
    from .planned_action_lifecycle import (
        PlannedActionObservation,
        PlannedActionPreparation,
    )


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
    weight_sensitivity: Any = None
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
    namespace_repair_attempts: list[Dict[str, Any]] = field(default_factory=list)
    response_conformance_repair_attempts: list[Dict[str, Any]] = field(default_factory=list)
    plan_conformance_rejections: list[Dict[str, Any]] = field(default_factory=list)
    requirement_semantic_analysis: Any = None
    requirement_input: Dict[str, Any] = field(default_factory=dict)
    ag_authoring_attempts: list[Dict[str, Any]] = field(default_factory=list)
    ag_binding_report: Any = None
    ag_non_degradation: Any = None
    planned_action_preparation: Optional["PlannedActionPreparation"] = None
    planned_action_observation: Optional["PlannedActionObservation"] = None
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
    refined_revision: Any = None
    projected_revision: Any = None
    refinement_closure_outcome: Any = None
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
    planned_action_preparation: Optional["PlannedActionPreparation"] = None
    planned_action_observation: Optional["PlannedActionObservation"] = None
    result: Dict[str, Any] = field(default_factory=dict)


def publish_handoff_transition(
    blackboard: Any,
    topic: str,
    producer: str,
    handoff: Any,
    status: str,
) -> None:
    """Terminate a handoff AUDITABLY: mutate the live typed record and publish
    the transition on the same topic.

    The opening ``publish_typed`` snapshots its payload while the handoff is
    ACTIVE and payloads are immutable (digest-bound), so mutating only the
    typed object left every archived event log showing both handoffs as
    permanently ACTIVE — a rejected handoff was indistinguishable from a
    completed one post-hoc. The follow-up record is the blackboard-idiomatic
    fix: history is appended, never edited.
    """
    from ..prototyping.blackboard import RecordType

    handoff.status = status
    blackboard.publish(
        RecordType.CONTROL,
        topic,
        producer,
        {
            "record_schema": f"{type(handoff).__name__}Transition",
            "task_id": handoff.task_id,
            "envelope_id": handoff.envelope_id,
            "session_id": handoff.session_id,
            "status": status,
        },
        task_id=handoff.task_id,
        session_id=handoff.session_id,
    )
