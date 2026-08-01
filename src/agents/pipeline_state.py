"""Board-backed runtime state compatibility for pipeline sources."""
from __future__ import annotations

from typing import Any

from .pipeline_records import (
    AGGenerationPlanRecord,
    ModelGenerationPlanRecord,
    PipelineRuntimeState,
)


_RUNTIME_TOPIC = "pipeline.runtime.state"


class PipelineStateMixin:
    def _init_pipeline_state(self) -> None:
        from ..prototyping.blackboard import Blackboard, RecordType

        self._runtime_board = Blackboard("pipeline-runtime")
        self._runtime_board.publish_typed(
            RecordType.CONTROL,
            _RUNTIME_TOPIC,
            "Orchestrator",
            {"record_schema": "PipelineRuntimeState", "status": "ACTIVE"},
            PipelineRuntimeState(),
        )

    @property
    def _pipeline_state(self) -> PipelineRuntimeState:
        return self._runtime_board.latest_typed(
            _RUNTIME_TOPIC, PipelineRuntimeState
        )

    @property
    def _active_ag_generation_plan(self):
        record = self._pipeline_state.ag_generation_plan
        return None if record is None else record.plan

    @_active_ag_generation_plan.setter
    def _active_ag_generation_plan(self, value) -> None:
        self._pipeline_state.ag_generation_plan = (
            None if value is None else AGGenerationPlanRecord(value)
        )

    @property
    def _active_model_generation_plan(self):
        record = self._pipeline_state.model_generation_plan
        return None if record is None else record.plan

    @_active_model_generation_plan.setter
    def _active_model_generation_plan(self, value) -> None:
        self._pipeline_state.model_generation_plan = (
            None if value is None else ModelGenerationPlanRecord(value)
        )


_STATE_FIELDS = {
    "last_recommended_design": "recommended_design",
    "last_pareto_designs": "pareto_designs",
    "last_recommended_bindings": "recommended_bindings",
    "last_recommended_realizable": "recommended_realizable",
    "last_realizable_front_count": "realizable_front_count",
    "last_recommended_by": "recommended_by",
    "last_recommended_estimator_feasible": "recommended_estimator_feasible",
    "last_recommendation_status": "recommendation_status",
    "last_recommendable_front_count": "recommendable_front_count",
    "last_exploratory_design": "exploratory_design",
    "last_exploratory_pareto_alternatives": "exploratory_pareto_alternatives",
    "last_constraint_counts": "constraint_counts",
    "last_search_coverage": "search_coverage",
    "last_variation_proposal_source": "variation_proposal_source",
    "last_functional_closure": "functional_closure",
    "last_verification_anchor_attempts": "verification_anchor_attempts",
    "last_requirement_semantic_analysis": "requirement_semantic_analysis",
    "last_requirement_input": "requirement_input",
    "last_ag_authoring_attempts": "ag_authoring_attempts",
    "last_ag_binding_report": "ag_binding_report",
    "last_ag_non_degradation": "ag_non_degradation",
    "last_estimator_calibration": "estimator_calibration",
}


def _state_property(field_name: str) -> property:
    def get(instance: PipelineStateMixin) -> Any:
        return getattr(instance._pipeline_state, field_name)

    def set_(instance: PipelineStateMixin, value: Any) -> None:
        setattr(instance._pipeline_state, field_name, value)

    return property(get, set_)


for _public_name, _field_name in _STATE_FIELDS.items():
    setattr(PipelineStateMixin, _public_name, _state_property(_field_name))
