"""Board-backed runtime state compatibility for pipeline sources."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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

        # Runtime-state retention is per generation run: keep every immutable
        # revision within the current bounded pipeline so failures and retries
        # have provenance, then replace the whole runtime board when the next
        # generate() call starts.  This prevents cross-run log growth without
        # discarding the history needed to audit one result.
        #
        # INVARIANT: nothing commits a model onto this board.  `commit_model` is
        # called only on the collaboration board and in `ag_repair`, never here,
        # so this board's model revision stays at its initial value for the whole
        # of one `generate()`.  The controller depends on that: it only sees
        # topics published at the board's current revision, so advancing the
        # revision mid-run would hide every phase topic published before the
        # commit and stall the chain.  `run(require_all=True)` at the call site
        # turns that stall into an error rather than a silently short run, but
        # the invariant is the actual protection — do not commit models here.
        self._runtime_board = Blackboard("pipeline-runtime")
        self._runtime_board.publish_typed(
            RecordType.CONTROL,
            _RUNTIME_TOPIC,
            "Orchestrator",
            {
                "record_schema": "PipelineRuntimeState",
                "status": "ACTIVE",
                "runtime_revision": 0,
                "parent_record_id": None,
                "changed_field": None,
            },
            PipelineRuntimeState(),
        )

    def _publish_pipeline_state(self, field_name: str, value: Any) -> None:
        from ..prototyping.blackboard import RecordType

        previous = self._runtime_board.records(topic=_RUNTIME_TOPIC)[-1]
        state = replace(
            self._pipeline_state,
            **{field_name: deepcopy(value)},
        )
        self._runtime_board.publish_typed(
            RecordType.CONTROL,
            _RUNTIME_TOPIC,
            "Orchestrator",
            {
                "record_schema": "PipelineRuntimeState",
                "status": "ACTIVE",
                "runtime_revision": int(
                    previous.payload.get("runtime_revision", 0)
                ) + 1,
                "parent_record_id": previous.record_id,
                "changed_field": field_name,
            },
            state,
        )

    def _append_pipeline_state_list(self, field_name: str, value: Any) -> int:
        values = deepcopy(list(getattr(self._pipeline_state, field_name)))
        values.append(value)
        self._publish_pipeline_state(field_name, values)
        return len(values) - 1

    def _replace_pipeline_state_list_item(
        self,
        field_name: str,
        index: int,
        value: Any,
    ) -> None:
        values = deepcopy(list(getattr(self._pipeline_state, field_name)))
        values[index] = value
        self._publish_pipeline_state(field_name, values)

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
        self._publish_pipeline_state("ag_generation_plan", (
            None if value is None else AGGenerationPlanRecord(value)
        ))

    @property
    def _active_model_generation_plan(self):
        record = self._pipeline_state.model_generation_plan
        return None if record is None else record.plan

    @_active_model_generation_plan.setter
    def _active_model_generation_plan(self, value) -> None:
        self._publish_pipeline_state("model_generation_plan", (
            None if value is None else ModelGenerationPlanRecord(value)
        ))


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
    "last_plan_conformance_rejections": "plan_conformance_rejections",
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
        instance._publish_pipeline_state(field_name, value)

    return property(get, set_)


for _public_name, _field_name in _STATE_FIELDS.items():
    setattr(PipelineStateMixin, _public_name, _state_property(_field_name))
