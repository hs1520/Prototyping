from __future__ import annotations

from dataclasses import replace

from src.agents.orchestrator import Orchestrator
from src.agents.pipeline_records import (
    DesignHandoffRecord,
    GenerationContext,
    PipelineRuntimeState,
)
from src.prototyping.blackboard import RecordType
from src.prototyping.controller import BlackboardController


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("LLM must not be called")


def test_runtime_state_is_a_typed_board_record_not_orchestrator_attributes():
    orchestrator = Orchestrator(_NoCallLLM())
    orchestrator.last_functional_closure = {"status": "CLOSED"}

    state = orchestrator._runtime_board.latest_typed(
        "pipeline.runtime.state", PipelineRuntimeState
    )
    assert state.functional_closure == {"status": "CLOSED"}
    assert not [
        name for name in vars(orchestrator)
        if name.startswith("last_") or name.startswith("_active_")
    ]


def test_design_handoff_is_resolved_from_its_typed_board_record():
    orchestrator = Orchestrator(
        _NoCallLLM(), revised_experiment_arm="R1-BBCTX"
    )
    orchestrator.last_requirement_input = {
        "mode": "frozen", "requirement_set_digest": "digest",
    }
    orchestrator._prepare_design_handoff(
        "DeliveryUAV", ["REQ-FUNC-001: The system shall operate."]
    )

    handoff = orchestrator.blackboard.latest_typed(
        orchestrator.DESIGN_HANDOFF_TOPIC, DesignHandoffRecord
    )
    task, envelope, session = orchestrator._design_handoff_objects(handoff)
    assert task.task_id == handoff.task_id
    assert envelope.envelope_id == handoff.envelope_id
    assert session.session_id == handoff.session_id
    snapshot_record = orchestrator.blackboard.snapshot()["records"][-1]
    assert snapshot_record["payload"]["record_schema"] == "DesignHandoffRecord"


def test_generation_dependency_stops_when_upstream_output_is_removed():
    orchestrator = Orchestrator(_NoCallLLM())
    context = GenerationContext(
        system_name="Test", system_description="", additional_requirements=[],
        parse_strict=None, platform_profile=None, frozen_requirements=None,
    )
    board = orchestrator._runtime_board
    board.publish(
        RecordType.SOURCE, "pipeline.request", "test", {"ready": True}
    )
    controller = BlackboardController(board)
    for source in orchestrator._generation_sources(context):
        outputs = source.output_topics

        def publish(outputs=outputs):
            for topic in outputs:
                board.publish(
                    RecordType.RESULT, topic, "fake-source", {"ready": True}
                )

        if source.name == "pre_ag_simulation":
            # Remove the source's output publication.  Its dependent must remain
            # ineligible even though sources were registered in reverse order.
            source = replace(source, output_topics=(), activate=lambda: None)
        else:
            source = replace(source, activate=publish)
        controller.register(source)

    controller.run()
    activated = [item["knowledge_source"] for item in controller.activation_log]
    assert "pre_ag_simulation" in activated
    assert "ag_contract_reconciliation" not in activated
    assert "generation_reporting" not in activated
