from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

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
    before = orchestrator._runtime_board.records(topic="pipeline.runtime.state")
    orchestrator.last_functional_closure = {"status": "OPEN"}
    orchestrator.last_functional_closure = {"status": "CLOSED"}

    state = orchestrator._runtime_board.latest_typed(
        "pipeline.runtime.state", PipelineRuntimeState
    )
    assert state.functional_closure == {"status": "CLOSED"}
    records = orchestrator._runtime_board.records(topic="pipeline.runtime.state")
    assert len(records) == len(before) + 2
    assert records[-1].payload["parent_record_id"] == records[-2].record_id
    assert records[-1].payload["changed_field"] == "functional_closure"
    open_state = orchestrator._runtime_board.typed_value(
        records[-2].record_id, PipelineRuntimeState
    )
    assert open_state.functional_closure == {"status": "OPEN"}
    with pytest.raises(FrozenInstanceError):
        open_state.functional_closure = {"status": "MUTATED"}
    assert not [
        name for name in vars(orchestrator)
        if name.startswith("last_") or name.startswith("_active_")
    ]


def test_runtime_state_history_is_bounded_to_one_generation_run():
    orchestrator = Orchestrator(_NoCallLLM())
    orchestrator.last_functional_closure = {"status": "CLOSED"}
    previous_board = orchestrator._runtime_board

    orchestrator._init_pipeline_state()

    assert len(previous_board.records(topic="pipeline.runtime.state")) > 1
    assert len(orchestrator._runtime_board.records(
        topic="pipeline.runtime.state"
    )) == 1
    assert orchestrator.last_functional_closure is None


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


def test_generation_sources_declare_their_agent_role_explicitly():
    """The role must not be re-derived from a substring of the source name.

    `pre_ag_simulation` is the case that made the old rule fragile: it contains
    "ag_" and is correctly an AssuranceAgent source, so the substring test
    happened to agree. Any future name that merely contains those characters
    would not, and the role travels into every activation record in the run
    artefacts.
    """
    import re
    from pathlib import Path

    source = Path("src/agents/generation_pipeline.py").read_text()
    # Strip comments: the rule being guarded against is quoted in one, and the
    # point is that it must not be live code.
    code = "\n".join(
        re.sub(r"#.*$", "", line) for line in source.splitlines()
    )
    assert '"ag_" not in name' not in code, (
        "agent role is being derived from the source name again"
    )
    block = source[source.index("phases = ("):source.index("sources = []")]
    rows = re.findall(r'\("([a-z_]+)",\s*(_ORCH|_ASSUR),', block)
    assert len(rows) == 21
    assurance = {name for name, role in rows if role == "_ASSUR"}
    assert assurance == {
        "ag_generation_planning",
        "pre_ag_simulation",
        "ag_contract_reconciliation",
        "ag_semantic_assurance",
        "ag_non_degradation",
    }


def test_runtime_control_board_never_commits_a_model():
    """The controller only sees topics at the board's current revision.

    Committing a model onto the runtime board would advance that revision and
    hide every phase topic published before it, stalling the chain. Nothing may
    call commit_model on `_runtime_board`.
    """
    from pathlib import Path

    for path in Path("src").rglob("*.py"):
        text = path.read_text()
        assert "_runtime_board.commit_model" not in text, (
            f"{path} commits a model onto the runtime control board"
        )
