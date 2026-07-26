"""Increment-4 evaluation core: coordination metrics + artifact writing."""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.prototyping.blackboard import Blackboard
from src.prototyping.context_builder import ContextBuilder
from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.run_artifacts import write_revised_run_artifacts
from src.prototyping.run_metrics import compute_coordination_metrics
from src.prototyping.task_session import TaskSessionRegistry
from src.sysml.lite_model import build_lite_model
from src.utils.sysml_text_utils import get_sysml_text


class _NoCallLLM:
    def complete(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("LLM should not be called")


_REQS = [
    "REQ-SAFE-005: deploy the parachute within 0.5 s of a critical propulsion "
    "failure.",
]


def _r2_run_result():
    """Drive the real R2 orchestrator seam and assemble a run-result dict."""
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", _REQS)
    model = build_lite_model(
        "package DeliveryUAV { requirement def REQ_SAFE_005 { doc /* deploy the "
        "parachute within 0.5 s of a critical propulsion failure. */ } "
        "part def SafetyMonitor {} }", model_name="DeliveryUAV"
    )
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}), model
    )
    final_sysml = orch._apply_ag_contract_layer(get_sysml_text(model), _REQS)
    orch._commit_terminal_model(final_sysml, producer="test")
    artifacts = orch._build_collaboration_artifacts(final_sysml)
    return {
        **artifacts,
        "model_sysml": final_sysml,
        "llm_usage": {"calls": 3, "total_tokens": 4200, "elapsed_seconds": 5.0},
    }


def test_coordination_metrics_are_computed_from_run_artifacts():
    result = _r2_run_result()
    m = compute_coordination_metrics(
        result["collaboration"], llm_usage=result["llm_usage"]
    )
    assert m["artifact_role"] == "R1_COORDINATION_METRICS"
    assert m["semantic_authority"] == "COMMITTED_SYSML_MODEL"
    # two migrated handoffs now (Requirements->Design, Design->Verification), each
    # revision-pinned; the metric denominator is no longer an illustrative single.
    assert m["context_revision_consistency"]["value"] == 1.0
    assert m["counts"]["migrated_handoffs"] == 2
    assert m["cross_agent_handoff_completeness"]["value"] == 1.0
    assert m["cross_agent_handoff_completeness"]["illustrative_single_handoff"] is False
    # invariants hold across both roles: no stale-revision use, no contamination
    assert m["stale_revision_use"]["count"] == 0
    assert m["cross_role_contamination"]["count"] == 0
    assert m["source_threshold_unit_preservation"]["value"] == 1.0
    assert m["cost"]["llm_calls"] == 3
    assert m["mvp_caveats"]


def test_write_revised_run_artifacts_serialises_the_producible_views(tmp_path):
    result = _r2_run_result()
    report = PrototypingPipeline.build_run_report(result)
    assert report["pattern_conformance_report"]["verdict"] == "PASS"
    assert report["failure_diagnostics"]["failures"] == []
    assert report["repair_decisions"]["decisions"] == []
    written = write_revised_run_artifacts(result, tmp_path)

    # the producible §14 subset is on disk
    for name in (
        "shared_model_final", "blackboard_snapshot", "model_revision_log",
        "blackboard_event_log", "context_envelopes", "task_sessions",
        "ag_contract_graph", "pattern_conformance_report",
        "failure_diagnostics", "repair_decisions", "coordination_metrics",
    ):
        assert name in written
        assert (tmp_path / written[name].split("/")[-1]).exists()

    # the A/G graph and metrics round-trip as valid JSON
    ag = json.loads((tmp_path / "ag_contract_graph.json").read_text())
    assert ag["verdict"] == "PASS"
    pattern = json.loads(
        (tmp_path / "pattern_conformance_report.json").read_text()
    )
    assert pattern["verdict"] == "PASS"
    failures = json.loads((tmp_path / "failure_diagnostics.json").read_text())
    assert failures["failures"] == []
    metrics = json.loads((tmp_path / "coordination_metrics.json").read_text())
    # two sessions: the DesignAgent handoff and the VerificationAgent handoff
    assert metrics["counts"]["sessions"] == 2
    # the event log is one JSON object per line
    log_lines = (tmp_path / "blackboard_event_log.jsonl").read_text().splitlines()
    assert all(json.loads(line)["topic"] for line in log_lines)


def test_writer_rejects_a_non_revised_run():
    import pytest
    with pytest.raises(ValueError, match="requires a BLACKBOARD_AG_V1 run"):
        write_revised_run_artifacts({"model_sysml": "x"}, "/tmp/nope")
    with pytest.raises(ValueError, match="requires a BLACKBOARD_AG_V1 run"):
        write_revised_run_artifacts({
            "revised_experiment": {
                "experiment_namespace": "LEGACY_EXTERNAL_CONTRACT_V1",
                "configuration": "B2",
            },
            "collaboration": {"blackboard": {}},
        }, "/tmp/nope")


def test_metrics_flag_stale_and_contamination_when_present():
    # A shared-session-style snapshot: two sessions on the same role/task (the
    # R1-LONG failure mode) and a session left on an uncommitted base revision.
    collaboration = {
        "blackboard": {
            "semantic_authority": "COMMITTED_SYSML_MODEL",
            "model_revisions": [{"revision": 0, "model_digest": "d0"}],
            "records": [],
            "tasks": [],
        },
        "contexts": {"envelopes": []},
        "task_sessions": {"sessions": [
            {"session_id": "s1", "agent_role": "DesignAgent", "task_id": "t",
             "status": "COMPLETED", "base_model_revision": 9,
             "base_model_digest": "stale", "used_tokens": 10, "assistant_turns": 1},
            {"session_id": "s2", "agent_role": "DesignAgent", "task_id": "t",
             "status": "COMPLETED", "base_model_revision": 0,
             "base_model_digest": "d0", "used_tokens": 10, "assistant_turns": 1},
        ]},
    }
    m = compute_coordination_metrics(collaboration)
    assert m["stale_revision_use"]["count"] == 1
    assert m["cross_role_contamination"]["count"] == 1


def test_contamination_is_reported_as_a_property_not_as_a_measured_rate():
    """§18-Q2, option B. A session owns one role and one task by construction, so
    contamination cannot occur — 0 is entailed by the design, not observed.

    Reporting it as a coordination rate would restate a definition as a finding,
    which is the same over-claim this project polices elsewhere (an arm with no
    A/G layer is `n/a`, not 0.00). It becomes measurable only under the R1-LONG
    shared-session diagnostic, which is not implemented.
    """
    metrics = compute_coordination_metrics({
        "blackboard": {
            "semantic_authority": "COMMITTED_SYSML_MODEL",
            "model_revisions": [{"revision": 0, "model_digest": "d0"}],
            "records": [], "tasks": [],
        },
        "contexts": {"envelopes": []},
        "task_sessions": {"sessions": [
            {"session_id": "s1", "agent_role": "DesignAgent", "task_id": "t1",
             "status": "COMPLETED", "base_model_revision": 0,
             "base_model_digest": "d0"},
        ]},
    })

    contamination = metrics["cross_role_contamination"]
    assert contamination["kind"] == "ARCHITECTURAL_PROPERTY"
    assert contamination["measured"] is False
    assert "structurally impossible" in contamination["note"]
    assert "R1-LONG" in contamination["note"]

    # and the pillar-2 evidence must still rest on metrics that CAN fail
    for falsifiable in ("context_revision_consistency", "stale_revision_use",
                        "required_context_coverage", "envelope_truncation",
                        "stale_session_detection"):
        assert falsifiable in metrics
        assert metrics[falsifiable].get("kind") != "ARCHITECTURAL_PROPERTY"
