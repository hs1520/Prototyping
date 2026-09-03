from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("benchmark", benchmark)
_SPEC.loader.exec_module(benchmark)


def test_suite_specs_complete():
    assert len(benchmark.SYSTEMS) >= 3
    for spec in benchmark.SYSTEMS.values():
        assert spec["system_name"] and spec["description"]
        assert len(spec["requirements"]) >= 5
        assert any("[SEV:" in r for r in spec["requirements"])


def test_fail_closed_archives_model(tmp_path):
    error = RuntimeError("terminal functional closure is not closed: REQ_FUNC_001")
    error.terminal_model_text = "package Demo { part def X { } }"
    error.functional_closure = {
        "status": "OPEN", "remaining_gap_req_ids": ["REQ_FUNC_001"],
    }
    error.plan_conformance_rejections = [
        {"stage": "FUNCTIONAL_CLOSURE", "pass": 1,
         "issues": ["CommunicationSystem.tlmData is missing"]},
    ]
    error.generation_plan_provenance = {
        "plan_status": "PASS",
        "plan_issues": [],
        "planned_event_symbols": ["WaypointCmdReceived"],
        "step1_plan_retries": 2,
        "step1_plan_attempts": [{"issues": ["REQ_FUNC_001 needs a planned navigate response"]}],
    }
    record = {}

    benchmark._archive_failure(
        error, tmp_path, {"system_name": "DeliveryDrone"}, 1, record
    )

    archived = tmp_path / "failed_runs" / "DeliveryDrone_seed1.sysml"
    assert archived.read_text(encoding="utf-8") == error.terminal_model_text
    assert record["failed_model_digest"] == hashlib.sha256(
        error.terminal_model_text.encode("utf-8")
    ).hexdigest()
    evidence = json.loads(
        (tmp_path / "failed_runs" / "DeliveryDrone_seed1.evidence.json").read_text()
    )
    assert evidence["plan_conformance_rejections"][0]["issues"] == [
        "CommunicationSystem.tlmData is missing"
    ]
    assert record["remaining_functional_gaps"] == ["REQ_FUNC_001"]
    assert evidence["generation_plan_provenance"]["step1_plan_retries"] == 2
    assert record["step1_plan_retries"] == 2 and record["plan_status"] == "PASS"


def test_failure_no_evidence_writes_nothing(tmp_path):
    record = {}

    benchmark._archive_failure(
        ValueError("unrelated"), tmp_path, {"system_name": "A"}, 0, record
    )

    assert not (tmp_path / "failed_runs").exists()
    assert record == {}


def test_aggregate_mean_std_per_system():
    records = [
        {"system": "A", "ok": True, "final_score": 0.8, "reachability": 1.0,
         "iterations": 2, "llm_calls": 10, "llm_total_tokens": 1000},
        {"system": "A", "ok": True, "final_score": 0.9, "reachability": 0.8,
         "iterations": 3, "llm_calls": 12, "llm_total_tokens": 1200},
        {"system": "A", "ok": False, "error": "boom"},
        {"system": "B", "ok": True, "final_score": 0.7, "reachability": None,
         "iterations": 1, "llm_calls": None, "llm_total_tokens": None},
    ]
    agg = benchmark.aggregate(records)
    assert agg["A"]["runs_ok"] == 2 and agg["A"]["runs_failed"] == 1
    assert abs(agg["A"]["final_score"]["mean"] - 0.85) < 1e-9
    assert agg["A"]["final_score"]["n"] == 2
    assert agg["A"]["final_score"]["std"] > 0
    assert agg["B"]["runs_ok"] == 1
    assert "llm_calls" not in agg["B"]
    assert agg["B"]["final_score"]["std"] == 0.0
