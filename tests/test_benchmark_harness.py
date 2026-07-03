"""Smoke tests for the multi-system × multi-seed benchmark harness."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("benchmark", benchmark)
_SPEC.loader.exec_module(benchmark)


def test_suite_specs_are_complete():
    assert len(benchmark.SYSTEMS) >= 3  # multi-system by construction (n>1)
    for spec in benchmark.SYSTEMS.values():
        assert spec["system_name"] and spec["description"]
        assert len(spec["requirements"]) >= 5
        # every suite system carries at least one severity-tagged SAFE requirement
        assert any("[SEV:" in r for r in spec["requirements"])


def test_aggregate_reports_mean_std_per_system():
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
    # None metrics are skipped, not crashed on
    assert agg["B"]["runs_ok"] == 1
    assert "llm_calls" not in agg["B"]
    assert agg["B"]["final_score"]["std"] == 0.0  # single run → no variance claim
