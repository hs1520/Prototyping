"""Cross-arm robustness report — the core results table, computed without gold.

Aggregates the three mechanisms the claim rests on: safety-pattern conformance,
blackboard-controlled context access, and requirement traceability. It consumes
archived committed models and coordination metrics only, so it needs no frozen
gold, no blind labels and no readiness gate — those protect the pooling of
gold-based accuracy, and there is none here.

The property these tests exist for is the honesty of the *absent* cells. An arm
without an A/G layer must read "not applicable", never 0.0: the A/G layer is the R2
intervention, so scoring R0/R1 against it would manufacture a difference out of a
definition. A run that archived no model at all is a third state again — missing
evidence, not a result.
"""
from __future__ import annotations

import json

import pytest

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.robustness_report import (
    assert_reads_no_gold,
    build_robustness_report,
    format_robustness_table,
)

_REQ = "deploy the ballistic recovery parachute within 0.5 s"
_BASE = f"package DeliveryUAV {{ requirement def REQ_SAFE_005 {{ doc /* {_REQ} */ }} }}"
_WITH_AG = _BASE + "\n\n" + emit_ag_package(REQ_SAFE_005_CHAIN)


def _pilot(tmp_path, arms: dict) -> str:
    """arms maps arm name -> model text, or None to archive no model."""
    for seed in ("seed-0", "seed-1"):
        for arm, model in arms.items():
            run = tmp_path / seed / arm
            run.mkdir(parents=True)
            (run / "coordination_metrics.json").write_text(json.dumps({
                "context_revision_consistency": {"value": 1.0},
                "stale_revision_use": {"count": 0},
            }))
            if model is not None:
                (run / "shared_model_final.sysml").write_text(model)
    return str(tmp_path)


def test_an_arm_without_an_ag_layer_is_not_applicable_not_zero(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R1-BBCTX": _BASE, "R2-BBAG": _WITH_AG}),
        declared_requirements=["REQ_SAFE_005"],
    )
    baseline = report["by_arm"]["R1-BBCTX"]
    assert baseline["runs_with_ag_layer"] == 0
    assert "mean_trace_completeness" not in baseline
    assert "not applicable rather than zero" in baseline["note"]

    intervention = report["by_arm"]["R2-BBAG"]
    assert intervention["mean_trace_completeness"] == 1.0
    assert intervention["pattern_pass"] == intervention["runs_with_ag_layer"]


def test_a_run_that_archived_no_model_is_missing_evidence_not_a_result(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R0-CURRENT": None, "R2-BBAG": _WITH_AG}),
        declared_requirements=["REQ_SAFE_005"],
    )
    run = next(r for r in report["runs"] if r["arm"] == "R0-CURRENT")
    assert run["model_archived"] is False
    assert "ag_layer_present" not in run
    assert "missing " in run["note"]
    assert report["by_arm"]["R0-CURRENT"]["models_archived"] == 0


def test_the_table_marks_unmeasurable_cells_rather_than_zeroing_them(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R1-BBCTX": _BASE, "R2-BBAG": _WITH_AG}),
        declared_requirements=["REQ_SAFE_005"],
    )
    table = format_robustness_table(report)
    baseline_row = next(
        line for line in table.splitlines() if line.startswith("R1-BBCTX")
    )
    assert "n/a" in baseline_row
    assert "0.00" not in baseline_row
    assert "not a score of zero" in table


def test_the_coordination_pillar_is_carried_through(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R2-BBAG": _WITH_AG}), declared_requirements=[]
    )
    assert report["runs"][0]["coordination"]["context_revision_consistency"]


def test_gold_can_never_enter_the_report(tmp_path):
    from pathlib import Path

    with pytest.raises(ValueError, match="gold"):
        assert_reads_no_gold([Path("x/human_frozen/gold.REQ_SAFE_005.json")])
    with pytest.raises(ValueError, match="not a permitted input"):
        assert_reads_no_gold([Path("x/blind_labels.json")])


def test_it_declares_it_is_not_an_accuracy_or_physical_claim(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R2-BBAG": _WITH_AG}), declared_requirements=[]
    )
    assert "no human gold" in report["measurement_boundary"]
    assert "NOT correctness" in report["metric_interpretation"]
    assert "physical system" in report["metric_interpretation"]


def test_the_archived_pilot_reproduces_the_measured_separation():
    """The real evidence, not a fixture: R2 is the only arm carrying an A/G layer,
    and it passes and traces fully on every seed."""
    from pathlib import Path

    pilot = Path("examples/output/revised_pilot_20260723_freeze2")
    if not pilot.exists():  # the evidence directory is gitignored
        pytest.skip("archived pilot not present")
    report = build_robustness_report(pilot, declared_requirements=["REQ_SAFE_005"])
    assert report["by_arm"]["R0-CURRENT"]["models_archived"] == 0
    assert report["by_arm"]["R1-BBCTX"]["runs_with_ag_layer"] == 0
    r2 = report["by_arm"]["R2-BBAG"]
    assert r2["runs_with_ag_layer"] == r2["runs"] == 3
    assert r2["pattern_pass"] == 3
    assert r2["mean_trace_completeness"] == 1.0
