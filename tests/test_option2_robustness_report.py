"""Cross-arm robustness report - the core results table, computed without gold.

Aggregates safety-pattern conformance, blackboard-controlled context access and
requirement traceability from archived committed models and coordination
metrics, so no frozen gold, blind labels or readiness gate are needed. The tests
cover the absent cells: an arm without an A/G layer reads "not applicable"
rather than 0.0, since the A/G layer is the R2 intervention, and a run that
archived no model is a third state - missing evidence, not a result.
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


def test_arm_without_ag_not_zero(tmp_path):
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


def test_no_model_is_missing_evidence(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R0-CURRENT": None, "R2-BBAG": _WITH_AG}),
        declared_requirements=["REQ_SAFE_005"],
    )
    run = next(r for r in report["runs"] if r["arm"] == "R0-CURRENT")
    assert run["model_archived"] is False
    assert "ag_layer_present" not in run
    assert "missing " in run["note"]
    assert report["by_arm"]["R0-CURRENT"]["models_archived"] == 0


def test_table_marks_unmeasurable_cells(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R1-BBCTX": _BASE, "R2-BBAG": _WITH_AG}),
        declared_requirements=["REQ_SAFE_005"],
    )
    table = format_robustness_table(report)
    baseline_row = next(
        line for line in table.splitlines() if line.startswith("R1-BBCTX")
    )
    assert "n/a" in baseline_row
    # The A/G columns are marked unmeasurable rather than scored. Checked on the
    # slice before the obligations column: an arm committing to zero executable
    # obligations did commit to zero, so matching "0.00" across the whole row
    # would forbid reporting it.
    ag_columns = baseline_row[: baseline_row.rindex("n/a") + len("n/a")]
    assert "0.00" not in ag_columns
    assert "—" in ag_columns
    assert ag_columns.count("n/a") == 4
    assert "not a score of zero" in table


def test_coordination_carried_through(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R2-BBAG": _WITH_AG}), declared_requirements=[]
    )
    assert report["runs"][0]["coordination"]["context_revision_consistency"]


def test_gold_rejected(tmp_path):
    from pathlib import Path

    with pytest.raises(ValueError, match="gold"):
        assert_reads_no_gold([Path("x/human_frozen/gold.REQ_SAFE_005.json")])
    with pytest.raises(ValueError, match="not a permitted input"):
        assert_reads_no_gold([Path("x/blind_labels.json")])


def test_declares_measurement_boundary(tmp_path):
    report = build_robustness_report(
        _pilot(tmp_path, {"R2-BBAG": _WITH_AG}), declared_requirements=[]
    )
    assert "no human gold" in report["measurement_boundary"]
    assert "NOT correctness" in report["metric_interpretation"]
    assert "physical system" in report["metric_interpretation"]


def test_archived_pilot_reproduces():
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


def test_modes_share_measurement_path():
    """`summarise_models` reuses `measure_model`, so a model measured either way gives
    the same answer; two tables computed two ways would not be comparable.
    """
    from src.prototyping.robustness_report import measure_model, summarise_models

    direct = measure_model(_WITH_AG, ["REQ_SAFE_005"])
    grouped = summarise_models(
        {"m": [_WITH_AG]}, declared_requirements=["REQ_SAFE_005"]
    )["by_group"]["m"]
    assert grouped["pattern_pass"] == (
        1 if direct["pattern_conformance"]["verdict"] == "PASS" else 0
    )
    assert grouped["mean_trace_completeness"] == (
        direct["traceability"]["mean_trace_completeness"]
    )


def test_group_without_ag_not_zero():
    from src.prototyping.robustness_report import summarise_models

    out = summarise_models({"baseline": [_BASE]})["by_group"]["baseline"]
    assert out["models_with_ag_layer"] == 0
    assert "mean_trace_completeness" not in out
    assert "rather than zero" in out["note"]


def test_groups_separated_by_conformance():
    from src.prototyping.robustness_report import summarise_models

    broken = "\n".join(
        line for line in _WITH_AG.splitlines()
        if "dependency discharge" not in line.lower()
    )
    out = summarise_models(
        {"good": [_WITH_AG], "broken": [broken]},
        declared_requirements=["REQ_SAFE_005"],
    )["by_group"]
    assert out["good"]["pattern_pass"] == 1
    assert out["broken"]["pattern_pass"] == 0
    assert out["broken"]["mean_errors"] > out["good"]["mean_errors"]


def test_artifacts_carry_matching_ids(tmp_path):
    """Wired into the run, not reconstructed post-hoc.

    A run carries "REQ-SAFE-005: The system shall ..." while the committed contract
    cites REQ_SAFE_005; passing the raw strings through matches nothing and reports
    every requirement as untraced.
    """
    import json as _json

    from src.prototyping.run_artifacts import (
        _declared_requirement_ids,
        write_revised_run_artifacts,
    )

    assert _declared_requirement_ids(
        ["REQ-SAFE-005: The system shall deploy...", "REQ-SAFE-004: other"]
    ) == ["REQ_SAFE_005", "REQ_SAFE_004"]

    written = write_revised_run_artifacts(
        {
            "revised_experiment": {
                "experiment_namespace": "BLACKBOARD_AG_V1",
                "configuration": "R2-BBAG",
            },
            "collaboration": {"blackboard": {}, "contexts": {}, "task_sessions": {}},
            "model_sysml": _WITH_AG,
            "requirements": ["REQ-SAFE-005: deploy the parachute within 0.5 s"],
        },
        tmp_path,
    )
    payload = _json.loads(
        (tmp_path / "requirement_traceability.json").read_text()
    )
    assert payload["declared_requirements"] == ["REQ_SAFE_005"]
    assert payload["traceability"]["untraced_requirements"] == []
    assert payload["traceability"]["fully_traced"] == 1
    assert "no human gold" in payload["measurement_boundary"]
    assert "requirement_traceability" in written


def test_partial_coverage_not_fully_traced(tmp_path):
    """The frozen set has two requirements but only one encoded A/G chain, so every
    run traces half of it. `fully_traced` is a count, and testing it for truthiness
    made the column read 3/3 for runs covering half the set.
    """
    report = build_robustness_report(
        _pilot(tmp_path, {"R2-BBAG": _WITH_AG}),
        declared_requirements=["REQ_SAFE_005", "REQ_FUNC_002"],
    )
    arm = report["by_arm"]["R2-BBAG"]
    assert arm["runs_with_ag_layer"] == 2
    assert arm["fully_traced_runs"] == 0, "half-covered runs are not fully traced"
    assert arm["mean_trace_completeness"] == 0.5
    run = report["runs"][0]
    assert run["traceability"]["untraced_requirements"] == ["REQ_FUNC_002"]


def test_denominator_is_declared_set():
    """Passing only the covered requirement reports 1.00. The declared set is what a
    run was asked to implement.
    """
    from src.prototyping.robustness_report import measure_model

    covered_only = measure_model(_WITH_AG, ["REQ_SAFE_005"])
    full_set = measure_model(_WITH_AG, ["REQ_SAFE_005", "REQ_FUNC_002"])
    assert covered_only["traceability"]["mean_trace_completeness"] == 1.0
    assert full_set["traceability"]["mean_trace_completeness"] == 0.5


def test_multi_chain_measured_per_chain():
    """Several A/G packages mean several system contracts.

    Extracting them as one graph leaves the decomposition root ambiguous and the
    checker reports what it cannot resolve as defects: a three-seed run whose nine
    chains all verified PASS still had `pattern_conformance: FAIL, 14 errors` in
    every traceability artifact.
    """
    from src.prototyping.ag_chains import REQ_SAFE_004_CHAIN, REQ_SAFE_008_CHAIN
    from src.prototyping.robustness_report import measure_model

    base = (
        "package DeliveryUAV { "
        f"requirement def REQ_SAFE_005 {{ doc /* {_REQ} */ }} "
        "requirement def REQ_SAFE_004 { doc /* self-test inhibit */ } "
        "requirement def REQ_SAFE_008 { doc /* locked until authorised */ } }"
    )
    model = base + "\n\n" + "\n\n".join(
        emit_ag_package(chain) for chain in
        (REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN, REQ_SAFE_008_CHAIN)
    )
    measured = measure_model(
        model, ["REQ_SAFE_004", "REQ_SAFE_005", "REQ_SAFE_008"]
    )
    conformance = measured["pattern_conformance"]
    assert conformance["verdict"] == "PASS", conformance
    assert conformance["errors"] == 0
    assert conformance["chains"] == {
        "REQ_SAFE_004": "PASS", "REQ_SAFE_005": "PASS", "REQ_SAFE_008": "PASS",
    }
    assert measured["traceability"]["fully_traced"] == 3


def test_runner_carries_scope_declaration(tmp_path):
    """A requirement the A/G layer is not built to decompose is not a gap.

    The runner passed no declaration, so REQ_FUNC_002 - a continuous control
    envelope with no trigger, deadline or invariant state - was archived as untraced
    (3/4 = 0.75). The declaration travels with the artifact and carries its reason,
    since a shrunk denominator is only reviewable if the reason is; an undeclared
    requirement still counts.
    """
    from src.prototyping.ag_traceability import DECLARED_OUT_OF_SCOPE
    from src.prototyping.run_artifacts import write_revised_run_artifacts

    assert DECLARED_OUT_OF_SCOPE["REQ_FUNC_002"], "declaration must state a reason"

    write_revised_run_artifacts(
        {
            "revised_experiment": {
                "experiment_namespace": "BLACKBOARD_AG_V1",
                "configuration": "R2-BBAG",
            },
            "collaboration": {"blackboard": {}, "contexts": {}, "task_sessions": {}},
            "model_sysml": _WITH_AG,
            "requirements": [
                "REQ-SAFE-005: deploy the parachute within 0.5 s",
                "REQ-FUNC-002: maintain separation while avoiding an obstacle",
                "REQ-FUNC-003: something else entirely",
            ],
        },
        tmp_path,
    )
    artifact = json.loads(
        (tmp_path / "requirement_traceability.json").read_text()
    )

    assert artifact["out_of_scope_declaration"] == {
        "REQ_FUNC_002": DECLARED_OUT_OF_SCOPE["REQ_FUNC_002"]
    }
    traceability = artifact["traceability"]
    assert [item["requirement"] for item in
            traceability["out_of_scope_requirements"]] == ["REQ_FUNC_002"]
    # REQ_FUNC_003 is undeclared, so it still counts as an in-scope gap
    assert traceability["untraced_requirements"] == ["REQ_FUNC_003"]
    assert traceability["fully_traced"] == 1


def test_obligations_counted_every_arm():
    """A conformance rate alone rewards the run that commits to least.

    In pilot_n6_20260802, seed-3 carried four plan constraints where every other
    seed carried one or two, discharged two, and lost the qualification gate.
    Without this column it reads as the worse run rather than the one that asked
    for more checking.
    """
    from src.prototyping.robustness_report import (
        count_committed_obligations,
        measure_model,
    )

    plain = "package P { part def Controller; }"
    assert count_committed_obligations(plain) == {
        "plan_constraints": 0,
        "state_execution_obligations": 0,
        "asserted_constraints": 0,
    }

    committed = (
        "package P {\n"
        "    state def B {\n"
        "        state S {\n"
        "            // PLAN-CONSTRAINT c1 provenance=A_G_GUARANTEE "
        "activation=STATE_ACTIVE verification=STATE_EXECUTION reference=B::S\n"
        "            assert constraint c1 { a == b }\n"
        "        }\n"
        "    }\n"
        "}"
    )
    counted = count_committed_obligations(committed)
    assert counted["plan_constraints"] == 1
    assert counted["state_execution_obligations"] == 1
    assert counted["asserted_constraints"] == 1

    # The baseline carries no A/G layer but still reports the count: an absent
    # measurement and a zero are different claims.
    baseline = measure_model(plain)
    assert baseline["ag_layer_present"] is False
    assert baseline["obligations"]["state_execution_obligations"] == 0
