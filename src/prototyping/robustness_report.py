"""Cross-arm robustness report, computed post-hoc and without human gold.

Aggregates the three mechanisms the robustness claim rests on, per archived run:

1. safety-pattern conformance — the gold-blind checker's verdict on the committed
   model;
2. blackboard-controlled context access — the coordination metrics already archived
   with the run;
3. requirement → implementation traceability — walked from the committed model.

None of it compares against a reviewed answer, so this needs no frozen gold, no
blind labels and no readiness gate: those exist to protect the pooling of
gold-based accuracy, and there is none here. `assert_reads_no_gold` makes that
checkable rather than merely asserted in prose.

Three states are reported distinctly, because collapsing them would misrepresent
the baseline:

``model_archived = False``
    The run did not archive its committed model, so nothing model-derived can be
    measured. This is a gap in the evidence, not a result.
``ag_layer_present = False``
    The run archived a model that carries no A/G decomposition. R0/R1 are like this
    by construction — the A/G layer *is* the R2 intervention. Reporting 0.0 here
    would score an arm against a mechanism it does not have; the asymmetry is the
    comparison, not a defeat.
otherwise
    Measured values.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROBUSTNESS_REPORT_SCHEMA_VERSION = "1.0"

#: Filenames this report is allowed to read. Gold and blind labels are absent by
#: construction, not by convention.
_ALLOWED_ARTIFACTS = ("shared_model_final.sysml", "coordination_metrics.json")


def assert_reads_no_gold(paths: Sequence[Path]) -> None:
    """Fail closed if a gold or blind-label artifact reaches this report."""
    for path in paths:
        name = path.name.lower()
        if name.startswith("gold.") or "human_frozen" in str(path).lower():
            raise ValueError(
                f"{path} is evaluator-only gold; the robustness report is "
                "gold-free and must never consume it"
            )
        if path.name not in _ALLOWED_ARTIFACTS:
            raise ValueError(f"{path.name} is not a permitted input")


#: A planned obligation the committed model asks an executor to discharge.
#: Counted from the model text so this stays inside the report's two-artifact
#: boundary; the executor's own pass counts live in `run_report.json`, which
#: this report is not permitted to read.
_PLAN_CONSTRAINT_MARKER = "// PLAN-CONSTRAINT"
_STATE_EXECUTION_MARKER = "verification=STATE_EXECUTION"


def count_committed_obligations(model_text: str) -> Dict[str, int]:
    """How much a committed model asks to be checked against it.

    Reported beside conformance because a rate alone rewards conservatism: a run
    that plans one executable constraint and discharges it scores better than one
    that plans four and discharges two, though the second committed to more.
    Measured on pilot_n6_20260802, where seed-3 carried four plan constraints and
    every other seed one or two — and seed-3 was the only run in its arm to lose
    the qualification gate. Without this column that reads as a worse run.

    Counted for every arm, not only those carrying an A/G layer, so the baseline
    is visible as the zero it actually is rather than as an absent measurement.
    """
    return {
        "plan_constraints": model_text.count(_PLAN_CONSTRAINT_MARKER),
        "state_execution_obligations": model_text.count(_STATE_EXECUTION_MARKER),
        "asserted_constraints": model_text.count("assert constraint"),
    }


def measure_model(
    model_text: str,
    declared_requirements: Sequence[str] = (),
    out_of_scope: Mapping[str, str] = MappingProxyType({}),
) -> Dict[str, Any]:
    """Pattern conformance + traceability for one committed model.

    Public so that a comparison of generation modes goes through exactly the same
    measurement path as the cross-arm table. Two tables computed two ways would not
    be comparable, and the difference would be invisible in the write-up.
    """
    from .ag_contracts import check_ag_graph
    from .ag_extractor import extract_ag_graphs
    from .ag_traceability import compute_traceability

    graphs = extract_ag_graphs(model_text)
    components = sum(len(getattr(g, "components", ()) or ()) for g in graphs)
    if not components:
        return {
            "ag_layer_present": False,
            "obligations": count_committed_obligations(model_text),
            "note": (
                "no A/G decomposition in the committed model; the A/G layer is "
                "the R2 intervention, so this is the baseline condition rather "
                "than a score of zero"
            ),
        }
    # Per chain, never pooled. A model carrying several A/G packages has several
    # system contracts, so a single extraction leaves the decomposition root
    # ambiguous and manufactures failures: on a three-chain model whose chains each
    # verify PASS, the pooled graph reported FAIL with 14 errors
    # (DECOMPOSITION_MISSING, GUARANTEE_NO_OWNER, REALIZATION_MISSING,
    # ASSUMPTION_UNDISCHARGED) — every one of them an artefact of the pooling. The
    # run verdict is the conjunction, as it is everywhere else.
    # This report replays historical archives whose checker version predates the
    # v8 response-provenance convention. Requiring a field that did not exist at
    # their frozen code revision would rewrite rather than reproduce the recorded
    # measurement. Current pipeline assurance uses the strict default.
    reports = [
        check_ag_graph(
            graph, require_priority_member_provenance=False
        )
        for graph in graphs
    ]
    traceability = compute_traceability(
        graphs,
        realization_links=[
            link
            for report in reports
            for link in ((report.to_dict().get("graph") or {}).get(
                "realization_links"
            ) or ())
        ],
        declared_requirements=declared_requirements,
        out_of_scope=out_of_scope,
    )
    return {
        "ag_layer_present": True,
        "obligations": count_committed_obligations(model_text),
        "pattern_conformance": {
            "verdict": (
                "PASS" if reports and all(
                    item.verdict == "PASS" for item in reports
                ) else "FAIL"
            ),
            "errors": sum(len(item.errors()) for item in reports),
            "codes": sorted({
                diagnostic.code
                for item in reports for diagnostic in item.errors()
            }),
            "chains": {
                str(item.source_requirement or f"chain_{index}"): item.verdict
                for index, item in enumerate(reports)
            },
        },
        "traceability": {
            "requirements": traceability["requirements"],
            "fully_traced": traceability["fully_traced"],
            "mean_trace_completeness": traceability["mean_trace_completeness"],
            "untraced_requirements": traceability["untraced_requirements"],
            "out_of_scope_requirements": traceability["out_of_scope_requirements"],
            "unattributed_chains": len(traceability["unattributed_chains"]),
        },
    }


def _coordination_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    keep = (
        "context_revision_consistency",
        "required_context_coverage",
        "cross_agent_handoff_completeness",
        "stale_revision_use",
        "stale_rejection_rate",
    )
    return {name: data.get(name) for name in keep if data.get(name) is not None}


def build_robustness_report(
    pilot_dir: Path | str,
    *,
    declared_requirements: Sequence[str] = (),
    out_of_scope: Mapping[str, str] = MappingProxyType({}),
) -> Dict[str, Any]:
    """Per-run and per-arm robustness across an archived pilot directory."""
    root = Path(pilot_dir)
    runs: List[Dict[str, Any]] = []
    for seed_dir in sorted(p for p in root.iterdir() if p.name.startswith("seed-")):
        for arm_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            model_path = arm_dir / "shared_model_final.sysml"
            metrics_path = arm_dir / "coordination_metrics.json"
            assert_reads_no_gold(
                [p for p in (model_path, metrics_path) if p.exists()]
            )
            entry: Dict[str, Any] = {
                "seed": seed_dir.name,
                "arm": arm_dir.name,
                "model_archived": model_path.exists(),
                "coordination": _coordination_summary(metrics_path),
            }
            if model_path.exists():
                entry.update(
                    measure_model(
                        model_path.read_text(), declared_requirements, out_of_scope
                    )
                )
            else:
                entry["note"] = (
                    "the run archived no committed model, so pattern conformance "
                    "and traceability cannot be measured; this is missing "
                    "evidence, not a result"
                )
            runs.append(entry)

    by_arm: Dict[str, Dict[str, Any]] = {}
    for arm in sorted({item["arm"] for item in runs}):
        arm_runs = [item for item in runs if item["arm"] == arm]
        measured = [item for item in arm_runs if item.get("ag_layer_present")]
        summary: Dict[str, Any] = {
            "runs": len(arm_runs),
            "models_archived": sum(1 for i in arm_runs if i["model_archived"]),
            "runs_with_ag_layer": len(measured),
        }
        # Deliberately outside the `measured` branch below: an arm that commits
        # to nothing executable must show a zero here, not an absent field.
        committed = [
            item["obligations"]["state_execution_obligations"]
            for item in arm_runs if item.get("obligations")
        ]
        if committed:
            summary["committed_obligations"] = {
                "total": sum(committed),
                "mean": round(mean(committed), 2),
                "min": min(committed),
                "max": max(committed),
                "per_run": committed,
            }
        if measured:
            traces = [
                item["traceability"]["mean_trace_completeness"] for item in measured
                if item["traceability"]["mean_trace_completeness"] is not None
            ]
            summary["pattern_pass"] = sum(
                1 for item in measured
                if item["pattern_conformance"]["verdict"] == "PASS"
            )
            summary["mean_errors"] = round(
                mean(item["pattern_conformance"]["errors"] for item in measured), 3
            )
            summary["mean_trace_completeness"] = (
                round(mean(traces), 4) if traces else None
            )
            # A run counts as fully traced only when EVERY declared requirement is
            # traced. `traceability["fully_traced"]` is a count, so testing it for
            # truthiness marked a run with one of two requirements traced as fully
            # traced — the column then read 3/3 for runs that each covered half the
            # requirement set.
            summary["fully_traced_runs"] = sum(
                1 for item in measured
                if item["traceability"]["requirements"]
                and item["traceability"]["fully_traced"]
                == item["traceability"]["requirements"]
            )
        else:
            summary["note"] = (
                "no run in this arm carries an A/G layer; pattern conformance and "
                "traceability are not applicable rather than zero"
            )
        by_arm[arm] = summary

    return {
        "schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION,
        "artifact_role": "CROSS_ARM_ROBUSTNESS_REPORT",
        "measurement_boundary": (
            "archived committed models and coordination metrics only; no human "
            "gold, no blind labels, no readiness gate required"
        ),
        "metric_interpretation": (
            "presence, conformance and completeness of the generated model; NOT "
            "correctness against a reviewed decomposition, and NOT evidence that "
            "the physical system satisfies the requirement"
        ),
        "declared_requirements": list(declared_requirements),
        "out_of_scope_declaration": dict(out_of_scope),
        "by_arm": by_arm,
        "runs": runs,
    }


def summarise_models(
    named_models: Mapping[str, Sequence[str]],
    *,
    declared_requirements: Sequence[str] = (),
) -> Dict[str, Any]:
    """Summarise labelled groups of committed models — e.g. generation modes.

    Uses `measure_model`, the same path as the cross-arm table, so the two results
    tables in the write-up are computed identically and can be read together. A
    group whose models carry no A/G layer is reported as such rather than as zero,
    for the same reason it is in the cross-arm table.
    """
    groups: Dict[str, Any] = {}
    for label, models in named_models.items():
        measured = [
            measure_model(text, declared_requirements) for text in models
        ]
        with_layer = [item for item in measured if item.get("ag_layer_present")]
        summary: Dict[str, Any] = {
            "models": len(measured),
            "models_with_ag_layer": len(with_layer),
        }
        if with_layer:
            traces = [
                item["traceability"]["mean_trace_completeness"] for item in with_layer
                if item["traceability"]["mean_trace_completeness"] is not None
            ]
            summary.update({
                "pattern_pass": sum(
                    1 for item in with_layer
                    if item["pattern_conformance"]["verdict"] == "PASS"
                ),
                "mean_errors": round(
                    mean(item["pattern_conformance"]["errors"] for item in with_layer),
                    3,
                ),
                "mean_trace_completeness": round(mean(traces), 4) if traces else None,
                # same rule as the cross-arm table: every declared requirement,
                # not merely at least one
                "fully_traced": sum(
                    1 for item in with_layer
                    if item["traceability"]["requirements"]
                    and item["traceability"]["fully_traced"]
                    == item["traceability"]["requirements"]
                ),
            })
        else:
            summary["note"] = (
                "no model in this group carries an A/G layer; not applicable "
                "rather than zero"
            )
        groups[label] = summary
    return {
        "schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION,
        "artifact_role": "GENERATION_MODE_ROBUSTNESS_SUMMARY",
        "measurement_boundary": (
            "committed models only; no human gold, no blind labels, identical "
            "measurement path to the cross-arm report"
        ),
        "metric_interpretation": (
            "presence, conformance and completeness of the generated model; NOT "
            "correctness against a reviewed decomposition"
        ),
        "by_group": groups,
    }


def format_robustness_table(report: Mapping[str, Any]) -> str:
    """The cross-arm table as text, with unmeasurable cells marked, never zeroed."""
    def obligations(summary: Mapping[str, Any]) -> str:
        committed = summary.get("committed_obligations")
        if not committed:
            return "—"
        return f"{committed['mean']:.2f} [{committed['min']}-{committed['max']}]"

    lines = [
        f"{'arm':<12} {'runs':>4} {'A/G':>4} {'PASS':>5} "
        f"{'mean err':>9} {'trace':>7} {'traced':>7} {'obligations':>13}",
        "-" * 70,
    ]
    for arm, summary in (report.get("by_arm") or {}).items():
        if not summary.get("runs_with_ag_layer"):
            lines.append(
                f"{arm:<12} {summary['runs']:>4} {'—':>4} {'n/a':>5} "
                f"{'n/a':>9} {'n/a':>7} {'n/a':>7} {obligations(summary):>13}"
            )
            continue
        trace = summary.get("mean_trace_completeness")
        lines.append(
            f"{arm:<12} {summary['runs']:>4} {summary['runs_with_ag_layer']:>4} "
            f"{summary['pattern_pass']:>5} {summary['mean_errors']:>9} "
            f"{(f'{trace:.2f}' if trace is not None else '—'):>7} "
            f"{summary['fully_traced_runs']:>7} {obligations(summary):>13}"
        )
    lines.append("")
    lines.append("n/a = the arm carries no A/G layer; not a score of zero")
    # Without this column a conformance rate rewards conservatism: the run that
    # commits to the fewest executable obligations has the fewest ways to fail.
    lines.append(
        "obligations = executable state-constraint obligations the committed "
        "model asks to be checked, mean [min-max] per run"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the cross-arm table beside an archived pilot.

    Exists because the report was previously only reachable by hand-written
    script, and four pilots were archived before anyone generated one. It reads
    the pilot's own frozen requirement set, so the declared denominator cannot
    drift from the run it describes.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="defaults to <pilot_dir>/robustness_report.json",
    )
    args = parser.parse_args(argv)

    config_path = args.pilot_dir / "pilot_config.json"
    if not config_path.exists():
        parser.error(f"no pilot_config.json in {args.pilot_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requirements = (
        config.get("frozen_requirement_set", {}).get("requirements") or ()
    )
    declared = [
        item.split(":", 1)[0].strip().replace("-", "_").upper()
        for item in requirements
    ]
    from .ag_traceability import DECLARED_OUT_OF_SCOPE

    report = build_robustness_report(
        args.pilot_dir,
        declared_requirements=declared,
        out_of_scope=DECLARED_OUT_OF_SCOPE,
    )
    out = args.out or (args.pilot_dir / "robustness_report.json")
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(format_robustness_table(report))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
