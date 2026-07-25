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


def measure_model(
    model_text: str, declared_requirements: Sequence[str] = ()
) -> Dict[str, Any]:
    """Pattern conformance + traceability for one committed model.

    Public so that a comparison of generation modes goes through exactly the same
    measurement path as the cross-arm table. Two tables computed two ways would not
    be comparable, and the difference would be invisible in the write-up.
    """
    from .ag_contracts import check_ag_graph
    from .ag_extractor import extract_ag_graph, extract_ag_graphs
    from .ag_traceability import compute_traceability

    graphs = extract_ag_graphs(model_text)
    components = sum(len(getattr(g, "components", ()) or ()) for g in graphs)
    if not components:
        return {
            "ag_layer_present": False,
            "note": (
                "no A/G decomposition in the committed model; the A/G layer is "
                "the R2 intervention, so this is the baseline condition rather "
                "than a score of zero"
            ),
        }
    report = check_ag_graph(extract_ag_graph(model_text))
    traceability = compute_traceability(
        graphs,
        realization_links=(report.to_dict().get("graph") or {}).get(
            "realization_links"
        ) or (),
        declared_requirements=declared_requirements,
    )
    return {
        "ag_layer_present": True,
        "pattern_conformance": {
            "verdict": report.verdict,
            "errors": len(report.errors()),
            "codes": sorted({item.code for item in report.errors()}),
        },
        "traceability": {
            "requirements": traceability["requirements"],
            "fully_traced": traceability["fully_traced"],
            "mean_trace_completeness": traceability["mean_trace_completeness"],
            "untraced_requirements": traceability["untraced_requirements"],
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
                    measure_model(model_path.read_text(), declared_requirements)
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
    lines = [
        f"{'arm':<12} {'runs':>4} {'A/G':>4} {'PASS':>5} "
        f"{'mean err':>9} {'trace':>7} {'traced':>7}",
        "-" * 56,
    ]
    for arm, summary in (report.get("by_arm") or {}).items():
        if not summary.get("runs_with_ag_layer"):
            lines.append(
                f"{arm:<12} {summary['runs']:>4} {'—':>4} {'n/a':>5} "
                f"{'n/a':>9} {'n/a':>7} {'n/a':>7}"
            )
            continue
        trace = summary.get("mean_trace_completeness")
        lines.append(
            f"{arm:<12} {summary['runs']:>4} {summary['runs_with_ag_layer']:>4} "
            f"{summary['pattern_pass']:>5} {summary['mean_errors']:>9} "
            f"{(f'{trace:.2f}' if trace is not None else '—'):>7} "
            f"{summary['fully_traced_runs']:>7}"
        )
    lines.append("")
    lines.append("n/a = the arm carries no A/G layer; not a score of zero")
    return "\n".join(lines)
