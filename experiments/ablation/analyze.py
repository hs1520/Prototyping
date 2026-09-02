"""Aggregate an ablation campaign: per-arm stats + per-seed paired deltas.

Statistical honesty boundary (see README.md): with 3 seeds per cell only
descriptive statistics are reported.  Score deltas are indicative, never
significance claims; the load-bearing comparisons are categorical (success
rate, gate rejections, intervention counts, token cost), where effect sizes
are large enough for n=3 to carry weight.

Usage:
    python experiments/ablation/analyze.py <campaign_dir>   # regenerate summary
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HEADLINE_METRICS = (
    "final_score",
    "generate_phase_score",
    "reachability",
    "controlled_pass_rate",
    "iterations",
    "llm_calls",
    "llm_total_tokens",
    "llm_live_calls",
    "llm_live_tokens",
    "replayed_calls",
    "replayed_tokens",
    "elapsed_s",
    # mechanism ledger (was the ablated layer exercised, and how much?)
    "mech_refine_iterations",
    "mech_plan_retries",
    "mech_tier0_fixes",
    "mech_syntax_gate_llm_attempts",
    "mech_det_connectivity_fixes",
    "mech_surgical_refinement_accepted",
    "mech_surgical_exit_passes",
    "mech_closure_initial_gaps",
    "mech_closure_attempts",
    "mech_closure_accepted",
    "mech_closure_full_rewrite_passes",
    "mech_closure_surgical_passes",
    "mech_plan_det_additions",
    "mech_plan_conformance_rejections",
)

#: Categorical gate outcomes counted over ALL runs of an arm (failed runs
#: stay in the denominator: a rejected model is the arm's result, not noise).
GATE_OUTCOMES = {
    "qualified": lambda r: r.get("qualification_status") == "QUALIFIED",
    "closure_closed": lambda r: r.get("functional_closure_status") == "CLOSED",
    "paths_7of7": lambda r: r.get("controlled_pass") == 7,
}

#: Metrics worth a per-seed paired comparison against FULL.
PAIRED_METRICS = (
    "final_score",
    "generate_phase_score",
    "controlled_pass_rate",
    "llm_calls",
    "llm_total_tokens",
    "elapsed_s",
)


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm mean/std over successful runs — the anti-lucky-run table."""
    out: Dict[str, Any] = {}
    for arm in sorted({record["arm"] for record in records}):
        ok_runs = [r for r in records if r["arm"] == arm and r.get("ok")]
        failed = [r for r in records if r["arm"] == arm and not r.get("ok")]
        # An infrastructure failure (rate limit, transport) is the
        # environment's, not the arm's: it must not masquerade as a quality
        # failure, especially in a single-seed wave where one 429 would be
        # an arm's entire failure rate.
        infra = [r for r in failed if r.get("infrastructure_failure")]
        genuine = [r for r in failed if not r.get("infrastructure_failure")]
        counted = ok_runs + genuine
        entry: Dict[str, Any] = {
            "runs_ok": len(ok_runs),
            "runs_failed": len(genuine),
            "runs_infra_failed": len(infra),
            "failed_seeds": sorted(r["seed"] for r in genuine),
            "infra_failed_seeds": sorted(r["seed"] for r in infra),
            "runs_counted": len(counted),
            "gates": {
                name: sum(1 for r in counted if predicate(r))
                for name, predicate in GATE_OUTCOMES.items()
            },
        }
        for metric in HEADLINE_METRICS:
            values = [
                float(r[metric]) for r in ok_runs
                if r.get(metric) is not None
            ]
            if values:
                entry[metric] = {
                    "mean": round(statistics.fmean(values), 4),
                    "std": (
                        round(statistics.stdev(values), 4)
                        if len(values) > 1 else 0.0
                    ),
                    "n": len(values),
                }
        out[arm] = entry
    return out


def paired_deltas(
    records: List[Dict[str, Any]], baseline: str = "FULL"
) -> Dict[str, Any]:
    """Per-seed (arm − baseline) deltas, pairing only seeds where both ran ok.

    Pairing on shared seeds removes cross-seed variance from the comparison;
    a seed where either side failed is excluded from the delta but counted.
    """
    by_arm_seed: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for record in records:
        by_arm_seed.setdefault(record["arm"], {})[record["seed"]] = record
    base = by_arm_seed.get(baseline, {})
    out: Dict[str, Any] = {}
    for arm, runs in sorted(by_arm_seed.items()):
        if arm == baseline:
            continue
        entry: Dict[str, Any] = {"paired_seeds": [], "unpaired_seeds": []}
        deltas: Dict[str, List[float]] = {m: [] for m in PAIRED_METRICS}
        for seed, record in sorted(runs.items()):
            reference = base.get(seed)
            if not (record.get("ok") and reference and reference.get("ok")):
                entry["unpaired_seeds"].append(seed)
                continue
            entry["paired_seeds"].append(seed)
            for metric in PAIRED_METRICS:
                a, b = record.get(metric), reference.get(metric)
                if a is not None and b is not None:
                    deltas[metric].append(round(float(a) - float(b), 4))
        for metric, values in deltas.items():
            if values:
                entry[f"delta_{metric}"] = {
                    "per_seed": values,
                    "mean": round(statistics.fmean(values), 4),
                }
        out[arm] = entry
    return out


def _fmt(cell: Optional[Dict[str, Any]], decimals: int = 3) -> str:
    if not cell:
        return "—"
    mean, std = cell["mean"], cell["std"]
    if abs(mean) >= 1000:
        return f"{mean:,.0f} ± {std:,.0f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def render_markdown(
    manifest: Dict[str, Any],
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Ablation campaign `{manifest.get('campaign')}`\n")
    lines.append(f"- system: {manifest.get('system')} "
                 f"(frozen requirements, digest "
                 f"`{str(manifest.get('requirements_digest'))[:12]}…`)")
    lines.append(f"- provider: {manifest.get('provider')}  ·  seeds: "
                 f"{manifest.get('seeds')}  ·  commit: "
                 f"`{str(manifest.get('git_commit'))[:12]}`"
                 f"{' **(dirty)**' if manifest.get('git_dirty') else ''}")
    lines.append(f"- base pipeline kwargs: "
                 f"`{json.dumps(manifest.get('base_pipeline_kwargs'))}`  ·  "
                 f"mcts_iterations: {manifest.get('mcts_iterations')}\n")

    lines.append("## Gate outcomes (all counted runs; genuine failures stay "
                 "in the denominator)\n")
    lines.append("| arm | runs | qualified | closure CLOSED | 7/7 role paths "
                 "| reachability | total tokens |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm, entry in summary["aggregate"].items():
        n = entry.get("runs_counted", 0)
        gates = entry.get("gates") or {}
        lines.append(
            f"| {arm} | {n} "
            f"| {gates.get('qualified', 0)}/{n} "
            f"| {gates.get('closure_closed', 0)}/{n} "
            f"| {gates.get('paths_7of7', 0)}/{n} "
            f"| {_fmt(entry.get('reachability'))} "
            f"| {_fmt(entry.get('llm_total_tokens'))} |"
        )
    lines.append("")

    lines.append("## Mechanism ledger (successful runs; mean ± std)\n")
    lines.append(
        "Whether the layer an arm removes was exercised. A zero row in FULL "
        "means the matching ablation is uninformative on that trajectory."
    )
    lines.append("")
    lines.append("| arm | refine iters | Tier-0 fixes | Tier-1 syntax LLM "
                 "| det. connectivity | surgical accepted | surgical exit "
                 "passes | closure gaps | closure att./acc. | plan det. adds |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm, entry in summary["aggregate"].items():
        def m(key: str, decimals: int = 1) -> str:
            return _fmt(entry.get(key), decimals)
        lines.append(
            f"| {arm} | {m('mech_refine_iterations')} | {m('mech_tier0_fixes')} "
            f"| {m('mech_syntax_gate_llm_attempts')} "
            f"| {m('mech_det_connectivity_fixes')} "
            f"| {m('mech_surgical_refinement_accepted')} "
            f"| {m('mech_surgical_exit_passes')} "
            f"| {m('mech_closure_initial_gaps')} "
            f"| {m('mech_closure_attempts')} / {m('mech_closure_accepted')} "
            f"| {m('mech_plan_det_additions')} |"
        )
    lines.append("")

    replayed = [r for r in records if r.get("replayed_calls")]
    if replayed:
        lines.append("## Prefix replay (trajectory-matched arms)\n")
        lines.append("| arm | seed | replayed calls | replayed tokens | live "
                     "calls | live tokens | total tokens |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(replayed, key=lambda x: (x["arm"], x["seed"])):
            lines.append(
                f"| {r['arm']} | {r['seed']} | {r.get('replayed_calls')} "
                f"| {r.get('replayed_tokens'):,} | {r.get('llm_live_calls')} "
                f"| {r.get('llm_live_tokens'):,} | {r.get('llm_total_tokens'):,} |"
            )
        lines.append("")

    lines.append("## Per-arm aggregate (successful runs)\n")
    lines.append("| arm | ok/failed | final_score | controlled 7-scenario "
                 "pass rate | reachability | LLM calls | total tokens | "
                 "wall s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, entry in summary["aggregate"].items():
        lines.append(
            f"| {arm} | {entry['runs_ok']}/{entry['runs_failed']} "
            f"| {_fmt(entry.get('final_score'))} "
            f"| {_fmt(entry.get('controlled_pass_rate'))} "
            f"| {_fmt(entry.get('reachability'))} "
            f"| {_fmt(entry.get('llm_calls'), 1)} "
            f"| {_fmt(entry.get('llm_total_tokens'))} "
            f"| {_fmt(entry.get('elapsed_s'), 0)} |"
        )
    lines.append("")

    lines.append("## Paired per-seed deltas vs FULL (arm − FULL)\n")
    lines.append("| arm | paired seeds | Δ final_score | Δ controlled pass "
                 "| Δ LLM calls | Δ total tokens |")
    lines.append("|---|---|---|---|---|---|")
    for arm, entry in summary["paired_vs_baseline"].items():
        def cell(metric: str) -> str:
            data = entry.get(f"delta_{metric}")
            if not data:
                return "—"
            return f"{data['mean']:+.3f} ({data['per_seed']})"
        lines.append(
            f"| {arm} | {entry['paired_seeds']} "
            f"| {cell('final_score')} | {cell('controlled_pass_rate')} "
            f"| {cell('llm_calls')} | {cell('llm_total_tokens')} |"
        )
    lines.append("")

    failures = [r for r in records if not r.get("ok")]
    if failures:
        lines.append("## Failed runs\n")
        for record in failures:
            kind = (
                "infrastructure — excluded from the arm's failure rate"
                if record.get("infrastructure_failure") else "arm failure"
            )
            lines.append(f"- **{record['arm']}** seed {record['seed']} "
                         f"({kind}): {record.get('error')} "
                         f"(evidence: {record.get('failed_evidence_path', '—')})")
        lines.append("")

    lines.append("## Reading guide\n")
    lines.append(
        "n per cell is small: means ± std and per-seed deltas are "
        "**descriptive**; no significance is claimed. Categorical outcomes "
        "(failure counts, gate rejections, LLM-call and token costs) carry "
        "the load-bearing comparisons; score deltas are indicative only. "
        "A genuine failed run stays in the denominator — failure is data, "
        "not noise; an infrastructure failure (rate limit, transport) is "
        "reported separately and re-run rather than charged to the arm. "
        "final_score is measured at different pipeline stages per arm (DSE "
        "arms re-score the enriched terminal snapshot; non-DSE arms keep "
        "the generate-phase score), so cross-arm score comparisons must "
        "use generate_phase_score; final_score differences between "
        "DSE-bearing and DSE-less arms reflect the measurement point and "
        "the terminal enrichment layer, not generation quality. Under "
        "--prefix-from-baseline every non-FULL arm replays FULL's archived "
        "call prefix for the same seed and goes live at the first request "
        "the ablated component changes: differences are attributable to the "
        "component, not to provider sampling; total tokens include the "
        "replayed prefix so effort is comparable with FULL, while wall time "
        "covers live calls only."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    campaign_dir = Path(sys.argv[1])
    manifest = json.loads((campaign_dir / "campaign.json").read_text())
    records = [
        json.loads(line)
        for line in (campaign_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = {
        "aggregate": aggregate(records),
        "paired_vs_baseline": paired_deltas(records),
    }
    (campaign_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (campaign_dir / "summary.md").write_text(
        render_markdown(manifest, records, summary), encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2, default=str))
    print(f"regenerated: {campaign_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
