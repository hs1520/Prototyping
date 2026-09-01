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
    "elapsed_s",
)

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
        entry: Dict[str, Any] = {
            "runs_ok": len(ok_runs),
            "runs_failed": len(genuine),
            "runs_infra_failed": len(infra),
            "failed_seeds": sorted(r["seed"] for r in genuine),
            "infra_failed_seeds": sorted(r["seed"] for r in infra),
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
        "the terminal enrichment layer, not generation quality."
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
