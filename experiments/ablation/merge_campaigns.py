"""Merge per-seed records from several campaigns into one analysable campaign."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "ablation"))
from analyze import aggregate, paired_deltas, render_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+", help="<campaign_dir>:<seed,seed,...>")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-root", default=str(REPO / "experiments/ablation/results"))
    args = ap.parse_args()

    records, provenance = [], []
    for spec in args.sources:
        path, _, seeds = spec.partition(":")
        wanted = {int(s) for s in seeds.split(",") if s}
        cdir = Path(path)
        manifest = json.loads((cdir / "campaign.json").read_text())
        taken = 0
        for line in (cdir / "records.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["seed"] in wanted:
                rec["source_campaign"] = manifest["campaign"]
                rec["source_commit"] = manifest.get("git_commit")
                records.append(rec)
                taken += 1
        provenance.append({"campaign": manifest["campaign"], "commit": manifest.get("git_commit"),
                           "seeds": sorted(wanted), "records": taken})
        print(f"{manifest['campaign']}: took {taken} record(s) for seeds {sorted(wanted)}")

    latest = {}
    for rec in records:
        latest[(rec["arm"], rec["seed"])] = rec
    records = sorted(latest.values(), key=lambda r: (r["arm"], r["seed"]))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_root) / f"{stamp}_{args.label}"
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((Path(args.sources[0].partition(":")[0]) / "campaign.json").read_text())
    manifest = {
        **{k: base.get(k) for k in ("system", "requirement_source", "requirements_digest",
                                    "provider", "base_pipeline_kwargs", "mcts_iterations", "arm_registry")},
        "campaign": out.name,
        "merged_from": provenance,
        "seeds": sorted({r["seed"] for r in records}),
        "git_commit": "merged: " + ", ".join(f"{p['campaign']}@{str(p['commit'])[:12]}" for p in provenance),
        "started": stamp,
    }
    (out / "campaign.json").write_text(json.dumps(manifest, indent=2, default=str))
    with open(out / "records.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    summary = {"aggregate": aggregate(records), "paired_vs_baseline": paired_deltas(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "summary.md").write_text(render_markdown(manifest, records, summary))
    print(f"merged {len(records)} record(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
