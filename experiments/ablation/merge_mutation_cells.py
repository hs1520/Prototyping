"""Merge mutation-study cells from several campaigns into one analysable campaign.

Latest record per (arm, set) wins in source order; each record keeps its
source campaign/commit; summary.md is regenerated with mutation_study.render.

Usage:
    python experiments/ablation/merge_mutation_cells.py --label mut_s0_merged \
        <campaign_A> <campaign_B> [...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "ablation"))
from mutation_study import render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+", help="campaign dirs, earliest first; later cells override")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-root", default=str(REPO / "experiments/ablation/results"))
    args = ap.parse_args()

    latest, provenance, manifests = {}, [], []
    for path in args.sources:
        cdir = Path(path)
        manifest = json.loads((cdir / "mutation_manifest.json").read_text())
        manifests.append(manifest)
        taken = []
        for line in (cdir / "records.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["source_campaign"] = manifest["campaign"]
            rec["source_commit"] = manifest.get("git_commit")
            rec["source_runs_dir"] = str(cdir / "runs")
            latest[(rec["arm"], rec["set"])] = rec
            taken.append(f"{rec['arm']}:{rec['set']}")
        provenance.append({"campaign": manifest["campaign"], "commit": manifest.get("git_commit"), "cells": taken})
        print(f"{manifest['campaign']}: {len(taken)} cell(s)")

    set_order = ["CONTROL", "SYNTAX", "CONNECT", "BEHAVIOUR", "ALL"]
    arm_order = ["FULL", "NO-REFINE", "NO-SURGICAL", "NO-DETFIX", "NO-REPAIR"]
    records = sorted(latest.values(), key=lambda r: (set_order.index(r["set"]) if r["set"] in set_order else 99,
                                                     arm_order.index(r["arm"]) if r["arm"] in arm_order else 99))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_root) / f"{stamp}_{args.label}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "runs").mkdir()
    for rec in records:
        stem = f"{rec['arm']}_{rec['set']}"
        for f in Path(rec["source_runs_dir"]).glob(f"{stem}.*"):
            shutil.copy2(f, out / "runs" / f.name)
    base = manifests[0]
    manifest = {
        **{k: base.get(k) for k in ("source", "source_digest", "sets", "provider", "base_pipeline_kwargs")},
        "campaign": out.name,
        "merged_from": provenance,
        "arms": sorted({r["arm"] for r in records}, key=lambda a: arm_order.index(a) if a in arm_order else 99),
        "git_commit": "merged: " + ", ".join(f"{p['campaign']}@{str(p['commit'])[:12]}" for p in provenance),
        "started": stamp,
    }
    (out / "mutation_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    with open(out / "records.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    (out / "summary.md").write_text(render(records, manifest))
    print(f"merged {len(records)} cell(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
