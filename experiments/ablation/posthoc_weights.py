"""W-UNIFORM arm - offline replay of the recommendation step (zero LLM cost).

The search is weight-free (it returns a Pareto front); weights enter only when
one design is recommended from that front. This replay re-runs the
recommendation on the fronts saved by FULL runs under two weight vectors:

  nominal - requirement-derived weights, recomputed with the production
            function (``variation_dse._recommendation_weights``);
  uniform - 1/n per objective (the ablation).

and reports whether the pick flips, plus each pick's weight-simplex
robustness. Fidelity check: when production picked by weights alone
(``recommended_by != "datasheet"``), the replayed nominal pick must match the
run's recorded ``best_config``, and a mismatch is flagged rather than
reported. Only variation-mode fronts are replayed; the bilevel arm recommends
through ``pipeline_adapter``, so pooling the two would compare unlike things.

Usage:
    python experiments/ablation/posthoc_weights.py <campaign_dir> [--arm FULL]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.dse.variation_dse import _recommendation_weights  # noqa: E402
from src.dse.weighting import recommend, sensitivity  # noqa: E402


def _state_key(state: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in state.items()))


def replay_report(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One run's replay record, or None when the run has no replayable front."""
    front_entries = report.get("pareto_alternatives") or []
    requirements = report.get("requirements") or []
    if not front_entries or not requirements:
        return None
    kwargs = (report.get("ablation") or {}).get(
        "effective_pipeline_kwargs"
    ) or {}
    if kwargs.get("dse_mode", "variation") != "variation":
        return None

    members = [
        (dict(entry["parameters"]), dict(entry["scores"]))
        for entry in front_entries
    ]
    labels = {
        _state_key(entry["parameters"]): entry.get("name")
        or json.dumps(entry["parameters"], sort_keys=True)
        for entry in front_entries
    }

    def label(state: Dict[str, Any]) -> str:
        return labels.get(_state_key(state), json.dumps(state, sort_keys=True))

    names = list(members[0][1].keys())
    nominal_weights = _recommendation_weights(names, requirements)
    uniform_weights = {name: 1.0 / len(names) for name in names}

    nominal_pick = label(
        recommend(members, nominal_weights, method="chebyshev")[0]
    )
    uniform_pick = label(
        recommend(members, uniform_weights, method="chebyshev")[0]
    )
    simplex = sensitivity(
        members, nominal_weights, label,
        n_samples=2000, method="chebyshev", random_seed=0,
    )

    recommended_by = report.get("recommended_by")
    best_config = report.get("best_config") or {}
    replay_matches_production: Optional[bool] = None
    if recommended_by != "datasheet" and best_config:
        replay_matches_production = (
            _state_key(best_config)
            == _state_key(recommend(
                members, nominal_weights, method="chebyshev"
            )[0])
        )

    return {
        "front_size": len(members),
        "objectives": names,
        "nominal_weights": nominal_weights,
        "nominal_pick": nominal_pick,
        "uniform_pick": uniform_pick,
        "flipped": nominal_pick != uniform_pick,
        "nominal_robustness": simplex.nominal_robustness,
        "uniform_pick_simplex_share": simplex.selection_frequency.get(
            uniform_pick, 0.0
        ),
        "selection_frequency": simplex.selection_frequency,
        "production_recommended_by": recommended_by,
        "production_used_datasheet_override": recommended_by == "datasheet",
        "replay_matches_production": replay_matches_production,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--arm", default="FULL",
                        help="which arm's saved fronts to replay (default FULL)")
    args = parser.parse_args()

    runs_dir = args.campaign_dir / "runs"
    reports = sorted(runs_dir.glob(f"{args.arm}_seed*.report.json"))
    if not reports:
        print(f"✗ no {args.arm} run reports under {runs_dir}")
        return 2

    records: List[Dict[str, Any]] = []
    for path in reports:
        report = json.loads(path.read_text())
        replay = replay_report(report)
        row: Dict[str, Any] = {"report": path.name}
        if replay is None:
            row["status"] = "NO_REPLAYABLE_FRONT"
        else:
            row.update({"status": "REPLAYED", **replay})
        records.append(row)

    replayed = [r for r in records if r["status"] == "REPLAYED"]
    flips = sum(1 for r in replayed if r["flipped"])
    mismatches = [
        r["report"] for r in replayed
        if r.get("replay_matches_production") is False
    ]
    out = {
        "arm_replayed": args.arm,
        "runs_total": len(records),
        "runs_replayed": len(replayed),
        "recommendation_flips": flips,
        "flip_rate": (flips / len(replayed)) if replayed else None,
        "replay_production_mismatches": mismatches,
        "records": records,
        "claim_boundary": (
            "Replay changes the recommendation step only; the searched front "
            "is identical by construction. A flip means the weight scheme "
            "matters for the delivered design; no-flip on a small front is "
            "weak evidence either way — report front sizes alongside."
        ),
    }
    out_path = args.campaign_dir / "w_uniform_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str),
                        encoding="utf-8")

    for row in records:
        if row["status"] != "REPLAYED":
            print(f"{row['report']}: {row['status']}")
            continue
        flag = "FLIP" if row["flipped"] else "same"
        note = (" [datasheet override in production]"
                if row["production_used_datasheet_override"] else "")
        print(f"{row['report']}: front={row['front_size']}  "
              f"nominal={row['nominal_pick']}  uniform={row['uniform_pick']}  "
              f"→ {flag}  (nominal robustness "
              f"{row['nominal_robustness']:.1%}){note}")
    if mismatches:
        print(f"\n⚠ REPLAY_MISMATCH in {len(mismatches)} run(s): replayed "
              "nominal pick ≠ production best_config although production "
              "picked by weights — investigate before citing this arm: "
              f"{mismatches}")
    print(f"\nflips: {flips}/{len(replayed)}  →  {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
