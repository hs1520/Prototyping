"""Diagnose real-catalog coverage against the supported DSE design domain.

Reports both operational matching and an exact-cell counterfactual, so a
non-empty match is not read as full catalog coverage.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.dse.gazebo_oracle import SUPPORTED_ROTOR_COUNTS
from src.dse.physics_estimator import DesignInputs
from src.realization.catalog import ComponentCatalog, DEFAULT_CATALOG
from src.realization.closure import close_the_loop
from src.realization.closure_types import requirement_verdicts_met
from src.realization.matcher import all_combinations, match


CELLS = (4, 6)
PROP_DIAMETERS_IN = (8, 10, 12, 15, 18, 22)
# Domain follows the catalog rather than a frozen 22Ah upper bound. Sub-3Ah
# racing packs are outside this heavy-payload grid but stay in the inventory.
CAPACITIES_MAH = tuple(sorted({
    int(pack.capacity_mah) for pack in DEFAULT_CATALOG.packs
    if 3000.0 <= pack.capacity_mah <= 30000.0
}))
MISSION_REQUIREMENTS = (
    "REQ-FUNC-003: payload gross mass up to 1.5 kg",
    "REQ-PERF-002: minimum endurance of 25 minutes with maximum rated payload",
    "REQ-CONS-003: maximum take-off mass shall not exceed 8.0 kg",
)


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def historical_runs(paths: Iterable[str]) -> dict:
    verdicts: Counter = Counter()
    best_configs = []
    realized = []
    total = 0
    for raw_path in sorted(paths):
        path = Path(raw_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            realized.append({"file": str(path), "read_error": f"{type(exc).__name__}: {exc}"})
            continue
        total += 1
        best_configs.append({"file": path.name, "best_config": data.get("best_config", {})})
        report = data.get("realization")
        if not isinstance(report, dict):
            continue
        verdict = str(report.get("verdict", "UNKNOWN"))
        verdicts[verdict] += 1
        chosen = report.get("chosen") or {}
        realized.append({
            "file": path.name,
            "verdict": verdict,
            "failed_checks": report.get("failed_checks", []),
            "combo": chosen.get("combo"),
            "pack": chosen.get("pack"),
            "frame": chosen.get("frame"),
            "design_drift": chosen.get("design_drift", []),
        })
    return {
        "reports_scanned": total,
        "reports_with_realization": sum(verdicts.values()),
        "realization_verdicts": _counter_dict(verdicts),
        "best_configs": best_configs,
        "realized_runs": realized,
    }


def _cell_catalog(cells: int) -> ComponentCatalog:
    return ComponentCatalog(
        combos=tuple(combo for combo in DEFAULT_CATALOG.combos if combo.cells == cells),
        packs=tuple(pack for pack in DEFAULT_CATALOG.packs if pack.cells == cells),
        frames=DEFAULT_CATALOG.frames,
        integration_bundles=DEFAULT_CATALOG.integration_bundles,
    )


def _architecture_conditioned_failures(
    design: DesignInputs, catalog: ComponentCatalog
) -> tuple[str, ...]:
    """Explain interface failure without letting a wrong-arm frame mask physics.

    The production report picks the globally nearest candidate. For catalog
    planning, hold cells/arms/prop fit fixed first, or an octocopter failing only
    ``arms_match`` hides a quad's hover-throttle shortfall.
    """
    candidates = all_combinations(design, list(MISSION_REQUIREMENTS), catalog)
    structural = {"cells_match", "arms_match", "prop_fits", "integration_architecture"}
    compatible = [
        candidate for candidate in candidates
        if all(check.passed for check in candidate.checks if check.name in structural)
    ]
    if not compatible:
        if not candidates:
            return ("empty_catalog_axis",)
        best = min(candidates, key=lambda c: (sum(not x.passed for x in c.checks), c.distance))
        return tuple(check.name for check in best.checks if not check.passed)
    best = min(compatible, key=lambda c: (sum(not x.passed for x in c.checks), c.distance))
    return tuple(check.name for check in best.checks if not check.passed)


def coverage_grid() -> dict:
    totals: Counter = Counter()
    by_rotor: dict[int, Counter] = {n: Counter() for n in SUPPORTED_ROTOR_COUNTS}
    by_cells: dict[int, Counter] = {cells: Counter() for cells in CELLS}
    operational_failures: Counter = Counter()
    exact_cell_failures: Counter = Counter()
    operational_conditioned_failures: Counter = Counter()
    exact_cell_conditioned_failures: Counter = Counter()
    chosen_combos: Counter = Counter()
    chosen_packs: Counter = Counter()
    chosen_frames: Counter = Counter()
    drift: Counter = Counter()
    closure_false_negative_examples = []

    for rotor_count in SUPPORTED_ROTOR_COUNTS:
        for cells in CELLS:
            exact_catalog = _cell_catalog(cells)
            for diameter_in in PROP_DIAMETERS_IN:
                for capacity_mah in CAPACITIES_MAH:
                    design = DesignInputs(
                        payload_mass_kg=1.5,
                        battery_capacity_mah=float(capacity_mah),
                        battery_cells=cells,
                        rotor_count=rotor_count,
                        rotor_radius_m=diameter_in * 0.0254 / 2.0,
                        cruise_speed_mps=0.0,
                    )
                    operational_matches = match(design, list(MISSION_REQUIREMENTS))
                    exact_matches = match(design, list(MISSION_REQUIREMENTS), exact_catalog)
                    report = close_the_loop(design, [], list(MISSION_REQUIREMENTS))
                    exact_report = close_the_loop(
                        design, [], list(MISSION_REQUIREMENTS), exact_catalog
                    )

                    buckets = (totals, by_rotor[rotor_count], by_cells[cells])
                    for bucket in buckets:
                        bucket["points"] += 1
                        bucket["operational_match"] += bool(operational_matches)
                        bucket["exact_cell_match"] += bool(exact_matches)
                        bucket[f"operational_{report.verdict}"] += 1
                        bucket[f"exact_cell_{exact_report.verdict}"] += 1

                    operational_failures.update(ch.name for ch in report.failed_checks)
                    exact_cell_failures.update(ch.name for ch in exact_report.failed_checks)
                    if not operational_matches:
                        operational_conditioned_failures.update(
                            _architecture_conditioned_failures(design, DEFAULT_CATALOG)
                        )
                    if not exact_matches:
                        exact_cell_conditioned_failures.update(
                            _architecture_conditioned_failures(design, exact_catalog)
                        )
                    if report.verdict == "INFEASIBLE_REALIZATION":
                        closing_alternatives = [
                            candidate for candidate in operational_matches
                            if requirement_verdicts_met(design, candidate.metrics,
                                                        list(MISSION_REQUIREMENTS))
                        ]
                        if closing_alternatives:
                            totals["closure_false_negative"] += 1
                            first = closing_alternatives[0]
                            if len(closure_false_negative_examples) < 5:
                                closure_false_negative_examples.append({
                                    "design": asdict(design),
                                    "chosen_nonclosing": None if report.chosen is None else {
                                        "combo": report.chosen.rd.combo.name,
                                        "pack": report.chosen.rd.pack.name,
                                    },
                                    "available_closing": {
                                        "combo": first.rd.combo.name,
                                        "pack": first.rd.pack.name,
                                        "endurance_min": first.metrics.endurance_min,
                                    },
                                })
                    if report.chosen is None:
                        continue
                    chosen_combos[report.chosen.rd.combo.name] += 1
                    chosen_packs[report.chosen.rd.pack.name] += 1
                    chosen_frames[report.chosen.rd.frame.name] += 1
                    design_drift = {item.name: item for item in report.chosen.design_drift}
                    if design_drift["battery_cells"].delta:
                        drift["battery_cells_changed"] += 1
                    if abs(design_drift["rotor_radius_m"].relative_delta) > 0.10:
                        drift["rotor_radius_over_10pct"] += 1
                    if abs(design_drift["battery_capacity_mah"].relative_delta) > 0.10:
                        drift["battery_capacity_over_10pct"] += 1

    return {
        "domain": {
            "rotor_counts": list(SUPPORTED_ROTOR_COUNTS),
            "cells": list(CELLS),
            "prop_diameters_in": list(PROP_DIAMETERS_IN),
            "capacities_mah": list(CAPACITIES_MAH),
            "payload_kg": 1.5,
            "endurance_min": 25.0,
            "mtom_kg": 8.0,
        },
        "totals": _counter_dict(totals),
        "by_rotor": {str(key): _counter_dict(value) for key, value in by_rotor.items()},
        "by_cells": {str(key): _counter_dict(value) for key, value in by_cells.items()},
        "operational_failed_checks": _counter_dict(operational_failures),
        "exact_cell_failed_checks": _counter_dict(exact_cell_failures),
        "operational_architecture_conditioned_failures": _counter_dict(
            operational_conditioned_failures
        ),
        "exact_cell_architecture_conditioned_failures": _counter_dict(
            exact_cell_conditioned_failures
        ),
        "drift": _counter_dict(drift),
        "closure_false_negative_examples": closure_false_negative_examples,
        "chosen_components": {
            "combos": _counter_dict(chosen_combos),
            "packs": _counter_dict(chosen_packs),
            "frames": _counter_dict(chosen_frames),
        },
    }


def catalog_inventory() -> dict:
    return {
        "combos": [asdict(combo) for combo in DEFAULT_CATALOG.combos],
        "packs": [asdict(pack) for pack in DEFAULT_CATALOG.packs],
        "frames": [asdict(frame) for frame in DEFAULT_CATALOG.frames],
        "integration_bundles": [
            asdict(bundle) for bundle in DEFAULT_CATALOG.integration_bundles
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs/run_*.json", help="historical run-report glob")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    result = {
        "history": historical_runs(glob.glob(args.logs)),
        "coverage": coverage_grid(),
        "catalog_inventory": catalog_inventory(),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
