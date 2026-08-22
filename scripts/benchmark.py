"""Multi-system × multi-seed benchmark harness (addresses the n=1 criticism).

Runs the full PrototypingPipeline over a suite of system specifications and
random seeds, saves one JSON run report per run (via
``PrototypingPipeline.build_run_report``), and aggregates mean/std of the
headline metrics so a single lucky run can't carry a claim.

Usage
-----
    # harness plumbing smoke (MockLLM can't produce a real design — the run is
    # recorded as failed, which exercises capture/aggregate/save end to end):
    .venv/bin/python scripts/benchmark.py --provider mock --systems drone --seeds 1

    # real benchmark (uses your provider keys from .env):
    .venv/bin/python scripts/benchmark.py --provider vertex --seeds 3 \
        --systems drone smart_building

Outputs ``logs/benchmark_<timestamp>.json`` with per-run records + aggregates.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from src.utils.digest import sha256_text

# ── Benchmark suite ─────────────────────────────────────────────────────────
# Each spec is deliberately self-contained (name, description, requirements)
# so results are reproducible from this file alone.

SYSTEMS: Dict[str, Dict[str, Any]] = {
    "drone": {
        "system_name": "DeliveryDrone",
        "description": (
            "An autonomous multirotor drone for last-mile package delivery. It "
            "navigates GPS waypoints, avoids obstacles, monitors battery state, "
            "returns to base on low charge, and releases packages at the target."
        ),
        "requirements": [
            "REQ-FUNC-001: The drone shall navigate to GPS waypoints with < 1 m precision",
            "REQ-FUNC-002: The drone shall release the package within 0.2 m of the target",
            "REQ-PERF-001: The drone shall sustain a cruise speed of at least 15 m/s",
            "REQ-PERF-002: The drone shall achieve at least 25 minutes of endurance at maximum rated payload",
            "REQ-SAFE-001: The drone shall return to base if battery charge drops below 25%. [SEV:Hazardous]",
            "REQ-SAFE-002: The drone shall deploy a parachute on complete motor failure. [SEV:Catastrophic]",
            "REQ-CONS-001: The drone total mass shall not exceed 25 kg",
            "REQ-INTF-001: The drone shall report telemetry over MAVLink at 1 Hz",
        ],
    },
    "smart_building": {
        "system_name": "SmartBuildingBMS",
        "description": (
            "A building management system for a 20-floor office building that "
            "controls HVAC from occupancy, manages lighting, monitors energy, "
            "and alerts operators to equipment faults."
        ),
        "requirements": [
            "REQ-FUNC-001: The BMS shall maintain indoor temperature at 21±1°C in occupied zones",
            "REQ-FUNC-002: The BMS shall control lighting based on occupancy and daylight",
            "REQ-PERF-001: The BMS shall respond to manual overrides within 2 seconds",
            "REQ-SAFE-001: The BMS shall maintain emergency lighting on power failure. [SEV:Major]",
            "REQ-SAFE-002: The BMS shall unlock all exit doors during a fire alarm. [SEV:Catastrophic]",
            "REQ-INTF-001: The BMS shall integrate with the city's smart grid API",
            "REQ-CONS-001: The BMS shall keep energy consumption at least 20% below baseline",
        ],
    },
    "agv": {
        "system_name": "WarehouseAGV",
        "description": (
            "An automated guided vehicle for warehouse pallet transport. It "
            "follows planned routes, detects and stops for humans, docks to "
            "charging stations autonomously, and lifts pallets up to 1 tonne."
        ),
        "requirements": [
            "REQ-FUNC-001: The AGV shall follow planned routes with < 5 cm lateral deviation",
            "REQ-FUNC-002: The AGV shall dock and recharge autonomously below 20% charge",
            "REQ-PERF-001: The AGV shall travel at up to 2 m/s when loaded",
            "REQ-SAFE-001: The AGV shall stop within 0.5 m when a human enters its path. [SEV:Catastrophic]",
            "REQ-SAFE-002: The AGV shall sound an alarm and stop on lift mechanism overload. [SEV:Hazardous]",
            "REQ-CONS-001: The AGV shall carry payloads up to 1000 kg",
            "REQ-INTF-001: The AGV shall report position to the fleet manager over WiFi at 2 Hz",
        ],
    },
}

HEADLINE_METRICS = ("final_score", "reachability", "iterations", "llm_calls", "llm_total_tokens")


def _archive_failure(
    error: BaseException,
    out_dir: Path,
    spec: Dict[str, Any],
    seed: int,
    record: Dict[str, Any],
) -> None:
    """Persist the rejected model and the evidence that rejected it.

    A fail-closed verdict is correct, but it kills the run before any artifact
    is written, so the exact revision that failed used to be lost and only the
    requirement ids survived. The gates already attach their evidence to the
    error; this writes it down.
    """
    model_text = getattr(error, "terminal_model_text", None)
    closure = getattr(error, "functional_closure", None)
    rejections = getattr(error, "plan_conformance_rejections", None)
    provenance = getattr(error, "generation_plan_provenance", None)
    if (
        model_text is None and closure is None
        and rejections is None and provenance is None
    ):
        return

    failed_dir = out_dir / "failed_runs"
    failed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec['system_name']}_seed{seed}"
    if isinstance(model_text, str) and model_text:
        model_path = failed_dir / f"{stem}.sysml"
        model_path.write_text(model_text, encoding="utf-8")
        record["failed_model_path"] = str(model_path)
        record["failed_model_digest"] = sha256_text(model_text)
    evidence = {
        "system": spec["system_name"],
        "seed": seed,
        "error": f"{type(error).__name__}: {error}",
        "functional_closure": closure,
        "plan_conformance_rejections": rejections,
        "generation_plan_provenance": provenance,
    }
    evidence_path = failed_dir / f"{stem}.evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8"
    )
    record["failed_evidence_path"] = str(evidence_path)
    if isinstance(closure, dict):
        record["functional_closure_status"] = closure.get("status")
        record["remaining_functional_gaps"] = list(
            closure.get("remaining_gap_req_ids") or ()
        )
    if isinstance(rejections, list):
        record["plan_conformance_rejections"] = len(rejections)
    if isinstance(provenance, dict):
        record["plan_status"] = provenance.get("plan_status")
        record["step1_plan_retries"] = provenance.get("step1_plan_retries")


def run_one(
    provider: str,
    spec: Dict[str, Any],
    seed: int,
    dse_mode: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """One benchmark run → flat record (never raises; failures are recorded)."""
    from src.app.pipeline import PrototypingPipeline
    from src.prototyping.provider_factory import create_llm

    started = time.time()
    record: Dict[str, Any] = {
        "system": spec["system_name"],
        "seed": seed,
        "provider": provider,
        "dse_mode": dse_mode,
    }
    try:
        provider_kwargs = (
            {"seed": seed}
            if provider.strip().lower() in {"vertex", "gemini"}
            else None
        )
        llm = create_llm(
            provider=provider,
            provider_kwargs=provider_kwargs,
        )
        record.update({
            "llm_model": getattr(llm, "model", None),
            "provider_seed": getattr(llm, "seed", None),
            "langsmith_enabled": bool(
                getattr(llm, "langsmith_enabled", False)
            ),
        })
        pipeline = PrototypingPipeline(llm=llm, dse_mode=dse_mode)
        gen = pipeline.generate_system(
            system_name=spec["system_name"],
            description=spec["description"],
            frozen_requirements=spec["requirements"],
        )
        result = pipeline.explore_design_space(gen, mcts_seed=seed)
        report = pipeline.build_run_report(result)
        closure = result.get("functional_closure") or {}
        qualification = result.get("model_qualification") or {}
        record.update({
            "ok": True,
            "final_score": report.get("final_score"),
            "reachability": (report.get("simulation") or {}).get("reachability_score"),
            "iterations": report.get("iterations"),
            "pareto_size": len(report.get("pareto_alternatives") or []),
            "best_config": report.get("best_config"),
            "llm_calls": (report.get("llm_usage") or {}).get("calls"),
            "llm_total_tokens": (report.get("llm_usage") or {}).get("total_tokens"),
            "qualification_status": qualification.get("status"),
            "functional_closure_status": closure.get("status"),
            "remaining_functional_gaps": list(
                closure.get("remaining_gap_req_ids") or ()
            ),
            "verification_anchor_attempts": len(
                result.get("verification_anchor_attempts") or ()
            ),
        })
    except Exception as e:
        record.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        try:
            _archive_failure(e, out_dir, spec, seed, record)
        except Exception as archive_error:   # never mask the real failure
            record["archive_error"] = f"{type(archive_error).__name__}: {archive_error}"
    record["elapsed_s"] = round(time.time() - started, 1)
    return record


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-system mean/std over successful runs — the anti-lucky-run table."""
    out: Dict[str, Any] = {}
    for system in sorted({r["system"] for r in records}):
        runs = [r for r in records if r["system"] == system and r.get("ok")]
        agg: Dict[str, Any] = {"runs_ok": len(runs),
                               "runs_failed": sum(1 for r in records
                                                  if r["system"] == system and not r.get("ok"))}
        for metric in HEADLINE_METRICS:
            vals = [float(r[metric]) for r in runs if r.get(metric) is not None]
            if vals:
                agg[metric] = {
                    "mean": round(statistics.fmean(vals), 4),
                    "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
        out[system] = agg
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="mock", help="LLM provider (mock/gemini/vertex/github_copilot)")
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds per system")
    ap.add_argument("--dse-mode", default="variation", choices=["variation", "bilevel"])
    ap.add_argument("--out-dir", default="logs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for name in args.systems:
        for seed in range(args.seeds):
            print(f"\n=== benchmark: {name} seed={seed} ({args.provider}, {args.dse_mode}) ===")
            records.append(
                run_one(args.provider, SYSTEMS[name], seed, args.dse_mode, out_dir)
            )

    result = {
        "provider": args.provider,
        "dse_mode": args.dse_mode,
        "seeds_per_system": args.seeds,
        "records": records,
        "aggregate": aggregate(records),
    }
    path = out_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))

    print("\n=== aggregate ===")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"\nsaved: {path}")
    return 0 if all(r.get("ok") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
