"""Gazebo flight verification for run3's recommended design (bundle leg).

Reconstructs the DSE-recommended DesignInputs from the archived
``DseDesignAnalysis.recommendedDesign`` block in the final model — the run's
own committed record — flies it in Gazebo (hover + single-motor-out when the
redundancy requirement is present), and writes the verdict plus the resulting
requirement-coverage upgrade next to the run's other artifacts.

Run (NOT concurrently with native SITL work — both bind tcp:5760):
    .venv/bin/python experiments/gazebo_leg_run3.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

MODEL_PATH = REPO / "examples" / "output" / "final_model.sysml"
OUT_PATH = REPO / "examples" / "output" / "run3_gazebo_verify_20260831.json"


def recommended_design():
    from src.dse.physics_estimator import DesignInputs

    text = MODEL_PATH.read_text()
    block = re.search(
        r"part recommendedDesign \{(.*?)\}", text, re.S
    )
    if block is None:
        raise SystemExit("final model carries no recommendedDesign block")
    attrs = dict(re.findall(
        r"attribute (\w+) : Real = ([0-9.]+);", block.group(1)
    ))
    return DesignInputs(
        payload_mass_kg=float(attrs["addedMassKg"]),
        battery_capacity_mah=float(attrs["capacityMah"]),
        battery_cells=int(float(attrs["cells"])),
        rotor_count=int(float(attrs["rotorCount"])),
        rotor_radius_m=float(attrs["rotorRadiusM"]),
    )


def main() -> int:
    from drone_system_v2 import DRONE_REQUIREMENTS
    from gazebo_poc.gazebo_verify import summary_line, verify_recommended_design
    from src.dse.requirement_coverage import (
        classify_requirement_coverage, coverage_summary,
    )

    design = recommended_design()
    print(f"[gazebo-leg] design: {design}")
    started = time.time()
    verdict = verify_recommended_design(design, DRONE_REQUIREMENTS)
    wall_s = time.time() - started
    print(f"[gazebo-leg] {summary_line(verdict)}")

    final_text = MODEL_PATH.read_text()
    coverage = classify_requirement_coverage(
        final_text, DRONE_REQUIREMENTS, dynamic=True, gazebo=verdict,
    )
    print(f"[gazebo-leg] {coverage_summary(coverage)}")

    OUT_PATH.write_text(json.dumps({
        "design": {
            "payload_mass_kg": design.payload_mass_kg,
            "battery_capacity_mah": design.battery_capacity_mah,
            "battery_cells": design.battery_cells,
            "rotor_count": design.rotor_count,
            "rotor_radius_m": design.rotor_radius_m,
        },
        "wall_s": round(wall_s, 1),
        "verdict": verdict,
        "summary": summary_line(verdict),
        "coverage_with_gazebo": coverage,
        "coverage_summary": coverage_summary(coverage),
    }, indent=2, ensure_ascii=False, default=str))
    print(f"[gazebo-leg] wrote {OUT_PATH}")
    return 0 if verdict.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
