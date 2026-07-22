"""Run the revised R0/R1/R2 ablation on one frozen requirement set (Increment 4).

Thin driver: for each arm it runs the pipeline's generate() on the SAME frozen
requirements, writes the derived artifacts (§14) for the blackboard arms, and
computes the coordination metrics (§13 Group A). It needs a real provider — the
MockLLM cannot synthesise multi-step SysML — so it is not a CI test; a human runs
it with ``--provider``.

R2-BBAG stays runnable-but-not-poolable: accuracy/F1 against gold is deliberately
NOT computed here (that requires supervisor-frozen human gold, §18-Q7).

Usage:
    PYTHONPATH=. python examples/run_revised_experiment.py \
        --provider vertex --out examples/output/revised_experiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.prototyping.run_artifacts import write_revised_run_artifacts

# One digest-checked frozen requirement set is shared across the three arms.
FROZEN_REQUIREMENTS = [
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses.",
    "REQ-FUNC-002: The system shall detect a stationary obstacle directly ahead "
    "within the forward sensor field of view and maintain at least 5 metres of "
    "separation while avoiding it.",
]

SYSTEM = "DeliveryUAV"
DESCRIPTION = (
    "An autonomous delivery UAV with ballistic parachute recovery and forward "
    "obstacle avoidance."
)

ARMS = ("R0-CURRENT", "R1-BBCTX", "R2-BBAG")


def _run_arm(arm: str, llm, out: Path) -> dict:
    pipe = PrototypingPipeline(
        llm=llm, max_iterations=3, verbose=False,
        revised_experiment_arm=arm,
    )
    result = pipe.orchestrator.generate(
        system_name=SYSTEM,
        system_description=DESCRIPTION,
        frozen_requirements=list(FROZEN_REQUIREMENTS),
    )
    report = pipe.build_run_report(result)
    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written = {}
    if result.get("revised_experiment") and result.get("collaboration"):
        written = write_revised_run_artifacts(result, arm_dir)
    print(f"  [{arm}] score={report.get('final_score')} "
          f"artifacts={len(written)} -> {arm_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--out", default="examples/output/revised_experiment")
    args = parser.parse_args()

    out = Path(args.out)
    llm = create_llm(args.provider)
    print(f"Revised R0/R1/R2 ablation on {len(FROZEN_REQUIREMENTS)} frozen "
          f"requirements (provider={args.provider})")
    summary = {arm: _run_arm(arm, llm, out) for arm in ARMS}
    (out / "summary.json").write_text(
        json.dumps(
            {arm: {"final_score": r.get("final_score"),
                   "configuration": r.get("configuration")}
             for arm, r in summary.items()},
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}/summary.json — note: R2-BBAG F1 needs frozen human gold "
          "(evaluation_ready is False).")


if __name__ == "__main__":
    main()
