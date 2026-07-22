"""Run the frozen revised Option 2 descriptive pilot.

This command performs external paid generation only when the caller passes the
explicit authorization flag. It creates a new output directory and refuses to
overwrite previous evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.prototyping.revised_pilot import RevisedPilotConfig, run_revised_pilot
from src.prototyping.run_artifacts import write_revised_run_artifacts


FROZEN_REQUIREMENTS = (
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses.",
    "REQ-FUNC-002: The system shall detect a stationary obstacle directly ahead "
    "within the forward sensor field of view and maintain at least 5 metres of "
    "separation while avoiding it.",
)


def _git_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "controlled pilot requires a clean worktree so code_revision is complete"
        )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs=3, type=int, default=(0, 1, 2))
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument(
        "--confirm-external-experiment",
        action="store_true",
        help="Confirm that this current invocation is authorised to call a provider.",
    )
    args = parser.parse_args()

    config = RevisedPilotConfig(
        provider=args.provider,
        model=args.model,
        seeds=tuple(args.seeds),
        max_iterations=args.max_iterations,
        code_revision=_git_revision(),
        requirements=FROZEN_REQUIREMENTS,
    )
    manifest = run_revised_pilot(
        config,
        Path(args.out),
        external_execution_authorized=args.confirm_external_experiment,
        llm_factory=create_llm,
        pipeline_factory=PrototypingPipeline,
        artifact_writer=write_revised_run_artifacts,
    )
    print(
        f"Pilot {manifest['status']}: {manifest['completed_run_count']}/"
        f"{manifest['run_count']} runs; manifest={Path(args.out) / 'pilot_manifest.json'}"
    )


if __name__ == "__main__":
    main()
