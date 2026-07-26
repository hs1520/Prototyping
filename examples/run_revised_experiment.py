"""Run the frozen revised Option 2 descriptive pilot.

This command performs external paid generation only when the caller passes the
explicit authorization flag. It creates a new output directory and refuses to
overwrite previous evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from src.prototyping.experiment_arms import (
    R2_DETERMINISTIC_GENERATION_MODE,
    R2_INTERVENTION_VERSION_BY_MODE,
)
from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.prototyping.revised_pilot import RevisedPilotConfig, run_revised_pilot
from src.prototyping.run_artifacts import write_revised_run_artifacts


# Three encoded chains covering the three bounded safety patterns, plus one
# requirement that instantiates none of them. REQ-FUNC-002 is a continuous control
# envelope — no trigger, no deadline, no invariant state — and is kept deliberately:
# a declared out-of-scope case is stronger evidence that the method's boundary is
# real than a set containing only requirements it can handle. The texts are the
# authoritative ones the gold drafts cite.
FROZEN_REQUIREMENTS = (
    "REQ-SAFE-004: The system shall not transition to the armed or airborne state "
    "if any onboard sensor reports a failure during the power-on self-test "
    "sequence.",
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses.",
    "REQ-SAFE-008: The payload-release actuator shall default to the mechanically "
    "locked state upon power-on, before any arming or flight authorisation.",
    "REQ-FUNC-002: The system shall detect a stationary obstacle directly ahead "
    "within the forward sensor field of view and maintain at least 5 metres of "
    "separation while avoiding it.",
)

#: The scope declaration lives in `ag_traceability.DECLARED_OUT_OF_SCOPE`, because
#: the runner writes it into every traceability artifact and a second copy here
#: would be a second table to drift. It was declared in this file first and never
#: read by anything; two statements of one design decision is how the reason and
#: the number stop matching.


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
        "--r2-generation-mode",
        default=R2_DETERMINISTIC_GENERATION_MODE,
        choices=sorted(R2_INTERVENTION_VERSION_BY_MODE),
        help=(
            "Which R2 intervention to execute. Each mode is a separate frozen "
            "intervention whose version is bound to it; results from different "
            "modes must never be pooled, so give each its own --out directory."
        ),
    )
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
        # must be explicit: the default selects REQ_SAFE_005 alone, so adding
        # requirements to the frozen set would otherwise leave the extra chains
        # unselected and silently unexercised
        selected_ag_chain_ids=("REQ_SAFE_004", "REQ_SAFE_005", "REQ_SAFE_008"),
        r2_generation_mode=args.r2_generation_mode,
        # bound, never chosen independently: a mode running under another
        # intervention's version would pool with it
        r2_intervention_version=R2_INTERVENTION_VERSION_BY_MODE[
            args.r2_generation_mode
        ],
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
