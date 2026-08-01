"""Fail-closed reporting policy for the revised descriptive pilot.

`n` follows the runs actually performed; three is the floor.  The classification
and the inference bans below are deliberately independent of `n`: more
repetitions buy tighter descriptive estimates, never confirmatory licence.
"""
from __future__ import annotations

from typing import Any, Iterable


def build_descriptive_pilot_manifest(
    run_ids: Iterable[str],
    *,
    experiment_namespace: str = "BLACKBOARD_AG_V1",
) -> dict[str, Any]:
    ids = tuple(str(item) for item in run_ids)
    if experiment_namespace != "BLACKBOARD_AG_V1":
        raise ValueError("revised pilot cannot pool a legacy experiment namespace")
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError(
            "minimum pilot requires at least three distinct paired runs"
        )
    return {
        "schema_version": "1.0",
        "experiment_namespace": experiment_namespace,
        "run_ids": list(ids),
        "n": len(ids),
        "study_classification": "DESCRIPTIVE_PILOT",
        "confirmatory_inference": False,
        "confirmatory_p_values_permitted": False,
        "allowed_reporting": [
            "counts", "paired_differences", "effect_estimates", "uncertainty",
        ],
    }
