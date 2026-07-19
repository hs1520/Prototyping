"""Load archived Option 2 runs with verified measurement sidecars."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_archived_run(value: str | Path) -> dict[str, Any]:
    """Load a run and bind optional sidecars to the exact final model.

    Every CLI that consumes an archived run uses this function so gold
    preparation, blind review, and aggregation cannot silently observe
    different artifacts for the same run directory.
    """
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        run_dir = path
        report_path = run_dir / "realization_run.json"
    else:
        run_dir = path.parent
        report_path = path
    run = json.loads(report_path.read_text(encoding="utf-8"))
    run["_archive_run_dir"] = str(run_dir)

    metadata_path = run_dir / "pilot_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run["pilot_metadata"] = metadata
        for key in (
            "run_id", "repetition", "mcts_seed", "generation_seed",
            "generation_seed_control",
        ):
            if metadata.get(key) is not None:
                run.setdefault(key, metadata[key])

    posthoc_path = run_dir / "posthoc_evaluation.json"
    if not posthoc_path.exists():
        return run

    model_path = run_dir / "final_model.sysml"
    if not model_path.exists():
        raise ValueError(
            f"uniform post-hoc measurement has no final model: {run_dir}"
        )
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
    if posthoc.get("artifact_type") != "OPTION2_UNIFORM_POSTHOC_EVALUATION":
        raise ValueError(f"invalid post-hoc artifact type: {run_dir}")
    if (
        posthoc.get("measurement_only") is not True
        or posthoc.get("mutation_permitted") is not False
    ):
        raise ValueError(
            f"post-hoc sidecar is not a read-only measurement artifact: {run_dir}"
        )
    actual_model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if posthoc.get("model_digest") != actual_model_digest:
        raise ValueError(
            f"post-hoc measurement does not match final_model.sysml: {run_dir}"
        )
    trace_digest = (posthoc.get("semantic_trace_report") or {}).get(
        "model_digest"
    )
    if trace_digest != actual_model_digest:
        raise ValueError(
            f"post-hoc semantic trace does not match final_model.sysml: {run_dir}"
        )
    run["posthoc_evaluation"] = posthoc
    run["posthoc_model_digest_verified"] = True
    return run
