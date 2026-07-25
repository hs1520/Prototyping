"""Reproducible, fail-closed runner for the revised Option 2 pilot.

This module owns experiment execution provenance only. It has no evaluator-gold
input and never upgrades runtime checker verdicts into accuracy or physical
verification claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .ag_contracts import AG_CHECKER_VERSION
from .evaluation_protocol import build_descriptive_pilot_manifest
from .experiment_arms import (
    REVISED_EXPERIMENT_NAMESPACE,
    R2_DETERMINISTIC_GENERATION_MODE,
    R2_DETERMINISTIC_INTERVENTION_VERSION,
    R2_INTERVENTION_VERSION_BY_MODE,
)
from .requirement_inputs import (
    build_frozen_requirement_set,
    normalise_requirement_id,
)


REVISED_PILOT_ARMS = ("R0-CURRENT", "R1-BBCTX", "R2-BBAG")
_EVALUATOR_ONLY_ROLE_TOKENS = {
    "evaluatorgold",
    "blindfailurereviewpacket",
    "blindfailurelabel",
    "failuretaxonomy",
    "frozenfailuretaxonomy",
    "posthochumangoldevaluation",
    "posthocevaluationreadinessmanifest",
}
_EVALUATOR_ONLY_KEY_TOKENS = {
    "gold",
    "humangold",
    "evaluatorgold",
    "blindlabel",
    "blindlabels",
}


def _json_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RevisedPilotConfig:
    provider: str
    model: str
    seeds: tuple[int, int, int]
    max_iterations: int
    code_revision: str
    requirements: tuple[str, ...]
    quality_threshold: float = 0.75
    llm_timeout_seconds: float = 300.0
    task_session_max_turns: int = 12
    task_session_max_tokens: int = 150000
    context_token_budget: int = 12000
    ag_checker_version: str = AG_CHECKER_VERSION
    pattern_profile_version: str = "bounded-ag-safety-profile-2.0"
    r2_generation_mode: str = R2_DETERMINISTIC_GENERATION_MODE
    r2_intervention_version: str = R2_DETERMINISTIC_INTERVENTION_VERSION
    system_name: str = "DeliveryUAV"
    system_description: str = (
        "An autonomous delivery UAV with ballistic parachute recovery and "
        "forward obstacle avoidance."
    )
    selected_ag_chain_ids: tuple[str, ...] = ("REQ_SAFE_005",)
    experiment_namespace: str = REVISED_EXPERIMENT_NAMESPACE
    arms: tuple[str, str, str] = REVISED_PILOT_ARMS

    def __post_init__(self) -> None:
        if self.experiment_namespace != REVISED_EXPERIMENT_NAMESPACE:
            raise ValueError("revised pilot requires BLACKBOARD_AG_V1")
        if tuple(self.arms) != REVISED_PILOT_ARMS:
            raise ValueError("revised pilot requires exact R0/R1/R2 arm ordering")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("descriptive pilot requires exactly three distinct seeds")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be in (0, 1]")
        if (
            self.llm_timeout_seconds <= 0
            or self.task_session_max_turns <= 0
            or self.task_session_max_tokens <= 0
            or self.context_token_budget <= 0
        ):
            raise ValueError("all frozen execution budgets must be positive")
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider and model must be explicit")
        if not self.code_revision.strip():
            raise ValueError("code_revision must be frozen before execution")
        # Each R2 generation mode is its own frozen intervention with its own
        # version, and results from different modes must never be pooled. That is
        # enforced by binding the mode to exactly one version — the same binding
        # `evaluation_readiness` gates on — rather than by pinning the runner to a
        # single mode, which prevented the other interventions from ever executing.
        expected_version = R2_INTERVENTION_VERSION_BY_MODE.get(
            self.r2_generation_mode
        )
        if expected_version is None:
            raise ValueError(
                f"unknown r2_generation_mode {self.r2_generation_mode!r}; "
                f"expected one of {sorted(R2_INTERVENTION_VERSION_BY_MODE)}"
            )
        if self.r2_intervention_version != expected_version:
            raise ValueError(
                f"r2_generation_mode {self.r2_generation_mode!r} is bound to "
                f"intervention version {expected_version!r}, not "
                f"{self.r2_intervention_version!r}; a mode may not run under "
                "another intervention's version or the results would pool"
            )
        frozen = build_frozen_requirement_set(self.requirements)
        selected = tuple(
            normalise_requirement_id(value) for value in self.selected_ag_chain_ids
        )
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("selected_ag_chain_ids must be non-empty and unique")
        available = set(frozen["source_digests"])
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(
                "selected A/G chains are absent from the frozen requirement set: "
                + ", ".join(missing)
            )

    def frozen_requirements(self) -> dict[str, Any]:
        return build_frozen_requirement_set(
            self.requirements,
            name="option2-minimum-credible-pilot",
            source="design-authority-selected",
        )

    def to_manifest(self) -> dict[str, Any]:
        frozen = self.frozen_requirements()
        payload = {
            "schema_version": "1.0",
            "artifact_role": "FROZEN_REVISED_PILOT_CONFIGURATION",
            "experiment_namespace": self.experiment_namespace,
            "arms": list(self.arms),
            "seeds": list(self.seeds),
            "provider": self.provider,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "quality_threshold": self.quality_threshold,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "task_session_max_turns": self.task_session_max_turns,
            "task_session_max_tokens": self.task_session_max_tokens,
            "context_token_budget": self.context_token_budget,
            "ag_checker_version": self.ag_checker_version,
            "pattern_profile_version": self.pattern_profile_version,
            "r2_generation_mode": self.r2_generation_mode,
            "r2_intervention_version": self.r2_intervention_version,
            "code_revision": self.code_revision,
            "system_name": self.system_name,
            "system_description": self.system_description,
            "frozen_requirement_set": frozen,
            "selected_ag_chain_ids": [
                normalise_requirement_id(value)
                for value in self.selected_ag_chain_ids
            ],
            "selected_r2_run_ids": [
                f"seed-{seed}:R2-BBAG" for seed in self.seeds
            ],
            "gold_input_permitted": False,
            "langsmith_permitted": False,
            "gazebo_permitted": False,
            "sitl_permitted": False,
        }
        return {**payload, "configuration_digest": _json_digest(payload)}


def _usage(report: Mapping[str, Any]) -> tuple[int | None, int | None]:
    usage = report.get("llm_usage") or {}
    calls = usage.get("calls")
    tokens = usage.get("total_tokens")
    if tokens is None:
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return (
        int(calls) if calls is not None else None,
        int(tokens) if tokens is not None else None,
    )


def _contains_evaluator_only_material(value: Any) -> bool:
    """Reject evaluator artifacts recursively, independent of key spelling."""
    if isinstance(value, Mapping):
        role = re.sub(
            r"[^a-z0-9]", "", str(value.get("artifact_role") or "").lower()
        )
        if role in _EVALUATOR_ONLY_ROLE_TOKENS:
            return True
        for key, item in value.items():
            token = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if token in _EVALUATOR_ONLY_KEY_TOKENS:
                return True
            if _contains_evaluator_only_material(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_evaluator_only_material(item) for item in value)
    return False


def _validate_run_result(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    arm: str,
    requirement_set_digest: str,
    ag_checker_version: str,
    r2_generation_mode: str,
    r2_intervention_version: str,
) -> None:
    revised = result.get("revised_experiment") or {}
    if (
        revised.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE
        or revised.get("configuration") != arm
    ):
        raise ValueError("run result namespace/configuration does not match frozen arm")
    if arm == "R2-BBAG" and revised.get("evaluation_ready") is not False:
        raise ValueError("R2-BBAG must remain evaluation_ready=false before gold freeze")
    if arm == "R2-BBAG" and (
        revised.get("r2_generation_mode") != r2_generation_mode
        or revised.get("r2_intervention_version") != r2_intervention_version
    ):
        raise ValueError(
            "R2-BBAG run intervention identity does not match frozen configuration"
        )
    if arm == "R2-BBAG" and (
        (result.get("ag_contract_graph") or {}).get("checker_version")
        != ag_checker_version
    ):
        raise ValueError("R2-BBAG checker version does not match frozen configuration")
    requirement_input = result.get("requirement_input") or {}
    if requirement_input.get("requirement_set_digest") != requirement_set_digest:
        raise ValueError("run result requirement digest does not match frozen configuration")
    if report.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        raise ValueError("run report is missing the revised experiment namespace")
    if report.get("configuration") != arm:
        raise ValueError("run report configuration does not match frozen arm")
    if (
        _contains_evaluator_only_material(result)
        or _contains_evaluator_only_material(report)
    ):
        raise ValueError("evaluator gold/blind labels cannot enter pilot generation")


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows or not all(row.get("status") == "COMPLETED" for row in rows):
        return {
            "status": "INCOMPLETE",
            "reason": "all nine arm/seed runs must complete before aggregation",
        }
    by_arm: dict[str, list[Mapping[str, Any]]] = {
        arm: [row for row in rows if row.get("configuration") == arm]
        for arm in REVISED_PILOT_ARMS
    }
    if any(len(items) != 3 for items in by_arm.values()):
        return {"status": "INCOMPLETE", "reason": "paired arm coverage is incomplete"}

    arms: dict[str, Any] = {}
    for arm, items in by_arm.items():
        scores = [float(item["final_score"]) for item in items]
        arms[arm] = {
            "n": len(scores),
            "score_mean": sum(scores) / len(scores),
            "score_range": [min(scores), max(scores)],
            "llm_calls_total": sum(int(item.get("llm_calls") or 0) for item in items),
            "llm_tokens_total": sum(
                int(item.get("llm_total_tokens") or 0) for item in items
            ),
        }
    indexed = {
        (int(row["seed"]), str(row["configuration"])): row for row in rows
    }
    paired_differences = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        r0 = float(indexed[(seed, "R0-CURRENT")]["final_score"])
        r1 = float(indexed[(seed, "R1-BBCTX")]["final_score"])
        r2 = float(indexed[(seed, "R2-BBAG")]["final_score"])
        paired_differences.append({
            "seed": seed,
            "R1_minus_R0": r1 - r0,
            "R2_minus_R1": r2 - r1,
            "R2_minus_R0": r2 - r0,
        })
    return {
        "status": "DESCRIPTIVE_ONLY",
        "confirmatory_inference": False,
        "confirmatory_p_values_permitted": False,
        "arms": arms,
        "paired_differences": paired_differences,
    }


def run_revised_pilot(
    config: RevisedPilotConfig,
    out_dir: str | Path,
    *,
    external_execution_authorized: bool,
    llm_factory: Callable[..., Any],
    pipeline_factory: Callable[..., Any],
    artifact_writer: Callable[[Mapping[str, Any], Path], Mapping[str, str]],
) -> dict[str, Any]:
    """Run exactly three paired R0/R1/R2 repetitions with full provenance.

    A fresh provider instance is created for every arm/seed run so token ledgers,
    call state, and seed configuration cannot leak across arms. Existing output
    directories are rejected rather than overwritten.
    """
    if config.provider.strip().lower() != "mock" and not external_execution_authorized:
        raise PermissionError(
            "external pilot execution requires explicit current-run authorization"
        )
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"pilot output directory already exists: {out}")
    out.mkdir(parents=True)

    config_manifest = config.to_manifest()
    config_digest = str(config_manifest["configuration_digest"])
    frozen = config.frozen_requirements()
    requirement_digest = str(frozen["requirement_set_digest"])
    _write_json(out / "pilot_config.json", config_manifest)

    protocol = build_descriptive_pilot_manifest(
        [f"paired-seed-{seed}" for seed in config.seeds],
        experiment_namespace=config.experiment_namespace,
    )
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        for arm in config.arms:
            run_id = f"seed-{seed}:{arm}"
            run_dir = out / f"seed-{seed}" / arm
            run_dir.mkdir(parents=True)
            started_at = _utc_now()
            started = time.monotonic()
            base = {
                "schema_version": "1.0",
                "artifact_role": "REVISED_PILOT_RUN_MANIFEST",
                "run_id": run_id,
                "paired_run_id": f"paired-seed-{seed}",
                "experiment_namespace": config.experiment_namespace,
                "configuration": arm,
                "seed": seed,
                "provider": config.provider,
                "model": config.model,
                "max_iterations": config.max_iterations,
                "code_revision": config.code_revision,
                "pilot_configuration_digest": config_digest,
                "requirement_set_digest": requirement_digest,
                "ag_checker_version": config.ag_checker_version,
                "pattern_profile_version": config.pattern_profile_version,
                "r2_generation_mode": config.r2_generation_mode,
                "r2_intervention_version": config.r2_intervention_version,
                "started_at": started_at,
                "gold_access": False,
                "formal_ag_proof": False,
                "physical_verification": False,
            }
            try:
                llm = llm_factory(
                    provider=config.provider,
                    model=config.model,
                    provider_kwargs={
                        "seed": seed,
                        "enable_langsmith": False,
                        "timeout_seconds": config.llm_timeout_seconds,
                    },
                )
                pipeline = pipeline_factory(
                    llm=llm,
                    max_iterations=config.max_iterations,
                    quality_threshold=config.quality_threshold,
                    verbose=False,
                    revised_experiment_arm=arm,
                    task_session_max_turns=config.task_session_max_turns,
                    task_session_max_tokens=config.task_session_max_tokens,
                )
                result = pipeline.orchestrator.generate(
                    system_name=config.system_name,
                    system_description=config.system_description,
                    frozen_requirements=frozen,
                )
                report = pipeline.build_run_report(result)
                _validate_run_result(
                    result,
                    report,
                    arm=arm,
                    requirement_set_digest=requirement_digest,
                    ag_checker_version=config.ag_checker_version,
                    r2_generation_mode=config.r2_generation_mode,
                    r2_intervention_version=config.r2_intervention_version,
                )
                _write_json(run_dir / "run_report.json", report)
                written: Mapping[str, str] = {}
                if result.get("collaboration"):
                    written = artifact_writer(result, run_dir)
                calls, tokens = _usage(report)
                row = {
                    **base,
                    "status": "COMPLETED",
                    "completed_at": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "final_score": report.get("final_score"),
                    "llm_calls": calls,
                    "llm_total_tokens": tokens,
                    "evaluation_ready": bool(
                        (result.get("revised_experiment") or {}).get(
                            "evaluation_ready", False
                        )
                    ),
                    "runtime_ag_verdict": (
                        (result.get("ag_contract_graph") or {}).get("verdict")
                    ),
                    "runtime_pattern_verdict": (
                        (result.get("pattern_conformance_report") or {}).get(
                            "verdict"
                        )
                    ),
                    "artifact_files": sorted(str(key) for key in written),
                }
            except Exception as exc:
                row = {
                    **base,
                    "status": "FAILED",
                    "completed_at": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            _write_json(run_dir / "run_manifest.json", row)
            rows.append(row)

    complete = all(row["status"] == "COMPLETED" for row in rows)
    manifest = {
        "schema_version": "1.0",
        "artifact_role": "REVISED_DESCRIPTIVE_PILOT_MANIFEST",
        "experiment_namespace": config.experiment_namespace,
        "configuration_digest": config_digest,
        "requirement_set_digest": requirement_digest,
        "protocol": protocol,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "run_count": len(rows),
        "completed_run_count": sum(row["status"] == "COMPLETED" for row in rows),
        "pooling_permitted": False,
        "pooling_note": (
            "descriptive pilot only; R2 accuracy/F1 additionally requires frozen "
            "independent human gold and blind review"
        ),
        "runs": rows,
        "descriptive_summary": _aggregate(rows),
    }
    _write_json(out / "pilot_manifest.json", manifest)
    return manifest
