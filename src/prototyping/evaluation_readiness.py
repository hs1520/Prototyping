"""Fail-closed evidence gate for post-hoc R2-BBAG evaluation.

This module is evaluator-only. It binds a frozen experiment configuration to the
complete selected chain/run set, immutable requirement digests, independently
frozen architecture boundaries, frozen human gold, and per-run blind labels.
It never selects chains from live code and never changes the global experiment
arm metadata.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .ag_gold_template import validate_frozen_gold
from .architecture_boundary import (
    architecture_boundary_digest,
    validate_frozen_boundary,
)
from .experiment_arms import REVISED_EXPERIMENT_NAMESPACE
from .requirement_inputs import (
    normalise_requirement_id,
    resolve_frozen_requirement_set,
)


READINESS_SCHEMA_VERSION = "1.0"
READINESS_ROLE = "POSTHOC_EVALUATION_READINESS_MANIFEST"
FAILURE_TAXONOMY_ROLE = "FROZEN_FAILURE_TAXONOMY"
BLIND_PACKET_ROLE = "BLIND_FAILURE_REVIEW_PACKET"
BLIND_LABEL_ROLE = "BLIND_FAILURE_LABEL"
R2_CONFIGURATION = "R2-BBAG"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_BLIND_KEYS = {
    "verdict",
    "runtime_verdict",
    "diagnostics",
    "failure_diagnostics",
    "repair_decision",
    "repair_decisions",
    "checker_result",
}


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    """Canonical SHA-256 excluding the self-referential digest slot."""
    content = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    raw = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "")))


def _valid_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _forbidden_keys(value: Any, *, path: str = "packet") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            token = str(key).strip().lower()
            child = f"{path}.{key}"
            if token in _FORBIDDEN_BLIND_KEYS:
                found.append(child)
            found.extend(_forbidden_keys(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_keys(item, path=f"{path}[{index}]"))
    return found


def validate_frozen_failure_taxonomy(taxonomy: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if taxonomy.get("schema_version") != READINESS_SCHEMA_VERSION:
        problems.append(f"taxonomy schema_version must be {READINESS_SCHEMA_VERSION!r}")
    if taxonomy.get("artifact_role") != FAILURE_TAXONOMY_ROLE:
        problems.append(f"taxonomy artifact_role must be {FAILURE_TAXONOMY_ROLE!r}")
    if taxonomy.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("taxonomy must use BLACKBOARD_AG_V1")
    if taxonomy.get("status") != "FROZEN":
        problems.append("taxonomy status must be 'FROZEN'")
    if not str(taxonomy.get("taxonomy_version") or "").strip():
        problems.append("taxonomy_version must be set")
    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or not classes:
        problems.append("taxonomy classes must be a non-empty list")
    elif len(classes) != len(set(map(str, classes))):
        problems.append("taxonomy classes must be unique")
    if not taxonomy.get("reviewer"):
        problems.append("taxonomy reviewer must be set")
    if not _valid_iso_date(taxonomy.get("reviewed_date")):
        problems.append("taxonomy reviewed_date must be ISO YYYY-MM-DD")
    digest = taxonomy.get("artifact_digest")
    if not _is_digest(digest) or digest != artifact_digest(taxonomy):
        problems.append("taxonomy artifact_digest is missing, malformed, or stale")
    return problems


def validate_blind_packet(packet: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if packet.get("schema_version") != READINESS_SCHEMA_VERSION:
        problems.append(f"blind packet schema_version must be {READINESS_SCHEMA_VERSION!r}")
    if packet.get("artifact_role") != BLIND_PACKET_ROLE:
        problems.append(f"blind packet artifact_role must be {BLIND_PACKET_ROLE!r}")
    if packet.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("blind packet must use BLACKBOARD_AG_V1")
    if packet.get("configuration") != R2_CONFIGURATION:
        problems.append("blind packet configuration must be R2-BBAG")
    for field in ("run_id", "chain_id"):
        if not str(packet.get(field) or "").strip():
            problems.append(f"blind packet {field} must be set")
    for field in ("model_digest", "requirement_digest"):
        if not _is_digest(packet.get(field)):
            problems.append(f"blind packet {field} must be a lowercase SHA-256 digest")
    review_material = packet.get("review_material")
    if not isinstance(review_material, Mapping) or not review_material:
        problems.append("blind packet review_material must be a non-empty object")
    forbidden = _forbidden_keys(packet)
    if forbidden:
        problems.append(
            "blind packet exposes forbidden runtime/repair fields: "
            + ", ".join(sorted(forbidden))
        )
    digest = packet.get("artifact_digest")
    if not _is_digest(digest) or digest != artifact_digest(packet):
        problems.append("blind packet artifact_digest is missing, malformed, or stale")
    return problems


def validate_blind_label(
    label: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    if label.get("schema_version") != READINESS_SCHEMA_VERSION:
        problems.append(f"blind label schema_version must be {READINESS_SCHEMA_VERSION!r}")
    if label.get("artifact_role") != BLIND_LABEL_ROLE:
        problems.append(f"blind label artifact_role must be {BLIND_LABEL_ROLE!r}")
    if label.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("blind label must use BLACKBOARD_AG_V1")
    if label.get("configuration") != R2_CONFIGURATION:
        problems.append("blind label configuration must be R2-BBAG")
    if label.get("status") != "FROZEN":
        problems.append("blind label status must be 'FROZEN'")
    for field in ("run_id", "chain_id", "model_digest", "requirement_digest"):
        if label.get(field) != packet.get(field):
            problems.append(f"blind label {field} does not match its blind packet")
    if label.get("blind_packet_digest") != packet.get("artifact_digest"):
        problems.append("blind label does not cite its blind packet digest")
    if label.get("failure_taxonomy_version") != taxonomy.get("taxonomy_version"):
        problems.append("blind label failure_taxonomy_version does not match taxonomy")
    if label.get("failure_taxonomy_digest") != taxonomy.get("artifact_digest"):
        problems.append("blind label failure_taxonomy_digest does not match taxonomy")
    if label.get("failure_class") not in (taxonomy.get("classes") or []):
        problems.append("blind label failure_class is not in the frozen taxonomy")
    if not label.get("reviewer"):
        problems.append("blind label reviewer must be set")
    if not _valid_iso_date(label.get("reviewed_date")):
        problems.append("blind label reviewed_date must be ISO YYYY-MM-DD")
    protocol = label.get("review_protocol") or {}
    if protocol.get("independent_human_review") is not True:
        problems.append("blind label independent_human_review must be true")
    if protocol.get("blind_to_runtime_verdict") is not True:
        problems.append("blind label blind_to_runtime_verdict must be true")
    digest = label.get("artifact_digest")
    if not _is_digest(digest) or digest != artifact_digest(label):
        problems.append("blind label artifact_digest is missing, malformed, or stale")
    return problems


def _configuration_problems(config: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if config.get("artifact_role") != "FROZEN_REVISED_PILOT_CONFIGURATION":
        problems.append("experiment config must be FROZEN_REVISED_PILOT_CONFIGURATION")
    if config.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        problems.append("experiment config must use BLACKBOARD_AG_V1")
    if R2_CONFIGURATION not in (config.get("arms") or []):
        problems.append("experiment config does not contain R2-BBAG")
    digest = config.get("configuration_digest")
    if not _is_digest(digest):
        problems.append("configuration_digest must be a lowercase SHA-256 digest")
    else:
        content = {key: value for key, value in config.items() if key != "configuration_digest"}
        if digest != artifact_digest(content):
            problems.append("configuration_digest does not match the frozen config")
    try:
        resolve_frozen_requirement_set(config.get("frozen_requirement_set") or {})
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid frozen requirement set: {exc}")
    chains = [
        normalise_requirement_id(str(value))
        for value in (config.get("selected_ag_chain_ids") or [])
    ]
    if not chains or len(chains) != len(set(chains)):
        problems.append("selected_ag_chain_ids must be non-empty and unique")
    runs = [str(value) for value in (config.get("selected_r2_run_ids") or [])]
    if not runs or len(runs) != len(set(runs)):
        problems.append("selected_r2_run_ids must be non-empty and unique")
    return problems


def build_evaluation_readiness_manifest(
    *,
    frozen_experiment_config: Mapping[str, Any],
    architecture_boundaries: Mapping[str, Mapping[str, Any]],
    gold_by_chain: Mapping[str, Mapping[str, Any]],
    archived_runs: Mapping[str, Mapping[str, Any]],
    blind_packets: Sequence[Mapping[str, Any]],
    blind_labels: Sequence[Mapping[str, Any]],
    failure_taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the sole fail-closed pooling decision for one frozen R2 evaluation."""
    problems = _configuration_problems(frozen_experiment_config)
    config_digest = frozen_experiment_config.get("configuration_digest")
    frozen_requirements = frozen_experiment_config.get("frozen_requirement_set") or {}
    requirement_set_digest = frozen_requirements.get("requirement_set_digest")
    source_digests = frozen_requirements.get("source_digests") or {}
    selected_chains = [
        normalise_requirement_id(str(value))
        for value in (frozen_experiment_config.get("selected_ag_chain_ids") or [])
    ]
    selected_runs = [
        str(value)
        for value in (frozen_experiment_config.get("selected_r2_run_ids") or [])
    ]

    problems.extend(validate_frozen_failure_taxonomy(failure_taxonomy))
    taxonomy_digest = failure_taxonomy.get("artifact_digest")

    boundary_evidence: dict[str, str | None] = {}
    gold_evidence: dict[str, str | None] = {}
    for chain_id in selected_chains:
        expected_source_digest = source_digests.get(chain_id)
        if not _is_digest(expected_source_digest):
            problems.append(
                f"{chain_id}: no matching requirement source digest in frozen config"
            )

        boundary = architecture_boundaries.get(chain_id)
        if boundary is None:
            problems.append(f"{chain_id}: frozen architecture boundary is missing")
            boundary_evidence[chain_id] = None
        else:
            problems.extend(
                f"{chain_id}: {issue}" for issue in validate_frozen_boundary(boundary)
            )
            if boundary.get("chain_id") != chain_id:
                problems.append(f"{chain_id}: architecture boundary chain_id mismatch")
            if boundary.get("requirement_set_digest") != requirement_set_digest:
                problems.append(
                    f"{chain_id}: architecture boundary requirement-set digest mismatch"
                )
            boundary_evidence[chain_id] = boundary.get("artifact_digest")

        gold = gold_by_chain.get(chain_id)
        if gold is None:
            problems.append(f"{chain_id}: frozen evaluator gold is missing")
            gold_evidence[chain_id] = None
            continue
        problems.extend(f"{chain_id}: {issue}" for issue in validate_frozen_gold(dict(gold)))
        if gold.get("chain_id") != chain_id:
            problems.append(f"{chain_id}: gold chain_id mismatch")
        if gold.get("source_digest") != expected_source_digest:
            problems.append(f"{chain_id}: gold source digest mismatch")
        if gold.get("requirement_set_digest") != requirement_set_digest:
            problems.append(f"{chain_id}: gold requirement-set digest mismatch")
        if gold.get("architecture_boundary_digest") != boundary_evidence.get(chain_id):
            problems.append(f"{chain_id}: gold architecture-boundary digest mismatch")
        if boundary is not None:
            gold_allocations = {
                (
                    str(item.get("owner") or ""),
                    str(item.get("contract") or ""),
                    str(item.get("guarantee") or ""),
                )
                for item in (gold.get("allocations") or [])
            }
            boundary_allocations = {
                (
                    str(item.get("owner") or ""),
                    str(item.get("contract") or ""),
                    str(item.get("guarantee") or ""),
                )
                for item in (boundary.get("allocations") or [])
            }
            if gold_allocations != boundary_allocations:
                problems.append(
                    f"{chain_id}: gold allocations differ from architecture boundary"
                )
        gold_evidence[chain_id] = artifact_digest(gold)

    packet_index = {
        (str(item.get("run_id")), normalise_requirement_id(str(item.get("chain_id")))): item
        for item in blind_packets
    }
    label_index = {
        (str(item.get("run_id")), normalise_requirement_id(str(item.get("chain_id")))): item
        for item in blind_labels
    }
    if len(packet_index) != len(blind_packets):
        problems.append("blind packets contain duplicate run_id/chain_id bindings")
    if len(label_index) != len(blind_labels):
        problems.append("blind labels contain duplicate run_id/chain_id bindings")

    blind_evidence: list[dict[str, Any]] = []
    expected_bindings = {
        (run_id, chain_id) for run_id in selected_runs for chain_id in selected_chains
    }
    if set(archived_runs) - set(selected_runs):
        problems.append("archived runs include run_ids outside the frozen selection")
    if set(packet_index) - expected_bindings:
        problems.append("blind packets include runs/chains outside the frozen selection")
    if set(label_index) - expected_bindings:
        problems.append("blind labels include runs/chains outside the frozen selection")
    for binding in sorted(expected_bindings):
        run_id, chain_id = binding
        archived = archived_runs.get(run_id)
        if archived is None:
            problems.append(f"{run_id}: archived run evidence is missing")
            continue
        run_manifest = archived.get("run_manifest") or {}
        if run_manifest.get("artifact_role") != "REVISED_PILOT_RUN_MANIFEST":
            problems.append(f"{run_id}: archived run manifest role mismatch")
        if run_manifest.get("run_id") != run_id:
            problems.append(f"{run_id}: archived run manifest run_id mismatch")
        if run_manifest.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
            problems.append(f"{run_id}: archived run namespace mismatch")
        if run_manifest.get("configuration") != R2_CONFIGURATION:
            problems.append(f"{run_id}: archived run configuration mismatch")
        if run_manifest.get("status") != "COMPLETED":
            problems.append(f"{run_id}: archived run is not COMPLETED")
        if run_manifest.get("pilot_configuration_digest") != config_digest:
            problems.append(f"{run_id}: archived run configuration digest mismatch")
        if run_manifest.get("requirement_set_digest") != requirement_set_digest:
            problems.append(f"{run_id}: archived run requirement-set digest mismatch")
        prediction = (archived.get("predictions") or {}).get(chain_id)
        if not isinstance(prediction, Mapping):
            problems.append(f"{run_id}/{chain_id}: archived prediction is missing")
            continue
        if prediction.get("artifact_role") != "RUNTIME_A_G_PREDICTION":
            problems.append(f"{run_id}/{chain_id}: archived prediction role mismatch")
        if prediction.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
            problems.append(f"{run_id}/{chain_id}: prediction namespace mismatch")
        if prediction.get("configuration") != R2_CONFIGURATION:
            problems.append(f"{run_id}/{chain_id}: prediction configuration mismatch")
        if normalise_requirement_id(
            str(prediction.get("source_requirement") or "")
        ) != chain_id:
            problems.append(f"{run_id}/{chain_id}: prediction source requirement mismatch")
        prediction_model_digest = prediction.get("source_model_digest")
        if not _is_digest(prediction_model_digest):
            problems.append(f"{run_id}/{chain_id}: prediction model digest is invalid")
        packet = packet_index.get(binding)
        label = label_index.get(binding)
        if packet is None:
            problems.append(f"{run_id}/{chain_id}: blind packet is missing")
            continue
        problems.extend(
            f"{run_id}/{chain_id}: {issue}" for issue in validate_blind_packet(packet)
        )
        if packet.get("requirement_digest") != source_digests.get(chain_id):
            problems.append(f"{run_id}/{chain_id}: packet requirement digest mismatch")
        if packet.get("model_digest") != prediction_model_digest:
            problems.append(f"{run_id}/{chain_id}: packet model digest mismatch")
        if label is None:
            problems.append(f"{run_id}/{chain_id}: frozen blind label is missing")
            continue
        problems.extend(
            f"{run_id}/{chain_id}: {issue}"
            for issue in validate_blind_label(
                label, packet=packet, taxonomy=failure_taxonomy
            )
        )
        blind_evidence.append({
            "run_id": run_id,
            "chain_id": chain_id,
            "packet_digest": packet.get("artifact_digest"),
            "label_digest": label.get("artifact_digest"),
            "run_manifest_digest": artifact_digest(run_manifest),
            "prediction_digest": artifact_digest(prediction),
        })

    unique_problems = list(dict.fromkeys(problems))
    ready = not unique_problems
    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "artifact_role": READINESS_ROLE,
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": R2_CONFIGURATION,
        "configuration_digest": config_digest,
        "requirement_set_digest": requirement_set_digest,
        "selected_chain_ids": selected_chains,
        "selected_run_ids": selected_runs,
        "architecture_boundary_digests": boundary_evidence,
        "gold_artifact_digests": gold_evidence,
        "failure_taxonomy_digest": taxonomy_digest,
        "blind_label_evidence": blind_evidence,
        "evaluation_ready": ready,
        "pooling_permitted": ready,
        "study_classification": "DESCRIPTIVE_PILOT",
        "confirmatory_inference_permitted": False,
        "problems": unique_problems,
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return payload


def require_evaluation_ready(manifest: Mapping[str, Any]) -> None:
    """Reject any attempt to pool without a self-consistent ready manifest."""
    if manifest.get("artifact_role") != READINESS_ROLE:
        raise ValueError(f"readiness artifact_role must be {READINESS_ROLE!r}")
    if manifest.get("experiment_namespace") != REVISED_EXPERIMENT_NAMESPACE:
        raise ValueError("readiness manifest must use BLACKBOARD_AG_V1")
    if manifest.get("configuration") != R2_CONFIGURATION:
        raise ValueError("readiness manifest must be for R2-BBAG")
    if manifest.get("artifact_digest") != artifact_digest(manifest):
        raise ValueError("readiness manifest artifact_digest is stale or malformed")
    if (
        manifest.get("evaluation_ready") is not True
        or manifest.get("pooling_permitted") is not True
        or manifest.get("problems")
    ):
        raise ValueError("R2 evaluation evidence gate is not ready")
