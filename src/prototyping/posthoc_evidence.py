"""Operator tooling for the human-gated Option 2 post-hoc evidence chain.

This module prepares and validates evidence; it does not make review decisions.
In particular, it never sets an artifact to ``FROZEN``, never asserts an
independent/blind-review flag, and never chooses a per-run failure class.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from .ag_gold_template import validate_frozen_gold
from .artifact_store import atomic_write_json, read_json_object
from .architecture_boundary import (
    architecture_boundary_digest,
    validate_frozen_boundary,
)
from .evaluation_readiness import (
    artifact_digest,
    build_blind_review_packet,
    build_evaluation_readiness_manifest,
    require_evaluation_ready,
    validate_blind_label,
    validate_blind_packet,
    validate_frozen_failure_taxonomy,
)
from .requirement_inputs import normalise_requirement_id


_NAMESPACE = "BLACKBOARD_AG_V1"
_R2 = "R2-BBAG"
_RESPONSIBILITY_CANDIDATES = {
    "SafetyResponseArbiterContract": (
        "When criticalPropulsionFailureDetected is asserted while airborne, "
        "apply the frozen approved response-priority policy, select "
        "PARACHUTE_DEPLOYMENT, expose that selection, and issue the parachute "
        "deployment command within 0.10 s of the triggering failure-detection "
        "event."
    ),
    "RecoveryPowerSupplyContract": (
        "While airborne, maintain recovery-actuation power independently of "
        "nominal propulsion power and continuously expose whether that power is "
        "available to the recovery system."
    ),
    "RecoverySystemContract": (
        "When a parachute deployment command is received with recovery-actuation "
        "power available, execute parachute deployment and expose the "
        "parachute-deployed observation within 0.35 s of command receipt."
    ),
}


_read_json = read_json_object
_write_json = atomic_write_json


def _source_by_chain(config: Mapping[str, Any]) -> dict[str, str]:
    frozen = config.get("frozen_requirement_set") or {}
    result: dict[str, str] = {}
    for source in frozen.get("requirements") or []:
        chain = normalise_requirement_id(str(source))
        if chain in result:
            raise ValueError(f"duplicate frozen requirement id: {chain}")
        result[chain] = str(source)
    return result


def _run_dir(pilot_dir: Path, run_id: str) -> Path:
    match = re.fullmatch(r"seed-([^:]+):(R2-BBAG)", run_id)
    if not match:
        raise ValueError(f"unsupported selected R2 run_id: {run_id!r}")
    return pilot_dir / f"seed-{match.group(1)}" / match.group(2)


def _prediction_path(run_dir: Path, chain_id: str) -> Path:
    aggregate = run_dir / "ag_contract_graph.json"
    prediction = _read_json(aggregate)
    if normalise_requirement_id(
        str(prediction.get("source_requirement") or "")
    ) == chain_id:
        return aggregate
    per_chain = run_dir / f"ag_contract_graph.{chain_id}.json"
    if per_chain.is_file():
        return per_chain
    raise ValueError(f"{run_dir}: prediction for {chain_id} is missing")


def load_completed_pilot(
    pilot_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and cross-check the frozen configuration and selected R2 archives."""
    root = Path(pilot_dir)
    config = _read_json(root / "pilot_config.json")
    manifest = _read_json(root / "pilot_manifest.json")
    problems: list[str] = []
    if config.get("experiment_namespace") != _NAMESPACE:
        problems.append("pilot config must use BLACKBOARD_AG_V1")
    if manifest.get("experiment_namespace") != _NAMESPACE:
        problems.append("pilot manifest must use BLACKBOARD_AG_V1")
    if manifest.get("status") != "COMPLETE":
        problems.append("pilot manifest is not COMPLETE")
    if manifest.get("completed_run_count") != manifest.get("run_count"):
        problems.append("pilot manifest has incomplete runs")
    if manifest.get("configuration_digest") != config.get("configuration_digest"):
        problems.append("pilot/config configuration_digest mismatch")

    selected_chains = [
        normalise_requirement_id(str(item))
        for item in config.get("selected_ag_chain_ids") or []
    ]
    selected_runs = [str(item) for item in config.get("selected_r2_run_ids") or []]
    if not selected_chains or not selected_runs:
        problems.append("pilot config has no selected chains/runs")
    sources = _source_by_chain(config)
    source_digests = (
        config.get("frozen_requirement_set", {}).get("source_digests") or {}
    )
    archived_runs: dict[str, dict[str, Any]] = {}
    for run_id in selected_runs:
        directory = _run_dir(root, run_id)
        run_manifest = _read_json(directory / "run_manifest.json")
        if run_manifest.get("run_id") != run_id:
            problems.append(f"{run_id}: run manifest id mismatch")
        if run_manifest.get("status") != "COMPLETED":
            problems.append(f"{run_id}: run is not COMPLETED")
        if run_manifest.get("configuration") != _R2:
            problems.append(f"{run_id}: run is not R2-BBAG")
        if run_manifest.get("experiment_namespace") != _NAMESPACE:
            problems.append(f"{run_id}: namespace mismatch")
        if (
            run_manifest.get("pilot_configuration_digest")
            != config.get("configuration_digest")
        ):
            problems.append(f"{run_id}: configuration digest mismatch")
        if (
            run_manifest.get("requirement_set_digest")
            != config.get("frozen_requirement_set", {}).get(
                "requirement_set_digest"
            )
        ):
            problems.append(f"{run_id}: requirement-set digest mismatch")
        if run_manifest.get("evaluation_ready") is not False:
            problems.append(f"{run_id}: R2 evaluation_ready must remain false")

        model_path = directory / "shared_model_final.sysml"
        model = model_path.read_text(encoding="utf-8")
        model_digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
        predictions: dict[str, Any] = {}
        for chain_id in selected_chains:
            if chain_id not in sources or chain_id not in source_digests:
                problems.append(f"{chain_id}: frozen source binding is missing")
                continue
            prediction = _read_json(_prediction_path(directory, chain_id))
            if prediction.get("artifact_role") != "RUNTIME_A_G_PREDICTION":
                problems.append(f"{run_id}/{chain_id}: prediction role mismatch")
            if prediction.get("experiment_namespace") != _NAMESPACE:
                problems.append(f"{run_id}/{chain_id}: prediction namespace mismatch")
            if prediction.get("configuration") != _R2:
                problems.append(f"{run_id}/{chain_id}: prediction arm mismatch")
            if prediction.get("source_model_digest") != model_digest:
                problems.append(f"{run_id}/{chain_id}: model digest mismatch")
            if (
                prediction.get("checker_version")
                != config.get("ag_checker_version")
                or run_manifest.get("ag_checker_version")
                != config.get("ag_checker_version")
            ):
                problems.append(f"{run_id}/{chain_id}: checker version mismatch")
            predictions[chain_id] = prediction
        archived_runs[run_id] = {
            "run_manifest": run_manifest,
            "predictions": predictions,
            "_candidate_models": {
                chain_id: model for chain_id in selected_chains
            },
        }
    if problems:
        raise ValueError("invalid pilot archive: " + "; ".join(problems))
    return config, manifest, archived_runs


def prepare_review_materials(
    *,
    pilot_dir: str | Path,
    output_dir: str | Path,
    gold_dir: str | Path,
) -> dict[str, Any]:
    """Create review-only DRAFT copies bound to one completed pilot config."""
    config, manifest, _archived = load_completed_pilot(pilot_dir)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"review output already exists: {out}")
    review = out / "review_inputs"
    review.mkdir(parents=True)
    gold_root = Path(gold_dir)
    requirement_set_digest = str(
        config["frozen_requirement_set"]["requirement_set_digest"]
    )
    source_digests = config["frozen_requirement_set"]["source_digests"]
    selected_chains = [
        normalise_requirement_id(str(item))
        for item in config["selected_ag_chain_ids"]
    ]
    files: dict[str, str] = {}
    for chain_id in selected_chains:
        boundary = _read_json(
            gold_root / f"{chain_id}_architecture_boundary.draft.json"
        )
        gold = _read_json(gold_root / f"{chain_id}_ag_gold.draft.json")
        if boundary.get("status") == "FROZEN" or gold.get("status") == "FROZEN":
            raise ValueError("prepare-review accepts DRAFT sources only")
        if boundary.get("chain_id") != chain_id or gold.get("chain_id") != chain_id:
            raise ValueError(f"{chain_id}: DRAFT artifact chain mismatch")
        boundary["requirement_set_digest"] = requirement_set_digest
        for component in boundary.get("components") or []:
            component_id = str(component.get("component_id") or "")
            candidate = _RESPONSIBILITY_CANDIDATES.get(component_id)
            if candidate:
                component["responsibility"] = candidate
                component["_responsibility_review"] = (
                    "HUMAN REVIEW REQUIRED: independently confirm or replace this "
                    "AI-assisted responsibility candidate; do not accept it from "
                    "the implementation or runtime result."
                )
        gold["requirement_set_digest"] = requirement_set_digest
        gold["architecture_boundary_digest"] = None
        if gold.get("source_digest") != source_digests.get(chain_id):
            raise ValueError(f"{chain_id}: gold source bytes differ from pilot config")
        boundary_path = review / f"architecture_boundary.{chain_id}.json"
        gold_path = review / f"gold.{chain_id}.json"
        _write_json(boundary_path, boundary)
        _write_json(gold_path, gold)
        files[f"architecture_boundary.{chain_id}"] = str(boundary_path)
        files[f"gold.{chain_id}"] = str(gold_path)

    taxonomy = _read_json(gold_root / "AG_FAILURE_TAXONOMY.draft.json")
    if taxonomy.get("status") == "FROZEN":
        raise ValueError("prepare-review accepts a DRAFT taxonomy only")
    taxonomy_path = review / "failure_taxonomy.json"
    _write_json(taxonomy_path, taxonomy)
    files["failure_taxonomy"] = str(taxonomy_path)

    review_manifest = {
        "schema_version": "1.0",
        "artifact_role": "OPTION2_HUMAN_REVIEW_PREPARATION",
        "experiment_namespace": _NAMESPACE,
        "status": "HUMAN_REVIEW_REQUIRED",
        "pilot_directory": str(Path(pilot_dir)),
        "pilot_configuration_digest": config["configuration_digest"],
        "pilot_manifest_status": manifest["status"],
        "selected_chain_ids": selected_chains,
        "selected_run_ids": list(config["selected_r2_run_ids"]),
        "requirements": {
            "requirement_set_digest": requirement_set_digest,
            "source_digests": {
                chain_id: source_digests[chain_id]
                for chain_id in selected_chains
            },
        },
        "review_files": files,
        "automated_decisions": {
            "set_frozen_status": False,
            "set_independent_review_flags": False,
            "set_blind_review_flags": False,
            "select_failure_labels": False,
        },
    }
    _write_json(out / "review_manifest.json", review_manifest)
    (out / "HUMAN_REVIEW_CHECKLIST.md").write_text(
        _review_checklist(selected_chains), encoding="utf-8"
    )
    return review_manifest


def _review_checklist(selected_chains: list[str]) -> str:
    chains = ", ".join(selected_chains)
    return f"""# Independent human review checklist

Selected chain(s): `{chains}`.

The files under `review_inputs/` remain DRAFT. The responsibility text is an
AI-assisted candidate and is not independent evidence.

Human-only decisions:

1. Independently review the architecture boundary without runtime verdicts,
   diagnostics, repair decisions, or `ag_contract_graph.json`.
2. Confirm or replace every responsibility, interface, allocation, timing,
   priority, discharge, and invariant fact.
3. Remove all `_review` markers; enter the real reviewer and ISO date.
4. The actual independent reviewer—not this tool—sets `status: "FROZEN"` and
   the applicable independent/blind flags.
5. Independently review the taxonomy definitions and adjudication before
   removing its review markers and freezing it.
6. Keep reviewer-facing blind packets separate from the pilot directory. The
   reviewer must not inspect runtime verdicts, diagnostics, or repairs.
7. Apply the readiness gate before classification. An invalid packet, missing
   or unverified frozen boundary, or failed provenance binding receives no
   taxonomy run label; it is not `VERIFIER_LIMITATION`.

`NO_FAILURE` means only that no visible static defect was found in the permitted
blind material and no required non-static adjudication question remains
unresolved. It is not formal A/G proof, LLM accuracy, dynamic performance, or
physical verification.
"""


def stamp_human_digest(*, path: str | Path, kind: str) -> str:
    """Compute only the digest after the human has already made all attestations."""
    artifact_path = Path(path)
    value = _read_json(artifact_path)
    if value.get("status") != "FROZEN":
        raise ValueError("human must set status='FROZEN' before digest stamping")
    if kind == "boundary":
        without_digest = copy.deepcopy(value)
        without_digest["artifact_digest"] = architecture_boundary_digest(
            without_digest
        )
        problems = validate_frozen_boundary(without_digest)
    elif kind == "taxonomy":
        without_digest = copy.deepcopy(value)
        without_digest["artifact_digest"] = artifact_digest(without_digest)
        problems = validate_frozen_failure_taxonomy(without_digest)
    else:
        raise ValueError("kind must be 'boundary' or 'taxonomy'")
    if problems:
        raise ValueError(
            "human attestations are incomplete; digest not written: "
            + "; ".join(problems)
        )
    _write_json(artifact_path, without_digest)
    return str(without_digest["artifact_digest"])


def bind_gold_provenance(
    *,
    pilot_dir: str | Path,
    boundary_path: str | Path,
    gold_path: str | Path,
) -> list[str]:
    """Bind gold to validated frozen evidence without setting review decisions."""
    config, _manifest, _archived = load_completed_pilot(pilot_dir)
    boundary = _read_json(Path(boundary_path))
    problems = validate_frozen_boundary(boundary)
    if problems:
        raise ValueError("boundary is not frozen: " + "; ".join(problems))
    gold_file = Path(gold_path)
    gold = _read_json(gold_file)
    if gold.get("chain_id") != boundary.get("chain_id"):
        raise ValueError("gold/boundary chain mismatch")
    gold["requirement_set_digest"] = config["frozen_requirement_set"][
        "requirement_set_digest"
    ]
    gold["architecture_boundary_digest"] = boundary["artifact_digest"]
    _write_json(gold_file, gold)
    return validate_frozen_gold(gold)


def build_blind_materials(
    *,
    pilot_dir: str | Path,
    evidence_dir: str | Path,
) -> dict[str, Any]:
    """Build blind-safe packets and unlabelled DRAFT templates."""
    config, _manifest, archived = load_completed_pilot(pilot_dir)
    root = Path(evidence_dir)
    human = root / "human_frozen"
    blind = root / "blind_review"
    if blind.exists():
        raise FileExistsError(f"blind-review output already exists: {blind}")
    taxonomy = _read_json(human / "failure_taxonomy.json")
    taxonomy_problems = validate_frozen_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        raise ValueError(
            "failure taxonomy is not independently frozen: "
            + "; ".join(taxonomy_problems)
        )
    sources = _source_by_chain(config)
    source_digests = config["frozen_requirement_set"]["source_digests"]
    packets: list[dict[str, Any]] = []
    label_templates: list[dict[str, Any]] = []
    for run_id in config["selected_r2_run_ids"]:
        for raw_chain in config["selected_ag_chain_ids"]:
            chain_id = normalise_requirement_id(str(raw_chain))
            boundary = _read_json(
                human / f"architecture_boundary.{chain_id}.json"
            )
            boundary_problems = validate_frozen_boundary(boundary)
            if boundary_problems:
                raise ValueError(
                    f"{chain_id} boundary is not independently frozen: "
                    + "; ".join(boundary_problems)
                )
            prediction = archived[run_id]["predictions"][chain_id]
            candidate_model = archived[run_id]["_candidate_models"][chain_id]
            packet = build_blind_review_packet(
                run_id=run_id,
                chain_id=chain_id,
                source_requirement=sources[chain_id],
                candidate_model=candidate_model,
                architecture_boundary=boundary,
                expected_requirement_digest=source_digests[chain_id],
                expected_model_digest=prediction["source_model_digest"],
            )
            packet_problems = validate_blind_packet(packet)
            if packet_problems:
                raise ValueError(
                    f"{run_id}/{chain_id}: invalid blind packet: "
                    + "; ".join(packet_problems)
                )
            safe_run = run_id.replace(":", "__")
            packet_path = blind / "packets" / f"{safe_run}__{chain_id}.json"
            _write_json(packet_path, packet)
            label = {
                "schema_version": "1.0",
                "artifact_role": "BLIND_FAILURE_LABEL",
                "experiment_namespace": _NAMESPACE,
                "configuration": _R2,
                "status": "DRAFT_FOR_INDEPENDENT_BLIND_REVIEW",
                "run_id": run_id,
                "chain_id": chain_id,
                "model_digest": packet["model_digest"],
                "requirement_digest": packet["requirement_digest"],
                "blind_packet_digest": packet["artifact_digest"],
                "failure_taxonomy_version": taxonomy["taxonomy_version"],
                "failure_taxonomy_digest": taxonomy["artifact_digest"],
                "failure_class": None,
                "reviewer": None,
                "reviewed_date": None,
                "review_protocol": {
                    "independent_human_review": False,
                    "blind_to_runtime_verdict": False,
                },
                "artifact_digest": None,
            }
            label_path = root / "operator_only" / "label_templates" / (
                f"{safe_run}__{chain_id}.json"
            )
            _write_json(label_path, label)
            packets.append(packet)
            label_templates.append(label)
    summary = {
        "status": "INDEPENDENT_BLIND_LABELS_REQUIRED",
        "packet_count": len(packets),
        "label_template_count": len(label_templates),
        "reviewer_visible_directory": str(blind),
        "operator_only_directory": str(root / "operator_only"),
    }
    _write_json(root / "blind_materials_manifest.json", summary)
    return summary


def stamp_blind_label_digests(*, evidence_dir: str | Path) -> list[str]:
    """Digest labels only after the human has supplied every review decision."""
    root = Path(evidence_dir)
    taxonomy = _read_json(root / "human_frozen" / "failure_taxonomy.json")
    taxonomy_problems = validate_frozen_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        raise ValueError(
            "failure taxonomy is not independently frozen: "
            + "; ".join(taxonomy_problems)
        )
    packet_index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((root / "blind_review" / "packets").glob("*.json")):
        packet = _read_json(path)
        problems = validate_blind_packet(packet)
        if problems:
            raise ValueError(f"{path}: invalid blind packet: " + "; ".join(problems))
        key = (
            str(packet.get("run_id") or ""),
            normalise_requirement_id(str(packet.get("chain_id") or "")),
        )
        if key in packet_index:
            raise ValueError(f"duplicate blind packet binding: {key!r}")
        packet_index[key] = packet

    prepared: list[tuple[Path, dict[str, Any]]] = []
    label_paths = sorted((root / "operator_only" / "labels").glob("*.json"))
    if not label_paths:
        raise ValueError("no human-authored blind labels found")
    for path in label_paths:
        label = _read_json(path)
        key = (
            str(label.get("run_id") or ""),
            normalise_requirement_id(str(label.get("chain_id") or "")),
        )
        packet = packet_index.get(key)
        if packet is None:
            raise ValueError(f"{path}: no matching blind packet")
        candidate = copy.deepcopy(label)
        candidate["artifact_digest"] = artifact_digest(candidate)
        problems = validate_blind_label(
            candidate,
            packet=packet,
            taxonomy=taxonomy,
        )
        if problems:
            raise ValueError(
                f"{path}: human label is incomplete; digest not written: "
                + "; ".join(problems)
            )
        prepared.append((path, candidate))
    if len(prepared) != len(packet_index):
        raise ValueError(
            "human labels must exactly cover all generated blind packets"
        )
    for path, candidate in prepared:
        _write_json(path, candidate)
    return [str(path) for path, _candidate in prepared]


def build_readiness_from_disk(
    *,
    pilot_dir: str | Path,
    evidence_dir: str | Path,
) -> dict[str, Any]:
    """Build and reproduce the sole readiness manifest from human evidence."""
    config, _pilot_manifest, archived = load_completed_pilot(pilot_dir)
    root = Path(evidence_dir)
    human = root / "human_frozen"
    labels_root = root / "operator_only" / "labels"
    packets_root = root / "blind_review" / "packets"
    chains = [
        normalise_requirement_id(str(item))
        for item in config["selected_ag_chain_ids"]
    ]
    boundaries = {
        chain: _read_json(human / f"architecture_boundary.{chain}.json")
        for chain in chains
    }
    gold = {
        chain: _read_json(human / f"gold.{chain}.json")
        for chain in chains
    }
    taxonomy = _read_json(human / "failure_taxonomy.json")
    packets = [
        _read_json(path) for path in sorted(packets_root.glob("*.json"))
    ]
    labels = [_read_json(path) for path in sorted(labels_root.glob("*.json"))]
    archived_for_gate = {
        run_id: {
            "run_manifest": item["run_manifest"],
            "predictions": item["predictions"],
        }
        for run_id, item in archived.items()
    }
    bundle = {
        "frozen_experiment_config": config,
        "architecture_boundaries": boundaries,
        "gold_by_chain": gold,
        "archived_runs": archived_for_gate,
        "blind_packets": packets,
        "blind_labels": labels,
        "failure_taxonomy": taxonomy,
    }
    readiness = build_evaluation_readiness_manifest(**bundle)
    _write_json(root / "operator_only" / "evaluation_readiness.json", readiness)
    _write_json(root / "operator_only" / "source_evidence_bundle.json", bundle)
    if readiness["evaluation_ready"]:
        require_evaluation_ready(readiness, evidence_bundle=bundle)
    return readiness
