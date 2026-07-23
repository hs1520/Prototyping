"""Strong-binding post-hoc R2 evaluation evidence gate."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_gold_template import build_gold_draft
from src.prototyping.architecture_boundary import (
    architecture_boundary_digest,
    build_architecture_boundary_draft,
)
from src.prototyping.evaluation_readiness import (
    artifact_digest,
    build_blind_review_packet,
    build_evaluation_readiness_manifest,
    require_evaluation_ready,
    validate_frozen_failure_taxonomy,
)
from src.prototyping.revised_pilot import RevisedPilotConfig


_SOURCE = (
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses."
)


def _strip_review_markers(value):
    if isinstance(value, dict):
        return {
            key: _strip_review_markers(item)
            for key, item in value.items()
            if not (str(key).startswith("_") and "review" in str(key))
        }
    if isinstance(value, list):
        return [_strip_review_markers(item) for item in value]
    return value


def _fixtures():
    config = RevisedPilotConfig(
        provider="mock",
        model="test-model",
        seeds=(0, 1, 2),
        max_iterations=1,
        code_revision="abc123",
        requirements=(_SOURCE,),
    ).to_manifest()
    frozen = config["frozen_requirement_set"]
    requirement_set_digest = frozen["requirement_set_digest"]
    requirement_digest = frozen["source_digests"]["REQ_SAFE_005"]

    boundary = _strip_review_markers(build_architecture_boundary_draft(
        REQ_SAFE_005_CHAIN,
        requirement_set_digest=requirement_set_digest,
    ))
    for component in boundary["components"]:
        component["responsibility"] = (
            f"owns {component['interfaces']['produces'][0]}"
        )
    boundary.update({
        "status": "FROZEN",
        "reviewer": "Dr. Supervisor",
        "reviewed_date": "2026-07-30",
        "review_protocol": {"independent_architecture_review": True},
    })
    boundary["artifact_digest"] = architecture_boundary_digest(boundary)

    gold = _strip_review_markers(build_gold_draft(
        REQ_SAFE_005_CHAIN,
        source_text=_SOURCE,
        requirement_set_digest=requirement_set_digest,
        architecture_boundary_digest=boundary["artifact_digest"],
    ))
    gold.update({
        "status": "FROZEN",
        "reviewer": "Dr. Supervisor",
        "reviewed_date": "2026-07-30",
        "review_protocol": {
            "blind_to_runtime_verdict": True,
            "independent_human_review": True,
        },
    })

    taxonomy = {
        "schema_version": "1.0",
        "artifact_role": "FAILURE_TAXONOMY",
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "status": "FROZEN",
        "taxonomy_version": "ag-failures-1.0",
        "classes": ["NO_FAILURE", "MODEL_SEMANTIC_FAULT", "VERIFIER_LIMITATION"],
        "class_definitions": {
            "NO_FAILURE": {
                "definition": "No failure is visible in the reviewed run/chain pair.",
                "include_when": ["all reviewed obligations are represented"],
                "exclude_when": [
                    "a contract, model, architecture, or evidence fault is visible"
                ],
            },
            "MODEL_SEMANTIC_FAULT": {
                "definition": "The candidate model does not realize its stated contract.",
                "include_when": ["required behavior is missing or unreachable"],
                "exclude_when": ["the reviewed material is insufficient to decide"],
            },
            "VERIFIER_LIMITATION": {
                "definition": "The review material is insufficient to decide conformance.",
                "include_when": ["required observation evidence is absent"],
                "exclude_when": ["a model fault is directly visible"],
            },
        },
        "adjudication": {
            "unit_of_analysis": "ONE_ARCHIVED_RUN_CHAIN_PAIR",
            "primary_label_count": 1,
            "multi_fault_rule": "label the most upstream supported root cause",
            "inconclusive_class": "VERIFIER_LIMITATION",
        },
        "reviewer": "Dr. Supervisor",
        "reviewed_date": "2026-07-30",
        "review_protocol": {"independent_human_review": True},
        "artifact_digest": None,
    }
    taxonomy["artifact_digest"] = artifact_digest(taxonomy)

    packets = []
    labels = []
    archived_runs = {}
    candidate_model = "package Candidate {}"
    # Packet model_digest binds the exact candidate-model bytes, not a wrapper.
    import hashlib
    candidate_model_digest = hashlib.sha256(
        candidate_model.encode("utf-8")
    ).hexdigest()
    for run_id in config["selected_r2_run_ids"]:
        archived_runs[run_id] = {
            "run_manifest": {
                "schema_version": "1.0",
                "artifact_role": "REVISED_PILOT_RUN_MANIFEST",
                "run_id": run_id,
                "experiment_namespace": "BLACKBOARD_AG_V1",
                "configuration": "R2-BBAG",
                "status": "COMPLETED",
                "pilot_configuration_digest": config["configuration_digest"],
                "requirement_set_digest": requirement_set_digest,
                "ag_checker_version": config["ag_checker_version"],
                "r2_generation_mode": config["r2_generation_mode"],
                "r2_intervention_version": config["r2_intervention_version"],
            },
            "predictions": {
                "REQ_SAFE_005": {
                    "artifact_role": "RUNTIME_A_G_PREDICTION",
                    "experiment_namespace": "BLACKBOARD_AG_V1",
                    "configuration": "R2-BBAG",
                    "source_requirement": "REQ_SAFE_005",
                    "source_model_digest": candidate_model_digest,
                    "checker_version": config["ag_checker_version"],
                    "graph": {},
                },
            },
        }
        packet = build_blind_review_packet(
            run_id=run_id,
            chain_id="REQ_SAFE_005",
            source_requirement=_SOURCE,
            candidate_model=candidate_model,
            expected_requirement_digest=requirement_digest,
            expected_model_digest=candidate_model_digest,
        )
        label = {
            "schema_version": "1.0",
            "artifact_role": "BLIND_FAILURE_LABEL",
            "experiment_namespace": "BLACKBOARD_AG_V1",
            "configuration": "R2-BBAG",
            "status": "FROZEN",
            "run_id": run_id,
            "chain_id": "REQ_SAFE_005",
            "model_digest": packet["model_digest"],
            "requirement_digest": requirement_digest,
            "blind_packet_digest": packet["artifact_digest"],
            "failure_taxonomy_version": taxonomy["taxonomy_version"],
            "failure_taxonomy_digest": taxonomy["artifact_digest"],
            "failure_class": "NO_FAILURE",
            "reviewer": "Independent Reviewer",
            "reviewed_date": "2026-07-31",
            "review_protocol": {
                "independent_human_review": True,
                "blind_to_runtime_verdict": True,
            },
            "artifact_digest": None,
        }
        label["artifact_digest"] = artifact_digest(label)
        packets.append(packet)
        labels.append(label)

    return {
        "frozen_experiment_config": config,
        "architecture_boundaries": {"REQ_SAFE_005": boundary},
        "gold_by_chain": {"REQ_SAFE_005": gold},
        "archived_runs": archived_runs,
        "blind_packets": packets,
        "blind_labels": labels,
        "failure_taxonomy": taxonomy,
    }


def test_complete_evidence_chain_opens_only_the_posthoc_manifest_gate():
    fixtures = _fixtures()
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert manifest["evaluation_ready"] is True
    assert manifest["pooling_permitted"] is True
    assert manifest["study_classification"] == "DESCRIPTIVE_PILOT"
    assert manifest["confirmatory_inference_permitted"] is False
    assert manifest["selected_chain_ids"] == ["REQ_SAFE_005"]
    assert len(manifest["blind_label_evidence"]) == 3
    require_evaluation_ready(manifest, evidence_bundle=fixtures)


def test_gate_fails_closed_when_any_selected_run_label_is_missing():
    fixtures = _fixtures()
    fixtures["blind_labels"].pop()
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert manifest["evaluation_ready"] is False
    assert any("frozen blind label is missing" in item for item in manifest["problems"])
    with pytest.raises(ValueError, match="not ready"):
        require_evaluation_ready(manifest)


def test_gate_binds_gold_to_requirement_and_architecture_digests():
    fixtures = _fixtures()
    fixtures["gold_by_chain"]["REQ_SAFE_005"]["source_digest"] = "c" * 64
    fixtures["gold_by_chain"]["REQ_SAFE_005"]["architecture_boundary_digest"] = "d" * 64
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any("gold source digest mismatch" in item for item in manifest["problems"])
    assert any(
        "gold architecture-boundary digest mismatch" in item
        for item in manifest["problems"]
    )


def test_gate_binds_blind_packet_to_the_archived_prediction_model():
    fixtures = _fixtures()
    run_id = fixtures["frozen_experiment_config"]["selected_r2_run_ids"][0]
    fixtures["archived_runs"][run_id]["predictions"]["REQ_SAFE_005"][
        "source_model_digest"
    ] = "c" * 64
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any("packet model digest mismatch" in item for item in manifest["problems"])


def test_blind_packet_binds_the_exact_reviewed_source_and_model_bytes():
    fixtures = _fixtures()
    packet = fixtures["blind_packets"][0]
    packet["review_material"]["candidate_model"] += "\n// altered"
    packet["review_material"]["source_requirement"] += " altered"
    packet["artifact_digest"] = artifact_digest(packet)
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any(
        "candidate model does not match model_digest" in item
        for item in manifest["problems"]
    )
    assert any(
        "source requirement does not match requirement_digest" in item
        for item in manifest["problems"]
    )


def test_blind_packet_builder_rejects_wrong_chain_or_archived_digest():
    with pytest.raises(ValueError, match="does not match chain_id"):
        build_blind_review_packet(
            run_id="seed-0:R2-BBAG",
            chain_id="REQ_SAFE_008",
            source_requirement=_SOURCE,
            candidate_model="package Candidate {}",
        )
    with pytest.raises(ValueError, match="expected_model_digest"):
        build_blind_review_packet(
            run_id="seed-0:R2-BBAG",
            chain_id="REQ_SAFE_005",
            source_requirement=_SOURCE,
            candidate_model="package Candidate {}",
            expected_model_digest="0" * 64,
        )


def test_taxonomy_requires_operational_definitions_and_adjudication():
    taxonomy = _fixtures()["failure_taxonomy"]
    taxonomy.pop("class_definitions")
    taxonomy.pop("adjudication")
    taxonomy["artifact_digest"] = artifact_digest(taxonomy)
    problems = validate_frozen_failure_taxonomy(taxonomy)
    assert any("class_definitions" in item for item in problems)
    assert any("adjudication" in item for item in problems)


def test_taxonomy_cannot_freeze_with_unresolved_review_markers():
    taxonomy = _fixtures()["failure_taxonomy"]
    taxonomy["class_definitions"]["NO_FAILURE"]["_review"] = "confirm"
    taxonomy["artifact_digest"] = artifact_digest(taxonomy)
    problems = validate_frozen_failure_taxonomy(taxonomy)
    assert any("leftover _review markers" in item for item in problems)


def test_committed_taxonomy_candidate_remains_unfrozen_and_non_independent():
    taxonomy = json.loads(
        Path("docs/gold/AG_FAILURE_TAXONOMY.draft.json").read_text(
            encoding="utf-8"
        )
    )
    assert taxonomy["status"] == "DRAFT_FOR_SUPERVISOR_REVIEW"
    assert taxonomy["review_protocol"]["independent_human_review"] is False
    assert taxonomy["reviewer"] is None
    problems = validate_frozen_failure_taxonomy(taxonomy)
    assert any("status must be 'FROZEN'" in item for item in problems)
    assert any("leftover _review markers" in item for item in problems)


def test_taxonomy_and_frozen_execution_permissions_fail_closed():
    fixtures = _fixtures()
    fixtures["failure_taxonomy"]["review_protocol"][
        "independent_human_review"
    ] = False
    fixtures["failure_taxonomy"]["artifact_digest"] = artifact_digest(
        fixtures["failure_taxonomy"]
    )
    fixtures["frozen_experiment_config"]["gold_input_permitted"] = True
    fixtures["frozen_experiment_config"]["configuration_digest"] = artifact_digest(
        {
            key: value
            for key, value in fixtures["frozen_experiment_config"].items()
            if key != "configuration_digest"
        }
    )
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any("taxonomy independent_human_review" in item for item in manifest["problems"])
    assert any("gold_input_permitted must be false" in item for item in manifest["problems"])


def test_static_failure_class_in_reference_gold_is_rejected():
    fixtures = _fixtures()
    fixtures["gold_by_chain"]["REQ_SAFE_005"]["failure_class"] = "NO_FAILURE"
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any(
        "failure_class must not appear" in item for item in manifest["problems"]
    )


def test_blind_packet_rejects_runtime_diagnostics_even_with_a_fresh_digest():
    fixtures = _fixtures()
    packet = fixtures["blind_packets"][0]
    packet["review_material"]["diagnostics"] = [{"code": "AG_X"}]
    packet["artifact_digest"] = artifact_digest(packet)
    fixtures["blind_labels"][0]["blind_packet_digest"] = packet["artifact_digest"]
    fixtures["blind_labels"][0]["artifact_digest"] = artifact_digest(
        fixtures["blind_labels"][0]
    )
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any(
        "exposes forbidden runtime/repair fields" in item
        for item in manifest["problems"]
    )


def test_gate_does_not_accept_live_or_tampered_configuration_selection():
    fixtures = _fixtures()
    config = copy.deepcopy(fixtures["frozen_experiment_config"])
    config["selected_ag_chain_ids"].append("REQ_SAFE_008")
    fixtures["frozen_experiment_config"] = config
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any("configuration_digest does not match" in item for item in manifest["problems"])
    assert any("REQ_SAFE_008" in item for item in manifest["problems"])


def test_gate_rejects_generation_mode_or_intervention_version_mixing():
    for field, replacement, expected in (
        ("r2_generation_mode", "LLM_AUTHORED", "generation mode"),
        ("r2_intervention_version", "llm-authored-v1", "intervention version"),
    ):
        fixtures = _fixtures()
        config = fixtures["frozen_experiment_config"]
        config[field] = replacement
        config["configuration_digest"] = artifact_digest({
            key: value
            for key, value in config.items()
            if key != "configuration_digest"
        })
        manifest = build_evaluation_readiness_manifest(**fixtures)
        assert manifest["evaluation_ready"] is False
        assert any(expected in item for item in manifest["problems"])


def test_ready_manifest_digest_detects_posthoc_tampering():
    manifest = build_evaluation_readiness_manifest(**_fixtures())
    manifest["selected_run_ids"].append("invented:R2-BBAG")
    with pytest.raises(ValueError, match="artifact_digest"):
        require_evaluation_ready(manifest)


def test_consumer_rejects_self_digested_manifest_without_bound_evidence():
    manifest = {
        "schema_version": "1.0",
        "artifact_role": "POSTHOC_EVALUATION_READINESS_MANIFEST",
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "configuration": "R2-BBAG",
        "evaluation_ready": True,
        "pooling_permitted": True,
        "problems": [],
    }
    manifest["artifact_digest"] = artifact_digest(manifest)
    with pytest.raises(ValueError, match="configuration_digest"):
        require_evaluation_ready(manifest)


def test_complete_shaped_manifest_still_requires_reproducible_source_evidence():
    manifest = build_evaluation_readiness_manifest(**_fixtures())
    with pytest.raises(ValueError, match="source evidence bundle is required"):
        require_evaluation_ready(manifest)

    forged = copy.deepcopy(manifest)
    forged["architecture_boundary_digests"]["REQ_SAFE_005"] = "c" * 64
    forged["gold_artifact_digests"]["REQ_SAFE_005"] = "d" * 64
    forged["failure_taxonomy_digest"] = "e" * 64
    for evidence in forged["blind_label_evidence"]:
        evidence["packet_digest"] = "1" * 64
        evidence["label_digest"] = "2" * 64
        evidence["run_manifest_digest"] = "3" * 64
        evidence["prediction_digest"] = "4" * 64
    forged["artifact_digest"] = artifact_digest(forged)
    with pytest.raises(ValueError, match="source evidence bundle is required"):
        require_evaluation_ready(forged)


def test_gate_rejects_extra_chain_evidence_and_embedded_runtime_artifact():
    fixtures = _fixtures()
    fixtures["architecture_boundaries"]["REQ_SAFE_008"] = copy.deepcopy(
        fixtures["architecture_boundaries"]["REQ_SAFE_005"]
    )
    fixtures["gold_by_chain"]["REQ_SAFE_008"] = copy.deepcopy(
        fixtures["gold_by_chain"]["REQ_SAFE_005"]
    )
    packet = fixtures["blind_packets"][0]
    packet["review_material"]["hidden"] = {
        "artifact_role": "RUNTIME_A_G_PREDICTION",
    }
    packet["artifact_digest"] = artifact_digest(packet)
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any("exactly cover" in item for item in manifest["problems"])
    assert any(
        "exposes forbidden runtime/repair fields" in item
        for item in manifest["problems"]
    )


def test_blind_packet_rejects_case_and_separator_variants():
    fixtures = _fixtures()
    packet = fixtures["blind_packets"][0]
    packet["review_material"].update({
        "runtimeVerdict": "FAIL",
        "runtime_diagnostics": ["AG_X"],
        "repairDecision": "rewrite",
    })
    packet["artifact_digest"] = artifact_digest(packet)
    manifest = build_evaluation_readiness_manifest(**fixtures)
    assert any(
        "exposes forbidden runtime/repair fields" in item
        for item in manifest["problems"]
    )
