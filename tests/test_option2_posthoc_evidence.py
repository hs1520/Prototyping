"""Operator evidence preparation remains fail-closed and human-gated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_gold_template import (
    build_gold_draft,
    validate_frozen_gold,
)
from src.prototyping.architecture_boundary import (
    build_architecture_boundary_draft,
    validate_frozen_boundary,
)
from src.prototyping.evaluation_readiness import (
    validate_blind_packet,
    validate_frozen_failure_taxonomy,
)
from src.prototyping.posthoc_evidence import (
    build_blind_materials,
    build_readiness_from_disk,
    load_completed_pilot,
    prepare_review_materials,
    stamp_blind_label_digests,
    stamp_human_digest,
)
from src.app.revised_pilot import RevisedPilotConfig


_SOURCE = (
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses."
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


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


def _pilot(tmp_path: Path) -> Path:
    config = RevisedPilotConfig(
        provider="mock",
        model="test-model",
        seeds=(0, 1, 2),
        max_iterations=1,
        code_revision="abc123",
        requirements=(_SOURCE,),
    ).to_manifest()
    root = tmp_path / "pilot"
    _write(root / "pilot_config.json", config)
    rows = []
    for seed in config["seeds"]:
        run_id = f"seed-{seed}:R2-BBAG"
        directory = root / f"seed-{seed}" / "R2-BBAG"
        model = f"package Candidate{seed} {{}}\n"
        digest = hashlib.sha256(model.encode()).hexdigest()
        run = {
            "artifact_role": "REVISED_PILOT_RUN_MANIFEST",
            "run_id": run_id,
            "status": "COMPLETED",
            "experiment_namespace": "BLACKBOARD_AG_V1",
            "configuration": "R2-BBAG",
            "pilot_configuration_digest": config["configuration_digest"],
            "requirement_set_digest": config["frozen_requirement_set"][
                "requirement_set_digest"
            ],
            "ag_checker_version": config["ag_checker_version"],
            "r2_generation_mode": config["r2_generation_mode"],
            "r2_intervention_version": config["r2_intervention_version"],
            "evaluation_ready": False,
        }
        prediction = {
            "artifact_role": "RUNTIME_A_G_PREDICTION",
            "experiment_namespace": "BLACKBOARD_AG_V1",
            "configuration": "R2-BBAG",
            "source_requirement": "REQ_SAFE_005",
            "source_model_digest": digest,
            "checker_version": config["ag_checker_version"],
            "graph": {},
        }
        _write(directory / "run_manifest.json", run)
        _write(directory / "shared_model_final.sysml", model)
        _write(directory / "ag_contract_graph.json", prediction)
        rows.append(run)
    _write(root / "pilot_manifest.json", {
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "configuration_digest": config["configuration_digest"],
        "status": "COMPLETE",
        "run_count": 9,
        "completed_run_count": 9,
        "runs": rows,
    })
    return root


def test_completed_pilot_loader_checks_exact_model_digest(tmp_path):
    root = _pilot(tmp_path)
    config, manifest, archived = load_completed_pilot(root)
    assert manifest["status"] == "COMPLETE"
    assert set(archived) == set(config["selected_r2_run_ids"])

    model = root / "seed-1" / "R2-BBAG" / "shared_model_final.sysml"
    model.write_text("package Tampered {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model digest mismatch"):
        load_completed_pilot(root)


def test_prepare_review_binds_config_but_keeps_human_decisions_open(
    tmp_path,
):
    root = _pilot(tmp_path)
    gold_dir = Path(__file__).parents[1] / "docs" / "gold"
    out = tmp_path / "evidence"
    result = prepare_review_materials(
        pilot_dir=root,
        output_dir=out,
        gold_dir=gold_dir,
    )
    assert result["status"] == "HUMAN_REVIEW_REQUIRED"
    boundary = json.loads(
        (out / "review_inputs" / "architecture_boundary.REQ_SAFE_005.json")
        .read_text()
    )
    gold = json.loads(
        (out / "review_inputs" / "gold.REQ_SAFE_005.json").read_text()
    )
    taxonomy = json.loads(
        (out / "review_inputs" / "failure_taxonomy.json").read_text()
    )
    assert boundary["requirement_set_digest"]
    assert all(item["responsibility"] for item in boundary["components"])
    assert boundary["status"] != "FROZEN"
    assert boundary["review_protocol"]["independent_architecture_review"] is False
    assert gold["architecture_boundary_digest"] is None
    assert gold["review_protocol"]["independent_human_review"] is False
    assert taxonomy["status"] != "FROZEN"
    assert taxonomy["review_protocol"]["independent_human_review"] is False


def test_digest_stamping_refuses_to_make_the_human_attestation(tmp_path):
    artifact = tmp_path / "boundary.json"
    _write(artifact, {
        "status": "DRAFT_FOR_SUPERVISOR_REVIEW",
        "artifact_digest": None,
    })
    before = artifact.read_text()
    with pytest.raises(ValueError, match="human must set"):
        stamp_human_digest(path=artifact, kind="boundary")
    assert artifact.read_text() == before


def test_blind_materials_are_built_only_after_synthetic_human_freeze(tmp_path):
    pilot = _pilot(tmp_path)
    config = json.loads((pilot / "pilot_config.json").read_text())
    evidence = tmp_path / "evidence"
    human = evidence / "human_frozen"

    boundary = _strip_review_markers(build_architecture_boundary_draft(
        REQ_SAFE_005_CHAIN,
        requirement_set_digest=config["frozen_requirement_set"][
            "requirement_set_digest"
        ],
    ))
    for component in boundary["components"]:
        component["responsibility"] = f"Owns {component['component_id']}"
    boundary.update({
        "status": "FROZEN",
        "reviewer": "Independent Architect",
        "reviewed_date": "2026-07-23",
        "review_protocol": {"independent_architecture_review": True},
    })
    _write(human / "architecture_boundary.REQ_SAFE_005.json", boundary)
    stamp_human_digest(
        path=human / "architecture_boundary.REQ_SAFE_005.json",
        kind="boundary",
    )
    boundary = json.loads(
        (human / "architecture_boundary.REQ_SAFE_005.json").read_text()
    )
    assert not validate_frozen_boundary(boundary)

    taxonomy = _strip_review_markers(json.loads(
        (Path(__file__).parents[1] / "docs" / "gold"
         / "AG_FAILURE_TAXONOMY.draft.json").read_text()
    ))
    taxonomy.update({
        "status": "FROZEN",
        "taxonomy_version": "ag-failures-1.1",
        "reviewer": "Independent Taxonomy Reviewer",
        "reviewed_date": "2026-07-23",
        "review_protocol": {"independent_human_review": True},
    })
    _write(human / "failure_taxonomy.json", taxonomy)
    stamp_human_digest(
        path=human / "failure_taxonomy.json",
        kind="taxonomy",
    )
    taxonomy = json.loads((human / "failure_taxonomy.json").read_text())
    assert not validate_frozen_failure_taxonomy(taxonomy)

    summary = build_blind_materials(
        pilot_dir=pilot,
        evidence_dir=evidence,
    )
    assert summary["packet_count"] == 3
    packets = sorted((evidence / "blind_review" / "packets").glob("*.json"))
    templates = sorted(
        (evidence / "operator_only" / "label_templates").glob("*.json")
    )
    assert len(packets) == len(templates) == 3
    assert all(not validate_blind_packet(json.loads(path.read_text()))
               for path in packets)
    for path in templates:
        label = json.loads(path.read_text())
        assert label["status"] != "FROZEN"
        assert label["failure_class"] is None
        assert label["review_protocol"]["independent_human_review"] is False

    gold = _strip_review_markers(build_gold_draft(
        REQ_SAFE_005_CHAIN,
        source_text=_SOURCE,
        requirement_set_digest=config["frozen_requirement_set"][
            "requirement_set_digest"
        ],
        architecture_boundary_digest=boundary["artifact_digest"],
    ))
    gold.update({
        "status": "FROZEN",
        "reviewer": "Independent Gold Reviewer",
        "reviewed_date": "2026-07-23",
        "review_protocol": {
            "blind_to_runtime_verdict": True,
            "independent_human_review": True,
        },
    })
    assert not validate_frozen_gold(gold)
    _write(human / "gold.REQ_SAFE_005.json", gold)

    readiness = build_readiness_from_disk(
        pilot_dir=pilot,
        evidence_dir=evidence,
    )
    assert readiness["evaluation_ready"] is False
    assert readiness["pooling_permitted"] is False
    assert any("blind label is missing" in item for item in readiness["problems"])

    labels_dir = evidence / "operator_only" / "labels"
    for template_path in templates:
        label = json.loads(template_path.read_text())
        label.update({
            "status": "FROZEN",
            "failure_class": "NO_FAILURE",
            "reviewer": "Independent Blind Reviewer",
            "reviewed_date": "2026-07-23",
            "review_protocol": {
                "independent_human_review": True,
                "blind_to_runtime_verdict": True,
            },
        })
        _write(labels_dir / template_path.name, label)
    stamped = stamp_blind_label_digests(evidence_dir=evidence)
    assert len(stamped) == 3

    readiness = build_readiness_from_disk(
        pilot_dir=pilot,
        evidence_dir=evidence,
    )
    assert readiness["problems"] == []
    assert readiness["evaluation_ready"] is True
    assert readiness["pooling_permitted"] is True
    assert readiness["study_classification"] == "DESCRIPTIVE_PILOT"
    assert readiness["confirmatory_inference_permitted"] is False
