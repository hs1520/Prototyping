"""Gate-protected descriptive reporting over archived Option 2 evidence."""
from __future__ import annotations

import copy

import pytest

from src.prototyping.evaluation_readiness import (
    build_evaluation_readiness_manifest,
)
from src.prototyping.posthoc_reporting import build_posthoc_descriptive_report
from tests.test_option2_ag_evaluation import _prediction
from tests.test_option2_evaluation_readiness import _fixtures


def _ready_bundle():
    bundle = _fixtures()
    runtime = _prediction()
    for archived in bundle["archived_runs"].values():
        prediction = archived["predictions"]["REQ_SAFE_005"]
        prediction["graph"] = copy.deepcopy(runtime["graph"])
    manifest = build_evaluation_readiness_manifest(**bundle)
    assert manifest["evaluation_ready"] is True
    return bundle, manifest


def test_ready_report_scores_separate_categories_and_preserves_claim_limits():
    bundle, manifest = _ready_bundle()
    report = build_posthoc_descriptive_report(
        readiness_manifest=manifest,
        evidence_bundle=bundle,
    )

    assert report["artifact_role"] == "POSTHOC_DESCRIPTIVE_PILOT_REPORT"
    assert report["metric_name"] == "decomposition_extraction_agreement"
    assert "f1" not in report
    assert len(report["evaluations"]) == 3
    assert set(report["separate_category_summary"]) == {
        "guarantee_allocation",
        "assumption_discharge",
        "timing_agreement",
        "priority_agreement",
        "realization_link_agreement",
        "observation_link_agreement",
    }
    assert all(
        item["exact_agreement_rate"] == 1.0
        for item in report["separate_category_summary"].values()
    )
    assert report["realization_link_agreement"]["status"] == "EVALUATED"
    assert report["observation_link_agreement"]["status"] == "EVALUATED"
    assert report["failure_classification"]["runtime_classified_pairs"] == 0
    assert report["failure_classification"][
        "accuracy_among_runtime_classified"
    ] is None
    assert report["repair_success"]["status"] == "ZERO_DENOMINATOR"
    assert report["cross_seed_graph_stability"]["mean_jaccard"] == 1.0
    assert report["claims"] == {
        "formal_ag_proof": False,
        "llm_accuracy": False,
        "physical_verification": False,
        "confirmatory_inference": False,
        "descriptive_pilot_only": True,
    }


def test_report_refuses_a_manifest_not_reproduced_from_the_source_bundle():
    bundle, manifest = _ready_bundle()
    forged = copy.deepcopy(manifest)
    forged["pooling_permitted"] = False
    with pytest.raises(ValueError, match="does not reproduce"):
        build_posthoc_descriptive_report(
            readiness_manifest=forged,
            evidence_bundle=bundle,
        )


def test_runtime_failure_class_is_compared_without_promoting_checker_pass():
    bundle, manifest = _ready_bundle()
    first_run = manifest["selected_run_ids"][0]
    routing = {
        (first_run, "REQ_SAFE_005"): {
            "failures": [{"classification": "MODEL_SEMANTIC_FAULT"}],
        },
    }
    report = build_posthoc_descriptive_report(
        readiness_manifest=manifest,
        evidence_bundle=bundle,
        failure_routing_by_binding=routing,
    )
    failure = report["failure_classification"]
    assert failure["runtime_classified_pairs"] == 1
    assert failure["classification_coverage"] == pytest.approx(1 / 3)
    # The synthetic fixture's blind labels are NO_FAILURE, so this is a mismatch.
    assert failure["matches_among_runtime_classified"] == 0
    unassigned = [
        item for item in report["evaluations"]
        if item["runtime_primary_failure_class"]["status"] == "UNASSIGNED"
    ]
    assert len(unassigned) == 2
