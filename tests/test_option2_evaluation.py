from __future__ import annotations

from copy import deepcopy

import pytest

from src.prototyping.robustness_evaluation import (
    evaluate_configurations,
    evaluate_run_against_gold,
)
from src.prototyping.robustness_gold import (
    prepare_gold_template,
    prepare_run_review_template,
    run_review_id,
    validate_gold_dataset,
    validate_run_review_dataset,
)
from src.prototyping.contract_types import source_digest
from src.prototyping.requirement_inputs import build_frozen_requirement_set
from src.prototyping.posthoc_evaluation import build_uniform_posthoc_evaluation


def _run(status: str, route: str):
    source = (
        "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
        "within 0.5 seconds."
    )
    frozen = build_frozen_requirement_set([source], source="test")
    return {
        "artifact_provenance": {"run_id": "fixture-run"},
        "requirements": [source],
        "requirement_input": {**frozen, "mode": "frozen", "frozen": True},
        "requirement_contracts": {
            "contracts": [{
                "req_id": "REQ_SAFE_005", "completeness": "READY",
                "source_text": source,
                "source_digest": source_digest(source),
                "obligations": [{
                    "obligation_id": "REQ_SAFE_005.O1", "kind": "timed_response",
                    "trigger": {"concept": "critical_propulsion_failure"},
                    "response": {"concept": "deploy_parachute"},
                    "criterion": {"metric": "response_latency", "value": 0.5, "unit": "s"},
                }],
            }],
        },
        "semantic_trace_report": {
            "counts": {status: 1},
            "traces": [{
                "req_id": "REQ_SAFE_005",
                "links": [{
                    "obligation_id": "REQ_SAFE_005.O1",
                    "link_kind": "state_to_action",
                    "expected_concept": "deploy_parachute",
                    "status": status,
                }],
            }],
        },
        "failure_diagnostics": ([] if status == "PASS" else [{
            "finding_code": "ACTION_PLATFORM_BINDING_MISMATCH"
        }]),
        "repair_decisions": {"decisions": [{
            "req_id": "REQ_SAFE_005", "failure_class": route,
            "route": "preserve_result", "repair_authorised": status != "PASS",
        }]},
    }


def test_aggregation_reports_raw_counts_and_cross_repetition_agreement():
    report = evaluate_configurations({
        "B2": [_run("PASS", "NO_FAILURE"), _run("PASS", "NO_FAILURE")]
    })
    summary = report["configurations"]["B2"]

    assert summary["run_count"] == 2
    assert summary["semantic_trace_pass_rate"] == 1.0
    assert summary["cross_repetition_contract_agreement"] == 1.0
    assert summary["cross_repetition_trace_agreement"] == 1.0
    assert "exhaustive" in report["claim_boundary"]


def test_aggregation_reports_repair_success_and_strict_preservation():
    run = _run("FAIL", "MODEL_SEMANTIC_FAULT")
    run["robustness_options"] = {"failure_routing": True}
    run["repair_decisions"]["semantic_repair_attempts"] = [
        {
            "accepted": True,
            "new_diagnostic_ids": [],
            "simulation_regressions": [],
            "syntax_passed": True,
            "score_preserved": True,
        },
        {
            "accepted": False,
            "new_diagnostic_ids": [],
            "simulation_regressions": [],
            "syntax_passed": None,
            "score_preserved": None,
        },
    ]

    summary = evaluate_configurations({"B2": [run]})["configurations"]["B2"]

    assert summary["repair_attempt_count"] == 2
    assert summary["repair_accepted_count"] == 1
    assert summary["repair_success"] == 0.5
    assert summary["regression_preserved_count"] == 1
    assert summary["regression_preservation"] == 0.5


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        evaluate_configurations({"B3": []})


def test_strict_design_requires_balanced_paired_b0_b1_b2_repetitions():
    with pytest.raises(ValueError, match="equal repetition counts"):
        evaluate_configurations(
            {"B0": [_run("PASS", "NO_FAILURE")]},
            require_complete_design=True,
        )


def test_complete_design_records_paired_repetitions_and_mcts_seeds():
    groups = {}
    for configuration in ("B0", "B1", "B2"):
        runs = []
        for repetition, seed in ((1, 20), (2, 21)):
            run = _run("PASS", "NO_FAILURE")
            run.update(
                configuration=configuration,
                repetition=repetition,
                mcts_seed=seed,
                generation_seed=1000 + repetition,
                generation_seed_control="PROVIDER_BEST_EFFORT",
            )
            runs.append(run)
        groups[configuration] = runs

    report = evaluate_configurations(
        groups, require_complete_design=True
    )

    assert report["experimental_design_control"] == {
        "required": True,
        "complete_configurations": True,
        "balanced_repetitions": True,
        "repetitions_recorded": True,
        "paired_repetitions": True,
        "paired_mcts_seeds": True,
        "paired_generation_seeds": True,
        "generation_seed_controls": ["PROVIDER_BEST_EFFORT"],
        "run_counts": {"B0": 2, "B1": 2, "B2": 2},
        "valid": True,
    }
    assert report["controlled_comparison_valid"] is False


def test_complete_design_with_uniform_posthoc_is_valid_controlled_comparison():
    base = _run("PASS", "NO_FAILURE")
    posthoc = build_uniform_posthoc_evaluation(
        model_text="package D {}",
        model_name="D",
        frozen_requirements=base["requirement_input"],
    )
    groups = {}
    for configuration in ("B0", "B1", "B2"):
        run = _run("PASS", "NO_FAILURE")
        run.update(
            configuration=configuration,
            repetition=1,
            mcts_seed=20,
            generation_seed=1020,
            generation_seed_control="PROVIDER_BEST_EFFORT",
            posthoc_evaluation=deepcopy(posthoc),
            posthoc_model_digest_verified=True,
        )
        groups[configuration] = [run]

    report = evaluate_configurations(
        groups,
        require_complete_design=True,
        require_uniform_posthoc=True,
    )

    assert report["controlled_comparison_valid"] is True
    assert report["posthoc_measurement_control"]["verified"] is True


def test_strict_measurement_rejects_different_posthoc_stack_versions():
    base = _run("PASS", "NO_FAILURE")
    posthoc = build_uniform_posthoc_evaluation(
        model_text="package D {}",
        model_name="D",
        frozen_requirements=base["requirement_input"],
    )
    groups = {}
    for configuration in ("B0", "B1", "B2"):
        run = _run("PASS", "NO_FAILURE")
        sidecar = deepcopy(posthoc)
        if configuration == "B2":
            sidecar["platform_binding_version"] = "different-version"
        run.update(
            configuration=configuration,
            repetition=1,
            mcts_seed=20,
            generation_seed=1020,
            posthoc_evaluation=sidecar,
            posthoc_model_digest_verified=True,
        )
        groups[configuration] = [run]

    with pytest.raises(ValueError, match="same complete versioned stack"):
        evaluate_configurations(groups, require_uniform_posthoc=True)


def test_run_cannot_be_supplied_under_the_wrong_configuration():
    run = _run("PASS", "NO_FAILURE")
    run["configuration"] = "B2"

    with pytest.raises(ValueError, match="supplied under B0"):
        evaluate_configurations({"B0": [run]})


def test_same_archived_run_cannot_be_counted_twice():
    run = _run("PASS", "NO_FAILURE")
    run["_archive_run_dir"] = "/archive/run-1"

    with pytest.raises(ValueError, match="more than once"):
        evaluate_configurations({"B0": [run, dict(run)]})


def test_different_requirement_sets_are_rejected():
    left = _run("PASS", "NO_FAILURE")
    right = _run("PASS", "NO_FAILURE")
    left["requirements"] = ["REQ-A: first"]
    right["requirements"] = ["REQ-B: second"]

    with pytest.raises(ValueError, match="same verified frozen"):
        evaluate_configurations({"B0": [left], "B2": [right]})


def test_variable_full_set_can_be_described_but_not_controlled():
    left = _run("PASS", "NO_FAILURE")
    right = _run("PASS", "NO_FAILURE")
    right["requirements"].append(
        "REQ-FUNC-999: The system shall log a diagnostic event."
    )

    report = evaluate_configurations(
        {"B0": [left], "B2": [right]}, require_frozen_inputs=False
    )

    assert report["requirement_input_control"]["verified"] is False
    assert report["requirement_input_control"]["full_set_variants"] == 2


def test_uniform_posthoc_measures_b0_without_enabling_intervention():
    run = _run("PASS", "NO_FAILURE")
    run.pop("requirement_contracts")
    run.pop("semantic_trace_report")
    run.pop("failure_diagnostics")
    run.pop("repair_decisions")
    model = """package D {
        requirement def REQ_SAFE_005 {
            doc /* Critical propulsion failure shall deploy the parachute within 0.5 seconds. */
        }
        part def SafetyMonitor {
            satisfy requirement REQ_SAFE_005;
        }
    }"""
    posthoc = build_uniform_posthoc_evaluation(
        model_text=model,
        model_name="D",
        frozen_requirements=run["requirement_input"],
    )
    run["posthoc_evaluation"] = posthoc
    run["posthoc_model_digest_verified"] = True

    report = evaluate_configurations({"B0": [run]})

    summary = report["configurations"]["B0"]
    assert summary["supported_trace_count"] == 1
    assert summary["diagnostic_count"] > 0
    assert summary["repair_authorised_count"] == 0
    assert summary["measurement_source"] == "UNIFORM_POSTHOC_MEASUREMENT"
    assert report["posthoc_measurement_control"]["verified"] is True
    assert posthoc["measurement_only"] is True
    assert posthoc["mutation_permitted"] is False
    assert all(
        item["intervention_applied"] is False
        for item in posthoc["failure_classifications"]["decisions"]
    )


def test_gold_and_run_review_bind_to_uniform_posthoc_artifacts():
    run = _run("PASS", "NO_FAILURE")
    run.pop("artifact_provenance")
    model = """package D {
        requirement def REQ_SAFE_005 { doc /* parachute */ }
        part def SafetyMonitor { satisfy requirement REQ_SAFE_005; }
    }"""
    posthoc = build_uniform_posthoc_evaluation(
        model_text=model,
        model_name="D",
        frozen_requirements=run["requirement_input"],
    )
    run["posthoc_evaluation"] = posthoc
    run["posthoc_model_digest_verified"] = True
    gold = prepare_gold_template(run, include_pipeline_candidates=True)
    gold["reviewer"] = "reviewer"
    gold["reviewed_at"] = "2026-07-19T00:00:00Z"
    row = gold["requirements"][0]
    row["review_status"] = "REVIEWED"
    row["reviewed_contract"] = row["pipeline_candidate"]["contract"]
    row["reviewed_trace_links"] = [
        {
            "obligation_id": link["obligation_id"],
            "link_kind": link["link_kind"],
            "expected_concept": link["expected_concept"],
        }
        for link in row["pipeline_candidate"]["trace_links"]
    ]

    review = prepare_run_review_template(run, gold)

    assert review["model_digest"] == posthoc["model_digest"]
    assert run_review_id(run) == f"model:{posthoc['model_digest']}"


def test_controlled_evaluation_rejects_mixed_measurement_sources():
    left = _run("PASS", "NO_FAILURE")
    right = _run("PASS", "NO_FAILURE")
    left["posthoc_evaluation"] = build_uniform_posthoc_evaluation(
        model_text="package D {}",
        model_name="D",
        frozen_requirements=left["requirement_input"],
    )

    with pytest.raises(ValueError, match="cannot mix"):
        evaluate_configurations({"B0": [left], "B2": [right]})


def test_gold_template_is_not_usable_until_independently_reviewed():
    gold = prepare_gold_template(_run("PASS", "NO_FAILURE"))

    errors = validate_gold_dataset(gold)

    assert any("not REVIEWED" in item for item in errors)
    assert "pipeline_candidate" not in gold["requirements"][0]
    assert gold["requirements"][0]["reviewed_contract"]["completeness"] is None


def test_candidate_context_is_explicit_second_pass_only():
    gold = prepare_gold_template(
        _run("PASS", "NO_FAILURE"), include_pipeline_candidates=True
    )

    assert gold["requirements"][0]["pipeline_candidate"]


def _reviewed_gold_and_run_review(run):
    gold = prepare_gold_template(run, include_pipeline_candidates=True)
    gold["reviewer"] = "fixture-reviewer"
    gold["reviewed_at"] = "2026-07-19T00:00:00Z"
    row = gold["requirements"][0]
    candidate = row["pipeline_candidate"]
    row["review_status"] = "REVIEWED"
    row["reviewed_contract"] = {
        "completeness": candidate["contract"]["completeness"],
        "obligations": candidate["contract"]["obligations"],
    }
    row["reviewed_trace_links"] = [
        {
            "obligation_id": link["obligation_id"],
            "link_kind": link["link_kind"],
            "expected_concept": link["expected_concept"],
        }
        for link in candidate["trace_links"]
    ]
    row["reviewed_pattern_ids"] = candidate["pattern_ids"]
    review = prepare_run_review_template(run, gold)
    review["reviewer"] = "fixture-run-reviewer"
    review["reviewed_at"] = "2026-07-19T01:00:00Z"
    review_row = review["requirements"][0]
    review_row["review_status"] = "REVIEWED"
    for link, candidate_link in zip(
        review_row["reviewed_trace_links"], candidate["trace_links"]
    ):
        link["status"] = candidate_link["status"]
    review_row["reviewed_failure_class"] = candidate["failure_class"]
    return gold, review


def test_reviewed_gold_produces_explicit_accuracy_and_f1_metrics():
    run = _run("PASS", "NO_FAILURE")
    gold, review = _reviewed_gold_and_run_review(run)

    metrics = evaluate_run_against_gold(run, gold, run_review=review)

    assert validate_gold_dataset(gold) == []
    assert validate_run_review_dataset(review, gold, run=run) == []
    assert metrics["contract_completeness_accuracy"] == 1.0
    assert metrics["contract_completeness_raw_counts"] == {
        "READY": {"expected": 1, "scored": 1, "correct": 1}
    }
    assert metrics["contract_slots"]["f1"] == 1.0
    assert metrics["semantic_trace_links"]["f1"] == 1.0
    assert metrics["routing_accuracy"] == 1.0


def test_selected_gold_remains_stable_when_unscored_llm_additions_change():
    run = _run("PASS", "NO_FAILURE")
    gold, review = _reviewed_gold_and_run_review(run)
    run["requirements"].append(
        "REQ-FUNC-999: The system shall log a diagnostic event."
    )

    metrics = evaluate_run_against_gold(run, gold, run_review=review)

    assert metrics["contract_completeness_accuracy"] == 1.0
    assert metrics["full_requirement_set_matches_gold_source_run"] is False


def test_repair_metrics_require_and_use_recorded_attempt_evidence():
    run = _run("PASS", "NO_FAILURE")
    gold, review = _reviewed_gold_and_run_review(run)
    assert evaluate_run_against_gold(run, gold)["repair_success"] is None

    run["repair_decisions"]["semantic_repair_attempts"] = [{
        "accepted": True,
        "new_diagnostic_ids": [],
        "simulation_regressions": [],
        "syntax_passed": True,
        "score_preserved": True,
    }]
    metrics = evaluate_run_against_gold(run, gold, run_review=review)

    assert metrics["repair_success"] == 1.0
    assert metrics["regression_preservation"] == 1.0


def test_unexecuted_preservation_gates_are_not_counted_as_preserved():
    run = _run("FAIL", "MODEL_SEMANTIC_FAULT")
    gold, review = _reviewed_gold_and_run_review(run)
    review["requirements"][0]["reviewed_rationale_codes"] = [
        "MODEL_SEMANTIC_FINDING"
    ]
    run["repair_decisions"]["semantic_repair_attempts"] = [{
        "accepted": False,
        "new_diagnostic_ids": [],
        "simulation_regressions": [],
        "syntax_passed": None,
        "score_preserved": None,
    }]

    metrics = evaluate_run_against_gold(run, gold, run_review=review)

    assert metrics["repair_success"] == 0.0
    assert metrics["regression_preservation"] == 0.0


def test_static_gold_cannot_freeze_a_run_dependent_trace_status():
    run = _run("PASS", "NO_FAILURE")
    gold, _ = _reviewed_gold_and_run_review(run)
    gold["requirements"][0]["reviewed_trace_links"][0]["status"] = "PASS"

    assert any("must not contain run status" in item for item in validate_gold_dataset(gold))


def test_slot_scoring_normalizes_omitted_defaults_and_null_fields():
    run = _run("PASS", "NO_FAILURE")
    obligation = run["requirement_contracts"]["contracts"][0]["obligations"][0]
    obligation["trigger"].update(
        condition_kind="event", variable=None, comparator=None, value=None,
        unit=None, qualifiers=[],
    )
    obligation["response"].update(polarity="required", qualifiers=[])
    gold, review = _reviewed_gold_and_run_review(run)
    reviewed = gold["requirements"][0]["reviewed_contract"]["obligations"][0]
    reviewed["trigger"] = {"concept": "critical_propulsion_failure"}
    reviewed["response"] = {"concept": "deploy_parachute"}

    metrics = evaluate_run_against_gold(run, gold, run_review=review)

    assert metrics["contract_slots"]["f1"] == 1.0
    assert metrics["invented_slot_rate"] == 0.0
