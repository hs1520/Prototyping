"""Gate-protected post-hoc reporting for the revised Option 2 pilot.

The only production consumer that pools independent human gold with archived
``RUNTIME_A_G_PREDICTION`` artifacts; it reproduces the complete strong-binding
readiness manifest before evaluating anything, imports no runtime
extractor/checker, and does not treat a checker PASS as ``NO_FAILURE``. The
report is descriptive: allocation, discharge, timing, priority and invariant
agreement stay separate, unavailable metrics are ``NOT_EVALUABLE``, and
no-proof/no-physical-validation claim flags travel with it.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .ag_evaluation import evaluate_ag_against_gold
from .artifact_store import atomic_write_json, read_json_object
from .evaluation_protocol import build_descriptive_pilot_manifest
from .evaluation_readiness import (
    artifact_digest,
    build_evaluation_readiness_manifest,
    require_evaluation_ready,
)


REPORT_SCHEMA_VERSION = "1.0"
REPORT_ROLE = "POSTHOC_DESCRIPTIVE_PILOT_REPORT"


_read_json = read_json_object


def _index(
    items: Iterable[Mapping[str, Any]],
    *,
    name: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in items:
        key = (str(item.get("run_id") or ""), str(item.get("chain_id") or ""))
        if not all(key):
            raise ValueError(f"{name} has an empty run_id/chain_id binding")
        if key in result:
            raise ValueError(f"{name} has duplicate binding {key!r}")
        result[key] = item
    return result


def _prf_exact(value: Mapping[str, Any]) -> bool:
    return (
        int(value.get("fp", -1)) == 0
        and int(value.get("fn", -1)) == 0
        and float(value.get("precision", -1)) == 1.0
        and float(value.get("recall", -1)) == 1.0
        and float(value.get("f1", -1)) == 1.0
    )


def _category_exact(name: str, value: Mapping[str, Any]) -> bool:
    if name in {
        "guarantee_allocation",
        "assumption_discharge",
        "realization_link_agreement",
        "observation_link_agreement",
    }:
        return _prf_exact(value)
    if name == "timing_agreement":
        return (
            value.get("origin_match") is True
            and value.get("deadline_match") is True
            and _prf_exact(value.get("segment_prf") or {})
            and value.get("additive_total_match") is True
            and value.get("within_deadline_match") is True
        )
    if name == "priority_agreement":
        return (
            value.get("response_set_id_match") is True
            and value.get("members_match") is True
            and _prf_exact(value.get("edge_prf") or {})
            and value.get("trigger_match") is True
            and value.get("arbitration_topology_conforms") is True
        )
    if name == "invariant_agreement":
        denominators = [
            item for item in (
                value.get("stakeholder"),
                value.get("student_derived_design_constraint"),
            )
            if isinstance(item, Mapping)
        ]
        return bool(denominators) and all(_prf_exact(item) for item in denominators)
    raise ValueError(f"unknown agreement category: {name}")


def _canonical_fact(category: str, value: Any) -> str:
    return category + ":" + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _graph_fact_set(graph: Mapping[str, Any]) -> set[str]:
    facts: set[str] = set()
    for item in graph.get("allocations") or ():
        facts.add(_canonical_fact("allocation", {
            "owner": item.get("owner"),
            "contract": item.get("contract"),
            "guarantee": item.get("guarantee"),
        }))
    for item in graph.get("discharge_edges") or ():
        facts.add(_canonical_fact("discharge", {
            "component": item.get("component"),
            "assumption": item.get("assumption"),
            "by": item.get("by"),
        }))
    for item in graph.get("realization_links") or ():
        for path in item.get("response_paths") or ():
            facts.add(_canonical_fact("realization", {
                "contract": item.get("contract"),
                "owner": item.get("owner"),
                "behavior": item.get("behavior"),
                "initial_state": item.get("initial_state"),
                "continuous_guarantee": item.get("continuous_guarantee"),
                "source": path.get("source"),
                "trigger": path.get("trigger"),
                "target": path.get("target"),
                "guard": path.get("guard"),
                "action": path.get("action"),
            }))
    for item in graph.get("observation_links") or ():
        facts.add(_canonical_fact("observation", {
            "contract": item.get("contract"),
            "verification": item.get("verification"),
            "observation": item.get("observation"),
        }))
    timing = graph.get("timing")
    if isinstance(timing, Mapping):
        facts.add(_canonical_fact("timing_origin", timing.get("origin")))
        facts.add(_canonical_fact("timing_deadline", timing.get("deadline")))
        for index, item in enumerate(timing.get("segments") or ()):
            facts.add(_canonical_fact("timing_segment", {
                "index": index,
                "component": item.get("component"),
                "budget": item.get("budget"),
            }))
    priority = graph.get("priority")
    if isinstance(priority, Mapping):
        facts.add(_canonical_fact(
            "priority_response_set", priority.get("response_set_id")
        ))
        for member in priority.get("members") or ():
            facts.add(_canonical_fact("priority_member", str(member)))
        for edge in priority.get("edges") or ():
            facts.add(_canonical_fact("priority_edge", {
                "higher": edge.get("higher"),
                "lower": edge.get("lower"),
            }))
        facts.add(_canonical_fact("priority_trigger", priority.get("trigger")))
    for item in graph.get("invariants") or ():
        facts.add(_canonical_fact("invariant", item))
    return facts


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _runtime_primary_class(
    routing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failures = list((routing or {}).get("failures") or ())
    classes = sorted({
        str(item.get("classification"))
        for item in failures
        if isinstance(item, Mapping) and item.get("classification")
    })
    if not classes:
        return {
            "status": "UNASSIGNED",
            "primary_class": None,
            "reason": (
                "runtime routing emitted no failure class; checker PASS is not "
                "promoted to NO_FAILURE or VERIFIER_LIMITATION"
            ),
        }
    if len(classes) == 1:
        return {"status": "CLASSIFIED", "primary_class": classes[0]}
    return {
        "status": "MULTIPLE_RUNTIME_CLASSES",
        "primary_class": None,
        "classes": classes,
        "reason": "runtime diagnostics did not select one taxonomy primary label",
    }


def _not_evaluable(reason: str) -> dict[str, Any]:
    return {"status": "NOT_EVALUABLE", "reason": reason}


def _arm_summary(run_manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        item for item in run_manifests
        if item.get("status") == "COMPLETED"
    ]
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row.get("configuration")), []).append(row)
    result: dict[str, Any] = {}
    for arm, items in sorted(by_arm.items()):
        scores = [float(item["final_score"]) for item in items
                  if item.get("final_score") is not None]
        result[arm] = {
            "n": len(items),
            "score_mean": mean(scores) if scores else None,
            "score_range": [min(scores), max(scores)] if scores else None,
            "llm_calls_total": sum(int(item.get("llm_calls") or 0) for item in items),
            "llm_tokens_total": sum(
                int(item.get("llm_total_tokens") or 0) for item in items
            ),
            "elapsed_seconds_total": sum(
                float(item.get("elapsed_seconds") or 0.0) for item in items
            ),
        }
    indexed = {
        (int(item["seed"]), str(item["configuration"])): item
        for item in rows
        if item.get("seed") is not None and item.get("configuration")
    }
    paired: list[dict[str, Any]] = []
    for seed in sorted({key[0] for key in indexed}):
        keys = [
            (seed, "R0-CURRENT"),
            (seed, "R1-BBCTX"),
            (seed, "R2-BBAG"),
        ]
        if not all(key in indexed for key in keys):
            continue
        r0, r1, r2 = (float(indexed[key]["final_score"]) for key in keys)
        paired.append({
            "seed": seed,
            "R1_minus_R0": r1 - r0,
            "R2_minus_R1": r2 - r1,
            "R2_minus_R0": r2 - r0,
        })
    return {"arms": result, "paired_score_differences": paired}


def build_posthoc_descriptive_report(
    *,
    readiness_manifest: Mapping[str, Any],
    evidence_bundle: Mapping[str, Any],
    failure_routing_by_binding: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
    repair_decisions_by_binding: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
    run_manifests: Iterable[Mapping[str, Any]] = (),
    coordination_metrics_by_run: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate the complete frozen pilot evidence after reproducing its gate."""
    reproduced = build_evaluation_readiness_manifest(**dict(evidence_bundle))
    if dict(readiness_manifest) != reproduced:
        raise ValueError(
            "saved readiness manifest does not reproduce from the supplied "
            "source-evidence bundle"
        )
    require_evaluation_ready(reproduced, evidence_bundle=evidence_bundle)

    selected_runs = [str(item) for item in reproduced["selected_run_ids"]]
    selected_chains = [str(item) for item in reproduced["selected_chain_ids"]]
    protocol = build_descriptive_pilot_manifest(selected_runs)
    archived = evidence_bundle.get("archived_runs") or {}
    gold_by_chain = evidence_bundle.get("gold_by_chain") or {}
    labels = _index(
        evidence_bundle.get("blind_labels") or (),
        name="blind labels",
    )
    routing = failure_routing_by_binding or {}
    repairs = repair_decisions_by_binding or {}

    evaluations: list[dict[str, Any]] = []
    graph_facts: dict[tuple[str, str], set[str]] = {}
    category_exact: dict[str, list[bool]] = {}
    runtime_classified = 0
    runtime_human_matches = 0
    attempted_repairs = 0
    accepted_repairs = 0

    for run_id in selected_runs:
        run = archived.get(run_id) or {}
        predictions = run.get("predictions") or {}
        for chain_id in selected_chains:
            prediction = predictions.get(chain_id)
            gold = gold_by_chain.get(chain_id)
            label = labels.get((run_id, chain_id))
            if not isinstance(prediction, Mapping):
                raise ValueError(f"{run_id}/{chain_id}: prediction is missing")
            if not isinstance(gold, Mapping):
                raise ValueError(f"{run_id}/{chain_id}: frozen gold is missing")
            if not isinstance(label, Mapping):
                raise ValueError(f"{run_id}/{chain_id}: blind label is missing")
            evaluation = evaluate_ag_against_gold(prediction, gold)
            categories: dict[str, bool] = {}
            for name in (
                "guarantee_allocation",
                "assumption_discharge",
                "timing_agreement",
                "priority_agreement",
                "invariant_agreement",
                "realization_link_agreement",
                "observation_link_agreement",
            ):
                value = evaluation.get(name)
                if isinstance(value, Mapping):
                    exact = _category_exact(name, value)
                    categories[name] = exact
                    category_exact.setdefault(name, []).append(exact)

            runtime_class = _runtime_primary_class(
                routing.get((run_id, chain_id))
            )
            human_class = str(label.get("failure_class") or "")
            classification_match = (
                runtime_class["status"] == "CLASSIFIED"
                and runtime_class["primary_class"] == human_class
            )
            if runtime_class["status"] == "CLASSIFIED":
                runtime_classified += 1
                runtime_human_matches += int(classification_match)

            repair_rows = list(
                (repairs.get((run_id, chain_id)) or {}).get("decisions") or ()
            )
            attempted_repairs += len(repair_rows)
            accepted_repairs += sum(
                1 for item in repair_rows
                if isinstance(item, Mapping)
                and str(item.get("decision") or item.get("status") or "").upper()
                in {"ACCEPT", "ACCEPTED", "COMPLETED"}
            )
            graph_facts[(run_id, chain_id)] = _graph_fact_set(
                prediction.get("graph") or {}
            )
            evaluations.append({
                "run_id": run_id,
                "chain_id": chain_id,
                "source_model_digest": prediction.get("source_model_digest"),
                "agreement": evaluation,
                "category_exact_agreement": categories,
                "runtime_primary_failure_class": runtime_class,
                "human_blind_failure_class": human_class,
                "failure_classification_match": classification_match,
                "human_label_digest": label.get("artifact_digest"),
            })

    category_summary = {
        name: {
            "evaluated_run_chain_pairs": len(values),
            "exact_agreement_pairs": sum(values),
            "exact_agreement_rate": (
                sum(values) / len(values) if values else None
            ),
        }
        for name, values in sorted(category_exact.items())
    }
    if "realization_link_agreement" in category_summary:
        realization = {
            "status": "EVALUATED",
            **category_summary["realization_link_agreement"],
        }
    elif not any("realization_links" in gold for gold in gold_by_chain.values()):
        realization = _not_evaluable(
            "frozen gold contains no independently reviewed realization_links; "
            "runtime realization output cannot serve as its own gold"
        )
    else:
        realization = _not_evaluable(
            "realization-link comparison support is not yet defined for this "
            "frozen evaluator schema"
        )
    if "observation_link_agreement" in category_summary:
        observation = {
            "status": "EVALUATED",
            **category_summary["observation_link_agreement"],
        }
    elif not any("observation_links" in gold for gold in gold_by_chain.values()):
        observation = _not_evaluable(
            "frozen gold contains no independently reviewed observation_links"
        )
    else:
        observation = _not_evaluable(
            "observation-link comparison support is not yet defined for this "
            "frozen evaluator schema"
        )

    stability_pairs: list[dict[str, Any]] = []
    for chain_id in selected_chains:
        for left_run, right_run in combinations(selected_runs, 2):
            left = graph_facts[(left_run, chain_id)]
            right = graph_facts[(right_run, chain_id)]
            stability_pairs.append({
                "chain_id": chain_id,
                "left_run_id": left_run,
                "right_run_id": right_run,
                "left_fact_count": len(left),
                "right_fact_count": len(right),
                "jaccard": _jaccard(left, right),
            })

    labels_requiring_non_static_evidence = {
        chain_id
        for (_run_id, chain_id), label in labels.items()
        if label.get("failure_class") == "VERIFIER_LIMITATION"
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_role": REPORT_ROLE,
        "experiment_namespace": reproduced["experiment_namespace"],
        "configuration": reproduced["configuration"],
        "producing_stage": "GATE_PROTECTED_POSTHOC_EVALUATION",
        "measurement_boundary": "EVALUATOR_ONLY",
        "readiness_manifest_digest": reproduced["artifact_digest"],
        "configuration_digest": reproduced["configuration_digest"],
        "requirement_set_digest": reproduced["requirement_set_digest"],
        "protocol": protocol,
        "metric_name": "decomposition_extraction_agreement",
        "metric_interpretation": (
            "deterministic-emitter round-trip agreement, not LLM accuracy"
        ),
        "posthoc_pooling_authorized": True,
        "evaluations": evaluations,
        "separate_category_summary": category_summary,
        "realization_link_agreement": realization,
        "observation_link_agreement": observation,
        "failure_classification": {
            "selected_run_chain_pairs": len(evaluations),
            "runtime_classified_pairs": runtime_classified,
            "classification_coverage": (
                runtime_classified / len(evaluations) if evaluations else None
            ),
            "matches_among_runtime_classified": runtime_human_matches,
            "accuracy_among_runtime_classified": (
                runtime_human_matches / runtime_classified
                if runtime_classified else None
            ),
            "note": (
                "empty runtime diagnostics are UNASSIGNED, not NO_FAILURE; "
                "human VERIFIER_LIMITATION labels describe evidence limits"
            ),
        },
        "repair_success": {
            "attempted_repairs": attempted_repairs,
            "accepted_repairs": accepted_repairs,
            "success_rate": (
                accepted_repairs / attempted_repairs
                if attempted_repairs else None
            ),
            "status": "EVALUATED" if attempted_repairs else "ZERO_DENOMINATOR",
        },
        "cross_seed_graph_stability": {
            "pairwise": stability_pairs,
            "mean_jaccard": (
                mean(item["jaccard"] for item in stability_pairs)
                if stability_pairs else None
            ),
        },
        "non_static_evidence_coverage": {
            "guarantees_requiring_independent_dynamic_or_physical_evidence": len(
                labels_requiring_non_static_evidence
            ),
            "guarantees_with_executed_dynamic_or_physical_evidence": 0,
            "coverage": (
                0.0 if labels_requiring_non_static_evidence else None
            ),
            "status": (
                "VERIFIER_LIMITATION"
                if labels_requiring_non_static_evidence else "NOT_APPLICABLE"
            ),
        },
        "arm_descriptive_summary": _arm_summary(run_manifests),
        "coordination_metrics": {
            run_id: dict(value)
            for run_id, value in sorted(
                (coordination_metrics_by_run or {}).items()
            )
        },
        "claims": {
            "formal_ag_proof": False,
            "llm_accuracy": False,
            "physical_verification": False,
            "confirmatory_inference": False,
            "descriptive_pilot_only": True,
        },
        "artifact_digest": None,
    }
    report["artifact_digest"] = artifact_digest(report)
    return report


def evaluate_ready_pilot_from_disk(
    *,
    pilot_dir: str | Path,
    evidence_dir: str | Path,
) -> dict[str, Any]:
    """Load immutable archives, reproduce readiness, and write the report."""
    pilot = Path(pilot_dir)
    evidence = Path(evidence_dir)
    operator = evidence / "operator_only"
    readiness = _read_json(operator / "evaluation_readiness.json")
    bundle = _read_json(operator / "source_evidence_bundle.json")
    pilot_manifest = _read_json(pilot / "pilot_manifest.json")

    failure_routing: dict[tuple[str, str], Mapping[str, Any]] = {}
    repair_decisions: dict[tuple[str, str], Mapping[str, Any]] = {}
    coordination: dict[str, Mapping[str, Any]] = {}
    for run_id in readiness.get("selected_run_ids") or ():
        seed, configuration = str(run_id).split(":", 1)
        run_dir = pilot / seed / configuration
        for chain_id in readiness.get("selected_chain_ids") or ():
            failure_path = run_dir / "failure_diagnostics.json"
            repair_path = run_dir / "repair_decisions.json"
            if failure_path.is_file():
                failure_routing[(str(run_id), str(chain_id))] = _read_json(
                    failure_path
                )
            if repair_path.is_file():
                repair_decisions[(str(run_id), str(chain_id))] = _read_json(
                    repair_path
                )

    for row in pilot_manifest.get("runs") or ():
        run_id = str(row.get("run_id") or "")
        if not run_id or ":" not in run_id:
            continue
        seed, configuration = run_id.split(":", 1)
        path = pilot / seed / configuration / "coordination_metrics.json"
        if path.is_file():
            coordination[run_id] = _read_json(path)

    report = build_posthoc_descriptive_report(
        readiness_manifest=readiness,
        evidence_bundle=bundle,
        failure_routing_by_binding=failure_routing,
        repair_decisions_by_binding=repair_decisions,
        run_manifests=pilot_manifest.get("runs") or (),
        coordination_metrics_by_run=coordination,
    )
    output = operator / "posthoc_descriptive_report.json"
    atomic_write_json(output, report)
    return report
