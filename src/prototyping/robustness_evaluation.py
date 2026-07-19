"""Controlled B0/B1/B2 aggregation for Option 2 run reports.

This module does not run LLMs or simulators.  It consumes archived run reports
from the same reviewed requirement set and reports raw counts plus
cross-repetition agreement, avoiding claims that natural failures provide
exhaustive coverage.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Iterable, Mapping


EVALUATION_SCHEMA_VERSION = "1.4"


def _posthoc(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = run.get("posthoc_evaluation")
    return value if isinstance(value, Mapping) else None


def _measurement_artifact(
    run: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    """Use the common read-only evaluator when attached; support legacy runs."""
    posthoc = _posthoc(run)
    if posthoc is not None:
        value = posthoc.get(key)
    else:
        value = run.get(key)
    return value if isinstance(value, Mapping) else {}


def measurement_artifact(
    run: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    """Return the uniform post-hoc artifact when present, else legacy data."""
    return _measurement_artifact(run, key)


def requirement_set_fingerprint(run: Mapping[str, Any]) -> str:
    requirements = run.get("requirements") or ()
    normalized = []
    for item in requirements:
        if isinstance(item, Mapping):
            normalized.append((
                str(item.get("id") or item.get("req_id") or ""),
                str(item.get("text") or item.get("requirement_text") or ""),
            ))
        else:
            normalized.append(("", str(item)))
    if not normalized:
        normalized = [
            (str(item.get("req_id", "")), str(item.get("source_digest", "")))
            for item in measurement_artifact(
                run, "requirement_contracts"
            ).get("contracts", ())
        ]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_source_fingerprint(
    run: Mapping[str, Any], req_ids: Iterable[str]
) -> str:
    """Fingerprint only reviewed source anchors, independent of LLM additions."""
    from .contract_types import normalise_req_id, source_digest
    from .requirement_inputs import normalise_requirement_id

    by_id: dict[str, str] = {}
    for item in measurement_artifact(
        run, "requirement_contracts"
    ).get("contracts", ()):
        req_id = normalise_req_id(str(item.get("req_id", "")))
        digest = str(item.get("source_digest", ""))
        if req_id and digest:
            by_id[req_id] = digest
    for item in run.get("requirements") or ():
        if isinstance(item, Mapping):
            req_id = normalise_req_id(str(item.get("id") or item.get("req_id") or ""))
            text = str(item.get("text") or item.get("requirement_text") or "")
        else:
            text = str(item)
            req_id = normalise_requirement_id(text)
        if req_id and text:
            by_id.setdefault(req_id, source_digest(text))
    ordered = [normalise_req_id(str(req_id)) for req_id in req_ids]
    missing = [req_id for req_id in ordered if req_id not in by_id]
    if missing:
        raise ValueError(
            "run is missing reviewed source requirements: "
            + ", ".join(missing)
        )
    payload = json.dumps(
        [(req_id, by_id[req_id]) for req_id in ordered],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frozen_input_valid(run: Mapping[str, Any]) -> bool:
    from .requirement_inputs import requirement_set_digest

    artifact = run.get("requirement_input") or {}
    if artifact.get("mode") != "frozen" or artifact.get("frozen") is not True:
        return False
    requirements = run.get("requirements") or ()
    try:
        actual = requirement_set_digest(str(item) for item in requirements)
    except ValueError:
        return False
    return bool(actual) and actual == artifact.get("requirement_set_digest")


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _pairwise_agreement(fingerprints: list[set[Any]]) -> float | None:
    if len(fingerprints) < 2:
        return None
    return _mean(_jaccard(left, right) for left, right in combinations(fingerprints, 2))


def _precision_recall_f1(
    predicted: set[Any], expected: set[Any]
) -> dict[str, float | int | None]:
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else None
    recall = true_positive / len(expected) if expected else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision == 0 or recall == 0:
        f1 = 0.0
    return {
        "predicted": len(predicted),
        "expected": len(expected),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _atomic_contract_slots(req_id: str, contract: Mapping[str, Any]) -> set[tuple]:
    """Canonicalize obligation slots so omitted default/null fields do not bias F1."""
    slots: set[tuple] = set()
    shapes = {
        "trigger": (
            ("concept", None), ("condition_kind", "event"),
            ("variable", None), ("comparator", None), ("value", None),
            ("unit", None), ("qualifiers", ()),
        ),
        "response": (
            ("concept", None), ("polarity", "required"), ("qualifiers", ()),
        ),
        "criterion": (
            ("metric", None), ("comparator", None), ("value", None),
            ("unit", None), ("timing_semantics", None),
        ),
        "verification_intent": (
            ("method", None), ("observation_concept", None),
            ("preferred_tier", "behavioral"),
        ),
    }
    for obligation in contract.get("obligations", ()):
        obligation_id = str(obligation.get("obligation_id", ""))
        for field in ("kind", "subject"):
            if obligation.get(field) is not None:
                slots.add((req_id, obligation_id, field, str(obligation.get(field))))
        for group, fields in shapes.items():
            value = obligation.get(group)
            if not isinstance(value, Mapping):
                continue
            for field, default in fields:
                item = value.get(field, default)
                if item is None:
                    continue
                normalized = json.dumps(
                    item, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                slots.add((req_id, obligation_id, f"{group}.{field}", normalized))
    return slots


def _contract_fingerprint(run: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    contracts = _measurement_artifact(run, "requirement_contracts").get(
        "contracts", ()
    )
    result: set[tuple[Any, ...]] = set()
    for contract in contracts:
        for obligation in contract.get("obligations", ()):
            trigger = obligation.get("trigger") or {}
            response = obligation.get("response") or {}
            criterion = obligation.get("criterion") or {}
            result.add((
                contract.get("req_id"), obligation.get("obligation_id"),
                contract.get("completeness"), obligation.get("kind"),
                trigger.get("concept"), trigger.get("comparator"), trigger.get("value"),
                response.get("concept"), criterion.get("metric"),
                criterion.get("comparator"), criterion.get("value"), criterion.get("unit"),
            ))
    return result


def _trace_fingerprint(run: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    traces = _measurement_artifact(run, "semantic_trace_report").get(
        "traces", ()
    )
    return {
        (
            trace.get("req_id"), link.get("obligation_id"),
            link.get("link_kind"), link.get("expected_concept"), link.get("status"),
        )
        for trace in traces for link in trace.get("links", ())
        if link.get("link_kind") != "pattern_selection"
    }


def _route_fingerprint(run: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    posthoc = _posthoc(run)
    if posthoc is not None:
        decisions = _measurement_artifact(
            run, "failure_classifications"
        ).get("decisions", ())
    else:
        decisions = (run.get("repair_decisions") or {}).get("decisions", ())
    return {
        (
            item.get("req_id"), item.get("failure_class"),
            item.get("recommended_route", item.get("route")),
        )
        for item in decisions
    }


def _attempt_preserved(item: Mapping[str, Any]) -> bool:
    """True only when every recorded preservation gate actually passed."""
    return bool(
        not item.get("new_diagnostic_ids")
        and not item.get("simulation_regressions")
        and item.get("syntax_passed") is True
        and item.get("score_preserved") is True
    )


def evaluate_run_against_gold(
    run: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    run_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one archived run against independently reviewed Option 2 gold."""
    from .robustness_gold import reviewed_rows, reviewed_run_rows

    rows = reviewed_rows(gold)
    selected_fingerprint = selected_source_fingerprint(run, rows)
    if selected_fingerprint != gold.get("selected_source_fingerprint"):
        raise ValueError("run selected source requirements do not match reviewed gold")
    run_fingerprint = requirement_set_fingerprint(run)
    predicted_contracts = {
        str(item.get("req_id", "")): item
        for item in _measurement_artifact(
            run, "requirement_contracts"
        ).get("contracts", ())
        if str(item.get("req_id", "")) in rows
    }
    for req_id, row in rows.items():
        predicted = predicted_contracts.get(req_id)
        if predicted and predicted.get("source_digest") != row.get("source_digest"):
            raise ValueError(f"{req_id}: source digest does not match reviewed gold")

    completeness_pairs = [
        (
            predicted_contracts.get(req_id, {}).get("completeness"),
            row["reviewed_contract"].get("completeness"),
            req_id in predicted_contracts,
        )
        for req_id, row in rows.items()
    ]
    completeness_correct = sum(
        left == right for left, right, _scored in completeness_pairs
    )
    completeness_by_status: dict[str, dict[str, int]] = {}
    for predicted, expected, scored in completeness_pairs:
        bucket = completeness_by_status.setdefault(
            str(expected), {"expected": 0, "scored": 0, "correct": 0}
        )
        bucket["expected"] += 1
        bucket["scored"] += int(scored)
        bucket["correct"] += int(predicted == expected)

    predicted_slots: set[tuple] = set()
    gold_slots: set[tuple] = set()
    for req_id, contract in predicted_contracts.items():
        predicted_slots |= _atomic_contract_slots(req_id, contract)
    for req_id, row in rows.items():
        gold_slots |= _atomic_contract_slots(req_id, row["reviewed_contract"])
    slot_scores = _precision_recall_f1(predicted_slots, gold_slots)

    predicted_trace = {
        item for item in _trace_fingerprint(run) if str(item[0]) in rows
    }
    run_rows = (
        reviewed_run_rows(run_review, gold, run=run)
        if run_review is not None else None
    )
    gold_trace = None
    if run_rows is not None:
        gold_trace = {
            (
                req_id, link.get("obligation_id"), link.get("link_kind"),
                link.get("expected_concept"), link.get("status"),
            )
            for req_id, row in run_rows.items()
            for link in row.get("reviewed_trace_links", ())
            if link.get("link_kind") != "pattern_selection"
        }
    predicted_patterns = {
        (str(item.get("req_id", "")), str(item.get("pattern_id", "")))
        for item in _measurement_artifact(
            run, "safety_pattern_bindings"
        ).get("bindings", ())
        if str(item.get("req_id", "")) in rows
    }
    gold_patterns = {
        (req_id, str(pattern_id))
        for req_id, row in rows.items()
        for pattern_id in row.get("reviewed_pattern_ids", ())
    }
    if _posthoc(run) is not None:
        measured_route_items = _measurement_artifact(
            run, "failure_classifications"
        ).get("decisions", ())
    else:
        measured_route_items = (
            run.get("repair_decisions") or {}
        ).get("decisions", ())
    predicted_routes = {
        str(item.get("req_id", "")): item
        for item in measured_route_items
        if str(item.get("req_id", "")) in rows
    }
    route_pairs = [] if run_rows is None else [
        (
            predicted_routes.get(req_id, {}).get("failure_class"),
            row.get("reviewed_failure_class"),
        )
        for req_id, row in run_rows.items()
    ]
    routing_cases = [] if run_rows is None else [
        {
            "req_id": req_id,
            "predicted_failure_class": predicted_routes.get(req_id, {}).get(
                "failure_class"
            ),
            "reviewed_failure_class": row.get("reviewed_failure_class"),
            "predicted_rationale_codes": sorted(set(
                predicted_routes.get(req_id, {}).get("rationale_codes", ())
            )),
            "reviewed_rationale_codes": sorted(set(
                row.get("reviewed_rationale_codes", ())
            )),
            "class_match": (
                predicted_routes.get(req_id, {}).get("failure_class")
                == row.get("reviewed_failure_class")
            ),
            "rationale_match": (
                req_id in predicted_routes
                and set(predicted_routes.get(req_id, {}).get("rationale_codes", ()))
                == set(row.get("reviewed_rationale_codes", ()))
            ),
        }
        for req_id, row in run_rows.items()
    ]
    options = run.get("robustness_options")
    intervention_items = (
        () if isinstance(options, Mapping) and not options.get("failure_routing")
        else (run.get("repair_decisions") or {}).get("decisions", ())
    )
    intervention_routes = {
        str(item.get("req_id", "")): item
        for item in intervention_items
        if str(item.get("req_id", "")) in rows
    }
    unsafe = [
        item for item in intervention_routes.values()
        if item.get("failure_class") != "MODEL_SEMANTIC_FAULT"
        and bool(item.get("repair_authorised"))
    ]
    authorised = [
        item for item in intervention_routes.values()
        if item.get("repair_authorised")
    ]
    attempts = list(
        (run.get("repair_decisions") or {}).get("semantic_repair_attempts", ())
    )
    accepted_attempts = [item for item in attempts if item.get("accepted")]
    preserved_attempts = [item for item in attempts if _attempt_preserved(item)]
    return {
        "reviewed_requirement_count": len(rows),
        "selected_source_fingerprint": selected_fingerprint,
        "full_requirement_set_fingerprint": run_fingerprint,
        "full_requirement_set_matches_gold_source_run": (
            run_fingerprint == gold.get("requirement_set_fingerprint")
        ),
        "contract_completeness_scored": sum(
            scored for _left, _right, scored in completeness_pairs
        ),
        "contract_completeness_accuracy": (
            completeness_correct / len(completeness_pairs)
            if completeness_pairs else None
        ),
        "contract_completeness_raw_counts": completeness_by_status,
        "contract_slots": slot_scores,
        "invented_slot_rate": (
            len(predicted_slots - gold_slots) / len(predicted_slots)
            if predicted_slots else None
        ),
        "semantic_trace_links": (
            _precision_recall_f1(predicted_trace, gold_trace)
            if gold_trace is not None else None
        ),
        "pattern_bindings": _precision_recall_f1(predicted_patterns, gold_patterns),
        "routing_expected": len(route_pairs),
        "routing_scored": sum(
            req_id in predicted_routes for req_id in (run_rows or {})
        ),
        "routing_accuracy": (
            sum(left == right for left, right in route_pairs) / len(route_pairs)
            if route_pairs else None
        ),
        "routing_rationale_accuracy": (
            sum(item["rationale_match"] for item in routing_cases)
            / len(routing_cases)
            if routing_cases else None
        ),
        "routing_cases": routing_cases,
        "run_review_status": (
            "BLIND_PER_RUN_REVIEWED" if run_rows is not None else "NOT_PROVIDED"
        ),
        "unsafe_repair_count": len(unsafe),
        "unsafe_repair_rate": (
            len(unsafe) / len(authorised) if authorised else 0.0
        ),
        "repair_attempt_count": len(attempts),
        "repair_success": (
            len(accepted_attempts) / len(attempts) if attempts else None
        ),
        "regression_preservation": (
            len(preserved_attempts) / len(attempts) if attempts else None
        ),
        "repair_metric_note": (
            "Metrics are null when no recorded semantic repair attempts exist; "
            "terminal PASS is never back-interpreted as repair success."
        ),
    }


def _configuration_summary(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    trace_counts = [
        _measurement_artifact(run, "semantic_trace_report").get("counts", {})
        for run in runs
    ]
    diagnostics = []
    for run in runs:
        artifact = _measurement_artifact(run, "failure_diagnostics")
        # Keep archived pre-envelope reports readable.
        if isinstance(artifact, Mapping):
            artifact = artifact.get("diagnostics", ())
        diagnostics.extend(artifact)
    intervention_decisions = []
    intervention_attempts = []
    for run in runs:
        options = run.get("robustness_options")
        # Older B1 bundles accidentally emitted terminal classifications even
        # though failure routing was disabled.  Do not reinterpret those
        # observer artifacts as an intervention.  Truly legacy fixtures/runs
        # without recorded options keep their historical behavior.
        if isinstance(options, Mapping) and not options.get("failure_routing"):
            continue
        intervention_decisions.extend(
            (run.get("repair_decisions") or {}).get("decisions", ())
        )
        intervention_attempts.extend(
            (run.get("repair_decisions") or {}).get(
                "semantic_repair_attempts", ()
            )
        )
    measurement_decisions = []
    for run in runs:
        if _posthoc(run) is not None:
            measurement_decisions.extend(
                _measurement_artifact(
                    run, "failure_classifications"
                ).get("decisions", ())
            )
        else:
            measurement_decisions.extend(
                (run.get("repair_decisions") or {}).get("decisions", ())
            )
    total_supported = sum(
        counts.get("PASS", 0) + counts.get("FAIL", 0) + counts.get("BLOCKED", 0)
        for counts in trace_counts
    )
    passed = sum(counts.get("PASS", 0) for counts in trace_counts)
    controlled_scenario_counts = [
        _measurement_artifact(
            run, "controlled_scenario_evaluation"
        ).get("counts", {})
        for run in runs
    ]
    controlled_scenario_total = sum(
        counts.get("PASS", 0) + counts.get("FAIL", 0)
        for counts in controlled_scenario_counts
    )
    controlled_scenario_passed = sum(
        counts.get("PASS", 0) for counts in controlled_scenario_counts
    )
    llm_calls = [
        float((run.get("llm_usage") or {}).get("calls"))
        for run in runs if (run.get("llm_usage") or {}).get("calls") is not None
    ]
    total_tokens = []
    for run in runs:
        usage = run.get("llm_usage") or {}
        value = usage.get("total_tokens")
        if value is None and (
            usage.get("input_tokens") is not None
            or usage.get("output_tokens") is not None
        ):
            value = float(usage.get("input_tokens") or 0) + float(
                usage.get("output_tokens") or 0
            )
        if value is not None:
            total_tokens.append(float(value))
    elapsed = [
        float(run["elapsed_s"]) for run in runs if run.get("elapsed_s") is not None
    ]
    return {
        "run_count": len(runs),
        "supported_trace_count": total_supported,
        "semantic_trace_pass_count": passed,
        "semantic_trace_pass_rate": passed / total_supported if total_supported else None,
        "controlled_scenario_count": controlled_scenario_total,
        "controlled_scenario_pass_count": controlled_scenario_passed,
        "controlled_scenario_pass_rate": (
            controlled_scenario_passed / controlled_scenario_total
            if controlled_scenario_total else None
        ),
        "diagnostic_count": len(diagnostics),
        "diagnostic_counts": _counts(
            item.get("finding_code", "UNKNOWN") for item in diagnostics
        ),
        "repair_authorised_count": sum(
            bool(item.get("repair_authorised"))
            for item in intervention_decisions
        ),
        "repair_attempt_count": len(intervention_attempts),
        "repair_accepted_count": sum(
            bool(item.get("accepted")) for item in intervention_attempts
        ),
        "repair_success": (
            sum(bool(item.get("accepted")) for item in intervention_attempts)
            / len(intervention_attempts)
            if intervention_attempts else None
        ),
        "regression_preserved_count": sum(
            _attempt_preserved(item) for item in intervention_attempts
        ),
        "regression_preservation": (
            sum(_attempt_preserved(item) for item in intervention_attempts)
            / len(intervention_attempts)
            if intervention_attempts else None
        ),
        "route_counts": _counts(
            item.get("failure_class", "UNKNOWN")
            for item in measurement_decisions
        ),
        "measurement_source": (
            "UNIFORM_POSTHOC_MEASUREMENT"
            if runs and all(_posthoc(run) is not None for run in runs)
            else "LEGACY_RUN_ARTIFACTS"
        ),
        "cross_repetition_contract_agreement": _pairwise_agreement([
            _contract_fingerprint(run) for run in runs
        ]),
        "cross_repetition_trace_agreement": _pairwise_agreement([
            _trace_fingerprint(run) for run in runs
        ]),
        "cross_repetition_route_agreement": _pairwise_agreement([
            _route_fingerprint(run) for run in runs
        ]),
        "mean_llm_calls": _mean(llm_calls),
        "mean_total_tokens": _mean(total_tokens),
        "mean_elapsed_s": _mean(elapsed),
    }


def _counts(items: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        result[item] = result.get(item, 0) + 1
    return result


def evaluate_configurations(
    runs_by_configuration: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    gold: Mapping[str, Any] | None = None,
    run_reviews_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    require_frozen_inputs: bool = True,
    require_complete_design: bool = False,
    require_uniform_posthoc: bool = False,
) -> dict[str, Any]:
    """Aggregate B0/B1/B2 reports from an identical reviewed requirement set."""
    allowed = {"B0", "B1", "B2"}
    unknown = set(runs_by_configuration) - allowed
    if unknown:
        raise ValueError(f"unknown robustness configurations: {sorted(unknown)}")
    materialized = {
        name: list(runs) for name, runs in runs_by_configuration.items()
    }
    for name, runs in materialized.items():
        for run in runs:
            recorded = run.get("configuration") or (
                run.get("pilot_metadata") or {}
            ).get("configuration")
            if recorded and str(recorded).upper() != name:
                raise ValueError(
                    f"run recorded as {recorded!r} was supplied under {name}"
                )
    archive_dirs = [
        str(run.get("_archive_run_dir"))
        for runs in materialized.values() for run in runs
        if run.get("_archive_run_dir")
    ]
    if len(archive_dirs) != len(set(archive_dirs)):
        raise ValueError("the same archived run was supplied more than once")
    fingerprints = {
        requirement_set_fingerprint(run)
        for runs in materialized.values() for run in runs
    }
    all_runs = [run for runs in materialized.values() for run in runs]
    posthoc_presence = [_posthoc(run) is not None for run in all_runs]
    if any(posthoc_presence) and not all(posthoc_presence):
        raise ValueError(
            "controlled comparison cannot mix uniform post-hoc measurements "
            "with intervention/legacy run artifacts"
        )
    posthoc_digests = {
        str((_posthoc(run) or {}).get("requirement_set_digest"))
        for run in all_runs if _posthoc(run) is not None
    }
    posthoc_inputs_match = all(
        str((_posthoc(run) or {}).get("requirement_set_digest"))
        == str((run.get("requirement_input") or {}).get(
            "requirement_set_digest"
        ))
        for run in all_runs if _posthoc(run) is not None
    )
    posthoc_models_match = all(
        run.get("posthoc_model_digest_verified") is True
        and str((_posthoc(run) or {}).get("model_digest"))
        == str(_measurement_artifact(run, "semantic_trace_report").get(
            "model_digest"
        ))
        for run in all_runs if _posthoc(run) is not None
    )
    frozen_valid = [_frozen_input_valid(run) for run in all_runs]
    input_digests = {
        str((run.get("requirement_input") or {}).get("requirement_set_digest"))
        for run in all_runs
        if (run.get("requirement_input") or {}).get("requirement_set_digest")
    }
    controlled_inputs = (
        bool(all_runs)
        and all(frozen_valid)
        and len(input_digests) == 1
        and len(fingerprints) == 1
    )
    if require_frozen_inputs and not controlled_inputs:
        raise ValueError(
            "controlled evaluation requires every run to use the same verified "
            "frozen requirement artifact"
        )
    run_counts = {name: len(materialized.get(name, ())) for name in sorted(allowed)}
    complete_configurations = all(run_counts[name] > 0 for name in allowed)
    balanced_repetitions = (
        complete_configurations and len(set(run_counts.values())) == 1
    )
    repetition_maps: dict[str, dict[int, int | None]] = {}
    generation_seed_maps: dict[str, dict[int, int | None]] = {}
    generation_seed_controls: set[str] = set()
    repetitions_recorded = True
    for name in allowed:
        per_rep: dict[int, int | None] = {}
        per_rep_generation: dict[int, int | None] = {}
        for run in materialized.get(name, ()):
            metadata = run.get("pilot_metadata") or {}
            repetition = run.get("repetition", metadata.get("repetition"))
            seed = run.get("mcts_seed", metadata.get("mcts_seed"))
            generation_seed = run.get(
                "generation_seed", metadata.get("generation_seed")
            )
            seed_control = run.get(
                "generation_seed_control",
                metadata.get("generation_seed_control"),
            )
            if seed_control:
                generation_seed_controls.add(str(seed_control))
            if repetition is None:
                repetitions_recorded = False
                continue
            rep = int(repetition)
            if rep in per_rep:
                raise ValueError(
                    f"{name} contains duplicate repetition identifier {rep}"
                )
            per_rep[rep] = int(seed) if seed is not None else None
            per_rep_generation[rep] = (
                int(generation_seed) if generation_seed is not None else None
            )
        repetition_maps[name] = per_rep
        generation_seed_maps[name] = per_rep_generation
    paired_repetitions = (
        complete_configurations
        and repetitions_recorded
        and len({tuple(sorted(value)) for value in repetition_maps.values()}) == 1
    )
    paired_mcts_seeds = paired_repetitions and all(
        repetition_maps["B0"].get(rep) is not None
        and len({repetition_maps[name].get(rep) for name in allowed}) == 1
        for rep in repetition_maps["B0"]
    )
    paired_generation_seeds = paired_repetitions and all(
        generation_seed_maps["B0"].get(rep) is not None
        and len({generation_seed_maps[name].get(rep) for name in allowed}) == 1
        for rep in repetition_maps["B0"]
    )
    design_complete = bool(
        complete_configurations
        and balanced_repetitions
        and paired_repetitions
        and paired_mcts_seeds
        and paired_generation_seeds
    )
    if require_complete_design and not design_complete:
        raise ValueError(
            "controlled B0/B1/B2 evaluation requires every configuration, "
            "equal repetition counts, unique paired repetition ids, and the "
            "same recorded LLM and MCTS seeds within each repetition"
        )
    measurement_stack_signatures = {
        (
            str(posthoc.get("schema_version") or ""),
            str(posthoc.get("contract_library_version") or ""),
            str(posthoc.get("pattern_library_version") or ""),
            str(posthoc.get("platform_binding_version") or ""),
            str((posthoc.get("semantic_trace_report") or {}).get(
                "schema_version"
            ) or ""),
            str((posthoc.get("controlled_scenario_evaluation") or {}).get(
                "schema_version"
            ) or ""),
        )
        for run in all_runs
        for posthoc in [_posthoc(run)]
        if posthoc is not None
    }
    measurement_roles_valid = all(
        (_posthoc(run) or {}).get("artifact_role")
        == "UNIFORM_POSTHOC_MEASUREMENT"
        and (_posthoc(run) or {}).get("measurement_only") is True
        and (_posthoc(run) or {}).get("mutation_permitted") is False
        for run in all_runs if _posthoc(run) is not None
    )
    measurement_stack_consistent = bool(
        len(measurement_stack_signatures) == 1
        and all(
            value
            for signature in measurement_stack_signatures
            for value in signature
        )
    )
    posthoc_control_verified = bool(
        all_runs
        and all(posthoc_presence)
        and len(posthoc_digests) == 1
        and posthoc_inputs_match
        and posthoc_models_match
        and measurement_roles_valid
        and measurement_stack_consistent
    )
    if require_uniform_posthoc and not posthoc_control_verified:
        raise ValueError(
            "controlled comparison requires a verified uniform post-hoc "
            "measurement sidecar from the same complete versioned stack for "
            "every run"
        )
    summaries = {
        name: _configuration_summary(runs)
        for name, runs in materialized.items()
    }
    if gold is not None:
        from .robustness_gold import run_review_id

        for name, runs in materialized.items():
            scored = []
            for run in runs:
                review = None
                if run_reviews_by_id is not None:
                    key = run_review_id(run)
                    if key not in run_reviews_by_id:
                        raise ValueError(f"missing blind per-run review for {key}")
                    review = run_reviews_by_id[key]
                scored.append(
                    evaluate_run_against_gold(run, gold, run_review=review)
                )
            summaries[name]["gold_runs"] = scored
            scalar_fields = (
                "contract_completeness_accuracy", "invented_slot_rate",
                "routing_accuracy", "routing_rationale_accuracy",
                "unsafe_repair_rate", "repair_success",
                "regression_preservation",
            )
            summaries[name]["gold_metric_means"] = {
                field: _mean(
                    item[field] for item in scored if item.get(field) is not None
                )
                for field in scalar_fields
            }
            summaries[name]["gold_metric_means"].update({
                "contract_slot_f1": _mean(
                    item["contract_slots"]["f1"] for item in scored
                    if item["contract_slots"]["f1"] is not None
                ),
                "semantic_trace_f1": _mean(
                    item["semantic_trace_links"]["f1"] for item in scored
                    if item.get("semantic_trace_links") is not None
                    and item["semantic_trace_links"]["f1"] is not None
                ),
                "pattern_binding_f1": _mean(
                    item["pattern_bindings"]["f1"] for item in scored
                    if item["pattern_bindings"]["f1"] is not None
                ),
            })
    elif run_reviews_by_id:
        raise ValueError("blind per-run reviews require reviewed source gold")
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "requirement_set_fingerprint": next(iter(fingerprints), None),
        "requirement_set_consistent": len(fingerprints) <= 1,
        "requirement_input_control": {
            "required": require_frozen_inputs,
            "verified": controlled_inputs,
            "modes": sorted(set(
                str((run.get("requirement_input") or {}).get("mode") or "UNRECORDED")
                for run in all_runs
            )),
            "digests": sorted(input_digests),
            "full_set_variants": len(fingerprints),
        },
        "experimental_design_control": {
            "required": require_complete_design,
            "complete_configurations": complete_configurations,
            "balanced_repetitions": balanced_repetitions,
            "repetitions_recorded": repetitions_recorded,
            "paired_repetitions": paired_repetitions,
            "paired_mcts_seeds": paired_mcts_seeds,
            "paired_generation_seeds": paired_generation_seeds,
            "generation_seed_controls": sorted(generation_seed_controls),
            "run_counts": run_counts,
            "valid": design_complete,
        },
        "posthoc_measurement_control": {
            "required": require_uniform_posthoc,
            "verified": posthoc_control_verified,
            "attached_runs": sum(posthoc_presence),
            "total_runs": len(all_runs),
            "requirement_set_digests": sorted(posthoc_digests),
            "requirement_digest_links_valid": posthoc_inputs_match,
            "model_digest_links_valid": posthoc_models_match,
            "artifact_roles_valid": measurement_roles_valid,
            "measurement_stack_consistent": measurement_stack_consistent,
            "measurement_stack_signatures": [
                list(signature)
                for signature in sorted(measurement_stack_signatures)
            ],
            "role": (
                "UNIFORM_POSTHOC_MEASUREMENT"
                if all_runs and all(posthoc_presence) else "LEGACY"
            ),
        },
        "controlled_comparison_valid": bool(
            controlled_inputs and design_complete and posthoc_control_verified
        ),
        "configurations": summaries,
        "claim_boundary": (
            "Natural/archived failures and independent stochastic repetitions "
            "measure observed cases; they do not establish exhaustive "
            "fault-detection coverage or absolute LLM determinism."
        ),
    }
