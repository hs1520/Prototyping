"""Independent reviewed-gold support for the Option 2 controlled evaluation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Mapping

from .contract_types import INCOMPLETE, READY, UNSUPPORTED
from .failure_routing import (
    DESIGN_OR_REALIZATION_NONCOMPLIANCE,
    INFRASTRUCTURE_FAILURE,
    MODEL_SEMANTIC_FAULT,
    NO_FAILURE,
    REQUIREMENT_AMBIGUITY,
    VERIFIER_LIMITATION,
)
from .robustness_evaluation import (
    measurement_artifact,
    requirement_set_fingerprint,
    selected_source_fingerprint,
)


GOLD_SCHEMA_VERSION = "1.2"
RUN_REVIEW_SCHEMA_VERSION = "1.1"
OPTION2_GOLD_REQUIREMENT_IDS = (
    "REQ_FUNC_002", "REQ_FUNC_005", "REQ_FUNC_006", "REQ_PERF_005",
    "REQ_SAFE_001", "REQ_SAFE_004", "REQ_SAFE_005", "REQ_SAFE_006",
    "REQ_SAFE_007", "REQ_SAFE_008", "REQ_FUNC_007", "REQ_PERF_001",
    "REQ_PERF_003", "REQ_INTF_002", "REQ_CONS_004",
)
_CONTRACT_STATUSES = {READY, INCOMPLETE, UNSUPPORTED}
_FAILURE_CLASSES = {
    REQUIREMENT_AMBIGUITY,
    MODEL_SEMANTIC_FAULT,
    DESIGN_OR_REALIZATION_NONCOMPLIANCE,
    VERIFIER_LIMITATION,
    INFRASTRUCTURE_FAILURE,
    NO_FAILURE,
}
ROUTING_POLICY = {
    "version": "option2-mvp-1",
    "tie_break_order": [
        "CONTRACT_INCOMPLETE",
        "CONTRACT_UNSUPPORTED",
        "MODEL_SEMANTIC_FINDING",
        "INFRASTRUCTURE_FAILURE",
        "VERIFIER_UNAVAILABLE",
        "ENGINEERING_ORACLE_FAILED",
        "NO_FAILURE",
    ],
    "rationale_distinctions": {
        "CONTRACT_UNSUPPORTED": "unsupported contract-family coverage",
        "VERIFIER_UNAVAILABLE": "required observer or executable check unavailable",
    },
}


def run_review_id(run: Mapping[str, Any]) -> str:
    """Return a stable identifier for binding blind labels to exactly one run."""
    provenance = run.get("artifact_provenance") or {}
    explicit = provenance.get("run_id") or run.get("run_id")
    if explicit:
        return str(explicit)
    model_digest = measurement_artifact(
        run, "semantic_trace_report"
    ).get("model_digest")
    if model_digest:
        return f"model:{model_digest}"
    payload = json.dumps(
        run, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_gold_template(
    run: Mapping[str, Any],
    *,
    include_pipeline_candidates: bool = False,
    selected_req_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create source-first static gold; candidate predictions are opt-in only."""
    contracts = {
        str(item.get("req_id", "")): item
        for item in measurement_artifact(
            run, "requirement_contracts"
        ).get("contracts", ())
    }
    traces = {
        str(item.get("req_id", "")): item
        for item in measurement_artifact(
            run, "semantic_trace_report"
        ).get("traces", ())
    }
    bindings: dict[str, list[str]] = {}
    for item in measurement_artifact(
        run, "safety_pattern_bindings"
    ).get("bindings", ()):
        bindings.setdefault(str(item.get("req_id", "")), []).append(
            str(item.get("pattern_id", ""))
        )
    routes = {
        str(item.get("req_id", "")): item
        for item in measurement_artifact(
            run, "failure_classifications"
        ).get("decisions", ())
    }
    if not routes:
        routes = {
            str(item.get("req_id", "")): item
            for item in (run.get("repair_decisions") or {}).get(
                "decisions", ()
            )
        }
    selected_order = (
        list(dict.fromkeys(
            str(req_id).upper().replace("-", "_")
            for req_id in selected_req_ids
        ))
        if selected_req_ids is not None else None
    )
    selected = set(selected_order) if selected_order is not None else None
    rows = []
    row_order = selected_order if selected_order is not None else list(contracts)
    for req_id in row_order:
        contract = contracts.get(req_id)
        if contract is None:
            continue
        trace = traces.get(req_id, {})
        row = {
            "req_id": req_id,
            "source_text": contract.get("source_text", ""),
            "source_digest": contract.get("source_digest", ""),
            "review_status": "PENDING",
            "review_notes": "",
            "reviewed_contract": {
                "completeness": None,
                "obligations": [],
            },
            "reviewed_trace_links": [],
            "reviewed_pattern_ids": [],
        }
        if include_pipeline_candidates:
            row["pipeline_candidate"] = {
                "contract": deepcopy(contract),
                "trace_links": deepcopy(trace.get("links", [])),
                "pattern_ids": sorted(set(bindings.get(req_id, ()))),
                "failure_class": routes.get(req_id, {}).get("failure_class"),
            }
        rows.append(row)
    if selected is not None:
        missing = selected - set(contracts)
        if missing:
            raise ValueError(
                "selected requirements are absent from run contracts: "
                + ", ".join(sorted(missing))
            )
    selected_ids = [row["req_id"] for row in rows]
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "review_mode": "SOURCE_FIRST",
        "selected_requirement_ids": selected_ids,
        "selected_source_fingerprint": selected_source_fingerprint(
            run, selected_ids
        ),
        "requirement_set_fingerprint": requirement_set_fingerprint(run),
        "routing_policy": deepcopy(ROUTING_POLICY),
        "reviewer": "",
        "reviewed_at": "",
        "review_protocol": (
            "Review source_text independently. Freeze contract fields, expected "
            "trace-link structure, and pattern ids; do not assign run-dependent "
            "trace statuses or failure classes here. If pipeline_candidate is "
            "included for a documented second pass, it is never gold authority."
        ),
        "requirements": rows,
    }


def prepare_run_review_template(
    run: Mapping[str, Any], gold: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a pipeline-blind, per-run worksheet for trace and route labels."""
    rows = reviewed_rows(gold)
    selected_fingerprint = selected_source_fingerprint(run, rows)
    if selected_fingerprint != gold.get("selected_source_fingerprint"):
        raise ValueError(
            "run selected source requirements do not match reviewed source gold"
        )
    contracts = {
        str(item.get("req_id", "")): item
        for item in measurement_artifact(
            run, "requirement_contracts"
        ).get("contracts", ())
    }
    review_rows = []
    for req_id, row in rows.items():
        predicted = contracts.get(req_id)
        if predicted and predicted.get("source_digest") != row.get("source_digest"):
            raise ValueError(f"{req_id}: source digest does not match reviewed gold")
        review_rows.append({
            "req_id": req_id,
            "source_text": row.get("source_text", ""),
            "source_digest": row.get("source_digest", ""),
            "review_status": "PENDING",
            "review_notes": "",
            "reviewed_trace_links": [
                {
                    "obligation_id": link.get("obligation_id"),
                    "link_kind": link.get("link_kind"),
                    "expected_concept": link.get("expected_concept"),
                    "status": None,
                }
                for link in row.get("reviewed_trace_links", ())
            ],
            "reviewed_failure_class": None,
            "reviewed_rationale_codes": [],
        })
    trace_report = measurement_artifact(run, "semantic_trace_report")
    return {
        "schema_version": RUN_REVIEW_SCHEMA_VERSION,
        "review_mode": "BLIND_PER_RUN",
        "run_id": run_review_id(run),
        "model_digest": trace_report.get("model_digest"),
        "requirement_set_fingerprint": requirement_set_fingerprint(run),
        "selected_source_fingerprint": selected_fingerprint,
        "reviewer": "",
        "reviewed_at": "",
        "review_protocol": (
            "Inspect source text, the generated model, and raw execution evidence. "
            "Assign trace status and failure class without consulting pipeline "
            "trace verdicts, diagnostics, repair decisions, or routed classes."
        ),
        "requirements": review_rows,
    }


def validate_gold_dataset(
    gold: Mapping[str, Any], *, require_complete: bool = True
) -> list[str]:
    """Return deterministic schema/review errors; an empty list means usable gold."""
    errors: list[str] = []
    if gold.get("schema_version") != GOLD_SCHEMA_VERSION:
        errors.append(f"unsupported gold schema_version: {gold.get('schema_version')!r}")
    if not gold.get("requirement_set_fingerprint"):
        errors.append("missing requirement_set_fingerprint")
    if not gold.get("selected_source_fingerprint"):
        errors.append("missing selected_source_fingerprint")
    if gold.get("routing_policy") != ROUTING_POLICY:
        errors.append("missing or modified frozen routing_policy")
    if require_complete and not gold.get("reviewer"):
        errors.append("missing source-gold reviewer")
    if require_complete and not gold.get("reviewed_at"):
        errors.append("missing source-gold reviewed_at")
    seen: set[str] = set()
    for index, row in enumerate(gold.get("requirements", ())):
        req_id = str(row.get("req_id", ""))
        label = req_id or f"row[{index}]"
        if not req_id:
            errors.append(f"{label}: missing req_id")
        elif req_id in seen:
            errors.append(f"{label}: duplicate req_id")
        seen.add(req_id)
        if not row.get("source_digest"):
            errors.append(f"{label}: missing source_digest")
        reviewed = row.get("review_status") == "REVIEWED"
        if require_complete and not reviewed:
            errors.append(f"{label}: review_status is not REVIEWED")
            continue
        if not reviewed:
            continue
        contract = row.get("reviewed_contract") or {}
        completeness = contract.get("completeness")
        if completeness not in _CONTRACT_STATUSES:
            errors.append(f"{label}: invalid reviewed completeness {completeness!r}")
        if completeness == READY and not contract.get("obligations"):
            errors.append(f"{label}: READY review requires at least one obligation")
        for link_index, link in enumerate(row.get("reviewed_trace_links", ())):
            for field in ("obligation_id", "link_kind", "expected_concept"):
                if not link.get(field):
                    errors.append(
                        f"{label}: reviewed_trace_links[{link_index}] missing {field}"
                    )
            if link.get("status") is not None:
                errors.append(
                    f"{label}: static gold trace link must not contain run status"
                )
    declared = [str(item) for item in gold.get("selected_requirement_ids", ())]
    if declared and declared != [
        str(row.get("req_id", "")) for row in gold.get("requirements", ())
    ]:
        errors.append("selected_requirement_ids do not match worksheet rows/order")
    return errors


def reviewed_rows(gold: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    errors = validate_gold_dataset(gold, require_complete=True)
    if errors:
        raise ValueError("invalid/incomplete gold dataset: " + "; ".join(errors))
    return {str(row["req_id"]): row for row in gold.get("requirements", ())}


def validate_run_review_dataset(
    review: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    run: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> list[str]:
    """Validate blind, run-dependent trace and routing labels."""
    errors: list[str] = []
    if review.get("schema_version") != RUN_REVIEW_SCHEMA_VERSION:
        errors.append(
            f"unsupported run-review schema_version: {review.get('schema_version')!r}"
        )
    if review.get("review_mode") != "BLIND_PER_RUN":
        errors.append("run review mode must be BLIND_PER_RUN")
    if require_complete and not review.get("reviewer"):
        errors.append("missing run-review reviewer")
    if require_complete and not review.get("reviewed_at"):
        errors.append("missing run-review reviewed_at")
    if review.get("selected_source_fingerprint") != gold.get(
        "selected_source_fingerprint"
    ):
        errors.append("run review selected source does not match source gold")
    gold_rows = reviewed_rows(gold)
    if run is not None:
        if review.get("run_id") != run_review_id(run):
            errors.append("run review is bound to a different run_id")
        model_digest = measurement_artifact(
            run, "semantic_trace_report"
        ).get("model_digest")
        if review.get("model_digest") != model_digest:
            errors.append("run review is bound to a different model_digest")
        try:
            selected = selected_source_fingerprint(run, gold_rows)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if review.get("selected_source_fingerprint") != selected:
                errors.append("run review selected source fingerprint mismatch")

    review_rows = {
        str(row.get("req_id", "")): row for row in review.get("requirements", ())
    }
    if set(review_rows) != set(gold_rows):
        errors.append("run review requirement ids do not match source gold")
    for req_id, gold_row in gold_rows.items():
        row = review_rows.get(req_id, {})
        if row.get("source_digest") != gold_row.get("source_digest"):
            errors.append(f"{req_id}: source digest does not match source gold")
        reviewed = row.get("review_status") == "REVIEWED"
        if require_complete and not reviewed:
            errors.append(f"{req_id}: review_status is not REVIEWED")
            continue
        if not reviewed:
            continue
        failure_class = row.get("reviewed_failure_class")
        if failure_class not in _FAILURE_CLASSES:
            errors.append(
                f"{req_id}: invalid reviewed failure class {failure_class!r}"
            )
        if (
            failure_class in _FAILURE_CLASSES - {NO_FAILURE}
            and not row.get("reviewed_rationale_codes")
        ):
            errors.append(
                f"{req_id}: non-NO_FAILURE label requires reviewed_rationale_codes"
            )
        expected = {
            (
                link.get("obligation_id"),
                link.get("link_kind"),
                link.get("expected_concept"),
            )
            for link in gold_row.get("reviewed_trace_links", ())
        }
        observed = set()
        for index, link in enumerate(row.get("reviewed_trace_links", ())):
            key = (
                link.get("obligation_id"),
                link.get("link_kind"),
                link.get("expected_concept"),
            )
            observed.add(key)
            if link.get("status") not in {"PASS", "FAIL"}:
                errors.append(
                    f"{req_id}: reviewed_trace_links[{index}] has invalid status "
                    f"{link.get('status')!r}"
                )
        if observed != expected:
            errors.append(f"{req_id}: trace-link structure differs from source gold")
    return errors


def reviewed_run_rows(
    review: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    run: Mapping[str, Any] | None = None,
) -> dict[str, Mapping[str, Any]]:
    errors = validate_run_review_dataset(review, gold, run=run)
    if errors:
        raise ValueError("invalid/incomplete run review: " + "; ".join(errors))
    return {
        str(row["req_id"]): row for row in review.get("requirements", ())
    }
