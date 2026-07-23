"""Independent post-hoc gold evaluator for the R2-BBAG A/G intervention.

Separation boundary (design §13 "Claim discipline"/"Dataset and gold boundary",
§16; source-first review finding F3): this evaluator scores an ARCHIVED run
prediction — the ``ag_contract_graph.json`` a checker already emitted — against
EVALUATOR-ONLY human gold. It is a different boundary from the runtime checker:

  * it never invokes the runtime extractor/checker (a checker must not score its
    own output), so this module imports neither ``ag_extractor`` nor
    ``ag_contracts`` — enforced by a test that greps this file;
  * it refuses inputs whose ``artifact_role`` is wrong, so evaluator gold can
    never be mistaken for a pipeline input and a prediction can never be scored
    against another export of the same checker;
  * gold is authored blind to the pipeline verdict (a human process); this module
    only computes agreement between that gold and the archived prediction.

The number produced here is a **decomposition/extraction agreement** F1 (over
guarantee allocation + assumption discharge), **not** an LLM-accuracy score: under
deterministic A/G emission the prediction is the reviewed decomposition rendered
and read back, so agreement is ~1.0 by construction and measures extract/check
faithfulness (it becomes a generation-accuracy metric only when R2 emits
LLM-authored A/G models under a separately frozen intervention). It may be pooled
for R2-BBAG only when a self-consistent
``POSTHOC_EVALUATION_READINESS_MANIFEST`` binds every selected chain/run and all
independent evidence. The global arm enum is deliberately not used as this gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

PREDICTION_ROLE = "RUNTIME_A_G_PREDICTION"
GOLD_ROLE = "EVALUATOR_GOLD"
REVISED_NAMESPACE = "BLACKBOARD_AG_V1"
R2_CONFIGURATION = "R2-BBAG"


@dataclass(frozen=True)
class PRF:
    """Set-based precision/recall/F1 counts."""
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        if denom:
            return self.tp / denom
        return 1.0 if self.fn == 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        if denom:
            return self.tp / denom
        return 1.0 if self.fp == 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _prf(predicted: set, gold: set) -> PRF:
    return PRF(
        tp=len(predicted & gold),
        fp=len(predicted - gold),
        fn=len(gold - predicted),
    )


def _tok(value: Any, aliases: Mapping[str, str]) -> str:
    key = str(value if value is not None else "").strip().lower()
    return aliases.get(key, key)


def _allocation_set(items, aliases):
    return {
        (_tok(i.get("owner"), aliases), _tok(i.get("guarantee"), aliases))
        for i in (items or [])
    }


def _discharge_set(items, aliases, *, claimed_only: bool):
    out = set()
    for i in (items or []):
        by = i.get("by")
        if claimed_only and by is None:
            # An undischarged prediction makes no discharge claim, so it is not a
            # predicted edge; it surfaces as a false negative against gold.
            continue
        out.add((
            _tok(i.get("component"), aliases),
            _tok(i.get("assumption"), aliases),
            _tok(by if by is not None else "undischarged", aliases),
        ))
    return out


def evaluate_ag_against_gold(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Score an archived A/G prediction against evaluator-only human gold.

    Returns per-category precision/recall/F1 for guarantee allocation and
    assumption discharge. Per-run failure classification is evaluated only from
    separately frozen blind labels, never from static reference gold. Raises on a
    role mismatch so gold and predictions cannot be swapped or self-scored.
    """
    if not isinstance(prediction, Mapping) or not isinstance(gold, Mapping):
        raise TypeError("prediction and gold must be mappings")
    if prediction.get("artifact_role") != PREDICTION_ROLE:
        raise ValueError(
            f"prediction artifact_role must be {PREDICTION_ROLE!r}; the evaluator "
            "scores an archived checker prediction, never gold or a live checker call"
        )
    if gold.get("artifact_role") != GOLD_ROLE:
        raise ValueError(
            f"gold artifact_role must be {GOLD_ROLE!r}; evaluator gold is "
            "evaluator-only and must never be a pipeline artifact"
        )
    if (
        prediction.get("experiment_namespace") != REVISED_NAMESPACE
        or prediction.get("configuration") != R2_CONFIGURATION
    ):
        raise ValueError(
            "prediction must be a BLACKBOARD_AG_V1:R2-BBAG artifact; legacy "
            "and revised evaluators/results cannot be mixed"
        )
    if gold.get("experiment_namespace") != REVISED_NAMESPACE:
        raise ValueError("gold must be scoped to BLACKBOARD_AG_V1")
    if gold.get("status") != "FROZEN":
        raise ValueError("accuracy/F1 requires supervisor-reviewed FROZEN gold")
    if "failure_class" in gold:
        raise ValueError(
            "reference gold must not contain failure_class; use a per-run blind label"
        )
    review = gold.get("review_protocol") or {}
    if (
        not gold.get("reviewer")
        or not gold.get("reviewed_date")
        or review.get("blind_to_runtime_verdict") is not True
        or review.get("independent_human_review") is not True
    ):
        raise ValueError(
            "accuracy/F1 requires documented independent blind human review"
        )
    alias_map = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in (aliases or {}).items()
    }

    pred_graph = prediction.get("graph", {}) or {}
    allocation = _prf(
        _allocation_set(pred_graph.get("allocations"), alias_map),
        _allocation_set(gold.get("allocations"), alias_map),
    )
    discharge = _prf(
        _discharge_set(pred_graph.get("discharge_edges"), alias_map, claimed_only=True),
        _discharge_set(gold.get("discharge_edges"), alias_map, claimed_only=False),
    )

    result: Dict[str, Any] = {
        "artifact_role": "POSTHOC_HUMAN_GOLD_EVALUATION",
        "producing_stage": "POSTHOC_EVALUATION",
        "measurement_boundary": "EVALUATOR_ONLY",
        "metric_name": "decomposition_extraction_agreement",
        "metric_interpretation": (
            "set F1 over guarantee allocation + assumption discharge. This is "
            "decomposition/extraction AGREEMENT, NOT LLM accuracy: under "
            "deterministic A/G emission the prediction is the reviewed "
            "decomposition rendered and read back, so agreement is ~1.0 by "
            "construction and measures extract/check faithfulness. It measures "
            "model-generation accuracy only when R2 emits LLM-authored A/G models. "
            "Timing, priority, and invariant agreement are reported as SEPARATE "
            "categories when the gold and prediction carry them (§6); they are never "
            "merged with allocation/discharge or with each other into a composite F1."
        ),
        "chain_id": gold.get("chain_id"),
        "checker_version": prediction.get("checker_version"),
        "source_model_revision": prediction.get("source_model_revision"),
        "source_model_digest": prediction.get("source_model_digest"),
        "evaluated_against": "independent_human_gold",
        "self_scored": False,
        "experiment_namespace": REVISED_NAMESPACE,
        "configuration": R2_CONFIGURATION,
        "pooling_permitted": False,
        "pooling_note": (
            "This single evaluation is not poolable until a strong-binding "
            "POSTHOC_EVALUATION_READINESS_MANIFEST clears"
        ),
        "guarantee_allocation": allocation.as_dict(),
        "assumption_discharge": discharge.as_dict(),
    }

    # Separate A2 agreement categories (§6), added only when both the gold and the
    # prediction carry the atomic facts. Each is reported on its own; they are never
    # merged with allocation/discharge or with each other into a composite F1.
    from .ag_eval_semantics import (
        invariant_agreement,
        priority_agreement,
        timing_agreement,
    )
    if gold.get("timing") is not None and pred_graph.get("timing") is not None:
        result["timing_agreement"] = timing_agreement(
            pred_graph["timing"], gold["timing"]
        )
    if gold.get("priority") is not None and pred_graph.get("priority") is not None:
        result["priority_agreement"] = priority_agreement(
            pred_graph["priority"], gold["priority"]
        )
    if (
        gold.get("invariants") is not None
        and pred_graph.get("invariants") is not None
    ):
        result["invariant_agreement"] = invariant_agreement(pred_graph, gold)
    return result
