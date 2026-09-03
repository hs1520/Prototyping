"""Stale-read/write rejection rate (design §13, Group A).

The metric is rejected / attempted operations against a superseded revision. A
rejection raises `StaleRevisionError` and aborts, so it never reaches the record
log and cannot be computed from archived artifacts; it is counted at the guard
and carried in the blackboard snapshot. A run that never went stale reports no
rate rather than 1.0, and a permitted stale read moves the denominator.
"""
from __future__ import annotations

import pytest

from src.prototyping.blackboard import (
    Blackboard,
    RecordType,
    StaleRevisionError,
)
from src.prototyping.run_metrics import compute_coordination_metrics

_MODEL = "package P { part def A; }"
_NEXT = "package P { part def A; part def B; }"


def _board() -> Blackboard:
    board = Blackboard("w", _MODEL)
    board.publish(
        RecordType.SOURCE, "requirements.authoritative", "requirements",
        {"text": "x"},
    )
    return board


def _metrics(board) -> dict:
    return compute_coordination_metrics(
        {"blackboard": board.snapshot(), "contexts": [], "task_sessions": []}
    )["stale_rejection_rate"]


def test_no_stale_reports_no_rate():
    out = _metrics(_board())
    assert out["attempted"] == 0
    assert out["value"] is None, "a run with no stale attempt must not report 1.0"


def test_rejected_stale_publish_counted():
    board = _board()
    stale = board.current_revision
    board.commit_model(
        _NEXT, base_revision=stale, base_digest=board.current_model.model_digest,
        producer="design",
    )
    with pytest.raises(StaleRevisionError):
        board.publish(
            RecordType.ANALYSIS, "t", "p", {"a": 1}, model_revision=stale
        )
    out = _metrics(board)
    assert (out["rejected"], out["permitted"], out["attempted"]) == (1, 0, 1)
    assert out["value"] == 1.0


def test_permitted_stale_moves_denominator():
    board = _board()
    stale = board.current_revision
    board.commit_model(
        _NEXT, base_revision=stale, base_digest=board.current_model.model_digest,
        producer="design",
    )
    board.publish(
        RecordType.ANALYSIS, "t", "p", {"a": 1}, model_revision=stale,
        allow_stale=True,
    )
    out = _metrics(board)
    assert (out["rejected"], out["permitted"], out["attempted"]) == (0, 1, 1)
    assert out["value"] == 0.0


def test_rejected_stale_commit_counted():
    board = _board()
    stale = board.current_revision
    digest = board.current_model.model_digest
    board.commit_model(
        _NEXT, base_revision=stale, base_digest=digest, producer="design"
    )
    with pytest.raises(StaleRevisionError):
        board.commit_model(
            "package P { part def C; }", base_revision=stale, base_digest=digest,
            producer="design",
        )
    assert _metrics(board)["rejected"] == 1


def test_superseded_digest_counted():
    board = _board()
    with pytest.raises(StaleRevisionError):
        board.commit_model(
            _NEXT, base_revision=board.current_revision,
            base_digest="not-the-committed-digest", producer="design",
        )
    assert _metrics(board)["rejected"] == 1


def test_irrelevance_uses_dependency_closure():
    out = compute_coordination_metrics(
        {"blackboard": _board().snapshot(), "contexts": [], "task_sessions": []}
    )
    ratio = out["irrelevant_context_ratio"]
    assert ratio["value"] is None
    assert "task.required_topics" in ratio["dependency_closure"]
    composition = out["envelope_composition"]
    assert set(composition) >= {
        "items", "required_topic_items", "supplementary_items", "note"
    }
    assert "value" not in composition
    assert "not evidence of" in composition["note"]


def test_gold_coverage_incomplete_freeze():
    """A gold fact family is optional in the schema, so a freeze can validate without
    one: the REQ_SAFE_005 freeze lacked realization_links and left a primary metric
    uncomputable. Surfaced at freeze time.
    """
    from src.prototyping.ag_gold_template import gold_metric_coverage

    complete = {
        "chain_id": "X", "allocations": [1], "discharge_edges": [1],
        "realization_links": [1], "timing": {"a": 1}, "priority": {"a": 1},
        "invariants": [1],
    }
    assert not gold_metric_coverage(complete)["unsupported_metrics"]

    missing = dict(complete)
    del missing["realization_links"]
    out = gold_metric_coverage(missing)
    assert "realization_links" in out["unsupported_metrics"]
    assert "realization" in out["unsupported_metrics"]["realization_links"]

    emptied = dict(complete, realization_links=[])
    assert "realization_links" in gold_metric_coverage(emptied)["unsupported_metrics"]
