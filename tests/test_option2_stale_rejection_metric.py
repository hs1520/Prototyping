"""Stale-read/write rejection rate (design §13, Group A).

The metric is rejected / attempted operations against a superseded revision. A
rejection raises `StaleRevisionError` and aborts, so nothing about it reached the
record log — which only ever holds operations that succeeded — and the metric could
not be computed from archived artifacts at all. It is therefore counted at the
guard itself and carried in the blackboard snapshot.

The denominator must stay honest: a run that never went stale reports no rate
rather than a vacuous 1.0, and a permitted stale read has to move the denominator,
or the rate would be 1.0 by construction and would evidence nothing.
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


def test_a_run_that_never_went_stale_reports_no_rate():
    out = _metrics(_board())
    assert out["attempted"] == 0
    assert out["value"] is None, "a run with no stale attempt must not report 1.0"


def test_a_rejected_stale_publish_is_counted():
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


def test_a_permitted_stale_read_moves_the_denominator():
    """Without this the rate is 1.0 by construction and evidences nothing."""
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


def test_a_rejected_stale_commit_is_counted():
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


def test_a_commit_on_a_superseded_digest_is_counted():
    board = _board()
    with pytest.raises(StaleRevisionError):
        board.commit_model(
            _NEXT, base_revision=board.current_revision,
            base_digest="not-the-committed-digest", producer="design",
        )
    assert _metrics(board)["rejected"] == 1


def test_envelope_composition_is_descriptive_not_an_irrelevance_ratio():
    """`irrelevant_context_ratio` is deliberately not computed: the builder carries
    the ids its caller passes, so "in closure" cannot be derived here. What is
    reported must therefore describe the envelope, not grade it."""
    out = compute_coordination_metrics(
        {"blackboard": _board().snapshot(), "contexts": [], "task_sessions": []}
    )
    assert "irrelevant_context_ratio" not in out
    composition = out["envelope_composition"]
    assert set(composition) >= {
        "items", "required_topic_items", "supplementary_items", "note"
    }
    # it must not present itself as a quality ratio
    assert "value" not in composition
    assert "not evidence of" in composition["note"]
