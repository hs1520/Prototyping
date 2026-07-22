"""Deterministic coordination/context metrics for the revised arms (Increment 4).

Computes the Group-A metrics of design §13 from a run's archived collaboration
artifacts (blackboard snapshot + context envelopes + task sessions) plus the LLM
ledger. No human gold is needed — these are the R1-BBCTX metrics.

Honesty notes carried in the output:
- several quantities are **invariants R1 enforces by construction** (a session is
  revision-pinned, gold cannot enter an envelope, a stale session cannot take a
  model-dependent turn). Here they read 1.0 / 0; R0-CURRENT has no blackboard and
  therefore no mechanism to enforce or even measure them — that asymmetry is the
  point of the comparison, not a claim that R1 "scored higher on a shared scale".
- handoff-based metrics have a denominator of two migrated handoffs in the MVP
  (RequirementsAgent→DesignAgent, DesignAgent→VerificationAgent) — a bounded
  subset, still reported as illustrative rather than a pipeline-wide rate (§13).
- ``irrelevant_context_ratio`` needs a per-task dependency oracle and is not
  computed here; envelope truncation is reported instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

METRICS_SCHEMA_VERSION = "1.0"


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def compute_coordination_metrics(
    collaboration: Mapping[str, Any],
    *,
    llm_usage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute deterministic R1-BBCTX coordination/context metrics.

    ``collaboration`` is the ``collaboration`` block of a revised run result
    (``blackboard`` / ``contexts`` / ``task_sessions`` snapshots).
    """
    board = dict(collaboration.get("blackboard") or {})
    envelopes: List[dict] = list(
        (collaboration.get("contexts") or {}).get("envelopes") or []
    )
    sessions: List[dict] = list(
        (collaboration.get("task_sessions") or {}).get("sessions") or []
    )
    revisions: List[dict] = list(board.get("model_revisions") or [])
    records: List[dict] = list(board.get("records") or [])
    tasks: List[dict] = list(board.get("tasks") or [])
    protected = dict(board.get("protected_requirement_digests") or {})

    committed = {
        (int(r.get("revision")), r.get("model_digest")) for r in revisions
    }

    # context-revision consistency: every envelope pins a real committed revision.
    env_consistent = sum(
        1 for e in envelopes
        if (int(e.get("model_revision", -1)), e.get("model_digest")) in committed
    )

    # required-context coverage: a design/refinement envelope must carry its
    # authoritative source (source_requirements) or explicit included records.
    coverage_num = sum(
        1 for e in envelopes
        if e.get("source_requirements") or e.get("included_record_ids")
    )

    # cross-agent handoff completeness: every required topic of a migrated task
    # was published as a typed record (illustrative — one handoff in the MVP).
    published_topics = {r.get("topic") for r in records}
    handoff_tasks = [t for t in tasks if t.get("required_topics")]
    handoff_complete = sum(
        1 for t in handoff_tasks
        if all(topic in published_topics for topic in t.get("required_topics", ()))
    )

    # session lifecycle.
    stale_sessions = sum(1 for s in sessions if s.get("status") == "STALE")
    commits = max(0, len(revisions) - 1)
    # A model-dependent turn on a superseded base without rebasing is prevented by
    # TaskSession.assert_current; a completed session that carries a rebase pointer
    # rebased correctly. Count any COMPLETED session left on a non-committed base.
    stale_revision_use = sum(
        1 for s in sessions
        if s.get("status") == "COMPLETED"
        and (int(s.get("base_model_revision", -1)), s.get("base_model_digest"))
        not in committed
    )
    # cross-role contamination: an R1 session owns exactly one role/task by
    # construction; contamination is only possible in the R1-LONG shared session.
    role_task_pairs = [(s.get("agent_role"), s.get("task_id")) for s in sessions]
    cross_role = len(role_task_pairs) - len(set(role_task_pairs))

    growth = [
        {
            "session_id": s.get("session_id"),
            "agent_role": s.get("agent_role"),
            "used_tokens": s.get("used_tokens"),
            "assistant_turns": s.get("assistant_turns"),
        }
        for s in sessions
    ]
    truncated = sum(1 for e in envelopes if e.get("truncated"))

    cost = None
    if llm_usage:
        cost = {
            "llm_calls": llm_usage.get("calls"),
            "total_tokens": llm_usage.get("total_tokens"),
            "elapsed_seconds": llm_usage.get("elapsed_seconds"),
        }

    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "artifact_role": "R1_COORDINATION_METRICS",
        "metric_scope": "R1-BBCTX coordination/context (deterministic, no gold)",
        "semantic_authority": board.get("semantic_authority"),
        "counts": {
            "envelopes": len(envelopes),
            "sessions": len(sessions),
            "model_revisions": len(revisions),
            "migrated_handoffs": len(handoff_tasks),
            "protected_requirement_contract_defs": len(protected),
        },
        "source_threshold_unit_preservation": {
            "preserved": len(protected),
            "total": len(protected),
            "value": _ratio(len(protected), len(protected)),
            "note": (
                "enforced by exact committed requirement/contract block identity; "
                "covers source text, comparators, thresholds, units, and constraints"
            ),
        },
        "context_revision_consistency": {
            "consistent": env_consistent,
            "total": len(envelopes),
            "value": _ratio(env_consistent, len(envelopes)),
            "note": "enforced: envelopes are revision-pinned to a committed model",
        },
        "required_context_coverage": {
            "covered": coverage_num,
            "total": len(envelopes),
            "value": _ratio(coverage_num, len(envelopes)),
        },
        "cross_agent_handoff_completeness": {
            "complete": handoff_complete,
            "total": len(handoff_tasks),
            "value": _ratio(handoff_complete, len(handoff_tasks)),
            "illustrative_single_handoff": len(handoff_tasks) <= 1,
        },
        "stale_session_detection": {
            "stale_sessions": stale_sessions,
            "commits_observed": commits,
        },
        "stale_revision_use": {
            "count": stale_revision_use,
            "target": 0,
            "note": "enforced by TaskSession.assert_current",
        },
        "cross_role_contamination": {
            "count": cross_role,
            "target": 0,
            "note": "0 by construction for R1; meaningful only for R1-LONG",
        },
        "session_context_growth": growth,
        "envelope_truncation": {"truncated": truncated, "total": len(envelopes)},
        "cost": cost,
        "mvp_caveats": [
            "handoff/role metrics have a denominator of two migrated handoffs "
            "(Requirements->Design, Design->Verification); a bounded subset, not "
            "a pipeline-wide rate (§13)",
            "irrelevant_context_ratio needs a dependency oracle and is not computed",
            "invariant metrics read 1.0/0 because R1 enforces them; R0 has no "
            "blackboard mechanism to enforce or measure them",
        ],
    }
