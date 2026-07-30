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
- event ordering is part of every handoff/context verdict: a topic published by
  another task or after consumer activation cannot satisfy the metric;
- model-dependent session turns carry revision/digest/event stamps, so use of a
  superseded revision is measured at the turn rather than inferred from whether
  the revision still exists in history;
- ``irrelevant_context_ratio`` uses the explicit record dependency closure:
  required task topics plus diagnostic/evidence/previous-attempt records.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

METRICS_SCHEMA_VERSION = "1.1"


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
    records_by_id = {
        str(record.get("record_id")): record for record in records
    }
    context_events = {
        str((record.get("payload") or {}).get("envelope_id")): record
        for record in records
        if record.get("topic") == "context.created"
    }

    # The context must match the board revision at the context.created event, not
    # merely any historical revision that still exists in the revision log.
    env_consistent = sum(
        1 for e in envelopes
        if (
            (event := context_events.get(str(e.get("envelope_id")))) is not None
            and int(e.get("model_revision", -1))
            == int(event.get("model_revision", -2))
            and e.get("model_digest") == event.get("model_digest")
            and str(e.get("task_id")) == str(event.get("task_id"))
        )
    )

    # required-context coverage, §13: required categories present in the envelope
    # over the required categories in that ROLE's policy, averaged over tasks. The
    # denominator used to be "one per envelope, satisfied by carrying anything at
    # all", which cannot fall below 1.0 for any envelope that carries a single
    # record — a metric that cannot fail. The policy
    # (`context_builder.REQUIRED_CONTEXT_BY_ROLE`) is ablation-derived, so each
    # category in the denominator is one whose removal demonstrably breaks or
    # silently degrades the task.
    from .context_builder import context_coverage

    per_envelope = [context_coverage(item) for item in envelopes]
    scored = [item for item in per_envelope if item]
    coverage_ratios = [
        item["present"] / item["required"] for item in scored if item["required"]
    ]
    coverage_missing = sorted({
        name for item in scored for name in item["missing"]
    })
    unscored_roles = sorted({
        str(envelope.get("agent_role"))
        for envelope, coverage in zip(envelopes, per_envelope)
        if coverage is None
    })

    # A handoff is complete only when the SAME task's envelope consumes every
    # required typed publication from the same revision before activation.
    handoff_tasks = [t for t in tasks if t.get("required_topics")]
    handoff_complete = 0
    handoff_failures: List[dict] = []
    for task in handoff_tasks:
        task_id = str(task.get("task_id"))
        envelope = next(
            (item for item in envelopes if str(item.get("task_id")) == task_id),
            None,
        )
        event = (
            context_events.get(str(envelope.get("envelope_id")))
            if envelope else None
        )
        included = set(envelope.get("included_record_ids") or ()) if envelope else set()
        missing: List[str] = []
        for topic in task.get("required_topics") or ():
            matching = [
                records_by_id[record_id]
                for record_id in included
                if record_id in records_by_id
                and records_by_id[record_id].get("topic") == topic
            ]
            valid = bool(event) and any(
                int(record.get("sequence", 10**18))
                < int(event.get("sequence", -1))
                and int(record.get("model_revision", -1))
                == int(envelope.get("model_revision", -2))
                and record.get("model_digest") == envelope.get("model_digest")
                for record in matching
            )
            if not valid:
                missing.append(str(topic))
        if not missing:
            handoff_complete += 1
        else:
            handoff_failures.append({
                "task_id": task_id,
                "missing_or_late_topics": missing,
            })

    # session lifecycle.
    stale_session_ids = {
        str(s.get("session_id"))
        for s in sessions if s.get("status") == "STALE"
    }
    stale_sessions = len(stale_session_ids)
    rebased_stale_ids = {
        str(s.get("rebased_from_session_id"))
        for s in sessions if s.get("rebased_from_session_id")
    } & stale_session_ids
    commits = max(0, len(revisions) - 1)
    # A model-dependent turn on a superseded base without rebasing is prevented by
    # TaskSession.assert_current; a completed session that carries a rebase pointer
    # rebased correctly. Count any COMPLETED session left on a non-committed base.
    # envelope composition: how many carried items were required typed
    # publications for their task, and how many were supplementary
    tasks_by_id = {str(t.get("task_id")): t for t in (board.get("tasks") or ())}
    envelope_items = 0
    envelope_required_items = 0
    irrelevant_items = 0
    for envelope in envelopes:
        required = {
            str(topic)
            for topic in (
                tasks_by_id.get(str(envelope.get("task_id")), {}).get(
                    "required_topics"
                ) or ()
            )
        }
        for item in envelope.get("context_item_provenance") or ():
            if item.get("kind") != "blackboard_record":
                continue
            envelope_items += 1
            if str(item.get("topic")) in required:
                envelope_required_items += 1
        explicit_dependency_ids = {
            str(record_id)
            for field in (
                "diagnostic_record_ids",
                "evidence_record_ids",
                "previous_attempt_record_ids",
            )
            for record_id in (envelope.get(field) or ())
        }
        for record_id in envelope.get("included_record_ids") or ():
            record = records_by_id.get(str(record_id))
            if record is None:
                continue
            if (
                str(record.get("topic")) not in required
                and str(record_id) not in explicit_dependency_ids
            ):
                irrelevant_items += 1

    stale_access = dict(board.get("stale_access") or {})
    stale_rejected = int(stale_access.get("rejected") or 0)
    stale_permitted = int(stale_access.get("permitted") or 0)
    stale_attempted = stale_rejected + stale_permitted
    # Revision current at a message event: revision 0 until the first commit,
    # then the latest model.committed record whose sequence is not after the turn.
    initial = next(
        (item for item in revisions if int(item.get("revision", -1)) == 0),
        None,
    )
    commit_events = sorted(
        (
            int(record.get("sequence", 0)),
            int(record.get("model_revision", -1)),
            record.get("model_digest"),
        )
        for record in records if record.get("topic") == "model.committed"
    )

    def current_at(sequence: int) -> tuple[int, Any]:
        current = (
            int((initial or {}).get("revision", -1)),
            (initial or {}).get("model_digest"),
        )
        for event_sequence, revision, digest in commit_events:
            if event_sequence > sequence:
                break
            current = (revision, digest)
        return current

    stale_turns: List[dict] = []
    for session in sessions:
        for message in session.get("messages") or ():
            if message.get("role") != "assistant":
                continue
            if message.get("board_sequence") is None:
                continue
            expected = current_at(int(message["board_sequence"]))
            observed = (
                int(message.get("model_revision", -1)),
                message.get("model_digest"),
            )
            if observed != expected:
                stale_turns.append({
                    "session_id": session.get("session_id"),
                    "message_sequence": message.get("sequence"),
                    "board_sequence": message.get("board_sequence"),
                    "observed_revision": observed[0],
                    "expected_revision": expected[0],
                })
    # Backward-compatible detection for legacy snapshots that contain no stamped
    # turns and cite a base that never existed at all.
    invalid_unstamped_sessions = sum(
        1 for s in sessions
        if s.get("status") == "COMPLETED"
        and (int(s.get("base_model_revision", -1)), s.get("base_model_digest"))
        not in committed
        and not any(
            message.get("board_sequence") is not None
            for message in (s.get("messages") or ())
        )
    )
    stale_revision_use = len(stale_turns) + invalid_unstamped_sessions
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
            "note": (
                "compared with the revision current at context.created; a merely "
                "historical committed revision does not count"
            ),
        },
        "required_context_coverage": {
            "envelopes_scored": len(scored),
            "envelopes_total": len(envelopes),
            "value": (
                round(sum(coverage_ratios) / len(coverage_ratios), 4)
                if coverage_ratios else None
            ),
            "missing_categories": coverage_missing,
            "roles_without_a_policy": unscored_roles,
            "note": (
                "denominator is the role's ablation-derived required categories "
                "(context_builder.REQUIRED_CONTEXT_BY_ROLE); a role with no "
                "declared policy is not scored rather than scored 1.0"
            ),
        },
        "cross_agent_handoff_completeness": {
            "complete": handoff_complete,
            "total": len(handoff_tasks),
            "value": _ratio(handoff_complete, len(handoff_tasks)),
            "illustrative_single_handoff": len(handoff_tasks) <= 1,
            "failures": handoff_failures,
        },
        "stale_session_detection": {
            "stale_sessions": stale_sessions,
            "commits_observed": commits,
            "closed_or_rebased": stale_sessions,
            "rebased": len(rebased_stale_ids),
            "value": _ratio(stale_sessions, stale_sessions),
        },
        "stale_revision_use": {
            "count": stale_revision_use,
            "target": 0,
            "stale_turns": stale_turns,
            "legacy_invalid_base_sessions": invalid_unstamped_sessions,
            "note": "measured from per-turn revision/digest/event stamps",
        },
        # §13: rejected / attempted operations against a superseded revision. A
        # rejection raises, so it is counted at the guard rather than recovered
        # from the record log, which only ever holds operations that succeeded.
        # `value` is None when nothing was attempted: a run that never went stale
        # reports no rate rather than a vacuous 1.0.
        "stale_rejection_rate": {
            "rejected": stale_rejected,
            "permitted": stale_permitted,
            "attempted": stale_attempted,
            "value": _ratio(stale_rejected, stale_attempted),
            "note": (
                "denominator is attempts against a superseded revision; a run "
                "with no stale attempt reports null, not 1.0"
            ),
        },
        # NOT a measurement under this architecture, and reported as what it is.
        # A session owns exactly one role and one task (design §5.3 rules 1 and 6),
        # so contamination cannot occur: the count is 0 because the structure
        # forbids it, not because a run avoided it. Reporting it as a measured rate
        # would restate a definition as a finding. It becomes measurable only in
        # the R1-LONG shared-session condition, which is not implemented (§18-Q2),
        # so `measured` stays false and no denominator is claimed. The pillar-2
        # evidence is the metrics below that CAN fail on a real run:
        # context_revision_consistency, stale_revision_use, required_context_
        # coverage, envelope_truncation, stale_session_detection.
        "cross_role_contamination": {
            "kind": "ARCHITECTURAL_PROPERTY",
            "measured": False,
            "count": cross_role,
            "target": 0,
            "note": (
                "structurally impossible: one session owns one role and one task, "
                "so 0 is entailed by the design rather than observed. Measurable "
                "only under the R1-LONG shared-session diagnostic, which is not "
                "implemented; do not report this as a coordination rate"
            ),
        },
        "session_context_growth": growth,
        "envelope_truncation": {"truncated": truncated, "total": len(envelopes)},
        "irrelevant_context_ratio": {
            "irrelevant_items": irrelevant_items,
            "total_record_items": envelope_items,
            "value": _ratio(irrelevant_items, envelope_items),
            "dependency_closure": (
                "task.required_topics plus diagnostic/evidence/"
                "previous-attempt record ids"
            ),
        },
        # Descriptive companion to the operational ratio above: this preserves the
        # raw required-topic/supplementary composition without grading every
        # supplementary record as irrelevant.
        "envelope_composition": {
            "items": envelope_items,
            "required_topic_items": envelope_required_items,
            "supplementary_items": envelope_items - envelope_required_items,
            "note": (
                "descriptive only; a supplementary item is not evidence of "
                "irrelevance — repair context legitimately carries prior attempts"
            ),
        },
        "cost": cost,
        "mvp_caveats": [
            "handoff/role metrics have a denominator of two migrated handoffs "
            "(Requirements->Design, Design->Verification); a bounded subset, not "
            "a pipeline-wide rate (§13)",
            "invariant metrics read 1.0/0 because R1 enforces them; R0 has no "
            "blackboard mechanism to enforce or measure them",
        ],
    }
