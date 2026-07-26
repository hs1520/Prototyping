"""The required-context policy is ablation-derived, and stays that way (§18-Q1).

The intended method was to read required categories off observed failures. The
archive has none: across every archived revised run every task is COMPLETED except
ten BLOCKED by design (§11 routes an integration gap to BLOCKED), and no session is
REJECTED or STALE. So "required" is given an operational meaning instead — remove
the category and the task raises, returns nothing, or silently returns a worse
answer — and every entry in `REQUIRED_CONTEXT_BY_ROLE` is re-derived here.

Without this file the policy would be a list of plausible-sounding categories, and
§13's required-context-coverage metric would have a denominator nobody can justify.
"""
from __future__ import annotations

import re

import pytest

from src.prototyping.blackboard import Blackboard, RecordType
from src.prototyping.context_builder import (
    REQUIRED_CONTEXT_BY_ROLE,
    context_coverage,
    ContextBuilder,
)
from src.prototyping.verification_planning import plan_verification

_REQ = "REQ-SAFE-005: The system shall deploy the parachute within 0.5 seconds."
_MODEL = """package Drone {
    requirement def REQ_SAFE_005 { doc /* deploy the parachute within 0.5 seconds */ }
    requirement def REQ_FUNC_002 { doc /* maintain at least 5 metres of separation */ }
}"""


def _board() -> tuple[Blackboard, str]:
    board = Blackboard("Drone")
    record = board.publish(
        RecordType.SOURCE, "requirements.authoritative", "RequirementsAgent",
        {"requirements": [_REQ]},
    )
    return board, record.record_id


def test_design_agent_cannot_work_without_the_authoritative_source():
    """failure_mode: raises. The strongest kind — it cannot go unnoticed."""
    board, record_id = _board()
    builder = ContextBuilder(board)

    task = board.create_task(
        "design.generate", "DesignAgent",
        required_topics=("requirements.authoritative",),
    )
    envelope = builder.build_design_context(
        task_id=task.task_id, system_name="Drone", source_record_ids=[record_id],
    )
    assert "REQ-SAFE-005" in envelope.render_for_prompt()

    ablated = board.create_task(
        "design.generate", "DesignAgent",
        required_topics=("requirements.authoritative",),
    )
    with pytest.raises(ValueError, match="required typed publications are missing"):
        builder.build_design_context(
            task_id=ablated.task_id, system_name="Drone", source_record_ids=[],
        )


def test_verification_planning_needs_the_committed_slice():
    """failure_mode: empty_result. It plans nothing rather than planning wrongly."""
    assert plan_verification(_MODEL)["planned"] == 2
    assert plan_verification("")["planned"] == 0


def test_stripping_the_source_text_degrades_the_plan_silently():
    """failure_mode: silent_degradation — the dangerous one.

    The count of planned requirements does not move, so nothing looks wrong, and
    every method quietly becomes `inspection`. A coverage metric that only asked
    "was a slice present" would score this 1.0.
    """
    full = plan_verification(_MODEL)
    stripped = plan_verification(re.sub(r"doc /\*.*?\*/", "", _MODEL, flags=re.S))

    assert stripped["planned"] == full["planned"]          # nothing failed
    assert set(full["tier_histogram"]) != {"inspection"}   # the real plan varies
    assert set(stripped["tier_histogram"]) == {"inspection"}
    assert all(not item["has_doc"] for item in stripped["entries"])


def test_every_policy_entry_names_a_measured_failure_mode():
    for role, requirements in REQUIRED_CONTEXT_BY_ROLE.items():
        assert requirements, role
        for item in requirements:
            assert item.failure_mode in {
                "raises", "blocked", "empty_result", "silent_degradation",
            }, (role, item.category)
            # the evidence must be specific enough to re-run, not a claim
            assert len(item.evidence) > 40, (role, item.category)


def test_the_repair_agent_is_blocked_without_resolvable_diagnostics():
    """failure_mode: blocked. The repair slice is derived from the failure
    record's element pointers, so an unresolvable diagnostic must fail closed —
    §11 allows no whole-model fallback — rather than repair a guessed scope."""
    import importlib.util

    from src.prototyping.ag_assurance import route_failure_diagnostics
    from src.prototyping.ag_repair import attempt_dependency_closed_ag_repair
    from src.prototyping.task_session import TaskSessionRegistry

    spec = importlib.util.spec_from_file_location(
        "_assurance_fixtures", "tests/test_option2_ag_assurance.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)

    broken = fixtures.GOOD.replace(
        "state deployed { entry action setParachuteDeployed; }",
        "state deployed { }",
    )
    _graph, report = fixtures._check(broken)
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"]
        if item["diagnostic_code"] == "REALIZATION_ACTION_MISSING"
    )

    def _repair(mutate):
        board = Blackboard("Drone")
        board.commit_model(
            broken, base_revision=0,
            base_digest=board.current_model.model_digest, producer="test",
        )
        analysis = board.publish(
            RecordType.ANALYSIS, "analysis.ag_trace", "AGChecker",
            {"diagnostics": []},
        )
        payload = dict(failure)
        mutate(payload)
        record = board.publish(
            RecordType.ANALYSIS, "diagnostic.failure", "AGFailureRouter", payload,
        )
        builder = ContextBuilder(board)
        decision = attempt_dependency_closed_ag_repair(
            llm=fixtures._RepairLLM(
                "```sysml\n"
                + fixtures._definition(
                    fixtures.GOOD, "state", "RecoverySystemBehavior"
                )
                + "\n```"
            ),
            board=board, context_builder=builder,
            sessions=TaskSessionRegistry(),
            failure_record_id=record.record_id,
            analysis_record_id=analysis.record_id,
        )
        return decision, builder.snapshot()["envelopes"]

    intact, envelopes = _repair(lambda payload: None)
    assert intact.status == "ACCEPTED", (intact.status, intact.reason)
    assert len(envelopes) == 1
    coverage = context_coverage(envelopes[0])
    assert coverage == {"required": 2, "present": 2, "missing": []}, coverage

    stripped, no_envelopes = _repair(
        lambda payload: payload.update(
            affected_elements=(), contract="", message="",
        )
    )
    assert stripped.status == "BLOCKED"
    assert stripped.reason == "dependency_closed_context_unresolved"
    assert no_envelopes == []


def test_a_role_without_a_policy_is_not_scored_as_perfect():
    """A missing denominator must read as "not scored", never as 1.0 — the same
    rule that makes an arm without an A/G layer `n/a` rather than 0.00."""
    assert context_coverage({"agent_role": "ArchitectureAgent"}) is None
    assert context_coverage({
        "agent_role": "VerificationAgent",
        "model_context": _MODEL,
        "source_requirements": [_REQ],
    }) == {"required": 2, "present": 2, "missing": []}
    # and the silent-degradation category is what a slice-only check would miss
    assert context_coverage({
        "agent_role": "VerificationAgent",
        "model_context": "requirement def REQ_SAFE_005 { }",
    }) == {
        "required": 2, "present": 1, "missing": ["requirement_source_text"],
    }


def test_the_archive_offered_no_context_failure_to_learn_from():
    """Pins the premise of the whole method, so a future reader does not assume
    the policy was derived from failures that never happened."""
    import glob
    import json

    statuses: set[str] = set()
    for path in glob.glob("examples/output/*/seed-*/R*/blackboard_snapshot.json"):
        snapshot = json.load(open(path))
        statuses.update(
            str(task.get("status")) for task in snapshot.get("tasks", ())
        )
    if not statuses:
        pytest.skip("no archived runs in this checkout")
    assert statuses <= {"COMPLETED", "BLOCKED"}, statuses
