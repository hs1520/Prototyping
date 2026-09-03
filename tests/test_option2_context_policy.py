"""The required-context policy is ablation-derived, and stays that way (§18-Q1).

Reading required categories off observed failures was not possible: at
derivation (2026-07-26) every archived task was COMPLETED except ten BLOCKED by
design, with no REJECTED or STALE session. "Required" therefore means
operational - remove the category and the task raises, returns nothing, or
silently returns a worse answer - and every entry in `REQUIRED_CONTEXT_BY_ROLE`
is re-derived here. One later run does carry failures, pinned by the archive
test below as a named exception.
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


def test_design_needs_authoritative_source():
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


def test_planning_needs_model_slice():
    assert plan_verification(_MODEL)["planned"] == 2
    assert plan_verification("")["planned"] == 0


def test_stripped_source_degrades_plan():
    """failure_mode: silent_degradation.

    The planned-requirement count does not move while every method becomes
    `inspection`, so a metric asking only whether a slice was present scores 1.0.
    """
    full = plan_verification(_MODEL)
    stripped = plan_verification(re.sub(r"doc /\*.*?\*/", "", _MODEL, flags=re.S))

    assert stripped["planned"] == full["planned"]
    assert set(full["tier_histogram"]) != {"inspection"}
    assert set(stripped["tier_histogram"]) == {"inspection"}
    assert all(not item["has_doc"] for item in stripped["entries"])


def test_policy_entries_name_failure_mode():
    for role, requirements in REQUIRED_CONTEXT_BY_ROLE.items():
        assert requirements, role
        for item in requirements:
            assert item.failure_mode in {
                "raises", "blocked", "empty_result", "silent_degradation",
            }, (role, item.category)
            # the evidence must be specific enough to re-run, not a claim
            assert len(item.evidence) > 40, (role, item.category)


def test_repair_blocked_without_diagnostics():
    """failure_mode: blocked. The repair slice comes from the failure record's
    element pointers, so an unresolvable diagnostic fails closed (§11 allows no
    whole-model fallback).
    """
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


def test_role_without_policy_unscored():
    assert context_coverage({"agent_role": "ArchitectureAgent"}) is None
    assert context_coverage({
        "agent_role": "VerificationAgent",
        "model_context": _MODEL,
        "source_requirements": [_REQ],
    }) == {"required": 2, "present": 2, "missing": []}
    assert context_coverage({
        "agent_role": "VerificationAgent",
        "model_context": "requirement def REQ_SAFE_005 { }",
    }) == {
        "required": 2, "present": 1, "missing": ["requirement_source_text"],
    }


# The one archived run allowed to carry failure statuses, from 2026-08-11 and the
# superseded commit its directory name records: its seed-1 R2 repair loop hit
# undischargeable diagnostics - 3 REJECTED (`target_not_removed_or_regression`)
# and 14 BLOCKED (`automatic_repair_budget_exhausted`). Reference batches
# (`pilot_n6_0c26731_20260821_*`) archive zero REJECTED.
_SUPERSEDED_FAILURE_RUN = "pilot_n6_4bb7544_20260811_2316"


def test_archive_has_no_context_failures():
    """Pins the premise: the policy was not derived from failures, because there were
    none.

    The policy predates every archived failure. `_SUPERSEDED_FAILURE_RUN` is the one
    post-derivation run that carries any, pinned by name so new rejections still
    fail the check.
    """
    import glob
    import json

    statuses: set[str] = set()
    superseded: set[str] = set()
    for path in glob.glob("examples/output/*/seed-*/R*/blackboard_snapshot.json"):
        snapshot = json.load(open(path))
        found = {
            str(task.get("status")) for task in snapshot.get("tasks", ())
        }
        if f"/{_SUPERSEDED_FAILURE_RUN}/" in path:
            superseded |= found
        else:
            statuses |= found
    if not (statuses or superseded):
        pytest.skip("no archived runs in this checkout")
    assert statuses <= {"COMPLETED", "BLOCKED"}, statuses
    assert superseded <= {"COMPLETED", "BLOCKED", "REJECTED"}, superseded


def test_repair_slice_covers_omission():
    """§17 risk row "context selection omits a necessary dependency", measured.

    The slicer closes over symbols the sliced elements reference, which runs the
    wrong way for an omission fault; the trigger diagnostic now names
    complete-concept signal candidates, which can pull the declaration into the base
    slice. Invariant: the final repair context carries the declaration and contract
    facts needed to restore the omitted edge. `ag_repair._ag_context_supplement`
    adds only those two A/G facts, as prompt context; the accept gates are
    unchanged.
    """
    import re

    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_contracts import check_ag_graph
    from src.prototyping.ag_emitter import emit_ag_package
    from src.prototyping.ag_extractor import extract_ag_graph
    from src.prototyping.ag_assurance import route_failure_diagnostics
    from src.prototyping.ag_repair import _ag_context_supplement
    from src.simulation.surgical_refiner import build_dependency_closed_context

    model = (
        "package Drone {\n    requirement def REQ_SAFE_005 { doc /* deploy the "
        "parachute within 0.5 seconds */ }\n}\n\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN)
    )
    transition = re.search(
        r"transition (\w+) first (\w+) accept (\w+) then (\w+);", model
    )
    signal = transition.group(3)
    injured = model.replace(transition.group(0), "")

    report = check_ag_graph(extract_ag_graph(injured))
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    failure = next(
        item for item in routed["failures"] if item.get("repair_authorized")
    )
    issue = (
        f"{failure.get('source_requirement')} {failure.get('contract') or ''} "
        + " ".join(str(x) for x in failure.get("affected_elements", ()))
        + f": {failure.get('message') or failure.get('diagnostic_code')}"
    )
    sliced = build_dependency_closed_context(
        injured, [issue],
        allowed_req_ids={str(failure.get("source_requirement"))},
    )
    assert sliced is not None

    declaration = f"item def {signal};"
    assert declaration in injured, "the declaration still exists in the model"

    supplemented = sliced.text + _ag_context_supplement(
        injured, str(failure.get("contract") or "")
    )
    assert declaration in supplemented
    assert f"requirement def {failure.get('contract')}" in supplemented
    assert len(supplemented.splitlines()) < len(injured.splitlines()) / 2
