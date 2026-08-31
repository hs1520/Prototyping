"""Bounded, issue-authorized revision of a frozen whole-model plan.

The generation pipeline freezes the typed plan after Phase 2 and constrains
every downstream repair to be plan-authorized. That invariant is what keeps
refinement from fabricating structure — and it also means no repair inside the
loop can ever repair the plan itself. Run 2026-08-31 spent 8 idle iterations
against `structural_repair_blocked` because the blocked message's own remedy
("requires a validated plan revision") had no code path.

This module is that path, mirrored on the pipeline's existing repair
discipline one level up:

- **Trigger**: only a `structural_repair_blocked` verdict — never free
  refinement preference.
- **Issue-authorized diff**: the revised plan may change only what the handed
  issues name. `components` and `connections` are frozen outright (a blockage
  is never license to rewire), and every untouched realization/behaviour must
  survive byte-identically. The mirror of "plan-authorized repair".
- **Validated**: the revision re-enters `ModelGenerationPlan.from_payload`
  under the full current validator set, so it cannot dodge any gate the
  original plan passed — or any gate added since.
- **Bounded**: a fixed attempt ceiling, every attempt recorded.

Monotonic acceptance (the revised plan must strictly shrink the unsatisfied
obligation set on the actual model text) is the caller's step, because it
needs the model; see ``GenerationPipelineMixin._attempt_frozen_plan_revision``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..llm.interface import ESCALATION_TEMPERATURES
from ..prototyping.generation_plan import ModelGenerationPlan
from ..prototyping.plan_patch import normalise_for_mention as _normalise_for_mention
from ..utils.req_id import normalise_req_id

if TYPE_CHECKING:
    from ..llm.chain_of_thought import ChainOfThoughtPrompter

#: Revision is a recovery path, not a second planning loop: one correction
#: retry after the first attempt, then fail closed with the evidence.
MAXIMUM_REVISION_ATTEMPTS = 2


@dataclass(frozen=True)
class PlanRevisionRequest:
    system_name: str
    requirements: Sequence[str]
    #: The frozen plan's archived payload (``ModelGenerationPlan.to_dict``).
    frozen_plan: Mapping[str, Any]
    #: The ``structural_repair_blocked`` record that triggered revision.
    blocked: Mapping[str, Any]
    verbose: bool = False


@dataclass(frozen=True)
class PlanRevisionOutcome:
    #: The accepted revised plan, or None when revision failed closed.
    plan: ModelGenerationPlan | None
    #: Full audit record for run metadata.
    record: dict[str, Any] = field(default_factory=dict)


def _blocked_issue_lines(blocked: Mapping[str, Any]) -> list[str]:
    """Flatten the blockage into the issue lines that authorize the revision."""
    lines: list[str] = []
    report = blocked.get("structural_obligation_report")
    if isinstance(report, Mapping):
        for result in report.get("results", ()):
            if not isinstance(result, Mapping):
                continue
            if result.get("status") == "PASS":
                continue
            for issue in result.get("issues", ()):
                lines.append(
                    f"{result.get('obligation_id')}: {issue}"
                )
    conformance = blocked.get("generation_plan_conformance")
    if isinstance(conformance, Mapping):
        for issue in conformance.get("issues", ()):
            lines.append(str(issue))
    return list(dict.fromkeys(lines))


def _issue_authorized_diff(
    frozen: ModelGenerationPlan,
    revised: ModelGenerationPlan,
    issue_lines: Sequence[str],
) -> list[str]:
    """Reject every change the handed issues do not name.

    A change is authorized iff some issue line mentions the changed entry's
    requirement id, owner, or behaviour name. Everything else must survive
    byte-identically — the revision gate is the mirror, one level up, of the
    plan-authorized repair gate it unblocks.
    """
    mention_pool = _normalise_for_mention(" ".join(issue_lines))

    def mentioned(*names: str | None) -> bool:
        return any(
            name and _normalise_for_mention(name) in mention_pool
            for name in names
        )

    violations: list[str] = []

    def canonical(items: Sequence[Any]) -> list[Any]:
        return sorted(
            (item.to_dict() for item in items),
            key=lambda value: json.dumps(value, sort_keys=True),
        )

    if canonical(revised.components) != canonical(frozen.components):
        violations.append(
            "components changed; a structural blockage does not authorize "
            "rewiring the component surface"
        )
    if canonical(revised.connections) != canonical(frozen.connections):
        violations.append(
            "connections changed; a structural blockage does not authorize "
            "rewiring the connection surface"
        )
    if canonical(revised.constraint_plans) != canonical(
        frozen.constraint_plans
    ):
        violations.append("constraints changed without authorization")

    frozen_realizations = {
        item.requirement_id: item.to_dict()
        for item in frozen.requirement_realizations
    }
    revised_realizations = {
        item.requirement_id: item.to_dict()
        for item in revised.requirement_realizations
    }
    for req_id in (
        set(frozen_realizations) | set(revised_realizations)
    ):
        if (
            frozen_realizations.get(req_id)
            == revised_realizations.get(req_id)
        ):
            continue
        if not mentioned(req_id):
            violations.append(
                f"requirement realization {req_id} changed but no issue "
                "names it"
            )

    frozen_behaviors = {
        (item.owner, item.behavior_id): item.to_dict()
        for item in frozen.planned_behaviors
    }
    revised_behaviors = {
        (item.owner, item.behavior_id): item.to_dict()
        for item in revised.planned_behaviors
    }
    for key in set(frozen_behaviors) | set(revised_behaviors):
        if frozen_behaviors.get(key) == revised_behaviors.get(key):
            continue
        owner, behavior_id = key
        source_req = (
            frozen_behaviors.get(key) or revised_behaviors.get(key) or {}
        ).get("provenance", {}).get("requirement_id")
        # The owner alone does not authorize: one part owns many behaviours
        # (FlightController owns eight in the reference plan), so an issue
        # naming the owner would license rewriting all of them. The issue
        # must name the behaviour itself or its source requirement.
        if not mentioned(behavior_id, source_req):
            violations.append(
                f"planned behavior {owner}::{behavior_id} changed but no "
                "issue names it"
            )
    return violations


class PlanRevision:
    """Own prompt context, parsing, validation, diff gate, and audit."""

    def __init__(
        self,
        prompter: ChainOfThoughtPrompter,
        *,
        maximum_attempts: int = MAXIMUM_REVISION_ATTEMPTS,
    ) -> None:
        if int(maximum_attempts) < 1:
            raise ValueError("maximum_attempts must be at least 1")
        self._prompter = prompter
        self._maximum_attempts = int(maximum_attempts)

    def revise(self, request: PlanRevisionRequest) -> PlanRevisionOutcome:
        requirements = list(request.requirements)
        frozen = ModelGenerationPlan.from_payload(
            dict(request.frozen_plan),
            requirements=requirements,
            source="FROZEN_PLAN_REVISION_BASE",
            require_source_anchored_paths=True,
        )
        issue_lines = _blocked_issue_lines(request.blocked)
        # The frozen payload re-validated under the CURRENT validator set may
        # carry issues the original freeze predates (a gate added since).
        # They are equally revision-authorizing: the reviser must satisfy
        # today's validators, not the freeze date's.
        issue_lines.extend(frozen.issues)
        issue_lines = list(dict.fromkeys(issue_lines))
        if not issue_lines:
            return PlanRevisionOutcome(
                plan=None,
                record={
                    "status": "NOT_ATTEMPTED",
                    "reason": "blockage carries no issue lines to "
                              "authorize a revision",
                },
            )

        blocked_requirements = sorted({
            normalise_req_id(match)
            for line in issue_lines
            for match in _req_ids_in(line)
        })
        base_context = "\n\n".join((
            "FROZEN MODEL PLAN REVISION — the committed model can no longer "
            "satisfy the frozen plan through plan-authorized repair, so the "
            "plan itself is being revised. Return exactly one complete "
            "replacement JSON object in a ```json block and no prose. Do "
            "not emit SysML.\n"
            "Change ONLY what the issues below implicate"
            + (
                " (requirements " + ", ".join(blocked_requirements) + ")"
                if blocked_requirements else ""
            )
            + ". components and connections must be identical to the "
            "frozen plan; every realization and behavior no issue names "
            "must be unchanged. Unauthorized changes are rejected "
            "deterministically.",
            "VALIDATION ISSUES:\n"
            + "\n".join(f"- {line}" for line in issue_lines),
            "FROZEN PLAN — the repair base; preserve every field not "
            "implicated by an issue:\n```json\n"
            + json.dumps(dict(request.frozen_plan), indent=2)
            + "\n```",
        ))

        attempts: list[dict[str, Any]] = []
        attempt_context = base_context
        accepted: ModelGenerationPlan | None = None
        while len(attempts) < self._maximum_attempts and accepted is None:
            attempt_index = len(attempts)
            response = self._prompter.decompose_architecture(
                system_name=request.system_name,
                requirements=requirements,
                context=attempt_context,
                temperature=ESCALATION_TEMPERATURES[
                    min(attempt_index, len(ESCALATION_TEMPERATURES) - 1)
                ],
            )
            attempt_issues: list[str] = []
            revised: ModelGenerationPlan | None = None
            if isinstance(response.extracted_json, Mapping):
                revised = ModelGenerationPlan.from_payload(
                    response.extracted_json,
                    requirements=requirements,
                    source="LLM_PLAN_REVISION",
                    require_source_anchored_paths=True,
                )
                attempt_issues.extend(revised.issues)
                if revised.status == "PASS":
                    diff_violations = _issue_authorized_diff(
                        frozen, revised, issue_lines
                    )
                    attempt_issues.extend(diff_violations)
                    if not diff_violations:
                        accepted = revised
            else:
                attempt_issues.append(
                    "revision response contained no parseable JSON object"
                )
            attempts.append({
                "attempt": attempt_index + 1,
                "plan_status": (
                    revised.status if revised is not None else "UNAVAILABLE"
                ),
                "issues": list(dict.fromkeys(attempt_issues)),
                "accepted": accepted is not None,
            })
            if accepted is None:
                attempt_context = "\n\n".join((
                    base_context,
                    "REVISION CORRECTION — the previous revision was "
                    "rejected. Resolve every issue below while keeping all "
                    "unimplicated fields identical to the frozen plan:\n"
                    + "\n".join(
                        f"- {issue}" for issue in attempts[-1]["issues"]
                    ),
                ))

        record = {
            "status": "ACCEPTED" if accepted is not None else "REJECTED",
            "authorizing_issues": issue_lines,
            "blocked_requirements": blocked_requirements,
            "attempts": attempts,
        }
        return PlanRevisionOutcome(plan=accepted, record=record)


def _req_ids_in(text: str) -> list[str]:
    import re

    return re.findall(r"\bREQ[_-][A-Za-z]+[_-]\d+\b", text)
