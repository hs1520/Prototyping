"""Static verification-readiness audit — feeds matrix `unassigned` gaps back into refinement.

The verification matrix runs at the END of the pipeline, so requirements that end up
``unassigned`` (no verification tier anchors them at all) are discovered only after
generation is over and never flow back to the model. ``build_matrix()`` with no
execution results already computes the linker/text tiers exactly (execution results
only upgrade planned tiers to verified; they never create a first anchor); the ONE
anchor source unavailable at refinement time is Phase 8 (datasheet/forward-flight
verdicts), so requirements whose quantified family Phase 8 will decide are excluded
statically via requirement_spec instead. This module runs that projection inside the
Phase 4-5 refinement loop and turns each remaining unassigned requirement into a
surgical-refinement issue, so the model can grow a genuine verification anchor (a
linkable guard/attribute or a state-machine behaviour) while the LLM is still in
the loop.

Honesty boundary: the issues only ask for MODEL anchors that the parameter linker or
the behavioural simulator can genuinely exercise. Requirements that explicitly need
external measurement (CEP, attitude RMS, RTCM corrections) are excluded from the
surgical issue list and remain visibly unassigned in the matrix. The surgical gates
(requirement-def set frozen, satisfy links may not shrink) prevent a model repair from
rewriting the spec instead.
"""
from __future__ import annotations

import re
from typing import Mapping, Iterable, List, Optional

from ..utils.req_id import normalise_req_id

_ISSUE_PREFIX = "[VERIFY-GAP]"
_EXTERNAL_EVIDENCE_PATTERNS = (
    re.compile(r"\bcircular\s+error\s+probable\b", re.IGNORECASE),
    re.compile(r"\bcep\b", re.IGNORECASE),
    re.compile(r"\brtcm\b", re.IGNORECASE),
    re.compile(r"\bdifferential\s+gnss\b", re.IGNORECASE),
    re.compile(r"\brtk(?:\s+bench)?\b", re.IGNORECASE),
    re.compile(r"\battitude\b.{0,160}\brms\b", re.IGNORECASE),
    re.compile(r"\broll\b.{0,100}\bpitch\b.{0,100}\brms\b", re.IGNORECASE),
)


def _requires_external_evidence(text: str) -> bool:
    """Return True for criteria a SysML edit cannot truthfully verify.

    This deliberately keys on the measured quantity, not a generic ``HIL`` tag:
    many behavioural requirements can still gain a useful executable model
    anchor even when HIL is their eventual acceptance method.
    """
    return any(pattern.search(text or "") for pattern in _EXTERNAL_EVIDENCE_PATTERNS)


def _phase8_will_anchor(req_id: str, text: str) -> bool:
    """True when Phase 8 (datasheet/forward-flight verdicts) will anchor this
    requirement downstream — statically decidable from the requirement text via
    requirement_spec, so the audit does not flag it as a gap at refinement time."""
    try:
        from ..dse.requirement_spec import extract_requirements
        from ..realization.closure_types import (
            CLOSURE_SCOPE_FAMILIES,
            FORWARD_FLIGHT_SCOPE_FAMILIES,
            _actionable_spec,
            _family_for_spec,
        )
        specs = extract_requirements([f"{req_id}: {text}"])
    except Exception:
        return False
    phase8_families = CLOSURE_SCOPE_FAMILIES | FORWARD_FLIGHT_SCOPE_FAMILIES
    return any(
        _family_for_spec(s) in phase8_families and _actionable_spec(s)
        for s in specs
    )


def verification_gap_issues(
    model_text: str,
    model_name: str,
    limit: int = 6,
    strict: bool = False,
    allowed_req_ids: Optional[Iterable[str]] = None,
    unmeasurable_req_ids: Optional[Iterable[str]] = None,
    planned_intents: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Return surgical-refinement issues for requirements no verification tier anchors.

    Best-effort: any parsing/linker failure returns [] — the audit must never break
    the refinement loop (same contract as the other best-effort phases).
    """
    if not (model_text or "").strip():
        return []
    try:
        from ..prototyping.verification_matrix import build_matrix
        from ..sitl.requirement_linker import RequirementLinker
        from ..sysml.lite_model import build_lite_model

        lite = build_lite_model(model_text, model_name=model_name)
        linker = RequirementLinker(lite)
        rows = build_matrix(
            lite, None, linker.compile_evidence(),
            planned_intents=dict(planned_intents) if planned_intents else None,
        )
    except Exception:
        if strict:
            raise
        return []
    allowed = (
        {normalise_req_id(str(req_id)) for req_id in allowed_req_ids}
        if allowed_req_ids is not None else None
    )
    unmeasurable = (
        {normalise_req_id(str(req_id)) for req_id in unmeasurable_req_ids}
        if unmeasurable_req_ids is not None else set()
    )
    issues: List[str] = []
    for row in rows:
        if allowed is not None and normalise_req_id(row.req_id) not in allowed:
            continue
        behavioral_failed = "behavioral_sim_failed" in row.tiers
        if row.status != "unassigned" and not behavioral_failed:
            continue
        if (
            not behavioral_failed
            and "planned_no_response" in row.tiers
            and normalise_req_id(row.req_id) in unmeasurable
        ):
            # Two independent signals agree that there is nothing here for a
            # model to anchor: the planner recorded that the requirement
            # obliges no discrete response, and the extractor flagged it as
            # carrying no measurable criterion. Asking the surgical LLM to
            # repair the model would ask it to invent an anchor the requirement
            # does not provide. The row stays UNASSIGNED in the matrix -- the
            # gap is real -- but it is a requirement-side gap, not a
            # model-side one, and it does not block closure.
            continue
        text = " ".join((row.text or "").split())[:220]
        if _phase8_will_anchor(row.req_id, text):
            continue
        if not behavioral_failed and _requires_external_evidence(text):
            # Keep the row UNASSIGNED in the verification matrix, but do not ask
            # the surgical LLM to fabricate model evidence for a measured
            # hardware/HIL quantity.
            continue
        if behavioral_failed:
            issues.append(
                f"{_ISSUE_PREFIX} {row.req_id} has a behavioral verification anchor, "
                f"but its executable model-level scenario FAILS. Requirement: \"{text}\". "
                "Repair the complete satisfying part def; do not add another declaration-only "
                "state machine. Every declared state must be reachable, the initial/default "
                "state must agree with its Boolean attribute, and the required transition or "
                "entry action must actually execute under simulation."
            )
            continue
        default_guidance = ""
        low = text.lower()
        if any(k in low for k in ("power-on", "power on", "default", "startup", "start-up")):
            default_guidance = (
                " For a power-on/default-state requirement, model an explicit initial state "
                "and a consistent Boolean attribute. A single-state invariant is allowed; "
                "if Locked/Unlocked (or any multiple states) are declared, add the real "
                "release/re-lock transitions so no state is unreachable. Tie the default "
                "state to an entry action or explicit initial attribute value."
            )
        elif any(k in low for k in ("inhibit", "prevent", "block", "suppress", "shall not release",
                                     "shall not actuate", "lock out", "lockout")):
            # An inhibition requirement is satisfied by the transition that is
            # NOT taken. Repeated probe runs showed the surgical LLM adding a
            # response action for it instead, which the simulator cannot credit:
            # nothing fires. Name the shape that does anchor.
            default_guidance = (
                " This is an INHIBITION requirement: it is satisfied when a response is "
                "withheld while a condition holds. Model it as a Boolean attribute for the "
                "inhibiting condition on the owning part (e.g. deliveryAbortActive : Boolean "
                "= false) and EITHER a guard `if not <condition>` on the transition that "
                "would otherwise perform the response, OR a transition from the response "
                "state back to the safe/secured state that fires on the condition. Do not "
                "add a new response action; the anchor is the guarded or reverting "
                "transition, which the behavioural simulator exercises by sweeping the "
                "condition across true and false."
            )
        issues.append(
            f"{_ISSUE_PREFIX} {row.req_id} has no verification anchor at any tier — it "
            f"will land UNASSIGNED in the verification matrix. Requirement: \"{text}\". "
            "Fix inside the part def that satisfies it: add EITHER (a) an attribute or "
            "state-machine guard carrying the requirement's explicit threshold/default "
            "so the parameter linker can anchor it (configuration/safety semantics, "
            "e.g. a power-on default state or a numeric limit), OR (b) a state-machine "
            "transition whose accept event or guard implements the real trigger and whose "
            "reachable target-state entry action invokes the required response so the "
            "behavioural simulator can exercise it. An action declaration alone is not an "
            "anchor. Preserve trigger qualifiers such as valid-command and landing-completed; "
            "when the requirement says `within N seconds`, also model max/current latency "
            "attributes and an assert constraint carrying that bound. Do NOT invent physics calcs, do "
            "NOT add or remove requirement defs, and do NOT fake external evidence."
            f"{default_guidance}"
        )
    return issues[:limit]


_FUNC_GAP_RE = re.compile(r"\bREQ[_-]FUNC[_-]\d+\b", re.IGNORECASE)


def functional_verification_gap_issues(
    model_text: str,
    model_name: str,
    limit: int = 100,
    strict: bool = False,
    allowed_req_ids: Optional[Iterable[str]] = None,
    unmeasurable_req_ids: Optional[Iterable[str]] = None,
    planned_intents: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Model-fixable functional gaps that require a dedicated closure pass.

    ``verification_gap_issues`` already excludes external-measurement and
    downstream Phase-8 evidence. This view only selects FUNC rows, so the
    pipeline can make executable functional semantics a terminal gate without
    asking the LLM to fabricate HIL/field evidence.
    """
    return [
        issue for issue in verification_gap_issues(
            model_text, model_name, limit=limit, strict=strict,
            allowed_req_ids=allowed_req_ids,
            unmeasurable_req_ids=unmeasurable_req_ids,
            planned_intents=planned_intents,
        )
        if _FUNC_GAP_RE.search(issue)
    ]


def behavioral_result_regressed(before_sim, after_sim) -> bool:
    """True when a surgical anchor makes behavioral execution strictly worse."""
    before = getattr(before_sim, "behavioral_result", None)
    after = getattr(after_sim, "behavioral_result", None)
    if before is None:
        return bool(after and after.failed_scenarios())
    if after is None:
        return bool(before.scenario_results)
    return (
        after.sim_score + 1e-12 < before.sim_score
        or len(after.failed_scenarios()) > len(before.failed_scenarios())
    )


def is_verify_gap_issue(issue: object) -> bool:
    return str(issue).startswith(_ISSUE_PREFIX)
