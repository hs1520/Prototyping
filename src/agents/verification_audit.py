"""Static verification-readiness audit - feeds matrix `unassigned` gaps back into refinement.

The verification matrix runs at the END of the pipeline, so requirements that end
up ``unassigned`` are found after generation and never flow back to the model.
``build_matrix()`` with no execution results already computes the linker/text tiers
(execution results only upgrade planned tiers to verified; they never create a
first anchor), and the one anchor source unavailable at refinement time is Phase 8,
so requirements whose quantified family Phase 8 will decide are excluded statically
via requirement_spec. Each remaining unassigned requirement becomes a
surgical-refinement issue asking only for model anchors the parameter linker or the
behavioural simulator can exercise; requirements needing external measurement (CEP,
attitude RMS, RTCM corrections) stay visibly unassigned in the matrix, and the
surgical gates (requirement-def set frozen, satisfy links may not shrink) stop a
repair from rewriting the spec.
"""
from __future__ import annotations

import re
from typing import Mapping, Iterable, List, Optional

from ..prototyping.verification_obligations import is_inhibition_requirement
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

_INHIBITION_REPAIR_GUIDANCE = (
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


def _requires_external_evidence(text: str) -> bool:
    """True for criteria a SysML edit cannot verify.

    Keys on the measured quantity, not a generic ``HIL`` tag: a behavioural
    requirement can still gain an executable model anchor even when HIL is its
    eventual acceptance method.
    """
    return any(pattern.search(text or "") for pattern in _EXTERNAL_EVIDENCE_PATTERNS)


def _phase8_will_anchor(req_id: str, text: str) -> bool:
    """True when Phase 8 (datasheet/forward-flight verdicts) anchors this requirement.

    Decidable statically from the requirement text via requirement_spec, so the
    audit does not flag it as a gap at refinement time.
    """
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
    planned_markers: Optional[Mapping[str, Iterable[str]]] = None,
) -> List[str]:
    """Return surgical-refinement issues for requirements no verification tier anchors.

    Best-effort: any parsing/linker failure returns [], so the audit does not break
    the refinement loop.
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
            planned_markers=(
                {k: frozenset(v) for k, v in planned_markers.items()}
                if planned_markers else None
            ),
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
            # Two signals agree there is nothing to anchor: the planner recorded no obliged
            # discrete response and the extractor found no measurable criterion. A surgical
            # repair would have to invent an anchor the requirement does not provide. The
            # row stays UNASSIGNED as a requirement-side gap and does not block closure.
            continue
        if not behavioral_failed and "planned_unverifiable_response" in row.tiers:
            # A discrete response is obliged but no reachable-action marker can evidence
            # it. No surgical repair makes the gate check what its vocabulary cannot
            # express, so asking for one fail-closes the run with no way out. The row stays
            # UNASSIGNED under its own tier: a gate-capability gap, not a model-side one.
            # The record is not free - plan validation demands a rationale, and declared
            # response_markers remain the route for any response a reachable action can
            # evidence.
            continue
        text = " ".join((row.text or "").split())[:220]
        if _phase8_will_anchor(row.req_id, text):
            continue
        if not behavioral_failed and _requires_external_evidence(text):
            # The row stays UNASSIGNED in the matrix; the surgical LLM is not asked to
            # invent model evidence for a measured hardware/HIL quantity.
            continue
        if behavioral_failed:
            repair_guidance = (
                _INHIBITION_REPAIR_GUIDANCE
                if is_inhibition_requirement(text) else ""
            )
            issues.append(
                f"{_ISSUE_PREFIX} {row.req_id} has a behavioral verification anchor, "
                f"but its executable model-level scenario FAILS. Requirement: \"{text}\". "
                "Repair the complete satisfying part def; do not add another declaration-only "
                "state machine. Every declared state must be reachable, the initial/default "
                "state must agree with its Boolean attribute, and the required transition or "
                "entry action must actually execute under simulation."
                f"{repair_guidance}"
            )
            continue
        default_guidance = ""
        low = text.lower()
        # Inhibition phrasing wins over power-on/default keywords: "shall not
        # transition ... during the power-on self-test" names the phase, not a default
        # state, and the matrix routes it the same way via is_inhibition_requirement.
        if not is_inhibition_requirement(low) and any(
            k in low
            for k in ("power-on", "power on", "default", "startup", "start-up")
        ):
            default_guidance = (
                " For a power-on/default-state requirement, model an explicit initial state "
                "and a consistent Boolean attribute. A single-state invariant is allowed; "
                "if Locked/Unlocked (or any multiple states) are declared, add the real "
                "release/re-lock transitions so no state is unreachable. Tie the default "
                "state to an entry action or explicit initial attribute value."
            )
        elif is_inhibition_requirement(low):
            # An inhibition requirement is satisfied by the transition not taken. Probe
            # runs showed the surgical LLM adding a response action instead, which the
            # simulator cannot credit because nothing fires. Name the shape that anchors.
            default_guidance = _INHIBITION_REPAIR_GUIDANCE
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
    planned_markers: Optional[Mapping[str, Iterable[str]]] = None,
) -> List[str]:
    """Model-fixable functional gaps that need a dedicated closure pass.

    ``verification_gap_issues`` already drops external-measurement and downstream
    Phase-8 evidence; this view keeps only FUNC rows, so executable functional
    semantics can be a terminal gate without asking for HIL/field evidence.
    """
    return [
        issue for issue in verification_gap_issues(
            model_text, model_name, limit=limit, strict=strict,
            allowed_req_ids=allowed_req_ids,
            unmeasurable_req_ids=unmeasurable_req_ids,
            planned_intents=planned_intents,
            planned_markers=planned_markers,
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
