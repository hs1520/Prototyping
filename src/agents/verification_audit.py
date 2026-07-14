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
the behavioural simulator can genuinely exercise. Requirements that need hardware/HIL
evidence (CEP, attitude RMS, RTCM corrections) will simply remain unassigned if the
LLM cannot anchor them truthfully — the audit creates the OPPORTUNITY for evidence,
never the evidence itself, and the surgical gates (requirement-def set frozen,
satisfy links may not shrink) prevent the fix from rewriting the spec instead.
"""
from __future__ import annotations

from typing import List, Optional

_ISSUE_PREFIX = "[VERIFY-GAP]"


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


def verification_gap_issues(model_text: str, model_name: str,
                            limit: int = 6) -> List[str]:
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
        rows = build_matrix(lite, None, linker)
    except Exception:
        return []
    issues: List[str] = []
    for row in rows:
        if row.status != "unassigned":
            continue
        text = " ".join((row.text or "").split())[:220]
        if _phase8_will_anchor(row.req_id, text):
            continue
        issues.append(
            f"{_ISSUE_PREFIX} {row.req_id} has no verification anchor at any tier — it "
            f"will land UNASSIGNED in the verification matrix. Requirement: \"{text}\". "
            "Fix inside the part def that satisfies it: add EITHER (a) an attribute or "
            "state-machine guard carrying the requirement's explicit threshold/default "
            "so the parameter linker can anchor it (configuration/safety semantics, "
            "e.g. a power-on default state or a numeric limit), OR (b) a state-machine "
            "transition whose guard implements the required behaviour so the "
            "behavioural simulator can exercise it. Do NOT invent physics calcs, do "
            "NOT add or remove requirement defs, and do NOT fake evidence for "
            "hardware-only criteria (CEP / attitude RMS / RTCM) — leaving those "
            "unassigned is the honest outcome."
        )
    return issues[:limit]


def is_verify_gap_issue(issue: object) -> bool:
    return str(issue).startswith(_ISSUE_PREFIX)
