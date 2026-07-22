"""Deterministic verification-planning knowledge source (design §15, 2nd handoff).

The VerificationAgent is the second board-mediated knowledge source. It consumes
the committed SysML model (the sole authority) that the DesignAgent produced and
the authoritative requirements, and produces a per-requirement verification
*plan*: each stakeholder ``requirement def`` in the committed model is traced and
assigned a planned IADT method (Inspection / Analysis / Demonstration / Test) from
its own text. This is planning, not execution — the SITL / Gazebo / analysis tiers
that actually run are downstream (``verification_matrix``). It reads only the
committed model; it never loads evaluator gold and it makes no LLM call, so the
handoff is reproducible.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..utils.req_id import normalise_req_id
from ..utils.sysml_text_utils import find_block_end

VERIFICATION_PLAN_ROLE = "VERIFICATION_PLAN"

_REQ_DEF_RE = re.compile(r"\brequirement\s+def\s+(REQ[_-][A-Za-z0-9]+[_-]\d+)\s*\{")
_DOC_RE = re.compile(r"\bdoc\s*/\*(.*?)\*/", re.DOTALL)
# A physical quantity: a number followed by an engineering unit. Deliberately
# unit-anchored so a bare id number (REQ-SAFE-005) never reads as a threshold.
_QUANTITY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:s|sec|secs|second|seconds|ms|m|metre|metres|meter|meters|km|"
    r"m/s|km/h|%|hz|khz|kg|g|v|volt|volts|a|amp|amps|min|minute|minutes|"
    r"degrees?|deg|rpm)\b",
    re.IGNORECASE,
)
_INVARIANT_RE = re.compile(
    r"\b(?:shall|must|will|do|does|can|could|may)\s+not\b|\bcannot\b|\bnever\b|"
    r"\bonly\b|\bremain\b|\bprevent\b|\binhibit\b|\blocked\b",
    re.IGNORECASE,
)


def requirement_def_slice(model_text: str) -> str:
    """Return only the stakeholder ``requirement def REQ_*`` blocks of the model.

    This is the relevant, complete context for verification planning: it carries
    every requirement to plan without the rest of the model, so the envelope stays
    small and no requirement is lost to token-budget truncation.
    """
    text = model_text or ""
    blocks: List[str] = []
    for match in _REQ_DEF_RE.finditer(text):
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        if end != -1:
            blocks.append(text[match.start():end + 1])
    return "\n".join(blocks)


def _plan_method(text: str) -> Dict[str, str]:
    """Assign one planned IADT method + tier from the requirement text."""
    body = " ".join((text or "").split())
    if _QUANTITY_RE.search(body):
        return {
            "planned_method": "Analysis/Test (quantified threshold)",
            "planned_tier": "analysis_or_sitl",
            "rationale": "carries a measurable physical threshold",
        }
    if _INVARIANT_RE.search(body):
        return {
            "planned_method": "Demonstration/Inspection (state invariant)",
            "planned_tier": "behavioral_or_inspection",
            "rationale": "a Boolean state/ordering invariant to demonstrate",
        }
    return {
        "planned_method": "Inspection",
        "planned_tier": "inspection",
        "rationale": "qualitative requirement, no measurable threshold",
    }


def plan_verification(model_text: str) -> Dict[str, Any]:
    """Build a per-requirement verification plan from the committed model.

    Every stakeholder ``requirement def REQ_*`` is traced (it is present in the
    committed model by construction) and assigned a planned method. A/G contract
    defs (``System*Contract`` etc.) are not stakeholder requirements and are
    skipped — only ``REQ_``-prefixed definitions are planned.
    """
    text = model_text or ""
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for match in _REQ_DEF_RE.finditer(text):
        req_id = normalise_req_id(match.group(1))
        if req_id in seen:
            continue
        seen.add(req_id)
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        block = text[brace + 1:end] if end != -1 else ""
        doc = _DOC_RE.search(block)
        doc_text = " ".join(doc.group(1).split()) if doc else ""
        plan = _plan_method(doc_text)
        entries.append({
            "requirement": req_id,
            "traceable_in_model": True,
            "has_doc": bool(doc_text),
            "source_text": doc_text,
            **plan,
        })

    tiers: Dict[str, int] = {}
    for entry in entries:
        tiers[entry["planned_tier"]] = tiers.get(entry["planned_tier"], 0) + 1
    return {
        "schema_version": "1.0",
        "artifact_role": VERIFICATION_PLAN_ROLE,
        "producing_stage": "R1_VERIFICATION_PLANNING",
        "measurement_boundary": "COORDINATION",
        "semantic_authority": "COMMITTED_SYSML_MODEL",
        "planned": len(entries),
        "tier_histogram": tiers,
        "entries": entries,
    }
