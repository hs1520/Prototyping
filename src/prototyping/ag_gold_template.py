"""Draft evaluator-only gold generator for bounded A/G chains.

Produces a DRAFT gold artifact for supervisor review and freeze (candidate doc
§4/§5, design §13/§16). The draft is derived from the reviewed Stage-2
decomposition (``ag_chains``), NOT from the runtime checker: this module imports
neither ``ag_extractor`` nor ``ag_contracts``, so gold can never be a checker
export (finding F3). Until a supervisor confirms and freezes it, the draft is not
authoritative, and its ``EVALUATOR_GOLD`` role keeps it out of the pipeline (the
ContextBuilder rejects that role/topic).

What the draft measures: with deterministic emission it verifies that the
emit → extract → check pipeline faithfully reproduces the reviewed decomposition
(expected F1≈1.0, and it already caught two real extractor/ordering bugs). The
same gold measures LLM A/G accuracy directly once generation is LLM-driven. Gold
cannot second-guess the reviewed decomposition itself — that is the human
reviewer's responsibility, which is why every field carries a review note.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict

from .ag_emitter import AGChainSpec
from .ag_evaluation import GOLD_ROLE
from ..utils.req_id import normalise_req_id

GOLD_STATUS_DRAFT = "DRAFT_FOR_SUPERVISOR_REVIEW"
GOLD_STATUS_FROZEN = "FROZEN"


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_gold_draft(
    spec: AGChainSpec,
    *,
    source_text: str,
    failure_class: str = "NO_FAILURE",
) -> Dict[str, Any]:
    """Build a review-ready DRAFT gold dict from a reviewed A/G chain decomposition.

    Allocations and discharge edges are pre-filled from the decomposition intent so
    the reviewer confirms rather than transcribes; an assumption that is neither an
    environment/system assumption nor produced by an upstream guarantee is left
    ``by = None`` and flagged UNRESOLVED for the reviewer.
    """
    producers = {comp.guarantee: comp.name for comp in spec.components}
    system_env = set(spec.system_assumptions)

    allocations = [
        {
            "owner": comp.name,
            "guarantee": comp.guarantee,
            "_review": "confirm the responsible owner",
        }
        for comp in spec.components
    ]

    discharge_edges = []
    for comp in spec.components:
        for assumption in comp.assumptions:
            if assumption.environment or assumption.concept in system_env:
                by = "environment"
            else:
                by = producers.get(assumption.concept)
            edge = {"component": comp.name, "assumption": assumption.concept, "by": by}
            edge["_review"] = (
                "UNRESOLVED — reviewer must set the discharging source"
                if by is None else "confirm the discharging source"
            )
            discharge_edges.append(edge)

    return {
        "artifact_role": GOLD_ROLE,
        "status": GOLD_STATUS_DRAFT,
        "chain_id": normalise_req_id(spec.source_requirement),
        "source_requirement": spec.source_requirement,
        "source_text": source_text,
        "source_digest": _digest(source_text),
        "reviewer": None,
        "reviewed_date": None,
        "review_instructions": (
            "Label blind to any pipeline verdict (design §13). Confirm or edit each "
            "field from the source requirement and the reviewed decomposition, drop "
            "the _review notes, set reviewer/reviewed_date, and change status to "
            f"{GOLD_STATUS_FROZEN!r}. This artifact is evaluator-only: never feed it "
            "into generation, context, checking, routing, or repair."
        ),
        "system": {
            "contract": spec.system_contract,
            "assumptions": list(spec.system_assumptions),
            "guarantee_observation": spec.observation,
            "deadline_s": spec.deadline,
            "_review": "confirm system assumptions, observation, and deadline",
        },
        "allocations": allocations,
        "discharge_edges": discharge_edges,
        "failure_class": failure_class,
        "_failure_class_review": "confirm the expected blind failure class",
    }
