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
            "owner": comp.owner_usage,
            "contract": comp.name,
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
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "status": GOLD_STATUS_DRAFT,
        "chain_id": normalise_req_id(spec.source_requirement),
        "source_requirement": spec.source_requirement,
        "source_text": source_text,
        "source_digest": _digest(source_text),
        "reviewer": None,
        "reviewed_date": None,
        "review_protocol": {
            "blind_to_runtime_verdict": False,
            "independent_human_review": False,
        },
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


_REQUIRED_NAMESPACE = "BLACKBOARD_AG_V1"


def _has_review_markers(obj: Any) -> bool:
    if isinstance(obj, dict):
        if any(
            str(key).startswith("_") and "review" in str(key) for key in obj
        ):
            return True
        return any(_has_review_markers(value) for value in obj.values())
    if isinstance(obj, list):
        return any(_has_review_markers(item) for item in obj)
    return False


def validate_frozen_gold(gold: Dict[str, Any]) -> list[str]:
    """Return the freeze-completeness problems of a supervisor gold file.

    An empty list means the file satisfies every gate the post-hoc evaluator
    enforces (``FROZEN`` status, evaluator role/namespace, a named reviewer and
    date, both blind+independent review flags) and carries no leftover DRAFT
    review markers or unresolved ``by: null`` discharge edges. This validates
    STRUCTURE only — it never authors or second-guesses gold values (F3), so it
    imports no checker and reads no prediction. It lets a supervisor confirm a
    freeze is complete before the gold is pooled.
    """
    problems: list[str] = []
    if gold.get("artifact_role") != GOLD_ROLE:
        problems.append(f"artifact_role must be {GOLD_ROLE!r}")
    if gold.get("experiment_namespace") != _REQUIRED_NAMESPACE:
        problems.append(f"experiment_namespace must be {_REQUIRED_NAMESPACE!r}")
    if gold.get("status") != GOLD_STATUS_FROZEN:
        problems.append(
            f"status must be {GOLD_STATUS_FROZEN!r} (still a draft?)"
        )
    if not gold.get("reviewer"):
        problems.append("reviewer must be set to the reviewing supervisor")
    if not gold.get("reviewed_date"):
        problems.append("reviewed_date must be set")
    review = gold.get("review_protocol") or {}
    if review.get("blind_to_runtime_verdict") is not True:
        problems.append("review_protocol.blind_to_runtime_verdict must be true")
    if review.get("independent_human_review") is not True:
        problems.append("review_protocol.independent_human_review must be true")
    if not gold.get("allocations"):
        problems.append("allocations must be non-empty")
    for edge in gold.get("discharge_edges") or []:
        if edge.get("by") is None:
            problems.append(
                f"discharge edge {edge.get('component')}/"
                f"{edge.get('assumption')} is unresolved (by=null)"
            )
    if _has_review_markers(gold):
        problems.append(
            "leftover _review markers remain — drop them after confirming each field"
        )
    return problems


def frozen_gold_gate(
    requirements, *, gold_dir: str = "docs/gold"
) -> list[str]:
    """Problems that must be empty before R2-BBAG accuracy may be pooled.

    Design P2: ``evaluation_ready`` is a whole-arm state, so it may open only when
    **every** chain selected for the run has an independent FROZEN gold on disk —
    never on a single frozen file. Returns the per-chain problems (missing frozen
    file, or a file that fails :func:`validate_frozen_gold`); empty ⇒ the gate is
    clear for these requirements. Reads gold files only; authors nothing (F3).
    """
    import json
    from pathlib import Path

    from .ag_chains import select_ag_chains

    chains = select_ag_chains(requirements)
    if not chains:
        return ["no reviewed A/G chain is selected for these requirements"]
    problems: list[str] = []
    for chain in chains:
        req = normalise_req_id(chain.source_requirement)
        path = Path(gold_dir) / f"{req}_ag_gold.json"
        if not path.exists():
            problems.append(
                f"{req}: no FROZEN gold at {path} (a .draft is not sufficient)"
            )
            continue
        try:
            gold = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{req}: gold file unreadable ({exc})")
            continue
        problems.extend(f"{req}: {issue}" for issue in validate_frozen_gold(gold))
    return problems
