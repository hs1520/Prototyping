"""Draft evaluator-only gold generator for bounded A/G chains.

Produces a DRAFT gold artifact for supervisor review and freeze (candidate doc
§4/§5, design §13/§16). The draft is derived from the student-approved Stage-2
decomposition (``ag_chains``), NOT from the runtime checker: this module imports
neither ``ag_extractor`` nor ``ag_contracts``, so gold can never be a checker
export (finding F3). Until a supervisor confirms and freezes it, the draft is not
authoritative, and its ``EVALUATOR_GOLD`` role keeps it out of the pipeline (the
ContextBuilder rejects that role/topic).

What the draft measures: with deterministic emission it verifies that the
emit → extract → check pipeline faithfully reproduces the selected decomposition
(expected F1≈1.0, and it already caught extractor/ordering bugs). Under the current
deterministic intervention this must not be called LLM accuracy. An LLM-authored
intervention would require its own frozen configuration/version and evidence gate.
Gold cannot independently validate the selected decomposition itself — that is
the human reviewer's responsibility, which is why every field carries a review note.
"""
from __future__ import annotations

from datetime import date
import hashlib
import re
from typing import Any, Dict, Mapping

from .ag_emitter import AGChainSpec
from ..utils.req_id import normalise_req_id

GOLD_ROLE = "EVALUATOR_GOLD"
GOLD_STATUS_DRAFT = "DRAFT_FOR_SUPERVISOR_REVIEW"
GOLD_STATUS_FROZEN = "FROZEN"
GOLD_SCHEMA_VERSION = "3.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _valid_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_gold_draft(
    spec: AGChainSpec,
    *,
    source_text: str,
    requirement_set_digest: str | None = None,
    architecture_boundary_digest: str | None = None,
) -> Dict[str, Any]:
    """Build a review-ready DRAFT from a student-approved A/G decomposition.

    Allocations and discharge edges are pre-filled from the decomposition intent so
    the reviewer confirms rather than transcribes; an assumption that is neither an
    environment/system assumption nor produced by an upstream guarantee is left
    ``by = None`` and flagged UNRESOLVED for the reviewer.
    """
    producers = {
        guarantee: comp.name
        for comp in spec.components
        for guarantee in comp.guarantees
    }
    system_env = set(spec.system_assumptions)

    allocations = [
        {
            "owner": comp.owner_usage,
            "contract": comp.name,
            "guarantee": guarantee,
            "_review": "confirm the responsible owner",
        }
        for comp in spec.components
        for guarantee in comp.guarantees
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

    semantic_fields: Dict[str, Any] = {}
    if spec.deadline is not None and spec.timing_origin:
        semantic_fields["timing"] = {
            "origin": spec.timing_origin,
            "deadline": {
                "value": format(spec.deadline, ".15g"),
                "unit": "s",
            },
            "segments": [
                {
                    "component": comp.name.removesuffix("Contract"),
                    "budget": {
                        "value": format(comp.latency_budget, ".15g"),
                        "unit": "s",
                    },
                }
                for comp in spec.components
                if comp.latency_budget is not None
                and comp.timing_segment_required is not False
            ],
        }
    if spec.priority is not None:
        semantic_fields["priority"] = {
            "response_set_id": spec.priority.response_set_id,
            "members": list(spec.priority.members),
            "edges": [
                {"higher": higher, "lower": lower}
                for higher, lower in spec.priority.edges
            ],
            "trigger": spec.priority.trigger,
        }
    if spec.invariants:
        semantic_fields["selected_model_elements"] = list(
            spec.selected_model_elements
        )
        semantic_fields["invariants"] = [
            {
                "invariant_id": item.invariant_id,
                "scope": item.scope,
                "trigger_or_antecedent_ast": dict(
                    item.trigger_or_antecedent_ast
                ),
                "required_consequent_ast": dict(
                    item.required_consequent_ast
                ),
                "source_kind": item.source_kind,
                "source_id": item.source_id,
            }
            for item in spec.invariants
        ]

    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "artifact_role": GOLD_ROLE,
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "status": GOLD_STATUS_DRAFT,
        "chain_id": normalise_req_id(spec.source_requirement),
        "source_requirement": spec.source_requirement,
        "source_text": source_text,
        "source_digest": _digest(source_text),
        "requirement_set_digest": requirement_set_digest,
        "architecture_boundary_digest": architecture_boundary_digest,
        "_provenance_review": (
            "bind the frozen requirement-set and independently frozen architecture "
            "boundary SHA-256 digests before freeze"
        ),
        "reviewer": None,
        "reviewed_date": None,
        "review_protocol": {
            "blind_to_runtime_verdict": False,
            "independent_human_review": False,
        },
        "review_instructions": (
            "Label blind to any pipeline verdict (design §13). Confirm or edit each "
            "field from the source requirement and the student-approved decomposition "
            "candidate, drop "
            "the _review notes, set reviewer/reviewed_date, and change status to "
            f"{GOLD_STATUS_FROZEN!r}. This artifact is evaluator-only: never feed it "
            "into generation, context, checking, routing, or repair."
        ),
        "system": {
            "contract": spec.system_contract,
            "assumptions": list(spec.system_assumptions),
            "guarantee_observation": spec.observation,
            "_review": "confirm system assumptions and observation",
        },
        "allocations": allocations,
        "discharge_edges": discharge_edges,
        **semantic_fields,
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


def _contains_key(obj: Any, target: str) -> bool:
    """Return whether a forbidden semantic key occurs anywhere in an artifact."""
    if isinstance(obj, Mapping):
        return any(
            str(key) == target or _contains_key(value, target)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(item, target) for item in obj)
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
    if gold.get("schema_version") != GOLD_SCHEMA_VERSION:
        problems.append(f"schema_version must be {GOLD_SCHEMA_VERSION!r}")
    if gold.get("artifact_role") != GOLD_ROLE:
        problems.append(f"artifact_role must be {GOLD_ROLE!r}")
    if gold.get("experiment_namespace") != _REQUIRED_NAMESPACE:
        problems.append(f"experiment_namespace must be {_REQUIRED_NAMESPACE!r}")
    if gold.get("status") != GOLD_STATUS_FROZEN:
        problems.append(
            f"status must be {GOLD_STATUS_FROZEN!r} (still a draft?)"
        )
    chain_id = str(gold.get("chain_id") or "")
    source_requirement = str(gold.get("source_requirement") or "")
    if not chain_id:
        problems.append("chain_id must be set")
    if not source_requirement:
        problems.append("source_requirement must be set")
    elif chain_id and normalise_req_id(source_requirement) != chain_id:
        problems.append("chain_id must match the normalised source_requirement")
    source_text = gold.get("source_text")
    if not isinstance(source_text, str) or not source_text:
        problems.append("source_text must be the immutable stakeholder text")
    elif gold.get("source_digest") != _digest(source_text):
        problems.append("source_digest does not match source_text")
    for field in ("requirement_set_digest", "architecture_boundary_digest"):
        if not _SHA256_RE.fullmatch(str(gold.get(field) or "")):
            problems.append(f"{field} must be a lowercase SHA-256 digest")
    if not gold.get("reviewer"):
        problems.append("reviewer must be set to the reviewing supervisor")
    if not _valid_iso_date(gold.get("reviewed_date")):
        problems.append("reviewed_date must be ISO YYYY-MM-DD")
    review = gold.get("review_protocol")
    if not isinstance(review, Mapping):
        problems.append("review_protocol must be an object")
        review = {}
    if review.get("blind_to_runtime_verdict") is not True:
        problems.append("review_protocol.blind_to_runtime_verdict must be true")
    if review.get("independent_human_review") is not True:
        problems.append("review_protocol.independent_human_review must be true")
    allocations = gold.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        problems.append("allocations must be a non-empty list")
        allocations = []
    allocation_keys: set[tuple[str, str, str]] = set()
    for index, allocation in enumerate(allocations):
        if not isinstance(allocation, Mapping):
            problems.append(f"allocations[{index}] must be an object")
            continue
        key = (
            str(allocation.get("owner") or ""),
            str(allocation.get("contract") or ""),
            str(allocation.get("guarantee") or ""),
        )
        if not all(key):
            problems.append(
                f"allocations[{index}] must set owner/contract/guarantee"
            )
        if key in allocation_keys:
            problems.append(f"duplicate allocation {key!r}")
        allocation_keys.add(key)
    if _contains_key(gold, "failure_class"):
        problems.append(
            "failure_class must not appear in reference gold; use a per-run blind label"
        )
    if _contains_key(gold, "deadline_s"):
        problems.append(
            "deadline_s must not appear in gold; timing authority is the atomic "
            "decimal-string quantity in timing.deadline"
        )
    discharge_edges = gold.get("discharge_edges")
    if not isinstance(discharge_edges, list) or not discharge_edges:
        problems.append("discharge_edges must be a non-empty list")
        discharge_edges = []
    discharge_keys: set[tuple[str, str, str]] = set()
    for index, edge in enumerate(discharge_edges):
        if not isinstance(edge, Mapping):
            problems.append(f"discharge_edges[{index}] must be an object")
            continue
        key = (
            str(edge.get("component") or ""),
            str(edge.get("assumption") or ""),
            str(edge.get("by") or ""),
        )
        if not key[0] or not key[1]:
            problems.append(
                f"discharge_edges[{index}] must set component/assumption"
            )
        if edge.get("by") is None or not key[2]:
            problems.append(
                f"discharge edge {edge.get('component')}/"
                f"{edge.get('assumption')} is unresolved (by=null)"
            )
        if key in discharge_keys:
            problems.append(f"duplicate discharge edge {key!r}")
        discharge_keys.add(key)

    # A2 is part of the frozen evaluator contract, not an optional reporting
    # decoration.  The selected bounded chains have explicit category coverage.
    required_categories = {
        "REQ_SAFE_004": ("invariants",),
        "REQ_SAFE_005": ("timing", "priority"),
        "REQ_SAFE_008": ("invariants",),
    }.get(chain_id, ())
    for category in required_categories:
        if gold.get(category) is None:
            problems.append(f"{chain_id}: required A2 category {category!r} is missing")

    timing = gold.get("timing")
    if timing is not None and not isinstance(timing, Mapping):
        problems.append("timing must be an object")
    elif isinstance(timing, Mapping):
        if not str(timing.get("origin") or ""):
            problems.append("timing origin must be set")
        if not isinstance(timing.get("deadline"), Mapping):
            problems.append("timing deadline must be an atomic quantity object")
        if not isinstance(timing.get("segments"), list) or not timing.get("segments"):
            problems.append("timing segments must be a non-empty ordered list")
        forbidden_derived = {
            "additive_total",
            "additive_total_s",
            "within_deadline",
            "computed",
        }
        present = {
            key for key in forbidden_derived if _contains_key(timing, key)
        }
        if present:
            problems.append(
                "timing gold contains evaluator-derived fields: "
                + ", ".join(sorted(present))
            )
        try:
            from .ag_eval_semantics import timing_agreement
            timing_agreement(timing, timing)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"timing gold is invalid: {exc}")

    priority = gold.get("priority")
    if priority is not None and not isinstance(priority, Mapping):
        problems.append("priority must be an object")
    elif isinstance(priority, Mapping):
        response_set_id = str(priority.get("response_set_id") or "")
        members = [str(item) for item in (priority.get("members") or [])]
        edges = priority.get("edges")
        trigger = str(priority.get("trigger") or "")
        if not response_set_id:
            problems.append("priority response_set_id must be set")
        if (
            not members
            or any(not item for item in members)
            or len(members) != len(set(members))
        ):
            problems.append("priority members must be non-empty strings and unique")
        if not isinstance(edges, list) or not edges:
            problems.append("priority edges must be a non-empty list")
        else:
            edge_pairs = [
                (str(item.get("higher") or ""), str(item.get("lower") or ""))
                for item in edges if isinstance(item, Mapping)
            ]
            if (
                len(edge_pairs) != len(edges)
                or any(not all(pair) for pair in edge_pairs)
                or len(edge_pairs) != len(set(edge_pairs))
            ):
                problems.append("priority edges must be complete and unique")
            elif any(a not in members or b not in members for a, b in edge_pairs):
                problems.append("priority edges must reference response-set members")
            elif any(a == b for a, b in edge_pairs):
                problems.append("priority precedence edges must not be self-loops")
            else:
                adjacency = {
                    member: {
                        lower for higher, lower in edge_pairs if higher == member
                    }
                    for member in members
                }

                def reaches_cycle(node: str, visiting: set[str], done: set[str]) -> bool:
                    if node in visiting:
                        return True
                    if node in done:
                        return False
                    visiting.add(node)
                    if any(
                        reaches_cycle(child, visiting, done)
                        for child in adjacency.get(node, ())
                    ):
                        return True
                    visiting.remove(node)
                    done.add(node)
                    return False

                done: set[str] = set()
                if any(reaches_cycle(member, set(), done) for member in members):
                    problems.append("priority precedence edges must be acyclic")
        if not trigger:
            problems.append("priority trigger must be set")

    invariants = gold.get("invariants")
    if invariants is not None:
        if not isinstance(invariants, list) or not invariants:
            problems.append("invariants must be a non-empty list")
        selected = gold.get("selected_model_elements")
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(item, str) or not item for item in selected)
            or len(selected) != len(set(selected))
        ):
            problems.append(
                "invariant gold requires unique non-empty selected_model_elements"
            )
        try:
            from .ag_eval_semantics import invariant_agreement
            invariant_agreement(gold, gold)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"invariant gold is invalid: {exc}")

    if _has_review_markers(gold):
        problems.append(
            "leftover _review markers remain — drop them after confirming each field"
        )
    return problems


def frozen_gold_gate(
    requirements, *, gold_dir: str = "docs/gold"
) -> list[str]:
    """Legacy convenience check for frozen gold coverage.

    This function still selects chains from live code, so an empty result is *not*
    authority to pool. It remains for draft/freeze diagnostics and compatibility.
    The sole pooling decision is
    :func:`evaluation_readiness.build_evaluation_readiness_manifest`, which binds
    a frozen experiment configuration, requirement and architecture digests, the
    complete selected chain/run sets, and blind labels.
    """
    import json
    from pathlib import Path

    from .ag_chains import select_ag_chains

    chains = select_ag_chains(requirements)
    if not chains:
        return ["no bounded A/G chain is selected for these requirements"]
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
