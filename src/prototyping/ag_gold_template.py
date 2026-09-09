"""Draft evaluator-only gold generator for bounded A/G chains.

Produces a DRAFT gold artifact for supervisor review and freeze (candidate doc
§4/§5, design §13/§16). The draft comes from the student-approved Stage-2
decomposition (``ag_chains``), not from the runtime checker: this module imports
neither ``ag_extractor`` nor ``ag_contracts``, so gold cannot be a checker export
(finding F3). Until a supervisor freezes it the draft is not authoritative, and its
``EVALUATOR_GOLD`` role keeps it out of the pipeline (the ContextBuilder rejects
that role/topic).

With deterministic emission the draft verifies that the emit -> extract -> check
pipeline reproduces the selected decomposition (expected F1~1.0; it has caught
extractor/ordering bugs), which is not LLM accuracy. An LLM-authored intervention
needs its own frozen configuration/version and evidence gate. Gold cannot validate
the selected decomposition itself - that is the reviewer's job, which is why every
field carries a review note.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .ag_emitter import AGComponentSpec
from .experiment_arms import REVISED_EXPERIMENT_NAMESPACE
from .frozen_artifact_protocol import (
    has_review_markers,
    validate_frozen_envelope,
)

GOLD_ROLE = "EVALUATOR_GOLD"
GOLD_STATUS_DRAFT = "DRAFT_FOR_SUPERVISOR_REVIEW"
GOLD_STATUS_FROZEN = "FROZEN"
GOLD_SCHEMA_VERSION = "3.0"


def _gold_realization_paths(
    component: AGComponentSpec,
) -> list[dict[str, Any]]:
    paths = component.realization_paths
    if not paths:
        return [{
            "source": component.initial_state,
            "trigger": component.trigger_signal,
            "target": component.response_state,
            "guard": None,
            "action": component.response_action,
        }]
    return [
        {
            "source": path.source,
            "trigger": path.trigger,
            "target": path.target,
            "guard": path.guard,
            "action": path.action,
        }
        for path in paths
    ]


_REQUIRED_NAMESPACE = REVISED_EXPERIMENT_NAMESPACE


def _contains_key(obj: Any, target: str) -> bool:
    if isinstance(obj, Mapping):
        return any(
            str(key) == target or _contains_key(value, target)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(item, target) for item in obj)
    return False


# Which §13 Group B metric each gold fact family makes computable. Families are
# optional in the schema, so a freeze can omit one and still validate: the
# REQ_SAFE_005 freeze lacked `realization_links`, leaving a named primary metric
# uncomputable.
GOLD_METRIC_SUPPORT: Dict[str, str] = {
    "allocations": "guarantee allocation accuracy",
    "discharge_edges": "assumption-discharge P/R/F1",
    "realization_links": "guarantee-realization trace P/R/F1",
    "timing": "timing origin / deadline / apportionment agreement",
    "priority": "safety topology (priority) conformance",
    "invariants": "safety invariant conformance",
}


def validate_frozen_gold(gold: Dict[str, Any]) -> list[str]:
    """Return the freeze-completeness problems of a supervisor gold file.

    An empty list means the file passes every gate the post-hoc evaluator enforces
    (``FROZEN`` status, evaluator role/namespace, a named reviewer and date, both
    blind+independent review flags) and carries no leftover DRAFT review markers or
    unresolved ``by: null`` discharge edges. Structure only: it authors no gold values
    (F3), imports no checker and reads no prediction, so a supervisor can confirm a
    freeze is complete before the gold is pooled.
    """
    problems = validate_frozen_envelope(
        gold,
        role=GOLD_ROLE,
        schema_version=GOLD_SCHEMA_VERSION,
        namespace=_REQUIRED_NAMESPACE,
        review_flags=(
            "blind_to_runtime_verdict",
            "independent_human_review",
        ),
    )
    chain_id = str(gold.get("chain_id") or "")
    source_text = gold.get("source_text")
    if not isinstance(source_text, str) or not source_text:
        problems.append("source_text must be the immutable stakeholder text")
    if not gold.get("requirement_set_digest"):
        problems.append("requirement_set_digest must name the frozen requirement set")
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

    # A2 is part of the frozen evaluator contract, not optional reporting. The
    # selected bounded chains have explicit category coverage.
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
        source_kind = str(priority.get("source_kind") or "")
        source_id = str(priority.get("source_id") or "")
        members = [str(item) for item in (priority.get("members") or [])]
        edges = priority.get("edges")
        trigger = str(priority.get("trigger") or "")
        if not response_set_id:
            problems.append("priority response_set_id must be set")
        if not source_kind or not source_id:
            problems.append(
                "priority response-set provenance must set source_kind/source_id"
            )
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

    realization_links = gold.get("realization_links")
    if realization_links is not None:
        if not isinstance(realization_links, list) or not realization_links:
            problems.append("realization_links must be a non-empty list")
        else:
            seen_contracts: set[str] = set()
            for index, link in enumerate(realization_links):
                if not isinstance(link, Mapping):
                    problems.append(f"realization_links[{index}] must be an object")
                    continue
                contract = str(link.get("contract") or "")
                scalar = (
                    contract,
                    str(link.get("owner") or ""),
                    str(link.get("behavior") or ""),
                    str(link.get("initial_state") or ""),
                )
                if not all(scalar):
                    problems.append(
                        f"realization_links[{index}] must set "
                        "contract/owner/behavior/initial_state"
                    )
                if contract in seen_contracts:
                    problems.append(
                        f"realization_links contains duplicate contract {contract!r}"
                    )
                seen_contracts.add(contract)
                paths = link.get("response_paths")
                if not isinstance(paths, list) or not paths:
                    problems.append(
                        f"realization_links[{index}].response_paths must be a "
                        "non-empty list"
                    )
                    continue
                continuous = link.get("continuous_guarantee")
                if not isinstance(continuous, bool):
                    problems.append(
                        f"realization_links[{index}].continuous_guarantee "
                        "must be Boolean"
                    )
                seen_paths: set[tuple[str, str, str, str, str]] = set()
                for path_index, path in enumerate(paths):
                    if not isinstance(path, Mapping):
                        problems.append(
                            f"realization_links[{index}].response_paths"
                            f"[{path_index}] must be an object"
                        )
                        continue
                    key = (
                        str(path.get("source") or ""),
                        str(path.get("trigger") or ""),
                        str(path.get("target") or ""),
                        str(path.get("guard") or ""),
                        str(path.get("action") or ""),
                    )
                    if not key[0] or not key[2] or not key[4]:
                        problems.append(
                            f"realization_links[{index}].response_paths"
                            f"[{path_index}] must set source/target/action"
                        )
                    if key in seen_paths:
                        problems.append(
                            f"realization_links[{index}] contains duplicate "
                            f"response path {key!r}"
                        )
                    seen_paths.add(key)
                    if continuous is True and path.get("trigger") is not None:
                        problems.append(
                            f"realization_links[{index}] continuous guarantee "
                            "must use trigger=null"
                        )
                    if continuous is False and not str(path.get("trigger") or ""):
                        problems.append(
                            f"realization_links[{index}] event-driven path must "
                            "set trigger"
                        )

    observation_links = gold.get("observation_links")
    if observation_links is not None:
        if not isinstance(observation_links, list) or not observation_links:
            problems.append("observation_links must be a non-empty list")
        else:
            seen_observations: set[tuple[str, str, str]] = set()
            for index, link in enumerate(observation_links):
                if not isinstance(link, Mapping):
                    problems.append(f"observation_links[{index}] must be an object")
                    continue
                key = (
                    str(link.get("contract") or ""),
                    str(link.get("verification") or ""),
                    str(link.get("observation") or ""),
                )
                if not all(key):
                    problems.append(
                        f"observation_links[{index}] must set "
                        "contract/verification/observation"
                    )
                if key in seen_observations:
                    problems.append(f"duplicate observation link {key!r}")
                seen_observations.add(key)

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

    if has_review_markers(gold):
        problems.append(
            "leftover _review markers remain — drop them after confirming each field"
        )
    return problems


