"""Immutable stakeholder requirement-input boundary.

Controlled experiments vary generation, not the requirement set, so this module
builds the frozen requirement artifact and the dependency graph used for change
propagation between runs.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from ..utils.digest import sha256_text
from ..utils.req_id import first_req_id, normalise_req_id


FROZEN_REQUIREMENT_SCHEMA_VERSION = "1.0"


def normalise_requirement_id(value: str) -> str:
    return normalise_req_id(first_req_id(value) or value)


def requirement_records(requirements: Iterable[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(requirements):
        text = str(raw)
        req_id = normalise_requirement_id(text)
        if first_req_id(text) is None:
            raise ValueError(f"frozen requirement[{index}] has no valid requirement id")
        if req_id in seen:
            raise ValueError(f"frozen requirements contain duplicate id {req_id}")
        seen.add(req_id)
        records.append({"req_id": req_id, "source_text": text})
    if not records:
        raise ValueError("frozen requirement set must not be empty")
    return records


def requirement_set_digest(requirements: Iterable[str]) -> str:
    """Order-sensitive digest of exact IDs and stakeholder-owned source text."""
    records = requirement_records(requirements)
    payload = json.dumps(
        [(item["req_id"], item["source_text"]) for item in records],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def build_requirement_dependency_graph(
    requirements: Iterable[str],
    dependencies: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the persisted requirement graph used for change propagation."""
    records = requirement_records(requirements)
    req_ids = {item["req_id"] for item in records}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for dependency in dependencies:
        dependent = normalise_requirement_id(str(dependency.get("from") or ""))
        prerequisite = normalise_requirement_id(str(dependency.get("to") or ""))
        if dependent not in req_ids or prerequisite not in req_ids:
            raise ValueError(
                f"requirement dependency has unknown endpoint: "
                f"{dependent} depends on {prerequisite}"
            )
        if dependent == prerequisite:
            raise ValueError(f"requirement dependency is self-referential: {dependent}")
        edge = (dependent, prerequisite)
        if edge not in seen:
            seen.add(edge)
            edges.append({"from": dependent, "to": prerequisite})
    graph = {
        "schema_version": "1.0",
        "artifact_type": "REQUIREMENT_DEPENDENCY_GRAPH",
        "nodes": [
            {"req_id": item["req_id"], "source_text": item["source_text"]}
            for item in sorted(records, key=lambda value: value["req_id"])
        ],
        "edges": sorted(edges, key=lambda value: (value["from"], value["to"])),
    }
    return graph


def requirement_change_impact(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return requirements whose evidence is invalid after a graph change."""
    old_graph = (previous or {}).get("dependency_graph") or {"nodes": [], "edges": []}
    new_graph = current.get("dependency_graph") or {"nodes": [], "edges": []}
    old_nodes = {
        str(item.get("req_id")): str(item.get("source_text"))
        for item in old_graph.get("nodes", ())
    }
    new_nodes = {
        str(item.get("req_id")): str(item.get("source_text"))
        for item in new_graph.get("nodes", ())
    }
    directly_changed = {
        req_id for req_id in set(old_nodes) | set(new_nodes)
        if old_nodes.get(req_id) != new_nodes.get(req_id)
    }
    old_edges = {
        (str(item.get("from")), str(item.get("to")))
        for item in old_graph.get("edges", ())
    }
    new_edges = {
        (str(item.get("from")), str(item.get("to")))
        for item in new_graph.get("edges", ())
    }
    changed_edges = old_edges ^ new_edges
    directly_changed.update(dependent for dependent, _ in changed_edges)

    dependents: dict[str, set[str]] = {}
    for dependent, prerequisite in old_edges | new_edges:
        dependents.setdefault(prerequisite, set()).add(dependent)
    impacted = set(directly_changed)
    pending = list(directly_changed)
    while pending:
        for dependent in dependents.get(pending.pop(), ()):
            if dependent not in impacted:
                impacted.add(dependent)
                pending.append(dependent)
    return {
        "schema_version": "1.0",
        "artifact_type": "REQUIREMENT_CHANGE_IMPACT",
        "baseline_requirement_set_digest": (previous or {}).get(
            "requirement_set_digest"
        ),
        "current_requirement_set_digest": current.get("requirement_set_digest"),
        "directly_changed_requirement_ids": sorted(directly_changed),
        "invalidated_requirement_ids": sorted(impacted),
        "evidence_invalidations": [
            {"req_id": req_id, "status": "STALE"}
            for req_id in sorted(impacted)
        ],
        "dependency_edges_changed": bool(changed_edges),
    }


def build_frozen_requirement_set(
    requirements: Iterable[str],
    *,
    name: str = "frozen-requirements",
    source: str = "manual",
    dependencies: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    items = list(requirements)
    records = requirement_records(items)
    return {
        "schema_version": FROZEN_REQUIREMENT_SCHEMA_VERSION,
        "artifact_type": "FROZEN_REQUIREMENT_SET",
        "name": name,
        "source": source,
        "requirement_set_digest": requirement_set_digest(items),
        "requirements": [item["source_text"] for item in records],
        "dependency_graph": build_requirement_dependency_graph(
            items, dependencies
        ),
    }


def resolve_frozen_requirement_set(
    value: Iterable[str] | Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate a frozen artifact or wrap an explicit list as one."""
    if isinstance(value, Mapping):
        if value.get("artifact_type") != "FROZEN_REQUIREMENT_SET":
            raise ValueError("frozen input must have artifact_type=FROZEN_REQUIREMENT_SET")
        if value.get("schema_version") != FROZEN_REQUIREMENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported frozen requirement schema_version: "
                f"{value.get('schema_version')!r}"
            )
        requirements = [str(item) for item in value.get("requirements", ())]
        canonical = build_frozen_requirement_set(
            requirements,
            name=str(value.get("name") or "frozen-requirements"),
            source=str(value.get("source") or "manual"),
            dependencies=(
                (value.get("dependency_graph") or {}).get("edges") or ()
            ),
        )
        return requirements, canonical
    requirements = [str(item) for item in value]
    return requirements, build_frozen_requirement_set(requirements)
