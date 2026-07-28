"""Requirement-traceable structural obligations for generated SysML models.

The whole-model plan already fixes typed components, ports, and connections.
This module compiles its requirement-tagged connection graph into stable
source-to-target obligations before SysML generation.  Later validation can
therefore evaluate the same paths for every candidate instead of deriving a
new scenario set from candidate-specific names or ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


_STRUCTURAL_CATEGORIES = ("REQ_FUNC_", "REQ_SAFE_", "REQ_INTF_", "REQ_OPER_")


@dataclass(frozen=True)
class StructuralConnectionRef:
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    item_type: str

    def key(self) -> tuple[str, str, str, str]:
        return (
            self.source_component,
            self.source_port,
            self.target_component,
            self.target_port,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_component": self.source_component,
            "source_port": self.source_port,
            "target_component": self.target_component,
            "target_port": self.target_port,
            "item_type": self.item_type,
        }


@dataclass(frozen=True)
class StructuralObligation:
    obligation_id: str
    requirement_id: str
    source_component: str
    target_component: str
    required_components: tuple[str, ...]
    required_connections: tuple[StructuralConnectionRef, ...]
    entry_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "requirement_id": self.requirement_id,
            "source_component": self.source_component,
            "target_component": self.target_component,
            "required_components": list(self.required_components),
            "required_connections": [
                item.to_dict() for item in self.required_connections
            ],
            "entry_kind": self.entry_kind,
        }


def requires_structural_path(requirement_id: str) -> bool:
    return str(requirement_id).upper().startswith(_STRUCTURAL_CATEGORIES)


def compile_structural_obligations(
    components: Sequence[Any],
    connections: Sequence[Any],
    *,
    allocated_requirements: Iterable[str] = (),
) -> tuple[tuple[StructuralObligation, ...], tuple[str, ...]]:
    """Compile stable causal paths from a validated typed model plan.

    Connections are grouped by their requirement provenance.  Every simple
    root-to-sink path becomes one obligation, preserving parallel safety paths
    instead of collapsing them into an existential role-level check.
    """
    component_ports = {
        component.name: tuple(component.ports)
        for component in components
    }
    traced_requirements = sorted({
        requirement_id
        for connection in connections
        for requirement_id in connection.requirements
        if requires_structural_path(requirement_id)
    })
    allocated = {
        requirement_id
        for requirement_id in allocated_requirements
        if requires_structural_path(requirement_id)
    }
    issues = [
        f"{requirement_id} has no requirement-traceable structural path"
        for requirement_id in sorted(allocated - set(traced_requirements))
    ]
    obligations: list[StructuralObligation] = []

    for requirement_id in traced_requirements:
        tagged = sorted(
            (
                connection
                for connection in connections
                if requirement_id in connection.requirements
            ),
            key=lambda item: (
                item.source_component,
                item.source_port,
                item.target_component,
                item.target_port,
            ),
        )
        adjacency: dict[str, list[Any]] = {}
        indegree: dict[str, int] = {}
        outdegree: dict[str, int] = {}
        nodes: set[str] = set()
        for connection in tagged:
            source = connection.source_component
            target = connection.target_component
            nodes.update((source, target))
            adjacency.setdefault(source, []).append(connection)
            outdegree[source] = outdegree.get(source, 0) + 1
            indegree[target] = indegree.get(target, 0) + 1
            indegree.setdefault(source, 0)
            outdegree.setdefault(target, 0)

        roots = sorted(node for node in nodes if indegree.get(node, 0) == 0)
        sinks = {node for node in nodes if outdegree.get(node, 0) == 0}
        paths: list[tuple[Any, ...]] = []

        def visit(
            node: str,
            path: tuple[Any, ...],
            visited: frozenset[str],
        ) -> None:
            if node in sinks and path:
                paths.append(path)
                return
            for connection in adjacency.get(node, ()):
                target = connection.target_component
                if target in visited:
                    continue
                visit(
                    target,
                    (*path, connection),
                    visited | {target},
                )

        for root in roots:
            visit(root, (), frozenset({root}))

        # A closed directed loop has no root/sink pair.  Preserve its exact
        # planned edges as bounded one-edge obligations rather than inventing
        # an arbitrary break point.
        if not paths:
            paths = [(connection,) for connection in tagged]

        unique_paths: dict[
            tuple[tuple[str, str, str, str], ...],
            tuple[Any, ...],
        ] = {}
        for path in paths:
            key = tuple(
                (
                    item.source_component,
                    item.source_port,
                    item.target_component,
                    item.target_port,
                )
                for item in path
            )
            unique_paths.setdefault(key, path)

        for index, path in enumerate(unique_paths.values(), 1):
            source = path[0].source_component
            target = path[-1].target_component
            required_components = (
                source,
                *(item.target_component for item in path),
            )
            external_entry = any(
                port.external and port.direction in {"in", "inout"}
                for port in component_ports.get(source, ())
            )
            obligations.append(StructuralObligation(
                obligation_id=f"STRUCT_{requirement_id}_{index:03d}",
                requirement_id=requirement_id,
                source_component=source,
                target_component=target,
                required_components=tuple(required_components),
                required_connections=tuple(
                    StructuralConnectionRef(
                        source_component=item.source_component,
                        source_port=item.source_port,
                        target_component=item.target_component,
                        target_port=item.target_port,
                        item_type=item.item_type,
                    )
                    for item in path
                ),
                entry_kind=(
                    "EXTERNAL_BOUNDARY"
                    if external_entry
                    else "INTERNAL_SOURCE"
                ),
            ))

    return tuple(obligations), tuple(issues)
