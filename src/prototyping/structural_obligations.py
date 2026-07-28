"""Requirement-traceable structural obligations for generated SysML models.

The whole-model plan already fixes typed components, ports, and connections.
This module compiles its requirement-tagged connection graph into stable
source-to-target obligations before SysML generation.  Later validation can
therefore evaluate the same paths for every candidate instead of deriving a
new scenario set from candidate-specific names or ports.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


def validate_structural_obligations(
    model_text: str,
    obligations: Sequence[StructuralObligation],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Validate frozen structural obligations against one terminal SysML model."""
    from ..simulation.extractor import extract_behavioral_graph

    graph = extract_behavioral_graph(
        model_text,
        root_package=model_name,
    )
    usages_by_definition: dict[str, list[str]] = {}
    for usage_name, part in graph.parts.items():
        usages_by_definition.setdefault(part.def_name, []).append(usage_name)
    for usages in usages_by_definition.values():
        usages.sort()

    def resolve(component: str) -> tuple[str | None, str | None]:
        usages = usages_by_definition.get(component, ())
        conventional = component[:1].lower() + component[1:]
        if conventional in usages:
            return conventional, None
        if len(usages) == 1:
            return usages[0], None
        if not usages:
            return None, f"component definition {component} has no system usage"
        return None, (
            f"component definition {component} has ambiguous usages: "
            f"{', '.join(usages)}"
        )

    actual_connections = {
        (
            connection.source.split(".", 1)[0],
            connection.source.split(".", 1)[1],
            connection.target.split(".", 1)[0],
            connection.target.split(".", 1)[1],
        )
        for connection in graph.connections
        if "." in connection.source and "." in connection.target
    }
    results: list[dict[str, Any]] = []

    for obligation in obligations:
        issues: list[str] = []
        resolved: dict[str, str] = {}
        for component in obligation.required_components:
            usage, issue = resolve(component)
            if issue:
                issues.append(issue)
            elif usage is not None:
                resolved[component] = usage

        missing_connections: list[str] = []
        for connection in obligation.required_connections:
            source = resolved.get(connection.source_component)
            target = resolved.get(connection.target_component)
            if source is None or target is None:
                continue
            key = (
                source,
                connection.source_port,
                target,
                connection.target_port,
            )
            if key not in actual_connections:
                missing_connections.append(
                    f"{source}.{connection.source_port} -> "
                    f"{target}.{connection.target_port}"
                )
        if missing_connections:
            issues.extend(
                f"missing required connection {item}"
                for item in missing_connections
            )

        observed_path: list[str] = []
        if not missing_connections and not issues:
            for index, connection in enumerate(
                obligation.required_connections
            ):
                source = resolved[connection.source_component]
                target = resolved[connection.target_component]
                segment = [
                    source,
                    f"{source}.{connection.source_port}",
                    f"{target}.{connection.target_port}",
                    target,
                ]
                observed_path.extend(
                    segment if index == 0 else segment[1:]
                )

        results.append({
            **obligation.to_dict(),
            "status": "PASS" if not issues else "FAIL",
            "resolved_usages": resolved,
            "observed_path": observed_path,
            "missing_connections": missing_connections,
            "issues": list(dict.fromkeys(issues)),
        })

    passed = sum(item["status"] == "PASS" for item in results)
    if not obligations:
        status = "UNVERIFIED"
    else:
        status = "PASS" if passed == len(results) else "FAIL"
    return {
        "schema_version": "1.0",
        "artifact_role": "REQUIREMENT_STRUCTURAL_OBLIGATION_VALIDATION",
        "status": status,
        "scenario_set_fixed": True,
        "model_name": model_name,
        "source_model_digest": hashlib.sha256(
            (model_text or "").encode("utf-8")
        ).hexdigest(),
        "passed": passed,
        "total": len(results),
        "results": results,
    }
