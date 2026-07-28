"""Typed whole-model generation plan and deterministic connectivity assembly.

The LLM decides the architecture and signal flows once.  Later generation
stages consume this validated plan instead of independently guessing component,
port, and connection names.  The committed SysML remains the semantic authority;
this object is generation input and an auditable conformance expectation only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .ag_behavior_plan import (
    BehaviorObligation,
    BehaviorObligationPlan,
)
from .structural_obligations import (
    StructuralObligation,
    compile_structural_obligations,
)
from ..utils.req_id import normalise_req_id


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_REQ_ID = re.compile(r"\bREQ[-_][A-Za-z]+[-_]\d+\b", re.IGNORECASE)


def _req_ids(values: Iterable[str]) -> tuple[str, ...]:
    found: list[str] = []
    for value in values:
        for match in _REQ_ID.findall(str(value)):
            req_id = normalise_req_id(match)
            if req_id not in found:
                found.append(req_id)
    return tuple(found)


@dataclass(frozen=True)
class PortPlan:
    name: str
    direction: str
    port_type: str = "DataPort"
    external: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "type": self.port_type,
            "external": self.external,
        }


@dataclass(frozen=True)
class ComponentPlan:
    name: str
    responsibility: str
    requirements: tuple[str, ...]
    ports: tuple[PortPlan, ...]
    attributes: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "responsibility": self.responsibility,
            "requirements": list(self.requirements),
            "ports": [item.to_dict() for item in self.ports],
            "attributes": [dict(item) for item in self.attributes],
        }


@dataclass(frozen=True)
class ConnectionPlan:
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    item_type: str = "DataPort"
    requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": {
                "component": self.source_component,
                "port": self.source_port,
            },
            "target": {
                "component": self.target_component,
                "port": self.target_port,
            },
            "item_type": self.item_type,
            "requirements": list(self.requirements),
        }


@dataclass
class ModelGenerationPlan:
    components: tuple[ComponentPlan, ...] = ()
    connections: tuple[ConnectionPlan, ...] = ()
    structural_obligations: tuple[StructuralObligation, ...] = ()
    behavior_obligations: tuple[BehaviorObligation, ...] = ()
    source: str = "LLM_TYPED_JSON"
    issues: tuple[str, ...] = ()
    schema_version: str = "1.0"

    @property
    def status(self) -> str:
        if not self.components or not self.connections:
            return "INCOMPLETE"
        return "PASS" if not self.issues else "INVALID"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_role": "WHOLE_MODEL_GENERATION_PLAN",
            "semantic_authority": "GENERATION_INPUT_ONLY_COMMITTED_SYSML_WINS",
            "source": self.source,
            "status": self.status,
            "components": [item.to_dict() for item in self.components],
            "connections": [item.to_dict() for item in self.connections],
            "structural_obligations": [
                item.to_dict() for item in self.structural_obligations
            ],
            "behavior_obligations": [
                item.to_dict() for item in self.behavior_obligations
            ],
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ModelGenerationPlan":
        if not isinstance(value, Mapping):
            return cls(issues=("typed generation plan is absent",))
        return cls.from_payload(value, source=str(value.get("source") or "ARCHIVED"))

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        requirements: Sequence[str] = (),
        source: str = "LLM_TYPED_JSON",
    ) -> "ModelGenerationPlan":
        raw_components = payload.get("components")
        raw_connections = payload.get("connections")
        issues: list[str] = []
        components: list[ComponentPlan] = []
        if not isinstance(raw_components, Sequence) or isinstance(
            raw_components, (str, bytes)
        ):
            raw_components = ()
            issues.append("components must be a list")
        for index, raw in enumerate(raw_components):
            if not isinstance(raw, Mapping):
                issues.append(f"components[{index}] must be an object")
                continue
            name = str(raw.get("name") or "").strip()
            if not _IDENTIFIER.fullmatch(name):
                issues.append(f"components[{index}].name is not a SysML identifier")
                continue
            responsibility = str(raw.get("responsibility") or "").strip()
            if not responsibility:
                issues.append(f"{name} has no stated responsibility")
            ports: list[PortPlan] = []
            for p_index, port in enumerate(raw.get("ports") or ()):
                if not isinstance(port, Mapping):
                    issues.append(f"{name}.ports[{p_index}] must be an object")
                    continue
                port_name = str(port.get("name") or "").strip()
                direction = str(port.get("direction") or "").strip().lower()
                port_type = str(
                    port.get("type") or port.get("port_type") or "DataPort"
                ).strip()
                if not _IDENTIFIER.fullmatch(port_name):
                    issues.append(f"{name}.ports[{p_index}] has invalid name")
                    continue
                if direction not in {"in", "out", "inout"}:
                    issues.append(
                        f"{name}.{port_name} has invalid direction {direction!r}"
                    )
                    continue
                if not _IDENTIFIER.fullmatch(port_type):
                    issues.append(f"{name}.{port_name} has invalid port type")
                    continue
                ports.append(PortPlan(
                    name=port_name,
                    direction=direction,
                    port_type=port_type,
                    external=bool(port.get("external", False)),
                ))
            port_names = [item.name for item in ports]
            if len(set(port_names)) != len(port_names):
                issues.append(f"{name} port names must be unique")
            attributes: list[dict[str, str]] = []
            for attribute in raw.get("attributes") or ():
                if isinstance(attribute, Mapping) and attribute.get("name"):
                    attributes.append({
                        "name": str(attribute["name"]),
                        "unit": str(attribute.get("unit") or "1"),
                    })
            components.append(ComponentPlan(
                name=name,
                responsibility=responsibility,
                requirements=_req_ids(raw.get("requirements") or ()),
                ports=tuple(ports),
                attributes=tuple(attributes),
            ))

        by_component = {item.name: item for item in components}
        if len(by_component) != len(components):
            issues.append("component names must be unique")
        port_lookup = {
            (component.name, port.name): port
            for component in components
            for port in component.ports
        }
        connections: list[ConnectionPlan] = []
        if not isinstance(raw_connections, Sequence) or isinstance(
            raw_connections, (str, bytes)
        ):
            raw_connections = ()
            issues.append("connections must be a list")
        driven_inputs: set[tuple[str, str]] = set()
        connection_keys: set[tuple[str, str, str, str]] = set()
        for index, raw in enumerate(raw_connections):
            if not isinstance(raw, Mapping):
                issues.append(f"connections[{index}] must be an object")
                continue
            source_endpoint = raw.get("source") or {}
            target_endpoint = raw.get("target") or {}
            if not isinstance(source_endpoint, Mapping) or not isinstance(
                target_endpoint, Mapping
            ):
                issues.append(f"connections[{index}] endpoints must be objects")
                continue
            sc = str(source_endpoint.get("component") or "").strip()
            sp = str(source_endpoint.get("port") or "").strip()
            tc = str(target_endpoint.get("component") or "").strip()
            tp = str(target_endpoint.get("port") or "").strip()
            source_port = port_lookup.get((sc, sp))
            target_port = port_lookup.get((tc, tp))
            if source_port is None or target_port is None:
                issues.append(
                    f"connections[{index}] references an undeclared endpoint "
                    f"{sc}.{sp} -> {tc}.{tp}"
                )
                continue
            if source_port.direction not in {"out", "inout"}:
                issues.append(f"{sc}.{sp} cannot be a connection source")
            if target_port.direction not in {"in", "inout"}:
                issues.append(f"{tc}.{tp} cannot be a connection target")
            if source_port.port_type != target_port.port_type:
                issues.append(
                    f"{sc}.{sp} and {tc}.{tp} have different port types"
                )
            item_type = str(raw.get("item_type") or source_port.port_type).strip()
            if not _IDENTIFIER.fullmatch(item_type):
                issues.append(f"connections[{index}] has invalid item_type")
            elif item_type != source_port.port_type:
                issues.append(
                    f"connections[{index}] item_type {item_type} does not match "
                    f"endpoint type {source_port.port_type}"
                )
            target_key = (tc, tp)
            if target_port.direction == "in" and target_key in driven_inputs:
                issues.append(f"{tc}.{tp} has more than one planned driver")
            driven_inputs.add(target_key)
            connection_key = (sc, sp, tc, tp)
            if connection_key in connection_keys:
                issues.append(
                    f"duplicate planned connection {sc}.{sp} -> {tc}.{tp}"
                )
            connection_keys.add(connection_key)
            connections.append(ConnectionPlan(
                source_component=sc,
                source_port=sp,
                target_component=tc,
                target_port=tp,
                item_type=item_type,
                requirements=_req_ids(raw.get("requirements") or ()),
            ))

        declared_requirements = set(_req_ids(requirements))
        allocated_requirements = {
            req_id for component in components for req_id in component.requirements
        }
        for req_id in sorted(declared_requirements - allocated_requirements):
            issues.append(f"{req_id} has no responsible component")
        if declared_requirements:
            for req_id in sorted(allocated_requirements - declared_requirements):
                issues.append(f"{req_id} is allocated but not declared")
            connection_requirements = {
                req_id
                for connection in connections
                for req_id in connection.requirements
            }
            for req_id in sorted(connection_requirements - declared_requirements):
                issues.append(f"{req_id} traces a connection but is not declared")

        consumed = {
            (item.target_component, item.target_port) for item in connections
        }
        produced = {
            (item.source_component, item.source_port) for item in connections
        }
        for component in components:
            if not component.ports:
                issues.append(f"{component.name} has no planned ports")
            for port in component.ports:
                key = (component.name, port.name)
                if port.external:
                    continue
                if port.direction == "in" and key not in consumed:
                    issues.append(f"{component.name}.{port.name} has no planned driver")
                if port.direction == "out" and key not in produced:
                    issues.append(f"{component.name}.{port.name} has no planned consumer")

        behavior_obligations = tuple(
            BehaviorObligation.from_dict(dict(item))
            for item in (payload.get("behavior_obligations") or ())
            if isinstance(item, Mapping)
        )
        structural_obligations, structural_issues = (
            compile_structural_obligations(
                components,
                connections,
                allocated_requirements=allocated_requirements,
            )
        )
        issues.extend(structural_issues)
        return cls(
            components=tuple(components),
            connections=tuple(connections),
            structural_obligations=structural_obligations,
            behavior_obligations=behavior_obligations,
            source=source,
            issues=tuple(dict.fromkeys(issues)),
            schema_version=(
                "3.0"
                if structural_obligations
                else str(payload.get("schema_version") or "1.0")
            ),
        )

    @classmethod
    def from_legacy_text(
        cls,
        text: str,
        *,
        requirements: Sequence[str] = (),
    ) -> "ModelGenerationPlan":
        """Best-effort compatibility for archived/tests that return prose."""
        component_pattern = re.compile(
            r"(?ms)^\s*\d+\.\s+([A-Za-z_]\w*)\s+[—-]\s*(.*?)"
            r"(?=^\s*\d+\.|\Z)"
        )
        components: list[dict[str, Any]] = []
        for match in component_pattern.finditer(text):
            name, block = match.group(1), match.group(2)
            addresses = re.search(r"(?im)^\s*Addresses:\s*(.+)$", block)
            ports_line = re.search(r"(?im)^\s*Ports needed:\s*(.+)$", block)
            ports = []
            if ports_line:
                for raw_port in ports_line.group(1).split(","):
                    token = raw_port.strip()
                    direction_match = re.search(
                        r"\b(inout|in|out)\b", token, re.IGNORECASE
                    )
                    if not direction_match:
                        continue
                    direction = direction_match.group(1).lower()
                    remainder = (
                        token[:direction_match.start()]
                        + " "
                        + token[direction_match.end():]
                    )
                    words = re.findall(r"[A-Za-z_]\w*", remainder)
                    if words:
                        ports.append({
                            "name": words[0],
                            "direction": direction,
                            "type": "DataPort",
                            "external": True,
                        })
            components.append({
                "name": name,
                "responsibility": block.splitlines()[0].strip(),
                "requirements": (
                    _req_ids((addresses.group(1),)) if addresses else ()
                ),
                "ports": ports,
            })
        plan = cls.from_payload(
            {"components": components, "connections": []},
            requirements=requirements,
            source="LEGACY_STRUCTURED_TEXT",
        )
        return plan

    def render_for_prompt(self) -> str:
        lines = ["TYPED WHOLE-MODEL GENERATION PLAN (validated once):"]
        for index, component in enumerate(self.components, 1):
            lines.extend([
                f"{index}. {component.name} — {component.responsibility}",
                "   Addresses: " + (
                    ", ".join(component.requirements) or "(none)"
                ),
                "   Ports needed: " + (
                    ", ".join(
                        f"{port.direction} {port.name} : {port.port_type}"
                        + (" [external]" if port.external else "")
                        for port in component.ports
                    ) or "(none)"
                ),
                "   Key attributes: " + (
                    ", ".join(
                        f"{item['name']} [{item['unit']}]"
                        for item in component.attributes
                    ) or "(derive from allocated requirements)"
                ),
            ])
        lines.append("Connections (emit exactly; do not reverse or substitute endpoints):")
        for connection in self.connections:
            trace = (
                f" [{', '.join(connection.requirements)}]"
                if connection.requirements else ""
            )
            lines.append(
                f"- {connection.source_component}.{connection.source_port} -> "
                f"{connection.target_component}.{connection.target_port} : "
                f"{connection.item_type}{trace}"
            )
        if self.structural_obligations:
            lines.append("")
            lines.append(
                "FROZEN REQUIREMENT STRUCTURAL OBLIGATIONS "
                "(validation input; do not add unrelated paths):"
            )
            for obligation in self.structural_obligations:
                path = " -> ".join(obligation.required_components)
                lines.append(
                    f"- {obligation.obligation_id} "
                    f"[{obligation.requirement_id}]: {path} "
                    f"({obligation.entry_kind})"
                )
        if self.behavior_obligations:
            lines.append("")
            lines.append(
                BehaviorObligationPlan(
                    self.behavior_obligations
                ).render_for_prompt()
            )
        if self.issues:
            lines.append("Plan validation issues (must be resolved; do not hide them):")
            lines.extend(f"- {item}" for item in self.issues)
        return "\n".join(lines)


def attach_ag_behavior_obligations(
    plan: ModelGenerationPlan,
    behavior_plan: BehaviorObligationPlan,
) -> ModelGenerationPlan:
    """Cross-validate and attach A/G behavior facts to the whole-model plan."""
    issues = list(plan.issues)
    component_names = {item.name for item in plan.components}
    allocated = {
        (component.name, requirement)
        for component in plan.components
        for requirement in component.requirements
    }
    contract_ids: set[str] = set()
    stable_ids: set[tuple[str, str]] = set()
    for obligation in behavior_plan.obligations:
        if obligation.contract_id in contract_ids:
            issues.append(
                f"duplicate A/G behavior contract {obligation.contract_id}"
            )
        contract_ids.add(obligation.contract_id)
        if obligation.owner_def not in component_names:
            issues.append(
                f"A/G behavior owner {obligation.owner_def} is absent from "
                "the whole-model component plan"
            )
        if (
            obligation.owner_def,
            normalise_req_id(obligation.requirement_id),
        ) not in allocated:
            issues.append(
                f"{obligation.owner_def} is not allocated "
                f"{normalise_req_id(obligation.requirement_id)}"
            )
        stable_key = (
            obligation.owner_def,
            obligation.stable_behavior_id,
        )
        if stable_key in stable_ids:
            issues.append(
                f"duplicate stable behavior id "
                f"{obligation.owner_def}::{obligation.stable_behavior_id}"
            )
        stable_ids.add(stable_key)
    if behavior_plan.status != "PASS":
        issues.append(
            f"A/G behavior obligation plan is {behavior_plan.status}"
        )
    return replace(
        plan,
        behavior_obligations=behavior_plan.obligations,
        issues=tuple(dict.fromkeys(issues)),
        schema_version="3.0",
    )


def apply_generation_plan(
    model_text: str,
    plan: ModelGenerationPlan,
) -> tuple[str, dict[str, Any]]:
    """Materialise missing planned connects and report exact conformance."""
    from ..simulation.connectivity_fixer import (
        build_port_directory,
        merge_connects,
        parse_connects,
        validate_connects,
    )
    from ..simulation.extractor import extract_behavioral_graph

    directory = build_port_directory(model_text)
    existing = parse_connects(model_text)
    instances_by_type: dict[str, list[str]] = {}
    for instance, component_type in directory.instance_type.items():
        instances_by_type.setdefault(component_type, []).append(instance)

    issues = list(plan.issues)
    candidate_lines: list[str] = []
    expected_keys: list[tuple[str, str, str, str]] = []

    def resolve(component: str) -> str | None:
        candidates = instances_by_type.get(component, ())
        if len(candidates) == 1:
            return candidates[0]
        conventional = component[:1].lower() + component[1:]
        if conventional in candidates:
            return conventional
        issues.append(
            f"cannot uniquely resolve usage for component {component}"
        )
        return None

    for connection in plan.connections:
        source = resolve(connection.source_component)
        target = resolve(connection.target_component)
        if source is None or target is None:
            continue
        key = (
            source,
            connection.source_port,
            target,
            connection.target_port,
        )
        expected_keys.append(key)
        if key not in {item.key() for item in existing}:
            candidate_lines.append(
                f"connect {source}.{connection.source_port} "
                f"to {target}.{connection.target_port};"
            )

    validation = validate_connects(candidate_lines, directory, existing)
    merged = merge_connects(model_text, validation.accepted)
    final_directory = build_port_directory(merged.merged_text)
    package_match = re.search(r"\bpackage\s+([A-Za-z_]\w*)\s*\{", model_text)
    root_package = package_match.group(1) if package_match else None
    scoped_graph = extract_behavioral_graph(
        merged.merged_text,
        root_package=root_package,
    )
    scoped_instances = set(scoped_graph.parts)
    if not scoped_instances:
        scoped_instances = set(final_directory.instances)
    actual_instance_types = {
        instance: component_type
        for instance, component_type in final_directory.instance_type.items()
        if instance in scoped_instances
    }
    final_connections = {
        item.key()
        for item in parse_connects(merged.merged_text)
        if (
            item.src_inst in scoped_instances
            and item.tgt_inst in scoped_instances
        )
    }
    missing = [
        ".".join((src, sp)) + " -> " + ".".join((tgt, tp))
        for src, sp, tgt, tp in expected_keys
        if (src, sp, tgt, tp) not in final_connections
    ]
    issues.extend(
        f"planned connect rejected: {line}: {reason}"
        for line, reason in validation.rejected
    )
    issues.extend(f"planned connect missing: {item}" for item in missing)

    planned_component_types = {item.name for item in plan.components}
    actual_by_type: dict[str, list[str]] = {}
    for instance, component_type in actual_instance_types.items():
        actual_by_type.setdefault(component_type, []).append(instance)
    missing_components = sorted(
        component
        for component in planned_component_types
        if not actual_by_type.get(component)
    )
    duplicate_component_usages = sorted(
        f"{component}: {', '.join(sorted(usages))}"
        for component, usages in actual_by_type.items()
        if component in planned_component_types and len(usages) != 1
    )
    unplanned_components = sorted(
        f"{instance} : {component_type}"
        for instance, component_type in actual_instance_types.items()
        if component_type not in planned_component_types
    )

    planned_ports = {
        (
            component.name,
            port.name,
            port.direction,
            port.port_type,
        )
        for component in plan.components
        for port in component.ports
    }
    actual_ports = {
        (
            component_type,
            port.name,
            port.direction,
            str(port.port_type or ""),
        )
        for instance, component_type in actual_instance_types.items()
        if component_type in planned_component_types
        for port in final_directory.instances.get(instance, {}).values()
    }
    missing_ports = sorted(
        f"{component}.{name} ({direction}:{port_type})"
        for component, name, direction, port_type
        in planned_ports - actual_ports
    )
    unplanned_ports = sorted(
        f"{component}.{name} ({direction}:{port_type})"
        for component, name, direction, port_type
        in actual_ports - planned_ports
    )

    expected_key_set = set(expected_keys)
    unplanned_connections = sorted(
        ".".join((src, source_port))
        + " -> "
        + ".".join((target, target_port))
        for src, source_port, target, target_port
        in final_connections - expected_key_set
    )

    internalized_external_ports: list[str] = []
    for component in plan.components:
        usages = actual_by_type.get(component.name, ())
        if len(usages) != 1:
            continue
        usage = usages[0]
        for port in component.ports:
            if not port.external:
                continue
            if (
                port.direction in {"in", "inout"}
                and any(
                    target == usage and target_port == port.name
                    for _, _, target, target_port in final_connections
                )
            ):
                internalized_external_ports.append(
                    f"{usage}.{port.name} receives an internal connection"
                )
            if (
                port.direction in {"out", "inout"}
                and any(
                    source == usage and source_port == port.name
                    for source, source_port, _, _ in final_connections
                )
            ):
                internalized_external_ports.append(
                    f"{usage}.{port.name} drives an internal connection"
                )

    issues.extend(
        f"planned component missing: {item}" for item in missing_components
    )
    issues.extend(
        f"planned component usage multiplicity mismatch: {item}"
        for item in duplicate_component_usages
    )
    issues.extend(
        f"unplanned component usage: {item}" for item in unplanned_components
    )
    issues.extend(f"planned port missing: {item}" for item in missing_ports)
    issues.extend(f"unplanned port: {item}" for item in unplanned_ports)
    issues.extend(
        f"unplanned connection: {item}" for item in unplanned_connections
    )
    issues.extend(
        f"external boundary violation: {item}"
        for item in internalized_external_ports
    )
    report = {
        "schema_version": "2.0",
        "artifact_role": "GENERATION_PLAN_CONFORMANCE",
        "status": (
            "PASS"
            if plan.status == "PASS" and not issues and not missing
            else "FAIL"
        ),
        "plan_status": plan.status,
        "planned_connection_count": len(plan.connections),
        "realized_connection_count": len(expected_keys) - len(missing),
        "deterministically_added_connections": list(merged.added_lines),
        "missing_connections": missing,
        "unplanned_connections": unplanned_connections,
        "missing_components": missing_components,
        "duplicate_component_usages": duplicate_component_usages,
        "unplanned_components": unplanned_components,
        "missing_ports": missing_ports,
        "unplanned_ports": unplanned_ports,
        "internalized_external_ports": internalized_external_ports,
        "issues": list(dict.fromkeys(issues)),
    }
    return merged.merged_text, report
