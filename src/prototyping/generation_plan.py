"""Typed whole-model generation plan and deterministic connectivity assembly.

The LLM decides the architecture and signal flows once.  Later generation
stages consume this validated plan instead of independently guessing component,
port, and connection names.  The committed SysML remains the semantic authority;
this object is generation input and an auditable conformance expectation only.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .ag_behavior_plan import (
    BehaviorObligation,
    BehaviorObligationPlan,
    behavior_boolean_concepts,
    materialize_owned_behavior_obligations,
    reserved_behavior_identities,
)
from .event_symbols import (
    collect_planned_event_symbols,
    materialize_planned_event_symbols,
    validate_planned_event_symbols,
)
from .structural_obligations import (
    RequirementRealizationPlan,
    StructuralObligation,
    compile_structural_obligations,
    compile_source_anchored_structural_obligations,
)
from .requirement_semantics import (
    RequirementSemanticObligation,
    SemanticBindingPlan,
    compile_requirement_semantic_obligations,
    materialize_semantic_bindings,
    quantity_type_for_unit,
    semantic_binding_matches_subject,
)
from .namespace_integrity import collect_package_definitions
from .activated_constraint_plan import (
    AttributePlan,
    ConstraintPlan,
    materialize_planned_attributes,
    materialize_planned_constraints,
    validate_constraint_plan,
)
from .planned_behavior import (
    PlannedBehavior,
    materialize_owned_planned_behaviors,
    validate_planned_behaviors,
)
from ..utils.req_id import normalise_req_id
from ..utils.sysml_text_utils import find_block_end


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_QUALIFIED_TYPE = re.compile(
    r"^[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*$"
)
_REQ_ID = re.compile(r"\bREQ[-_][A-Za-z]+[-_]\d+\b", re.IGNORECASE)
_GENERIC_PORT_TYPES = {"DataPort", "StatusPort", "CommandPort"}
PLAN_APPLICATION_HISTORY_KEY = "plan_application_history"
_STANDARD_LIBRARY_TYPES = {
    "ScalarValues": {
        "Boolean",
        "Complex",
        "Integer",
        "Rational",
        "Real",
        "String",
    },
    "ISQ": {
        "AccelerationValue",
        "AngleValue",
        "AngularMeasureValue",
        "ChargeValue",
        "CurrentValue",
        "DurationValue",
        "EnergyValue",
        "ForceValue",
        "FrequencyValue",
        "LengthValue",
        "MassValue",
        "PowerValue",
        "SpeedValue",
        "TemperatureValue",
        "ThermodynamicTemperatureValue",
        "TimeValue",
        "VelocityValue",
        "VoltageValue",
    },
}
_SI_UNIT_NAMES = {
    "A", "C", "Hz", "J", "K", "N", "Pa", "V", "W",
    "cm", "deg", "g", "h", "kg", "km", "m", "min", "mm", "ms", "rad", "s",
}


_DECLARED_PORT = re.compile(
    r"\b(?P<direction>in|out|inout)\s+port\s+(?P<name>[A-Za-z_]\w*)"
    r"\s*:\s*(?P<type>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*;"
)


def normalise_planned_port_types(
    model_text: str, components: Sequence[Any]
) -> tuple[str, list[str]]:
    """Give a planned port the planned type when generation used another one.

    The plan owns a port's type exactly as it owns an attribute's. Generation
    writing `in port overrideCmd : DataPort` where the plan says `CommandPort`
    used to satisfy the "does this port exist?" test, which looks only at the
    name, so nothing added it and nothing corrected it — and conformance then
    reported the same port as BOTH a missing planned port and an unplanned one.
    Four such pairs failed one measured run.

    Direction is deliberately not rewritten. A wrong direction changes what the
    connections mean, which is a design question rather than a notation one.
    """
    text = str(model_text)
    changes: list[str] = []
    for component in components:
        planned = {
            item.name: item for item in getattr(component, "ports", ())
        }
        if not planned:
            continue
        match = re.search(
            rf"\bpart\s+def\s+{re.escape(component.name)}\s*\{{", text
        )
        if match is None:
            continue
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        body = text[brace + 1:end]
        rewritten = []
        for declaration in _DECLARED_PORT.finditer(body):
            want = planned.get(declaration.group("name"))
            if want is None:
                continue
            if declaration.group("direction") != want.direction:
                continue
            if declaration.group("type") == want.port_type:
                continue
            rewritten.append((
                declaration.start(), declaration.end(),
                f"{want.direction} port {want.name} : {want.port_type};",
                f"{component.name}.{want.name} "
                f"({declaration.group('type')} -> {want.port_type})",
            ))
        for start, stop, replacement, note in reversed(rewritten):
            body = body[:start] + replacement + body[stop:]
            changes.append(note)
        if rewritten:
            text = text[:brace + 1] + body + text[end:]
    return text, changes


def materialize_standard_library_imports(
    model_text: str,
) -> tuple[str, dict[str, Any]]:
    """Close standard-library dependencies in the first/root package."""
    text = str(model_text)
    package = re.search(r"\bpackage\s+([A-Za-z_]\w*)\s*\{", text)
    if package is None:
        return text, {
            "artifact_role": "STANDARD_LIBRARY_IMPORT_CLOSURE",
            "status": "FAIL",
            "root_package": None,
            "required_packages": [],
            "added_imports": [],
            "issues": ["terminal model has no root package"],
        }
    opening = text.find("{", package.start(), package.end())
    closing = find_block_end(text, opening)
    if opening == -1 or closing == -1:
        return text, {
            "artifact_role": "STANDARD_LIBRARY_IMPORT_CLOSURE",
            "status": "FAIL",
            "root_package": package.group(1),
            "required_packages": [],
            "added_imports": [],
            "issues": ["root package body cannot be parsed"],
        }
    body = text[opening + 1:closing]
    used_types = set(re.findall(
        r":\s*([A-Za-z_]\w*)"
        r"(?!\s*::)",
        body,
    ))
    required: dict[str, set[str]] = {
        namespace: used_types & names
        for namespace, names in _STANDARD_LIBRARY_TYPES.items()
    }
    unit_tokens = {
        token
        for bracket in re.findall(r"\[([^\]]+)\]", body)
        for token in re.findall(r"[A-Za-z_]\w*", bracket)
    }
    if unit_tokens & _SI_UNIT_NAMES:
        required["SI"] = unit_tokens & _SI_UNIT_NAMES
    required = {
        namespace: names for namespace, names in required.items() if names
    }

    additions: list[str] = []
    for namespace, names in sorted(required.items()):
        imported = set(re.findall(
            rf"\b(?:private|public)\s+import\s+"
            rf"{re.escape(namespace)}::([A-Za-z_]\w*|\*{{1,2}})\s*;",
            body,
        ))
        if "*" in imported or "**" in imported or names <= imported:
            continue
        additions.append(f"private import {namespace}::*;")
    if additions:
        insertion = "".join(f"\n    {line}" for line in additions) + "\n"
        text = text[:opening + 1] + insertion + text[opening + 1:]
    return text, {
        "artifact_role": "STANDARD_LIBRARY_IMPORT_CLOSURE",
        "status": "PASS",
        "root_package": package.group(1),
        "required_packages": sorted(required),
        "added_imports": additions,
        "issues": [],
    }


def append_plan_application_history(
    metadata: dict[str, Any],
    conformance: Mapping[str, Any],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """Append one immutable plan-application event to model metadata."""
    history = [
        dict(item)
        for item in metadata.get(PLAN_APPLICATION_HISTORY_KEY, ())
        if isinstance(item, Mapping)
    ]
    semantic = conformance.get("semantic_binding_conformance")
    semantic_changes = (
        list(semantic.get("deterministic_changes") or ())
        if isinstance(semantic, Mapping)
        else []
    )
    event = {
        "stage": stage,
        "status": str(conformance.get("status") or "UNKNOWN"),
        "input_model_digest": conformance.get("input_model_digest"),
        "output_model_digest": conformance.get("output_model_digest"),
        "semantic_changes": semantic_changes,
        "added_ports": list(
            conformance.get("deterministically_added_ports") or ()
        ),
        "added_connections": list(
            conformance.get("deterministically_added_connections") or ()
        ),
    }
    key = (
        event["stage"],
        event["input_model_digest"],
        event["output_model_digest"],
        tuple(event["semantic_changes"]),
        tuple(event["added_ports"]),
        tuple(event["added_connections"]),
    )
    existing_keys = {
        (
            item.get("stage"),
            item.get("input_model_digest"),
            item.get("output_model_digest"),
            tuple(item.get("semantic_changes") or ()),
            tuple(item.get("added_ports") or ()),
            tuple(item.get("added_connections") or ()),
        )
        for item in history
    }
    if key not in existing_keys:
        history.append(event)
    metadata[PLAN_APPLICATION_HISTORY_KEY] = history
    return history


def validate_part_definition_fragment(
    fragment: str,
    plan: "ModelGenerationPlan",
) -> dict[str, Any]:
    """Check that Step 2 emits exactly the plan's component definitions."""
    expected = {item.name for item in plan.components}
    observed_list = [
        item["name"]
        for item in collect_package_definitions(fragment)
        if item["kind"] == "part def"
    ]
    observed = set(observed_list)
    duplicates = sorted({
        name for name in observed_list if observed_list.count(name) > 1
    })
    missing = sorted(expected - observed)
    unplanned = sorted(observed - expected)
    return {
        "status": (
            "PASS"
            if not missing and not unplanned and not duplicates
            else "FAIL"
        ),
        "missing_part_definitions": missing,
        "unplanned_part_definitions": unplanned,
        "duplicate_part_definitions": duplicates,
    }


def _is_planned_assembly_container(
    model_text: str,
    definition_name: str,
    planned_components: set[str],
) -> bool:
    """Allow an unplanned wrapper only when it contains planned part usages."""
    match = re.search(
        rf"\bpart\s+def\s+{re.escape(definition_name)}\s*\{{",
        model_text,
    )
    if match is None:
        return False
    opening = model_text.find("{", match.start(), match.end())
    closing = find_block_end(model_text, opening)
    if closing == -1:
        return False
    usage_types = {
        item.group(1)
        for item in re.finditer(
            r"\bpart\s+[A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)\s*;",
            model_text[opening + 1:closing],
        )
    }
    return bool(usage_types) and usage_types <= planned_components


def _definition_contract_report(
    model_text: str,
    plan: "ModelGenerationPlan",
) -> dict[str, Any]:
    """Validate plan-owned definition names and their SysML declaration kind."""
    definitions = collect_package_definitions(model_text)
    by_name: dict[str, list[str]] = {}
    for item in definitions:
        by_name.setdefault(item["name"], []).append(item["kind"])

    planned_components = {item.name for item in plan.components}
    expected: dict[str, str] = {
        name: "part def" for name in planned_components
    }
    for binding in plan.semantic_bindings:
        expected[binding.item_type] = "item def"
        expected[binding.port_type] = "port def"

    missing: list[str] = []
    conflicts: list[str] = []
    for name, expected_kind in sorted(expected.items()):
        observed = by_name.get(name, [])
        if expected_kind not in observed:
            missing.append(f"{expected_kind} {name}")
        wrong = sorted(kind for kind in observed if kind != expected_kind)
        if wrong:
            conflicts.append(
                f"{name}: expected {expected_kind}, found "
                + ", ".join(wrong)
            )

    unplanned = sorted(
        name
        for name, kinds in by_name.items()
        if "part def" in kinds
        and name not in planned_components
        and not _is_planned_assembly_container(
            model_text, name, planned_components
        )
    )
    return {
        "status": (
            "PASS"
            if not missing and not conflicts and not unplanned
            else "FAIL"
        ),
        "missing_definitions": missing,
        "definition_kind_conflicts": conflicts,
        "unplanned_part_definitions": unplanned,
    }


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
    attributes: tuple[AttributePlan, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "responsibility": self.responsibility,
            "requirements": list(self.requirements),
            "ports": [item.to_dict() for item in self.ports],
            "attributes": [item.to_dict() for item in self.attributes],
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
    requirement_realizations: tuple[RequirementRealizationPlan, ...] = ()
    structural_obligations: tuple[StructuralObligation, ...] = ()
    semantic_obligations: tuple[RequirementSemanticObligation, ...] = ()
    semantic_bindings: tuple[SemanticBindingPlan, ...] = ()
    constraint_plans: tuple[ConstraintPlan, ...] = ()
    planned_behaviors: tuple[PlannedBehavior, ...] = ()
    behavior_obligations: tuple[BehaviorObligation, ...] = ()
    behavior_identity_reconciliations: tuple[str, ...] = ()
    constraint_identity_reconciliations: tuple[str, ...] = ()
    source: str = "LLM_TYPED_JSON"
    issues: tuple[str, ...] = ()
    schema_version: str = "1.0"

    @property
    def status(self) -> str:
        if not self.components or not self.connections:
            return "INCOMPLETE"
        return "PASS" if not self.issues else "INVALID"

    @property
    def planned_event_symbols(self):
        return collect_planned_event_symbols(
            self.planned_behaviors,
            self.behavior_obligations,
            self.components,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_role": "WHOLE_MODEL_GENERATION_PLAN",
            "semantic_authority": "GENERATION_INPUT_ONLY_COMMITTED_SYSML_WINS",
            "source": self.source,
            "status": self.status,
            "components": [item.to_dict() for item in self.components],
            "connections": [item.to_dict() for item in self.connections],
            "requirement_realizations": [
                item.to_dict() for item in self.requirement_realizations
            ],
            "structural_obligations": [
                item.to_dict() for item in self.structural_obligations
            ],
            "semantic_obligations": [
                item.to_dict() for item in self.semantic_obligations
            ],
            "semantic_bindings": [
                item.to_dict() for item in self.semantic_bindings
            ],
            "constraints": [
                item.to_dict() for item in self.constraint_plans
            ],
            "behaviors": [
                item.to_dict() for item in self.planned_behaviors
            ],
            "behavior_obligations": [
                item.to_dict() for item in self.behavior_obligations
            ],
            "planned_event_symbols": [
                item.to_dict() for item in self.planned_event_symbols
            ],
            "behavior_identity_reconciliations": list(
                self.behavior_identity_reconciliations
            ),
            "constraint_identity_reconciliations": list(
                self.constraint_identity_reconciliations
            ),
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
        require_source_anchored_paths: bool = False,
        ag_behavior_plan: BehaviorObligationPlan | None = None,
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
            attributes: list[AttributePlan] = []
            for attribute in raw.get("attributes") or ():
                if isinstance(attribute, Mapping) and attribute.get("name"):
                    planned_attribute = AttributePlan.from_dict(
                        attribute,
                        requirements=requirements,
                    )
                    if not _IDENTIFIER.fullmatch(planned_attribute.name):
                        issues.append(
                            f"{name} attribute name "
                            f"{planned_attribute.name!r} is not a SysML identifier"
                        )
                        continue
                    attributes.append(planned_attribute)
            attribute_names = [item.name for item in attributes]
            if len(set(attribute_names)) != len(attribute_names):
                issues.append(f"{name} attribute names must be unique")
            for member_name in sorted(
                set(port_names) & set(attribute_names)
            ):
                issues.append(
                    f"{name} direct member {member_name!r} cannot be both "
                    "a port and an attribute"
                )
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
        allocated_component_requirements = {
            (component.name, requirement)
            for component in components
            for requirement in component.requirements
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
        behavior_identity_reconciliations = tuple(
            str(item)
            for item in (
                payload.get("behavior_identity_reconciliations") or ()
            )
            if str(item).strip()
        )
        archived_semantic_obligations = tuple(
            RequirementSemanticObligation.from_dict(dict(item))
            for item in (payload.get("semantic_obligations") or ())
            if isinstance(item, Mapping)
        )
        semantic_obligations = (
            compile_requirement_semantic_obligations(requirements)
            if requirements else archived_semantic_obligations
        )
        raw_semantic_bindings = payload.get("semantic_bindings")
        if not isinstance(raw_semantic_bindings, Sequence) or isinstance(
            raw_semantic_bindings, (str, bytes)
        ):
            raw_semantic_bindings = ()
            if semantic_obligations:
                issues.append(
                    "semantic_bindings must contain one typed binding for "
                    "each semantic obligation"
                )
        semantic_bindings: list[SemanticBindingPlan] = []
        obligations_by_id = {
            item.obligation_id: item for item in semantic_obligations
        }
        seen_binding_ids: set[str] = set()
        port_payloads: dict[str, tuple[str, str]] = {}
        item_features: dict[
            tuple[str, str], tuple[str, str]
        ] = {}
        for index, raw in enumerate(raw_semantic_bindings):
            if not isinstance(raw, Mapping):
                issues.append(
                    f"semantic_bindings[{index}] must be an object"
                )
                continue
            binding = SemanticBindingPlan.from_dict(raw)
            semantic_bindings.append(binding)
            prefix = f"semantic_bindings[{index}]"
            obligation = obligations_by_id.get(binding.obligation_id)
            if obligation is None:
                issues.append(
                    f"{prefix} references unknown obligation "
                    f"{binding.obligation_id!r}"
                )
            elif binding.requirement_id != obligation.requirement_id:
                issues.append(
                    f"{prefix} requirement_id does not match "
                    f"{binding.obligation_id}"
                )
            if binding.obligation_id in seen_binding_ids:
                issues.append(
                    f"duplicate semantic binding for "
                    f"{binding.obligation_id}"
                )
            seen_binding_ids.add(binding.obligation_id)

            identifier_fields = {
                "source.component": binding.source_component,
                "source.port": binding.source_port,
                "target.component": binding.target_component,
                "target.port": binding.target_port,
                "payload.port_type": binding.port_type,
                "payload.port_feature": binding.port_feature,
                "payload.item_type": binding.item_type,
                "payload.item_feature": binding.item_feature,
                "target.runtime_attribute": binding.runtime_attribute,
                "constraint.threshold_attribute": (
                    binding.threshold_attribute
                ),
                "constraint.name": binding.constraint_name,
            }
            for field_name, field_value in identifier_fields.items():
                if not _IDENTIFIER.fullmatch(field_value):
                    issues.append(
                        f"{prefix}.{field_name} is not a SysML identifier"
                    )
            if not _QUALIFIED_TYPE.fullmatch(binding.value_type):
                issues.append(
                    f"{prefix}.payload.value_type is not a SysML type"
                )
            expected_quantity_type = quantity_type_for_unit(binding.unit)
            if expected_quantity_type is None:
                issues.append(
                    f"{prefix}.payload.unit {binding.unit!r} has no supported "
                    "SysML v2 ISQ quantity-type mapping"
                )
            elif binding.value_type != expected_quantity_type:
                issues.append(
                    f"{prefix}.payload.value_type must be "
                    f"{expected_quantity_type} for [{binding.unit}]"
                )
            source_port = port_lookup.get((
                binding.source_component,
                binding.source_port,
            ))
            target_port = port_lookup.get((
                binding.target_component,
                binding.target_port,
            ))
            if source_port is None or target_port is None:
                issues.append(
                    f"{prefix} references an undeclared semantic endpoint"
                )
            else:
                if (
                    source_port.port_type != binding.port_type
                    or target_port.port_type != binding.port_type
                ):
                    issues.append(
                        f"{prefix}.payload.port_type does not match both "
                        "planned endpoints"
                    )
                if (
                    binding.source_component,
                    binding.source_port,
                    binding.target_component,
                    binding.target_port,
                ) not in connection_keys:
                    issues.append(
                        f"{prefix} endpoints are not a planned connection"
                    )
            if binding.port_type in _GENERIC_PORT_TYPES:
                issues.append(
                    f"{prefix} must use a requirement-relevant dedicated "
                    f"port type, not generic {binding.port_type}"
                )
            component_names = {item.name for item in components}
            if binding.item_type in component_names:
                issues.append(
                    f"{prefix}.payload.item_type collides with planned "
                    f"component {binding.item_type}"
                )
            if binding.port_type in component_names:
                issues.append(
                    f"{prefix}.payload.port_type collides with planned "
                    f"component {binding.port_type}"
                )
            if binding.item_type == binding.port_type:
                issues.append(
                    f"{prefix} cannot use the same definition name for "
                    "item_type and port_type"
                )
            if (
                binding.target_component,
                binding.requirement_id,
            ) not in allocated_component_requirements:
                issues.append(
                    f"{prefix} target component is not allocated "
                    f"{binding.requirement_id}"
                )
            if obligation is not None:
                if binding.unit != obligation.unit:
                    issues.append(
                        f"{prefix}.payload.unit does not preserve "
                        f"{obligation.unit}"
                    )
                if not semantic_binding_matches_subject(binding, obligation):
                    issues.append(
                        f"{prefix} feature/attribute names do not preserve "
                        "the frozen subject"
                    )
            if binding.runtime_attribute == binding.threshold_attribute:
                issues.append(
                    f"{prefix} runtime and threshold attributes must differ"
                )
            payload_key = (
                binding.item_type,
                binding.port_feature,
            )
            prior_payload = port_payloads.setdefault(
                binding.port_type, payload_key
            )
            if prior_payload != payload_key:
                issues.append(
                    f"port type {binding.port_type} has conflicting "
                    "semantic payload plans"
                )
            feature_key = (
                binding.value_type,
                binding.unit,
            )
            prior_feature = item_features.setdefault(
                (binding.item_type, binding.item_feature),
                feature_key,
            )
            if prior_feature != feature_key:
                issues.append(
                    f"item feature {binding.item_type}."
                    f"{binding.item_feature} has conflicting type/unit plans"
                )
        for obligation_id in sorted(
            set(obligations_by_id) - seen_binding_ids
        ):
            issues.append(
                f"{obligation_id} has no typed semantic binding"
            )

        # Source-derived semantic bindings own the typed data chain.  The
        # activated-constraint plan is the sole constraint writer. Reconcile
        # both views by semantic identity rather than by LLM-chosen names.
        explicit_constraints = [
            ConstraintPlan.from_dict(item, requirements=requirements)
            for item in (payload.get("constraints") or ())
            if isinstance(item, Mapping)
        ]
        semantic_by_id = {
            item.obligation_id: item for item in semantic_obligations
        }
        constraint_reconciliations: list[str] = []
        derived_constraints: list[ConstraintPlan] = []
        derived_attributes: dict[str, list[AttributePlan]] = {}
        explicit_by_semantics: dict[
            tuple[str, str, str, str, str], list[ConstraintPlan]
        ] = {}
        for constraint in explicit_constraints:
            semantic_key = (
                constraint.source_requirement_id or "",
                constraint.owner,
                constraint.lhs,
                constraint.operator,
                constraint.rhs,
            )
            explicit_by_semantics.setdefault(semantic_key, []).append(
                constraint
            )
        for semantic_key, matches in explicit_by_semantics.items():
            if len(matches) > 1:
                issues.append(
                    "duplicate semantic constraint identity "
                    + "::".join(semantic_key)
                    + ": "
                    + ", ".join(item.constraint_id for item in matches)
                )

        reconciled_bindings: list[SemanticBindingPlan] = []
        for binding in semantic_bindings:
            obligation = semantic_by_id.get(binding.obligation_id)
            if obligation is None:
                reconciled_bindings.append(binding)
                continue
            derived_attributes.setdefault(
                binding.target_component, []
            ).extend([
                AttributePlan(
                    name=binding.runtime_attribute,
                    value_type=binding.value_type,
                    unit=binding.unit,
                    role="RUNTIME_MEASUREMENT",
                    input_binding=binding.source_path,
                    provenance="FROZEN_REQUIREMENT",
                    source_requirement_id=binding.requirement_id,
                    source_digest=obligation.source_digest,
                ),
                AttributePlan(
                    name=binding.threshold_attribute,
                    value_type=binding.value_type,
                    unit=binding.unit,
                    role="FROZEN_THRESHOLD",
                    initial_value=(
                        f"{obligation.threshold:g} [{obligation.unit}]"
                    ),
                    provenance="FROZEN_REQUIREMENT",
                    source_requirement_id=binding.requirement_id,
                    source_digest=obligation.source_digest,
                ),
            ])
            semantic_key = (
                binding.requirement_id,
                binding.target_component,
                binding.runtime_attribute,
                obligation.operator,
                binding.threshold_attribute,
            )
            matches = explicit_by_semantics.get(semantic_key, [])
            if len(matches) == 1:
                canonical = matches[0]
                if (
                    obligation.activation_kind == "CONTEXTUAL"
                    and canonical.activation_kind != "STATE_ACTIVE"
                ):
                    issues.append(
                        f"{binding.obligation_id} preserves contextual clause "
                        f"{obligation.activation_clause!r} but its canonical "
                        "constraint is not STATE_ACTIVE"
                    )
                if binding.constraint_name != canonical.constraint_id:
                    constraint_reconciliations.append(
                        f"{binding.target_component}."
                        f"{binding.constraint_name} -> "
                        f"{canonical.constraint_id} "
                        f"({binding.obligation_id})"
                    )
                    binding = replace(
                        binding,
                        constraint_name=canonical.constraint_id,
                    )
            elif not matches:
                if obligation.activation_kind == "CONTEXTUAL":
                    issues.append(
                        f"{binding.obligation_id} contextual bound "
                        f"{obligation.activation_clause!r} requires exactly "
                        "one matching STATE_ACTIVE constraint"
                    )
                else:
                    derived_constraints.append(ConstraintPlan(
                        constraint_id=binding.constraint_name,
                        owner=binding.target_component,
                        lhs=binding.runtime_attribute,
                        operator=obligation.operator,
                        rhs=binding.threshold_attribute,
                        activation_kind="ALWAYS",
                        provenance="FROZEN_REQUIREMENT",
                        verification_tier="PARAMETRIC_SWEEP",
                        source_requirement_id=binding.requirement_id,
                        source_digest=obligation.source_digest,
                    ))
            reconciled_bindings.append(binding)
        semantic_bindings = reconciled_bindings
        if derived_attributes:
            enriched_components: list[ComponentPlan] = []
            for component in components:
                by_name = {
                    item.name: item for item in component.attributes
                }
                for item in derived_attributes.get(component.name, ()):
                    by_name[item.name] = item
                enriched_components.append(replace(
                    component,
                    attributes=tuple(by_name.values()),
                ))
            components = enriched_components
        constraint_plans = tuple([
            *explicit_constraints,
            *derived_constraints,
        ])
        issues.extend(validate_constraint_plan(
            constraint_plans,
            components,
            requirements,
        ))
        raw_behaviors = payload.get("behaviors")
        archived_schema = str(payload.get("schema_version") or "").strip()
        legacy_behavior_schema = archived_schema in {
            "1.0", "3.0", "5.0", "6.0", "7.0", "8.0",
        }
        if not isinstance(raw_behaviors, Sequence) or isinstance(
            raw_behaviors, (str, bytes)
        ):
            raw_behaviors = ()
            if any(
                item.activation_kind == "STATE_ACTIVE"
                for item in constraint_plans
            ) and not legacy_behavior_schema:
                issues.append(
                    "behaviors must declare the typed state machine for "
                    "every STATE_ACTIVE constraint"
                )
        planned_behaviors = tuple(
            PlannedBehavior.from_dict(item, requirements=requirements)
            for item in raw_behaviors
            if isinstance(item, Mapping)
        )
        issues.extend(validate_planned_behaviors(
            planned_behaviors,
            component_names={item.name for item in components},
            component_port_names={
                item.name: {port.name for port in item.ports}
                for item in components
            },
            requirements=requirements,
            state_active_constraints=tuple(
                item for item in constraint_plans
                if item.activation_kind == "STATE_ACTIVE"
            ) if not legacy_behavior_schema else (),
            require_executable_responses=not legacy_behavior_schema,
        ))
        ordinary_event_symbols = collect_planned_event_symbols(
            planned_behaviors,
            components=components,
        )
        issues.extend(validate_planned_event_symbols(
            ordinary_event_symbols,
            planned_behaviors,
            components=components,
        ))
        raw_realizations = payload.get("requirement_realizations")
        if not isinstance(raw_realizations, Sequence) or isinstance(
            raw_realizations, (str, bytes)
        ):
            raw_realizations = ()
            if require_source_anchored_paths:
                issues.append("requirement_realizations must be a list")
        requirement_realizations = tuple(
            RequirementRealizationPlan.from_dict(item)
            for item in raw_realizations
            if isinstance(item, Mapping)
        )
        if requirements and requirement_realizations:
            requirement_source_by_id: dict[str, str] = {}
            for requirement in requirements:
                source_text = str(requirement or "").strip()
                for req_id in _req_ids((source_text,)):
                    requirement_source_by_id[req_id] = source_text
            requirement_realizations = tuple(
                replace(
                    item,
                    source_digest=hashlib.sha256(
                        requirement_source_by_id[item.requirement_id].encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                )
                if item.requirement_id in requirement_source_by_id
                else item
                for item in requirement_realizations
            )
        if requirement_realizations or require_source_anchored_paths:
            structural_obligations, structural_issues = (
                compile_source_anchored_structural_obligations(
                    requirement_realizations,
                    components,
                    connections,
                    requirements=requirements,
                    require_complete=require_source_anchored_paths,
                    planned_behaviors=planned_behaviors,
                    behavior_obligations=(
                        ag_behavior_plan.obligations
                        if ag_behavior_plan is not None
                        else behavior_obligations
                    ),
                )
            )
        else:
            # Compatibility for archived plans and test doubles created before
            # schema 6.0. New provider calls must declare source anchors.
            structural_obligations, structural_issues = (
                compile_structural_obligations(
                    components,
                    connections,
                    allocated_requirements=allocated_requirements,
                )
            )
        realization_by_requirement = {
            item.requirement_id: item
            for item in requirement_realizations
        }
        for binding in semantic_bindings:
            realization = realization_by_requirement.get(
                binding.requirement_id
            )
            if realization is None:
                continue
            prefix = (
                f"source-derived semantic realization "
                f"{binding.requirement_id}"
            )
            if (
                realization.realization_kind != "CAUSAL_PATH"
                or not realization.connection_path
            ):
                issues.append(
                    f"{prefix} must be a CAUSAL_PATH aligned with "
                    f"{binding.obligation_id}"
                )
                continue
            first = realization.connection_path[0]
            last = realization.connection_path[-1]
            actual_source = (
                first.source_component,
                first.source_port,
            )
            expected_source = (
                binding.source_component,
                binding.source_port,
            )
            if actual_source != expected_source:
                issues.append(
                    f"{prefix} must start at semantic binding source "
                    f"{binding.source_component}.{binding.source_port}, "
                    f"found {first.source_component}.{first.source_port}"
                )
            actual_target = (
                last.target_component,
                last.target_port,
            )
            expected_target = (
                binding.target_component,
                binding.target_port,
            )
            if actual_target != expected_target:
                issues.append(
                    f"{prefix} must end at semantic binding target "
                    f"{binding.target_component}.{binding.target_port}, "
                    f"found {last.target_component}.{last.target_port}"
                )
        issues.extend(structural_issues)
        return cls(
            components=tuple(components),
            connections=tuple(connections),
            requirement_realizations=requirement_realizations,
            structural_obligations=structural_obligations,
            semantic_obligations=semantic_obligations,
            semantic_bindings=tuple(semantic_bindings),
            constraint_plans=constraint_plans,
            planned_behaviors=planned_behaviors,
            behavior_obligations=behavior_obligations,
            behavior_identity_reconciliations=(
                behavior_identity_reconciliations
            ),
            constraint_identity_reconciliations=tuple(
                constraint_reconciliations
            ),
            source=source,
            issues=tuple(dict.fromkeys(issues)),
            schema_version=(
                archived_schema
                if planned_behaviors and archived_schema == "8.0"
                else "9.0"
                if planned_behaviors
                else "7.0"
                if constraint_plans
                else "6.0"
                if requirement_realizations
                else "5.0"
                if semantic_obligations
                else "3.0"
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
                        f"{item.name} : {item.value_type} "
                        f"[{item.unit}] role={item.role} "
                        f"source={item.provenance}"
                        + (
                            f" initial={item.initial_value}"
                            if item.initial_value else ""
                        )
                        + (
                            f" binding={item.input_binding}"
                            if item.input_binding else ""
                        )
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
                "FROZEN SOURCE-ANCHORED CAUSAL OBLIGATIONS "
                "(validation input; do not add unrelated paths):"
            )
            for obligation in self.structural_obligations:
                path = " -> ".join(obligation.required_components)
                behavior = (
                    f"; behavior={obligation.behavior_kind} "
                    f"{obligation.behavior_name}"
                    if obligation.realization_kind == "LOCAL_BEHAVIOR"
                    else ""
                )
                lines.append(
                    f"- {obligation.obligation_id} "
                    f"[{obligation.requirement_id}]: {path} "
                    f"({obligation.entry_kind}); trigger="
                    f"{obligation.trigger_concept!r}; effect="
                    f"{obligation.effect_concept!r}{behavior}"
                )
        if self.semantic_obligations:
            lines.append("")
            lines.append(
                "FROZEN REQUIREMENT SEMANTIC OBLIGATIONS "
                "(model fidelity only; not physical proof):"
            )
            lines.extend([
                "- Represent each subject as a typed item feature delivered "
                "through a planned input port to the satisfying part.",
                "- Bind the constrained runtime attribute to that port item "
                "feature; a numeric literal placeholder is not a measurement.",
                "- Emit the canonical planned assert constraint at part scope "
                "for ALWAYS, or inside the referenced state body for "
                "STATE_ACTIVE; preserve the exact operator, threshold, and unit.",
                "- Any avoidance/maintenance transition must activate before "
                "the frozen boundary is violated.",
            ])
            for obligation in self.semantic_obligations:
                subject = "/".join(obligation.subject_terms)
                lines.append(
                    f"- {obligation.obligation_id} "
                    f"[{obligation.requirement_id}]: maintain {subject} "
                    f"{obligation.operator} {obligation.threshold:g} "
                    f"[{obligation.unit}]"
                )
        if self.semantic_bindings:
            lines.append("")
            lines.append(
                "FROZEN TYPED SEMANTIC BINDINGS "
                "(emit exactly; deterministic closure will enforce them):"
            )
            for binding in self.semantic_bindings:
                lines.extend([
                    f"- {binding.obligation_id}: "
                    f"{binding.source_component}.{binding.source_port} -> "
                    f"{binding.target_component}.{binding.target_port}",
                    f"  payload {binding.port_type}."
                    f"{binding.port_feature} : {binding.item_type}; "
                    f"{binding.item_type}.{binding.item_feature} : "
                    f"{binding.value_type} [{binding.unit}]",
                    f"  bind {binding.target_component}."
                    f"{binding.runtime_attribute} = {binding.source_path}; "
                    f"threshold {binding.threshold_attribute}; "
                    f"constraint {binding.constraint_name}",
                ])
        if self.constraint_plans:
            lines.append("")
            lines.append(
                "TYPED ACTIVATED CONSTRAINT PLAN "
                "(emit each item once at its declared activation scope; "
                "never invent another assert):"
            )
            for constraint in self.constraint_plans:
                source = (
                    f"; source={constraint.source_requirement_id}"
                    if constraint.source_requirement_id else ""
                )
                lines.append(
                    f"- {constraint.owner}.{constraint.constraint_id}: "
                    f"{constraint.expression}; "
                    f"activation={constraint.activation_kind}"
                    + (
                        f"({constraint.activation_ref})"
                        if constraint.activation_ref else ""
                    )
                    + f"; provenance={constraint.provenance}; "
                    f"verification={constraint.verification_tier}{source}"
                )
        if self.planned_behaviors:
            lines.append("")
            lines.append(
                "TYPED BEHAVIOR IDENTITY PLAN "
                "(compiler-owned SysML v2 member identities; emit exactly):"
            )
            for behavior in self.planned_behaviors:
                lines.append(
                    f"- owner={behavior.owner}; behavior="
                    f"{behavior.behavior_id}; initial="
                    f"{behavior.initial_state}; source="
                    f"{behavior.source_requirement_id or behavior.provenance}"
                )
                for state in behavior.states:
                    lines.append(
                        f"  state {behavior.behavior_id}::"
                        f"{state.state_id}; role={state.role}"
                        + (
                            f"; entry_action={state.entry_action}"
                            if state.entry_action else ""
                        )
                        + (
                            f"; do_action={state.do_action}"
                            if state.do_action else ""
                        )
                    )
                for transition in behavior.transitions:
                    lines.append(
                        f"  transition {transition.transition_id}: "
                        f"{transition.source} -> {transition.target}; "
                        f"{transition.trigger_kind} "
                        f"{transition.trigger}"
                    )
        if self.behavior_obligations:
            lines.append("")
            lines.append(
                BehaviorObligationPlan(
                    self.behavior_obligations
                ).render_for_prompt()
            )
        if self.planned_event_symbols:
            lines.append("")
            lines.append(
                "PLANNED EVENT SYMBOLS "
                "(package-level, compiler-owned, exact SysML v2 kind):"
            )
            lines.extend(
                f"- {symbol.name}: item def"
                for symbol in self.planned_event_symbols
            )
            lines.append(
                "Transitions may accept these event item types. Do not "
                "declare action/attribute/port/state definitions with these "
                "names; executable entry/do responses use separate action "
                "definitions."
            )
            lines.append(
                "An ACCEPT transition must reference one of these exact item "
                "classifier names. Never use an owner port name as an accept "
                "target; ports carry structure, not event-type identity."
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
    # An A/G concept used as a Boolean operand is typed by the contract, not by
    # generation — a guard or invariant reading `not (airborne)` is meaningless
    # over a Real. Nothing owned that type before: these concepts come from the
    # frozen A/G decisions, so the typed plan never carried them and generation
    # was free to write `attribute airborne : Real = 0.0;`. It did, in two
    # measured runs, and the only thing left to catch it was the terminal gate,
    # which could then only fail the whole run.
    #
    # Declaring them here puts them under the existing planned-attribute
    # materialiser, so the type is enforced during generation and the terminal
    # gate goes back to being a check that should never fire rather than the
    # first line of defence. A concept the component already plans is left
    # alone: the plan is authority for its own attributes.
    #
    # Only what the component CONSUMES. A guarantee is an output: it flows to the
    # component that assumes it, so generation realises it as a directed port,
    # and the terminal binder accepts a port as a valid carrier of the truth
    # concept. Planning it as an attribute as well produced exactly the collision
    # the binder reports as AMBIGUOUS — one measured run had
    # `recoveryActuationPowerAvailable` as both.
    boolean_by_owner: dict[str, set[str]] = {}
    for obligation in behavior_plan.obligations:
        consumed = set(obligation.assumptions) - set(obligation.guarantees)
        for concept in behavior_boolean_concepts(obligation):
            if concept in consumed:
                boolean_by_owner.setdefault(
                    obligation.owner_def, set()
                ).add(concept)
    if boolean_by_owner:
        components = []
        for component in plan.components:
            wanted = boolean_by_owner.get(component.name, set())
            # a name the plan already uses for a port is that port's, not ours
            existing = (
                {item.name for item in component.attributes}
                | {item.name for item in component.ports}
            )
            missing = sorted(wanted - existing)
            if not missing:
                components.append(component)
                continue
            components.append(replace(component, attributes=(
                *component.attributes,
                *(
                    AttributePlan(
                        name=concept,
                        value_type="Boolean",
                        unit="1",
                        role="LOCAL_STATE",
                        provenance="DESIGN_DECISION",
                    )
                    for concept in missing
                ),
            )))
        plan = replace(plan, components=tuple(components))

    reserved = {
        (item["owner_def"], item["name"]): item
        for item in reserved_behavior_identities(behavior_plan)
    }
    for behavior in plan.planned_behaviors:
        identity = reserved.get((behavior.owner, behavior.behavior_id))
        if identity is not None and identity["kind"] != "state def":
            issues.append(
                "ordinary planned behavior "
                f"{behavior.owner}::{behavior.behavior_id} collides with "
                "frozen A/G reserved identity "
                f"(expected {identity['kind']})"
            )
    combined_event_symbols = collect_planned_event_symbols(
        plan.planned_behaviors,
        behavior_plan.obligations,
        plan.components,
    )
    issues.extend(validate_planned_event_symbols(
        combined_event_symbols,
        plan.planned_behaviors,
        behavior_plan.obligations,
        components=plan.components,
    ))

    # A source-anchored LOCAL_BEHAVIOR and an A/G state-machine obligation for
    # the same requirement and owner identify one model element, not two
    # independently named behaviors.  The A/G plan is frozen before ordinary
    # generation, so its stable id is the canonical identity.  Reconcile that
    # notation deterministically and keep an explicit audit record.
    reconciliations = list(plan.behavior_identity_reconciliations)
    canonical_names: dict[tuple[str, str], str] = {}
    for obligation in behavior_plan.obligations:
        if obligation.realization_kind != "STATE_MACHINE":
            continue
        key = (
            normalise_req_id(obligation.requirement_id),
            obligation.owner_def,
        )
        prior = canonical_names.setdefault(
            key, obligation.stable_behavior_id
        )
        if prior != obligation.stable_behavior_id:
            issues.append(
                f"ambiguous A/G behavior identity for {key[0]} "
                f"at {key[1]}: {prior}, "
                f"{obligation.stable_behavior_id}"
            )

    realization_names: dict[tuple[str, str], str] = {}
    reconciled_realizations: list[RequirementRealizationPlan] = []
    for realization in plan.requirement_realizations:
        key = (
            normalise_req_id(realization.requirement_id),
            realization.owner_component,
        )
        canonical = canonical_names.get(key)
        if (
            realization.realization_kind == "LOCAL_BEHAVIOR"
            and realization.behavior_kind == "STATE_DEF"
            and canonical
        ):
            realization_names[key] = canonical
            if realization.behavior_name != canonical:
                reconciliations.append(
                    f"{key[0]}::{key[1]}::{realization.behavior_name} -> "
                    f"{canonical}"
                )
                realization = replace(
                    realization,
                    behavior_name=canonical,
                )
        reconciled_realizations.append(realization)

    reconciled_structural: list[StructuralObligation] = []
    for obligation in plan.structural_obligations:
        key = (
            normalise_req_id(obligation.requirement_id),
            obligation.source_component,
        )
        canonical = realization_names.get(key)
        if (
            obligation.realization_kind == "LOCAL_BEHAVIOR"
            and obligation.behavior_kind == "STATE_DEF"
            and canonical
        ):
            obligation = replace(
                obligation,
                behavior_name=canonical,
            )
        reconciled_structural.append(obligation)

    return replace(
        plan,
        requirement_realizations=tuple(reconciled_realizations),
        structural_obligations=tuple(reconciled_structural),
        behavior_obligations=behavior_plan.obligations,
        behavior_identity_reconciliations=tuple(dict.fromkeys(
            reconciliations
        )),
        issues=tuple(dict.fromkeys(issues)),
        schema_version=(
            plan.schema_version
            if plan.planned_behaviors and plan.schema_version == "8.0"
            else "9.0"
            if plan.planned_behaviors
            else "7.0"
            if plan.constraint_plans
            else "6.0"
            if plan.requirement_realizations
            else "5.0"
            if plan.semantic_obligations
            else "3.0"
        ),
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
    from ..simulation.port_fixer import (
        PortAdd,
        collect_port_defs,
        merge_port_additions,
        validate_port_additions,
    )

    semantic_text, semantic_binding_conformance = (
        materialize_semantic_bindings(
            model_text,
            plan.semantic_bindings,
            plan.semantic_obligations,
        )
    )
    semantic_text, attribute_conformance = materialize_planned_attributes(
        semantic_text,
        plan.components,
    )
    directory = build_port_directory(semantic_text)
    instances_by_type: dict[str, list[str]] = {}
    for instance, component_type in directory.instance_type.items():
        instances_by_type.setdefault(component_type, []).append(instance)

    issues = list(plan.issues)
    if semantic_binding_conformance["status"] == "FAIL":
        issues.extend(
            "semantic binding: " + item
            for item in semantic_binding_conformance["issues"]
        )
    if attribute_conformance["status"] == "FAIL":
        issues.extend(
            "planned attribute: " + item
            for item in attribute_conformance["missing_attributes"]
        )
    resolved_usages: dict[str, str | None] = {}
    candidate_lines: list[str] = []
    expected_keys: list[tuple[str, str, str, str]] = []

    def resolve(component: str) -> str | None:
        if component in resolved_usages:
            return resolved_usages[component]
        candidates = instances_by_type.get(component, ())
        if len(candidates) == 1:
            resolved = candidates[0]
        else:
            conventional = component[:1].lower() + component[1:]
            if conventional in candidates:
                resolved = conventional
            else:
                issues.append(
                    f"cannot uniquely resolve usage for component {component}"
                )
                resolved = None
        resolved_usages[component] = resolved
        return resolved

    # The plan owns a planned port's type; correct it before deciding what to
    # add, so a type-mismatched port is repaired rather than reported as both
    # missing and unplanned.
    semantic_text, retyped_ports = normalise_planned_port_types(
        semantic_text, plan.components
    )

    planned_port_additions: list[PortAdd] = []
    for component in plan.components:
        usage = resolve(component.name)
        if usage is None:
            continue
        existing_ports = directory.instances.get(usage, {})
        for port in component.ports:
            if port.name not in existing_ports:
                planned_port_additions.append(PortAdd(
                    part_def=component.name,
                    direction=port.direction,
                    name=port.name,
                    port_type=port.port_type,
                ))
    port_validation = validate_port_additions(
        planned_port_additions,
        directory,
        collect_port_defs(semantic_text),
    )
    port_merge = merge_port_additions(
        semantic_text,
        port_validation.accepted,
    )
    issues.extend(
        f"planned port rejected: {line}: {reason}"
        for line, reason in port_validation.rejected
    )
    working_text = port_merge.merged_text
    directory = build_port_directory(working_text)
    existing = parse_connects(working_text)

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
    merged = merge_connects(working_text, validation.accepted)
    behavior_text, planned_behavior_conformance = (
        materialize_owned_planned_behaviors(
            merged.merged_text,
            plan.planned_behaviors,
            event_symbols=plan.planned_event_symbols,
        )
        if plan.planned_behaviors
        else (
            merged.merged_text,
            {
                "artifact_role": "OWNED_PLANNED_BEHAVIOR_CONFORMANCE",
                "status": "NOT_APPLICABLE",
                "checked": [],
                "issues": [],
                "materialized": [],
            },
        )
    )
    if planned_behavior_conformance["status"] == "FAIL":
        issues.extend(
            "planned behavior: " + item
            for item in planned_behavior_conformance["issues"]
        )
    constraint_text, constraint_conformance = materialize_planned_constraints(
        behavior_text,
        plan.constraint_plans,
    )
    if constraint_conformance["status"] != "PASS":
        issues.extend(
            "activated constraint: " + item
            for item in (
                constraint_conformance["missing_constraints"]
                + constraint_conformance["activation_issues"]
            )
        )
    if plan.behavior_obligations:
        final_text, ag_behavior_conformance = (
            materialize_owned_behavior_obligations(
                constraint_text,
                BehaviorObligationPlan(plan.behavior_obligations),
                event_symbols=plan.planned_event_symbols,
            )
        )
    else:
        final_text = constraint_text
        ag_behavior_conformance = {
            "artifact_role": "A_G_OWNED_BEHAVIOR_MATERIALIZATION",
            "status": "NOT_APPLICABLE",
            "issues": [],
            "checked": [],
            "materialized": [],
            "replaced_inconsistent": [],
            "removed_kind_conflicts": [],
            "reserved_identity_conformance": {
                "artifact_role": "A_G_RESERVED_IDENTITY_CONFORMANCE",
                "status": "NOT_APPLICABLE",
                "checked": [],
                "issues": [],
            },
        }
    if ag_behavior_conformance["status"] == "FAIL":
        issues.extend(
            "A/G reserved behavior: " + item
            for item in ag_behavior_conformance["issues"]
        )
    final_text, event_symbol_conformance = (
        materialize_planned_event_symbols(
            final_text,
            plan.planned_event_symbols,
        )
    )
    if event_symbol_conformance["status"] == "FAIL":
        issues.extend(
            "planned event symbol: " + item
            for item in event_symbol_conformance["issues"]
        )
    final_text, standard_import_conformance = (
        materialize_standard_library_imports(final_text)
    )
    if standard_import_conformance["status"] == "FAIL":
        issues.extend(
            "standard library import: " + item
            for item in standard_import_conformance["issues"]
        )
    final_directory = build_port_directory(final_text)
    package_match = re.search(
        r"\bpackage\s+([A-Za-z_]\w*)\s*\{", semantic_text
    )
    root_package = package_match.group(1) if package_match else None
    scoped_graph = extract_behavioral_graph(
        final_text,
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
        for item in parse_connects(final_text)
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
    definition_contract = _definition_contract_report(
        final_text,
        plan,
    )
    issues.extend(
        f"planned definition missing: {item}"
        for item in definition_contract["missing_definitions"]
    )
    issues.extend(
        f"definition kind conflict: {item}"
        for item in definition_contract["definition_kind_conflicts"]
    )
    issues.extend(
        f"unplanned part definition: {item}"
        for item in definition_contract["unplanned_part_definitions"]
    )
    report = {
        "schema_version": "4.0",
        "artifact_role": "GENERATION_PLAN_CONFORMANCE",
        "input_model_digest": hashlib.sha256(
            str(model_text or "").encode("utf-8")
        ).hexdigest(),
        "output_model_digest": hashlib.sha256(
            final_text.encode("utf-8")
        ).hexdigest(),
        "status": (
            "PASS"
            if plan.status == "PASS" and not issues and not missing
            else "FAIL"
        ),
        "plan_status": plan.status,
        "planned_connection_count": len(plan.connections),
        "realized_connection_count": len(expected_keys) - len(missing),
        "deterministically_added_connections": list(merged.added_lines),
        "deterministically_added_ports": list(
            port_merge.added_descriptions
        ),
        "deterministically_retyped_ports": list(retyped_ports),
        "semantic_binding_conformance": (
            semantic_binding_conformance
        ),
        "planned_attribute_conformance": attribute_conformance,
        "planned_behavior_conformance": planned_behavior_conformance,
        "activated_constraint_conformance": constraint_conformance,
        "ag_behavior_conformance": ag_behavior_conformance,
        "ag_reserved_identity_conformance": (
            ag_behavior_conformance[
                "reserved_identity_conformance"
            ]
        ),
        "planned_event_symbol_conformance": event_symbol_conformance,
        "standard_library_import_conformance": (
            standard_import_conformance
        ),
        "missing_connections": missing,
        "unplanned_connections": unplanned_connections,
        "missing_components": missing_components,
        "duplicate_component_usages": duplicate_component_usages,
        "unplanned_components": unplanned_components,
        "missing_ports": missing_ports,
        "unplanned_ports": unplanned_ports,
        "internalized_external_ports": internalized_external_ports,
        "definition_contract": definition_contract,
        "issues": list(dict.fromkeys(issues)),
    }
    return final_text, report
