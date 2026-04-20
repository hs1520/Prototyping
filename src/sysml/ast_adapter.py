"""Adapter that maps a SysML v2 JSON AST into the project's SysMLModel IR."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

from .model import (
    Action,
    Attribute,
    Block,
    Connector,
    FeatureDirection,
    Multiplicity,
    Port,
    Requirement,
    SysMLElement,
    SysMLModel,
)


class SysMLAstMappingError(RuntimeError):
    """Raised when JSON AST cannot be mapped into SysMLModel."""


@dataclass
class SysMLAstAdapter:
    """Translate Java-side AST payloads into the Python IR used by the pipeline."""

    def ast_to_model(
        self,
        ast: Mapping[str, Any],
        *,
        existing_model: Optional[SysMLModel] = None,
    ) -> SysMLModel:
        if not isinstance(ast, Mapping):
            raise SysMLAstMappingError(
                f"AST payload must be a mapping, got {type(ast).__name__}."
            )

        schema_version = ast.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise SysMLAstMappingError("AST payload missing non-empty 'schema_version'.")
        if ast.get("status") != "ok":
            raise SysMLAstMappingError(
                f"AST payload status must be 'ok', got {ast.get('status')!r}."
            )
        if not isinstance(ast.get("ast"), Mapping):
            raise SysMLAstMappingError("AST payload missing mapping field 'ast'.")

        root = self._unwrap_root(ast)
        model = existing_model or SysMLModel(
            name=self._first_str(root, "name", "packageName", "qualifiedName", default="UnnamedSystem"),
            description=self._first_str(root, "description", "doc", default=""),
        )

        self._map_model_metadata(model, ast, root)
        self._map_model_scaffolding(model, root)

        for req_node in self._collect_nodes(root, "requirements", "ownedRequirements", "requirementNodes"):
            requirement = self._map_requirement(req_node)
            self._add_or_merge_requirement(model, requirement)

        for block_node in self._collect_nodes(root, "blocks", "partDefs", "parts", "components", "ownedElements"):
            block = self._map_block(block_node)
            self._add_or_merge_block(model, block)

        for conn_node in self._collect_nodes(root, "connectors", "connections", "links"):
            connector = self._map_connector(conn_node)
            self._add_or_merge_connector(model, connector)

        for constraint in self._collect_string_nodes(root, "constraints", "ownedRules"):
            model.add_constraint(constraint)
        for expression in self._collect_string_nodes(root, "expressions"):
            model.add_expression(expression)
        for package in self._collect_string_nodes(root, "packages", "ownedPackages"):
            model.add_package(package)
        for import_name in self._collect_string_nodes(root, "imports", "ownedImports"):
            model.add_import(import_name)
        for generalization in self._collect_string_nodes(root, "generalizations"):
            model.add_generalization(generalization)
        for specialization in self._collect_string_nodes(root, "specializations"):
            model.add_specialization(specialization)

        for diagnostic in self._collect_nodes(root, "diagnostics", "parseDiagnostics", "warnings"):
            model.add_diagnostic(self._normalize_node(diagnostic))

        if "confidence" in root:
            try:
                model.confidence = float(root["confidence"])
            except (TypeError, ValueError):
                model.add_mapping_note(f"Invalid confidence value ignored: {root['confidence']!r}")
        if "astVersion" in root:
            model.ast_version = str(root["astVersion"])
        if "sourceUri" in root:
            model.source_uri = str(root["sourceUri"])

        model.metadata.setdefault("raw_ast", copy.deepcopy(ast))
        model.metadata.setdefault("ast_root", copy.deepcopy(root))
        model.metadata.setdefault("ast_adapter", "python-json-ast-v1")
        return model

    def _unwrap_root(self, ast: Mapping[str, Any]) -> Mapping[str, Any]:
        for key in ("ast", "root", "model", "data", "result", "payload"):
            candidate = ast.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        return ast

    def _map_model_metadata(self, model: SysMLModel, ast: Mapping[str, Any], root: Mapping[str, Any]) -> None:
        model.namespace = self._first_str(root, "namespace", "package", default=model.namespace)
        model.qualified_name = self._first_str(root, "qualifiedName", "qualified_name", default=model.qualified_name)
        model.source_uri = self._first_str(root, "sourceUri", "source_uri", default=model.source_uri)
        if "description" in root and not model.description:
            model.description = self._first_str(root, "description", "doc", default=model.description)
        if "confidence" in ast and not model.confidence:
            try:
                model.confidence = float(ast["confidence"])
            except (TypeError, ValueError):
                pass
        if "schema_version" in ast and not model.ast_version:
            model.ast_version = str(ast["schema_version"])

    def _map_model_scaffolding(self, model: SysMLModel, root: Mapping[str, Any]) -> None:
        for key in ("imports", "ownedImports"):
            for item in self._collect_string_nodes(root, key):
                model.add_import(item)
        for key in ("packages", "ownedPackages"):
            for item in self._collect_string_nodes(root, key):
                model.add_package(item)

    def _map_requirement(self, node: Mapping[str, Any]) -> Requirement:
        requirement = Requirement(
            name=self._first_str(node, "name", "id", default="UnnamedRequirement"),
            text=self._first_str(node, "text", "doc", "description", default=""),
            short_description=self._first_str(node, "shortDescription", "short_description", default=""),
        )
        requirement.namespace = self._first_str(node, "namespace", default="")
        requirement.qualified_name = self._first_str(node, "qualifiedName", "qualified_name", default="")
        requirement.source_uri = self._first_str(node, "sourceUri", "source_uri", default="")
        requirement.status = self._first_str(node, "status", default="")
        requirement.parent_id = self._first_str(node, "parentId", "parent_id", default="") or None
        requirement.satisfaction_level = float(self._first_number(node, "satisfactionLevel", "satisfaction_level", default=0.0))
        requirement.derived_from = self._collect_string_nodes(node, "derivedFrom", "derived_from")
        requirement.refined_by = self._collect_string_nodes(node, "refinedBy", "refined_by")
        requirement.constraints = self._collect_string_nodes(node, "constraints", "ownedRules")
        requirement.source_span = self._source_span(node)
        requirement.metadata.update(self._element_metadata(node))
        return requirement

    def _map_block(self, node: Mapping[str, Any]) -> Block:
        block = Block(
            name=self._first_str(node, "name", "id", default="UnnamedBlock"),
            block_type=self._first_str(node, "kind", "type", default="part def"),
            short_description=self._first_str(node, "shortDescription", "short_description", "doc", default=""),
        )
        block.namespace = self._first_str(node, "namespace", default="")
        block.qualified_name = self._first_str(node, "qualifiedName", "qualified_name", default="")
        block.visibility = self._first_str(node, "visibility", default=block.visibility)
        block.is_abstract = bool(node.get("isAbstract", node.get("abstract", False)))
        block.is_final = bool(node.get("isFinal", node.get("final", False)))
        block.source_uri = self._first_str(node, "sourceUri", "source_uri", default="")
        block.source_span = self._source_span(node)
        block.satisfies = self._collect_string_nodes(node, "satisfies", "satisfiedRequirements")
        block.refines = self._collect_string_nodes(node, "refines", "refinedBy")
        block.generalizations = self._collect_string_nodes(node, "generalizations", "generalizes")
        block.specializations = self._collect_string_nodes(node, "specializations", "specializes")
        block.constraints = self._collect_string_nodes(node, "constraints", "ownedRules")
        block.expressions = self._collect_string_nodes(node, "expressions")
        block.imports = self._collect_string_nodes(node, "imports", "ownedImports")
        block.metadata.update(self._element_metadata(node))

        for port_node in self._collect_nodes(node, "ports", "ownedPorts", "features"):
            block.add_port(self._map_port(port_node))
        for attr_node in self._collect_nodes(node, "attributes", "valueProperties", "ownedAttributes"):
            block.add_attribute(self._map_attribute(attr_node))
        for action_node in self._collect_nodes(node, "actions", "behaviors", "ownedBehaviors"):
            block.add_action(self._map_action(action_node))
        for child_node in self._collect_nodes(node, "children", "ownedElements", "parts", "nestedBlocks", "subParts"):
            child = self._map_child_block(child_node)
            if child is not None:
                block.add_child(child)
                if child.name and child.name not in block.satisfies and child.block_type.lower().startswith("part"):
                    block.add_sub_part(child)

        return block

    def _map_child_block(self, node: Mapping[str, Any]) -> Optional[Block]:
        kind = self._first_str(node, "kind", "type", default="").lower()
        if kind and not any(token in kind for token in ("part", "block", "component")):
            return None
        return self._map_block(node)

    def _map_port(self, node: Mapping[str, Any]) -> Port:
        direction = self._normalize_direction(self._first_str(node, "direction", default="inout"))
        multiplicity = self._normalize_multiplicity(self._first_str(node, "multiplicity", default="1"))
        port = Port(
            name=self._first_str(node, "name", "id", default="UnnamedPort"),
            direction=direction,
            port_type=self._first_str(node, "portType", "port_type", "type", default=""),
            multiplicity=multiplicity,
            conjugated=bool(node.get("conjugated", node.get("isConjugated", False))),
            short_description=self._first_str(node, "shortDescription", "short_description", "doc", default=""),
        )
        port.qualified_name = self._first_str(node, "qualifiedName", "qualified_name", default="")
        port.type_path = self._first_str(node, "typePath", "type_path", default="")
        port.visibility = self._first_str(node, "visibility", default=port.visibility)
        port.source_uri = self._first_str(node, "sourceUri", "source_uri", default="")
        port.source_span = self._source_span(node)
        port.metadata.update(self._element_metadata(node))
        return port

    def _map_attribute(self, node: Mapping[str, Any]) -> Attribute:
        attribute = Attribute(
            name=self._first_str(node, "name", "id", default="UnnamedAttribute"),
            attribute_type=self._first_str(node, "attributeType", "attribute_type", "type", default="Real"),
            default_value=node.get("defaultValue", node.get("default_value")),
            unit=self._first_str(node, "unit", default=""),
            direction=self._normalize_direction(self._first_str(node, "direction", default="none")),
            short_description=self._first_str(node, "shortDescription", "short_description", "doc", default=""),
        )
        attribute.qualified_name = self._first_str(node, "qualifiedName", "qualified_name", default="")
        attribute.type_path = self._first_str(node, "typePath", "type_path", default="")
        attribute.visibility = self._first_str(node, "visibility", default=attribute.visibility)
        attribute.is_read_only = bool(node.get("isReadOnly", node.get("readOnly", False)))
        attribute.source_uri = self._first_str(node, "sourceUri", "source_uri", default="")
        attribute.source_span = self._source_span(node)
        attribute.metadata.update(self._element_metadata(node))
        return attribute

    def _map_action(self, node: Mapping[str, Any]) -> Action:
        action = Action(
            name=self._first_str(node, "name", "id", default="UnnamedAction"),
            inputs=self._collect_string_nodes(node, "inputs", "inputParameters", "ownedInputs"),
            outputs=self._collect_string_nodes(node, "outputs", "outputParameters", "ownedOutputs"),
            description=self._first_str(node, "description", "doc", "shortDescription", default=""),
        )
        action.qualified_name = self._first_str(node, "qualifiedName", "qualified_name", default="")
        action.parameters = [dict(item) for item in self._collect_nodes(node, "parameters", "ownedParameters")]
        action.body = self._first_str(node, "body", "expression", default="")
        action.preconditions = self._collect_string_nodes(node, "preconditions", "ownedPreconditions")
        action.postconditions = self._collect_string_nodes(node, "postconditions", "ownedPostconditions")
        action.source_uri = self._first_str(node, "sourceUri", "source_uri", default="")
        action.source_span = self._source_span(node)
        action.metadata.update(self._element_metadata(node))
        return action

    def _map_connector(self, node: Mapping[str, Any]) -> Connector:
        connector = Connector(
            name=self._first_str(node, "name", "id", default="UnnamedConnector"),
            source_block_id=self._first_str(node, "sourceBlockId", "source_block_id", "source", default=""),
            source_port_id=self._first_str(node, "sourcePortId", "source_port_id", default=""),
            target_block_id=self._first_str(node, "targetBlockId", "target_block_id", "target", default=""),
            target_port_id=self._first_str(node, "targetPortId", "target_port_id", default=""),
            qualified_name=self._first_str(node, "qualifiedName", "qualified_name", default=""),
            connector_type=self._first_str(node, "kind", "type", default=""),
            multiplicity=self._first_str(node, "multiplicity", default=""),
            source_uri=self._first_str(node, "sourceUri", "source_uri", default=""),
        )
        connector.end_roles = self._first_mapping(node, "endRoles", "ends", default={})
        connector.source_span = self._source_span(node)
        connector.metadata.update(self._element_metadata(node))
        return connector

    def _add_or_merge_requirement(self, model: SysMLModel, requirement: Requirement) -> None:
        existing = model.get_requirement_by_name(requirement.name)
        if existing is None:
            model.add_requirement(requirement)
            return
        self._merge_dataclass(existing, requirement)

    def _add_or_merge_block(self, model: SysMLModel, block: Block) -> None:
        existing = model.get_block_by_name(block.name)
        if existing is None:
            model.add_block(block)
            return
        self._merge_dataclass(existing, block)
        self._merge_lists(existing.ports, block.ports, key="name", add_method=existing.add_port)
        self._merge_lists(existing.attributes, block.attributes, key="name", add_method=existing.add_attribute)
        self._merge_lists(existing.actions, block.actions, key="name", add_method=existing.add_action)
        for child in block.children:
            existing.add_child(child)

    def _add_or_merge_connector(self, model: SysMLModel, connector: Connector) -> None:
        key = (
            connector.source_block_id,
            connector.source_port_id,
            connector.target_block_id,
            connector.target_port_id,
        )
        existing_keys = {
            (c.source_block_id, c.source_port_id, c.target_block_id, c.target_port_id)
            for c in model.connectors
        }
        if key not in existing_keys:
            model.add_connector(connector)

    def _merge_dataclass(self, target: SysMLElement, source: SysMLElement) -> None:
        for field_name, value in source.__dict__.items():
            if field_name == "metadata":
                target.metadata.update(value)
                continue
            if field_name == "id":
                continue
            current = getattr(target, field_name, None)
            if current in (None, "", [], {}, False) and value not in (None, "", [], {}, False):
                setattr(target, field_name, value)

    def _merge_lists(self, existing: Sequence[Any], new_items: Sequence[Any], *, key: str, add_method: Any) -> None:
        existing_values = {getattr(item, key, None) for item in existing}
        for item in new_items:
            if getattr(item, key, None) not in existing_values:
                add_method(item)
                existing_values.add(getattr(item, key, None))

    def _collect_nodes(self, node: Mapping[str, Any], *keys: str) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for key in keys:
            raw = node.get(key)
            if isinstance(raw, Mapping):
                values.append(dict(raw))
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                values.extend([dict(item) for item in raw if isinstance(item, Mapping)])
        return values

    def _collect_string_nodes(self, node: Mapping[str, Any], *keys: str) -> List[str]:
        values: List[str] = []
        for key in keys:
            raw = node.get(key)
            if isinstance(raw, str):
                values.append(raw)
            elif isinstance(raw, Mapping):
                maybe_name = self._first_str(raw, "name", "qualifiedName", "qualified_name", default="")
                if maybe_name:
                    values.append(maybe_name)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    if isinstance(item, str):
                        values.append(item)
                    elif isinstance(item, Mapping):
                        maybe_name = self._first_str(item, "name", "qualifiedName", "qualified_name", default="")
                        if maybe_name:
                            values.append(maybe_name)
        deduped: List[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _first_str(self, node: Mapping[str, Any], *keys: str, default: str = "") -> str:
        for key in keys:
            value = node.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                return value
            return str(value)
        return default

    def _first_number(self, node: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = node.get(key)
            if value is None:
                continue
            try:
                return float(cast(Any, value))
            except (TypeError, ValueError):
                continue
        return default

    def _first_mapping(self, node: Mapping[str, Any], *keys: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for key in keys:
            value = node.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        return default or {}

    def _normalize_direction(self, raw: str) -> FeatureDirection:
        normalized = (raw or "").strip().lower()
        if normalized in {"in", "input"}:
            return FeatureDirection.IN
        if normalized in {"out", "output"}:
            return FeatureDirection.OUT
        if normalized in {"inout", "in_out", "bidirectional"}:
            return FeatureDirection.INOUT
        return FeatureDirection.NONE if normalized in {"none", ""} else FeatureDirection.INOUT

    def _normalize_multiplicity(self, raw: str) -> Multiplicity:
        normalized = (raw or "").strip()
        mapping = {
            "1": Multiplicity.ONE,
            "0..1": Multiplicity.ZERO_OR_ONE,
            "0..*": Multiplicity.ZERO_OR_MANY,
            "1..*": Multiplicity.ONE_OR_MANY,
        }
        return mapping.get(normalized, Multiplicity.ONE)

    def _source_span(self, node: Mapping[str, Any]) -> Dict[str, Any]:
        span = node.get("sourceSpan") or node.get("source_span") or node.get("span")
        return dict(span) if isinstance(span, Mapping) else {}

    def _element_metadata(self, node: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = {}
        for key, value in node.items():
            if key in {
                "name",
                "id",
                "kind",
                "type",
                "description",
                "doc",
                "shortDescription",
                "short_description",
                "namespace",
                "qualifiedName",
                "qualified_name",
                "sourceUri",
                "source_uri",
                "sourceSpan",
                "source_span",
                "span",
                "ports",
                "attributes",
                "actions",
                "connectors",
                "requirements",
                "blocks",
                "children",
                "ownedElements",
                "parts",
                "nestedBlocks",
                "subParts",
                "constraints",
                "expressions",
                "imports",
                "ownedImports",
                "packages",
                "ownedPackages",
                "generalizations",
                "specializations",
                "satisfies",
                "refines",
                "derivedFrom",
                "refinedBy",
                "diagnostics",
                "parseDiagnostics",
                "warnings",
                "confidence",
                "astVersion",
                "version",
            }:
                continue
            metadata[key] = copy.deepcopy(value)
        metadata.setdefault("ast_kind", self._first_str(node, "kind", "type", default=""))
        return metadata

    def _normalize_node(self, node: Mapping[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(dict(node))





