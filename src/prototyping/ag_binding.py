"""Bind planning-time A/G contracts to concrete terminal SysML elements."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .ag_emitter import AGChainSpec, ag_event_signals
from .ag_behavior_plan import (
    BehaviorObligationPlan,
    INVARIANT,
    behavior_boolean_concepts as _behavior_boolean_concepts,
    compile_behavior_obligation_plan,
)
from .ag_planning import (
    emit_ag_planning_package,
)
from ..sysml.text_normalization import strip_named_item_definitions
from ..utils.sysml_text_utils import find_block_end

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:  # pragma: no cover - exercised in fail-closed environments
    _syside = None  # type: ignore
    _SYSIDE_OK = False


TERMINAL_REALIZATION = "TERMINAL_REALIZATION"


@dataclass(frozen=True)
class FeatureTypeBinding:
    concept: str
    owner_feature: str
    observed_kind: str
    observed_type: str
    status: str
    issues: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "owner_feature": self.owner_feature,
            "observed_kind": self.observed_kind,
            "observed_type": self.observed_type,
            "expected_semantics": "BOOLEAN_BEHAVIOR_OPERAND",
            "status": self.status,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class ComponentBinding:
    contract: str
    owner_definition: str
    owner_usage: str
    behavior: str
    status: str
    realization_kind: str = "STATE_MACHINE"
    feature_type_bindings: Tuple[FeatureTypeBinding, ...] = ()
    issues: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "owner_definition": self.owner_definition,
            "owner_usage": self.owner_usage,
            "behavior": self.behavior,
            "realization_kind": self.realization_kind,
            "feature_type_bindings": [
                item.to_dict() for item in self.feature_type_bindings
            ],
            "status": self.status,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class EventTypeBinding:
    package: str
    event: str
    canonical_type: str
    expected_kind: str
    status: str
    issues: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package": self.package,
            "event": self.event,
            "canonical_type": self.canonical_type,
            "expected_kind": self.expected_kind,
            "status": self.status,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AGBindingReport:
    status: str
    system_package: str
    profile: str
    bindings: Tuple[ComponentBinding, ...]
    event_type_bindings: Tuple[EventTypeBinding, ...] = ()
    issues: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_role": "A_G_TERMINAL_BINDING",
            "profile": self.profile,
            "status": self.status,
            "system_package": self.system_package,
            "bindings": [item.to_dict() for item in self.bindings],
            "event_type_bindings": [
                item.to_dict() for item in self.event_type_bindings
            ],
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AGBindingResult:
    model_text: str
    packages: Tuple[str, ...]
    report: AGBindingReport


def _qualified_name(element) -> str:
    qualified = getattr(element, "qualified_name", None)
    return str(qualified) if qualified is not None else str(
        getattr(element, "name", "") or ""
    )


def _without_named_package(model_text: str, package_name: str) -> str:
    pattern = re.compile(rf"\bpackage\s+{re.escape(package_name)}\s*\{{")
    text = str(model_text)
    while True:
        match = pattern.search(text)
        if match is None:
            return text
        brace = text.find("{", match.start())
        end = find_block_end(text, brace)
        if end == -1:
            return text
        text = text[:match.start()] + text[end + 1:]


def _insert_package_members(
    package_text: str,
    *,
    imports: Iterable[str],
    relationships: Iterable[str],
) -> str:
    text = str(package_text)
    opening = text.find("{")
    closing = find_block_end(text, opening)
    if opening == -1 or closing == -1:
        return text
    import_lines = "".join(f"\n    {line}" for line in dict.fromkeys(imports))
    text = text[:opening + 1] + import_lines + text[opening + 1:]
    closing = find_block_end(text, opening)
    relationship_lines = "".join(
        f"\n    {line}" for line in relationships
    )
    return text[:closing] + relationship_lines + "\n" + text[closing:]


def _definition_paths(usage: Any) -> set[str]:
    return {
        _qualified_name(item)
        for item in getattr(usage, "definitions", ())
    }


def _resolve_owner_usage(
    usages: Mapping[str, Any],
    *,
    system_package: str,
    owner_usage: str,
    definition_path: str,
) -> tuple[str | None, Any | None, tuple[str, ...]]:
    """Resolve one frozen local usage identity to its exact terminal path."""
    candidates = sorted(
        [
            (path, usage)
            for path, usage in usages.items()
            if path.startswith(f"{system_package}::")
            and path.rsplit("::", 1)[-1] == owner_usage
            and definition_path in _definition_paths(usage)
        ],
        key=lambda item: item[0],
    )
    if len(candidates) == 1:
        path, usage = candidates[0]
        return path, usage, ()
    expected = f"{system_package}::{owner_usage}"
    if not candidates:
        return None, None, (f"missing owner usage {expected}",)
    paths = ", ".join(path for path, _ in candidates)
    return (
        None,
        None,
        (
            f"ambiguous owner usage {expected}; typed candidates: {paths}",
        ),
    )




def _declared_attribute_type(attribute: Any, model_text: str) -> str:
    for item in getattr(attribute, "types", ()):
        qualified = _qualified_name(item)
        if qualified and "placeholder" not in qualified:
            if qualified not in {"Base::DataValue", "Base::Anything"}:
                return qualified
    node = getattr(attribute, "cst_node", None)
    if node is not None:
        try:
            source = str(node.text(model_text))
        except Exception:
            source = ""
        match = re.search(
            r"\battribute\s+[A-Za-z_]\w*\s*:\s*"
            r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)",
            source,
        )
        if match is not None:
            return match.group(1)
    return "UNRESOLVED"


def _feature_type_bindings(
    definition: Any,
    obligation: Any,
    *,
    model_text: str,
) -> tuple[FeatureTypeBinding, ...]:
    """Validate actual features used as Boolean A/G behavior operands."""
    attributes = {
        str(getattr(item, "name", "") or ""): item
        for item in getattr(definition, "owned_attributes", ())
    }
    ports = {
        str(getattr(item, "name", "") or ""): item
        for item in getattr(definition, "owned_ports", ())
    }
    bindings: list[FeatureTypeBinding] = []
    for concept in _behavior_boolean_concepts(obligation):
        attribute = attributes.get(concept)
        port = ports.get(concept)
        issues: list[str] = []
        if attribute is not None and port is not None:
            observed_kind = "AMBIGUOUS"
            observed_type = "MULTIPLE"
            owner_feature = (
                f"{_qualified_name(definition)}::{concept}"
            )
            issues.append(
                f"Boolean behavior concept {concept} resolves to both an "
                "attribute and a port"
            )
        elif attribute is not None:
            observed_kind = "attribute"
            observed_type = _declared_attribute_type(
                attribute, model_text
            )
            owner_feature = _qualified_name(attribute)
            if observed_type not in {
                "Boolean",
                "ScalarValues::Boolean",
            }:
                issues.append(
                    f"Boolean behavior concept {owner_feature} is declared "
                    f"as {observed_type}, expected Boolean"
                )
        elif port is not None:
            # A directed signal feature can carry the truth concept used by the
            # bounded profile. Its payload semantics remain structural evidence;
            # a contradictory scalar attribute declaration is never accepted.
            observed_kind = "port"
            observed_type = next(
                (
                    _qualified_name(item)
                    for item in getattr(port, "types", ())
                    if _qualified_name(item)
                    not in {
                        "Ports::Port",
                        "Objects::Object",
                        "Occurrences::Occurrence",
                        "Base::Anything",
                    }
                ),
                "UNRESOLVED",
            )
            owner_feature = _qualified_name(port)
        else:
            observed_kind = "MISSING"
            observed_type = "UNRESOLVED"
            owner_feature = (
                f"{_qualified_name(definition)}::{concept}"
            )
            issues.append(
                f"Boolean behavior concept {owner_feature} has no owner feature"
            )
        bindings.append(FeatureTypeBinding(
            concept=concept,
            owner_feature=owner_feature,
            observed_kind=observed_kind,
            observed_type=observed_type,
            status="PASS" if not issues else "FAIL",
            issues=tuple(issues),
        ))
    return tuple(bindings)


def bind_ag_contracts_to_model(
    model_text: str,
    specs: Iterable[AGChainSpec],
    *,
    system_package: str,
    behavior_plan: BehaviorObligationPlan | None = None,
) -> AGBindingResult:
    """Replace planning/shadow packages with contracts bound to real elements.

    Binding is intentionally exact: the expected owner definition, owner usage,
    and behavior name must all resolve in the requested system package.  The
    binder does not guess a semantically similar behavior because that would
    turn a missing realization into fabricated evidence.
    """
    specs = tuple(specs)
    behavior_plan = behavior_plan or compile_behavior_obligation_plan(specs)
    obligations = {
        item.contract_id: item for item in behavior_plan.obligations
    }
    if not _SYSIDE_OK:
        report = AGBindingReport(
            status="FAIL",
            system_package=system_package,
            profile=TERMINAL_REALIZATION,
            bindings=(),
            issues=("Syside is unavailable; terminal A/G binding is unverified",),
        )
        return AGBindingResult(str(model_text), (), report)

    try:
        model, _diagnostics = _syside.try_load_model(sysml_source=str(model_text))
    except Exception as exc:
        report = AGBindingReport(
            status="FAIL",
            system_package=system_package,
            profile=TERMINAL_REALIZATION,
            bindings=(),
            issues=(f"Syside could not load terminal model: {exc}",),
        )
        return AGBindingResult(str(model_text), (), report)

    definitions = {
        _qualified_name(item): item
        for item in model.elements(_syside.PartDefinition)
    }
    usages = {
        _qualified_name(item): item
        for item in model.elements(_syside.PartUsage)
    }
    states = {
        _qualified_name(item): item
        for item in model.elements(_syside.StateDefinition)
    }
    constraints = {
        _qualified_name(item): item
        for item in model.elements(_syside.AssertConstraintUsage)
    }
    item_definitions = {
        _qualified_name(item): item
        for item in model.elements(_syside.ItemDefinition)
    }

    all_bindings: List[ComponentBinding] = []
    event_type_bindings: List[EventTypeBinding] = []
    bound_packages: List[str] = []
    aggregate_issues: List[str] = []
    cleaned_model = str(model_text)
    for spec in specs:
        cleaned_model = _without_named_package(cleaned_model, spec.package)
        imports: List[str] = []
        relationships: List[str] = []
        behavior_by_contract: Dict[str, str] = {}

        for event_name in ag_event_signals(spec):
            canonical_type = f"{system_package}::{event_name}"
            event_issues: List[str] = []
            if canonical_type not in item_definitions:
                event_issues.append(
                    f"missing canonical item definition {canonical_type}"
                )
            else:
                imports.append(f"private import {canonical_type};")
            event_binding = EventTypeBinding(
                package=spec.package,
                event=event_name,
                canonical_type=canonical_type,
                expected_kind="item def",
                status="PASS" if not event_issues else "FAIL",
                issues=tuple(event_issues),
            )
            event_type_bindings.append(event_binding)
            aggregate_issues.extend(
                f"{spec.package}::{event_name}: {issue}"
                for issue in event_issues
            )

        for component in spec.components:
            obligation = obligations.get(component.name)
            definition_path = (
                f"{system_package}::{component.owner_def}"
            )
            usage_path, usage, usage_issues = _resolve_owner_usage(
                usages,
                system_package=system_package,
                owner_usage=component.owner_usage,
                definition_path=definition_path,
            )
            reported_usage_path = (
                usage_path or f"{system_package}::{component.owner_usage}"
            )
            realization_id = (
                obligation.stable_behavior_id
                if obligation is not None else component.behavior
            )
            realization_kind = (
                obligation.realization_kind
                if obligation is not None else "STATE_MACHINE"
            )
            behavior_path = (
                f"{system_package}::{component.owner_def}::"
                f"{realization_id}"
            )
            issues: List[str] = []
            definition = definitions.get(definition_path)
            behavior = (
                constraints.get(behavior_path)
                if realization_kind == INVARIANT
                else states.get(behavior_path)
            )
            if definition is None:
                issues.append(
                    f"missing owner definition {definition_path}"
                )
            issues.extend(usage_issues)
            if usage is not None:
                if definition is None:
                    pass
                elif definition_path not in _definition_paths(usage):
                    issues.append(
                        f"{reported_usage_path} is not typed by "
                        f"{definition_path}"
                    )
            if behavior is None:
                issues.append(
                    (
                        "missing realizing behavior "
                        if realization_kind != INVARIANT
                        else "missing invariant realization "
                    )
                    + behavior_path
                )
            feature_bindings = (
                _feature_type_bindings(
                    definition,
                    obligation,
                    model_text=str(model_text),
                )
                if definition is not None else ()
            )
            issues.extend(
                issue
                for binding in feature_bindings
                for issue in binding.issues
            )

            status = "PASS" if not issues else "FAIL"
            all_bindings.append(ComponentBinding(
                contract=component.name,
                owner_definition=definition_path,
                owner_usage=reported_usage_path,
                behavior=behavior_path,
                realization_kind=realization_kind,
                feature_type_bindings=feature_bindings,
                status=status,
                issues=tuple(issues),
            ))
            aggregate_issues.extend(
                f"{component.name}: {issue}" for issue in issues
            )

            if usage is not None:
                imports.append(f"private import {reported_usage_path};")
                label = component.name[:1].lower() + component.name[1:]
                relationships.append(
                    f"satisfy requirement {label} : {component.name} "
                    f"by {component.owner_usage};"
                )
            if behavior is not None:
                imports.append(f"private import {behavior_path};")
                relationships.append(
                    f"dependency realize{component.name} "
                    f"from {component.name} to {realization_id};"
                )
                behavior_by_contract[component.name] = realization_id

        if spec.priority is not None:
            priority_owner = next(
                (
                    component
                    for component in spec.components
                    if component.behavior == "SafetyResponseArbitration"
                ),
                None,
            )
            if (
                priority_owner is not None
                and priority_owner.name in behavior_by_contract
            ):
                relationships.append(
                    "dependency realizeSafetyResponsePriority "
                    "from SafetyResponsePriorityContract "
                    f"to {priority_owner.behavior};"
                )

        terminal_package = strip_named_item_definitions(
            emit_ag_planning_package(spec),
            ag_event_signals(spec),
        )
        package = _insert_package_members(
            terminal_package,
            imports=imports,
            relationships=relationships,
        )
        bound_packages.append(package)

    merged = cleaned_model.rstrip()
    if bound_packages:
        merged += "\n\n" + "\n\n".join(bound_packages) + "\n"
    report = AGBindingReport(
        status=(
            "PASS"
            if all_bindings
            and all(item.status == "PASS" for item in all_bindings)
            and all(
                item.status == "PASS" for item in event_type_bindings
            )
            and not aggregate_issues
            else "FAIL"
        ),
        system_package=system_package,
        profile=TERMINAL_REALIZATION,
        bindings=tuple(all_bindings),
        event_type_bindings=tuple(event_type_bindings),
        issues=tuple(aggregate_issues),
    )
    return AGBindingResult(merged, tuple(bound_packages), report)
