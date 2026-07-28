"""Bind planning-time A/G contracts to concrete terminal SysML elements."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .ag_emitter import AGChainSpec
from .ag_behavior_plan import (
    BehaviorObligationPlan,
    INVARIANT,
    compile_behavior_obligation_plan,
)
from .ag_planning import emit_ag_planning_package
from ..utils.sysml_text_utils import find_block_end

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:  # pragma: no cover - exercised in fail-closed environments
    _syside = None  # type: ignore
    _SYSIDE_OK = False


TERMINAL_REALIZATION = "TERMINAL_REALIZATION"


@dataclass(frozen=True)
class ComponentBinding:
    contract: str
    owner_definition: str
    owner_usage: str
    behavior: str
    status: str
    realization_kind: str = "STATE_MACHINE"
    issues: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "owner_definition": self.owner_definition,
            "owner_usage": self.owner_usage,
            "behavior": self.behavior,
            "realization_kind": self.realization_kind,
            "status": self.status,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AGBindingReport:
    status: str
    system_package: str
    profile: str
    bindings: Tuple[ComponentBinding, ...]
    issues: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_role": "A_G_TERMINAL_BINDING",
            "profile": self.profile,
            "status": self.status,
            "system_package": self.system_package,
            "bindings": [item.to_dict() for item in self.bindings],
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

    all_bindings: List[ComponentBinding] = []
    bound_packages: List[str] = []
    aggregate_issues: List[str] = []
    cleaned_model = str(model_text)
    for spec in specs:
        cleaned_model = _without_named_package(cleaned_model, spec.package)
        imports: List[str] = []
        relationships: List[str] = []
        behavior_by_contract: Dict[str, str] = {}

        for component in spec.components:
            obligation = obligations.get(component.name)
            definition_path = (
                f"{system_package}::{component.owner_def}"
            )
            usage_path = f"{system_package}::{component.owner_usage}"
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
            usage = usages.get(usage_path)
            behavior = (
                constraints.get(behavior_path)
                if realization_kind == INVARIANT
                else states.get(behavior_path)
            )
            if definition is None:
                issues.append(
                    f"missing owner definition {definition_path}"
                )
            if usage is None:
                issues.append(f"missing owner usage {usage_path}")
            elif definition is not None:
                resolved_definitions = {
                    _qualified_name(item)
                    for item in getattr(usage, "definitions", ())
                }
                if definition_path not in resolved_definitions:
                    issues.append(
                        f"{usage_path} is not typed by {definition_path}"
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

            status = "PASS" if not issues else "FAIL"
            all_bindings.append(ComponentBinding(
                contract=component.name,
                owner_definition=definition_path,
                owner_usage=usage_path,
                behavior=behavior_path,
                realization_kind=realization_kind,
                status=status,
                issues=tuple(issues),
            ))
            aggregate_issues.extend(
                f"{component.name}: {issue}" for issue in issues
            )

            if usage is not None:
                imports.append(f"private import {usage_path};")
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

        package = _insert_package_members(
            emit_ag_planning_package(spec),
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
            else "FAIL"
        ),
        system_package=system_package,
        profile=TERMINAL_REALIZATION,
        bindings=tuple(all_bindings),
        issues=tuple(aggregate_issues),
    )
    return AGBindingResult(merged, tuple(bound_packages), report)
