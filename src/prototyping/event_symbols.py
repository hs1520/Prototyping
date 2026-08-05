"""Plan-first SysML v2 event symbols with one package-level writer.

Transition ``accept`` syntax owns an accept-action usage.  Its referenced event
type is an ``item def``; executable state responses remain ``action def``.
This module prevents independently generated interface and behavior fragments
from declaring the same event name using both definition kinds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..utils.sysml_text_utils import find_block_end
from .namespace_integrity import _mask_comments_and_strings


@dataclass(frozen=True)
class PlannedEventSymbol:
    name: str
    sources: tuple[str, ...]
    definition_kind: str = "item def"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "definition_kind": self.definition_kind,
            "sources": list(self.sources),
        }


def collect_planned_event_symbols(
    planned_behaviors: Sequence[Any] = (),
    behavior_obligations: Sequence[Any] = (),
    components: Sequence[Any] = (),
) -> tuple[PlannedEventSymbol, ...]:
    """Derive every accepted event type from the frozen behavior plans.

    ``components`` remains in the signature for archived callers, but structural
    ports are deliberately not excluded.  An ``accept`` trigger is a classifier
    identity, never a reference to an owning port usage.  Plan validation rejects
    such collisions before this registry is frozen.
    """
    del components
    sources: dict[str, list[str]] = {}
    for behavior in planned_behaviors:
        owner = str(getattr(behavior, "owner", ""))
        for transition in getattr(behavior, "transitions", ()) or ():
            if getattr(transition, "trigger_kind", "") != "ACCEPT":
                continue
            trigger = str(getattr(transition, "trigger", "") or "")
            if not trigger:
                continue
            sources.setdefault(trigger, []).append(
                f"BEHAVIOR::{owner}::"
                f"{getattr(behavior, 'behavior_id', '')}"
            )
    for obligation in behavior_obligations:
        for transition in getattr(obligation, "transitions", ()) or ():
            trigger = str(getattr(transition, "trigger", "") or "")
            if not trigger or trigger == "continuous":
                continue
            sources.setdefault(trigger, []).append(
                "A_G::"
                f"{getattr(obligation, 'owner_def', '')}::"
                f"{getattr(obligation, 'stable_behavior_id', '')}"
            )
    return tuple(
        PlannedEventSymbol(
            name=name,
            sources=tuple(dict.fromkeys(event_sources)),
        )
        for name, event_sources in sorted(sources.items())
    )


def validate_planned_event_symbols(
    symbols: Sequence[PlannedEventSymbol],
    planned_behaviors: Sequence[Any] = (),
    behavior_obligations: Sequence[Any] = (),
    components: Sequence[Any] = (),
) -> list[str]:
    """Reject event identities reserved for actions or planned port types."""
    executable_actions = {
        str(action)
        for behavior in planned_behaviors
        for state in (getattr(behavior, "states", ()) or ())
        for action in (
            getattr(state, "entry_action", None),
            getattr(state, "do_action", None),
        )
        if action
    }
    executable_actions.update(
        str(getattr(transition, "action", ""))
        for obligation in behavior_obligations
        for transition in (getattr(obligation, "transitions", ()) or ())
        if getattr(transition, "action", None)
    )
    issues = [
        f"planned event {symbol.name} collides with executable action identity"
        for symbol in symbols
        if symbol.name in executable_actions
    ]
    planned_port_types = {
        str(getattr(port, "port_type", ""))
        for component in components
        for port in (getattr(component, "ports", ()) or ())
        if getattr(port, "port_type", None)
    }
    issues.extend(
        f"planned event {symbol.name} collides with planned port definition type"
        for symbol in symbols
        if symbol.name in planned_port_types
    )
    return issues


_DEFINITION_RE = re.compile(
    r"\b(?P<kind>"
    r"(?:action|attribute|constraint|enum|item|part|port|requirement|state|"
    r"verification)\s+def"
    r")\s+(?P<name>[A-Za-z_]\w*)"
    r"(?P<header>(?:\s*:>\s*[A-Za-z_]\w*"
    r"(?:::[A-Za-z_]\w*)*)*)\s*(?P<tail>[;{])"
)


def _scope(text: str) -> tuple[int, int]:
    package = re.search(r"\bpackage\s+[A-Za-z_]\w*\s*\{", text)
    if package is None:
        return 0, len(text)
    opening = text.find("{", package.start(), package.end())
    closing = find_block_end(text, opening)
    if opening == -1 or closing == -1:
        return 0, len(text)
    return opening + 1, closing


def _direct_declarations(
    text: str,
    name: str,
) -> list[dict[str, Any]]:
    start, end = _scope(text)
    body = text[start:end]
    masked = _mask_comments_and_strings(body)
    depth = 0
    depths: list[int] = []
    for char in masked:
        depths.append(depth)
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
    result: list[dict[str, Any]] = []
    for match in _DEFINITION_RE.finditer(masked):
        if depths[match.start()] != 0 or match.group("name") != name:
            continue
        absolute_start = start + match.start()
        if match.group("tail") == ";":
            absolute_end = start + match.end()
            body_text = ""
        else:
            opening = text.find(
                "{", absolute_start, start + match.end()
            )
            closing = find_block_end(text, opening)
            if opening == -1 or closing == -1:
                continue
            absolute_end = closing + 1
            body_text = text[opening + 1:closing]
        result.append({
            "kind": " ".join(match.group("kind").split()),
            "start": absolute_start,
            "end": absolute_end,
            "header": match.group("header").strip(),
            "body": body_text,
            "text": text[absolute_start:absolute_end],
        })
    return result


def _empty_body(value: str) -> bool:
    return not _mask_comments_and_strings(value).strip()


def _has_semantics(declaration: dict[str, Any]) -> bool:
    return bool(declaration["header"]) or not _empty_body(
        declaration["body"]
    )


def check_planned_event_symbol_conformance(
    model_text: str,
    symbols: Sequence[PlannedEventSymbol],
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    issues: list[str] = []
    symbol_names = {item.name for item in symbols}
    for symbol in symbols:
        declarations = _direct_declarations(model_text, symbol.name)
        item_count = sum(
            item["kind"] == "item def" for item in declarations
        )
        wrong = [
            item["kind"]
            for item in declarations
            if item["kind"] != "item def"
        ]
        item_issues: list[str] = []
        if item_count != 1:
            item_issues.append(
                f"expected exactly one package item def, found {item_count}"
            )
        if wrong:
            item_issues.append(
                "wrong definition kinds: " + ", ".join(sorted(wrong))
            )
        checked.append({
            **symbol.to_dict(),
            "status": "PASS" if not item_issues else "FAIL",
            "issues": item_issues,
        })
        issues.extend(
            f"{symbol.name}: {issue}" for issue in item_issues
        )
    accept_bindings: list[dict[str, Any]] = []
    masked = _mask_comments_and_strings(str(model_text or ""))
    for match in re.finditer(
        r"\baccept\s+(?P<name>[A-Za-z_]\w*)\b",
        masked,
    ):
        name = match.group("name")
        line = masked.count("\n", 0, match.start()) + 1
        binding_issues: list[str] = []
        if name not in symbol_names:
            binding_issues.append(
                "accept target is not present in the frozen event registry"
            )
        declarations = _direct_declarations(model_text, name)
        item_count = sum(
            item["kind"] == "item def" for item in declarations
        )
        if item_count != 1:
            binding_issues.append(
                f"accept target must resolve to exactly one package item def; "
                f"found {item_count}"
            )
        accept_bindings.append({
            "name": name,
            "line": line,
            "expected_kind": "item def",
            "status": "PASS" if not binding_issues else "FAIL",
            "issues": binding_issues,
        })
        issues.extend(
            f"accept {name} at line {line}: {issue}"
            for issue in binding_issues
        )
    return {
        "artifact_role": "PLANNED_EVENT_SYMBOL_CONFORMANCE",
        "status": (
            "NOT_APPLICABLE"
            if not symbols and not accept_bindings
            else "PASS" if not issues else "FAIL"
        ),
        "checked": checked,
        "accept_bindings": accept_bindings,
        "issues": issues,
    }


def materialize_planned_event_symbols(
    model_text: str,
    symbols: Sequence[PlannedEventSymbol],
) -> tuple[str, dict[str, Any]]:
    """Canonicalize declarations to the frozen event item identity."""
    text = str(model_text)
    materialized: list[str] = []
    removed: list[dict[str, str]] = []
    unsafe: list[str] = []
    for symbol in symbols:
        declarations = _direct_declarations(text, symbol.name)
        item_defs = [
            item for item in declarations if item["kind"] == "item def"
        ]
        action_defs = [
            item for item in declarations if item["kind"] == "action def"
        ]
        port_defs = [
            item for item in declarations if item["kind"] == "port def"
        ]
        other_defs = [
            item for item in declarations
            if item["kind"] not in {"item def", "action def", "port def"}
        ]
        rich_items = [
            item for item in item_defs if _has_semantics(item)
        ]
        rich_actions = [
            item for item in action_defs if _has_semantics(item)
        ]
        if len(rich_items) > 1:
            unsafe.append(
                f"{symbol.name}: multiple non-empty item definitions"
            )
        if rich_actions:
            unsafe.append(
                f"{symbol.name}: non-empty action definition cannot be "
                "reclassified as an event type"
            )
        if other_defs:
            unsafe.append(
                f"{symbol.name}: reserved event name uses "
                + ", ".join(sorted({
                    item["kind"] for item in other_defs
                }))
            )
        if any(issue.startswith(f"{symbol.name}:") for issue in unsafe):
            continue

        canonical = (
            rich_items[0]["text"]
            if rich_items else f"item def {symbol.name};"
        )
        removable = [*item_defs, *action_defs, *port_defs]
        for declaration in sorted(
            removable, key=lambda item: item["start"], reverse=True
        ):
            text = (
                text[:declaration["start"]]
                + text[declaration["end"]:]
            )
            removed.append({
                "name": symbol.name,
                "kind": declaration["kind"],
            })
        scope_start, _ = _scope(text)
        indentation = "    " if scope_start else ""
        insertion = indentation + canonical + "\n"
        text = text[:scope_start] + insertion + text[scope_start:]
        materialized.append(symbol.name)

    report = check_planned_event_symbol_conformance(text, symbols)
    if unsafe:
        report["status"] = "FAIL"
        report["issues"] = list(dict.fromkeys([
            *report["issues"], *unsafe,
        ]))
    report["materialized"] = materialized
    report["removed_declarations"] = list(reversed(removed))
    report["unsafe_conflicts"] = unsafe
    return text, report
