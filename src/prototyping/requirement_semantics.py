"""Bounded, source-derived semantic anchors for whole-model generation.

This is not a temporal proof system.  It extracts only explicit numeric
bounds stated by the stakeholder (for example, "maintain at least 5 metres of
separation while avoiding it").  The resulting anchor preserves comparator,
threshold, unit, subject terms, and a bounded activation qualifier across
generation and remains digest-bound to its source.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from ..utils.sysml_text_utils import find_block_end
from ..utils.req_id import normalise_req_id
from .namespace_integrity import collect_package_definitions


NUMERIC_INVARIANT = "NUMERIC_INVARIANT"

_REQ_ID_RE = re.compile(
    r"\bREQ[-_][A-Za-z]+[-_]\d+\b",
    re.IGNORECASE,
)
_MAINTAIN_BOUND_RE = re.compile(
    r"\b(?:maintain|keep|ensure)\s+"
    r"(?:(?P<subject_before>[A-Za-z][A-Za-z0-9 _-]{0,80}?)\s+)?"
    r"(?P<comparator>"
    r"at\s+least|no\s+less\s+than|minimum(?:\s+of)?|"
    r"at\s+most|no\s+more\s+than|maximum(?:\s+of)?"
    r")\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>"
    r"milliseconds?|ms|seconds?|s|"
    r"metres?|meters?|m|kilometres?|kilometers?|km|"
    r"degrees?|deg|percent|%|hertz|hz"
    r")"
    r"(?:\s+of\s+(?P<subject_after>"
    r"[A-Za-z][A-Za-z0-9 _-]{0,80}?))?"
    r"(?=\s+(?:while|when|during|after|before|unless|and)\b|[.,;]|$)",
    re.IGNORECASE,
)
_ACTIVATION_CLAUSE_RE = re.compile(
    r"^\s+(?P<clause>(?:while|when|during|after|before|unless)\b"
    r"[^.,;]*)(?=[.,;]|$)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "the", "current", "required", "minimum", "maximum",
    "distance", "value", "level",
}
_UNIT_CANONICAL = {
    "millisecond": "ms",
    "milliseconds": "ms",
    "ms": "ms",
    "second": "s",
    "seconds": "s",
    "s": "s",
    "metre": "m",
    "metres": "m",
    "meter": "m",
    "meters": "m",
    "m": "m",
    "kilometre": "km",
    "kilometres": "km",
    "kilometer": "km",
    "kilometers": "km",
    "km": "km",
    "degree": "deg",
    "degrees": "deg",
    "deg": "deg",
    "percent": "%",
    "%": "%",
    "hertz": "Hz",
    "hz": "Hz",
}
_UNIT_QUANTITY_TYPES = {
    "m": "LengthValue",
    "km": "LengthValue",
    "s": "DurationValue",
    "ms": "DurationValue",
    "Hz": "FrequencyValue",
}
_PART_DEF_RE = re.compile(r"\bpart\s+def\s+(?P<name>[A-Za-z_]\w*)\s*\{")
_PORT_RE = re.compile(
    r"\b(?P<direction>in|out|inout)\s+port\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>[A-Za-z_]\w*)\s*;"
)
_ATTRIBUTE_RE = re.compile(
    r"\battribute\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*:\s*[A-Za-z_][\w:]*)?\s*=\s*(?P<value>[^;{}]+)\s*;"
)
_ASSERT_RE = re.compile(
    r"\bassert\s+constraint\s+(?P<name>[A-Za-z_]\w*)\s*\{"
)
_PLAN_CONSTRAINT_RE = re.compile(
    r"//\s*PLAN-CONSTRAINT\s+[A-Za-z_]\w*\s+"
    r"provenance=[A-Z_]+\s+activation=(?P<activation>[A-Z_]+)\s+"
    r"verification=[A-Z_]+"
)
_COMPARISON_RE = re.compile(
    r"^\s*(?P<left>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*"
    r"(?P<operator>>=|<=|>|<)\s*"
    r"(?P<right>"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|"
    r"\d+(?:\.\d+)?(?:\s*\[[A-Za-z_%][A-Za-z0-9_%]*\])?"
    r")\s*$"
)
_NUMERIC_VALUE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*\[(?P<unit>[A-Za-z_%][A-Za-z0-9_%]*)\])?\s*$"
)
_PATH_RE = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$"
)
_TRANSITION_GUARD_RE = re.compile(
    r"\btransition\s+(?P<name>[A-Za-z_]\w*)"
    r"(?P<body>[^;]*?\bif\s+(?P<guard>.*?)\s+then\s+"
    r"(?P<target>[A-Za-z_]\w*)\s*;)",
    re.DOTALL,
)


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.strip().encode("utf-8")).hexdigest()


def _subject_terms(value: str) -> tuple[str, ...]:
    words = [
        item.lower()
        for item in re.findall(r"[A-Za-z][A-Za-z0-9]*", value or "")
    ]
    return tuple(dict.fromkeys(
        item for item in words if item not in _STOPWORDS
    ))


def _identifier_terms(value: str) -> tuple[str, ...]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value or "")
    return tuple(
        item.lower()
        for item in re.findall(r"[A-Za-z][A-Za-z0-9]*", expanded)
    )


def _matches_subject(value: str, terms: Sequence[str]) -> bool:
    observed = set(_identifier_terms(value.replace(".", " ")))
    return bool(terms) and all(term.lower() in observed for term in terms)


def _normalise_unit(value: str | None) -> str:
    raw = str(value or "").strip()
    return _UNIT_CANONICAL.get(raw.lower(), raw)


def quantity_type_for_unit(value: str | None) -> str | None:
    """Return the SysML v2 ISQ quantity type for a supported SI unit."""
    return _UNIT_QUANTITY_TYPES.get(_normalise_unit(value))


def _extract_part_definitions(model_text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for match in _PART_DEF_RE.finditer(model_text):
        opening = model_text.find("{", match.start(), match.end())
        closing = find_block_end(model_text, opening)
        if closing != -1:
            parts.append((
                match.group("name"),
                model_text[opening + 1:closing],
            ))
    return parts


def _extract_assertions(
    block: str,
) -> list[tuple[str, str, str | None]]:
    assertions: list[tuple[str, str, str | None]] = []
    for match in _ASSERT_RE.finditer(block):
        opening = block.find("{", match.start(), match.end())
        closing = find_block_end(block, opening)
        if closing != -1:
            line_start = block.rfind("\n", 0, match.start()) + 1
            previous_start = block.rfind(
                "\n", 0, max(0, line_start - 1)
            ) + 1
            marker = _PLAN_CONSTRAINT_RE.search(
                block[previous_start:line_start]
            )
            assertions.append((
                match.group("name"),
                block[opening + 1:closing].strip(),
                marker.group("activation") if marker else None,
            ))
    return assertions


def _numeric_value(
    expression: str,
    attributes: dict[str, str],
) -> tuple[float, str] | None:
    candidate = expression.strip()
    visited: set[str] = set()
    while candidate in attributes and candidate not in visited:
        visited.add(candidate)
        candidate = attributes[candidate].strip()
    match = _NUMERIC_VALUE_RE.fullmatch(candidate)
    if match is None:
        return None
    return float(match.group("value")), _normalise_unit(match.group("unit"))


def _source_bound_subject(
    expression: str,
    *,
    inputs: set[str],
    attributes: dict[str, str],
    subject_terms: Sequence[str],
) -> tuple[bool, str]:
    candidate = expression.strip()
    if candidate in attributes:
        binding = attributes[candidate].strip()
        if (
            _PATH_RE.fullmatch(binding)
            and binding.split(".", 1)[0] in inputs
            and (
                _matches_subject(candidate, subject_terms)
                or _matches_subject(binding, subject_terms)
            )
        ):
            return True, f"{candidate} = {binding}"
    if (
        _PATH_RE.fullmatch(candidate)
        and candidate.split(".", 1)[0] in inputs
        and _matches_subject(candidate, subject_terms)
    ):
        return True, candidate
    return False, ""


def _comparison_fidelity(
    expression: str,
    obligation: "RequirementSemanticObligation",
    *,
    inputs: set[str],
    attributes: dict[str, str],
) -> dict[str, Any] | None:
    match = _COMPARISON_RE.fullmatch(" ".join(expression.split()))
    if match is None:
        return None
    left = match.group("left")
    right = match.group("right")
    operator = match.group("operator")
    subject = left
    bound = right
    if not _matches_subject(left, obligation.subject_terms):
        if not _matches_subject(right, obligation.subject_terms):
            return None
        subject = right
        bound = left
        operator = {
            ">=": "<=",
            "<=": ">=",
            ">": "<",
            "<": ">",
        }[operator]
    source_bound, binding = _source_bound_subject(
        subject,
        inputs=inputs,
        attributes=attributes,
        subject_terms=obligation.subject_terms,
    )
    numeric = _numeric_value(bound, attributes)
    return {
        "subject_expression": subject,
        "operator": operator,
        "bound_expression": bound,
        "numeric_bound": numeric,
        "source_bound": source_bound,
        "source_binding": binding,
    }


def _late_response_guards(
    block: str,
    obligation: "RequirementSemanticObligation",
    *,
    attributes: dict[str, str],
) -> list[str]:
    """Find an avoidance/maintenance transition that waits for violation."""
    issues: list[str] = []
    for transition in _TRANSITION_GUARD_RE.finditer(block):
        response_name = (
            transition.group("name") + " " + transition.group("target")
        )
        response_terms = set(_identifier_terms(response_name))
        if not ({"avoid", "avoidance", "maintain"} & response_terms):
            continue
        comparison = _COMPARISON_RE.fullmatch(
            " ".join(transition.group("guard").split())
        )
        if comparison is None:
            continue
        left = comparison.group("left")
        right = comparison.group("right")
        operator = comparison.group("operator")
        if not _matches_subject(left, obligation.subject_terms):
            continue
        numeric = _numeric_value(right, attributes)
        if numeric is None:
            continue
        threshold, unit = numeric
        same_unit = _normalise_unit(unit) == obligation.unit
        if obligation.operator == ">=":
            too_late = (
                (operator == "<" and threshold <= obligation.threshold)
                or (operator == "<=" and threshold < obligation.threshold)
            )
        else:
            too_late = (
                (operator == ">" and threshold >= obligation.threshold)
                or (operator == ">=" and threshold > obligation.threshold)
            )
        if same_unit and too_late:
            issues.append(
                f"transition {transition.group('name')} activates "
                f"{transition.group('target')} only after the frozen "
                f"{obligation.threshold:g} [{obligation.unit}] boundary"
            )
    return issues


@dataclass(frozen=True)
class RequirementSemanticObligation:
    obligation_id: str
    requirement_id: str
    kind: str
    subject_terms: tuple[str, ...]
    operator: str
    threshold: float
    unit: str
    source_clause: str
    source_digest: str
    activation_kind: str = "UNCONDITIONAL"
    activation_clause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "requirement_id": self.requirement_id,
            "kind": self.kind,
            "subject_terms": list(self.subject_terms),
            "operator": self.operator,
            "threshold": self.threshold,
            "unit": self.unit,
            "source_clause": self.source_clause,
            "source_digest": self.source_digest,
            "activation": {
                "kind": self.activation_kind,
                "source_clause": self.activation_clause,
            },
            "claim_boundary": "MODEL_SEMANTIC_FIDELITY_NOT_PHYSICAL_PROOF",
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any]
    ) -> "RequirementSemanticObligation":
        activation_value = value.get("activation")
        activation = (
            activation_value
            if isinstance(activation_value, Mapping)
            else {}
        )
        return cls(
            obligation_id=str(value.get("obligation_id") or ""),
            requirement_id=normalise_req_id(
                str(value.get("requirement_id") or "")
            ),
            kind=str(value.get("kind") or ""),
            subject_terms=tuple(
                str(item).lower()
                for item in (value.get("subject_terms") or ())
                if str(item).strip()
            ),
            operator=str(value.get("operator") or ""),
            threshold=float(value.get("threshold") or 0.0),
            unit=str(value.get("unit") or ""),
            source_clause=str(value.get("source_clause") or ""),
            source_digest=str(value.get("source_digest") or ""),
            activation_kind=str(
                activation.get("kind")
                or value.get("activation_kind")
                or "UNCONDITIONAL"
            ).strip().upper(),
            activation_clause=(
                str(
                    activation.get("source_clause")
                    or value.get("activation_clause")
                ).strip()
                if (
                    activation.get("source_clause")
                    or value.get("activation_clause")
                )
                else None
            ),
        )


@dataclass(frozen=True)
class SemanticBindingPlan:
    """Frozen source-to-constraint realization for one semantic obligation."""

    obligation_id: str
    requirement_id: str
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    port_type: str
    port_feature: str
    item_type: str
    item_feature: str
    value_type: str
    unit: str
    runtime_attribute: str
    threshold_attribute: str
    constraint_name: str

    @property
    def source_path(self) -> str:
        return (
            f"{self.target_port}.{self.port_feature}.{self.item_feature}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "requirement_id": self.requirement_id,
            "source": {
                "component": self.source_component,
                "port": self.source_port,
            },
            "target": {
                "component": self.target_component,
                "port": self.target_port,
                "runtime_attribute": self.runtime_attribute,
            },
            "payload": {
                "port_type": self.port_type,
                "port_feature": self.port_feature,
                "item_type": self.item_type,
                "item_feature": self.item_feature,
                "value_type": self.value_type,
                "unit": self.unit,
            },
            "constraint": {
                "name": self.constraint_name,
                "threshold_attribute": self.threshold_attribute,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticBindingPlan":
        source_value = value.get("source")
        target_value = value.get("target")
        payload_value = value.get("payload")
        constraint_value = value.get("constraint")
        source = source_value if isinstance(source_value, Mapping) else {}
        target = target_value if isinstance(target_value, Mapping) else {}
        payload = payload_value if isinstance(payload_value, Mapping) else {}
        constraint = (
            constraint_value
            if isinstance(constraint_value, Mapping) else {}
        )
        unit = _normalise_unit(str(payload.get("unit") or ""))
        # Unit is frozen source evidence, so the quantity type is deterministic
        # rather than another LLM choice. Unsupported units remain visible to
        # plan validation instead of being silently represented as Real.
        value_type = (
            quantity_type_for_unit(unit)
            or str(payload.get("value_type") or "").strip()
        )
        return cls(
            obligation_id=str(value.get("obligation_id") or "").strip(),
            requirement_id=normalise_req_id(
                str(value.get("requirement_id") or "")
            ),
            source_component=str(source.get("component") or "").strip(),
            source_port=str(source.get("port") or "").strip(),
            target_component=str(target.get("component") or "").strip(),
            target_port=str(target.get("port") or "").strip(),
            port_type=str(payload.get("port_type") or "").strip(),
            port_feature=str(
                payload.get("port_feature") or "payload"
            ).strip(),
            item_type=str(payload.get("item_type") or "").strip(),
            item_feature=str(payload.get("item_feature") or "").strip(),
            value_type=value_type,
            unit=unit,
            runtime_attribute=str(
                target.get("runtime_attribute") or ""
            ).strip(),
            threshold_attribute=str(
                constraint.get("threshold_attribute") or ""
            ).strip(),
            constraint_name=str(constraint.get("name") or "").strip(),
        )


def semantic_binding_matches_subject(
    binding: SemanticBindingPlan,
    obligation: RequirementSemanticObligation,
) -> bool:
    """Return whether the planned feature/attribute preserves subject terms."""
    names = " ".join((
        binding.item_feature,
        binding.runtime_attribute,
        binding.item_type,
    ))
    observed = set(_identifier_terms(names))
    return bool(obligation.subject_terms) and all(
        term.lower() in observed for term in obligation.subject_terms
    )


def render_semantic_binding_planning_guidance(
    obligations: Sequence[RequirementSemanticObligation],
) -> str:
    """Render source facts the architecture planner must bind explicitly."""
    if not obligations:
        return ""
    lines = [
        "SOURCE-DERIVED SEMANTIC BINDINGS REQUIRED IN PLAN:",
        "For every obligation below, add exactly one semantic_bindings entry.",
        "Choose an existing planned source->target connection. The target must "
        "own the requirement and receive a dedicated, non-generic port type.",
    ]
    for obligation in obligations:
        subject = "/".join(obligation.subject_terms)
        lines.append(
            f"- {obligation.obligation_id} [{obligation.requirement_id}]: "
            f"{subject} {obligation.operator} "
            f"{obligation.threshold:g} [{obligation.unit}]; "
            f"activation={obligation.activation_kind}"
            + (
                f" ({obligation.activation_clause})"
                if obligation.activation_clause else ""
            )
        )
    return "\n".join(lines)


def compile_requirement_semantic_obligations(
    requirements: Sequence[str],
) -> tuple[RequirementSemanticObligation, ...]:
    """Compile only explicit, directly source-supported numeric invariants."""
    obligations: list[RequirementSemanticObligation] = []
    for requirement in requirements:
        source = str(requirement or "").strip()
        req_match = _REQ_ID_RE.search(source)
        if req_match is None:
            continue
        requirement_id = normalise_req_id(req_match.group(0))
        body = source.split(":", 1)[1].strip() if ":" in source else source
        matches = list(_MAINTAIN_BOUND_RE.finditer(body))
        for index, match in enumerate(matches, 1):
            subject = (
                match.group("subject_after")
                or match.group("subject_before")
                or ""
            )
            terms = _subject_terms(subject)
            if not terms:
                continue
            comparator = " ".join(
                match.group("comparator").lower().split()
            )
            operator = (
                ">="
                if comparator in {
                    "at least", "no less than", "minimum", "minimum of",
                }
                else "<="
            )
            raw_unit = match.group("unit").lower()
            activation_match = _ACTIVATION_CLAUSE_RE.match(
                body[match.end():]
            )
            activation_clause = (
                " ".join(activation_match.group("clause").split())
                if activation_match else None
            )
            obligations.append(RequirementSemanticObligation(
                obligation_id=f"SEM_{requirement_id}_{index:03d}",
                requirement_id=requirement_id,
                kind=NUMERIC_INVARIANT,
                subject_terms=terms,
                operator=operator,
                threshold=float(match.group("value")),
                unit=_UNIT_CANONICAL[raw_unit],
                source_clause=" ".join(
                    (
                        match.group(0)
                        + (
                            f" {activation_clause}"
                            if activation_clause else ""
                        )
                    ).split()
                ),
                source_digest=_source_digest(source),
                activation_kind=(
                    "CONTEXTUAL" if activation_clause else "UNCONDITIONAL"
                ),
                activation_clause=activation_clause,
            ))
    return tuple(obligations)


def _definition_span(
    text: str,
    kind: str,
    name: str,
) -> tuple[int, int, int | None] | None:
    match = re.search(
        rf"\b{re.escape(kind)}\s+def\s+{re.escape(name)}\s*(?P<tail>[;{{])",
        text,
    )
    if match is None:
        return None
    if match.group("tail") == ";":
        return match.start(), match.end(), None
    opening = text.find("{", match.start(), match.end())
    closing = find_block_end(text, opening)
    if closing == -1:
        return None
    return match.start(), closing + 1, closing


def _insert_package_member(text: str, snippet: str) -> str | None:
    package = re.search(r"\bpackage\s+[A-Za-z_]\w*\s*\{", text)
    if package is None:
        return None
    opening = text.find("{", package.start(), package.end())
    return text[:opening + 1] + "\n" + snippet + text[opening + 1:]


def _has_root_package_import(text: str, library: str) -> bool:
    package = re.search(r"\bpackage\s+[A-Za-z_]\w*\s*\{", text)
    if package is None:
        return False
    opening = text.find("{", package.start(), package.end())
    closing = find_block_end(text, opening)
    if closing == -1:
        return False
    body = text[opening + 1:closing]
    pattern = re.compile(
        rf"\b(?:private\s+)?import\s+{re.escape(library)}::\*\s*;"
    )
    for match in pattern.finditer(body):
        prefix = body[:match.start()]
        if prefix.count("{") == prefix.count("}"):
            return True
    return False


def _ensure_quantity_imports(
    text: str,
    changes: list[str],
    issues: list[str],
) -> str:
    missing = [
        library
        for library in ("ISQ", "SI")
        if not _has_root_package_import(text, library)
    ]
    if not missing:
        return text
    snippet = "\n".join(
        f"    private import {library}::*;" for library in missing
    )
    inserted = _insert_package_member(text, snippet + "\n")
    if inserted is None:
        issues.append("cannot locate package for ISQ/SI quantity imports")
        return text
    changes.extend(f"private import {library}::*" for library in missing)
    return inserted


def _indent_definition(definition: str) -> str:
    return "\n".join(
        f"    {line}" if line else line
        for line in definition.splitlines()
    )


def _item_attribute_usages(
    body: str,
    binding: SemanticBindingPlan,
) -> list[re.Match[str]]:
    return list(re.finditer(
        rf"\battribute\s+{re.escape(binding.item_feature)}"
        rf"\s*:\s*(?P<type>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
        rf"(?:\s*=\s*(?P<value>[^;{{}}]+))?\s*;",
        body,
    ))


def _named_item_attribute_statements(
    body: str,
    binding: SemanticBindingPlan,
) -> list[re.Match[str]]:
    """Find the feature by identity before interpreting its trailing syntax."""
    return list(re.finditer(
        rf"\battribute\s+{re.escape(binding.item_feature)}\b"
        rf"(?P<tail>[^;{{}}]*)\s*;",
        body,
    ))


def _port_item_usages(
    body: str,
    binding: SemanticBindingPlan,
) -> list[re.Match[str]]:
    return list(re.finditer(
        rf"\b(?:(?P<direction>in|out|inout)\s+)?item\s+"
        rf"{re.escape(binding.port_feature)}\s*:\s*"
        rf"(?P<type>[A-Za-z_]\w*)\s*;",
        body,
    ))


def _named_port_item_statements(
    body: str,
    binding: SemanticBindingPlan,
) -> list[re.Match[str]]:
    return list(re.finditer(
        rf"\b(?:(?:in|out|inout)\s+)?item\s+"
        rf"{re.escape(binding.port_feature)}\b"
        rf"(?P<tail>[^;{{}}]*)\s*;",
        body,
    ))


def _conflicting_definition_kinds(
    text: str,
    name: str,
    expected_kind: str,
) -> list[str]:
    return sorted(
        item["kind"]
        for item in collect_package_definitions(text)
        if item["name"] == name and item["kind"] != expected_kind
    )


def _canonicalize_plan_owned_definition_kind(
    text: str,
    *,
    name: str,
    expected_kind: str,
    desired_definition: str,
    changes: list[str],
    issues: list[str],
) -> str:
    """Correct the narrow definition-kind drift owned by a semantic binding.

    A semantic binding freezes ``item_type`` and ``port_type`` before textual
    generation.  LLM fragments still occasionally serialize those names as an
    ``attribute def``.  That is notation drift, not a new engineering decision,
    so the compiler may replace exactly one such root declaration.  All other
    collisions remain fail-closed: in particular, a ``part def`` may carry real
    structure and must never be silently rewritten into an item or port.
    """
    conflicts = _conflicting_definition_kinds(
        text, name, expected_kind
    )
    if not conflicts:
        return text
    if (
        len(conflicts) != 1
        or conflicts[0] != "attribute def"
        or _definition_span(
            text, expected_kind.removesuffix(" def"), name
        ) is not None
    ):
        return text
    span = _definition_span(text, "attribute", name)
    if span is None:
        issues.append(
            f"cannot locate conflicting attribute def {name} for "
            f"canonical {expected_kind}"
        )
        return text
    start, end, _closing = span
    line_start = text.rfind("\n", 0, start) + 1
    indent = text[line_start:start]
    replacement = desired_definition.replace("\n", "\n" + indent)
    changes.append(
        f"canonicalized definition kind {name}: attribute def -> "
        f"{expected_kind}"
    )
    return text[:start] + replacement + text[end:]


def _validate_item_attribute_usage(
    match: re.Match[str],
    binding: SemanticBindingPlan,
) -> list[str]:
    issues: list[str] = []
    if match.group("type") != binding.value_type:
        issues.append(
            f"{binding.item_type}.{binding.item_feature} has type "
            f"{match.group('type')}, expected {binding.value_type}"
        )
    initializer = match.group("value")
    if initializer is not None:
        numeric = _numeric_value(initializer, {})
        if numeric is None:
            issues.append(
                f"{binding.item_type}.{binding.item_feature} has a "
                "non-numeric default incompatible with its numeric binding"
            )
        elif binding.unit and _normalise_unit(numeric[1]) != binding.unit:
            issues.append(
                f"{binding.item_type}.{binding.item_feature} default uses "
                f"{numeric[1] or '(missing unit)'}, expected {binding.unit}"
            )
    return issues


def _ensure_item_feature(
    text: str,
    binding: SemanticBindingPlan,
    changes: list[str],
    issues: list[str],
) -> str:
    desired = (
        f"item def {binding.item_type} {{\n"
        f"    attribute {binding.item_feature} : "
        f"{binding.value_type};\n"
        f"}}"
    )
    text = _canonicalize_plan_owned_definition_kind(
        text,
        name=binding.item_type,
        expected_kind="item def",
        desired_definition=desired,
        changes=changes,
        issues=issues,
    )
    conflicting = _conflicting_definition_kinds(
        text, binding.item_type, "item def"
    )
    if conflicting:
        issues.append(
            f"{binding.item_type} must be item def, but is already "
            f"declared as {', '.join(conflicting)}"
        )
        return text
    span = _definition_span(text, "item", binding.item_type)
    if span is None:
        inserted = _insert_package_member(
            text, _indent_definition(desired) + "\n"
        )
        if inserted is None:
            issues.append("cannot locate package for semantic item definition")
            return text
        changes.append(f"item def {binding.item_type}")
        return inserted
    start, end, closing = span
    if closing is None:
        line_start = text.rfind("\n", 0, start) + 1
        indent = text[line_start:start]
        replacement = desired.replace("\n", "\n" + indent)
        changes.append(
            f"expanded item def {binding.item_type} with "
            f"{binding.item_feature}"
        )
        return text[:start] + replacement + text[end:]
    body = text[text.find("{", start, end) + 1:closing]
    named_features = _named_item_attribute_statements(body, binding)
    if len(named_features) > 1:
        issues.append(
            f"{binding.item_type}.{binding.item_feature} appears "
            f"{len(named_features)} times; expected exactly once"
        )
        return text
    features = _item_attribute_usages(body, binding)
    if len(named_features) == 1:
        if len(features) == 1:
            feature = features[0]
            feature_issues = _validate_item_attribute_usage(
                feature, binding
            )
            initializer = feature.group("value")
            type_issue = (
                feature.group("type") != binding.value_type
            )
            initializer_issues = [
                item for item in feature_issues
                if " has type " not in item
            ]
            if type_issue and not initializer_issues:
                canonical = (
                    f"attribute {binding.item_feature} : "
                    f"{binding.value_type}"
                )
                if initializer is not None:
                    canonical += f" = {initializer.strip()}"
                canonical += ";"
                body = (
                    body[:feature.start()]
                    + canonical
                    + body[feature.end():]
                )
                changes.append(
                    f"dimensioned item feature "
                    f"{binding.item_type}.{binding.item_feature}"
                )
                return (
                    text[:text.find("{", start, end) + 1]
                    + body
                    + text[closing:]
                )
            issues.extend(feature_issues)
            return text
        canonical = (
            f"attribute {binding.item_feature} : {binding.value_type};"
        )
        feature = named_features[0]
        body = body[:feature.start()] + canonical + body[feature.end():]
        changes.append(
            f"canonicalized item feature "
            f"{binding.item_type}.{binding.item_feature}"
        )
        return text[:text.find("{", start, end) + 1] + body + text[closing:]
    if features:
        issues.extend(
            _validate_item_attribute_usage(features[0], binding)
        )
        return text
    insertion = (
        f"\n        attribute {binding.item_feature} : "
        f"{binding.value_type};\n    "
    )
    changes.append(
        f"item feature {binding.item_type}.{binding.item_feature}"
    )
    return text[:closing] + insertion + text[closing:]


def _ensure_port_payload(
    text: str,
    binding: SemanticBindingPlan,
    changes: list[str],
    issues: list[str],
) -> str:
    desired = (
        f"port def {binding.port_type} {{\n"
        f"    in item {binding.port_feature} : {binding.item_type};\n"
        f"}}"
    )
    text = _canonicalize_plan_owned_definition_kind(
        text,
        name=binding.port_type,
        expected_kind="port def",
        desired_definition=desired,
        changes=changes,
        issues=issues,
    )
    conflicting = _conflicting_definition_kinds(
        text, binding.port_type, "port def"
    )
    if conflicting:
        issues.append(
            f"{binding.port_type} must be port def, but is already "
            f"declared as {', '.join(conflicting)}"
        )
        return text
    span = _definition_span(text, "port", binding.port_type)
    if span is None:
        inserted = _insert_package_member(
            text, _indent_definition(desired) + "\n"
        )
        if inserted is None:
            issues.append("cannot locate package for semantic port definition")
            return text
        changes.append(f"port def {binding.port_type}")
        return inserted
    start, end, closing = span
    if closing is None:
        line_start = text.rfind("\n", 0, start) + 1
        indent = text[line_start:start]
        replacement = desired.replace("\n", "\n" + indent)
        changes.append(
            f"expanded port def {binding.port_type} with "
            f"{binding.port_feature}"
        )
        return text[:start] + replacement + text[end:]
    body = text[text.find("{", start, end) + 1:closing]
    named_features = _named_port_item_statements(body, binding)
    attribute_features = list(re.finditer(
        rf"\battribute\s+{re.escape(binding.port_feature)}\b"
        rf"[^;{{}}]*\s*;",
        body,
    ))
    if attribute_features:
        if named_features or len(attribute_features) != 1:
            issues.append(
                f"{binding.port_type}.{binding.port_feature} appears in "
                "multiple incompatible feature forms"
            )
            return text
        feature = attribute_features[0]
        canonical = (
            f"in item {binding.port_feature} : {binding.item_type};"
        )
        body = body[:feature.start()] + canonical + body[feature.end():]
        changes.append(
            f"canonicalized port payload "
            f"{binding.port_type}.{binding.port_feature}"
        )
        return text[:text.find("{", start, end) + 1] + body + text[closing:]
    if len(named_features) > 1:
        issues.append(
            f"{binding.port_type}.{binding.port_feature} appears "
            f"{len(named_features)} times; expected exactly once"
        )
        return text
    features = _port_item_usages(body, binding)
    if len(named_features) == 1:
        if len(features) != 1:
            issues.append(
                f"{binding.port_type}.{binding.port_feature} is not a "
                "canonical typed item feature"
            )
            return text
        if features[0].group("type") != binding.item_type:
            issues.append(
                f"{binding.port_type}.{binding.port_feature} carries "
                f"{features[0].group('type')}, expected {binding.item_type}"
            )
        return text
    insertion = (
        f"\n        in item {binding.port_feature} : "
        f"{binding.item_type};\n    "
    )
    changes.append(
        f"port payload {binding.port_type}.{binding.port_feature}"
    )
    return text[:closing] + insertion + text[closing:]


def _ensure_owner_binding(
    text: str,
    binding: SemanticBindingPlan,
    obligation: RequirementSemanticObligation,
    changes: list[str],
    issues: list[str],
) -> str:
    span = _definition_span(text, "part", binding.target_component)
    if span is None or span[2] is None:
        issues.append(
            f"cannot locate part def {binding.target_component} for "
            f"{binding.obligation_id}"
        )
        return text
    start, end, closing = span
    assert closing is not None
    opening = text.find("{", start, end)
    body = text[opening + 1:closing]

    runtime_statement = (
        f"attribute {binding.runtime_attribute} : {binding.value_type} = "
        f"{binding.source_path};"
    )
    threshold_statement = (
        f"attribute {binding.threshold_attribute} : {binding.value_type} = "
        f"{obligation.threshold:g} [{obligation.unit}];"
    )
    statements = (
        (binding.runtime_attribute, runtime_statement),
        (binding.threshold_attribute, threshold_statement),
    )
    for name, statement in statements:
        # The type may carry a unit suffix (`: LengthValue [m]`) and the
        # declaration may have no initializer at all. A pattern that requires
        # neither form cannot see an existing declaration written that way, and
        # then this appends a second one — a duplicate the namespace-integrity
        # check correctly rejects, measured twice on real runs. Matching both
        # routes the declaration into the rebinding branch below instead.
        match = re.search(
            rf"\battribute\s+{re.escape(name)}"
            rf"(?:\s*:\s*[A-Za-z_][\w:]*(?:\s*\[[^\]{{}}]*\])?)?"
            rf"(?:\s*=\s*[^;{{}}]+)?\s*;",
            body,
        )
        if match is None:
            body += f"\n        {statement}"
            changes.append(
                f"attribute {binding.target_component}.{name}"
            )
        else:
            observed = " ".join(match.group(0).split())
            expected = " ".join(statement.split())
            if observed != expected:
                body = body[:match.start()] + statement + body[match.end():]
                changes.append(
                    f"rebound attribute {binding.target_component}.{name}"
                )

    body = body.rstrip() + "\n    "
    return text[:opening + 1] + body + text[closing:]


def _check_materialized_binding(
    text: str,
    binding: SemanticBindingPlan,
    obligation: RequirementSemanticObligation,
) -> list[str]:
    issues: list[str] = []
    for library in ("ISQ", "SI"):
        if not _has_root_package_import(text, library):
            issues.append(
                f"root package must import {library}::* for "
                f"{binding.value_type} [{binding.unit}]"
            )
    item_span = _definition_span(text, "item", binding.item_type)
    if item_span is None or item_span[2] is None:
        issues.append(f"item def {binding.item_type} is missing")
    else:
        body = text[
            text.find("{", item_span[0], item_span[1]) + 1:item_span[2]
        ]
        named_features = _named_item_attribute_statements(body, binding)
        features = _item_attribute_usages(body, binding)
        if len(named_features) != 1 or len(features) != 1:
            issues.append(
                f"item feature {binding.item_type}."
                f"{binding.item_feature} must appear exactly once"
            )
        elif feature_issues := _validate_item_attribute_usage(
            features[0], binding
        ):
            issues.extend(feature_issues)

    port_span = _definition_span(text, "port", binding.port_type)
    if port_span is None or port_span[2] is None:
        issues.append(f"port def {binding.port_type} is missing")
    else:
        body = text[
            text.find("{", port_span[0], port_span[1]) + 1:port_span[2]
        ]
        named_features = _named_port_item_statements(body, binding)
        features = _port_item_usages(body, binding)
        if len(named_features) != 1 or len(features) != 1:
            issues.append(
                f"port payload {binding.port_type}."
                f"{binding.port_feature} must appear exactly once"
            )
        elif features[0].group("type") != binding.item_type:
            issues.append(
                f"port payload {binding.port_type}."
                f"{binding.port_feature} carries "
                f"{features[0].group('type')}, expected {binding.item_type}"
            )

    owner_span = _definition_span(text, "part", binding.target_component)
    if owner_span is None or owner_span[2] is None:
        issues.append(f"part def {binding.target_component} is missing")
        return issues
    owner_body = text[
        text.find("{", owner_span[0], owner_span[1]) + 1:owner_span[2]
    ]
    runtime = re.search(
        rf"\battribute\s+{re.escape(binding.runtime_attribute)}"
        rf"\s*:\s*{re.escape(binding.value_type)}\s*=\s*"
        rf"(?P<value>[^;{{}}]+)\s*;",
        owner_body,
    )
    if runtime is None or " ".join(runtime.group("value").split()) != (
        binding.source_path
    ):
        issues.append(
            f"{binding.target_component}.{binding.runtime_attribute} is not "
            f"bound to {binding.source_path}"
        )
    threshold = re.search(
        rf"\battribute\s+{re.escape(binding.threshold_attribute)}"
        rf"\s*:\s*{re.escape(binding.value_type)}\s*=\s*"
        rf"(?P<value>[^;{{}}]+)\s*;",
        owner_body,
    )
    numeric = (
        _numeric_value(threshold.group("value"), {})
        if threshold is not None else None
    )
    if numeric != (obligation.threshold, obligation.unit):
        issues.append(
            f"{binding.target_component}.{binding.threshold_attribute} does "
            "not preserve the frozen threshold and unit"
        )
    return issues


def materialize_semantic_bindings(
    model_text: str,
    bindings: Sequence[SemanticBindingPlan],
    obligations: Sequence[RequirementSemanticObligation],
) -> tuple[str, dict[str, Any]]:
    """Transactionally materialize the frozen typed semantic data chain."""
    original = str(model_text or "")
    obligations_by_id = {
        item.obligation_id: item for item in obligations
    }
    working = original
    changes: list[str] = []
    issues: list[str] = []
    results: list[dict[str, Any]] = []

    if bindings:
        working = _ensure_quantity_imports(
            working, changes, issues
        )
    for binding in bindings:
        obligation = obligations_by_id.get(binding.obligation_id)
        if obligation is None:
            issues.append(
                f"{binding.obligation_id} has no frozen semantic obligation"
            )
            continue
        working = _ensure_item_feature(
            working, binding, changes, issues
        )
        working = _ensure_port_payload(
            working, binding, changes, issues
        )
        working = _ensure_owner_binding(
            working, binding, obligation, changes, issues
        )

    for binding in bindings:
        obligation = obligations_by_id.get(binding.obligation_id)
        binding_issues = (
            _check_materialized_binding(working, binding, obligation)
            if obligation is not None else [
                "frozen semantic obligation is missing"
            ]
        )
        results.append({
            "obligation_id": binding.obligation_id,
            "requirement_id": binding.requirement_id,
            "source_path": binding.source_path,
            "quantity_type": binding.value_type,
            "unit": binding.unit,
            "status": "PASS" if not binding_issues else "FAIL",
            "issues": binding_issues,
        })
        issues.extend(binding_issues)

    expected_ids = set(obligations_by_id)
    bound_ids = {item.obligation_id for item in bindings}
    for obligation_id in sorted(expected_ids - bound_ids):
        issues.append(f"{obligation_id} has no typed semantic binding")

    committed = not issues
    output = working if committed else original
    return (working if committed else original), {
        "schema_version": "1.0",
        "artifact_role": "TYPED_SEMANTIC_BINDING_CONFORMANCE",
        "status": (
            "PASS"
            if bindings and committed
            else "NOT_APPLICABLE"
            if not obligations
            else "FAIL"
        ),
        "transaction_committed": committed,
        "input_model_digest": hashlib.sha256(
            original.encode("utf-8")
        ).hexdigest(),
        "output_model_digest": hashlib.sha256(
            output.encode("utf-8")
        ).hexdigest(),
        "planned_binding_count": len(bindings),
        "materialized_binding_count": sum(
            item["status"] == "PASS" for item in results
        ),
        "deterministic_changes": list(dict.fromkeys(changes)),
        "results": results,
        "issues": list(dict.fromkeys(issues)),
    }


def validate_semantic_bindings(
    model_text: str,
    bindings: Sequence[SemanticBindingPlan],
    obligations: Sequence[RequirementSemanticObligation],
) -> dict[str, Any]:
    """Check the exact frozen binding chain without modifying the candidate."""
    text = str(model_text or "")
    obligations_by_id = {
        item.obligation_id: item for item in obligations
    }
    results: list[dict[str, Any]] = []
    issues: list[str] = []
    for binding in bindings:
        obligation = obligations_by_id.get(binding.obligation_id)
        binding_issues = (
            _check_materialized_binding(text, binding, obligation)
            if obligation is not None else [
                "frozen semantic obligation is missing"
            ]
        )
        results.append({
            "obligation_id": binding.obligation_id,
            "requirement_id": binding.requirement_id,
            "source_path": binding.source_path,
            "quantity_type": binding.value_type,
            "unit": binding.unit,
            "status": "PASS" if not binding_issues else "FAIL",
            "issues": binding_issues,
        })
        issues.extend(binding_issues)
    expected_ids = set(obligations_by_id)
    bound_ids = {item.obligation_id for item in bindings}
    for obligation_id in sorted(expected_ids - bound_ids):
        issues.append(f"{obligation_id} has no typed semantic binding")
    return {
        "schema_version": "1.0",
        "artifact_role": "TYPED_SEMANTIC_BINDING_CONFORMANCE",
        "status": (
            "PASS"
            if bindings and not issues
            else "NOT_APPLICABLE"
            if not obligations
            else "FAIL"
        ),
        "planned_binding_count": len(bindings),
        "materialized_binding_count": sum(
            item["status"] == "PASS" for item in results
        ),
        "results": results,
        "issues": list(dict.fromkeys(issues)),
    }


def validate_requirement_semantic_obligations(
    model_text: str,
    obligations: Sequence[RequirementSemanticObligation],
    *,
    model_name: str,
    bindings: Sequence[SemanticBindingPlan] | None = None,
) -> dict[str, Any]:
    """Validate frozen numeric semantics against the terminal SysML model.

    Passing means the model preserves a source-bound quantity and its explicit
    constraint.  It does not establish environmental validity, controller
    adequacy, or physical safety.
    """
    text = str(model_text or "")
    parts = _extract_part_definitions(text)
    results: list[dict[str, Any]] = []
    bindings_by_obligation = {
        item.obligation_id: item for item in (bindings or ())
    }

    for obligation in obligations:
        candidates = [
            (name, block)
            for name, block in parts
            if re.search(
                rf"\bsatisfy\s+requirement\s+"
                rf"{re.escape(obligation.requirement_id)}\s*;",
                block,
            )
        ]
        evidence: list[dict[str, Any]] = []
        candidate_issues: list[str] = []
        passed_owner: str | None = None

        if not candidates:
            candidate_issues.append(
                f"no part definition satisfies {obligation.requirement_id}"
            )

        for owner, block in candidates:
            inputs = {
                match.group("name")
                for match in _PORT_RE.finditer(block)
                if match.group("direction") in {"in", "inout"}
            }
            attributes = {
                match.group("name"): match.group("value").strip()
                for match in _ATTRIBUTE_RE.finditer(block)
            }
            owner_evidence: list[dict[str, Any]] = []
            owner_issues: list[str] = []
            matching_assertions = 0

            expected_binding = bindings_by_obligation.get(
                obligation.obligation_id
            )
            for (
                assertion_name,
                expression,
                assertion_activation,
            ) in _extract_assertions(block):
                if (
                    expected_binding is not None
                    and assertion_name != expected_binding.constraint_name
                ):
                    continue
                comparison = _comparison_fidelity(
                    expression,
                    obligation,
                    inputs=inputs,
                    attributes=attributes,
                )
                if comparison is None:
                    continue
                matching_assertions += 1
                numeric = comparison["numeric_bound"]
                issues: list[str] = []
                if comparison["operator"] != obligation.operator:
                    issues.append(
                        f"operator {comparison['operator']} does not preserve "
                        f"{obligation.operator}"
                    )
                if numeric is None:
                    issues.append("constraint bound is not a resolvable number")
                else:
                    threshold, unit = numeric
                    if threshold != obligation.threshold:
                        issues.append(
                            f"threshold {threshold:g} does not preserve "
                            f"{obligation.threshold:g}"
                        )
                    if _normalise_unit(unit) != obligation.unit:
                        issues.append(
                            f"unit {unit or '(missing)'} does not preserve "
                            f"{obligation.unit}"
                        )
                if not comparison["source_bound"]:
                    issues.append(
                        "constrained subject is not bound to an input-port "
                        "data feature"
                    )
                if (
                    obligation.activation_kind == "CONTEXTUAL"
                    and assertion_activation != "STATE_ACTIVE"
                ):
                    issues.append(
                        "contextual source clause is not preserved by a "
                        "STATE_ACTIVE planned constraint"
                    )
                owner_evidence.append({
                    "assertion": assertion_name,
                    "activation": assertion_activation,
                    **comparison,
                    "status": "PASS" if not issues else "FAIL",
                    "issues": issues,
                })

            if matching_assertions == 0:
                owner_issues.append(
                    "no assert constraint expresses the frozen subject bound"
                )
            owner_issues.extend(
                _late_response_guards(
                    block,
                    obligation,
                    attributes=attributes,
                )
            )
            passing_assertion = any(
                item["status"] == "PASS" for item in owner_evidence
            )
            if not passing_assertion and matching_assertions:
                owner_issues.append(
                    "no matching assertion preserves runtime binding, "
                    "operator, threshold, and unit"
                )
            evidence.append({
                "owner": owner,
                "input_ports": sorted(inputs),
                "assertions": owner_evidence,
                "status": (
                    "PASS"
                    if passing_assertion and not owner_issues
                    else "FAIL"
                ),
                "issues": list(dict.fromkeys(owner_issues)),
            })
            if passing_assertion and not owner_issues:
                passed_owner = owner
                break

        if passed_owner is None:
            candidate_issues.extend(
                issue
                for item in evidence
                for issue in item["issues"]
            )
            candidate_issues.extend(
                issue
                for item in evidence
                for assertion in item["assertions"]
                for issue in assertion["issues"]
            )
        results.append({
            **obligation.to_dict(),
            "status": "PASS" if passed_owner is not None else "FAIL",
            "satisfying_owner": passed_owner,
            "candidate_evidence": evidence,
            "issues": list(dict.fromkeys(candidate_issues)),
        })

    passed = sum(item["status"] == "PASS" for item in results)
    binding_report: dict[str, Any] | None = None
    if bindings is not None:
        binding_report = validate_semantic_bindings(
            text,
            bindings,
            obligations,
        )
    if not obligations:
        status = "UNVERIFIED"
    else:
        obligations_pass = passed == len(results)
        bindings_pass = (
            binding_report is None
            or binding_report["status"] == "PASS"
        )
        status = "PASS" if obligations_pass and bindings_pass else "FAIL"
    return {
        "schema_version": "2.0" if bindings is not None else "1.0",
        "artifact_role": "REQUIREMENT_MODEL_SEMANTIC_FIDELITY",
        "status": status,
        "claim_boundary": "MODEL_SEMANTIC_FIDELITY_NOT_PHYSICAL_PROOF",
        "scenario_set_fixed": True,
        "model_name": model_name,
        "source_model_digest": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "passed": passed,
        "total": len(results),
        "results": results,
        "typed_binding_conformance": binding_report,
    }
