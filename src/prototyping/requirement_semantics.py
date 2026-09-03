"""Bounded, source-derived semantic anchors for whole-model generation.

Extracts only explicit numeric bounds stated by the stakeholder (for example,
"maintain at least 5 metres of separation while avoiding it"); the anchor
preserves comparator, threshold, unit, subject terms and a bounded activation
qualifier across generation, and stays digest-bound to its source. Not a
temporal proof system.
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
# Bound-declaring verbs, inflected: the flagship requirement set writes
# "maintaining ..." and "shall achieve/sustain ...", and a base-form-only verb
# list compiled zero obligations from all 29 drone_v2 requirements, so the
# semantic-fidelity layer ran vacuously on the main experiment input (ablation
# pilot 20260829). Every extension below stays anchored on verb + comparator +
# number + unit; trigger conditions ("when ... reaches 25%") and scenario
# envelopes ("at a closing speed no greater than 1.5 m/s") carry no bound verb
# and compile to nothing.
_BOUND_VERBS = (
    r"\b(?:maintain(?:s|ing)?|keep(?:s|ing)?|ensur(?:e|es|ing)|"
    r"sustain(?:s|ing)?|achiev(?:e|es|ing))"
)
# "m/s" and "minutes|min" precede the bare "m" alternative, or they match
# only as their prefixes (same lesson as activated_constraint_plan).
_BOUND_UNITS = (
    r"milliseconds?|ms|seconds?|s|minutes?|min|"
    r"m/s|metres?|meters?|m|kilometres?|kilometers?|km|"
    r"kilograms?|kg|degrees?|deg|percent|%|hertz|hz"
)
# Where a bound clause may end: clause connectives and common prepositions,
# so trailing context ("... 18 m/s in nil-wind", "... 120 metres above ground
# level") terminates the match instead of failing it.
_CLAUSE_BOUNDARY = (
    r"(?=\s+(?:while|when|during|after|before|unless|and|in|on|at|for|"
    r"with|from|above|below|over|under|per|throughout)\b|[.,;]|$)"
)
_GE_COMPARATORS = {
    "at least", "no less than", "minimum", "minimum of",
}
_MAINTAIN_BOUND_RE = re.compile(
    _BOUND_VERBS + r"\s+"
    r"(?:(?P<subject_before>[A-Za-z][A-Za-z0-9 _-]{0,80}?)\s+)?"
    r"(?P<comparator>"
    r"at\s+least|no\s+less\s+than|minimum(?:\s+of)?|"
    r"at\s+most|no\s+more\s+than|no\s+greater\s+than|maximum(?:\s+of)?|"
    r"within"
    r")\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>" + _BOUND_UNITS + r")"
    r"(?:\s+RMS)?"
    r"(?:\s+of\s+(?P<subject_after>"
    r"[A-Za-z][A-Za-z0-9 _-]{0,80}?))?"
    + _CLAUSE_BOUNDARY,
    re.IGNORECASE,
)
# "maintain a minimum forward ground speed of 2 m/s" - the comparator
# precedes the subject, so the maintain rule (comparator directly before the
# value) cannot see it.
_MINMAX_OF_BOUND_RE = re.compile(
    _BOUND_VERBS + r"\s+"
    r"(?:(?:a|an|the)\s+)?(?P<comparator>minimum|maximum)\s+"
    r"(?P<subject>[A-Za-z][A-Za-z0-9 _-]{0,80}?)\s+of\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>" + _BOUND_UNITS + r")"
    r"(?:\s+RMS)?"
    + _CLAUSE_BOUNDARY,
    re.IGNORECASE,
)
_NOT_EXCEED_BOUND_RE = re.compile(
    r"(?:(?P<subject_before>[A-Za-z][A-Za-z0-9 _,()-]{0,90}?)\s+)?"
    r"(?:shall|must)\s+not\s+exceed\s+"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?P<subject_after>[A-Za-z][A-Za-z0-9 _-]{0,80}?)\s+of\s+)?"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>" + _BOUND_UNITS + r")"
    r"(?:\s+RMS)?"
    + _CLAUSE_BOUNDARY,
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
    # Glue words the broadened bound patterns can sweep into a subject span.
    # _matches_subject requires every term in the bound identifier, so one stray
    # connective makes an obligation unbindable.
    "and", "for", "of", "to", "including", "system",
}
# Unit identity comes from the single registry (unit_registry.py); when it
# was fragmented, `m/s` was planned as "m/s", authored as "[m_s]", never
# resolved and compared unequal (ablation pilots). Binding validation refuses a
# unit without a quantity-type mapping, so every unit the bound patterns emit
# carries one, syside-verified by tests/test_stdlib_vocabulary.py.
from .unit_registry import (  # noqa: E402
    CANONICAL_BY_SPELLING as _UNIT_CANONICAL,
    EMISSION_BY_SPELLING as _UNIT_EMISSION,
    QUANTITY_TYPE_BY_CANONICAL as _UNIT_QUANTITY_TYPES,
)


def _emission_unit(value: str | None) -> str:
    raw = str(value or "").strip()
    return _UNIT_EMISSION.get(raw.lower(), raw)
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
# The transition name is optional in SysML v2, so it is captured only when
# present; the pattern used to demand one, letting an unnamed guarded
# transition escape the late-response check.
_TRANSITION_GUARD_RE = re.compile(
    r"\btransition\b(?:\s+(?P<name>(?!first\b)[A-Za-z_]\w*))?"
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


def _constraint_delegation(text: str, constraint_name: str) -> str | None:
    """The declared higher-fidelity tier for a planned constraint, if any.

    Recognises the structured marker the capability normaliser emits when it
    removes a mission-end invariant, plus the legacy waiver line it used to emit
    (both deterministic emitter output). The legacy form is accepted only adjacent
    to the constraint's own PLAN-CONSTRAINT marker, so an unrelated comment cannot
    delegate an obligation.
    """
    if not constraint_name:
        return None
    structured = re.search(
        rf"//\s*DELEGATED-CONSTRAINT\s+{re.escape(constraint_name)}\s+"
        rf"tier=(?P<tier>[A-Z_]+)",
        text,
    )
    if structured:
        return structured.group("tier")
    legacy = re.search(
        rf"//\s*PLAN-CONSTRAINT\s+{re.escape(constraint_name)}\b[^\n]*\n"
        rf"[ \t]*//\s*Operational range is a mission-end capability",
        text,
    )
    if legacy:
        return "FORWARD_FLIGHT_FIDELITY"
    return None


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
    bound_subject: str | None = None,
) -> tuple[bool, str]:
    candidate = expression.strip()

    def _subject_ok(identifier: str) -> bool:
        # The plan's semantic binding declares the runtime attribute that
        # carries this obligation's subject (currentAltitude for "flight
        # altitude"), and that identity outranks term matching: the declared
        # name drops the requirement text's qualifier words, so demanding every
        # term rejected the attributes the plan had materialised (s0v16: all
        # four "failing" obligations had their planned assert in place under
        # the planned name). Term matching still covers obligations no binding
        # covers.
        if bound_subject is not None and identifier == bound_subject:
            return True
        return _matches_subject(identifier, subject_terms)

    if candidate in attributes:
        binding = attributes[candidate].strip()
        if (
            _PATH_RE.fullmatch(binding)
            and binding.split(".", 1)[0] in inputs
            and (
                _subject_ok(candidate)
                or _matches_subject(binding, subject_terms)
            )
        ):
            return True, f"{candidate} = {binding}"
    if (
        _PATH_RE.fullmatch(candidate)
        and candidate.split(".", 1)[0] in inputs
        and _subject_ok(candidate)
    ):
        return True, candidate
    return False, ""


def _comparison_fidelity(
    expression: str,
    obligation: "RequirementSemanticObligation",
    *,
    inputs: set[str],
    attributes: dict[str, str],
    bound_subject: str | None = None,
) -> dict[str, Any] | None:
    match = _COMPARISON_RE.fullmatch(" ".join(expression.split()))
    if match is None:
        return None
    left = match.group("left")
    right = match.group("right")
    operator = match.group("operator")

    def _is_subject(identifier: str) -> bool:
        if bound_subject is not None and identifier.strip() == bound_subject:
            return True
        return _matches_subject(identifier, obligation.subject_terms)

    subject = left
    bound = right
    if not _is_subject(left):
        if not _is_subject(right):
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
        bound_subject=bound_subject,
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
    issues: list[str] = []
    for transition in _TRANSITION_GUARD_RE.finditer(block):
        response_name = (
            (transition.group("name") or "")
            + " " + transition.group("target")
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
        # (span start, span end, subject text, operator, match) per rule. The rules
        # partition the phrasings (comparator directly before the value, embedded
        # min/max subject, not-exceed verb phrase), and the span-overlap drop below
        # keeps a pattern change from compiling one clause twice.
        found: list[tuple[int, int, str, str, "re.Match[str]"]] = []
        for match in _MAINTAIN_BOUND_RE.finditer(body):
            subject = (
                match.group("subject_after")
                or match.group("subject_before")
                or ""
            )
            comparator = " ".join(
                match.group("comparator").lower().split()
            )
            operator = ">=" if comparator in _GE_COMPARATORS else "<="
            found.append(
                (match.start(), match.end(), subject, operator, match)
            )
        for match in _MINMAX_OF_BOUND_RE.finditer(body):
            operator = (
                ">="
                if match.group("comparator").lower() == "minimum"
                else "<="
            )
            found.append((
                match.start(), match.end(),
                match.group("subject"), operator, match,
            ))
        for match in _NOT_EXCEED_BOUND_RE.finditer(body):
            subject = (
                match.group("subject_after")
                or match.group("subject_before")
                or ""
            )
            found.append(
                (match.start(), match.end(), subject, "<=", match)
            )

        found.sort(key=lambda item: (item[0], item[1]))
        index = 0
        previous_end = -1
        for start, end, subject, operator, match in found:
            if start < previous_end:
                continue   # a longer earlier match already compiled this span
            terms = _subject_terms(subject)
            if not terms:
                continue
            previous_end = end
            index += 1
            raw_unit = match.group("unit").lower()
            activation_match = _ACTIVATION_CLAUSE_RE.match(body[end:])
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
        rf"\b{re.escape(kind)}\s+def\s+{re.escape(name)}\b[^{{;]*(?P<tail>[;{{])",
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

    A binding freezes ``item_type`` and ``port_type`` before textual generation,
    and LLM fragments still occasionally serialize those names as an ``attribute
    def``. That is notation drift, so the compiler replaces exactly one such root
    declaration; other collisions stay fail-closed, since a ``part def`` may carry
    real structure and is not rewritten into an item or port.
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
        f"{obligation.threshold:g} [{_emission_unit(obligation.unit)}];"
    )
    statements = (
        (binding.runtime_attribute, runtime_statement),
        (binding.threshold_attribute, threshold_statement),
    )
    for name, statement in statements:
        # The type may carry a unit suffix (`: LengthValue [m]`) and the
        # declaration may have no initializer. A pattern blind to those forms misses
        # the existing declaration and appends a second one - a duplicate the
        # namespace-integrity check rejects, measured twice on real runs. Matching
        # both routes the declaration into the rebinding branch below.
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
    # The model carries the emission token (m_s), the obligation the canonical
    # spelling (m/s) - equality is judged on canonical identity, same as the
    # assertion check below.
    preserved = (
        numeric is not None
        and numeric[0] == obligation.threshold
        and _normalise_unit(numeric[1]) == _normalise_unit(obligation.unit)
    )
    if not preserved:
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
    """Materialize the frozen typed semantic data chain, one transaction per binding.

    ``transaction_committed`` is True only when every binding and every
    obligation's coverage succeeded; a failing binding reverts itself without
    discarding its siblings, and the returned text is what the report describes.
    """
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
    # Each binding is its own transaction: ensure on a candidate copy, commit
    # when the materialized chain checks out, revert only that binding otherwise.
    # Under all-or-nothing semantics (pilot 2), two m/s bindings failed on a unit
    # token, the shared rollback discarded seven healthy bindings with them, and
    # the reported conformance described a working copy the published model never
    # contained. The returned text and this report now describe each other.
    for binding in bindings:
        obligation = obligations_by_id.get(binding.obligation_id)
        if obligation is None:
            binding_issues = [
                f"{binding.obligation_id} has no frozen semantic obligation"
            ]
        else:
            candidate = working
            candidate_changes: list[str] = []
            binding_issues = []
            candidate = _ensure_item_feature(
                candidate, binding, candidate_changes, binding_issues
            )
            candidate = _ensure_port_payload(
                candidate, binding, candidate_changes, binding_issues
            )
            candidate = _ensure_owner_binding(
                candidate, binding, obligation, candidate_changes,
                binding_issues,
            )
            binding_issues = list(dict.fromkeys(
                binding_issues
                + _check_materialized_binding(candidate, binding, obligation)
            ))
            if not binding_issues:
                working = candidate
                changes.extend(candidate_changes)
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
    output = working
    return working, {
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
        # The plan may place the runtime constraint chain in a different component
        # from the one carrying the satisfy allocation (run 219eb9bb: MTOW and
        # endurance asserts in FlightController, satisfy links on Airframe and
        # PowerSystem). The typed binding's declared target owner is part of the
        # frozen chain, so it counts as a candidate; the assertion checks stay strict.
        expected_binding = bindings_by_obligation.get(obligation.obligation_id)
        if expected_binding is not None and not any(
            name == expected_binding.target_component
            for name, _ in candidates
        ):
            for name, block in parts:
                if name == expected_binding.target_component:
                    candidates.append((name, block))
                    break

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
                    bound_subject=(
                        expected_binding.runtime_attribute
                        if expected_binding is not None else None
                    ),
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

        # A planned constraint the capability normaliser removed (a mission-end
        # bound is not a runtime invariant) is verified by a higher-fidelity tier
        # rather than an inline assert; the delegation marker records that routing so
        # it is auditable.
        delegation: str | None = None
        if passed_owner is None and expected_binding is not None:
            delegation = _constraint_delegation(
                text, expected_binding.constraint_name
            )
        if passed_owner is None and delegation is None:
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
            "status": (
                "PASS"
                if passed_owner is not None
                else "DELEGATED"
                if delegation is not None
                else "FAIL"
            ),
            "satisfying_owner": passed_owner,
            "delegated_to": delegation,
            "candidate_evidence": evidence,
            "issues": (
                [] if delegation is not None and passed_owner is None
                else list(dict.fromkeys(candidate_issues))
            ),
        })

    passed = sum(item["status"] == "PASS" for item in results)
    delegated = sum(item["status"] == "DELEGATED" for item in results)
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
        obligations_pass = passed + delegated == len(results)
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
        "delegated": delegated,
        "total": len(results),
        "results": results,
        "typed_binding_conformance": binding_report,
    }
