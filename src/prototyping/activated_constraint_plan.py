"""Plan-first attributes and activated constraint obligations.

The plan is a generation/audit input.  The committed SysML v2 model remains
the semantic authority. ``ALWAYS`` obligations are materialised at part scope;
``STATE_ACTIVE`` obligations are materialised as assert constraints owned by
the referenced state usage. This uses valid SysML v2 containment semantics
instead of inventing a temporal expression language.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..utils.req_id import normalise_req_id
from ..utils.sysml_text_utils import find_block_end


#: Value types an attribute may be planned with. The plan has no way to declare
#: a new type, so anything outside this set is emitted as a reference to
#: something that does not exist.
#:
#: Measured: one run planned `lockState : StateEnum = Locked`, the materialiser
#: wrote it faithfully, and the model failed Syside with "No Type named
#: 'StateEnum' found." Validation had only checked that the name was a
#: well-formed identifier, which `StateEnum` is. Failing the plan instead lets
#: generation retry, which is what the bounded attempt budget is for.
RESOLVABLE_VALUE_TYPES = frozenset({
    # ScalarValues
    "Boolean", "Integer", "Natural", "Rational", "Real", "String",
    # ISQ / SI quantity values used by the emitters
    "DurationValue", "LengthValue", "MassValue", "TimeValue",
    "SpeedValue", "AccelerationValue", "AngleValue", "TemperatureValue",
    "ElectricCurrentValue", "PowerValue", "EnergyValue", "FrequencyValue",
})


def _bare_type(value_type: str) -> str:
    """The last segment of a possibly qualified type name."""
    return str(value_type or "").rsplit("::", 1)[-1].strip()


ATTRIBUTE_ROLES = {
    "RUNTIME_MEASUREMENT",
    "FROZEN_THRESHOLD",
    "DESIGN_PARAMETER",
    "LOCAL_STATE",
}
PROVENANCE_KINDS = {
    "FROZEN_REQUIREMENT",
    "A_G_GUARANTEE",
    "DESIGN_DECISION",
}
ACTIVATION_KINDS = {"ALWAYS", "STATE_ACTIVE"}
VERIFICATION_TIERS = {
    "PARAMETRIC_SWEEP",
    "STATE_EXECUTION",
    "EXTERNAL_ANALYSIS",
    "INSPECTION",
}
#: Operators whose satisfaction boundary the STATE_EXECUTION executor can probe.
#: It perturbs the right-hand value by epsilon and requires one side to satisfy
#: and the other not to. Equality fails that by construction — both perturbed
#: sides violate it — so `behavioral_sim` reports "boundary == N is not live" for
#: every `==` constraint regardless of the model. Measured on
#: pilot_n6_20260802/seed-3, the only seed that planned equality state
#: constraints and the only one in its arm to lose the qualification gate.
_LIVE_BOUNDARY_OPERATORS = frozenset({"<=", ">=", "<", ">"})


def _state_execution_obstacle(constraint, lhs) -> str | None:
    """Why STATE_EXECUTION cannot discharge this constraint, or None.

    Mirrors what `behavioral_sim._run_state_active_constraint_scenario` actually
    requires. The two were previously allowed to disagree: the plan validator
    forced every STATE_ACTIVE constraint to claim STATE_EXECUTION, and the
    executor then rejected the ones it had no machinery for. A plan that commits
    to evidence the system cannot produce is worse than one that says so.
    """
    if constraint.operator not in _LIVE_BOUNDARY_OPERATORS:
        return (
            f"`{constraint.operator}` has no live satisfaction boundary to "
            "execute against"
        )
    # Deliberately NOT checked here: the executor also demands the subject be an
    # attribute bound to a dotted input path, and rejects `LOCAL_STATE` values
    # that carry only an initial value. The two components already disagreed on
    # that before this change — `test_state_active_property_is_serialized_inside_
    # owning_state` plans exactly such a constraint and the validator has always
    # passed it. Aligning them would reject plans that are legal today, and it is
    # not established which side is wrong: the executor may be over-strict rather
    # than the plan over-permissive. Reported, not silently repaired.
    return None


def state_execution_advisories(
    constraints: Sequence[Any],
    components: Sequence[Any],
) -> list[str]:
    """Report STATE_EXECUTION constraints the executor cannot discharge.

    Deliberately advisory. ``_state_execution_obstacle`` documents why the
    plan validator does not reject these: it is not established whether the
    executor is over-strict or the plan over-permissive, and rejecting them
    would invalidate plans that are legal today. Until that is decided, the
    disagreement should at least be visible in the run artefacts instead of
    passing silently and reappearing as an unanchored requirement six phases
    later.
    """
    attributes = {
        (str(component.name), attribute.name): attribute
        for component in components
        for attribute in component.attributes
    }
    advisories: list[str] = []
    for constraint in constraints:
        if constraint.verification_tier != "STATE_EXECUTION":
            continue
        lhs = attributes.get((constraint.owner, constraint.lhs))
        if lhs is None:
            continue
        if lhs.input_binding:
            continue
        advisories.append(
            f"constraints[{constraint.constraint_id}] claims STATE_EXECUTION "
            f"but its subject {constraint.owner}.{constraint.lhs} carries "
            f"only the value {lhs.initial_value!r} and no input binding; the "
            "behavioural executor requires a bound runtime measurement, so "
            "this constraint cannot anchor "
            f"{constraint.source_requirement_id or 'its requirement'}"
        )
    return advisories


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_QUALIFIED = re.compile(r"^[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*$")
_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_REQ_ID = re.compile(r"\bREQ[-_][A-Za-z]+[-_]\d+\b", re.IGNORECASE)
_ASSERT = re.compile(
    r"\bassert\s+constraint\s+(?P<name>[A-Za-z_]\w*)\s*\{"
)
_COMPARISON = re.compile(
    r"^\s*(?P<lhs>[A-Za-z_]\w*)\s*"
    r"(?P<operator><=|>=|<|>|==)\s*"
    r"(?P<rhs>[A-Za-z_]\w*|[-+]?\d+(?:\.\d+)?)\s*$"
)
_ATTRIBUTE = re.compile(
    r"\battribute\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*:\s*(?P<type>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
    # A bracket suffix after the type — `LengthValue [m]` — is not how this
    # project writes units (they go on the value: `= 5 [m]`), but generation
    # produces it, and a pattern that cannot see such a declaration reports the
    # attribute as absent and appends a second one. Matching it is what lets the
    # existing replace branch normalise it instead of duplicating it.
    r"(?P<type_suffix>\s*\[[^\]{}]*\])?)?"
    r"(?:\s*=\s*(?P<value>[^;{}]+))?\s*;"
)
_PLAN_COMMENT = re.compile(
    r"(?m)^[ \t]*//\s*PLAN-CONSTRAINT\s+"
    r"(?P<id>[A-Za-z_]\w*)\s+"
    r"provenance=(?P<provenance>[A-Z_]+)\s+"
    r"activation=(?P<activation>[A-Z_]+)\s+"
    r"verification=(?P<verification>[A-Z_]+)"
    r"(?:\s+reference=(?P<reference>[A-Za-z_]\w*::[A-Za-z_]\w*))?\s*$"
)


def _source_map(requirements: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in requirements:
        source = str(raw or "").strip()
        for match in _REQ_ID.findall(source):
            result[normalise_req_id(match)] = source
    return result


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalise_expression(value: str) -> str:
    return " ".join(str(value or "").split())


def _numeric(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _unit_tokens(source: str) -> set[str]:
    tokens = {
        item.lower().replace("%", "percent")
        for item in re.findall(
            # "m/s" must be tried before the bare "m" alternative, or it is
            # only ever seen as the two separate units "m" and "s".
            r"m/s|km/h|"
            r"\b(?:ms|s|m|km|m_s|km_h|N|V|A|W|Hz|kg|g|percent|"
            r"metres?|meters?|seconds?|milliseconds?|kilometres?|kilometers?|"
            r"degrees?|deg|minutes?|min)\b|%",
            source,
            flags=re.IGNORECASE,
        )
    }
    aliases = {
        "metre": "m",
        "metres": "m",
        "meter": "m",
        "meters": "m",
        "second": "s",
        "seconds": "s",
        "millisecond": "ms",
        "milliseconds": "ms",
        "kilometre": "km",
        "kilometres": "km",
        "kilometer": "km",
        "kilometers": "km",
        # A requirement writes "1.0 degree" / "25 minutes" / "15 m/s"; a plan
        # may legitimately carry either spelling, so accept both rather than
        # rejecting every form of a unit the frozen text actually states.
        "degree": "deg",
        "degrees": "deg",
        "minute": "min",
        "minutes": "min",
        "m/s": "m_s",
        "km/h": "km_h",
    }
    return tokens | {aliases[item] for item in tokens if item in aliases}


@dataclass(frozen=True)
class AttributePlan:
    name: str
    value_type: str = "Real"
    unit: str = "1"
    role: str = "DESIGN_PARAMETER"
    initial_value: str | None = None
    input_binding: str | None = None
    provenance: str = "DESIGN_DECISION"
    source_requirement_id: str | None = None
    source_digest: str | None = None

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        requirements: Sequence[str] = (),
    ) -> "AttributePlan":
        sources = _source_map(requirements)
        req_id = str(
            value.get("source_requirement_id")
            or value.get("requirement_id")
            or ""
        ).strip()
        req_id = normalise_req_id(req_id) if req_id else None
        source = sources.get(req_id or "")
        initial = value.get("initial_value")
        binding = value.get("input_binding")
        return cls(
            name=str(value.get("name") or "").strip(),
            value_type=str(value.get("value_type") or "Real").strip(),
            unit=str(value.get("unit") or "1").strip(),
            role=str(value.get("role") or "DESIGN_PARAMETER").strip().upper(),
            initial_value=(
                str(initial).strip() if initial not in (None, "") else None
            ),
            input_binding=(
                str(binding).strip() if binding not in (None, "") else None
            ),
            provenance=str(
                value.get("provenance") or "DESIGN_DECISION"
            ).strip().upper(),
            source_requirement_id=req_id,
            source_digest=(
                _source_digest(source)
                if source else (
                    str(value.get("source_digest") or "").strip() or None
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "unit": self.unit,
            "role": self.role,
            "initial_value": self.initial_value,
            "input_binding": self.input_binding,
            "provenance": self.provenance,
            "source_requirement_id": self.source_requirement_id,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class ConstraintPlan:
    constraint_id: str
    owner: str
    lhs: str
    operator: str
    rhs: str
    activation_kind: str = "ALWAYS"
    activation_ref: str | None = None
    provenance: str = "DESIGN_DECISION"
    verification_tier: str = "INSPECTION"
    source_requirement_id: str | None = None
    source_digest: str | None = None

    @property
    def expression(self) -> str:
        return f"{self.lhs} {self.operator} {self.rhs}"

    @property
    def qualification_effect(self) -> str:
        if self.verification_tier == "EXTERNAL_ANALYSIS":
            return "EXTERNAL_VERIFICATION_READINESS"
        if self.provenance == "FROZEN_REQUIREMENT":
            return "REQUIREMENT_BEHAVIOR_EXECUTION"
        if self.provenance == "A_G_GUARANTEE":
            return "A_G_BEHAVIOR_EXECUTION"
        return "DESIGN_CONSTRAINT_CONSISTENCY"

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        requirements: Sequence[str] = (),
    ) -> "ConstraintPlan":
        expression = value.get("expression")
        if isinstance(expression, Mapping):
            lhs = str(expression.get("lhs") or "").strip()
            operator = str(expression.get("operator") or "").strip()
            rhs = str(expression.get("rhs") or "").strip()
        else:
            match = _COMPARISON.fullmatch(str(expression or ""))
            lhs = match.group("lhs") if match else str(value.get("lhs") or "")
            operator = (
                match.group("operator")
                if match else str(value.get("operator") or "")
            )
            rhs = match.group("rhs") if match else str(value.get("rhs") or "")
        activation = value.get("activation")
        if not isinstance(activation, Mapping):
            activation = {}
        provenance = value.get("provenance")
        if isinstance(provenance, Mapping):
            req_id = str(
                provenance.get("requirement_id")
                or value.get("source_requirement_id")
                or ""
            ).strip()
            provenance_kind = str(
                provenance.get("kind") or "DESIGN_DECISION"
            ).strip().upper()
        else:
            req_id = str(value.get("source_requirement_id") or "").strip()
            provenance_kind = str(
                provenance or "DESIGN_DECISION"
            ).strip().upper()
        req_id = normalise_req_id(req_id) if req_id else None
        source = _source_map(requirements).get(req_id or "")
        archived_digest = None
        if isinstance(provenance, Mapping):
            archived_digest = provenance.get("source_digest")
        if archived_digest in (None, ""):
            archived_digest = value.get("source_digest")
        return cls(
            constraint_id=str(
                value.get("constraint_id") or value.get("name") or ""
            ).strip(),
            owner=str(value.get("owner") or "").strip(),
            lhs=lhs.strip(),
            operator=operator.strip(),
            rhs=rhs.strip(),
            activation_kind=str(
                activation.get("kind")
                or value.get("activation_kind")
                or "ALWAYS"
            ).strip().upper(),
            activation_ref=(
                str(
                    activation.get("state")
                    or activation.get("reference")
                    or value.get("activation_ref")
                ).strip()
                if (
                    activation.get("state")
                    or activation.get("reference")
                    or value.get("activation_ref")
                )
                else None
            ),
            provenance=provenance_kind,
            verification_tier=str(
                value.get("verification_tier") or "INSPECTION"
            ).strip().upper(),
            source_requirement_id=req_id,
            source_digest=(
                _source_digest(source)
                if source else (
                    str(archived_digest).strip()
                    if archived_digest not in (None, "") else None
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "owner": self.owner,
            "expression": {
                "lhs": self.lhs,
                "operator": self.operator,
                "rhs": self.rhs,
            },
            "activation": {
                "kind": self.activation_kind,
                "reference": self.activation_ref,
            },
            "provenance": {
                "kind": self.provenance,
                "requirement_id": self.source_requirement_id,
                "source_digest": self.source_digest,
            },
            "verification_tier": self.verification_tier,
            "qualification_effect": self.qualification_effect,
        }


def validate_constraint_plan(
    constraints: Sequence[ConstraintPlan],
    components: Sequence[Any],
    requirements: Sequence[str],
) -> list[str]:
    """Fail closed on invented requirement facts and invalid activation."""
    issues: list[str] = []
    owners = {str(item.name) for item in components}
    attributes = {
        (str(component.name), attribute.name): attribute
        for component in components
        for attribute in component.attributes
    }
    sources = _source_map(requirements)
    for component in components:
        names: set[str] = set()
        for index, attribute in enumerate(component.attributes):
            prefix = f"{component.name}.attributes[{index}]"
            if attribute.name in names:
                issues.append(
                    f"{component.name} attribute names must be unique"
                )
            names.add(attribute.name)
            if not _IDENTIFIER.fullmatch(attribute.name):
                issues.append(f"{prefix}.name is not a SysML identifier")
            if not _QUALIFIED.fullmatch(attribute.value_type):
                issues.append(f"{prefix}.value_type is not a SysML type")
            elif _bare_type(attribute.value_type) not in RESOLVABLE_VALUE_TYPES:
                issues.append(
                    f"{prefix}.value_type {attribute.value_type!r} names no "
                    "type this pipeline can emit; the plan cannot declare new "
                    "types, so it would be written as an unresolvable reference"
                )
            if attribute.role not in ATTRIBUTE_ROLES:
                issues.append(f"{prefix}.role is unsupported")
            if attribute.provenance not in PROVENANCE_KINDS:
                issues.append(f"{prefix}.provenance is unsupported")
            if (
                attribute.role == "RUNTIME_MEASUREMENT"
                and not attribute.input_binding
            ):
                issues.append(
                    f"{prefix} runtime measurement has no input binding"
                )
            if (
                attribute.provenance == "FROZEN_REQUIREMENT"
                and requirements
                and attribute.source_requirement_id not in sources
            ):
                issues.append(
                    f"{prefix} has no matching frozen source requirement"
                )
    seen: set[tuple[str, str]] = set()
    for index, constraint in enumerate(constraints):
        prefix = f"constraints[{index}]"
        key = (constraint.owner, constraint.constraint_id)
        if key in seen:
            issues.append(
                f"duplicate planned constraint "
                f"{constraint.owner}.{constraint.constraint_id}"
            )
        seen.add(key)
        if not _IDENTIFIER.fullmatch(constraint.constraint_id):
            issues.append(f"{prefix}.constraint_id is not a SysML identifier")
        if constraint.owner not in owners:
            issues.append(f"{prefix}.owner is not a planned component")
        if not _IDENTIFIER.fullmatch(constraint.lhs):
            issues.append(f"{prefix}.expression.lhs is not an attribute name")
        if constraint.operator not in {"<=", ">=", "<", ">", "=="}:
            issues.append(f"{prefix}.expression.operator is unsupported")
        if not (
            _IDENTIFIER.fullmatch(constraint.rhs)
            or _NUMBER.fullmatch(constraint.rhs)
        ):
            issues.append(f"{prefix}.expression.rhs is not a simple value")
        if constraint.activation_kind not in ACTIVATION_KINDS:
            issues.append(f"{prefix}.activation.kind is unsupported")
        if (
            constraint.activation_kind == "STATE_ACTIVE"
            and not constraint.activation_ref
        ):
            issues.append(f"{prefix}.activation requires a state reference")
        if (
            constraint.activation_kind == "STATE_ACTIVE"
            and constraint.activation_ref
            and "::" not in constraint.activation_ref
        ):
            issues.append(
                f"{prefix}.activation must use BehaviorId::StateId"
            )
        if constraint.activation_ref and not _QUALIFIED.fullmatch(
            constraint.activation_ref
        ):
            issues.append(f"{prefix}.activation reference is invalid")
        if constraint.provenance not in PROVENANCE_KINDS:
            issues.append(f"{prefix}.provenance.kind is unsupported")
        if constraint.verification_tier not in VERIFICATION_TIERS:
            issues.append(f"{prefix}.verification_tier is unsupported")
        if (
            constraint.activation_kind != "ALWAYS"
            and constraint.verification_tier == "PARAMETRIC_SWEEP"
        ):
            issues.append(
                f"{prefix} parametric sweep cannot verify conditional activation"
            )
        lhs = attributes.get((constraint.owner, constraint.lhs))
        rhs = attributes.get((constraint.owner, constraint.rhs))
        if constraint.activation_kind == "STATE_ACTIVE":
            obstacle = _state_execution_obstacle(constraint, lhs)
            required = "INSPECTION" if obstacle else "STATE_EXECUTION"
            if constraint.verification_tier != required:
                reason = obstacle or "the subject is a live runtime measurement"
                issues.append(
                    f"{prefix} STATE_ACTIVE constraint requires "
                    f"{required}: {reason}"
                )
        if lhs is None:
            issues.append(
                f"{prefix}.expression.lhs is not a planned owner attribute"
            )
        if rhs is None and not _NUMBER.fullmatch(constraint.rhs):
            issues.append(
                f"{prefix}.expression.rhs is not a planned owner attribute"
            )
        if (
            lhs is not None
            and lhs.role == "RUNTIME_MEASUREMENT"
            and not lhs.input_binding
            and constraint.verification_tier != "STATE_EXECUTION"
        ):
            issues.append(
                f"{prefix} external runtime measurement {lhs.name} "
                "has no input binding"
            )
        if (
            constraint.provenance == "FROZEN_REQUIREMENT"
            and requirements
        ):
            source = sources.get(constraint.source_requirement_id or "")
            if source is None:
                issues.append(
                    f"{prefix} has no matching frozen source requirement"
                )
            else:
                bound = (
                    _numeric(rhs.initial_value)
                    if rhs is not None else _numeric(constraint.rhs)
                )
                source_without_id = _REQ_ID.sub("", source)
                source_numbers = {
                    float(item)
                    for item in re.findall(
                        r"[-+]?\d+(?:\.\d+)?", source_without_id
                    )
                }
                if bound is None or bound not in source_numbers:
                    issues.append(
                        f"{prefix} numeric bound is absent from frozen source"
                    )
                unit = rhs.unit if rhs is not None else (lhs.unit if lhs else "1")
                if (
                    unit not in {"", "1"}
                    and unit.lower().replace("%", "percent")
                    not in _unit_tokens(source)
                ):
                    issues.append(
                        f"{prefix} unit {unit!r} is absent from frozen source"
                    )
        if (
            constraint.activation_kind == "ALWAYS"
            and lhs is not None
            and lhs.initial_value is not None
        ):
            left_value = _numeric(lhs.initial_value)
            right_value = (
                _numeric(rhs.initial_value)
                if rhs is not None else _numeric(constraint.rhs)
            )
            if (
                left_value is not None
                and right_value is not None
                and not _compare(
                    left_value, constraint.operator, right_value
                )
            ):
                issues.append(
                    f"{prefix} ALWAYS constraint is false at planned initial state"
                )
    return list(dict.fromkeys(issues))


def _compare(lhs: float, operator: str, rhs: float) -> bool:
    if operator == "<=":
        return lhs <= rhs
    if operator == ">=":
        return lhs >= rhs
    if operator == "<":
        return lhs < rhs
    if operator == ">":
        return lhs > rhs
    return lhs == rhs


def _definition_spans(
    text: str,
    kind: str = "part",
) -> Iterable[tuple[str, int, int, int]]:
    pattern = re.compile(
        rf"\b{re.escape(kind)}\s+def\s+([A-Za-z_]\w*)\s*\{{"
    )
    for match in pattern.finditer(text):
        opening = text.find("{", match.start(), match.end())
        closing = find_block_end(text, opening)
        if closing != -1:
            yield match.group(1), match.start(), opening, closing


def _owner_at(text: str, offset: int) -> str:
    containing = [
        (name, start, closing)
        for name, start, _opening, closing in _definition_spans(text)
        if start < offset < closing
    ]
    if not containing:
        return "unknown"
    return max(containing, key=lambda item: item[1])[0]


def _assertions(text: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for match in _ASSERT.finditer(text):
        opening = text.find("{", match.start(), match.end())
        closing = find_block_end(text, opening)
        if closing == -1:
            continue
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        previous_line_start = text.rfind(
            "\n", 0, max(0, line_start - 1)
        ) + 1
        previous_line = text[previous_line_start:line_start]
        comment = list(_PLAN_COMMENT.finditer(previous_line))
        metadata = comment[-1].groupdict() if comment else {}
        result.append({
            "name": match.group("name"),
            "owner": _owner_at(text, match.start()),
            "expression": _normalise_expression(text[opening + 1:closing]),
            "start": start,
            "end": closing + 1,
            "line_start": line_start,
            "metadata": metadata,
        })
    return result


def _comment(constraint: ConstraintPlan, indent: str) -> str:
    return (
        f"{indent}// PLAN-CONSTRAINT {constraint.constraint_id} "
        f"provenance={constraint.provenance} "
        f"activation={constraint.activation_kind} "
        f"verification={constraint.verification_tier}"
        + (
            f" reference={constraint.activation_ref}"
            if constraint.activation_ref else ""
        )
        + "\n"
    )


def _activation_state_span(
    text: str,
    constraint: ConstraintPlan,
) -> tuple[tuple[int, int] | None, str | None]:
    """Return the absolute body delimiters for the referenced state usage."""
    if not constraint.activation_ref:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: "
            "activation reference is absent"
        )
    behavior_name, separator, state_name = (
        constraint.activation_ref.partition("::")
    )
    if not separator or not behavior_name or not state_name:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: activation "
            "must use BehaviorId::StateId"
        )
    owner_span = next(
        (
            span for span in _definition_spans(text)
            if span[0] == constraint.owner
        ),
        None,
    )
    if owner_span is None:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: owner is absent"
        )
    behavior_match = re.search(
        rf"\bstate\s+def\s+{re.escape(behavior_name)}\s*\{{",
        text[owner_span[2] + 1:owner_span[3]],
    )
    if behavior_match is None:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: activation "
            f"behavior {behavior_name} is absent from owner"
        )
    behavior_offset = owner_span[2] + 1
    behavior_opening = text.find(
        "{",
        behavior_offset + behavior_match.start(),
        behavior_offset + behavior_match.end(),
    )
    behavior_closing = find_block_end(text, behavior_opening)
    if behavior_closing == -1:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: activation "
            f"behavior {behavior_name} cannot be parsed"
        )
    behavior_body_start = behavior_opening + 1
    behavior_body = text[behavior_body_start:behavior_closing]
    state_match = re.search(
        rf"\bstate\s+(?!def\b){re.escape(state_name)}\s*(?P<tail>[;{{])",
        behavior_body,
    )
    if state_match is None:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: activation state "
            f"{constraint.activation_ref} is absent"
        )
    if state_match.group("tail") != "{":
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: activation state "
            f"{constraint.activation_ref} has no body for owned constraint"
        )
    state_opening = text.find(
        "{",
        behavior_body_start + state_match.start(),
        behavior_body_start + state_match.end(),
    )
    state_closing = find_block_end(text, state_opening)
    if state_closing == -1:
        return None, (
            f"{constraint.owner}.{constraint.constraint_id}: activation state "
            f"{constraint.activation_ref} cannot be parsed"
        )
    return (state_opening, state_closing), None


def _constraint_is_placed(
    text: str,
    assertion: Mapping[str, Any],
    constraint: ConstraintPlan,
) -> bool:
    if constraint.activation_kind == "STATE_ACTIVE":
        span, _issue = _activation_state_span(text, constraint)
        return bool(
            span
            and span[0] < int(assertion["start"]) < span[1]
        )
    # An ALWAYS invariant must not be weakened by nesting it in any state.
    for match in re.finditer(r"\bstate\s+(?!def\b)[A-Za-z_]\w*\s*\{", text):
        opening = text.find("{", match.start(), match.end())
        closing = find_block_end(text, opening)
        if opening < int(assertion["start"]) < closing:
            return False
    return True


def _ensure_activation_state_body(
    text: str,
    constraint: ConstraintPlan,
) -> tuple[str, bool]:
    """Expand a declaration-only target state so it can own the constraint."""
    if not constraint.activation_ref or "::" not in constraint.activation_ref:
        return text, False
    behavior_name, state_name = constraint.activation_ref.split("::", 1)
    owner_span = next(
        (
            span for span in _definition_spans(text)
            if span[0] == constraint.owner
        ),
        None,
    )
    if owner_span is None:
        return text, False
    owner_body_start = owner_span[2] + 1
    owner_body = text[owner_body_start:owner_span[3]]
    behavior = re.search(
        rf"\bstate\s+def\s+{re.escape(behavior_name)}\s*\{{",
        owner_body,
    )
    if behavior is None:
        return text, False
    behavior_opening = text.find(
        "{",
        owner_body_start + behavior.start(),
        owner_body_start + behavior.end(),
    )
    behavior_closing = find_block_end(text, behavior_opening)
    if behavior_closing == -1:
        return text, False
    state = re.search(
        rf"\bstate\s+(?!def\b){re.escape(state_name)}\s*;",
        text[behavior_opening + 1:behavior_closing],
    )
    if state is None:
        return text, False
    semicolon = behavior_opening + 1 + state.end() - 1
    return text[:semicolon] + " {\n            }" + text[semicolon + 1:], True


def _activation_issue(
    text: str,
    constraint: ConstraintPlan,
) -> str | None:
    span, span_issue = _activation_state_span(text, constraint)
    if span_issue is not None:
        return span_issue
    assert span is not None and constraint.activation_ref is not None
    behavior_name, separator, state_name = (
        constraint.activation_ref.partition("::")
    )
    owner_span = next(
        (
            span for span in _definition_spans(text)
            if span[0] == constraint.owner
        ),
        None,
    )
    if owner_span is None:
        return (
            f"{constraint.owner}.{constraint.constraint_id}: owner is absent"
        )
    owner_body_start = owner_span[2] + 1
    owner_body = text[owner_body_start:owner_span[3]]
    behavior_match = re.search(
        rf"\bstate\s+def\s+{re.escape(behavior_name)}\s*\{{",
        owner_body,
    )
    if behavior_match is None:
        return (
            f"{constraint.owner}.{constraint.constraint_id}: activation "
            f"behavior {behavior_name} is absent from owner"
        )
    opening = owner_body.find(
        "{", behavior_match.start(), behavior_match.end()
    )
    closing = find_block_end(owner_body, opening)
    if closing == -1:
        return (
            f"{constraint.owner}.{constraint.constraint_id}: activation "
            f"behavior {behavior_name} cannot be parsed"
        )
    behavior_body = owner_body[opening + 1:closing]
    declared_states = set(re.findall(
        r"\bstate\s+(?!def\b)([A-Za-z_]\w*)\b",
        behavior_body,
    ))
    initial = re.search(
        r"\btransition\s+initial\s+then\s+([A-Za-z_]\w*)\s*;",
        behavior_body,
    )
    if initial is None:
        initial = re.search(
            r"\bentry\s*;\s*then\s+([A-Za-z_]\w*)\s*;",
            behavior_body,
        )
    if initial is None:
        return (
            f"{constraint.owner}.{constraint.constraint_id}: activation "
            f"behavior {behavior_name} has no initial transition"
        )
    edges = [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r"\btransition\b[^;]*?\bfirst\s+([A-Za-z_]\w*)"
            r"\b[^;]*?\bthen\s+([A-Za-z_]\w*)\s*;",
            behavior_body,
            flags=re.DOTALL,
        )
    ]
    reachable = {initial.group(1)}
    changed = True
    while changed:
        changed = False
        for source, target in edges:
            if source in reachable and target not in reachable:
                reachable.add(target)
                changed = True
    if state_name not in reachable:
        return (
            f"{constraint.owner}.{constraint.constraint_id}: activation state "
            f"{constraint.activation_ref} is unreachable from initial"
        )
    return None


def materialize_planned_attributes(
    model_text: str,
    components: Sequence[Any],
) -> tuple[str, dict[str, Any]]:
    """Materialise attributes explicitly authorised by the typed plan."""
    text = model_text
    added: list[str] = []
    restored: list[str] = []
    missing: list[str] = []
    for component in components:
        for attribute in component.attributes:
            owner_span = next(
                (
                    span for span in _definition_spans(text)
                    if span[0] == component.name
                ),
                None,
            )
            if owner_span is None:
                missing.append(
                    f"{component.name}.{attribute.name}: owner missing"
                )
                continue
            closing = owner_span[3]
            body = text[owner_span[2] + 1:closing]
            initializer = (
                attribute.input_binding
                if attribute.input_binding
                else attribute.initial_value
            )
            if (
                initializer
                and not attribute.input_binding
                and _NUMBER.fullmatch(initializer)
                and attribute.unit not in {"", "1"}
            ):
                initializer = f"{initializer} [{attribute.unit}]"
            desired = (
                f"attribute {attribute.name} : {attribute.value_type}"
                + (f" = {initializer}" if initializer else "")
                + ";"
            )
            matches = [
                match for match in _ATTRIBUTE.finditer(body)
                if match.group("name") == attribute.name
            ]
            if not matches:
                text = (
                    text[:closing]
                    + f"\n        {desired}\n    "
                    + text[closing:]
                )
                added.append(f"{component.name}.{attribute.name}")
                continue
            match = matches[0]
            observed = _normalise_expression(match.group(0))
            if observed != _normalise_expression(desired):
                absolute_start = owner_span[2] + 1 + match.start()
                absolute_end = owner_span[2] + 1 + match.end()
                text = (
                    text[:absolute_start] + desired + text[absolute_end:]
                )
                restored.append(f"{component.name}.{attribute.name}")
    return text, {
        "schema_version": "1.0",
        "artifact_role": "PLANNED_ATTRIBUTE_MATERIALIZATION",
        "status": "PASS" if not missing else "FAIL",
        "materialized_attributes": added,
        "restored_attributes": restored,
        "missing_attributes": missing,
    }


def materialize_planned_constraints(
    model_text: str,
    constraints: Sequence[ConstraintPlan],
) -> tuple[str, dict[str, Any]]:
    """Make the activated-constraint plan the sole constraint writer."""
    planned = {
        (item.owner, item.constraint_id): item
        for item in constraints
        if item.verification_tier != "EXTERNAL_ANALYSIS"
    }
    observed = _assertions(model_text)
    unplanned = [
        item for item in observed
        if (
            (item["owner"], item["name"]) not in planned
            or not _constraint_is_placed(
                model_text,
                item,
                planned[(item["owner"], item["name"])],
            )
        )
    ]
    orphan_candidates: set[tuple[str, str]] = set()
    for item in unplanned:
        comparison = _COMPARISON.fullmatch(item["expression"])
        if comparison is None:
            continue
        orphan_candidates.add((item["owner"], comparison.group("lhs")))
        rhs = comparison.group("rhs")
        if _IDENTIFIER.fullmatch(rhs):
            orphan_candidates.add((item["owner"], rhs))
    text = model_text
    removed: list[str] = []
    for item in sorted(unplanned, key=lambda value: value["start"], reverse=True):
        start = item["line_start"]
        # Remove an immediately preceding stale plan annotation with the block.
        previous_line_start = text.rfind("\n", 0, max(0, start - 1)) + 1
        if _PLAN_COMMENT.fullmatch(text[previous_line_start:start].rstrip("\n")):
            start = previous_line_start
        text = text[:start] + text[item["end"]:]
        removed.append(f"{item['owner']}.{item['name']}")

    added: list[str] = []
    annotated: list[str] = []
    restored: list[str] = []
    missing: list[str] = []
    for key, constraint in planned.items():
        matches = [
            item for item in _assertions(text)
            if (item["owner"], item["name"]) == key
            and _constraint_is_placed(text, item, constraint)
        ]
        if matches:
            item = matches[0]
            if item["expression"] != _normalise_expression(
                constraint.expression
            ):
                opening = text.find("{", item["start"], item["end"])
                indent = text[item["line_start"]:item["start"]]
                text = (
                    text[:opening + 1]
                    + f"\n{indent}    {constraint.expression}\n{indent}"
                    + text[item["end"] - 1:]
                )
                restored.append(
                    f"{constraint.owner}.{constraint.constraint_id}"
                )
                item = next(
                    current for current in _assertions(text)
                    if (
                        current["owner"],
                        current["name"],
                    ) == key
                )
            if not item["metadata"]:
                indent = text[item["line_start"]:item["start"]]
                text = (
                    text[:item["line_start"]]
                    + _comment(constraint, indent)
                    + text[item["line_start"]:]
                )
                annotated.append(
                    f"{constraint.owner}.{constraint.constraint_id}"
                )
            continue
        if constraint.activation_kind == "STATE_ACTIVE":
            text, expanded = _ensure_activation_state_body(text, constraint)
            if expanded:
                restored.append(
                    f"{constraint.owner}.{constraint.activation_ref} state body"
                )
            state_span, state_issue = _activation_state_span(
                text, constraint
            )
            if state_issue is not None or state_span is None:
                missing.append(
                    state_issue
                    or f"{constraint.owner}.{constraint.constraint_id}: "
                    "activation state missing"
                )
                continue
            closing = state_span[1]
            indent = "            "
        else:
            owner_span = next(
                (
                    span for span in _definition_spans(text)
                    if span[0] == constraint.owner
                ),
                None,
            )
            if owner_span is None:
                missing.append(
                    f"{constraint.owner}.{constraint.constraint_id}: owner missing"
                )
                continue
            closing = owner_span[3]
            indent = "        "
        block = (
            "\n"
            + _comment(constraint, indent)
            + f"{indent}assert constraint {constraint.constraint_id} {{\n"
            + f"{indent}    {constraint.expression}\n"
            + f"{indent}}}\n"
            + indent[:-4]
        )
        text = text[:closing] + block + text[closing:]
        added.append(f"{constraint.owner}.{constraint.constraint_id}")

    # A fabricated min/current pair often exists only to support the fabricated
    # assertion.  Once that assertion is rejected, remove those now-orphaned
    # declarations as one compiler-owned cleanup.  Attributes used by a guard,
    # action, binding, or another constraint are preserved.
    removed_orphan_attributes: list[str] = []
    for owner, name in sorted(orphan_candidates):
        owner_span = next(
            (
                span for span in _definition_spans(text)
                if span[0] == owner
            ),
            None,
        )
        if owner_span is None:
            continue
        body = text[owner_span[2] + 1:owner_span[3]]
        match = next(
            (
                item for item in _ATTRIBUTE.finditer(body)
                if item.group("name") == name
            ),
            None,
        )
        if match is None:
            continue
        absolute_start = owner_span[2] + 1 + match.start()
        absolute_end = owner_span[2] + 1 + match.end()
        without_declaration = text[:absolute_start] + text[absolute_end:]
        if re.search(rf"\b{re.escape(name)}\b", without_declaration):
            continue
        line_start = text.rfind("\n", 0, absolute_start) + 1
        line_end = text.find("\n", absolute_end)
        if line_end == -1:
            line_end = absolute_end
        text = text[:line_start] + text[line_end + 1:]
        removed_orphan_attributes.append(f"{owner}.{name}")

    final_assertions = _assertions(text)
    final_keys = {
        (item["owner"], item["name"]): item for item in final_assertions
    }
    final_missing = [
        f"{item.owner}.{item.constraint_id}"
        for item in planned.values()
        if (
            (item.owner, item.constraint_id) not in final_keys
            or not _constraint_is_placed(
                text,
                final_keys[(item.owner, item.constraint_id)],
                item,
            )
        )
    ]
    activated = [
        item for item in constraints if item.activation_kind == "STATE_ACTIVE"
    ]
    activation_issues = [
        issue for issue in (
            _activation_issue(text, item) for item in activated
        )
        if issue is not None
    ]
    external = [
        item for item in constraints
        if item.verification_tier == "EXTERNAL_ANALYSIS"
    ]
    report = {
        "schema_version": "1.0",
        "artifact_role": "ACTIVATED_CONSTRAINT_CONFORMANCE",
        "status": (
            "PASS"
            if not missing and not final_missing and not activation_issues
            else "FAIL"
        ),
        "planned_count": len(constraints),
        "planned_invariant_count": sum(
            item.activation_kind == "ALWAYS"
            for item in planned.values()
        ),
        "planned_state_active_count": sum(
            item.activation_kind == "STATE_ACTIVE"
            for item in planned.values()
        ),
        "observed_unplanned_constraints": [
            f"{item['owner']}.{item['name']}" for item in unplanned
        ],
        "removed_unplanned_constraints": sorted(removed),
        "removed_orphan_constraint_attributes": (
            removed_orphan_attributes
        ),
        "materialized_constraints": added,
        "restored_constraints": restored,
        "annotated_constraints": annotated,
        "missing_constraints": list(dict.fromkeys(missing + final_missing)),
        "activation_issues": activation_issues,
        "external_verification_readiness": {
            "status": (
                "PASS"
                if external and all(
                    item.source_requirement_id or item.provenance == "DESIGN_DECISION"
                    for item in external
                )
                else "NOT_APPLICABLE" if not external else "FAIL"
            ),
            "planned": len(external),
        },
    }
    return text, report
