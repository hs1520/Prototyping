"""Bounded, source-derived semantic anchors for whole-model generation.

This is not a temporal proof system.  It extracts only explicit numeric
invariants stated by the stakeholder (for example, "maintain at least 5 metres
of separation").  The resulting anchor preserves comparator, threshold, unit,
and subject terms across generation and remains digest-bound to its source.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Sequence

from ..utils.sysml_text_utils import find_block_end
from ..utils.req_id import normalise_req_id


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


def _extract_assertions(block: str) -> list[tuple[str, str]]:
    assertions: list[tuple[str, str]] = []
    for match in _ASSERT_RE.finditer(block):
        opening = block.find("{", match.start(), match.end())
        closing = find_block_end(block, opening)
        if closing != -1:
            assertions.append((
                match.group("name"),
                block[opening + 1:closing].strip(),
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
            "claim_boundary": "MODEL_SEMANTIC_FIDELITY_NOT_PHYSICAL_PROOF",
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any]
    ) -> "RequirementSemanticObligation":
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
        )


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
            obligations.append(RequirementSemanticObligation(
                obligation_id=f"SEM_{requirement_id}_{index:03d}",
                requirement_id=requirement_id,
                kind=NUMERIC_INVARIANT,
                subject_terms=terms,
                operator=operator,
                threshold=float(match.group("value")),
                unit=_UNIT_CANONICAL[raw_unit],
                source_clause=" ".join(match.group(0).split()),
                source_digest=_source_digest(source),
            ))
    return tuple(obligations)


def validate_requirement_semantic_obligations(
    model_text: str,
    obligations: Sequence[RequirementSemanticObligation],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Validate frozen numeric semantics against the terminal SysML model.

    Passing means the model preserves a source-bound quantity and its explicit
    constraint.  It does not establish environmental validity, controller
    adequacy, or physical safety.
    """
    text = str(model_text or "")
    parts = _extract_part_definitions(text)
    results: list[dict[str, Any]] = []

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

            for assertion_name, expression in _extract_assertions(block):
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
                owner_evidence.append({
                    "assertion": assertion_name,
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
    if not obligations:
        status = "UNVERIFIED"
    else:
        status = "PASS" if passed == len(results) else "FAIL"
    return {
        "schema_version": "1.0",
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
    }
