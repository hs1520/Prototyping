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
