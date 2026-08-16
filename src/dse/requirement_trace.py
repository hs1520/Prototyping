"""One requirement-definition and satisfy-allocation index for DSE analyses."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..utils.req_id import first_req_id, normalise_req_id
from ..utils.sysml_text_utils import find_block_end


_REQUIREMENT_DEF_RE = re.compile(
    r"\brequirement\s+def\s+(REQ[_-]\w+)", re.IGNORECASE
)
_SATISFY_RE = re.compile(
    r"\bsatisfy\s+(?:requirement\s+)?(\w*REQ[_-]\w+)", re.IGNORECASE
)
_PART_DEF_RE = re.compile(
    r"\bpart\s+def\s+(\w+)\s*(?::>[^\{]*)?\{", re.IGNORECASE
)


def dse_req_id(value: str) -> str:
    """Canonical DSE display form (upper-case with hyphen separators)."""
    return normalise_req_id(first_req_id(value) or value).replace("_", "-")


@dataclass(frozen=True)
class RequirementTraceIndex:
    declared: tuple[str, ...]
    satisfied: frozenset[str]
    owners: Mapping[str, frozenset[str]]
    source_by_id: Mapping[str, str]


def extract_requirement_trace(
    model_text: str,
    requirements: Sequence[str] = (),
) -> RequirementTraceIndex:
    """Index declarations, allocations, owners and immutable source text once."""
    text = model_text or ""
    declared = tuple(
        dse_req_id(match.group(1)) for match in _REQUIREMENT_DEF_RE.finditer(text)
    )
    satisfied = frozenset(
        dse_req_id(match.group(1)) for match in _SATISFY_RE.finditer(text)
    )
    mutable_owners: dict[str, set[str]] = {}
    for part in _PART_DEF_RE.finditer(text):
        opening = text.index("{", part.start())
        closing = find_block_end(text, opening)
        body = text[opening + 1:closing] if closing != -1 else ""
        for allocation in _SATISFY_RE.finditer(body):
            mutable_owners.setdefault(
                dse_req_id(allocation.group(1)), set()
            ).add(part.group(1))
    sources = {
        dse_req_id(source): str(source)
        for source in requirements
        if first_req_id(str(source)) is not None
    }
    return RequirementTraceIndex(
        declared=declared,
        satisfied=satisfied,
        owners={
            req_id: frozenset(parts)
            for req_id, parts in mutable_owners.items()
        },
        source_by_id=sources,
    )
