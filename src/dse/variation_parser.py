"""Parse LLM-declared SysML v2 variation points into a searchable design space.

The "replace" path (user-chosen): the explored space comes from the
``variation``/``variant`` points the LLM declares in the generated model, not from
the fixed operator catalog. A point is rejected unless it carries an objectivity
rationale - a ``doc`` comment with a rationale and at least one linked
requirement - so the variant space stays traceable.

  parse_variation_points(text)   -> [VariationPoint]
  admitted(points)               -> (admitted, rejected) by the objectivity rule
  resolve_model(text, points, choices) -> concrete SysML (each point bound)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..utils.sysml_text_utils import find_block_end

_VARIATION_RE = re.compile(r"\bvariation\s+(part|item|attribute)\s+(\w+)\s*(?::[^\{]*)?\{")
_VARIANT_RE = re.compile(r"\bvariant\s+(?:part|item|attribute)\s+(\w+)\s*(?::\s*(\w+))?")
_DOC_RE = re.compile(r"doc\s*/\*(.*?)\*/", re.DOTALL)
_REQ_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")


@dataclass
class VariationPoint:
    point_id: str
    kind: str
    variants: List[Tuple[str, str]]
    rationale: str = ""
    requirements: List[str] = field(default_factory=list)
    span: Tuple[int, int] = (0, 0)

    @property
    def variant_names(self) -> List[str]:
        return [n for n, _ in self.variants]

    def type_of(self, variant_name: str) -> str:
        for n, t in self.variants:
            if n == variant_name:
                return t
        raise KeyError(variant_name)

    def is_objective(self) -> bool:
        """Admissible iff it has a rationale AND links >= 1 requirement."""
        return bool(self.rationale.strip()) and bool(self.requirements)


def parse_variation_points(sysml_text: str) -> List[VariationPoint]:
    points: List[VariationPoint] = []
    for m in _VARIATION_RE.finditer(sysml_text):
        kind, name = m.group(1), m.group(2)
        brace = sysml_text.index("{", m.start())
        end = find_block_end(sysml_text, brace)
        if end == -1:
            continue
        body = sysml_text[brace + 1 : end]
        variants = [(vn, vt or "") for vn, vt in _VARIANT_RE.findall(body)]
        if not variants:
            continue
        doc_m = _DOC_RE.search(body)
        doc = doc_m.group(1) if doc_m else ""
        reqs = _REQ_RE.findall(doc)
        points.append(
            VariationPoint(
                point_id=name,
                kind=kind,
                variants=variants,
                rationale=doc.strip(),
                requirements=reqs,
                span=(m.start(), end + 1),
            )
        )
    return points


def admitted(points: List[VariationPoint]) -> Tuple[List[VariationPoint], List[VariationPoint]]:
    """Split into (admitted, rejected) by the objectivity rule (rationale + requirement)."""
    ok = [p for p in points if p.is_objective()]
    bad = [p for p in points if not p.is_objective()]
    return ok, bad


def _variant_base(model_text: str, type_name: str) -> str:
    m = re.search(rf"\bpart\s+def\s+{re.escape(type_name)}\s*:>\s*(\w+)", model_text)
    return m.group(1) if m else ""


def port_safe(vp: VariationPoint, model_text: str) -> bool:
    """True iff every variant type specialises the same interface part def.

    All variants then share that interface's ports, so binding any variant keeps the
    host's connects valid (they target ports present in every variant).
    """
    bases = {_variant_base(model_text, t) for _, t in vp.variants if t}
    return len(bases) == 1 and "" not in bases


def port_safe_split(
    points: List[VariationPoint], model_text: str
) -> Tuple[List[VariationPoint], List[VariationPoint]]:
    """Split admitted points into (resolve-safe, unsafe) by shared port interface."""
    safe = [p for p in points if port_safe(p, model_text)]
    unsafe = [p for p in points if not port_safe(p, model_text)]
    return safe, unsafe


def resolve_model(sysml_text: str, points: List[VariationPoint], choices: Dict[str, str]) -> str:
    """Return concrete SysML with each variation point bound to its chosen variant.

    Replaces ``variation <kind> X { ... }`` with ``<kind> X : <chosen variant type>;``
    Replacements are applied back-to-front so spans stay valid.
    """
    result = sysml_text
    for p in sorted(points, key=lambda q: q.span[0], reverse=True):
        if p.point_id not in choices:
            continue
        variant = choices[p.point_id]
        vtype = p.type_of(variant)
        if not vtype:
            continue
        replacement = f"{p.kind} {p.point_id} : {vtype};"
        s, e = p.span
        result = result[:s] + replacement + result[e:]
    return result
