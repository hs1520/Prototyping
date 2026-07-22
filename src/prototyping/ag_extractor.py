"""Extract a bounded Assume-Guarantee graph from committed SysML v2 text.

This is the extractor half of Increment 2 (§6.2, §15 of
``docs/OPTION2_IMPLEMENTATION_DESIGN.md``). The committed SysML model is the sole
semantic authority: A/G facts are read *out of* the model using the validated
bounded convention and never supplied from JSON. The output :class:`AGGraph`
carries the source revision/digest and per-element spans so the derived
``ag_contract_graph.json`` view (§14) is fully attributable and regenerable.

Bounded convention (validated against the Syside gate — see
``tests/test_option2_ag_sysml_spike.py`` and ``test_option2_ag_checker.py``):

    requirement def <ContractName> {
        attribute <var> : <Type> [= <default>];
        attribute latencyBudget : Real = 0.10;   // component timing budget
        attribute maxLatency    : Real = 0.5;     // system deadline
        assume  constraint [env_]<name> { <boolean-id | var <cmp> rhs> }
        require constraint <name>        { <boolean-id | var <cmp> rhs> }
    }
    dependency decomposition from <System> to <Component>;

An ``assume constraint`` whose name begins with ``env`` is an explicit
environment assumption. The decomposition edge source is the system contract; the
targets are components.
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple

from .ag_contracts import (
    AGDiagnostic,
    AGEdge,
    AGGraph,
    Assumption,
    Contract,
    Guarantee,
    Span,
)
from .blackboard import text_digest
from ..utils.sysml_text_utils import find_block_end

_REQ_DEF_RE = re.compile(r"\brequirement\s+def\s+(\w+)\s*\{")
_ATTR_RE = re.compile(
    r"\battribute\s+(\w+)\s*:\s*(\w+)\s*(?:=\s*(-?[\d.]+))?\s*;"
)
_ASSUME_RE = re.compile(r"\bassume\s+constraint\s+(\w+)?\s*\{([^{}]*)\}")
_REQUIRE_RE = re.compile(r"\brequire\s+constraint\s+(\w+)?\s*\{([^{}]*)\}")
_DEP_RE = re.compile(
    r"\bdependency\s+(\w+)\s+from\s+(\w+)\s+to\s+(\w+)\s*;"
)
_CMP_RE = re.compile(r"^(\w+)\s*(<=|>=|==|<|>)\s*([A-Za-z_][\w]*|-?[\d.]+)$")
_IDENT_RE = re.compile(r"^(\w+)$")

# Attribute names that carry a timing budget/deadline.
_COMPONENT_BUDGET_KEYS = ("latencybudget",)
_SYSTEM_BUDGET_KEYS = ("maxlatency", "deadline", "systemdeadline")


def _parse_expr(
    expr: str, attrs: Mapping[str, Optional[float]]
) -> Tuple[str, str, Dict[str, object]]:
    """Return (concept, kind, extra) for one constraint body.

    kind ∈ {"boolean", "numeric", "unsupported"}. Numeric right-hand sides that
    name a declared attribute are resolved to that attribute's default value.
    """
    body = " ".join((expr or "").split())
    m = _CMP_RE.match(body)
    if m:
        var, cmp_op, rhs = m.group(1), m.group(2), m.group(3)
        try:
            value: Optional[float] = float(rhs)
        except ValueError:
            value = attrs.get(rhs.lower())
        return var, "numeric", {"variable": var, "comparator": cmp_op, "value": value}
    m = _IDENT_RE.match(body)
    if m:
        return m.group(1), "boolean", {}
    return body, "unsupported", {}


def _parse_contract(name: str, block: str, span: Span) -> Contract:
    attrs: Dict[str, Optional[float]] = {}
    for m in _ATTR_RE.finditer(block):
        attr_name = m.group(1)
        default = m.group(3)
        attrs[attr_name.lower()] = float(default) if default is not None else None

    assumptions: List[Assumption] = []
    for m in _ASSUME_RE.finditer(block):
        cname, expr = m.group(1), m.group(2)
        concept, kind, extra = _parse_expr(expr, attrs)
        is_env = bool(cname and cname.lower().startswith("env"))
        assumptions.append(Assumption(
            concept=concept, expr=" ".join(expr.split()), kind=kind,
            is_environment=is_env, constraint_name=cname,
            variable=extra.get("variable"), comparator=extra.get("comparator"),
            value=extra.get("value"),
        ))

    guarantees: List[Guarantee] = []
    for m in _REQUIRE_RE.finditer(block):
        cname, expr = m.group(1), m.group(2)
        concept, kind, extra = _parse_expr(expr, attrs)
        guarantees.append(Guarantee(
            concept=concept, expr=" ".join(expr.split()), kind=kind,
            constraint_name=cname, variable=extra.get("variable"),
            comparator=extra.get("comparator"), value=extra.get("value"),
        ))

    timing_budget: Optional[float] = None
    for key in _COMPONENT_BUDGET_KEYS + _SYSTEM_BUDGET_KEYS:
        if attrs.get(key) is not None:
            timing_budget = attrs[key]
            break

    return Contract(
        name=name,
        role="component",  # provisional; fixed once edges are known
        assumptions=tuple(assumptions),
        guarantees=tuple(guarantees),
        timing_budget=timing_budget,
        observation=None,  # set for the system contract only
        element_id=name,
        span=span,
    )


def extract_ag_graph(
    sysml_text: str,
    *,
    revision: Optional[int] = None,
    model_digest: Optional[str] = None,
) -> AGGraph:
    """Parse committed SysML v2 text into a bounded A/G graph (§6.2)."""
    text = sysml_text or ""
    digest = model_digest if model_digest is not None else text_digest(text)
    parse_diags: List[AGDiagnostic] = []

    raw: Dict[str, Contract] = {}
    for m in _REQ_DEF_RE.finditer(text):
        name = m.group(1)
        brace = text.index("{", m.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        block = text[brace + 1:end]
        contract = _parse_contract(name, block, Span(brace + 1, end))
        # Only a requirement def that declares assume/require constraints is an A/G
        # contract (§6.2). Ordinary stakeholder requirement defs imported into the
        # model carry no A/G semantics and must not pollute the graph.
        if contract.assumptions or contract.guarantees:
            raw[name] = contract

    edges: List[AGEdge] = []
    for m in _DEP_RE.finditer(text):
        kind, src, dst = m.group(1).lower(), m.group(2), m.group(3)
        edge_kind = "decomposes" if kind.startswith("decompos") else kind
        edges.append(AGEdge(kind=edge_kind, src=src, dst=dst))

    sources = {e.src for e in edges if e.kind == "decomposes"}
    targets = {e.dst for e in edges if e.kind == "decomposes"}
    system_names = sources - targets

    system_name: Optional[str] = None
    if len(system_names) == 1:
        system_name = next(iter(system_names))
    elif len(raw) == 1:
        system_name = next(iter(raw))
    else:
        # Ambiguous: fall back to a name-based heuristic and record it.
        candidates = [n for n in raw if "system" in n.lower()]
        if len(candidates) == 1:
            system_name = candidates[0]
        parse_diags.append(AGDiagnostic(
            "CONTRACT_INCOMPLETE",
            ("system contract could not be uniquely identified from decomposition "
             f"edges (sources={sorted(system_names)})"),
            severity="warning",
        ))

    system: Optional[Contract] = None
    components: List[Contract] = []
    for name, contract in raw.items():
        if name == system_name:
            # The system observation is its first Boolean guarantee concept.
            obs = next(
                (g.concept for g in contract.guarantees if g.kind == "boolean"),
                None,
            )
            system = Contract(
                name=contract.name, role="system",
                assumptions=contract.assumptions, guarantees=contract.guarantees,
                timing_budget=contract.timing_budget, timing_unit=contract.timing_unit,
                observation=obs, element_id=contract.element_id, span=contract.span,
            )
        else:
            components.append(contract)

    return AGGraph(
        system=system,
        components=tuple(components),
        edges=tuple(edges),
        revision=revision,
        model_digest=digest,
        parse_diagnostics=tuple(parse_diags),
    )
