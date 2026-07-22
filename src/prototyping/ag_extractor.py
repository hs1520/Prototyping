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
    BehaviorRealization,
    BehaviorTransition,
    Contract,
    Guarantee,
    Span,
)
from .blackboard import text_digest
from ..utils.sysml_text_utils import find_block_end

_REQ_DEF_RE = re.compile(r"\brequirement\s+def\s+(\w+)\s*\{")
_ATTR_RE = re.compile(
    r"\battribute\s+(\w+)\s*:\s*(\w+)\s*(?:=\s*(-?[\d.]+)"
    r"\s*(?:\[([^\]]+)\])?)?\s*;"
)
_ASSUME_RE = re.compile(r"\bassume\s+constraint\s+(\w+)?\s*\{([^{}]*)\}")
_REQUIRE_RE = re.compile(r"\brequire\s+constraint\s+(\w+)?\s*\{([^{}]*)\}")
_DEP_RE = re.compile(
    r"\bdependency\s+(\w+)\s+from\s+(\w+)\s+to\s+(\w+)\s*;"
)
_SATISFY_RE = re.compile(
    r"\bsatisfy\s+requirement\s+\w+\s*:\s*(\w+)\s+by\s+(\w+)\s*;"
)
_STATE_DEF_RE = re.compile(r"\bstate\s+def\s+(\w+)\s*\{")
_TRANSITION_RE = re.compile(
    r"\btransition\s+\w+\s+first\s+(\w+)\s+accept\s+(\w+)\s+then\s+(\w+)\s*;",
    re.DOTALL,
)
_VERIFICATION_DEF_RE = re.compile(r"\bverification\s+def\s+(\w+)\s*\{")
_VERIFY_REQ_RE = re.compile(
    r"\bverify\s+requirement\s+\w+\s*:\s*(\w+)\s*;"
)
_SOURCE_REQ_RE = re.compile(r"bounded\s+A/G\s+system\s+contract\s+for\s+(REQ[_-]\w+)", re.I)
_SAFETY_PATTERN_RE = re.compile(r"safety_pattern\s*=\s*(\w+)", re.I)
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
    attr_units: Dict[str, Optional[str]] = {}
    for m in _ATTR_RE.finditer(block):
        attr_name = m.group(1)
        default = m.group(3)
        attrs[attr_name.lower()] = float(default) if default is not None else None
        attr_units[attr_name.lower()] = m.group(4)

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
    timing_unit: Optional[str] = None
    for key in _COMPONENT_BUDGET_KEYS + _SYSTEM_BUDGET_KEYS:
        if attrs.get(key) is not None:
            timing_budget = attrs[key]
            timing_unit = attr_units.get(key)
            break

    source = _SOURCE_REQ_RE.search(block)
    pattern = _SAFETY_PATTERN_RE.search(block)

    return Contract(
        name=name,
        role="component",  # provisional; fixed once edges are known
        assumptions=tuple(assumptions),
        guarantees=tuple(guarantees),
        timing_budget=timing_budget,
        timing_unit=timing_unit,
        observation=None,  # set for the system contract only
        element_id=name,
        span=span,
        source_requirement=source.group(1).upper().replace("-", "_") if source else None,
        declared_pattern=pattern.group(1).upper() if pattern else None,
    )


def _parse_behavior(name: str, block: str, span: Span) -> BehaviorRealization:
    initial = re.search(r"\bentry\s*;\s*then\s+(\w+)\s*;", block)
    transitions = tuple(
        BehaviorTransition(m.group(1), m.group(2), m.group(3))
        for m in _TRANSITION_RE.finditer(block)
    )
    entry_actions: Dict[str, str] = {}
    for state in re.finditer(r"\bstate\s+(\w+)\s*\{", block):
        brace = block.find("{", state.start())
        end = find_block_end(block, brace)
        if end == -1:
            continue
        action = re.search(r"\bentry\s+action\s+(\w+)", block[brace + 1:end])
        if action:
            entry_actions[state.group(1)] = action.group(1)
    return BehaviorRealization(
        name=name,
        initial_state=initial.group(1) if initial else None,
        transitions=transitions,
        entry_actions=entry_actions,
        element_id=name,
        span=span,
    )


_PACKAGE_RE = re.compile(r"\bpackage\s+(\w+)\s*\{")
# The emitter stamps this exact marker on every A/G system contract (see
# ``ag_emitter.emit_ag_package``); it is the reliable signal that a top-level
# package carries a reviewed A/G chain rather than base model content.
_AG_PACKAGE_MARKER = "bounded A/G system contract for"


def _top_level_packages(text: str) -> List[Tuple[str, int, int]]:
    """Return (name, start, end_exclusive) for each top-level package block."""
    result: List[Tuple[str, int, int]] = []
    consumed_to = 0
    for m in _PACKAGE_RE.finditer(text):
        if m.start() < consumed_to:  # nested inside an already-consumed package
            continue
        brace = text.index("{", m.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        result.append((m.group(1), m.start(), end + 1))
        consumed_to = end + 1
    return result


def extract_ag_graphs(
    sysml_text: str,
    *,
    revision: Optional[int] = None,
    model_digest: Optional[str] = None,
) -> List[AGGraph]:
    """Extract one bounded A/G graph per reviewed chain in the committed model.

    A model may carry several independent A/G chains (one reviewed decomposition
    per selected requirement — the drone system co-selects REQ_SAFE_004 and
    REQ_SAFE_005). Each chain is emitted as its own top-level package with a
    single system contract, so each is a self-contained A/G decomposition that
    must be checked independently: pooling two system contracts into one graph
    would make the decomposition root ambiguous (``system=None``).

    With zero or one A/G package this returns exactly ``[extract_ag_graph(...)]``
    — byte-identical to the single-chain path. With two or more, the base model
    (which carries the immutable source ``requirement def`` provenance) is paired
    with each A/G package in turn so every per-chain graph both resolves its
    system contract uniquely and keeps its source-requirement provenance. Every
    per-chain graph reports the committed model's revision/digest, not the slice's.
    """
    text = sysml_text or ""
    digest = model_digest if model_digest is not None else text_digest(text)
    ag_spans = [
        (start, end)
        for (_name, start, end) in _top_level_packages(text)
        if _AG_PACKAGE_MARKER in text[start:end]
    ]
    if len(ag_spans) <= 1:
        return [extract_ag_graph(text, revision=revision, model_digest=digest)]

    # Base model = everything that is not an A/G package (source requirement defs
    # live here); it is prepended to each A/G package so provenance resolves.
    others_parts: List[str] = []
    cursor = 0
    for start, end in sorted(ag_spans):
        others_parts.append(text[cursor:start])
        cursor = end
    others_parts.append(text[cursor:])
    base = "".join(others_parts).rstrip()

    graphs: List[AGGraph] = []
    for start, end in sorted(ag_spans):
        slice_text = base + "\n\n" + text[start:end] + "\n"
        graphs.append(
            extract_ag_graph(slice_text, revision=revision, model_digest=digest)
        )
    return graphs


def extract_ag_graph(
    sysml_text: str,
    *,
    revision: Optional[int] = None,
    model_digest: Optional[str] = None,
) -> AGGraph:
    """Parse committed SysML v2 text into a bounded A/G graph (§6.2).

    This resolves a single system contract. For a model that may carry more than
    one reviewed chain, use :func:`extract_ag_graphs`, which returns one graph
    per chain and degrades to ``[this]`` when only one chain is present.
    """
    text = sysml_text or ""
    digest = model_digest if model_digest is not None else text_digest(text)
    parse_diags: List[AGDiagnostic] = []

    raw: Dict[str, Contract] = {}
    all_requirement_ids: List[str] = []
    for m in _REQ_DEF_RE.finditer(text):
        name = m.group(1)
        all_requirement_ids.append(name.upper().replace("-", "_"))
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

    owners: Dict[str, List[str]] = {}
    for match in _SATISFY_RE.finditer(text):
        owners.setdefault(match.group(1), []).append(match.group(2))

    behaviors: List[BehaviorRealization] = []
    for match in _STATE_DEF_RE.finditer(text):
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        if end != -1:
            behaviors.append(_parse_behavior(
                match.group(1), text[brace + 1:end], Span(brace + 1, end)
            ))

    verification_targets: Dict[str, Tuple[str, ...]] = {}
    for match in _VERIFICATION_DEF_RE.finditer(text):
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        if end != -1:
            verification_targets[match.group(1)] = tuple(
                item.group(1) for item in _VERIFY_REQ_RE.finditer(text[brace + 1:end])
            )

    edges: List[AGEdge] = []
    for m in _DEP_RE.finditer(text):
        raw_kind, src, dst = m.group(1), m.group(2), m.group(3)
        kind = raw_kind.lower()
        subject = None
        if kind.startswith("decompos"):
            edge_kind = "decomposes"
        elif kind.startswith("discharge"):
            edge_kind = "discharges"
            subject = raw_kind[len("discharge"):]
        elif kind.startswith("realize"):
            edge_kind = "realized_by"
        elif kind.startswith("observe"):
            edge_kind = "observed_by"
        else:
            edge_kind = kind
        edges.append(AGEdge(kind=edge_kind, src=src, dst=dst, subject=subject))

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
                owners=tuple(owners.get(name, ())),
                source_requirement=contract.source_requirement,
                declared_pattern=contract.declared_pattern,
            )
        else:
            components.append(Contract(
                name=contract.name, role=contract.role,
                assumptions=contract.assumptions, guarantees=contract.guarantees,
                timing_budget=contract.timing_budget, timing_unit=contract.timing_unit,
                observation=contract.observation, element_id=contract.element_id,
                span=contract.span, owners=tuple(owners.get(name, ())),
                source_requirement=contract.source_requirement,
            ))

    return AGGraph(
        system=system,
        components=tuple(components),
        edges=tuple(edges),
        revision=revision,
        model_digest=digest,
        parse_diagnostics=tuple(parse_diags),
        behaviors=tuple(behaviors),
        verification_targets=verification_targets,
        source_requirement_ids=tuple(dict.fromkeys(all_requirement_ids)),
    )
