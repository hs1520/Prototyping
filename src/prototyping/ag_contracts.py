"""Bounded Assume-Guarantee contract model and compositional checker (R2-BBAG).

This implements the checker half of Increment 2 in
``docs/OPTION2_IMPLEMENTATION_DESIGN.md`` (§6, §9). The data structures mirror
§6.1 (SystemContract / ComponentContract + decomposes/discharged_by/… edges) and
the checker runs the bounded §6.4 / §9 obligations:

  * per-contract completeness (READY / INCOMPLETE / UNSUPPORTED, §6.3);
  * one responsible owner per component guarantee (§16);
  * assumption discharge by explicit environment or upstream guarantee (§6.4),
    computed as a monotone availability fixpoint — the bounded A/G composition;
  * unit-safe numeric / Boolean compatibility (§6.4);
  * additive timing-budget composition (§6.4, §7);
  * whether the component guarantees collectively support the system guarantee;
  * circular-assumption detection (§17 unsoundness risk).

Boundary (§2, §13): this is bounded, A/G-*inspired* compositional checking, not a
sound formal proof calculus. A model-level PASS means the extracted graph is
complete and internally compatible — never that the physical system passed.

The checker consumes an already-extracted :class:`AGGraph` (see
``ag_extractor.py``); the SysML model remains the sole semantic authority and a
fact absent from the committed model is INCOMPLETE/UNSUPPORTED here, never
supplied from JSON (§6.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

AG_CHECKER_VERSION = "ag-mvp-1"

# Completeness states (§6.3).
READY = "READY"
INCOMPLETE = "INCOMPLETE"
UNSUPPORTED = "UNSUPPORTED"

# Diagnostic codes. Kept aligned with the failure-routing table (§10) so a
# downstream router can map them without re-deriving intent.
CODE_CONTRACT_INCOMPLETE = "CONTRACT_INCOMPLETE"
CODE_CONTRACT_UNSUPPORTED = "CONTRACT_UNSUPPORTED"
CODE_GUARANTEE_NO_OWNER = "GUARANTEE_NO_OWNER"
CODE_GUARANTEE_MULTIPLE_OWNERS = "GUARANTEE_MULTIPLE_OWNERS"
CODE_ASSUMPTION_UNDISCHARGED = "ASSUMPTION_UNDISCHARGED"
CODE_CIRCULAR_ASSUMPTION = "CIRCULAR_ASSUMPTION"
CODE_UNIT_INCOMPATIBLE = "UNIT_INCOMPATIBLE"
CODE_TIMING_BUDGET_EXCEEDED = "TIMING_BUDGET_EXCEEDED"
CODE_DECOMPOSITION_INSUFFICIENT = "DECOMPOSITION_INSUFFICIENT"

_ERROR_CODES = frozenset({
    CODE_GUARANTEE_NO_OWNER,
    CODE_GUARANTEE_MULTIPLE_OWNERS,
    CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION,
    CODE_UNIT_INCOMPATIBLE,
    CODE_TIMING_BUDGET_EXCEEDED,
    CODE_DECOMPOSITION_INSUFFICIENT,
})


def _norm(concept: str, aliases: Mapping[str, str]) -> str:
    """Canonical concept token: lower-cased, alias-resolved (§6.4 declared aliases)."""
    key = (concept or "").strip().lower()
    return aliases.get(key, key)


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def as_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class Assumption:
    concept: str
    expr: str
    kind: str  # "boolean" | "numeric" | "timing" | "unsupported"
    is_environment: bool = False
    variable: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    constraint_name: Optional[str] = None


@dataclass(frozen=True)
class Guarantee:
    concept: str
    expr: str
    kind: str  # "boolean" | "numeric" | "timing" | "unsupported"
    variable: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    constraint_name: Optional[str] = None


@dataclass(frozen=True)
class Contract:
    name: str
    role: str  # "system" | "component"
    assumptions: Tuple[Assumption, ...] = ()
    guarantees: Tuple[Guarantee, ...] = ()
    timing_budget: Optional[float] = None  # component latency budget / system deadline
    timing_unit: Optional[str] = None
    observation: Optional[str] = None  # system-level observed signal concept
    element_id: Optional[str] = None
    span: Optional[Span] = None

    def boolean_guarantee_concepts(self) -> Tuple[str, ...]:
        return tuple(g.concept for g in self.guarantees if g.kind == "boolean")


@dataclass(frozen=True)
class AGEdge:
    kind: str  # "decomposes" (§6.1); others reserved for later increments
    src: str
    dst: str


@dataclass(frozen=True)
class AGDiagnostic:
    code: str
    message: str
    contract: Optional[str] = None
    subject: Optional[str] = None  # assumption/guarantee concept
    severity: str = "error"  # "error" | "warning"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "contract": self.contract,
            "subject": self.subject,
            "severity": self.severity,
            "provenance": self.provenance,
        }


@dataclass
class AGGraph:
    system: Optional[Contract]
    components: Tuple[Contract, ...] = ()
    edges: Tuple[AGEdge, ...] = ()
    revision: Optional[int] = None
    model_digest: Optional[str] = None
    parse_diagnostics: Tuple[AGDiagnostic, ...] = ()

    def all_contracts(self) -> Tuple[Contract, ...]:
        return ((self.system,) if self.system else ()) + tuple(self.components)


@dataclass
class AGReport:
    verdict: str  # "PASS" | "FAIL" | "INCOMPLETE"
    system_completeness: str
    component_completeness: Dict[str, str]
    diagnostics: Tuple[AGDiagnostic, ...]
    timing: Dict[str, Any]
    discharge: Dict[str, str]
    revision: Optional[int]
    model_digest: Optional[str]
    checker_version: str = AG_CHECKER_VERSION

    def errors(self) -> Tuple[AGDiagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    def to_dict(self) -> Dict[str, Any]:
        # Shape of the derived, read-only ``ag_contract_graph.json`` audit view
        # (§14): every derived report cites the source model revision and digest
        # and the checker version, and can be regenerated deterministically.
        return {
            "artifact_role": "POSTHOC_A_G_TRACE",
            "checker_version": self.checker_version,
            "source_model_revision": self.revision,
            "source_model_digest": self.model_digest,
            "verdict": self.verdict,
            "system_completeness": self.system_completeness,
            "component_completeness": dict(self.component_completeness),
            "timing": dict(self.timing),
            "discharge": dict(self.discharge),
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }


def _classify_completeness(
    contract: Contract, *, owner_count: Optional[int]
) -> Tuple[str, List[AGDiagnostic]]:
    """READY / INCOMPLETE / UNSUPPORTED for one contract (§6.3)."""
    diags: List[AGDiagnostic] = []
    prov = {"element_id": contract.element_id,
            "span": contract.span.as_dict() if contract.span else None}

    # UNSUPPORTED: the bounded checker cannot represent a required property.
    if any(g.kind == "unsupported" for g in contract.guarantees) or \
       any(a.kind == "unsupported" for a in contract.assumptions):
        diags.append(AGDiagnostic(
            CODE_CONTRACT_UNSUPPORTED,
            f"{contract.name} uses a constraint the bounded checker cannot represent",
            contract=contract.name, severity="warning", provenance=prov,
        ))
        return UNSUPPORTED, diags

    # INCOMPLETE: a required semantic field or discharge link is missing.
    reasons: List[str] = []
    if not contract.guarantees:
        reasons.append("no guarantee (require constraint)")
    if not contract.assumptions:
        reasons.append("no assumption (assume constraint)")
    if contract.role == "component" and owner_count == 0:
        reasons.append("no responsible owner (decomposition edge)")
    if contract.role == "system" and contract.observation is None:
        reasons.append("no system observation concept")
    if reasons:
        diags.append(AGDiagnostic(
            CODE_CONTRACT_INCOMPLETE,
            f"{contract.name} is INCOMPLETE: {'; '.join(reasons)}",
            contract=contract.name, severity="warning", provenance=prov,
        ))
        return INCOMPLETE, diags

    return READY, diags


def _check_ownership(graph: AGGraph) -> Tuple[Dict[str, int], List[AGDiagnostic]]:
    """Every component guarantee has exactly one responsible owner (§16)."""
    diags: List[AGDiagnostic] = []
    owner_count: Dict[str, int] = {c.name: 0 for c in graph.components}
    for edge in graph.edges:
        if edge.kind == "decomposes" and edge.dst in owner_count:
            owner_count[edge.dst] += 1
    for comp in graph.components:
        n = owner_count[comp.name]
        prov = {"element_id": comp.element_id,
                "span": comp.span.as_dict() if comp.span else None}
        if n == 0:
            diags.append(AGDiagnostic(
                CODE_GUARANTEE_NO_OWNER,
                f"{comp.name} has no decomposition owner; its guarantee is unallocated",
                contract=comp.name, provenance=prov,
            ))
        elif n > 1:
            diags.append(AGDiagnostic(
                CODE_GUARANTEE_MULTIPLE_OWNERS,
                f"{comp.name} is allocated by {n} decomposition edges; ownership must be unique",
                contract=comp.name, provenance=prov,
            ))
    return owner_count, diags


def _check_discharge(
    graph: AGGraph, aliases: Mapping[str, str]
) -> Tuple[Dict[str, str], List[AGDiagnostic]]:
    """Assumption discharge as a monotone availability fixpoint (§6.4).

    A component activates once every non-environment Boolean assumption concept is
    available; activation publishes its Boolean guarantee concepts. Seeded by the
    system assumptions (environment/trigger) plus any environment-marked component
    assumptions. This is the bounded A/G composition: a concept is either an
    explicit environment assumption or discharged by an upstream guarantee.
    """
    diags: List[AGDiagnostic] = []
    discharge: Dict[str, str] = {}

    available: set = set()
    if graph.system:
        for a in graph.system.assumptions:
            if a.kind == "boolean":
                available.add(_norm(a.concept, aliases))
    for comp in graph.components:
        for a in comp.assumptions:
            if a.is_environment and a.kind == "boolean":
                available.add(_norm(a.concept, aliases))

    activated: set = set()
    changed = True
    while changed:
        changed = False
        for comp in graph.components:
            if comp.name in activated:
                continue
            needed = {
                _norm(a.concept, aliases)
                for a in comp.assumptions
                if a.kind == "boolean" and not a.is_environment
            }
            if needed <= available:
                activated.add(comp.name)
                for concept in comp.boolean_guarantee_concepts():
                    available.add(_norm(concept, aliases))
                changed = True

    # Distinguish a genuine cycle from a cascade behind an upstream gap (§16, §17).
    # producer[c] = component whose Boolean guarantee publishes concept c; C depends
    # on producer[c] for each non-environment Boolean assumption c. An assumption is
    # CIRCULAR only when its producer can transitively reach C back through the
    # dependency graph (a real cycle); otherwise it is a plain undischarged gap.
    producer: Dict[str, str] = {}
    for comp in graph.components:
        for concept in comp.boolean_guarantee_concepts():
            producer.setdefault(_norm(concept, aliases), comp.name)
    dep: Dict[str, set] = {c.name: set() for c in graph.components}
    for comp in graph.components:
        for a in comp.assumptions:
            if a.kind != "boolean" or a.is_environment:
                continue
            prod = producer.get(_norm(a.concept, aliases))
            if prod is not None and prod != comp.name:
                dep[comp.name].add(prod)

    def _reaches(start: str, target: str) -> bool:
        seen: set = set()
        stack = list(dep.get(start, ()))
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(dep.get(node, ()))
        return False

    inactive = [c for c in graph.components if c.name not in activated]
    for comp in inactive:
        prov = {"element_id": comp.element_id,
                "span": comp.span.as_dict() if comp.span else None}
        for a in comp.assumptions:
            if a.kind != "boolean" or a.is_environment:
                continue
            concept = _norm(a.concept, aliases)
            if concept in available:
                continue
            prod = producer.get(concept)
            circular = prod is not None and _reaches(prod, comp.name)
            code = CODE_CIRCULAR_ASSUMPTION if circular else CODE_ASSUMPTION_UNDISCHARGED
            discharge[f"{comp.name}.{a.concept}"] = (
                "circular" if circular else "undischarged"
            )
            diags.append(AGDiagnostic(
                code,
                (f"{comp.name} assumption '{a.concept}' is part of a circular "
                 f"assumption chain (no acyclic upstream guarantee)" if circular else
                 f"{comp.name} assumption '{a.concept}' is neither an environment "
                 f"assumption nor discharged by an upstream guarantee"),
                contract=comp.name, subject=a.concept, provenance=prov,
            ))

    for comp in graph.components:
        for a in comp.assumptions:
            if a.kind != "boolean":
                continue
            key = f"{comp.name}.{a.concept}"
            if key in discharge:
                continue
            discharge[key] = "environment" if a.is_environment else "discharged"

    return discharge, diags


def _check_timing(graph: AGGraph) -> Tuple[Dict[str, Any], List[AGDiagnostic]]:
    """Additive timing-budget composition (§6.4, §7): sum(components) <= system."""
    diags: List[AGDiagnostic] = []
    budgets = {c.name: c.timing_budget for c in graph.components
               if c.timing_budget is not None}
    system_budget = graph.system.timing_budget if graph.system else None
    total = round(sum(budgets.values()), 12) if budgets else None

    # Unit safety: all declared timing units must agree (§6.4).
    units = {c.timing_unit for c in graph.all_contracts()
             if c.timing_budget is not None and c.timing_unit is not None}
    if len(units) > 1:
        diags.append(AGDiagnostic(
            CODE_UNIT_INCOMPATIBLE,
            f"timing budgets mix incompatible units {sorted(units)}",
            severity="error",
        ))

    ok: Optional[bool] = None
    if total is not None and system_budget is not None:
        ok = total <= system_budget + 1e-9
        if not ok:
            diags.append(AGDiagnostic(
                CODE_TIMING_BUDGET_EXCEEDED,
                f"component timing budgets sum to {total} > system deadline {system_budget}",
                contract=graph.system.name if graph.system else None,
            ))
    return (
        {"component_budgets": budgets, "sum": total,
         "system_deadline": system_budget, "ok": ok},
        diags,
    )


def _check_sufficiency(
    graph: AGGraph, discharge_diags: List[AGDiagnostic], aliases: Mapping[str, str]
) -> List[AGDiagnostic]:
    """Do the component guarantees collectively support the system guarantee (§9.9)?

    Bounded MVP: the system observation concept must be produced by some component
    Boolean guarantee, and no component may be left undischarged.
    """
    diags: List[AGDiagnostic] = []
    if not graph.system or graph.system.observation is None:
        return diags
    produced = {
        _norm(concept, aliases)
        for c in graph.components for concept in c.boolean_guarantee_concepts()
    }
    observation = _norm(graph.system.observation, aliases)
    blocked = any(
        d.code in (CODE_ASSUMPTION_UNDISCHARGED, CODE_CIRCULAR_ASSUMPTION)
        for d in discharge_diags
    )
    if observation not in produced or blocked:
        prov = {"element_id": graph.system.element_id,
                "span": graph.system.span.as_dict() if graph.system.span else None}
        diags.append(AGDiagnostic(
            CODE_DECOMPOSITION_INSUFFICIENT,
            (f"component guarantees do not collectively support the system "
             f"observation '{graph.system.observation}'"),
            contract=graph.system.name, subject=graph.system.observation,
            provenance=prov,
        ))
    return diags


def check_ag_graph(
    graph: AGGraph, *, aliases: Optional[Mapping[str, str]] = None
) -> AGReport:
    """Run the bounded compositional A/G checks (§9) over an extracted graph."""
    alias_map = {k.strip().lower(): v.strip().lower() for k, v in (aliases or {}).items()}
    diagnostics: List[AGDiagnostic] = list(graph.parse_diagnostics)

    owner_count, own_diags = _check_ownership(graph)
    diagnostics.extend(own_diags)

    system_completeness = UNSUPPORTED
    component_completeness: Dict[str, str] = {}
    if graph.system:
        system_completeness, sys_diags = _classify_completeness(
            graph.system, owner_count=None
        )
        diagnostics.extend(sys_diags)
    for comp in graph.components:
        state, comp_diags = _classify_completeness(
            comp, owner_count=owner_count.get(comp.name, 0)
        )
        component_completeness[comp.name] = state
        diagnostics.extend(comp_diags)

    discharge, dis_diags = _check_discharge(graph, alias_map)
    diagnostics.extend(dis_diags)

    timing, timing_diags = _check_timing(graph)
    diagnostics.extend(timing_diags)

    diagnostics.extend(_check_sufficiency(graph, dis_diags, alias_map))

    has_error = any(d.code in _ERROR_CODES for d in diagnostics)
    any_incomplete = (
        system_completeness in (INCOMPLETE, UNSUPPORTED)
        or any(s in (INCOMPLETE, UNSUPPORTED) for s in component_completeness.values())
    )
    if has_error:
        verdict = "FAIL"
    elif any_incomplete:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    return AGReport(
        verdict=verdict,
        system_completeness=system_completeness,
        component_completeness=component_completeness,
        diagnostics=tuple(diagnostics),
        timing=timing,
        discharge=discharge,
        revision=graph.revision,
        model_digest=graph.model_digest,
    )
