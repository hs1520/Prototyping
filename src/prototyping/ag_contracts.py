"""Bounded Assume-Guarantee contract model and compositional checker (R2-BBAG).

The checker half of Increment 2 in ``docs/OPTION2_IMPLEMENTATION_DESIGN.md``
(§6, §9). Data structures mirror §6.1 (SystemContract / ComponentContract plus
decomposes/discharged_by/... edges); the bounded §6.4 / §9 obligations are:

  * per-contract completeness (READY / INCOMPLETE / UNSUPPORTED, §6.3);
  * one responsible owner per component guarantee (§16);
  * assumption discharge by explicit environment or upstream guarantee (§6.4),
    computed as a monotone availability fixpoint;
  * unit-safe numeric / Boolean compatibility (§6.4);
  * additive timing-budget composition (§6.4, §7);
  * whether the component guarantees support the system guarantee;
  * circular-assumption detection (§17 unsoundness risk).

This is A/G-inspired bounded checking, not a proof calculus (§2, §13): a PASS
means the extracted graph is complete and internally compatible. The input is
an already-extracted :class:`AGGraph` (``ag_extractor.py``); a fact absent
from the committed model is INCOMPLETE/UNSUPPORTED, never supplied from JSON
(§6.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ag_profile import (
    CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION,
    CODE_COMPONENT_GUARANTEE_NONATOMIC,
    CODE_CONTRACT_INCOMPLETE,
    CODE_CONTRACT_UNSUPPORTED,
    CODE_DECOMPOSITION_INSUFFICIENT,
    CODE_DECOMPOSITION_MISSING,
    CODE_DISCHARGE_EDGE_MISSING,
    CODE_GUARANTEE_MULTIPLE_OWNERS,
    CODE_GUARANTEE_NO_OWNER,
    CODE_INVARIANT_SEMANTICS_INVALID,
    CODE_INVARIANT_SEMANTICS_MISSING,
    CODE_OBSERVATION_MISSING,
    CODE_PATTERN_DECLARATION_INCONSISTENT,
    CODE_PATTERN_TOPOLOGY_INCOMPLETE,
    CODE_PRIORITY_TOPOLOGY_INCOMPLETE,
    CODE_PRIORITY_TOPOLOGY_MISSING,
    CODE_REALIZATION_ACTION_MISSING,
    CODE_REALIZATION_MISSING,
    CODE_REALIZATION_TRIGGER_MISSING,
    CODE_REALIZATION_UNREACHABLE,
    CODE_SOURCE_PROVENANCE_MISSING,
    CODE_SYSTEM_OBSERVATION_BINDING_MISSING,
    CODE_TIMING_BUDGET_EXCEEDED,
    CODE_TIMING_BUDGET_MISSING,
    CODE_UNIT_INCOMPATIBLE,
    DERIVED_SOURCE_KIND,
    INVARIANT_PATTERNS as PROFILE_INVARIANT_PATTERNS,
    INVARIANT_SOURCE_KINDS as PROFILE_INVARIANT_SOURCE_KINDS,
    KNOWN_PATTERNS as PROFILE_KNOWN_PATTERNS,
    LOCKED_UNTIL_RELEASE_PATTERN,
    PATTERN_INVARIANT_ROLES,
    STARTUP_INHIBIT_PATTERN,
    TIMED_PATTERN,
    TRIGGERED_PATTERNS as PROFILE_TRIGGERED_PATTERNS,
    UNTIMED_PATTERNS as PROFILE_UNTIMED_PATTERNS,
)

# ag-bounded-8: priority response members carry provenance in committed SysML.
# The checker knows no reviewed response names; it checks only that each
# declared member says whether it came from existing model behavior or an
# approved/derived design source. Gold-blind, but differs from v7, so the two
# are not pooled.
#
# ag-bounded-9 admits THRESHOLD_TRIGGERED_RESPONSE, a triggered pattern with
# no deadline: it keeps the timed pattern's arbitration obligations and drops
# the timing ones - no apportioned budget, no trigger_matches_timing_origin
# cross-check. A boundary component stops being mandatory where nothing is
# apportioned, and any declared one is still held to its obligation. The timed
# pattern is unchanged, but accepted declarations and untimed verdicts differ,
# so v8 and v9 are not pooled.
# ag-bounded-10: the extractor now reads a deadline written with a qualified
# or unit-suffixed type (`maxLatency : DurationValue [s] = ...`), which it
# previously missed, so a timed chain extracted with no budget. Verdicts on
# all 14 archived R2 runs are unchanged because the emitter never wrote that
# spelling; the visible input space changed, so v9 and v10 are not pooled.
AG_CHECKER_VERSION = "ag-bounded-10"

READY = "READY"
INCOMPLETE = "INCOMPLETE"
UNSUPPORTED = "UNSUPPORTED"

_ERROR_CODES = frozenset({
    CODE_GUARANTEE_NO_OWNER,
    CODE_GUARANTEE_MULTIPLE_OWNERS,
    CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION,
    CODE_UNIT_INCOMPATIBLE,
    CODE_TIMING_BUDGET_EXCEEDED,
    CODE_TIMING_BUDGET_MISSING,
    CODE_DECOMPOSITION_INSUFFICIENT,
    CODE_DECOMPOSITION_MISSING,
    CODE_DISCHARGE_EDGE_MISSING,
    CODE_REALIZATION_MISSING,
    CODE_REALIZATION_UNREACHABLE,
    CODE_REALIZATION_TRIGGER_MISSING,
    CODE_REALIZATION_ACTION_MISSING,
    CODE_OBSERVATION_MISSING,
    CODE_SOURCE_PROVENANCE_MISSING,
    CODE_PATTERN_DECLARATION_INCONSISTENT,
    CODE_PRIORITY_TOPOLOGY_MISSING,
    CODE_PRIORITY_TOPOLOGY_INCOMPLETE,
    CODE_INVARIANT_SEMANTICS_MISSING,
    CODE_INVARIANT_SEMANTICS_INVALID,
    CODE_PATTERN_TOPOLOGY_INCOMPLETE,
})

_TIMED_PATTERN = TIMED_PATTERN
_TRIGGERED_PATTERNS = frozenset(PROFILE_TRIGGERED_PATTERNS)
_INVARIANT_PATTERNS = frozenset(PROFILE_INVARIANT_PATTERNS)
# Patterns that apportion no deadline; the untimed triggered pattern owns no
# interval, like the invariant ones.
_UNTIMED_PATTERNS = frozenset(PROFILE_UNTIMED_PATTERNS)
_KNOWN_PATTERNS = frozenset(PROFILE_KNOWN_PATTERNS)

# The roles each invariant pattern is defined by. An invariant set leaving one
# unfilled has not stated the pattern, so this is a completeness obligation
# rather than a comparison against a reviewed answer.
#
# Public so authors can be told: ``ag_convention`` republishes these role names
# as an authoring rule and the tests pin the two tables together, so a role
# added here cannot go unstated in the prompts.
_INVARIANT_SOURCE_KINDS = frozenset(PROFILE_INVARIANT_SOURCE_KINDS)
_PRIORITY_MEMBER_SOURCE_KINDS = {
    "EXISTING_MODEL_BEHAVIOR",
    "STUDENT_APPROVED_DECOMPOSITION",
    DERIVED_SOURCE_KIND,
}
# REQ_SAFE_005's reviewed response set and precedence ordering used to be
# pinned here, which made a gold-blind runtime verdict depend on the reviewed
# answer and left the priority topology unmeasurable in the LLM-authored arm.
# The comparisons now live only in `ag_eval_semantics.priority_agreement`,
# which scores them against frozen gold.


def _norm(concept: str, aliases: Mapping[str, str]) -> str:
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
    kind: str
    is_environment: bool = False
    variable: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    constraint_name: Optional[str] = None
    ast: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class Guarantee:
    concept: str
    expr: str
    kind: str
    variable: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    constraint_name: Optional[str] = None
    ast: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class Contract:
    name: str
    role: str
    assumptions: Tuple[Assumption, ...] = ()
    guarantees: Tuple[Guarantee, ...] = ()
    timing_budget: Optional[float] = None
    timing_unit: Optional[str] = None
    timing_value_literal: Optional[str] = None
    timing_segment_required: Optional[bool] = None
    # Segments sharing a group run concurrently, so the group contributes its
    # maximum, not its sum. Undeclared means own group, i.e. serial, which is what
    # plain addition assumed.
    timing_segment_group: Optional[int] = None
    timing_margin: Optional[float] = None
    timing_margin_unit: Optional[str] = None
    timing_origin: Optional[str] = None
    observation: Optional[str] = None
    element_id: Optional[str] = None
    span: Optional[Span] = None
    owners: Tuple[str, ...] = ()
    source_requirement: Optional[str] = None
    declared_pattern: Optional[str] = None

    def boolean_guarantee_concepts(self) -> Tuple[str, ...]:
        return tuple(g.concept for g in self.guarantees if g.kind == "boolean")


@dataclass(frozen=True)
class AGEdge:
    kind: str
    src: str
    dst: str
    subject: Optional[str] = None


@dataclass(frozen=True)
class BehaviorTransition:
    source: str
    trigger: str
    target: str
    guard: Optional[str] = None


@dataclass(frozen=True)
class BehaviorRealization:
    name: str
    initial_state: Optional[str]
    transitions: Tuple[BehaviorTransition, ...] = ()
    entry_actions: Mapping[str, str] = field(default_factory=dict)
    element_id: Optional[str] = None
    span: Optional[Span] = None


@dataclass(frozen=True)
class InvariantRealization:
    name: str
    expression: str
    element_id: Optional[str] = None
    span: Optional[Span] = None


@dataclass(frozen=True)
class AGDiagnostic:
    code: str
    message: str
    contract: Optional[str] = None
    subject: Optional[str] = None
    severity: str = "error"
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
    parse_diagnostics: Tuple[AGDiagnostic, ...] = ()
    behaviors: Tuple[BehaviorRealization, ...] = ()
    invariant_realizations: Tuple[InvariantRealization, ...] = ()
    verification_targets: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    source_requirement_ids: Tuple[str, ...] = ()
    priority: Mapping[str, Any] = field(default_factory=dict)
    invariants: Tuple[Mapping[str, Any], ...] = ()
    selected_model_elements: Tuple[str, ...] = ()
    declared_event_signals: Tuple[str, ...] = ()

    def all_contracts(self) -> Tuple[Contract, ...]:
        return ((self.system,) if self.system else ()) + tuple(self.components)


@dataclass
class AGReport:
    verdict: str
    system_completeness: str
    component_completeness: Dict[str, str]
    diagnostics: Tuple[AGDiagnostic, ...]
    timing: Dict[str, Any]
    discharge: Dict[str, str]
    revision: Optional[int]
    allocations: Tuple[Dict[str, str], ...] = ()
    discharge_edges: Tuple[Dict[str, Any], ...] = ()
    realization_links: Tuple[Dict[str, Any], ...] = ()
    observation_links: Tuple[Dict[str, Any], ...] = ()
    source_requirement: Optional[str] = None
    timing_atomic: Optional[Mapping[str, Any]] = None
    priority: Optional[Mapping[str, Any]] = None
    invariants: Tuple[Mapping[str, Any], ...] = ()
    selected_model_elements: Tuple[str, ...] = ()
    checker_version: str = AG_CHECKER_VERSION

    def errors(self) -> Tuple[AGDiagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    def to_dict(self) -> Dict[str, Any]:
        # Shape of the read-only ``ag_contract_graph.json`` audit view (§14): the
        # report cites the source model revision and checker version, and is
        # deterministically regenerable. The ``graph`` block carries the predicted
        # structure so the independent post-hoc evaluator can score it without
        # re-running the checker (§13 separation).
        return {
            "artifact_role": "RUNTIME_A_G_PREDICTION",
            "evidence_role": "INTERVENTION_RUNTIME_CHECK",
            "producing_stage": "R2_COMPOSITIONAL_TRACE",
            "measurement_boundary": "INTERVENTION",
            "experiment_namespace": "BLACKBOARD_AG_V1",
            "configuration": "R2-BBAG",
            "checker_version": self.checker_version,
            "source_model_revision": self.revision,
            "verdict": self.verdict,
            "system_completeness": self.system_completeness,
            "component_completeness": dict(self.component_completeness),
            "timing": dict(self.timing),
            "discharge": dict(self.discharge),
            "graph": {
                "allocations": [dict(a) for a in self.allocations],
                "discharge_edges": [dict(e) for e in self.discharge_edges],
                "realization_links": [dict(e) for e in self.realization_links],
                "observation_links": [dict(e) for e in self.observation_links],
                **(
                    {"timing": dict(self.timing_atomic)}
                    if self.timing_atomic is not None else {}
                ),
                **(
                    {"priority": dict(self.priority)}
                    if self.priority is not None else {}
                ),
                **(
                    {
                        "invariants": [dict(item) for item in self.invariants],
                        "selected_model_elements": list(self.selected_model_elements),
                    }
                    if self.invariants else {}
                ),
            },
            "source_requirement": self.source_requirement,
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }


def _classify_completeness(
    contract: Contract, *, owner_count: Optional[int]
) -> Tuple[str, List[AGDiagnostic]]:
    diags: List[AGDiagnostic] = []
    prov = {"element_id": contract.element_id,
            "span": contract.span.as_dict() if contract.span else None}

    if any(g.kind == "unsupported" for g in contract.guarantees) or \
       any(a.kind == "unsupported" for a in contract.assumptions):
        diags.append(AGDiagnostic(
            CODE_CONTRACT_UNSUPPORTED,
            f"{contract.name} uses a constraint the bounded checker cannot represent",
            contract=contract.name, severity="warning", provenance=prov,
        ))
        return UNSUPPORTED, diags

    reasons: List[str] = []
    if not contract.guarantees:
        reasons.append("no guarantee (require constraint)")
    if contract.role == "component":
        non_atomic = [
            guarantee
            for guarantee in contract.guarantees
            if (
                guarantee.kind == "boolean"
                and (
                    not isinstance(guarantee.ast, Mapping)
                    or guarantee.ast.get("node") != "Identifier"
                )
            )
        ]
        for guarantee in non_atomic:
            diags.append(AGDiagnostic(
                CODE_COMPONENT_GUARANTEE_NONATOMIC,
                f"{contract.name} component guarantee "
                f"{guarantee.constraint_name or guarantee.concept} is compound; "
                "component guarantees must be one atomic Boolean concept so a "
                "realizing action can establish it",
                contract=contract.name,
                subject=guarantee.constraint_name or guarantee.concept,
                severity="warning",
                provenance=prov,
            ))
        if non_atomic:
            reasons.append(
                f"{len(non_atomic)} non-atomic component guarantee(s)"
            )
    pure_system_invariant = (
        contract.role == "system"
        and contract.declared_pattern in {
            STARTUP_INHIBIT_PATTERN,
            LOCKED_UNTIL_RELEASE_PATTERN,
        }
    )
    # A component contract may use A=true (no assume constraints), e.g. a
    # normally-safe lock mechanism. System contracts still need an explicit
    # envelope unless their profile is a pure invariant.
    if (
        contract.role == "system"
        and not contract.assumptions
        and not pure_system_invariant
    ):
        reasons.append("no assumption (assume constraint)")
    if contract.role == "component" and owner_count == 0:
        reasons.append("no responsible owner (satisfy relationship)")
    if contract.role == "system" and contract.observation is None:
        reasons.append("no system observation concept")
    if (
        contract.role == "system"
        and any(
            guarantee.constraint_name is not None
            for guarantee in contract.guarantees
        )
        and not any(
            guarantee.constraint_name == "g_observed"
            for guarantee in contract.guarantees
        )
    ):
        reasons.append("no `g_observed` system observation constraint")
        diags.append(AGDiagnostic(
            CODE_SYSTEM_OBSERVATION_BINDING_MISSING,
            f"{contract.name} has no distinct `require constraint g_observed "
            "{ <observation> }`; invariant or differently named constraints do "
            "not bind the system observation",
            contract=contract.name,
            severity="warning",
            provenance=prov,
        ))
    if contract.role == "system" and contract.source_requirement is None:
        reasons.append("no immutable source-requirement provenance")
        diags.append(AGDiagnostic(
            CODE_SOURCE_PROVENANCE_MISSING,
            f"{contract.name} does not cite its source requirement in SysML",
            contract=contract.name, provenance=prov,
        ))
    if reasons:
        diags.append(AGDiagnostic(
            CODE_CONTRACT_INCOMPLETE,
            f"{contract.name} is INCOMPLETE: {'; '.join(reasons)}",
            contract=contract.name, severity="warning", provenance=prov,
        ))
        return INCOMPLETE, diags

    return READY, diags


def _check_ownership(graph: AGGraph) -> Tuple[Dict[str, int], List[AGDiagnostic]]:
    diags: List[AGDiagnostic] = []
    owner_count: Dict[str, int] = {
        c.name: len(set(c.owners)) for c in graph.components
    }
    for comp in graph.components:
        n = owner_count[comp.name]
        prov = {"element_id": comp.element_id,
                "span": comp.span.as_dict() if comp.span else None}
        if n == 0:
            diags.append(AGDiagnostic(
                CODE_GUARANTEE_NO_OWNER,
                f"{comp.name} has no satisfying component owner; its guarantee is unallocated",
                contract=comp.name, provenance=prov,
            ))
        elif n > 1:
            diags.append(AGDiagnostic(
                CODE_GUARANTEE_MULTIPLE_OWNERS,
                f"{comp.name} is satisfied by {n} component owners; ownership must be unique",
                contract=comp.name, provenance=prov,
            ))
        decomposition_count = sum(
            edge.kind == "decomposes" and edge.dst == comp.name
            for edge in graph.edges
        )
        if decomposition_count != 1:
            diags.append(AGDiagnostic(
                CODE_DECOMPOSITION_MISSING,
                f"{comp.name} must have exactly one system decomposition edge; "
                f"found {decomposition_count}",
                contract=comp.name, provenance=prov,
            ))
    return owner_count, diags


def _check_discharge(
    graph: AGGraph, aliases: Mapping[str, str]
) -> Tuple[Dict[str, str], List[AGDiagnostic]]:
    diags: List[AGDiagnostic] = []
    discharge: Dict[str, str] = {}
    discharge_edges: List[Dict[str, Any]] = []

    available: set = set()
    available_by: Dict[str, str] = {}
    if graph.system:
        for a in graph.system.assumptions:
            if a.kind == "boolean":
                c = _norm(a.concept, aliases)
                available.add(c)
                available_by.setdefault(c, "environment")
    for comp in graph.components:
        for a in comp.assumptions:
            if a.is_environment and a.kind == "boolean":
                c = _norm(a.concept, aliases)
                available.add(c)
                available_by.setdefault(c, "environment")

    explicit_discharge = {
        (edge.src, edge.dst, _norm(edge.subject or "", aliases))
        for edge in graph.edges if edge.kind == "discharges"
    }
    guarantee_producer: Dict[str, str] = {}
    for comp in graph.components:
        for concept in comp.boolean_guarantee_concepts():
            guarantee_producer.setdefault(_norm(concept, aliases), comp.name)

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
            edges_ready = all(
                (
                    guarantee_producer.get(concept),
                    comp.name,
                    concept,
                ) in explicit_discharge
                for concept in needed
                if guarantee_producer.get(concept) is not None
            )
            if needed <= available and edges_ready:
                activated.add(comp.name)
                for concept in comp.boolean_guarantee_concepts():
                    c = _norm(concept, aliases)
                    available.add(c)
                    available_by.setdefault(c, comp.name)
                changed = True

    # Distinguish a cycle from a cascade behind an upstream gap (§16, §17).
    # producer[c] = component whose Boolean guarantee publishes concept c; C
    # depends on producer[c] for each non-environment Boolean assumption c. An
    # assumption is CIRCULAR only when its producer transitively reaches C back
    # through the dependency graph; otherwise it is an undischarged gap.
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

    for comp in graph.components:
        prov = {"element_id": comp.element_id,
                "span": comp.span.as_dict() if comp.span else None}
        for a in comp.assumptions:
            if a.kind != "boolean":
                continue
            concept = _norm(a.concept, aliases)
            if a.is_environment:
                status, by = "environment", "environment"
            elif concept in available:
                by = available_by.get(concept, "environment")
                has_edge = (
                    by, comp.name, concept
                ) in explicit_discharge if by != "environment" else True
                if has_edge:
                    status = "environment" if by == "environment" else "discharged"
                else:
                    status = "undischarged"
                    diags.append(AGDiagnostic(
                        CODE_DISCHARGE_EDGE_MISSING,
                        f"{comp.name} assumption '{a.concept}' has a matching "
                        f"upstream guarantee but no explicit discharge dependency",
                        contract=comp.name, subject=a.concept, provenance=prov,
                    ))
                    by = None
            else:
                prod = producer.get(concept)
                circular = prod is not None and _reaches(prod, comp.name)
                status = "circular" if circular else "undischarged"
                by = None
                code = (CODE_CIRCULAR_ASSUMPTION if circular
                        else CODE_ASSUMPTION_UNDISCHARGED)
                diags.append(AGDiagnostic(
                    code,
                    (f"{comp.name} assumption '{a.concept}' is part of a circular "
                     f"assumption chain (no acyclic upstream guarantee)" if circular
                     else f"{comp.name} assumption '{a.concept}' is neither an "
                     f"environment assumption nor discharged by an upstream guarantee"),
                    contract=comp.name, subject=a.concept, provenance=prov,
                ))
            discharge[f"{comp.name}.{a.concept}"] = status
            discharge_edges.append({
                "component": comp.name, "assumption": a.concept,
                "status": status, "by": by,
            })

    return discharge, discharge_edges, diags


def _compose_timing(
    components: Sequence[Contract],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compose component budgets by declared structure, not by blanket addition.

    Plain addition is sound only for a serial chain: two concurrent 0.3 s
    responses occupy 0.3 s, not 0.6 s, so a valid design would be reported as
    exceeding its deadline. Segments sharing a ``timingSegmentGroup`` are
    concurrent and contribute their maximum; groups compose serially and
    contribute their sum. An undeclared segment is its own group, so a model that
    says nothing composes as before.
    """
    groups: Dict[Any, List[Tuple[str, float]]] = {}
    for index, component in enumerate(components):
        if component.timing_budget is None:
            continue
        key = (
            component.timing_segment_group
            if component.timing_segment_group is not None
            else f"_serial_{index}"
        )
        groups.setdefault(key, []).append((component.name, component.timing_budget))
    if not groups:
        return None, {"groups": [], "structure": "none"}
    structure = []
    total = 0.0
    for key, members in groups.items():
        concurrent = len(members) > 1
        contribution = (
            max(value for _name, value in members) if concurrent
            else members[0][1]
        )
        total += contribution
        structure.append({
            "group": key if isinstance(key, int) else None,
            "members": [name for name, _value in members],
            "composition": "concurrent_max" if concurrent else "serial",
            "contributes": contribution,
        })
    return round(total, 12), {
        "groups": structure,
        "structure": (
            "serial_and_concurrent"
            if any(item["composition"] == "concurrent_max" for item in structure)
            else "serial"
        ),
    }


def _check_timing(graph: AGGraph) -> Tuple[Dict[str, Any], List[AGDiagnostic]]:
    diags: List[AGDiagnostic] = []
    budgets = {c.name: c.timing_budget for c in graph.components
               if c.timing_budget is not None}
    system_budget = graph.system.timing_budget if graph.system else None
    margin = (graph.system.timing_margin if graph.system else None) or 0.0
    total, composition = _compose_timing(graph.components)

    units = {c.timing_unit for c in graph.all_contracts()
             if c.timing_budget is not None and c.timing_unit is not None}
    missing_units = [
        c.name for c in graph.all_contracts()
        if c.timing_budget is not None and c.timing_unit is None
    ]
    # a margin is a duration too: without units 50 ms of reserve is compared
    # against a deadline in seconds
    if graph.system is not None and graph.system.timing_margin is not None:
        if graph.system.timing_margin_unit is None:
            missing_units.append(f"{graph.system.name}.timingMargin")
        else:
            units.add(graph.system.timing_margin_unit)
    if missing_units:
        diags.append(AGDiagnostic(
            CODE_UNIT_INCOMPATIBLE,
            "timing budgets lack explicit SysML units: "
            + ", ".join(sorted(missing_units)),
        ))
    if len(units) > 1:
        diags.append(AGDiagnostic(
            CODE_UNIT_INCOMPATIBLE,
            f"timing budgets mix incompatible units {sorted(units)}",
            severity="error",
        ))

    ok: Optional[bool] = None
    if system_budget is not None:
        missing_budgets = [
            c.name for c in graph.components
            if c.timing_budget is None
            and c.timing_segment_required is not False
        ]
        if missing_budgets:
            diags.append(AGDiagnostic(
                CODE_TIMING_BUDGET_MISSING,
                "timed system contract has components without additive budgets: "
                + ", ".join(sorted(missing_budgets)),
            ))
    if total is not None and system_budget is not None:
        committed = round(total + margin, 12)
        ok = committed <= system_budget + 1e-9
        if not ok:
            detail = (
                f"composed timing {total}"
                + (f" + declared margin {margin}" if margin else "")
                + f" = {committed} > system deadline {system_budget}"
            )
            diags.append(AGDiagnostic(
                CODE_TIMING_BUDGET_EXCEEDED,
                detail,
                contract=graph.system.name if graph.system else None,
            ))
    return (
        {"component_budgets": budgets, "sum": total,
         "margin": margin or None,
         "committed": round(total + margin, 12) if total is not None else None,
         "composition": composition,
         "system_deadline": system_budget, "ok": ok},
        diags,
    )


def _flat_token(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _check_realization(
    graph: AGGraph, aliases: Mapping[str, str]
) -> Tuple[List[Dict[str, Any]], List[AGDiagnostic]]:
    diags: List[AGDiagnostic] = []
    links: List[Dict[str, Any]] = []
    behaviors = {item.name: item for item in graph.behaviors}
    invariant_realizations = {
        item.name: item for item in graph.invariant_realizations
    }
    system_triggers = {
        _norm(a.concept, aliases) for a in (graph.system.assumptions if graph.system else ())
        if a.kind == "boolean"
    }
    for comp in graph.components:
        realization_edges = [
            edge for edge in graph.edges
            if edge.kind == "realized_by" and edge.src == comp.name
        ]
        if len(realization_edges) != 1:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_MISSING,
                f"{comp.name} has no unique realization dependency",
                contract=comp.name,
            ))
            continue
        realization_name = realization_edges[0].dst
        invariant = invariant_realizations.get(realization_name)
        if invariant is not None:
            expected = {
                _flat_token(item.concept)
                for item in comp.guarantees
                if item.kind == "boolean"
            }
            expression = _flat_token(invariant.expression)
            missing = sorted(
                token for token in expected
                if token and token not in expression
            )
            if missing:
                diags.append(AGDiagnostic(
                    CODE_REALIZATION_ACTION_MISSING,
                    f"{comp.name} invariant realization does not reference "
                    f"guarantees {missing}",
                    contract=comp.name,
                ))
            links.append({
                "contract": comp.name,
                "owner": comp.owners[0] if len(comp.owners) == 1 else None,
                "behavior": invariant.name,
                "realization_kind": "INVARIANT",
                "invariant_expression": invariant.expression,
                "continuous_guarantee": True,
                "status": "PASS" if not missing else "FAIL",
            })
            continue
        if realization_name not in behaviors:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_MISSING,
                f"{comp.name} realization target {realization_name} is absent",
                contract=comp.name,
            ))
            continue
        behavior = behaviors[realization_name]
        adjacency: Dict[str, List[BehaviorTransition]] = {}
        for transition in behavior.transitions:
            adjacency.setdefault(transition.source, []).append(transition)
        reachable = set()
        frontier = [behavior.initial_state] if behavior.initial_state else []
        used_transitions: List[BehaviorTransition] = []
        while frontier:
            state = frontier.pop()
            if state in reachable:
                continue
            reachable.add(state)
            for transition in adjacency.get(state, ()):
                used_transitions.append(transition)
                frontier.append(transition.target)

        expected_triggers = {
            _flat_token(a.concept) for a in comp.assumptions if a.kind == "boolean"
        } | {_flat_token(item) for item in system_triggers}
        trigger_ok = any(
            any(expected and expected in _flat_token(t.trigger)
                for expected in expected_triggers)
            for t in used_transitions
        )
        expected_trigger_concepts = sorted({
            a.concept for a in comp.assumptions if a.kind == "boolean"
        } | {
            a.concept
            for a in (graph.system.assumptions if graph.system else ())
            if a.kind == "boolean"
        })
        expected_signal_examples = [
            f"{concept[:1].upper()}{concept[1:]}Signal"
            for concept in expected_trigger_concepts
            if concept
        ]
        compatible_declared_signals = sorted(
            signal for signal in graph.declared_event_signals
            if any(
                expected and expected in _flat_token(signal)
                for expected in expected_triggers
            )
        )
        component_trigger_tokens = {
            _flat_token(a.concept)
            for a in comp.assumptions
            if a.kind == "boolean"
        }
        scoped_repair_compatible_signals = sorted(
            signal for signal in graph.declared_event_signals
            if any(
                expected and expected in _flat_token(signal)
                for expected in component_trigger_tokens
            )
        )

        guarantee_tokens = {
            _flat_token(g.concept) for g in comp.guarantees if g.kind == "boolean"
        }
        action_matches = [
            (state, action) for state, action in behavior.entry_actions.items()
            if (
                state in reachable
                # A clear/reset action negates a positive guarantee; a substring match alone
                # would count it as a realization.
                and not _flat_token(action).startswith("clear")
                and any(
                    token and token in _flat_token(action)
                    for token in guarantee_tokens
                )
            )
        ]
        # An untimed availability invariant is established in the initial state at the
        # chain boundary and has no activation transition. Other component guarantees
        # require a reachable trigger-response transition.
        availability_invariant = (
            comp.timing_segment_required is False
            and behavior.initial_state in behavior.entry_actions
            and bool(action_matches)
        )
        # An A=true lifecycle component is driven by typed interface events, not a
        # permanent environment predicate, so its reachable transitions are the
        # trigger evidence; a conjunctive assumption would change the event
        # semantics of power-on/power-loss.
        unconditional_lifecycle = not comp.assumptions and bool(used_transitions)
        trigger_ok = trigger_ok or availability_invariant or unconditional_lifecycle
        if not trigger_ok:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_TRIGGER_MISSING,
                f"{comp.name} realization has no reachable trigger compatible "
                "with its/system assumptions; an accepted signal name must "
                "contain one complete assumption concept, for example one of "
                f"{expected_signal_examples}",
                contract=comp.name,
                provenance={
                    "expected_trigger_concepts": expected_trigger_concepts,
                    "compatible_declared_signals": compatible_declared_signals,
                    "scoped_repair_compatible_signals":
                        scoped_repair_compatible_signals,
                },
            ))
        if not reachable or (
            not used_transitions
            and not availability_invariant
            and not unconditional_lifecycle
        ):
            diags.append(AGDiagnostic(
                CODE_REALIZATION_UNREACHABLE,
                f"{comp.name} realization has no reachable trigger-response path",
                contract=comp.name,
                provenance={
                    "expected_trigger_concepts": expected_trigger_concepts,
                    "compatible_declared_signals": compatible_declared_signals,
                    "scoped_repair_compatible_signals":
                        scoped_repair_compatible_signals,
                },
            ))
        if not action_matches:
            # Name the concept the action carries. The message used to say only "for its
            # guarantee", and a repair answered it by naming the action after its state
            # (`setParachuteDeploymentSelected` for a contract guaranteeing
            # `parachuteResponseSelected`), which the gate refused. The concept is already
            # in the committed model, so naming it keeps the checker gold-blind.
            diags.append(AGDiagnostic(
                CODE_REALIZATION_ACTION_MISSING,
                f"{comp.name} has no reachable response entry action naming one "
                f"of its guarantee concepts {sorted(guarantee_tokens)}",
                contract=comp.name,
            ))
        links.append({
            "contract": comp.name,
            "owner": comp.owners[0] if len(comp.owners) == 1 else None,
            "behavior": behavior.name,
            "realization_kind": "STATE_MACHINE",
            "initial_state": behavior.initial_state,
            "reachable_states": sorted(reachable),
            "trigger_ok": trigger_ok,
            "response_actions": [action for _state, action in action_matches],
            "response_states": [state for state, _action in action_matches],
            # Committed-model paths used by the post-hoc realization evaluator. Competing
            # transitions that do not enter a guarantee-producing state are excluded;
            # priority topology has its own category.
            "response_paths": [
                {
                    "source": transition.source,
                    "trigger": transition.trigger,
                    "target": transition.target,
                    "guard": transition.guard,
                    "action": action,
                }
                for state, action in action_matches
                for transition in used_transitions
                if transition.target == state
            ] or (
                [
                    {
                        "source": behavior.initial_state,
                        "trigger": None,
                        "target": behavior.initial_state,
                        "guard": None,
                        "action": action,
                    }
                    for state, action in action_matches
                    if state == behavior.initial_state
                ]
                if availability_invariant else []
            ),
            "continuous_guarantee": availability_invariant,
            "status": (
                "PASS"
                if trigger_ok
                and action_matches
                and (
                    used_transitions
                    or availability_invariant
                    or unconditional_lifecycle
                )
                else "FAIL"
            ),
        })
    return links, diags


def _check_observation(
    graph: AGGraph,
) -> Tuple[List[Dict[str, Any]], List[AGDiagnostic]]:
    links: List[Dict[str, Any]] = []
    diags: List[AGDiagnostic] = []
    if graph.system is None:
        return links, diags
    edges = [
        edge for edge in graph.edges
        if edge.kind == "observed_by" and edge.src == graph.system.name
    ]
    valid = [
        edge for edge in edges
        if graph.system.name in graph.verification_targets.get(edge.dst, ())
    ]
    for edge in edges:
        links.append({
            "contract": edge.src,
            "verification": edge.dst,
            "observation": graph.system.observation,
            "status": "PASS" if edge in valid else "FAIL",
        })
    if len(valid) != 1:
        diags.append(AGDiagnostic(
            CODE_OBSERVATION_MISSING,
            f"{graph.system.name} requires exactly one verification observation "
            f"linked by dependency; found {len(valid)}",
            contract=graph.system.name,
            subject=graph.system.observation,
        ))
    return links, diags


def _ast_identifiers(node: Any) -> set[str]:
    if not isinstance(node, Mapping):
        return set()
    kind = node.get("node")
    if kind == "Identifier":
        return {str(node.get("name") or "")}
    if kind == "Not":
        return _ast_identifiers(node.get("expr"))
    if kind in {"And", "Or"}:
        return set().union(*(
            _ast_identifiers(item) for item in (node.get("operands") or ())
        ))
    if kind == "Implies":
        return (
            _ast_identifiers(node.get("antecedent"))
            | _ast_identifiers(node.get("consequent"))
        )
    return set()


def _ast_shape(node: Any) -> tuple:
    """Canonical structural shape for the fixed runtime-profile AST subset.

    Profile validation only, so an arbitrary invariant does not pass just because
    it carries an approved identifier.
    """
    if not isinstance(node, Mapping):
        return ("INVALID",)
    kind = str(node.get("node") or "")
    if kind == "Identifier":
        return ("Identifier", str(node.get("name") or ""))
    if kind == "Not":
        return ("Not", _ast_shape(node.get("expr")))
    if kind in {"And", "Or"}:
        return (
            kind,
            tuple(sorted(_ast_shape(item) for item in (node.get("operands") or ()))),
        )
    if kind == "Implies":
        return (
            "Implies",
            _ast_shape(node.get("antecedent")),
            _ast_shape(node.get("consequent")),
        )
    return ("INVALID", kind)


# REQ_SAFE_004's and REQ_SAFE_008's reviewed invariant sets (ids, ASTs,
# provenance) used to be pinned here and compared against directly, which
# left the runtime checker holding the answer for those chains.


def _behavior_for_contract(
    graph: AGGraph,
    realization_links: List[Dict[str, Any]],
    contract_name: str,
) -> Optional[BehaviorRealization]:
    link = next(
        (
            item for item in realization_links
            if item.get("contract") == contract_name
        ),
        None,
    )
    behavior_name = str((link or {}).get("behavior") or "")
    return next(
        (item for item in graph.behaviors if item.name == behavior_name),
        None,
    )


def _negated_identifiers(node: Any) -> set[str]:
    """Identifiers appearing under a negation in a bounded Boolean AST.

    Negated consequents name what the invariant forbids, e.g. the states a
    startup inhibit keeps the system out of, so they are read from the AST rather
    than from a list of state names.
    """
    if not isinstance(node, Mapping):
        return set()
    if node.get("node") == "Not":
        return _ast_identifiers(node.get("expr"))
    operands = node.get("operands") or ()
    result: set[str] = set()
    for item in operands:
        result |= _negated_identifiers(item)
    return result


def _invariant_roles(graph: AGGraph) -> Dict[str, str]:
    """The concepts a locked-until-authorised-release chain's invariants define.

    The topology follows from the invariants rather than from element names:

      * ``<power-on> => <locked>``            fixes the default-safe state;
      * ``<unlocked> => <authorisation>``     fixes the only way out of it;
      * ``not <power> => <locked>``           fixes where power loss returns to.

    Deriving the roles this way lets the obligation be checked without the checker
    holding REQ_SAFE_008's state names, signals and action spellings. The unlocked
    state may be a concept of its own or simply ``not <locked>``: both state the
    release obligation, and a measured run wrote it the second way.
    """
    roles: Dict[str, str] = {}
    implications: List[Tuple[Any, str]] = []
    negated: List[Tuple[str, str]] = []
    concluded: Dict[str, int] = {}
    for invariant in graph.invariants or ():
        if not isinstance(invariant, Mapping):
            continue
        antecedent = invariant.get("trigger_or_antecedent_ast") or {}
        consequents = _ast_identifiers(invariant.get("required_consequent_ast") or {})
        if len(consequents) != 1:
            continue
        consequent_name = next(iter(consequents))
        concluded[consequent_name] = concluded.get(consequent_name, 0) + 1
        # A negated antecedent is either de-energise-to-lock or the release rule over
        # the complement of the locked concept. Collect both and tell them apart once
        # every conclusion is known; declaration order made the roles depend on how
        # the author listed them.
        if isinstance(antecedent, Mapping) and antecedent.get("node") == "Not":
            operand = _ast_identifiers(antecedent.get("expr"))
            if len(operand) == 1:
                negated.append((next(iter(operand)), consequent_name))
            continue
        implications.append((antecedent, consequent_name))
    # Two invariants conclude the locked concept (power-on defaults to it, power
    # loss returns to it), so the concept concluded most often is it. Identifying
    # it first tells the two negated-antecedent forms apart.
    ranked = sorted(concluded.items(), key=lambda item: -item[1])
    locked_concept = (
        ranked[0][0]
        if ranked and ranked[0][1] > 1 and (len(ranked) == 1 or ranked[1][1] < ranked[0][1])
        else None
    )
    for operand, consequent_name in negated:
        if operand == locked_concept:
            roles["locked"] = operand
            roles["unlocked"] = operand
            roles["authorisation"] = consequent_name
        else:
            roles["power"] = operand
            roles["locked"] = consequent_name
    for antecedent, consequent_name in implications:
        antecedents = _ast_identifiers(antecedent)
        if len(antecedents) != 1:
            continue
        antecedent_name = next(iter(antecedents))
        # <power-on> => <locked> fixes the default-safe state; anything else implying
        # a single concept is the guarded-release rule. With no de-energise invariant
        # the locked concept is unknown, so the pattern's own word for it is the only
        # signal left; that model fails on the missing power role anyway.
        if consequent_name == roles.get("locked") or (
            "locked" not in roles and "lock" in consequent_name.lower()
        ):
            roles.setdefault("locked", consequent_name)
            roles.setdefault("power_on", antecedent_name)
        else:
            roles["unlocked"] = antecedent_name
            roles["authorisation"] = consequent_name
    return roles


def _startup_inhibit_roles(graph: AGGraph) -> Dict[str, Any]:
    """The concepts a startup-inhibit chain's invariants define.

    As with de-energise-to-lock, the topology follows from the invariants rather
    than from element names:

      * ``<condition> => not <forbidden> ...``  names the states inhibition forbids;
      * ``<latch> => <inhibited> ...``          names the latch and what it inhibits;
      * ``<reset> => not <latch>``              names the event that clears it.
    """
    roles: Dict[str, Any] = {"forbidden": set()}
    for invariant in graph.invariants or ():
        if not isinstance(invariant, Mapping):
            continue
        antecedent = invariant.get("trigger_or_antecedent_ast") or {}
        consequent = invariant.get("required_consequent_ast") or {}
        negated = _negated_identifiers(consequent)
        antecedents = _ast_identifiers(antecedent)
        if negated and len(antecedents) == 1 and len(negated) == 1:
            roles["reset"] = next(iter(antecedents))
            roles["latch"] = next(iter(negated))
        elif negated:
            roles["forbidden"] |= negated
        elif len(antecedents) == 1:
            roles.setdefault("latch", next(iter(antecedents)))
            roles["inhibited"] = _ast_identifiers(consequent)
    return roles


def _startup_inhibit_topology_obligations(
    graph: AGGraph,
    realization_links: List[Dict[str, Any]],
) -> Tuple[Dict[str, bool], Optional[str]]:
    """Whether the model latches inhibition on failure and clears it only on a pass.

    Judged against the chain's own invariants, so a conforming model may name its
    states and signals freely.
    """
    roles = _startup_inhibit_roles(graph)
    latch = roles.get("latch")
    if not latch:
        return {"latch_role_present": False}, None

    behavior = next(
        (
            _behavior_for_contract(graph, realization_links, item.name)
            for item in graph.components
            if latch in item.boolean_guarantee_concepts()
        ),
        None,
    )
    if behavior is None:
        return {
            "latch_role_present": True,
            "latch_behavior_present": False,
        }, None
    if not behavior.initial_state:
        return {
            "latch_role_present": True,
            "latch_behavior_present": True,
            "initial_state_present": False,
        }, behavior.name

    set_token, clear_token = _flat_token(f"set{latch}"), _flat_token(f"clear{latch}")
    inhibit_states = {
        state for state, action in behavior.entry_actions.items()
        if _flat_token(action) == set_token
    }
    reset_states = {
        state for state, action in behavior.entry_actions.items()
        if _flat_token(action) == clear_token
    }
    obligations: Dict[str, bool] = {
        "latch_role_present": True,
        "latch_behavior_present": True,
        "initial_state_present": True,
        "single_inhibit_state": len(inhibit_states) == 1,
        "reset_state_present": bool(reset_states),
    }
    if len(inhibit_states) != 1 or not reset_states:
        return obligations, behavior.name
    inhibit_state = next(iter(inhibit_states))
    obligations["inhibit_not_initial"] = inhibit_state != behavior.initial_state

    # inhibition and the pass that clears it are alternatives of one decision
    # point, otherwise the latch is not a self-test outcome
    entering = {
        transition.source for transition in behavior.transitions
        if transition.target == inhibit_state
    }
    clearing = {
        transition.source for transition in behavior.transitions
        if transition.target in reset_states
    }
    obligations["common_decision_source"] = bool(entering and (entering & clearing))
    # The triggers leaving the inhibited state differ from the ones reaching it;
    # otherwise the failure signal both sets and clears the latch.
    latching_triggers = {
        transition.trigger for transition in behavior.transitions
        if transition.target == inhibit_state
    }
    obligations["distinct_latch_and_reset_triggers"] = not any(
        transition.trigger in latching_triggers
        for transition in behavior.transitions
        if transition.source == inhibit_state
    )
    forbidden = {_flat_token(item) for item in roles.get("forbidden") or ()}
    obligations["forbidden_states_unreachable"] = not any(
        _flat_token(transition.target) in forbidden
        for transition in behavior.transitions
    )
    return obligations, behavior.name


def _locked_release_topology_obligations(
    graph: AGGraph,
    realization_links: List[Dict[str, Any]],
) -> Tuple[Dict[str, bool], Optional[str]]:
    """Whether the model instantiates de-energise-to-lock, judged against its own
    invariants rather than against REQ_SAFE_008's spellings.

    The obligations are the default state being the locked one, authorisation
    being the only way out of it, and power loss returning to it. Which
    identifiers carry those roles comes from the declared invariants, so a model
    that names its states differently still conforms, and reviewed names with
    wrong wiring do not.
    """
    roles = _invariant_roles(graph)
    locked = roles.get("locked")
    authorisation = roles.get("authorisation")
    if not locked or not authorisation:
        return {
            "locked_role_present": bool(locked),
            "authorisation_role_present": bool(authorisation),
        }, None

    mechanism = next(
        (
            _behavior_for_contract(graph, realization_links, item.name)
            for item in graph.components
            if locked in item.boolean_guarantee_concepts()
        ),
        None,
    )
    if mechanism is None:
        return {
            "locked_role_present": True,
            "authorisation_role_present": True,
            "lock_behavior_present": False,
        }, None
    if not mechanism.initial_state:
        return {
            "locked_role_present": True,
            "authorisation_role_present": True,
            "lock_behavior_present": True,
            "initial_state_present": False,
        }, mechanism.name

    signal = f"{authorisation[:1].upper()}{authorisation[1:]}Signal"
    unlock_transitions = [
        transition for transition in mechanism.transitions
        if transition.trigger == signal
    ]
    obligations: Dict[str, bool] = {
        "locked_role_present": True,
        "authorisation_role_present": True,
        "lock_behavior_present": True,
        "initial_state_present": True,
        "authorised_unlock_transition": bool(unlock_transitions),
    }
    if not unlock_transitions:
        return obligations, mechanism.name
    unlocked_states = {transition.target for transition in unlock_transitions}
    obligations["single_unlocked_state"] = len(unlocked_states) == 1
    if len(unlocked_states) != 1:
        return obligations, mechanism.name
    unlocked_state = next(iter(unlocked_states))
    obligations["unlocked_not_initial"] = unlocked_state != mechanism.initial_state

    obligations["authorisation_is_only_unlock_path"] = not any(
        transition.target == unlocked_state and transition.trigger != signal
        for transition in mechanism.transitions
    )
    # Losing power returns to the default-safe state on an event distinct from the
    # one that energises it; without that, a model satisfies the return edge with
    # the power-on signal itself and one event both energises and de-energises.
    energising = {
        transition.trigger for transition in mechanism.transitions
        if transition.source == mechanism.initial_state
    }
    obligations["power_loss_returns_to_locked"] = any(
        transition.source == unlocked_state
        and transition.target == mechanism.initial_state
        and transition.trigger not in energising
        for transition in mechanism.transitions
    )
    obligations["initial_state_establishes_locked"] = (
        _flat_token(locked) in _flat_token(
        mechanism.entry_actions.get(mechanism.initial_state, "")
        )
    )
    return obligations, mechanism.name


def _check_profile_semantics(
    graph: AGGraph,
    realization_links: List[Dict[str, Any]],
    observation_links: List[Dict[str, Any]],
    *,
    require_priority_member_provenance: bool,
) -> List[AGDiagnostic]:
    """Check bounded-profile facts extracted only from committed SysML.

    This is structural/internal consistency checking, not evaluator-gold
    comparison and not a formal proof.
    """
    if graph.system is None:
        return []
    diagnostics: List[AGDiagnostic] = []
    system = graph.system
    declared_pattern = system.declared_pattern
    effective_pattern = (
        declared_pattern
        or (
            _TIMED_PATTERN
            if system.timing_budget is not None
            else STARTUP_INHIBIT_PATTERN
        )
    )
    # Which pattern a requirement ought to instantiate is an accuracy question the
    # evaluator answers against frozen gold. This checker is gold-blind: it asks
    # only whether the model is internally consistent with the declared pattern.
    if declared_pattern is not None and declared_pattern not in _KNOWN_PATTERNS:
        diagnostics.append(AGDiagnostic(
            CODE_PATTERN_DECLARATION_INCONSISTENT,
            f"{system.name} declares unknown safety_pattern={declared_pattern!r}; "
            f"expected one of {sorted(_KNOWN_PATTERNS)}",
            contract=system.name,
            subject=system.source_requirement,
        ))
    if (
        (effective_pattern == _TIMED_PATTERN and system.timing_budget is None)
        or (
            effective_pattern in _UNTIMED_PATTERNS
            and system.timing_budget is not None
        )
    ):
        diagnostics.append(AGDiagnostic(
            CODE_PATTERN_DECLARATION_INCONSISTENT,
            f"{system.name} pattern/timing declaration is inconsistent",
            contract=system.name,
            subject=effective_pattern,
        ))

    if effective_pattern in _TRIGGERED_PATTERNS:
        priority = graph.priority
        if not isinstance(priority, Mapping) or not priority:
            diagnostics.append(AGDiagnostic(
                CODE_PRIORITY_TOPOLOGY_MISSING,
                f"{system.name} triggered response ({effective_pattern}) has no "
                "extracted priority contract/topology",
                contract=system.name,
            ))
            return diagnostics
        members = {str(item) for item in (priority.get("members") or ())}
        member_provenance = [
            item for item in (priority.get("member_provenance") or ())
            if isinstance(item, Mapping)
        ]
        provenance_members = {
            str(item.get("response") or "")
            for item in member_provenance
            if (
                str(item.get("source_kind") or "")
                in _PRIORITY_MEMBER_SOURCE_KINDS
                and str(item.get("source_id") or "")
            )
        }
        edges = {
            (str(item.get("higher") or ""), str(item.get("lower") or ""))
            for item in (priority.get("edges") or ())
            if isinstance(item, Mapping)
        }
        topology = priority.get("arbitration_topology")
        selected = None
        selection_when = ""
        guards: set[str] = set()
        reachable = False
        if isinstance(topology, Mapping):
            selection = topology.get("selection")
            if isinstance(selection, Mapping):
                selected = str(selection.get("selected_response") or "")
                selection_when = str(selection.get("when") or "")
            guards = {
                str(item.get("response") or "")
                for item in (topology.get("competing_transition_guards") or ())
                if isinstance(item, Mapping)
            }
            reachable = topology.get("parachute_transition_reachable") is True
        trigger = str(priority.get("trigger") or "")
        higher = {item[0] for item in edges}
        lowers = {item[1] for item in edges}
        selection_action_connected = (
            isinstance(topology, Mapping)
            and topology.get("selection_action_connected") is True
        )
        # Identify participants by the role each plays in the committed model, not by
        # this chain's reviewed element names.
        observation = system.observation or ""
        observing_component = next(
            (
                item for item in graph.components
                if observation in item.boolean_guarantee_concepts()
            ),
            None,
        )
        observing_link = next(
            (
                item for item in realization_links
                if observing_component
                and item.get("contract") == observing_component.name
            ),
            {},
        )
        # The arbiter is whichever contract the published arbitration behavior
        # realizes. The behavior is kept as structured provenance because the failure
        # router needs an existing state-def target before authorizing repair.
        arbitration_behavior = next(
            (
                behavior
                for item in graph.components
                if (
                    (behavior := _behavior_for_contract(
                        graph, realization_links, item.name
                    ))
                    and behavior.name == "SafetyResponseArbitration"
                )
            ),
            None,
        )
        arbiter_guarantees = {
            concept
            for item in graph.components
            if (
                (behavior := _behavior_for_contract(
                    graph, realization_links, item.name
                ))
                and arbitration_behavior is not None
                and behavior.name == arbitration_behavior.name
            )
            for concept in item.boolean_guarantee_concepts()
        }
        # a guarantee available at the boundary carries no timing segment: its
        # behaviour is a single initial state that establishes it
        boundary_components = [
            item for item in graph.components
            if item.timing_segment_required is False
            and item.boolean_guarantee_concepts()
        ]
        def _establishes_at_boundary(component: "Contract") -> bool:
            link = next(
                (
                    item for item in realization_links
                    if item.get("contract") == component.name
                ),
                {},
            )
            if link.get("realization_kind") == "INVARIANT":
                return (
                    link.get("status") == "PASS"
                    and link.get("continuous_guarantee") is True
                )
            behavior = _behavior_for_contract(
                graph, realization_links, component.name
            )
            if not behavior or behavior.transitions or not behavior.initial_state:
                return False
            entry = behavior.entry_actions.get(behavior.initial_state, "")
            return any(
                _flat_token(entry) == _flat_token(f"set{concept}")
                for concept in component.boolean_guarantee_concepts()
            )
        boundary_guarantees_established = all(
            _establishes_at_boundary(item) for item in boundary_components
        )
        # A timed chain apportions its deadline, so at least one participant is
        # available at the boundary and excluded from the apportionment; without one,
        # every component was charged time and the composition is not the pattern's.
        # An untimed chain apportions nothing, so no boundary component is an ordinary
        # shape, but any it declares is held to the same obligation.
        recovery_power_available_at_boundary = (
            boundary_guarantees_established
            if effective_pattern != _TIMED_PATTERN
            else bool(boundary_components) and boundary_guarantees_established
        )
        deployment_action_connected = bool(observation) and any(
            _flat_token(action) == _flat_token(f"set{observation}")
            for action in (observing_link.get("response_actions") or ())
        )
        observation_connected = bool(
            observation_links
            and all(item.get("status") == "PASS" for item in observation_links)
        )
        arbitration_elements = (
            [arbitration_behavior.name] if arbitration_behavior else []
        )
        boundary_behavior_elements = [
            behavior.name
            for component in boundary_components
            if (
                behavior := _behavior_for_contract(
                    graph, realization_links, component.name
                )
            )
        ]
        observation_behavior_elements = [
            str(observing_link.get("behavior"))
        ] if observing_link.get("behavior") else []
        obligation_affected_elements = {
            "selection_guarded_by_trigger": arbitration_elements,
            "competing_transitions_guarded": arbitration_elements,
            "selected_transition_reachable": arbitration_elements,
            "selection_action_connected": arbitration_elements,
            "recovery_power_available_at_boundary": boundary_behavior_elements,
            "deployment_action_connected": observation_behavior_elements,
        }
        # Each obligation is named so a failure says which fact is wrong; lumping
        # fourteen conditions under one message leaves the author nothing to repair.
        # Still one diagnostic, so the error count stays comparable with runs measured
        # before the message was itemised.
        # Every obligation below is internal: it relates the model to itself, never to
        # a reviewed answer. Whether the arbitration matches the gold response set,
        # ordering or winner is an accuracy question answered by
        # `ag_eval_semantics.priority_agreement` against frozen gold.
        edge_endpoints = {name for edge in edges for name in edge}
        obligations = (
            ("response_member_provenance",
             (
                 not require_priority_member_provenance
                 or (bool(members) and provenance_members == members)
             )),
            ("response_set_members",
             bool(members) and edge_endpoints <= members),
            ("precedence_edges",
             bool(edges) and lowers == (members - {selected} if selected else set())),
            ("single_highest_response", len(higher) == 1),
            ("selected_response", bool(selected) and selected in members),
            ("competing_transitions_guarded", guards == lowers),
            ("trigger_concept",
             bool(trigger)
             and trigger in {a.concept for a in system.assumptions}),
            # Cross-check only where the trigger is declared twice. A timed chain states
            # it in the priority contract and as the system contract's interval origin,
            # and the two must agree. An untimed chain declares no interval and states
            # the trigger once, so there is nothing to cross-check.
            ("trigger_matches_timing_origin",
             effective_pattern != _TIMED_PATTERN
             or trigger == (system.timing_origin or "")),
            ("selection_guarded_by_trigger", selection_when == trigger),
            ("selected_transition_reachable", reachable),
            ("selection_action_connected", selection_action_connected),
            # the arbiter both commands the downstream responder and records the
            # selection, which one guarantee cannot do
            ("arbiter_guarantees",
             len(arbiter_guarantees) >= 2
             and bool(
                 arbiter_guarantees
                 & {
                     a.concept for a in (
                         observing_component.assumptions
                         if observing_component else ()
                     )
                 }
             )),
            ("recovery_power_available_at_boundary",
             recovery_power_available_at_boundary),
            ("deployment_action_connected", deployment_action_connected),
            ("observation_connected", observation_connected),
        )
        unmet = [name for name, satisfied in obligations if not satisfied]
        if unmet:
            diagnostics.append(AGDiagnostic(
                CODE_PRIORITY_TOPOLOGY_INCOMPLETE,
                f"{system.name} priority/arbitration topology is incomplete or "
                f"internally inconsistent; unsatisfied: {', '.join(unmet)}",
                contract=system.name,
                subject=trigger or None,
                provenance={
                    "unsatisfied_obligations": list(unmet),
                    "obligation_affected_elements": {
                        name: list(obligation_affected_elements.get(name, ()))
                        for name in unmet
                        if name in obligation_affected_elements
                    },
                },
            ))
    elif effective_pattern in _INVARIANT_PATTERNS:
        if not graph.invariants:
            diagnostics.append(AGDiagnostic(
                CODE_INVARIANT_SEMANTICS_MISSING,
                f"{system.name} has no extracted invariant constraints",
                contract=system.name,
            ))
        else:
            selected_elements = set(graph.selected_model_elements)
            ids: set[str] = set()
            invalid_items: List[Dict[str, Any]] = []
            for invariant in graph.invariants:
                invariant_id = str(invariant.get("invariant_id") or "")
                identifiers = (
                    _ast_identifiers(invariant.get("trigger_or_antecedent_ast"))
                    | _ast_identifiers(invariant.get("required_consequent_ast"))
                )
                reasons = []
                if not invariant_id:
                    reasons.append("missing_invariant_id")
                if invariant_id in ids:
                    reasons.append("duplicate_invariant_id")
                if invariant.get("scope") != system.name:
                    reasons.append("wrong_system_scope")
                if invariant.get("source_kind") not in _INVARIANT_SOURCE_KINDS:
                    reasons.append("invalid_source_kind")
                if not str(invariant.get("source_id") or ""):
                    reasons.append("missing_source_id")
                if not identifiers:
                    reasons.append("empty_boolean_ast")
                undeclared = sorted(identifiers - selected_elements)
                if undeclared:
                    reasons.append("undeclared_model_elements")
                if reasons:
                    invalid_items.append({
                        "invariant_id": invariant_id or None,
                        "reasons": reasons,
                        "undeclared_model_elements": undeclared,
                    })
                ids.add(invariant_id)
            # Which invariants a requirement ought to state is an accuracy question the
            # evaluator answers against frozen gold. The pattern itself requires every
            # role it depends on to be filled: a de-energise-to-lock chain that never
            # says where power loss leads has not stated the pattern.
            #
            # Removing the per-requirement table without this lost detection: deleting a
            # required invariant passed.
            needed = PATTERN_INVARIANT_ROLES.get(effective_pattern)
            roles: Dict[str, Any] = {}
            missing_roles: List[str] = []
            if needed:
                roles = (
                    _startup_inhibit_roles(graph)
                    if effective_pattern == STARTUP_INHIBIT_PATTERN
                    else _invariant_roles(graph)
                )
                missing_roles = [
                    name for name in needed if not roles.get(name)
                ]
            if invalid_items or missing_roles:
                unsatisfied = [
                    *(
                        f"invariant:{item.get('invariant_id') or '<missing>'}:"
                        + ",".join(item["reasons"])
                        for item in invalid_items
                    ),
                    *(f"missing_role:{name}" for name in missing_roles),
                ]
                diagnostics.append(AGDiagnostic(
                    CODE_INVARIANT_SEMANTICS_INVALID,
                    f"{system.name} invariant semantics are incomplete or "
                    f"inconsistent; unsatisfied: {', '.join(unsatisfied)}",
                    contract=system.name,
                    provenance={
                        "unsatisfied_obligations": unsatisfied,
                        "invalid_invariants": invalid_items,
                        "missing_roles": missing_roles,
                        "derived_roles": {
                            name: sorted(value) if isinstance(value, set) else value
                            for name, value in roles.items()
                        },
                    },
                ))
        topology_obligations, topology_behavior = (
            _startup_inhibit_topology_obligations(graph, realization_links)
            if effective_pattern == STARTUP_INHIBIT_PATTERN
            else _locked_release_topology_obligations(graph, realization_links)
        )
        if effective_pattern == LOCKED_UNTIL_RELEASE_PATTERN:
            # "Default safe" means the locking component holds its guarantee with no
            # assumptions: a mechanism that assumes a condition is locked only while that
            # condition holds. Identified by the role its invariants give it, not by this
            # chain's contract name, and required to carry more than the lock so the
            # authorisation and de-energise obligations have an owner.
            locked_concept = _invariant_roles(graph).get("locked")
            mechanism = next(
                (
                    item for item in graph.components
                    if locked_concept
                    and locked_concept in item.boolean_guarantee_concepts()
                ),
                None,
            )
            topology_obligations.update({
                "default_safe_mechanism_present": bool(mechanism),
                "default_safe_no_assumptions": bool(
                    mechanism and not mechanism.assumptions
                ),
                "default_safe_responsibility_complete": bool(
                    mechanism
                    and len(mechanism.boolean_guarantee_concepts()) >= 3
                ),
            })
        unsatisfied_topology = [
            name
            for name, satisfied in topology_obligations.items()
            if not satisfied
        ]
        if unsatisfied_topology:
            affected = [topology_behavior] if topology_behavior else []
            diagnostics.append(AGDiagnostic(
                CODE_PATTERN_TOPOLOGY_INCOMPLETE,
                f"{system.name} {effective_pattern} topology is incomplete; "
                f"unsatisfied: {', '.join(unsatisfied_topology)}",
                contract=system.name,
                subject=effective_pattern,
                provenance={
                    "unsatisfied_obligations": unsatisfied_topology,
                    "obligation_affected_elements": {
                        name: list(affected)
                        for name in unsatisfied_topology
                    },
                },
            ))
    return diagnostics


def _check_sufficiency(
    graph: AGGraph, discharge_diags: List[AGDiagnostic], aliases: Mapping[str, str]
) -> List[AGDiagnostic]:
    diags: List[AGDiagnostic] = []
    if not graph.system or graph.system.observation is None:
        return diags
    produced = {
        _norm(concept, aliases)
        for c in graph.components for concept in c.boolean_guarantee_concepts()
    }
    def positive_obligations(
        node: Optional[Mapping[str, Any]], *, negated: bool = False
    ) -> set[str]:
        """Return positively asserted identifiers from the bounded Boolean AST.

        A condition in negative polarity (for example ``not powerOn`` in the
        implication form ``not powerOn or payloadLocked``) is a trigger, not a
        component guarantee that the decomposition must produce.
        """
        if not isinstance(node, Mapping):
            return set()
        if node.get("node") == "Identifier":
            return set() if negated else {str(node.get("name") or "")}
        if node.get("node") == "Not":
            return positive_obligations(
                node.get("expr"), negated=not negated
            )
        if node.get("node") in {"And", "Or"}:
            return set().union(*(
                positive_obligations(item, negated=negated)
                for item in (node.get("operands") or ())
            ))
        if node.get("node") == "Implies":
            return (
                positive_obligations(
                    node.get("antecedent"), negated=not negated
                )
                | positive_obligations(
                    node.get("consequent"), negated=negated
                )
            )
        return set()

    observed_guarantee = next(
        (
            guarantee for guarantee in graph.system.guarantees
            if guarantee.constraint_name == "g_observed"
        ),
        None,
    )
    required_observations = (
        {
            _norm(item, aliases)
            for item in positive_obligations(observed_guarantee.ast)
        }
        if observed_guarantee and observed_guarantee.ast
        else {_norm(graph.system.observation, aliases)}
    )
    blocked = any(
        d.code in (CODE_ASSUMPTION_UNDISCHARGED, CODE_CIRCULAR_ASSUMPTION)
        for d in discharge_diags
    )
    if not required_observations.issubset(produced) or blocked:
        prov = {"element_id": graph.system.element_id,
                "span": graph.system.span.as_dict() if graph.system.span else None}
        diags.append(AGDiagnostic(
            CODE_DECOMPOSITION_INSUFFICIENT,
            (f"component guarantees do not collectively support the system "
             f"observation(s) {sorted(required_observations)}"),
            contract=graph.system.name, subject=graph.system.observation,
            provenance=prov,
        ))
    return diags


def check_ag_graph(
    graph: AGGraph,
    *,
    aliases: Optional[Mapping[str, str]] = None,
    require_priority_member_provenance: bool = True,
) -> AGReport:
    """Run the bounded compositional A/G checks (§9) over an extracted graph."""
    alias_map = {k.strip().lower(): v.strip().lower() for k, v in (aliases or {}).items()}
    diagnostics: List[AGDiagnostic] = list(graph.parse_diagnostics)

    if (
        graph.system
        and graph.system.source_requirement
        and graph.system.source_requirement not in graph.source_requirement_ids
    ):
        diagnostics.append(AGDiagnostic(
            CODE_SOURCE_PROVENANCE_MISSING,
            f"authoritative source requirement {graph.system.source_requirement} "
            "is absent from the committed SysML model",
            contract=graph.system.name,
            subject=graph.system.source_requirement,
        ))

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

    discharge, discharge_edges, dis_diags = _check_discharge(graph, alias_map)
    diagnostics.extend(dis_diags)

    allocated = {e.dst for e in graph.edges if e.kind == "decomposes"}
    allocations = [
        {"owner": comp.owners[0], "contract": comp.name, "guarantee": concept}
        for comp in graph.components
        if comp.name in allocated and len(comp.owners) == 1
        for concept in comp.boolean_guarantee_concepts()
    ]

    timing, timing_diags = _check_timing(graph)
    diagnostics.extend(timing_diags)

    realization_links, realization_diags = _check_realization(graph, alias_map)
    diagnostics.extend(realization_diags)
    observation_links, observation_diags = _check_observation(graph)
    diagnostics.extend(observation_diags)
    diagnostics.extend(
        _check_profile_semantics(
            graph,
            realization_links,
            observation_links,
            require_priority_member_provenance=(
                require_priority_member_provenance
            ),
        )
    )

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

    timing_atomic = None
    if graph.system is not None and graph.system.timing_budget is not None:
        timing_atomic = {
            "origin": graph.system.timing_origin or "",
            "deadline": {
                "value": (
                    graph.system.timing_value_literal
                    or str(graph.system.timing_budget)
                ),
                "unit": "s",
            },
            "segments": [
                {
                    "component": component.name.removesuffix("Contract"),
                    "budget": {
                        "value": (
                            component.timing_value_literal
                            or str(component.timing_budget)
                        ),
                        "unit": "s",
                    },
                }
                for component in graph.components
                if component.timing_budget is not None
                and component.timing_segment_required is not False
            ],
        }

    priority = dict(graph.priority) if graph.priority else None
    if priority is not None:
        topology = dict(priority.get("arbitration_topology") or {})
        # the observing component is identified by producing the system's observed
        # guarantee, not by this chain's element names
        observed = (graph.system.observation or "") if graph.system else ""
        observing = next(
            (
                item for item in graph.components
                if observed in item.boolean_guarantee_concepts()
            ),
            None,
        )
        observing_link = next(
            (
                item for item in realization_links
                if observing and item.get("contract") == observing.name
            ),
            {},
        )
        topology["deployment_action_connected"] = bool(observed) and any(
            _flat_token(action) == _flat_token(f"set{observed}")
            for action in (observing_link.get("response_actions") or ())
        )
        topology["observation_connected"] = bool(
            observation_links
            and all(item.get("status") == "PASS" for item in observation_links)
        )
        priority["arbitration_topology"] = topology

    return AGReport(
        verdict=verdict,
        system_completeness=system_completeness,
        component_completeness=component_completeness,
        diagnostics=tuple(diagnostics),
        timing=timing,
        discharge=discharge,
        revision=graph.revision,
        allocations=tuple(allocations),
        discharge_edges=tuple(discharge_edges),
        realization_links=tuple(realization_links),
        observation_links=tuple(observation_links),
        source_requirement=(
            graph.system.source_requirement if graph.system else None
        ),
        timing_atomic=timing_atomic,
        priority=priority,
        invariants=graph.invariants,
        selected_model_elements=graph.selected_model_elements,
    )
