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
CODE_TIMING_BUDGET_MISSING = "TIMING_BUDGET_MISSING"
CODE_DECOMPOSITION_INSUFFICIENT = "DECOMPOSITION_INSUFFICIENT"
CODE_DECOMPOSITION_MISSING = "DECOMPOSITION_MISSING"
CODE_DISCHARGE_EDGE_MISSING = "DISCHARGE_EDGE_MISSING"
CODE_REALIZATION_MISSING = "REALIZATION_MISSING"
CODE_REALIZATION_UNREACHABLE = "REALIZATION_UNREACHABLE"
CODE_REALIZATION_TRIGGER_MISSING = "REALIZATION_TRIGGER_MISSING"
CODE_REALIZATION_ACTION_MISSING = "REALIZATION_ACTION_MISSING"
CODE_OBSERVATION_MISSING = "OBSERVATION_MISSING"
CODE_SOURCE_PROVENANCE_MISSING = "SOURCE_PROVENANCE_MISSING"

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
    owners: Tuple[str, ...] = ()
    source_requirement: Optional[str] = None
    declared_pattern: Optional[str] = None  # reviewed safety pattern (system only)

    def boolean_guarantee_concepts(self) -> Tuple[str, ...]:
        return tuple(g.concept for g in self.guarantees if g.kind == "boolean")


@dataclass(frozen=True)
class AGEdge:
    kind: str  # "decomposes" (§6.1); others reserved for later increments
    src: str
    dst: str
    subject: Optional[str] = None


@dataclass(frozen=True)
class BehaviorTransition:
    source: str
    trigger: str
    target: str


@dataclass(frozen=True)
class BehaviorRealization:
    name: str
    initial_state: Optional[str]
    transitions: Tuple[BehaviorTransition, ...] = ()
    entry_actions: Mapping[str, str] = field(default_factory=dict)
    element_id: Optional[str] = None
    span: Optional[Span] = None


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
    behaviors: Tuple[BehaviorRealization, ...] = ()
    verification_targets: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    source_requirement_ids: Tuple[str, ...] = ()

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
    allocations: Tuple[Dict[str, str], ...] = ()
    discharge_edges: Tuple[Dict[str, Any], ...] = ()
    realization_links: Tuple[Dict[str, Any], ...] = ()
    observation_links: Tuple[Dict[str, Any], ...] = ()
    source_requirement: Optional[str] = None
    checker_version: str = AG_CHECKER_VERSION

    def errors(self) -> Tuple[AGDiagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    def to_dict(self) -> Dict[str, Any]:
        # Shape of the derived, read-only ``ag_contract_graph.json`` audit view
        # (§14): every derived report cites the source model revision and digest
        # and the checker version, and can be regenerated deterministically. The
        # ``graph`` block carries the predicted structure so the independent
        # post-hoc evaluator can score it against gold without re-running the
        # checker (§13 separation).
        return {
            "artifact_role": "RUNTIME_A_G_PREDICTION",
            "evidence_role": "INTERVENTION_RUNTIME_CHECK",
            "producing_stage": "R2_COMPOSITIONAL_TRACE",
            "measurement_boundary": "INTERVENTION",
            "experiment_namespace": "BLACKBOARD_AG_V1",
            "configuration": "R2-BBAG",
            "checker_version": self.checker_version,
            "source_model_revision": self.revision,
            "source_model_digest": self.model_digest,
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
            },
            "source_requirement": self.source_requirement,
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
        reasons.append("no responsible owner (satisfy relationship)")
    if contract.role == "system" and contract.observation is None:
        reasons.append("no system observation concept")
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
    """Every component guarantee has exactly one responsible owner (§16)."""
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
    """Assumption discharge as a monotone availability fixpoint (§6.4).

    A component activates once every non-environment Boolean assumption concept is
    available; activation publishes its Boolean guarantee concepts. Seeded by the
    system assumptions (environment/trigger) plus any environment-marked component
    assumptions. This is the bounded A/G composition: a concept is either an
    explicit environment assumption or discharged by an upstream guarantee.
    """
    diags: List[AGDiagnostic] = []
    discharge: Dict[str, str] = {}
    discharge_edges: List[Dict[str, Any]] = []

    # available_by[concept] = "environment" (system/env-declared) or the producing
    # component, so each emitted discharge edge records what discharged it.
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

    # One discharge edge per Boolean assumption, recording status and source.
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
    missing_units = [
        c.name for c in graph.all_contracts()
        if c.timing_budget is not None and c.timing_unit is None
    ]
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
            c.name for c in graph.components if c.timing_budget is None
        ]
        if missing_budgets:
            diags.append(AGDiagnostic(
                CODE_TIMING_BUDGET_MISSING,
                "timed system contract has components without additive budgets: "
                + ", ".join(sorted(missing_budgets)),
            ))
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


def _flat_token(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _check_realization(
    graph: AGGraph, aliases: Mapping[str, str]
) -> Tuple[List[Dict[str, Any]], List[AGDiagnostic]]:
    """Require a reachable trigger -> response entry-action for each guarantee."""
    diags: List[AGDiagnostic] = []
    links: List[Dict[str, Any]] = []
    behaviors = {item.name: item for item in graph.behaviors}
    system_triggers = {
        _norm(a.concept, aliases) for a in (graph.system.assumptions if graph.system else ())
        if a.kind == "boolean"
    }
    for comp in graph.components:
        realization_edges = [
            edge for edge in graph.edges
            if edge.kind == "realized_by" and edge.src == comp.name
        ]
        if len(realization_edges) != 1 or realization_edges[0].dst not in behaviors:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_MISSING,
                f"{comp.name} has no unique state-behavior realization dependency",
                contract=comp.name,
            ))
            continue
        behavior = behaviors[realization_edges[0].dst]
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
        if not trigger_ok:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_TRIGGER_MISSING,
                f"{comp.name} realization has no reachable trigger compatible "
                "with its/system assumptions",
                contract=comp.name,
            ))

        guarantee_tokens = {
            _flat_token(g.concept) for g in comp.guarantees if g.kind == "boolean"
        }
        action_matches = [
            (state, action) for state, action in behavior.entry_actions.items()
            if state in reachable and any(
                token and token in _flat_token(action) for token in guarantee_tokens
            )
        ]
        if not reachable or not used_transitions:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_UNREACHABLE,
                f"{comp.name} realization has no reachable trigger-response path",
                contract=comp.name,
            ))
        if not action_matches:
            diags.append(AGDiagnostic(
                CODE_REALIZATION_ACTION_MISSING,
                f"{comp.name} has no reachable response entry action for its guarantee",
                contract=comp.name,
            ))
        links.append({
            "contract": comp.name,
            "owner": comp.owners[0] if len(comp.owners) == 1 else None,
            "behavior": behavior.name,
            "initial_state": behavior.initial_state,
            "reachable_states": sorted(reachable),
            "trigger_ok": trigger_ok,
            "response_actions": [action for _state, action in action_matches],
            "response_states": [state for state, _action in action_matches],
            "status": "PASS" if trigger_ok and action_matches and used_transitions else "FAIL",
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

    # Predicted guarantee allocation: each decomposed component owns the Boolean
    # guarantee concepts it publishes (§6.1 allocated_to), for gold F1 scoring.
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
        allocations=tuple(allocations),
        discharge_edges=tuple(discharge_edges),
        realization_links=tuple(realization_links),
        observation_links=tuple(observation_links),
        source_requirement=(
            graph.system.source_requirement if graph.system else None
        ),
    )
