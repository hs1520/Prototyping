"""Deterministic emitter for the bounded A/G contract layer (Stage 2-3, §12).

A student-approved A/G *decomposition candidate* is
rendered into valid SysML v2 and merged into the model, so the committed model —
the sole semantic authority (§6.2) — actually carries the contracts and R2-BBAG
traces are non-empty. Emission is deterministic and syntax-gate-valid; it uses
only the validated convention (`requirement def` + `assume`/`require constraint`
+ `dependency decomposition`) and invents no keywords. The emitted layer is the
inverse of ``ag_extractor`` and round-trips to a checker PASS for a sound chain.

For the three bounded MVP chains it also renders the selected minimal behaviour
topology used by the conformance checker. It does not claim formal proof or
physical verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class AGAssumptionSpec:
    concept: str
    environment: bool = False


@dataclass(frozen=True)
class AGRealizationPathSpec:
    """One approved guarantee-producing path in the bounded behavior profile.

    These facts describe the student-approved decomposition candidate used to
    prepare evaluator-gold drafts.  They are not runtime checker output and do
    not make a draft independent or frozen.
    """

    source: str
    trigger: Optional[str]
    target: str
    action: str
    guard: Optional[str] = None


@dataclass(frozen=True)
class AGComponentSpec:
    name: str                       # component requirement-def name
    owner_def: str                  # responsible component type
    owner_usage: str                # concrete satisfying part usage
    guarantee: str                  # Boolean concept the component publishes
    behavior: str                   # realizing state definition
    # ``None`` represents a continuously maintained state guarantee rather than
    # an event-triggered transition (for example recovery-power availability).
    trigger_signal: Optional[str]
    initial_state: str
    response_state: str
    response_action: str
    assumptions: Tuple[AGAssumptionSpec, ...] = ()
    interface_inputs: Tuple[str, ...] = ()
    additional_guarantees: Tuple[str, ...] = ()
    latency_budget: Optional[float] = None
    timing_segment_required: Optional[bool] = None
    #: Segments sharing a group are concurrent; the group contributes its maximum.
    #: None means "own group", i.e. serial — the composition addition assumed.
    timing_segment_group: Optional[int] = None
    realization_paths: Tuple[AGRealizationPathSpec, ...] = ()

    @property
    def guarantees(self) -> Tuple[str, ...]:
        return (self.guarantee, *self.additional_guarantees)


@dataclass(frozen=True)
class AGPrioritySpec:
    response_set_id: str
    members: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    trigger: str
    selected_response: str
    # Evaluator gold must identify the approved design input that supplies the
    # concrete response-set members; the stakeholder requirement only says
    # "all other safety responses" and does not enumerate them.
    source_kind: str
    source_id: str
    #: Per-member provenance: (response id, source kind, source element id).
    #: Runtime-generated catalogs use EXISTING_MODEL_BEHAVIOR; reviewed specs may
    #: retain their independently approved source for backward compatibility.
    member_provenance: Tuple[Tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class AGInvariantSpec:
    invariant_id: str
    scope: str
    trigger_or_antecedent_ast: Mapping[str, Any]
    required_consequent_ast: Mapping[str, Any]
    source_kind: str
    source_id: str


@dataclass(frozen=True)
class AGChainSpec:
    source_requirement: str         # immutable stakeholder req id (provenance)
    package: str                    # SysML package name for the contract layer
    system_contract: str            # system requirement-def name
    system_assumptions: Tuple[str, ...]  # Boolean environment/trigger concepts
    observation: str                # observed system-level Boolean concept
    deadline: Optional[float]       # system deadline (maxLatency), seconds
    components: Tuple[AGComponentSpec, ...]
    verification: str = "ParachuteDeploymentVerification"
    # Student-selected safety-pattern kind for this chain (provenance). The
    # conformance checker independently infers the pattern from the emitted
    # topology (a timing budget ⇒ timed failsafe; none ⇒ startup-inhibit invariant).
    pattern: str = "TRIGGERED_TIMED_FAILSAFE_RESPONSE"
    timing_origin: Optional[str] = None
    priority: Optional[AGPrioritySpec] = None
    invariants: Tuple[AGInvariantSpec, ...] = ()
    selected_model_elements: Tuple[str, ...] = ()
    system_observation_concepts: Tuple[str, ...] = ()
    #: Deadline the design deliberately does not apportion. Explicit, because
    #: "how much reserve a safety response keeps" is a design decision the
    #: requirement does not contain — the measured divergence between 0.1+0.35
    #: (0.05 s held back) and 0.2+0.3 (none) was exactly this decision, unstated.
    timing_margin: Optional[float] = None


def _fmt(value: float) -> str:
    return repr(float(value))


def _sysml_identifier(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    if not token or token[0].isdigit():
        token = f"id_{token}"
    return token


def _capitalise(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _dedup(concepts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(c for c in concepts if c))


def _render_ast(node: Mapping[str, Any]) -> str:
    kind = node.get("node")
    if kind == "Identifier":
        return str(node["name"])
    if kind == "Not":
        return f"not ({_render_ast(node['expr'])})"
    if kind in {"And", "Or"}:
        operator = " and " if kind == "And" else " or "
        return "(" + operator.join(
            _render_ast(item) for item in node.get("operands", ())
        ) + ")"
    if kind == "Implies":
        return (
            f"(not ({_render_ast(node['antecedent'])}) or "
            f"({_render_ast(node['consequent'])}))"
        )
    raise ValueError(f"unsupported invariant AST node {kind!r}")


def emit_ag_package(spec: AGChainSpec) -> str:
    """Render one student-approved A/G chain candidate as valid SysML v2."""
    out: list[str] = [f"package {spec.package} {{"]
    # Official release-corpus models explicitly import the standard-library
    # namespaces they use. Keeping these imports in every standalone package makes
    # raw Syside semantic diagnostics clean rather than relying on diagnostic
    # suppression in the project-wide legacy syntax wrapper.
    out.extend([
        "    private import ScalarValues::*;",
        "    private import ISQ::*;",
        "    private import SI::*;",
    ])

    # System contract. The doc annotation carries the immutable source-requirement
    # provenance and selected safety_pattern, so the committed model (the sole
    # authority) declares which bounded pattern the conformance checker must apply
    # — three untimed patterns cannot be told apart by topology alone.
    out.append(f"    requirement def {spec.system_contract} {{")
    out.append(
        f"        doc /* bounded A/G system contract for {spec.source_requirement}"
        f"; safety_pattern={spec.pattern}"
        + (f"; timing_origin={spec.timing_origin}" if spec.timing_origin else "")
        + " */"
    )
    observation_concepts = (
        spec.system_observation_concepts
        or (spec.observation,)
    )
    for concept in _dedup([
        *spec.system_assumptions,
        *observation_concepts,
        *spec.selected_model_elements,
    ]):
        out.append(f"        attribute {concept} : Boolean;")
    if spec.deadline is not None:
        out.append(
            "        attribute maxLatency : DurationValue = "
            f"{_fmt(spec.deadline)} [s];"
        )
    if spec.timing_margin is not None:
        out.append(
            "        attribute timingMargin : DurationValue = "
            f"{_fmt(spec.timing_margin)} [s];"
        )
    for concept in spec.system_assumptions:
        out.append(f"        assume constraint a_{concept} {{ {concept} }}")
    out.append(f"        require constraint g_observed {{ {spec.observation} }}")
    for invariant in spec.invariants:
        name = (
            f"inv__{_sysml_identifier(invariant.invariant_id)}"
            f"__source__{_sysml_identifier(invariant.source_id)}"
            f"__kind__{_sysml_identifier(invariant.source_kind)}"
        )
        implication = {
            "node": "Implies",
            "antecedent": invariant.trigger_or_antecedent_ast,
            "consequent": invariant.required_consequent_ast,
        }
        out.append(
            f"        require constraint {name} {{ {_render_ast(implication)} }}"
        )
    out.append("    }")

    # Component contracts.
    for comp in spec.components:
        out.append(f"    requirement def {comp.name} {{")
        concepts = _dedup([
            *(a.concept for a in comp.assumptions),
            *comp.interface_inputs,
            *comp.guarantees,
        ])
        for concept in concepts:
            out.append(f"        attribute {concept} : Boolean;")
        if comp.latency_budget is not None:
            out.append(
                f"        attribute latencyBudget : DurationValue = "
                f"{_fmt(comp.latency_budget)} [s];"
            )
        if comp.timing_segment_group is not None:
            out.append(
                "        attribute timingSegmentGroup : Integer = "
                f"{int(comp.timing_segment_group)};"
            )
        if comp.timing_segment_required is not None:
            literal = "true" if comp.timing_segment_required else "false"
            out.append(
                "        attribute timingSegmentRequired : Boolean = "
                f"{literal};"
            )
        for assumption in comp.assumptions:
            prefix = "env_" if assumption.environment else "a_"
            out.append(
                f"        assume constraint {prefix}{assumption.concept} "
                f"{{ {assumption.concept} }}"
            )
        for guarantee in comp.guarantees:
            out.append(
                f"        require constraint g_{guarantee} {{ {guarantee} }}"
            )
        out.append("    }")

    # Explicit priority semantics for the bounded SAFE_005 profile.  The enum,
    # requirement constraints, and guarded state transitions are all authoritative
    # SysML; the evaluator JSON is extracted from these constructs.
    if spec.priority is not None:
        priority = spec.priority
        enum_name = _sysml_identifier(priority.response_set_id)
        out.append(f"    enum def {enum_name} {{")
        for member in priority.members:
            out.append(f"        enum {_sysml_identifier(member)};")
        out.append("    }")
        out.append("    requirement def SafetyResponsePriorityContract {")
        out.append(
            "        doc /* bounded A/G evaluator semantic auxiliary; "
            f"response_set_id={priority.response_set_id} */"
        )
        member_provenance = priority.member_provenance or tuple(
            (member, priority.source_kind, priority.source_id)
            for member in priority.members
        )
        for member, source_kind, source_id in member_provenance:
            out.append(
                "        doc /* response_member="
                f"{_sysml_identifier(member)}; source_kind="
                f"{_sysml_identifier(source_kind)}; source_id="
                f"{_sysml_identifier(source_id)} */"
            )
        out.append(f"        attribute {priority.trigger} : Boolean;")
        out.append(f"        attribute selectedResponse : {enum_name};")
        out.append(
            "        assume constraint priorityTrigger "
            f"{{ {priority.trigger} }}"
        )
        out.append(
            "        require constraint selectHighestPriority "
            f"{{ selectedResponse == {enum_name}::"
            f"{_sysml_identifier(priority.selected_response)} }}"
        )
        for higher, lower in priority.edges:
            out.append(
                f"        require constraint precedence_{_sysml_identifier(higher)}"
                f"_over_{_sysml_identifier(lower)} "
                f"{{ not {priority.trigger} or selectedResponse != "
                f"{enum_name}::{_sysml_identifier(lower)} }}"
            )
        out.append("    }")

        # The arbitration behavior itself consumes this signal.  It therefore
        # needs a package-level definition even when a component spec names the
        # same trigger but delegates its behavior to this shared topology.
        # LLM-decided priority is a runtime catalog even when the architecture
        # boundary has no pre-existing member provenance.  The reviewed
        # deterministic fixture remains on its historical topology so independent
        # evaluator gold is not silently rewritten to follow an implementation
        # change.  The previous check used only ``priority.member_provenance`` and
        # therefore sent a valid decided spec down the legacy hard-coded
        # parachute path: its enum selected ``parachuteResponseSelected`` while
        # behavior targeted ``parachuteDeploymentSelected``.
        runtime_catalog_bound = (
            bool(priority.member_provenance)
            or priority.source_kind == "STUDENT_DERIVED_DESIGN_CONSTRAINT"
        )
        trigger_signal = (
            f"{_capitalise(priority.trigger)}Signal"
            if runtime_catalog_bound
            else "CriticalPropulsionFailureDetectedSignal"
        )
        out.append(
            f"    action def {_sysml_identifier(trigger_signal)} {{}}"
        )
        for lower in (edge[1] for edge in priority.edges):
            out.append(
                f"    action def "
                f"{_sysml_identifier(lower.title())}RequestSignal {{}}"
            )
        out.append("    state def SafetyResponseArbitration {")
        out.append(f"        attribute {priority.trigger} : Boolean;")
        out.append("        entry; then awaitingResponse;")
        out.append("        state awaitingResponse;")
        selected_token = (
            _sysml_identifier(priority.selected_response)
            if runtime_catalog_bound
            else "parachuteDeploymentSelected"
        )
        arbiter = next(
            (
                component for component in spec.components
                if component.behavior == "SafetyResponseArbitration"
            ),
            None,
        )
        selected_action = (
            arbiter.response_action
            if arbiter is not None
            else f"set{_capitalise(priority.selected_response)}"
        )
        selection_transition_name = (
            f"select{_capitalise(_sysml_identifier(priority.selected_response))}"
            if runtime_catalog_bound
            else "selectParachute"
        )
        out.append(
            f"        transition {selection_transition_name} "
            "first awaitingResponse "
            f"accept {_sysml_identifier(trigger_signal)} "
            f"if {priority.trigger} "
            f"then {selected_token};"
        )
        for _higher, lower in priority.edges:
            lower_token = _sysml_identifier(lower)
            out.append(
                f"        transition select{_sysml_identifier(lower.title())} "
                "first awaitingResponse "
                f"accept {_sysml_identifier(lower.title())}RequestSignal "
                f"if not {priority.trigger} then {lower_token};"
            )
        out.append(
            f"        state {selected_token} "
            f"{{ entry action {_sysml_identifier(selected_action)}; }}"
        )
        for _higher, lower in priority.edges:
            out.append(f"        state {_sysml_identifier(lower)};")
        out.append("    }")
        out.append(
            "    dependency realizeSafetyResponsePriority "
            "from SafetyResponsePriorityContract to SafetyResponseArbitration;"
        )

    # Explicit component owners and legal satisfaction relationships. Ownership
    # is never inferred from the name of a contract definition.
    for comp in spec.components:
        out.append(f"    part def {comp.owner_def};")
        out.append(f"    part {comp.owner_usage} : {comp.owner_def};")
        usage = comp.name[0].lower() + comp.name[1:]
        out.append(
            f"    satisfy requirement {usage} : {comp.name} by {comp.owner_usage};"
        )

    # Reachable trigger -> response-state -> entry-action realizations.
    emitted_signal_defs: set[str] = (
        {"CriticalPropulsionFailureDetectedSignal"}
        if spec.priority is not None
        else set()
    )
    for comp in spec.components:
        if comp.behavior == "SafetyResponseArbitration":
            continue
        if comp.name == "ReleaseCommandGatewayContract":
            for signal in (
                "ReceivedReleaseCommandSignal",
                "PowerOnSignal",
                "PowerLostSignal",
            ):
                if signal not in emitted_signal_defs:
                    out.append(f"    action def {signal} {{}}")
                    emitted_signal_defs.add(signal)
            out.extend([
                f"    state def {comp.behavior} {{",
                "        attribute authorisationDataValid : Boolean;",
                "        entry; then awaitingAuthorisation;",
                "        state awaitingAuthorisation;",
                "        transition acceptAuthorisedCommand "
                "first awaitingAuthorisation "
                "accept ReceivedReleaseCommandSignal "
                "if authorisationDataValid then authorisationGranted;",
                "        state authorisationGranted "
                "{ entry action setAuthorisedReleaseCommandReceived; }",
                "        transition clearOnPowerLoss first authorisationGranted "
                "accept PowerLostSignal then awaitingAuthorisation;",
                "        transition clearOnNewPowerCycle first authorisationGranted "
                "accept PowerOnSignal then awaitingAuthorisation;",
                "    }",
            ])
            continue
        if comp.name == "SelfTestStatusLatchContract":
            for signal in (
                "PowerOnSignal",
                "SensorFailureReportedSignal",
                "SelfTestPassedSignal",
                "PowerCycleSignal",
            ):
                if signal not in emitted_signal_defs:
                    out.append(f"    action def {signal} {{}}")
                    emitted_signal_defs.add(signal)
            out.extend([
                f"    state def {comp.behavior} {{",
                "        entry; then poweredOff;",
                "        state poweredOff;",
                "        transition beginSelfTest first poweredOff "
                "accept PowerOnSignal then selfTesting;",
                "        state selfTesting;",
                "        transition latchFailure first selfTesting "
                "accept SensorFailureReportedSignal then startupInhibited;",
                "        transition completePassingSelfTest first selfTesting "
                "accept SelfTestPassedSignal then selfTestPassed;",
                "        state selfTestPassed "
                "{ entry action clearStartupInhibitActive; }",
                "        state startupInhibited "
                "{ entry action setStartupInhibitActive; }",
                "        transition resetAfterPowerCycle first startupInhibited "
                "accept PowerCycleSignal then poweredOff;",
                "    }",
            ])
            continue
        if comp.name == "PayloadLockMechanismContract":
            for signal in (
                "PowerOnSignal",
                "PowerLostSignal",
                "AuthorisedReleaseCommandReceivedSignal",
            ):
                if signal not in emitted_signal_defs:
                    out.append(f"    action def {signal} {{}}")
                    emitted_signal_defs.add(signal)
            out.extend([
                f"    state def {comp.behavior} {{",
                "        entry; then lockedUnpowered;",
                "        state lockedUnpowered "
                "{ entry action setPayloadLockedForDeenergiseToLock; }",
                "        transition powerApplied first lockedUnpowered "
                "accept PowerOnSignal then lockedPowered;",
                "        state lockedPowered "
                "{ entry action maintainPayloadLocked; }",
                "        transition authorisedUnlock first lockedPowered "
                "accept AuthorisedReleaseCommandReceivedSignal "
                "then unlockedPowered;",
                "        state unlockedPowered "
                "{ entry action enforceAuthorisedUnlockOnly; }",
                "        transition powerLostLocks first unlockedPowered "
                "accept PowerLostSignal then lockedUnpowered;",
                "    }",
            ])
            continue
        if comp.name == "RecoveryPowerSupplyContract":
            out.extend([
                f"    state def {comp.behavior} {{",
                "        entry; then recoveryPowerAvailable;",
                "        state recoveryPowerAvailable "
                "{ entry action setRecoveryActuationPowerAvailable; }",
                "    }",
            ])
            continue
        if comp.trigger_signal not in emitted_signal_defs:
            out.append(f"    action def {comp.trigger_signal} {{}}")
            emitted_signal_defs.add(comp.trigger_signal)
        out.append(f"    state def {comp.behavior} {{")
        out.append(f"        entry; then {comp.initial_state};")
        out.append(f"        state {comp.initial_state};")
        out.append(
            f"        transition on{comp.trigger_signal} "
            f"first {comp.initial_state} accept {comp.trigger_signal} "
            f"then {comp.response_state};"
        )
        out.append(
            f"        state {comp.response_state} "
            f"{{ entry action {comp.response_action}; }}"
        )
        out.append("    }")

    # Decomposition edges (unique names → no namespace-shadowing warning).
    for comp in spec.components:
        out.append(
            f"    dependency decompose{comp.name} "
            f"from {spec.system_contract} to {comp.name};"
        )
        out.append(
            f"    dependency realize{comp.name} "
            f"from {comp.name} to {comp.behavior};"
        )

    producers = {
        guarantee: comp.name
        for comp in spec.components
        for guarantee in comp.guarantees
    }
    system_environment = set(spec.system_assumptions)
    for comp in spec.components:
        for assumption in comp.assumptions:
            if assumption.environment or assumption.concept in system_environment:
                continue
            producer = producers.get(assumption.concept)
            if producer is not None:
                subject = assumption.concept[0].upper() + assumption.concept[1:]
                out.append(
                    f"    dependency discharge{subject}__to__{comp.name} "
                    f"from {producer} to {comp.name};"
                )

    out.append(f"    verification def {spec.verification} {{")
    out.append("        objective deploymentObservation {")
    out.append(
        f"            verify requirement observedContract : {spec.system_contract};"
    )
    out.append("        }")
    out.append("    }")
    out.append(
        f"    dependency observe{spec.system_contract} "
        f"from {spec.system_contract} to {spec.verification};"
    )
    out.append("}")
    return "\n".join(out)


def merge_ag_contracts(model_text: str, specs: Iterable[AGChainSpec]) -> str:
    """Append the emitted A/G contract packages to the model text.

    Multiple top-level packages are valid SysML; the extractor scans the whole
    text. The base model is preserved verbatim.
    """
    packages = [emit_ag_package(spec) for spec in specs]
    if not packages:
        return model_text
    return (model_text or "").rstrip() + "\n\n" + "\n\n".join(packages) + "\n"
