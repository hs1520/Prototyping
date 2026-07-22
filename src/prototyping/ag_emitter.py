"""Deterministic emitter for the bounded A/G contract layer (Stage 2-3, §12).

A reviewed A/G *decomposition* (the Stage-2 human-reviewed chain structure) is
rendered into valid SysML v2 and merged into the model, so the committed model —
the sole semantic authority (§6.2) — actually carries the contracts and R2-BBAG
traces are non-empty. Emission is deterministic and syntax-gate-valid; it uses
only the validated convention (`requirement def` + `assume`/`require constraint`
+ `dependency decomposition`) and invents no keywords. The emitted layer is the
inverse of ``ag_extractor`` and round-trips to a checker PASS for a sound chain.

This renders the *contract* layer only. Behaviour that realises each guarantee
(state machines, actions) remains ordinary LLM generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class AGAssumptionSpec:
    concept: str
    environment: bool = False


@dataclass(frozen=True)
class AGComponentSpec:
    name: str                       # component requirement-def name
    owner_def: str                  # responsible component type
    owner_usage: str                # concrete satisfying part usage
    guarantee: str                  # Boolean concept the component publishes
    behavior: str                   # realizing state definition
    trigger_signal: str             # accepted event/signal
    initial_state: str
    response_state: str
    response_action: str
    assumptions: Tuple[AGAssumptionSpec, ...] = ()
    latency_budget: Optional[float] = None


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
    # Reviewed safety-pattern kind for this chain (documentation/provenance). The
    # conformance checker independently infers the pattern from the emitted
    # topology (a timing budget ⇒ timed failsafe; none ⇒ startup-inhibit invariant).
    pattern: str = "TRIGGERED_TIMED_FAILSAFE_RESPONSE"


def _fmt(value: float) -> str:
    return repr(float(value))


def _dedup(concepts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(c for c in concepts if c))


def emit_ag_package(spec: AGChainSpec) -> str:
    """Render one reviewed A/G chain as a valid SysML v2 package."""
    out: list[str] = [f"package {spec.package} {{"]

    # System contract. The doc annotation carries the immutable source-requirement
    # provenance and the reviewed safety_pattern, so the committed model (the sole
    # authority) declares which bounded pattern the conformance checker must apply
    # — three untimed patterns cannot be told apart by topology alone.
    out.append(f"    requirement def {spec.system_contract} {{")
    out.append(
        f"        doc /* bounded A/G system contract for {spec.source_requirement}"
        f"; safety_pattern={spec.pattern} */"
    )
    for concept in _dedup([*spec.system_assumptions, spec.observation]):
        out.append(f"        attribute {concept} : Boolean;")
    if spec.deadline is not None:
        out.append(
            f"        attribute maxLatency : Real = {_fmt(spec.deadline)} [SI::s];"
        )
    for concept in spec.system_assumptions:
        out.append(f"        assume constraint a_{concept} {{ {concept} }}")
    out.append(f"        require constraint g_observed {{ {spec.observation} }}")
    out.append("    }")

    # Component contracts.
    for comp in spec.components:
        out.append(f"    requirement def {comp.name} {{")
        concepts = _dedup([*(a.concept for a in comp.assumptions), comp.guarantee])
        for concept in concepts:
            out.append(f"        attribute {concept} : Boolean;")
        if comp.latency_budget is not None:
            out.append(
                f"        attribute latencyBudget : Real = "
                f"{_fmt(comp.latency_budget)} [SI::s];"
            )
        for assumption in comp.assumptions:
            prefix = "env_" if assumption.environment else "a_"
            out.append(
                f"        assume constraint {prefix}{assumption.concept} "
                f"{{ {assumption.concept} }}"
            )
        out.append(
            f"        require constraint g_{comp.guarantee} {{ {comp.guarantee} }}"
        )
        out.append("    }")

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
    for comp in spec.components:
        out.append(f"    attribute def {comp.trigger_signal};")
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

    producers = {comp.guarantee: comp.name for comp in spec.components}
    system_environment = set(spec.system_assumptions)
    for comp in spec.components:
        for assumption in comp.assumptions:
            if assumption.environment or assumption.concept in system_environment:
                continue
            producer = producers.get(assumption.concept)
            if producer is not None:
                subject = assumption.concept[0].upper() + assumption.concept[1:]
                out.append(
                    f"    dependency discharge{subject} "
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
