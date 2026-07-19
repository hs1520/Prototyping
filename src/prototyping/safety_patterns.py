"""Small, versioned library of machine-checkable safety behavior patterns.

Patterns are selected from typed contract semantics.  Their prompt text is
guidance; conformance is evaluated independently from extracted model traces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contract_types import (
    ContractBundle,
    READY,
    RequirementObligation,
    contract_bundle_from_dict,
)
from .platform_semantics import platform_binding


PATTERN_LIBRARY_VERSION = "option2-mvp-1"


@dataclass(frozen=True)
class TopologyRule:
    requires_trigger: bool = True
    requires_transition: bool = True
    requires_reachable_target: bool = True
    requires_entry_action: bool = True
    requires_initial_state: bool = True


@dataclass(frozen=True)
class InvariantRule:
    code: str
    description: str


@dataclass(frozen=True)
class SafetyPattern:
    pattern_id: str
    version: str
    applicable_contract_kinds: tuple[str, ...]
    trigger_concepts: tuple[str, ...]
    required_topology: TopologyRule
    required_response_concepts: tuple[str, ...]
    invariants: tuple[InvariantRule, ...]
    verification_capabilities: tuple[str, ...]
    prompt_guidance: str


@dataclass(frozen=True)
class PatternBinding:
    req_id: str
    obligation_id: str
    pattern_id: str
    pattern_version: str
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PATTERNS: tuple[SafetyPattern, ...] = (
    SafetyPattern(
        pattern_id="TriggeredFailsafeResponse",
        version="1.0",
        applicable_contract_kinds=("triggered_response", "timed_response"),
        trigger_concepts=(
            "critical_propulsion_failure", "sensor_self_test_failure",
            "gcs_link_absent", "battery_state_of_charge",
            "delivery_abort_condition", "contingency_condition",
        ),
        required_topology=TopologyRule(),
        required_response_concepts=(
            "deploy_parachute", "prevent_arming", "alert_gcs", "lock_payload",
            "return_to_base", "controlled_landing",
        ),
        invariants=(),
        verification_capabilities=("behavioral", "sitl"),
        prompt_guidance=(
            "Represent the real fault/event as a guard or accept trigger, add a "
            "reachable response state, and invoke the required engineering response "
            "from that state's entry action."
        ),
    ),
    SafetyPattern(
        pattern_id="TimedEmergencyActuation",
        version="1.0",
        applicable_contract_kinds=("timed_response",),
        trigger_concepts=("critical_propulsion_failure",),
        required_topology=TopologyRule(),
        required_response_concepts=("deploy_parachute",),
        invariants=(InvariantRule(
            "TIMING_BOUND_REPRESENTED",
            "The response latency upper bound is represented as an assertable constraint.",
        ),),
        verification_capabilities=("behavioral", "sitl", "gazebo"),
        prompt_guidance=(
            "Model a reachable emergency actuation state and preserve the contract "
            "deadline as a maximum/current latency pair plus an assert constraint."
        ),
    ),
    SafetyPattern(
        pattern_id="StartupInhibit",
        version="1.0",
        applicable_contract_kinds=("state_invariant", "triggered_response"),
        trigger_concepts=("sensor_self_test_failure", "power_on"),
        required_topology=TopologyRule(),
        required_response_concepts=("prevent_arming", "alert_gcs"),
        invariants=(InvariantRule(
            "FAILED_STARTUP_CANNOT_ARM",
            "A failed startup/self-test condition cannot reach an armed state.",
        ),),
        verification_capabilities=("behavioral", "sitl"),
        prompt_guidance=(
            "Use the self-test failure condition as the real trigger; the failed "
            "path must inhibit arming, and any separately required alert is a second "
            "obligation rather than a substitute for inhibit."
        ),
    ),
    SafetyPattern(
        pattern_id="LockedUntilAuthorisedRelease",
        version="1.0",
        applicable_contract_kinds=("state_invariant", "triggered_response"),
        trigger_concepts=(
            "delivery_abort_condition", "delivery_waypoint_proximity", "power_on",
        ),
        required_topology=TopologyRule(),
        required_response_concepts=("lock_payload", "release_payload"),
        invariants=(InvariantRule(
            "LOCKED_DEFAULT_OR_ABORT",
            "Payload remains locked under every lock condition stated by the contract.",
        ),),
        verification_capabilities=("behavioral", "sitl", "gazebo"),
        prompt_guidance=(
            "Represent a reachable locked state for every contract-stated abort/default "
            "condition. Use locked as the initial state only when the contract states a "
            "default. Release is reachable only through its fully qualified trigger."
        ),
    ),
)

_BY_ID = {pattern.pattern_id: pattern for pattern in PATTERNS}


def pattern_by_id(pattern_id: str) -> SafetyPattern | None:
    return _BY_ID.get(pattern_id)


def _applicable(pattern: SafetyPattern, obligation: RequirementObligation) -> bool:
    trigger = obligation.trigger.concept if obligation.trigger else ""
    response = obligation.response.concept if obligation.response else ""
    return (
        obligation.kind in pattern.applicable_contract_kinds
        and trigger in pattern.trigger_concepts
        and response in pattern.required_response_concepts
    )


def select_patterns(
    bundle: ContractBundle | Mapping[str, Any] | None,
) -> tuple[PatternBinding, ...]:
    typed = contract_bundle_from_dict(bundle)
    bindings: list[PatternBinding] = []
    for contract in typed.contracts:
        if contract.completeness != READY:
            continue
        for obligation in contract.obligations:
            for pattern in PATTERNS:
                if not _applicable(pattern, obligation):
                    continue
                bindings.append(PatternBinding(
                    req_id=contract.req_id,
                    obligation_id=obligation.obligation_id,
                    pattern_id=pattern.pattern_id,
                    pattern_version=pattern.version,
                    selection_reason=(
                        f"kind={obligation.kind}; trigger="
                        f"{obligation.trigger.concept if obligation.trigger else 'none'}; "
                        f"response={obligation.response.concept if obligation.response else 'none'}"
                    ),
                ))
    return tuple(bindings)


def bindings_to_dict(bindings: Iterable[PatternBinding]) -> dict[str, Any]:
    return {
        "library_version": PATTERN_LIBRARY_VERSION,
        "bindings": [binding.to_dict() for binding in bindings],
    }


def _relevant_contracts(
    bundle: ContractBundle | Mapping[str, Any] | None,
    req_ids: Iterable[str] | None = None,
) -> tuple[ContractBundle, list[Any]]:
    typed = contract_bundle_from_dict(bundle)
    all_req_ids = {contract.req_id for contract in typed.contracts}
    wanted = {item.upper().replace("-", "_") for item in (req_ids or all_req_ids)}
    relevant_contracts = [
        contract for contract in typed.contracts
        if contract.req_id in wanted and contract.completeness == READY
    ]
    return typed, relevant_contracts


def render_step_guidance(
    bundle: ContractBundle | Mapping[str, Any] | None,
    bindings: Iterable[PatternBinding],
    *,
    step: str,
    req_ids: Iterable[str] | None = None,
) -> str:
    """Render the minimum authoritative semantic slice for one generation step.

    Endpoint/component choices are intentionally not invented: contracts encode
    required engineering meaning, while the architecture supplies concrete owners
    and interfaces.
    """
    step = step.strip().lower()
    if step not in {"parts", "interfaces", "behavior", "assembly"}:
        raise ValueError(f"unknown generation guidance step: {step}")
    _typed, relevant_contracts = _relevant_contracts(bundle, req_ids)
    if not relevant_contracts:
        return ""
    wanted = {contract.req_id for contract in relevant_contracts}
    selected = [binding for binding in bindings if binding.req_id in wanted]

    if step == "parts":
        lines = [
            "TYPED CONTRACT GUIDANCE — PARTS (source-derived values are immutable):",
            "- Assign exactly one responsible satisfying owner per requirement.",
            "- Declare runtime trigger and current/max criterion attributes on the owner; do not add behavior here.",
        ]
        for contract in relevant_contracts:
            for obligation in contract.obligations:
                trigger = obligation.trigger
                criterion = obligation.criterion
                trigger_text = (
                    f"trigger={trigger.concept}; trigger_variable={trigger.variable or 'architecture_defined'}"
                    if trigger else "trigger=none"
                )
                criterion_text = ""
                if criterion:
                    criterion_text = (
                        f"; criterion={criterion.metric} {criterion.comparator} "
                        f"{criterion.value:g} {criterion.unit}"
                    )
                states = ",".join(contract.envelope.operating_states) or "not_specified"
                lines.append(
                    f"- {obligation.obligation_id}: {trigger_text}; "
                    f"response={obligation.response.concept if obligation.response else 'none'}"
                    f"{criterion_text}; operating_states={states}"
                )
        return "\n".join(lines)

    if step == "interfaces":
        lines = [
            "TYPED CONTRACT GUIDANCE — INTERFACES (do not invent endpoints or protocols):",
            "- Use the architecture and existing ports to choose sender/receiver ownership.",
            "- Expose only the data/command/observation concepts needed for the obligations below.",
        ]
        for contract in relevant_contracts:
            for obligation in contract.obligations:
                response = obligation.response.concept if obligation.response else "none"
                platform = platform_binding(response)
                commands = (
                    ",".join(platform.canonical_commands)
                    if platform and platform.canonical_commands else "none"
                )
                lines.append(
                    f"- {obligation.obligation_id}: trigger_concept="
                    f"{obligation.trigger.concept if obligation.trigger else 'none'}; "
                    f"response_concept={response}; "
                    f"observation_concept={obligation.verification_intent.observation_concept}; "
                    f"platform_semantic_tag={platform.semantic_tag if platform else 'none'}; "
                    f"canonical_commands={commands}; "
                    "endpoint_authority=architecture"
                )
        return "\n".join(lines)

    if step == "assembly":
        lines = [
            "TYPED CONTRACT GUIDANCE — ASSEMBLY (no requirement reinterpretation):",
            "- Preserve every source requirement and source-derived threshold exactly.",
            "- Place each satisfy link on the selected responsible owner and connect only existing compatible ports.",
            "- Preserve behavior trigger/response/criterion semantics; assembly must not substitute actions or commands.",
        ]
        for contract in relevant_contracts:
            for obligation in contract.obligations:
                criterion = obligation.criterion
                criterion_text = (
                    f"{criterion.metric} {criterion.comparator} "
                    f"{criterion.value:g} {criterion.unit}"
                    if criterion else "none"
                )
                lines.append(
                    f"- {obligation.obligation_id}: source_digest="
                    f"{contract.source_digest}; trigger="
                    f"{obligation.trigger.concept if obligation.trigger else 'none'}; "
                    f"response={obligation.response.concept if obligation.response else 'none'}; "
                    f"criterion={criterion_text}"
                )
        return "\n".join(lines)

    lines = [
        "CONTRACT AND SAFETY PATTERN GUIDANCE (acceptance values are immutable):"
    ]
    binding_map: dict[str, list[PatternBinding]] = {}
    for binding in selected:
        binding_map.setdefault(binding.obligation_id, []).append(binding)
    for contract in relevant_contracts:
        for obligation in contract.obligations:
            trigger = obligation.trigger.concept if obligation.trigger else "none"
            response = obligation.response.concept if obligation.response else "none"
            criterion = ""
            if obligation.criterion:
                criterion = (
                    f"; criterion={obligation.criterion.metric} "
                    f"{obligation.criterion.comparator} {obligation.criterion.value:g} "
                    f"{obligation.criterion.unit}"
                )
            lines.append(
                f"- {obligation.obligation_id}: trigger={trigger}; "
                f"response={response}{criterion}"
            )
            platform = platform_binding(response)
            if platform is not None:
                commands = ", ".join(platform.canonical_commands) or "none"
                lines.append(
                    f"  Platform binding: semantic_tag={platform.semantic_tag}; "
                    f"canonical_command={commands}; "
                    f"oracle={platform.observation_concept}. Use the canonical "
                    "command symbol when a command is required."
                )
            for binding in binding_map.get(obligation.obligation_id, ()):
                pattern = pattern_by_id(binding.pattern_id)
                if pattern:
                    lines.append(
                        f"  Apply {pattern.pattern_id}: {pattern.prompt_guidance}"
                    )
            lines.append(
                "  Do not substitute another response, threshold, command family, or oracle."
            )
    return "\n".join(lines)


def render_generation_guidance(
    bundle: ContractBundle | Mapping[str, Any] | None,
    bindings: Iterable[PatternBinding],
    *,
    req_ids: Iterable[str] | None = None,
) -> str:
    """Backward-compatible alias for the behavior-step guidance renderer."""
    return render_step_guidance(
        bundle, bindings, step="behavior", req_ids=req_ids
    )
