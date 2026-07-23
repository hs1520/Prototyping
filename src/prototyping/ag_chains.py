"""Student-approved bounded A/G chain library (Stage 2 decomposition, design §7).

Each entry is an implementation candidate, not independent evaluator gold or a
human-frozen architecture.  The student-approved decisions are recorded in
``docs/gold/STUDENT_DESIGN_DECISIONS.md`` and remain subject to independent review.
Three chains are encoded, each a distinct bounded safety pattern:
REQ_SAFE_005 (critical propulsion failure → parachute deployment, a timed
failsafe), REQ_SAFE_004 (power-on self-test → arming inhibit, a Boolean
startup-inhibit invariant), and REQ_SAFE_008 (stakeholder power-on default lock
plus student-derived authorised-unlock/de-energise rules — a
locked-until-authorised-release invariant). Adding a chain is a controlled design activity; the emitter
renders it into the model deterministically.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .ag_emitter import (
    AGAssumptionSpec,
    AGChainSpec,
    AGComponentSpec,
    AGInvariantSpec,
    AGPrioritySpec,
)
from ..utils.req_id import normalise_req_id


# Small typed-AST constructors keep the decision record and emitted SysML aligned.
def _id(name: str) -> dict[str, str]:
    return {"node": "Identifier", "name": name}


def _not(name: str) -> dict[str, object]:
    return {"node": "Not", "expr": _id(name)}


def _and(*items: dict[str, object]) -> dict[str, object]:
    return {"node": "And", "operands": list(items)}


# REQ_SAFE_005 — timing starts at the already-detected failure event. Detection
# latency is outside the chain; 0.10 + 0.35 = 0.45 s and the remaining 0.05 s is
# unallocated margin (STUDENT_DESIGN_DECISIONS §4).
REQ_SAFE_005_CHAIN = AGChainSpec(
    source_requirement="REQ_SAFE_005",
    package="REQ_SAFE_005_AG",
    system_contract="SystemParachuteContract",
    system_assumptions=("airborne", "criticalPropulsionFailureDetected"),
    observation="parachuteDeployed",
    deadline=0.5,
    components=(
        AGComponentSpec(
            name="SafetyResponseArbiterContract",
            owner_def="SafetyResponseArbiter",
            owner_usage="safetyResponseArbiter",
            guarantee="parachuteDeploymentCommand",
            behavior="SafetyResponseArbitration",
            trigger_signal="CriticalPropulsionFailureDetectedSignal",
            initial_state="awaitingFailure",
            response_state="parachuteSelected",
            response_action=(
                "setParachuteResponseSelectedAndIssueParachuteDeploymentCommand"
            ),
            additional_guarantees=("parachuteResponseSelected",),
            assumptions=(
                AGAssumptionSpec("airborne", environment=True),
                AGAssumptionSpec(
                    "criticalPropulsionFailureDetected", environment=True
                ),
            ),
            latency_budget=0.10,
            timing_segment_required=True,
        ),
        AGComponentSpec(
            name="RecoveryPowerSupplyContract",
            owner_def="RecoveryPowerSupply",
            owner_usage="recoveryPowerSupply",
            guarantee="recoveryActuationPowerAvailable",
            behavior="RecoveryPowerSupplyBehavior",
            trigger_signal="AirborneRecoveryModeSignal",
            initial_state="recoveryPowerAvailable",
            response_state="recoveryPowerAvailable",
            response_action="setRecoveryActuationPowerAvailable",
            assumptions=(
                AGAssumptionSpec("airborne", environment=True),
            ),
            timing_segment_required=False,
        ),
        AGComponentSpec(
            name="RecoverySystemContract",
            owner_def="RecoverySystem",
            owner_usage="recoverySystem",
            guarantee="parachuteDeployed",
            behavior="RecoverySystemBehavior",
            trigger_signal="ParachuteDeploymentCommandSignal",
            initial_state="stowed",
            response_state="deployed",
            response_action="setParachuteDeployed",
            assumptions=(
                AGAssumptionSpec("parachuteDeploymentCommand"),
                AGAssumptionSpec("recoveryActuationPowerAvailable"),
            ),
            latency_budget=0.35,
            timing_segment_required=True,
        ),
    ),
    timing_origin="criticalPropulsionFailureDetected",
    priority=AGPrioritySpec(
        response_set_id="FLIGHT_RESPONSES_V1",
        members=(
            "PARACHUTE_DEPLOYMENT",
            "CONTROLLED_BATTERY_LANDING",
            "COMMUNICATION_LOSS_SAFE_LANDING",
            "LOW_BATTERY_RETURN_TO_BASE",
        ),
        edges=(
            ("PARACHUTE_DEPLOYMENT", "CONTROLLED_BATTERY_LANDING"),
            ("PARACHUTE_DEPLOYMENT", "COMMUNICATION_LOSS_SAFE_LANDING"),
            ("PARACHUTE_DEPLOYMENT", "LOW_BATTERY_RETURN_TO_BASE"),
        ),
        trigger="criticalPropulsionFailureDetected",
        selected_response="PARACHUTE_DEPLOYMENT",
    ),
    selected_model_elements=(
        "airborne",
        "criticalPropulsionFailureDetected",
        "parachuteDeploymentCommand",
        "recoveryActuationPowerAvailable",
        "parachuteDeployed",
        "selectedResponse",
    ),
)

# REQ_SAFE_004 — startup self-test → arming inhibit. A Boolean *invariant*
# (failed self-test ↛ armed), not a timed chain: no latency budgets, and the
# StartupInhibit pattern carries no timing obligation. Second selected chain,
# exercising a structurally different property KIND + safety pattern.
REQ_SAFE_004_CHAIN = AGChainSpec(
    source_requirement="REQ_SAFE_004",
    package="REQ_SAFE_004_AG",
    system_contract="SystemArmingInhibitContract",
    system_assumptions=("powerOnSelfTestActive", "sensorFailureReported"),
    observation="armingTransitionInhibited and airborneTransitionInhibited",
    deadline=None,
    verification="ArmingInhibitVerification",
    pattern="STARTUP_INHIBIT",
    components=(
        AGComponentSpec(
            name="SelfTestStatusLatchContract",
            owner_def="SelfTestStatusLatch",
            owner_usage="selfTestStatusLatch",
            guarantee="startupInhibitActive",
            behavior="SelfTestStatusLatchBehavior",
            trigger_signal="SensorFailureReportedSignal",
            initial_state="poweredOff",
            response_state="startupInhibited",
            response_action="setStartupInhibitActive",
            assumptions=(
                AGAssumptionSpec("powerOnSelfTestActive", environment=True),
                AGAssumptionSpec("sensorFailureReported", environment=True),
            ),
        ),
        AGComponentSpec(
            name="ArmingAuthorityContract",
            owner_def="ArmingAuthority",
            owner_usage="armingAuthority",
            guarantee="armingTransitionInhibited",
            behavior="ArmingAuthorityBehavior",
            trigger_signal="StartupInhibitActiveSignal",
            initial_state="preArm",
            response_state="armingInhibited",
            response_action="setArmingTransitionInhibited",
            assumptions=(
                AGAssumptionSpec("startupInhibitActive"),
            ),
        ),
        AGComponentSpec(
            name="FlightModeAuthorityContract",
            owner_def="FlightModeAuthority",
            owner_usage="flightModeAuthority",
            guarantee="airborneTransitionInhibited",
            behavior="FlightModeAuthorityBehavior",
            trigger_signal="StartupInhibitActiveSignal",
            initial_state="grounded",
            response_state="airborneInhibited",
            response_action="setAirborneTransitionInhibited",
            assumptions=(
                AGAssumptionSpec("startupInhibitActive"),
            ),
        ),
    ),
    invariants=(
        AGInvariantSpec(
            invariant_id="SAFE004_STARTUP_INHIBIT",
            scope="SystemArmingInhibitContract",
            trigger_or_antecedent_ast=_and(
                _id("powerOnSelfTestActive"), _id("sensorFailureReported")
            ),
            required_consequent_ast=_and(_not("armed"), _not("airborne")),
            source_kind="STAKEHOLDER",
            source_id="REQ_SAFE_004",
        ),
        AGInvariantSpec(
            invariant_id="SAFE004_LATCH_EFFECT",
            scope="SystemArmingInhibitContract",
            trigger_or_antecedent_ast=_id("startupInhibitActive"),
            required_consequent_ast=_and(
                _id("armingTransitionInhibited"),
                _id("airborneTransitionInhibited"),
            ),
            source_kind="STUDENT_DERIVED_DESIGN_CONSTRAINT",
            source_id="SAFE004_LATCH_PROPAGATION_V1",
        ),
        AGInvariantSpec(
            invariant_id="SAFE004_LATCH_RESET_AFTER_PASS",
            scope="SystemArmingInhibitContract",
            trigger_or_antecedent_ast=_id("selfTestPassed"),
            required_consequent_ast=_not("startupInhibitActive"),
            source_kind="STUDENT_DERIVED_DESIGN_CONSTRAINT",
            source_id="SAFE004_LATCH_RESET_V1",
        ),
    ),
    selected_model_elements=(
        "powerOnSelfTestActive",
        "sensorFailureReported",
        "armed",
        "airborne",
        "startupInhibitActive",
        "armingTransitionInhibited",
        "airborneTransitionInhibited",
        "selfTestPassed",
    ),
    system_observation_concepts=(
        "armingTransitionInhibited",
        "airborneTransitionInhibited",
    ),
)

# REQ_SAFE_008 — the selected stakeholder source requires a locked power-on
# default before arming/flight authorisation. The guarded-unlock and
# de-energise-to-lock rules are separately identified student architecture
# constraints. Together they instantiate the third bounded pattern,
# LockedUntilAuthorisedRelease. No timing budget applies.
REQ_SAFE_008_CHAIN = AGChainSpec(
    source_requirement="REQ_SAFE_008",
    package="REQ_SAFE_008_AG",
    system_contract="SystemPayloadLockContract",
    system_assumptions=(),
    observation="not powerOnInitialisation or payloadLocked",
    deadline=None,
    verification="PayloadLockVerification",
    pattern="LOCKED_UNTIL_AUTHORISED_RELEASE",
    components=(
        AGComponentSpec(
            name="ReleaseCommandGatewayContract",
            owner_def="ReleaseCommandGateway",
            owner_usage="releaseCommandGateway",
            guarantee="authorisedReleaseCommandReceived",
            behavior="ReleaseCommandGatewayBehavior",
            trigger_signal="ReceivedReleaseCommandSignal",
            initial_state="awaitingAuthorisation",
            response_state="authorisationGranted",
            response_action="setAuthorisedReleaseCommandReceived",
            assumptions=(
                AGAssumptionSpec("receivedReleaseCommand", environment=True),
                AGAssumptionSpec("authorisationDataValid", environment=True),
            ),
        ),
        AGComponentSpec(
            name="PayloadLockMechanismContract",
            owner_def="PayloadLockMechanism",
            owner_usage="payloadLockMechanism",
            guarantee="payloadLocked",
            behavior="PayloadLockLifecycle",
            trigger_signal="PowerOnSignal",
            initial_state="lockedUnpowered",
            response_state="lockedPowered",
            response_action="setPayloadLockedForDeenergiseToLock",
            interface_inputs=(
                "powerOnEvent",
                "powerLostEvent",
                "authorisedReleaseCommandReceived",
            ),
            additional_guarantees=(
                "authorisedUnlockOnly",
                "deenergiseToLock",
            ),
        ),
    ),
    invariants=(
        AGInvariantSpec(
            invariant_id="SAFE008_POWER_ON_LOCKED",
            scope="SystemPayloadLockContract",
            trigger_or_antecedent_ast=_id("powerOnInitialisation"),
            required_consequent_ast=_id("payloadLocked"),
            source_kind="STAKEHOLDER",
            source_id="REQ_SAFE_008",
        ),
        AGInvariantSpec(
            invariant_id="SAFE008_UNLOCK_AUTHORISED",
            scope="SystemPayloadLockContract",
            trigger_or_antecedent_ast=_id("payloadUnlocked"),
            required_consequent_ast=_id("authorisedReleaseCommandReceived"),
            source_kind="STUDENT_DERIVED_DESIGN_CONSTRAINT",
            source_id="SAFE008_UNLOCK_AUTHORIZATION_V1",
        ),
        AGInvariantSpec(
            invariant_id="SAFE008_DEENERGISE_TO_LOCK",
            scope="SystemPayloadLockContract",
            trigger_or_antecedent_ast=_not("actuatorPowerAvailable"),
            required_consequent_ast=_id("payloadLocked"),
            source_kind="STUDENT_DERIVED_DESIGN_CONSTRAINT",
            source_id="SAFE008_DEENERGISE_TO_LOCK_V1",
        ),
    ),
    selected_model_elements=(
        "powerOnInitialisation",
        "payloadLocked",
        "payloadUnlocked",
        "authorisedReleaseCommandReceived",
        "actuatorPowerAvailable",
    ),
    system_observation_concepts=(
        "powerOnInitialisation",
        "payloadLocked",
    ),
)

_CHAIN_LIBRARY: Tuple[AGChainSpec, ...] = (
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_008_CHAIN,
)

_REQ_ID_RE = re.compile(r"^\s*(REQ[-_][A-Za-z]+[-_]\d+)")


def _requirement_ids(requirements: Iterable[str]) -> set:
    ids = set()
    for req in requirements or ():
        match = _REQ_ID_RE.match(str(req))
        if match:
            ids.add(normalise_req_id(match.group(1)))
    return ids


def select_ag_chains(requirements: Iterable[str]) -> Tuple[AGChainSpec, ...]:
    """Return student-selected chains whose source requirement is present."""
    present = _requirement_ids(requirements)
    return tuple(
        chain for chain in _CHAIN_LIBRARY
        if normalise_req_id(chain.source_requirement) in present
    )
