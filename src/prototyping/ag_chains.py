"""Reviewed bounded A/G chain library (Stage 2 decomposition, design §7).

Each entry is a human-reviewed A/G decomposition for one selected requirement
chain. The primary case is REQ_SAFE_005 (critical propulsion failure → parachute
deployment). Adding a chain is a reviewed activity (candidate doc §2); the
emitter renders it into the model deterministically.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .ag_emitter import AGAssumptionSpec, AGChainSpec, AGComponentSpec
from ..utils.req_id import normalise_req_id

# REQ_SAFE_005 — additive budget 0.05 + 0.10 + 0.35 = 0.5 s (design §7).
REQ_SAFE_005_CHAIN = AGChainSpec(
    source_requirement="REQ_SAFE_005",
    package="REQ_SAFE_005_AG",
    system_contract="SystemParachuteContract",
    system_assumptions=("airborne", "criticalPropulsionFailure"),
    observation="parachuteDeployed",
    deadline=0.5,
    components=(
        AGComponentSpec(
            name="PropulsionMonitorContract",
            owner_def="PropulsionMonitor",
            owner_usage="propulsionMonitor",
            guarantee="criticalFailureEvent",
            behavior="PropulsionMonitorBehavior",
            trigger_signal="CriticalPropulsionFailureSignal",
            initial_state="monitoring",
            response_state="failureDetected",
            response_action="setCriticalFailureEvent",
            assumptions=(AGAssumptionSpec("failureSensingAvailable", environment=True),),
            latency_budget=0.05,
        ),
        AGComponentSpec(
            name="SafetyMonitorContract",
            owner_def="SafetyMonitor",
            owner_usage="safetyMonitor",
            guarantee="parachuteCommand",
            behavior="SafetyMonitorBehavior",
            trigger_signal="CriticalFailureEventSignal",
            initial_state="monitoring",
            response_state="emergencyCommanded",
            response_action="setParachuteCommand",
            assumptions=(
                AGAssumptionSpec("airborne", environment=True),
                AGAssumptionSpec("criticalFailureEvent"),
            ),
            latency_budget=0.10,
        ),
        AGComponentSpec(
            name="RecoverySystemContract",
            owner_def="RecoverySystem",
            owner_usage="recoverySystem",
            guarantee="parachuteDeployed",
            behavior="RecoverySystemBehavior",
            trigger_signal="ParachuteCommandSignal",
            initial_state="stowed",
            response_state="deployed",
            response_action="setParachuteDeployed",
            assumptions=(
                AGAssumptionSpec("parachuteCommand"),
                AGAssumptionSpec("actuatorPower", environment=True),
            ),
            latency_budget=0.35,
        ),
    ),
)

_CHAIN_LIBRARY: Tuple[AGChainSpec, ...] = (REQ_SAFE_005_CHAIN,)

_REQ_ID_RE = re.compile(r"^\s*(REQ[-_][A-Za-z]+[-_]\d+)")


def _requirement_ids(requirements: Iterable[str]) -> set:
    ids = set()
    for req in requirements or ():
        match = _REQ_ID_RE.match(str(req))
        if match:
            ids.add(normalise_req_id(match.group(1)))
    return ids


def select_ag_chains(requirements: Iterable[str]) -> Tuple[AGChainSpec, ...]:
    """Return reviewed chains whose source requirement is present in the run."""
    present = _requirement_ids(requirements)
    return tuple(
        chain for chain in _CHAIN_LIBRARY
        if normalise_req_id(chain.source_requirement) in present
    )
