"""Shared fixture for the assume-guarantee tests: the reviewed gold record for REQ-SAFE-005."""
import hashlib

from src.prototyping.ag_evaluation import GOLD_ROLE


REQ_SAFE_005_GOLD = {
    "schema_version": "3.0",
    "artifact_role": GOLD_ROLE,
    "experiment_namespace": "BLACKBOARD_AG_V1",
    "status": "FROZEN",
    "reviewer": "independent-reviewer",
    "reviewed_date": "2026-07-22",
    "review_protocol": {
        "blind_to_runtime_verdict": True,
        "independent_human_review": True,
    },
    "chain_id": "REQ_SAFE_005",
    "source_requirement": "REQ_SAFE_005",
    "source_text": (
        "REQ-SAFE-005: independently authored evaluator fixture for the "
        "parachute-deployment chain."
    ),
    "source_digest": hashlib.sha256(
        (
            "REQ-SAFE-005: independently authored evaluator fixture for the "
            "parachute-deployment chain."
        ).encode("utf-8")
    ).hexdigest(),
    "requirement_set_digest": "a" * 64,
    "architecture_boundary_digest": "b" * 64,
    "allocations": [
        {
            "owner": "safetyResponseArbiter",
            "contract": "SafetyResponseArbiterContract",
            "guarantee": "parachuteDeploymentCommand",
        },
        {
            "owner": "safetyResponseArbiter",
            "contract": "SafetyResponseArbiterContract",
            "guarantee": "parachuteResponseSelected",
        },
        {
            "owner": "recoveryPowerSupply",
            "contract": "RecoveryPowerSupplyContract",
            "guarantee": "recoveryActuationPowerAvailable",
        },
        {
            "owner": "recoverySystem",
            "contract": "RecoverySystemContract",
            "guarantee": "parachuteDeployed",
        },
    ],
    "discharge_edges": [
        {"component": "SafetyResponseArbiterContract", "assumption": "airborne", "by": "environment"},
        {"component": "SafetyResponseArbiterContract", "assumption": "criticalPropulsionFailureDetected", "by": "environment"},
        {"component": "RecoveryPowerSupplyContract", "assumption": "airborne", "by": "environment"},
        {"component": "RecoverySystemContract", "assumption": "parachuteDeploymentCommand", "by": "SafetyResponseArbiterContract"},
        {"component": "RecoverySystemContract", "assumption": "recoveryActuationPowerAvailable", "by": "RecoveryPowerSupplyContract"},
    ],
    "timing": {
        "origin": "criticalPropulsionFailureDetected",
        "deadline": {"value": "0.5", "unit": "s"},
        "segments": [
            {
                "component": "SafetyResponseArbiter",
                "budget": {"value": "0.1", "unit": "s"},
            },
            {
                "component": "RecoverySystem",
                "budget": {"value": "0.35", "unit": "s"},
            },
        ],
    },
    "priority": {
        "response_set_id": "FLIGHT_RESPONSES_V1",
        "source_kind": "STUDENT_APPROVED_DECOMPOSITION",
        "source_id": "STUDENT_DESIGN_DECISIONS.md§4.3",
        "members": [
            "PARACHUTE_DEPLOYMENT",
            "CONTROLLED_BATTERY_LANDING",
            "COMMUNICATION_LOSS_SAFE_LANDING",
            "LOW_BATTERY_RETURN_TO_BASE",
        ],
        "edges": [
            {
                "higher": "PARACHUTE_DEPLOYMENT",
                "lower": "CONTROLLED_BATTERY_LANDING",
            },
            {
                "higher": "PARACHUTE_DEPLOYMENT",
                "lower": "COMMUNICATION_LOSS_SAFE_LANDING",
            },
            {
                "higher": "PARACHUTE_DEPLOYMENT",
                "lower": "LOW_BATTERY_RETURN_TO_BASE",
            },
        ],
        "trigger": "criticalPropulsionFailureDetected",
    },
}
