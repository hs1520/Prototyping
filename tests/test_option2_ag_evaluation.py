"""Independent A/G gold evaluator — Increment 4 (gold F1) tests.

Enforces the §13/§16 separation (finding F3): accuracy/F1 is computed from an
archived checker prediction versus evaluator-only human gold, never by the checker
scoring itself. The gold below is authored by hand (evaluator-only), not derived
from the prediction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_evaluation import (
    GOLD_ROLE,
    PREDICTION_ROLE,
    evaluate_ag_against_gold,
)
from src.prototyping.ag_extractor import extract_ag_graph
from tests.test_option2_ag_checker import REQ_SAFE_005_SYSML


# Evaluator-only human gold for the REQ_SAFE_005 chain (authored blind to the
# pipeline verdict, NOT copied from the prediction).
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


def _prediction(sysml: str = REQ_SAFE_005_SYSML) -> dict:
    """An archived checker prediction (what the pipeline stored for this run)."""
    return check_ag_graph(extract_ag_graph(sysml, revision=3)).to_dict()


def test_perfect_prediction_scores_f1_one_against_gold():
    result = evaluate_ag_against_gold(_prediction(), REQ_SAFE_005_GOLD)
    assert result["guarantee_allocation"]["f1"] == 1.0
    assert result["assumption_discharge"]["f1"] == 1.0
    assert result["assumption_discharge"]["tp"] == 5
    assert result["self_scored"] is False
    assert result["evaluated_against"] == "independent_human_gold"
    assert result["chain_id"] == "REQ_SAFE_005"


def test_wrong_discharge_source_is_penalised():
    # Prediction where RecoveryPowerSupply emits nothing leaves the RecoverySystem
    # power assumption undischarged.
    broken = REQ_SAFE_005_SYSML.replace(
        "require constraint g_recoveryActuationPowerAvailable "
        "{ recoveryActuationPowerAvailable }", ""
    )
    result = evaluate_ag_against_gold(_prediction(broken), REQ_SAFE_005_GOLD)
    assert result["assumption_discharge"]["recall"] < 1.0
    assert result["assumption_discharge"]["fn"] >= 1
    # allocation also drops RecoveryPowerSupply's guarantee
    assert result["guarantee_allocation"]["recall"] < 1.0


def test_static_failure_class_is_rejected_in_favour_of_per_run_blind_labels():
    pred = _prediction()
    pred["failure_class"] = "NO_FAILURE"
    gold = {**REQ_SAFE_005_GOLD, "failure_class": "NO_FAILURE"}
    with pytest.raises(ValueError, match="per-run blind label"):
        evaluate_ag_against_gold(pred, gold)
    nested = {**REQ_SAFE_005_GOLD, "metadata": {"failure_class": "NO_FAILURE"}}
    with pytest.raises(ValueError, match="per-run blind label"):
        evaluate_ag_against_gold(pred, nested)


def test_role_guards_prevent_swapping_or_self_scoring():
    pred = _prediction()
    # gold passed where a prediction is expected
    with pytest.raises(ValueError, match="prediction artifact_role"):
        evaluate_ag_against_gold(REQ_SAFE_005_GOLD, REQ_SAFE_005_GOLD)
    # prediction passed where gold is expected (a checker cannot be its own gold)
    with pytest.raises(ValueError, match="gold artifact_role"):
        evaluate_ag_against_gold(pred, pred)


def test_evaluator_rejects_flag_shaped_but_structurally_invalid_gold():
    invalid = {
        **REQ_SAFE_005_GOLD,
        "timing": "not-an-atomic-timing-object",
    }
    with pytest.raises(ValueError, match="structural validation"):
        evaluate_ag_against_gold(_prediction(), invalid)

    wrong_chain = _prediction()
    wrong_chain["source_requirement"] = "REQ_SAFE_008"
    with pytest.raises(ValueError, match="does not match gold"):
        evaluate_ag_against_gold(wrong_chain, REQ_SAFE_005_GOLD)

    with pytest.raises(ValueError, match="ad-hoc"):
        evaluate_ag_against_gold(
            _prediction(),
            REQ_SAFE_005_GOLD,
            aliases={"wrongOwner": "safetyResponseArbiter"},
        )


def test_evaluator_never_imports_the_runtime_checker():
    # Structural F3 guarantee: the evaluator scores archived data only and must not
    # reach into the extractor/checker (which would let it score its own output).
    # Inspect import statements, not prose — the docstring explains the boundary.
    source = Path("src/prototyping/ag_evaluation.py").read_text(encoding="utf-8")
    imports = "\n".join(
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "ag_extractor" not in imports
    assert "ag_contracts" not in imports
    assert "extract_ag_graph" not in imports
    assert "check_ag_graph" not in imports
    assert PREDICTION_ROLE == "RUNTIME_A_G_PREDICTION"
    assert GOLD_ROLE == "EVALUATOR_GOLD"


def test_gold_role_matches_what_the_context_builder_rejects():
    # The same role string the ContextBuilder refuses as pipeline input (F3): gold
    # can be scored here but never fed back into generation/repair.
    from src.prototyping import context_builder as cb
    source = Path(cb.__file__).read_text(encoding="utf-8")
    assert "gold" in source.lower()


def test_missing_prediction_semantic_categories_are_reported_not_suppressed():
    prediction = _prediction()
    gold = {
        **REQ_SAFE_005_GOLD,
        "timing": {
            "origin": "criticalPropulsionFailureDetected",
            "deadline": {"value": "0.5", "unit": "s"},
            "segments": [{
                "component": "RecoverySystem",
                "budget": {"value": "0.35", "unit": "s"},
            }],
        },
        "priority": {
            "response_set_id": "FLIGHT_RESPONSES_V1",
            "source_kind": "STUDENT_APPROVED_DECOMPOSITION",
            "source_id": "STUDENT_DESIGN_DECISIONS.md§4.3",
            "members": ["PARACHUTE_DEPLOYMENT", "LOW_BATTERY_RETURN_TO_BASE"],
            "edges": [{
                "higher": "PARACHUTE_DEPLOYMENT",
                "lower": "LOW_BATTERY_RETURN_TO_BASE",
            }],
            "trigger": "criticalPropulsionFailureDetected",
        },
    }
    prediction["graph"].pop("timing", None)
    prediction["graph"].pop("priority", None)
    result = evaluate_ag_against_gold(prediction, gold)
    assert result["timing_agreement"]["segment_prf"]["recall"] == 0.0
    assert result["priority_agreement"]["arbitration_topology_conforms"] is False
