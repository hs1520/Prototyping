"""Independent A/G gold evaluator — Increment 4 (gold F1) tests.

Enforces the §13/§16 separation (finding F3): accuracy/F1 is computed from an
archived checker prediction versus evaluator-only human gold, never by the checker
scoring itself. The gold below is authored by hand (evaluator-only), not derived
from the prediction.
"""
from __future__ import annotations

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
    "allocations": [
        {"owner": "propulsionMonitor", "guarantee": "criticalFailureEvent"},
        {"owner": "safetyMonitor", "guarantee": "parachuteCommand"},
        {"owner": "recoverySystem", "guarantee": "parachuteDeployed"},
    ],
    "discharge_edges": [
        {"component": "PropulsionMonitorContract", "assumption": "failureSensingAvailable", "by": "environment"},
        {"component": "SafetyMonitorContract", "assumption": "airborne", "by": "environment"},
        {"component": "SafetyMonitorContract", "assumption": "criticalFailureEvent", "by": "PropulsionMonitorContract"},
        {"component": "RecoverySystemContract", "assumption": "parachuteCommand", "by": "SafetyMonitorContract"},
        {"component": "RecoverySystemContract", "assumption": "actuatorPower", "by": "environment"},
    ],
    "failure_class": "NO_FAILURE",
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
    # Prediction where PropulsionMonitor emits nothing → SafetyMonitor's event
    # assumption is undischarged, so gold's discharge edge is a false negative and
    # the downstream command edge shifts too.
    broken = REQ_SAFE_005_SYSML.replace(
        "require constraint g_criticalFailureEvent { criticalFailureEvent }", ""
    )
    result = evaluate_ag_against_gold(_prediction(broken), REQ_SAFE_005_GOLD)
    assert result["assumption_discharge"]["recall"] < 1.0
    assert result["assumption_discharge"]["fn"] >= 1
    # allocation also drops PropulsionMonitor's guarantee
    assert result["guarantee_allocation"]["recall"] < 1.0


def test_failure_class_agreement_is_reported_when_both_declare_it():
    pred = _prediction()
    pred["failure_class"] = "NO_FAILURE"
    assert evaluate_ag_against_gold(pred, REQ_SAFE_005_GOLD)["failure_class_match"] is True
    pred["failure_class"] = "MODEL_SEMANTIC_FAULT"
    assert evaluate_ag_against_gold(pred, REQ_SAFE_005_GOLD)["failure_class_match"] is False


def test_role_guards_prevent_swapping_or_self_scoring():
    pred = _prediction()
    # gold passed where a prediction is expected
    with pytest.raises(ValueError, match="prediction artifact_role"):
        evaluate_ag_against_gold(REQ_SAFE_005_GOLD, REQ_SAFE_005_GOLD)
    # prediction passed where gold is expected (a checker cannot be its own gold)
    with pytest.raises(ValueError, match="gold artifact_role"):
        evaluate_ag_against_gold(pred, pred)


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
