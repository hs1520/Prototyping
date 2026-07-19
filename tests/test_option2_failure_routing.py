from __future__ import annotations

from src.prototyping.contract_types import INCOMPLETE, READY, UNSUPPORTED
from src.prototyping.failure_routing import (
    DESIGN_OR_REALIZATION_NONCOMPLIANCE,
    INFRASTRUCTURE_FAILURE,
    MODEL_SEMANTIC_FAULT,
    REQUIREMENT_AMBIGUITY,
    VERIFIER_LIMITATION,
    Diagnostic,
    classify_failure,
    classify_bundle_evidence,
)
from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.semantic_trace import build_semantic_trace
from examples.finalize_authoritative_run import _terminal_evidence


def _diagnostic() -> Diagnostic:
    return Diagnostic(
        "REQ_SAFE_005:O1:ACTION", "REQ_SAFE_005", "semantic_trace",
        "ACTION_PLATFORM_BINDING_MISMATCH",
        {"response": "deploy_parachute"}, {"command": "CMD_LAND"},
        affected_elements=("deployParachute",),
    )


def test_only_model_semantic_fault_authorises_repair():
    decision = classify_failure(
        "REQ_SAFE_005", contract_status=READY, diagnostics=[_diagnostic()]
    )
    assert decision.failure_class == MODEL_SEMANTIC_FAULT
    assert decision.repair_authorised
    assert decision.repair_scope == ("deployParachute",)


def test_classification_order_is_fail_closed():
    assert classify_failure(
        "R", contract_status=INCOMPLETE, diagnostics=[_diagnostic()]
    ).failure_class == REQUIREMENT_AMBIGUITY
    assert classify_failure(
        "R", contract_status=UNSUPPORTED
    ).failure_class == VERIFIER_LIMITATION
    assert classify_failure(
        "R", contract_status=READY, infrastructure_ok=False
    ).failure_class == INFRASTRUCTURE_FAILURE
    assert classify_failure(
        "R", contract_status=READY, oracle_executed=True, oracle_passed=False
    ).failure_class == DESIGN_OR_REALIZATION_NONCOMPLIANCE


def test_terminal_gazebo_failure_routes_conformant_design_invariant_to_dse():
    bundle = build_contract_bundle([
        "REQ-SAFE-007: The system shall maintain controlled flight following "
        "the failure of any single propulsion motor."
    ])
    model = """package D {
        requirement def REQ_SAFE_007 { doc /* motor-out controlled flight */ }
        part def PropulsionSystem { satisfy requirement REQ_SAFE_007; }
    }"""
    trace = build_semantic_trace(model, bundle, model_name="D")

    report = classify_bundle_evidence(
        bundle, trace, {
            "REQ_SAFE_007": {
                "verifier_available": True,
                "infrastructure_ok": True,
                "oracle_executed": True,
                "oracle_passed": False,
            }
        }
    )

    decision = report["decisions"][0]
    assert decision["failure_class"] == DESIGN_OR_REALIZATION_NONCOMPLIANCE
    assert decision["repair_authorised"] is False


def test_sitl_launch_failure_is_not_misclassified_as_oracle_failure():
    evidence = _terminal_evidence(
        {},
        {"safety_l2": [{
            "req_id": "REQ-SAFE-005", "passed": False,
            "message": "独立 SITL 启动失败", "note": "",
        }]},
        {"rows": []},
    )["REQ_SAFE_005"]

    assert evidence["infrastructure_ok"] is False
    assert evidence["oracle_executed"] is False


def test_gazebo_observer_gap_is_a_verifier_limitation_not_design_failure():
    evidence = _terminal_evidence(
        {
            "status": "PARTIAL",
            "req_results": [{
                "req_id": "REQ-SAFE-005", "status": "INCONCLUSIVE",
                "message": "parachute observer unavailable",
            }],
        },
        {},
        {"rows": []},
    )["REQ_SAFE_005"]

    assert evidence["verifier_available"] is False
    assert evidence["oracle_executed"] is False
