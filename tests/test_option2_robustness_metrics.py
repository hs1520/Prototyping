"""Robustness metrics for the bounded A/G assurance enhancement.

Honest "detection + bounded repair" measure: the A/G check surfaces incompleteness
a no-contract pipeline cannot see, and the dependency-closed loop auto-repairs the
model-semantic subset while routing integration gaps as explicit BLOCKED.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.prototyping.robustness_metrics import compute_robustness_metrics
from src.sysml.lite_model import build_lite_model
from src.utils.sysml_text_utils import get_sysml_text


def test_clean_run_shows_no_detected_errors_and_full_fulfilment():
    assurance = {
        "ag_contract_graph": {
            "verdict": "PASS",
            "component_completeness": {"C1": "READY", "C2": "READY"},
            "graph": {
                "discharge_edges": [{"by": "C1"}, {"by": "environment"}],
                "realization_links": [{"status": "PASS"}, {"status": "PASS"}],
            },
        },
        "pattern_conformance_report": {"verdict": "PASS"},
        "failure_diagnostics": {
            "failures": [],
            "analysis_history": [
                {"analysis_round": 0, "verdict": "PASS", "failure_ids": []}
            ],
        },
        "repair_decisions": {"decisions": []},
    }
    m = compute_robustness_metrics(assurance)
    assert m["detection"]["errors_detected"] == 0
    assert m["requirement_fulfillment_final"]["ag_verdict"] == "PASS"
    assert m["requirement_fulfillment_final"]["pattern_conformance"] == "PASS"
    assert m["completeness_final"]["components_ready"] == 1.0
    assert m["completeness_final"]["assumption_discharge"] == 1.0
    assert m["completeness_final"]["guarantee_realization"] == 1.0


def test_detection_and_bounded_repair_are_separated_honestly():
    # 3 detected: 1 model-semantic (auto-repaired), 2 integration gaps (BLOCKED)
    assurance = {
        "ag_contract_graph": {
            "verdict": "FAIL",
            "component_completeness": {"C1": "READY", "C2": "INCOMPLETE", "C3": "READY"},
            "graph": {
                "discharge_edges": [{"by": "C1"}, {"by": None}, {"by": "environment"}],
                "realization_links": [{"status": "PASS"}, {"status": "FAIL"}],
            },
        },
        "pattern_conformance_report": {"verdict": "FAIL"},
        "failure_diagnostics": {
            "failures": [
                {"classification": "MODEL_SEMANTIC_FAULT"},
                {"classification": "INTEGRATION_DECOMPOSITION_GAP"},
                {"classification": "INTEGRATION_DECOMPOSITION_GAP"},
            ],
            "analysis_history": [
                {"analysis_round": 0, "verdict": "FAIL", "failure_ids": ["a", "b", "c"]},
                {"analysis_round": 1, "verdict": "FAIL", "failure_ids": ["b", "c"]},
            ],
        },
        "repair_decisions": {"decisions": [
            {"status": "BLOCKED"}, {"status": "BLOCKED"},
        ]},
    }
    m = compute_robustness_metrics(assurance)
    assert m["detection"]["errors_detected"] == 3
    assert m["detection"]["by_class"] == {
        "MODEL_SEMANTIC_FAULT": 1, "INTEGRATION_DECOMPOSITION_GAP": 2,
    }
    # the enhancement is honestly split: 1 auto-repaired, 2 routed for humans
    assert m["repair"]["auto_repaired"] == 1
    assert m["repair"]["routed_blocked"] == 2
    assert m["robustness_delta"]["errors_before_assurance_repair"] == 3
    assert m["robustness_delta"]["errors_after_assurance_repair"] == 2
    assert m["measurement"].endswith("(NOT full auto-fix)")
    # completeness reflects the final model (C1 + environment discharged; 1/2 realised)
    assert m["completeness_final"]["components_ready"] == round(2 / 3, 4)
    assert m["completeness_final"]["assumption_discharge"] == round(2 / 3, 4)
    assert m["completeness_final"]["guarantee_realization"] == 0.5


def test_missing_sections_degrade_without_raising():
    m = compute_robustness_metrics({})
    assert m["detection"]["errors_detected"] == 0
    assert m["requirement_fulfillment_final"]["ag_verdict"] is None
    assert m["completeness_final"]["components_ready"] is None


def test_consumes_a_real_r2_assurance_trace():
    """Smoke test against the actual orchestrator _build_ag_trace output shape."""
    class _NoCallLLM:
        def complete(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("LLM should not be called")

    reqs = ["REQ-SAFE-005: deploy the parachute within 0.5 s of a critical "
            "propulsion failure."]
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm="R2-BBAG")
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", reqs)
    model = build_lite_model(
        "package DeliveryUAV { requirement def REQ_SAFE_005 { doc /* deploy the "
        "parachute within 0.5 s of a critical propulsion failure. */ } "
        "part def SafetyMonitor {} }", model_name="DeliveryUAV")
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="gen", metadata={}), model)
    final = orch._apply_ag_contract_layer(get_sysml_text(model), reqs)
    orch._commit_terminal_model(final, producer="test")
    assurance = orch._build_collaboration_artifacts(final)

    m = compute_robustness_metrics(assurance)
    # deterministic reviewed A/G -> a robust final model: PASS, no residual errors
    assert m["requirement_fulfillment_final"]["ag_verdict"] == "PASS"
    assert m["robustness_delta"]["errors_after_assurance_repair"] == 0
    assert m["completeness_final"]["guarantee_realization"] == 1.0
