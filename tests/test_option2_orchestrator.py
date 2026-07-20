from __future__ import annotations

from src.agents.orchestrator import Orchestrator
from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.requirement_inputs import (
    APPROVED_CONTRACT_PROTOCOL_VERSION,
    build_frozen_requirement_set,
    requirement_set_digest,
)
from src.prototyping.robustness import RobustnessOptions
from src.prototyping.safety_patterns import select_patterns
from src.sysml.lite_model import build_lite_model
from types import SimpleNamespace


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("LLM should not be called")


_REQ = (
    "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
    "within 0.5 seconds."
)

_MODEL = """package D {
    action def CMD_LAND { }
    requirement def REQ_SAFE_005 {
        doc /* Critical propulsion failure shall deploy the parachute within 0.5 seconds. */
    }
    part def SafetyMonitor {
        out port chuteCmd : DataPort;
        attribute propulsionCriticalFailure : Boolean = false;
        satisfy requirement REQ_SAFE_005;
        action def deployParachute { send CMD_LAND() to chuteCmd; }
        state def Monitor {
            state nominal;
            state chute { entry action deploy : deployParachute; }
            transition initial then nominal;
            transition failed first nominal if propulsionCriticalFailure then chute;
        }
    }
}"""


def test_b0_does_not_build_option2_artifacts():
    orchestrator = Orchestrator(_NoCallLLM(), robustness_options=RobustnessOptions.b0())
    assert orchestrator._build_robustness_artifacts(_MODEL, "D") == {}


def test_b2_builds_all_declared_run_artifacts_and_routes_model_fault():
    orchestrator = Orchestrator(_NoCallLLM(), robustness_options=RobustnessOptions.b2())
    bundle = build_contract_bundle([_REQ])
    orchestrator.last_contract_bundle = bundle
    orchestrator.last_pattern_bindings = select_patterns(bundle)
    orchestrator.last_requirement_semantic_analysis = {
        "contract_summary": {
            "counts": {"READY": 1, "INCOMPLETE": 0, "UNSUPPORTED": 0}
        }
    }

    artifacts = orchestrator._build_robustness_artifacts(_MODEL, "D")

    assert {
        "requirement_contracts", "safety_pattern_bindings",
        "semantic_trace_report", "failure_diagnostics",
        "repair_decisions", "robustness_metrics",
    } <= artifacts.keys()
    assert artifacts["semantic_trace_report"]["model_digest"]
    assert artifacts["failure_diagnostics"]["artifact_schema_version"] == "1.0"
    assert artifacts["failure_diagnostics"]["requirement_source_digests"]
    assert artifacts["repair_decisions"]["platform_binding_version"]
    decision = artifacts["repair_decisions"]["decisions"][0]
    assert decision["failure_class"] == "MODEL_SEMANTIC_FAULT"
    assert decision["repair_authorised"] is True


def test_b1_records_contract_trace_but_not_b2_routing_or_pattern_intervention():
    orchestrator = Orchestrator(_NoCallLLM(), robustness_options=RobustnessOptions.b1())
    bundle = build_contract_bundle([_REQ])
    orchestrator.last_contract_bundle = bundle
    orchestrator.last_pattern_bindings = select_patterns(bundle)
    orchestrator.last_requirement_semantic_analysis = {
        "contract_summary": {
            "counts": {"READY": 1, "INCOMPLETE": 0, "UNSUPPORTED": 0}
        }
    }

    artifacts = orchestrator._build_robustness_artifacts(_MODEL, "D")

    assert "requirement_contracts" in artifacts
    assert "semantic_trace_report" in artifacts
    assert "failure_diagnostics" in artifacts
    assert "safety_pattern_bindings" not in artifacts
    assert "repair_decisions" not in artifacts
    assert artifacts["robustness_metrics"]["failure_route_counts"] == {}


def test_semantic_repair_budget_stops_before_a_third_llm_attempt():
    orchestrator = Orchestrator(_NoCallLLM(), robustness_options=RobustnessOptions.b2())
    orchestrator.last_semantic_repair_attempts = [{"accepted": False}, {"accepted": False}]
    model = build_lite_model(_MODEL, model_name="D")

    candidate = orchestrator._generate_refinement_candidate(
        model,
        _MODEL,
        SimpleNamespace(
            issues=["[SEMANTIC-TRACE] REQ_SAFE_005 mismatch"],
            recommendations=[],
        ),
        "repair only the mismatch",
        [_REQ],
    )

    assert candidate is None
    assert len(orchestrator.last_semantic_repair_attempts) == 2


def test_empty_semantic_repair_packet_blocks_before_llm(monkeypatch):
    orchestrator = Orchestrator(
        _NoCallLLM(), robustness_options=RobustnessOptions.b2(),
        use_surgical_refinement=True,
    )
    bundle = build_contract_bundle([_REQ])
    orchestrator.last_contract_bundle = bundle
    orchestrator.last_pattern_bindings = select_patterns(bundle)
    monkeypatch.setattr(
        "src.prototyping.repair_packet.build_scoped_repair_packet",
        lambda *args, **kwargs: {},
    )

    candidate = orchestrator._generate_refinement_candidate(
        build_lite_model(_MODEL, model_name="D"),
        _MODEL,
        SimpleNamespace(
            issues=["[SEMANTIC-TRACE] REQ_SAFE_005 mismatch"],
            recommendations=[],
        ),
        "repair only the mismatch",
        [_REQ],
    )

    assert candidate is None
    attempt = orchestrator.last_semantic_repair_attempts[0]
    assert attempt["accepted"] is False
    assert attempt["llm_invoked"] is False
    assert attempt["reason"] == "repair_not_authorised_or_packet_empty"


def test_b2_semantic_surgery_records_an_accepted_diagnostic_reduction(monkeypatch):
    broken = _MODEL.replace("package D {", "package D {\n    port def DataPort;").replace(
        "attribute propulsionCriticalFailure : Boolean = false;",
        "attribute propulsionCriticalFailure : Boolean = false;\n"
        "        attribute maxParachuteLatency : Real = 0.5 [s];\n"
        "        attribute currentParachuteLatency : Real = 0.0 [s];\n"
        "        assert constraint parachuteLatencyBound {\n"
        "            currentParachuteLatency <= maxParachuteLatency\n"
        "        }",
    )
    fixed = broken.replace("CMD_LAND", "MAV_CMD_DO_PARACHUTE")
    orchestrator = Orchestrator(
        _NoCallLLM(), robustness_options=RobustnessOptions.b2(),
        use_surgical_refinement=True, max_iterations=2,
    )
    bundle = build_contract_bundle([_REQ])
    orchestrator.last_contract_bundle = bundle
    orchestrator.last_pattern_bindings = select_patterns(bundle)
    captured = {}

    def _fake_surgery(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(merged_text=fixed, summary=lambda: "fixed")

    monkeypatch.setattr(
        "src.agents.surgical_refiner.attempt_surgical_refinement",
        _fake_surgery,
    )

    candidate = orchestrator._generate_refinement_candidate(
        build_lite_model(broken, model_name="D"),
        broken,
        SimpleNamespace(
            issues=[
                "[SEMANTIC-TRACE] REQ_SAFE_005 command mismatch",
                "unrelated structural recommendation",
            ],
            recommendations=["add an unrelated cosmetic annotation"],
        ),
        "repair the command only",
        [_REQ],
    )
    simulation = SimpleNamespace(
        passed_scenarios=lambda: [], failed_scenarios=lambda: []
    )
    monkeypatch.setattr(
        orchestrator, "_run_simulation", lambda *args, **kwargs: simulation
    )
    orchestrator.evaluator = SimpleNamespace(
        evaluate=lambda **kwargs: SimpleNamespace(weighted_total=0.9)
    )

    accepted = orchestrator._accept_refinement_candidate(
        candidate=candidate,
        current_sysml=broken,
        rule_score=0.9,
        dse_best_config=None,
        requirements=[_REQ],
        connectivity_floor=False,
    )

    assert accepted is candidate
    assert captured["issues"] == [
        "[SEMANTIC-TRACE] REQ_SAFE_005 command mismatch"
    ]
    assert "unrelated structural recommendation" not in captured["feedback"]
    assert "cosmetic annotation" not in captured["feedback"]
    assert captured["repair_packet"]["scope"]["req_ids"] == ["REQ_SAFE_005"]
    assert captured["repair_packet"]["contracts"][0]["source_text"] == _REQ
    attempt = orchestrator.last_semantic_repair_attempts[0]
    assert {
        key: attempt[key]
        for key in (
            "attempt", "before_diagnostic_ids", "after_diagnostic_ids",
            "removed_diagnostic_ids", "new_diagnostic_ids", "syntax_passed",
            "simulation_regressions", "score_preserved", "accepted", "reason",
        )
    } == {
        "attempt": 1,
        "before_diagnostic_ids": [
            "REQ_SAFE_005:REQ_SAFE_005.O1:ACTION_PLATFORM_BINDING_MISMATCH"
        ],
        "after_diagnostic_ids": [],
        "removed_diagnostic_ids": [
            "REQ_SAFE_005:REQ_SAFE_005.O1:ACTION_PLATFORM_BINDING_MISMATCH"
        ],
        "new_diagnostic_ids": [],
        "syntax_passed": True,
        "simulation_regressions": [],
        "score_preserved": True,
        "accepted": True,
        "reason": "accepted_all_preservation_gates",
    }
    assert attempt["repair_packet_req_ids"] == ["REQ_SAFE_005"]
    assert len(attempt["repair_packet_digest"]) == 64
    assert attempt["repair_packet_contract_count"] == 1
    assert attempt["repair_packet_trace_count"] == 1
    assert attempt["repair_packet_platform_binding_count"] == 1


def test_frozen_requirement_input_skips_llm_extraction_and_builds_contracts():
    orchestrator = Orchestrator(
        _NoCallLLM(), robustness_options=RobustnessOptions.b2()
    )
    artifact = build_frozen_requirement_set([_REQ], source="test")

    requirements = orchestrator._use_frozen_requirements(artifact)

    assert requirements == [_REQ]
    assert orchestrator.last_requirement_input["mode"] == "frozen"
    assert orchestrator.last_requirement_input["frozen"] is True
    assert orchestrator.last_contract_bundle.contracts[0].req_id == "REQ_SAFE_005"


def test_approved_contract_bundle_is_distinct_from_gold_and_installed():
    orchestrator = Orchestrator(
        _NoCallLLM(), robustness_options=RobustnessOptions.b2()
    )
    orchestrator._use_frozen_requirements([_REQ])
    approved = build_contract_bundle([_REQ]).to_dict()
    approved["approval"] = {
        "protocol_version": APPROVED_CONTRACT_PROTOCOL_VERSION,
        "status": "APPROVED",
        "reviewer": "Supervisor",
        "approved_at": "2026-07-19T10:00:00Z",
        "requirement_set_digest": requirement_set_digest([_REQ]),
    }

    orchestrator._apply_approved_contract_bundle(approved, [_REQ])

    assert orchestrator.last_approved_contract_provenance["reviewer"] == "Supervisor"
    assert (
        orchestrator.last_requirement_semantic_analysis["contract_source"]
        == "approved_contract_bundle"
    )


def test_generate_rejects_mixed_frozen_and_additional_inputs_before_llm_call():
    orchestrator = Orchestrator(_NoCallLLM())

    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        orchestrator.generate(
            "D",
            "A sufficiently detailed system description for deterministic testing.",
            additional_requirements=[_REQ],
            frozen_requirements=[_REQ],
        )
