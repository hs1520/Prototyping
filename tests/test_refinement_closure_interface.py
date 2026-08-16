from __future__ import annotations

from dataclasses import replace
import pytest

from src.agents.orchestrator import Orchestrator, PrototypingState
from src.agents.refinement import (
    ModelRevision,
    RefinementClosure,
    RefinementClosureRequest,
)
from src.agents.refinement_intelligence import ScriptedRefinementIntelligence
from src.llm.interface import MockLLM
from src.simulation.validator import SimulationResult
from src.sysml.lite_model import build_lite_model


def _model(name: str = "D"):
    return build_lite_model(f"package {name} {{}}", model_name=name)


def test_model_revision_detects_text_tampering():
    revision = ModelRevision.capture(_model())

    with pytest.raises(ValueError, match="digest"):
        replace(revision, sysml="package Changed {}").materialize()


def test_typestate_carries_requirements_without_hidden_cross_run_state():
    runtime = Orchestrator(
        llm=MockLLM(),
        max_iterations=1,
        quality_threshold=0.5,
        use_surgical_refinement=False,
    )
    runtime.state = PrototypingState(
        system_name="Test",
        system_description="",
    )
    intelligence = ScriptedRefinementIntelligence(evaluate=(
        lambda payload: type("Evaluation", (), {
            "weighted_total": 1.0,
            "issues": [],
            "recommendations": [],
            "criteria_scores": {},
        })(),
        lambda payload: type("Evaluation", (), {
            "weighted_total": 1.0,
            "issues": [],
            "recommendations": [],
            "criteria_scores": {},
        })(),
    ))
    closure = RefinementClosure(
        runtime,
        intelligence=intelligence,
        simulation_runner=lambda _text, name: SimulationResult(model_name=name),
        verification_gap_audit=lambda _text, _name: [],
        functional_gap_audit=lambda _text, _name: [],
    )

    first = closure.refine(RefinementClosureRequest(
        base=ModelRevision.capture(_model("First")),
        requirements=("REQ-FIRST",),
    ))
    closure.refine(RefinementClosureRequest(
        base=ModelRevision.capture(_model("Second")),
        requirements=("REQ-SECOND",),
    ))

    projected = closure.project_parameters(first, None)
    outcome = closure.close(projected)

    assert [
        tuple(call["requirements"])
        for call in intelligence.calls["evaluate"]
    ] == [("REQ-FIRST",), ("REQ-SECOND",)]
    assert projected.refined.requirements == ("REQ-FIRST",)
    assert outcome.projected.refined.requirements == ("REQ-FIRST",)
    assert outcome.evidence["status"] == "CLOSED"
    assert not hasattr(closure, "_engine")
    with pytest.raises(TypeError):
        first.evidence["syntax_error_count"] = 99


def test_fail_closed_verdict_carries_the_repairs_the_plan_refused():
    """The run dies at the gate, so the diagnosis has to travel with the error."""
    runtime = Orchestrator(
        llm=MockLLM(),
        max_iterations=1,
        quality_threshold=0.5,
        use_surgical_refinement=False,
    )
    runtime.state = PrototypingState(system_name="Test", system_description="")
    runtime._append_pipeline_state_list("plan_conformance_rejections", {
        "stage": "FUNCTIONAL_CLOSURE",
        "pass": 1,
        "issues": ["CommunicationSystem.tlmData is missing"],
    })
    closure = RefinementClosure(
        runtime,
        simulation_runner=lambda _text, name: SimulationResult(model_name=name),
        verification_gap_audit=lambda _text, _name: [],
        functional_gap_audit=lambda _text, _name: [
            "[VERIFY-GAP] REQ_FUNC_001 has no verification anchor at any tier",
        ],
    )

    with pytest.raises(RuntimeError) as raised:
        closure.verify_terminal("package Demo {}", "Demo")

    error = raised.value
    assert error.terminal_model_text == "package Demo {}"
    assert error.functional_closure["remaining_gap_req_ids"] == ["REQ_FUNC_001"]
    assert error.plan_conformance_rejections[0]["issues"] == [
        "CommunicationSystem.tlmData is missing"
    ]


def test_orchestrator_uses_composition_not_refinement_inheritance():
    assert "src.agents.refinement" not in {
        base.__module__ for base in Orchestrator.__mro__
    }
