"""Connectivity-regression guard for the variation-DSE refinement path.

The resolved variation model arrives at Phase 4-5 fully wired (reachability 1.0).
A full LLM rewrite tends to drop `connect` statements on the converted components,
isolating them. ``RefinementClosure.refine`` with
``preserve_connectivity=True`` must reject any candidate that sheds connects
relative to the model it was refined from.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

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

_WIRED = """package Drone {
    part def A { out port o; in port i; }
    part def B { in port i; out port o; }
    part def Sys {
        part a : A;
        part b : B;
        connect a.o to b.i;
        connect b.o to a.i;
    }
}"""


def test_connectivity_floor_defaults_off():
    # opt-in: existing (non-variation) paths keep their behaviour unchanged
    from src.agents.refinement import RefinementClosureRequest

    sig = inspect.signature(RefinementClosureRequest)
    assert sig.parameters["preserve_connectivity"].default is False


def _execute(candidate_text: str):
    current = build_lite_model(_WIRED, model_name="Drone")
    candidate = build_lite_model(candidate_text, model_name="Drone")
    runtime = Orchestrator(
        llm=MockLLM(),
        max_iterations=1,
        quality_threshold=0.95,
        use_surgical_refinement=False,
    )
    runtime.state = PrototypingState(
        system_name="Drone",
        system_description="",
    )
    evaluation = lambda issues=(): SimpleNamespace(
        weighted_total=0.8,
        issues=list(issues),
        recommendations=[],
        criteria_scores={},
    )
    intelligence = ScriptedRefinementIntelligence(
        generate=(SimpleNamespace(success=True, output=candidate),),
        evaluate=(evaluation(("improve",)), evaluation()),
        evaluate_design=(SimpleNamespace(
            final_answer="preserve the wiring",
            get_scores=lambda: {"overall": 0.8},
        ),),
    )
    closure = RefinementClosure(
        runtime,
        intelligence=intelligence,
        simulation_runner=lambda _text, name: SimulationResult(model_name=name),
        verification_gap_audit=lambda _text, _name: [],
    )
    return current, closure.refine(RefinementClosureRequest(
        base=ModelRevision.capture(current),
        requirements=(),
        preserve_connectivity=True,
    ))


def test_refine_rejects_candidate_that_drops_connectivity():
    dropped = _WIRED.replace("        connect b.o to a.i;\n", "")
    current, outcome = _execute(dropped)

    assert outcome.revision.digest == ModelRevision.capture(current).digest
    assert outcome.evidence["events"][-1]["decision"] == (
        "CONNECTIVITY_REGRESSION"
    )


def test_refine_accepts_candidate_that_preserves_connectivity():
    current, outcome = _execute(_WIRED)

    assert outcome.revision.digest == ModelRevision.capture(current).digest
    assert outcome.evidence["events"][-1]["decision"] == "ACCEPTED"
