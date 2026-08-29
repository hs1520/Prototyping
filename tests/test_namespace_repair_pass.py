"""Name collisions reach the author in-loop, with one bounded repair pass.

Measured on authoritative run 33f87cc6: an action def and a state def twice
shared one name inside FlightController, and the collision surfaced only at
the terminal qualification (USER_NAMESPACE_INTEGRITY + two shadowing
warnings) — the refinement loop never saw it.  The integrity findings now
ride along as advisory refinement issues, and a clean exit gets exactly one
surgical namespace-repair pass, accepted only when the duplicate count falls
and nothing regresses.  A deterministic rename is deliberately not attempted:
references to the shared name are ambiguous about which declaration they
meant.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.agents.refinement import RefinementClosure
from src.agents.refinement_intelligence import ScriptedRefinementIntelligence
from src.prototyping.namespace_integrity import namespace_integrity_issues
from src.simulation.validator import SimulationResult
from src.sysml.lite_model import build_lite_model

_REPO = Path(__file__).resolve().parents[1]


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("the pass must talk through the intelligence port")


_DUPLICATE_MODEL = """package M {
    part def Ctl {
        action def ContingencyReturn {}
        state def ContingencyReturn {
            entry; then Idle;
            state Idle;
        }
    }
}
"""

_RENAMED_BLOCK = """```sysml
part def Ctl {
    action def PerformContingencyReturn {}
    state def ContingencyReturn {
        entry; then Idle;
        state Idle;
    }
}
```"""


def _engine(intelligence, use_surgical_refinement: bool = True):
    orchestrator = Orchestrator(
        _NoCallLLM(),
        max_iterations=1,
        quality_threshold=0.5,
        use_surgical_refinement=use_surgical_refinement,
    )
    closure = RefinementClosure(
        orchestrator,
        intelligence=intelligence,
        simulation_runner=lambda _text, name: SimulationResult(model_name=name),
    )
    return orchestrator, closure._RefinementClosure__implementation


def _run(engine, model):
    return engine._namespace_repair_pass(
        model,
        SimulationResult(model_name="M"),
        rule_score=0.9,
        requirements=[],
        dse_best_config=None,
    )


def test_issue_builder_pins_the_archived_33f87cc6_collisions():
    text = (
        _REPO / "examples/output/runs"
        / "33f87cc6-5b61-4447-9df0-e10c022ec4a2/final_model.sysml"
    ).read_text()
    issues = namespace_integrity_issues(text)
    assert len(issues) == 2
    joined = " ".join(issues)
    assert "ContingencyReturnBehavior" in joined
    assert "WaypointModificationBehavior" in joined
    assert all(issue.startswith("[NAMESPACE]") for issue in issues)


def test_issue_builder_is_silent_on_a_clean_model():
    text = (
        _REPO / "experiments/ablation/results"
        / "20260829_193628_pilot5/runs/FULL_seed0.final.sysml"
    ).read_text()
    assert namespace_integrity_issues(text) == []


def test_repair_pass_accepts_a_rename_that_clears_the_duplicate():
    intelligence = ScriptedRefinementIntelligence(
        chat=(_RENAMED_BLOCK,),
        evaluate=(SimpleNamespace(
            weighted_total=0.9, issues=[], recommendations=[],
            criteria_scores={},
        ),),
    )
    orchestrator, engine = _engine(intelligence)
    model = build_lite_model(_DUPLICATE_MODEL, model_name="M")

    repaired_model, _sim, accepted = _run(engine, model)

    assert accepted is True
    from src.sysml.lite_model import build_lite_model as _  # noqa: F401
    repaired_text = repaired_model.metadata.get("last_sysml_text") or ""
    assert namespace_integrity_issues(repaired_text) == []
    attempts = orchestrator.last_namespace_repair_attempts
    assert [item["status"] for item in attempts] == ["ACCEPTED"]


def test_repair_pass_keeps_the_model_when_the_llm_output_fails_the_gates():
    intelligence = ScriptedRefinementIntelligence(
        chat=("no sysml here",),
    )
    orchestrator, engine = _engine(intelligence)
    model = build_lite_model(_DUPLICATE_MODEL, model_name="M")

    repaired_model, _sim, accepted = _run(engine, model)

    assert accepted is False
    assert repaired_model is model
    attempts = orchestrator.last_namespace_repair_attempts
    assert [item["status"] for item in attempts] == ["REJECTED"]


def test_repair_pass_is_skipped_without_surgical_refinement():
    intelligence = ScriptedRefinementIntelligence()
    orchestrator, engine = _engine(intelligence, use_surgical_refinement=False)
    model = build_lite_model(_DUPLICATE_MODEL, model_name="M")

    _model, _sim, accepted = _run(engine, model)

    assert accepted is False
    assert orchestrator.last_namespace_repair_attempts == []
    assert intelligence.calls["chat"] == []
