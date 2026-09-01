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


def _failing_scenario():
    from src.simulation.simulator import ScenarioResult

    return ScenarioResult(
        scenario_name="emergency_x_to_y", description="pre-existing miss",
        tags=["safety"], reachable=False, path=[],
        missing_nodes=["y"], unreachable_targets=["y"], issues=["no path"],
    )


def test_a_pre_existing_scenario_failure_does_not_veto_the_repair():
    """Measured on the s0v7 anchor: the baseline carried one failed advisory
    scenario (16/17), and the old absolute gate — repaired model must have
    ZERO failed scenarios — auto-rejected the namespace repair as
    'regression'; terminal qualification then failed on the very duplicate
    the repair had fixed. Regression is relative: worse than before."""
    baseline = SimulationResult(
        model_name="M", scenario_results=[_failing_scenario()],
    )
    intelligence = ScriptedRefinementIntelligence(
        chat=(_RENAMED_BLOCK,),
        evaluate=(SimpleNamespace(
            weighted_total=0.9, issues=[], recommendations=[],
            criteria_scores={},
        ),),
    )
    orchestrator = Orchestrator(
        _NoCallLLM(), max_iterations=1, quality_threshold=0.5,
        use_surgical_refinement=True,
    )
    closure = RefinementClosure(
        orchestrator,
        intelligence=intelligence,
        # the repaired model re-simulates with the SAME pre-existing miss
        simulation_runner=lambda _text, name: SimulationResult(
            model_name=name, scenario_results=[_failing_scenario()],
        ),
    )
    engine = closure._RefinementClosure__implementation
    model = build_lite_model(_DUPLICATE_MODEL, model_name="M")

    repaired_model, _sim, accepted = engine._namespace_repair_pass(
        model, baseline, rule_score=0.9, requirements=[],
        dse_best_config=None,
    )

    assert accepted is True
    attempts = orchestrator.last_namespace_repair_attempts
    assert [item["status"] for item in attempts] == ["ACCEPTED"]


def test_a_repair_that_adds_a_scenario_failure_is_still_rejected():
    baseline = SimulationResult(model_name="M", scenario_results=[])
    intelligence = ScriptedRefinementIntelligence(
        chat=(_RENAMED_BLOCK,),
        evaluate=(SimpleNamespace(
            weighted_total=0.9, issues=[], recommendations=[],
            criteria_scores={},
        ),),
    )
    orchestrator = Orchestrator(
        _NoCallLLM(), max_iterations=1, quality_threshold=0.5,
        use_surgical_refinement=True,
    )
    closure = RefinementClosure(
        orchestrator,
        intelligence=intelligence,
        simulation_runner=lambda _text, name: SimulationResult(
            model_name=name, scenario_results=[_failing_scenario()],
        ),
    )
    engine = closure._RefinementClosure__implementation
    model = build_lite_model(_DUPLICATE_MODEL, model_name="M")

    repaired_model, _sim, accepted = engine._namespace_repair_pass(
        model, baseline, rule_score=0.9, requirements=[],
        dse_best_config=None,
    )

    assert accepted is False
    assert repaired_model is model


def test_plan_conformance_residue_rides_along_in_the_loop():
    """s0v9 anchor: an invented airframe->perception connect was invisible
    to every in-loop mechanism and failed qualification at terminal. The
    rider runs the terminal gate's own projection read-only and surfaces
    exactly the post-remediation residue."""
    import json as _json
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures" / "plan_deadlock_20260831"
    text = (fixtures / "final_model.sysml").read_text()
    payload = _json.loads(
        (fixtures / "whole_model_generation_plan.json").read_text()
    )
    requirements = _json.loads((fixtures / "requirements.json").read_text())

    class _Model:
        metadata = {"whole_model_generation_plan": payload}

    orchestrator, engine = _engine(ScriptedRefinementIntelligence())
    baseline = engine._plan_conformance_issues(text, _Model(), requirements)
    tampered = text.replace(
        "part airframe : Airframe;",
        "part airframe : Airframe;\n"
        "    connect airframe.structuralMount to "
        "perceptionSystem.obstacleData;",
        1,
    )
    assert tampered != text
    tampered_issues = engine._plan_conformance_issues(
        tampered, _Model(), requirements
    )
    new_issues = [i for i in tampered_issues if i not in baseline]
    assert any(
        "unplanned connection" in issue and "airframe.structuralMount" in issue
        for issue in new_issues
    )
    assert all(issue.startswith("[PLAN-CONFORMANCE]") for issue in new_issues)

    class _PlanlessModel:
        metadata = {}

    assert engine._plan_conformance_issues(text, _PlanlessModel(), []) == []


def test_requirement_coverage_issues_use_the_terminal_gates_own_regexes():
    from src.agents.refinement import _requirement_coverage_issues

    text = """package M {
    requirement def REQ_FUNC_001 { doc /* x */ }
    part def A { satisfy requirement REQ_FUNC_001; }
}"""
    reqs = ["REQ_FUNC_001: do x", "REQ_SAFE_002: stay safe"]
    issues = _requirement_coverage_issues(text, reqs)
    assert len(issues) == 1
    assert "requirement def REQ_SAFE_002 is missing" in issues[0]
    # def present but no satisfy link
    text_no_satisfy = text.replace(
        "satisfy requirement REQ_FUNC_001;", ""
    )
    issues2 = _requirement_coverage_issues(text_no_satisfy, ["REQ_FUNC_001: x"])
    assert len(issues2) == 1 and "no satisfy link" in issues2[0]
    assert _requirement_coverage_issues(text, ["REQ_FUNC_001: x"]) == []


def _s0_style_setup(tampered_connect: str):
    import json as _json
    from pathlib import Path
    from src.simulation.validator import SimulationResult

    fixtures = Path(__file__).parent / "fixtures" / "plan_deadlock_20260831"
    text = (fixtures / "final_model.sysml").read_text()
    payload = _json.loads(
        (fixtures / "whole_model_generation_plan.json").read_text()
    )
    requirements = _json.loads((fixtures / "requirements.json").read_text())
    tampered = text.replace(
        "part airframe : Airframe;",
        f"part airframe : Airframe;\n    {tampered_connect}",
        1,
    )
    assert tampered != text
    model = build_lite_model(tampered, model_name="AutonomousDrone")
    model.metadata["whole_model_generation_plan"] = payload
    return model, requirements, SimulationResult(model_name="AutonomousDrone")


def test_unplanned_connect_removal_pass_removes_and_audits():
    """s0v9/s0v10 died twice on the same invented connect: visible in-loop
    via the rider, fatal at terminal, and nothing empowered to remove it.
    The plan is the sole writer of connectivity — an unjustified unplanned
    connect is removed deterministically, evidence-gated."""
    connect = "connect airframe.structuralMount to perceptionSystem.obstacleData;"
    model, requirements, baseline = _s0_style_setup(connect)
    orchestrator = Orchestrator(
        _NoCallLLM(), max_iterations=1, quality_threshold=0.5,
    )
    closure = RefinementClosure(
        orchestrator, intelligence=ScriptedRefinementIntelligence(),
        simulation_runner=lambda _t, n: SimulationResult(model_name=n),
    )
    engine = closure._RefinementClosure__implementation

    repaired, _sim, accepted = engine._unplanned_connect_removal_pass(
        model, baseline, rule_score=0.9,
        requirements=requirements, dse_best_config=None,
    )

    assert accepted is True
    from src.utils.sysml_text_utils import get_sysml_text
    assert connect not in get_sysml_text(repaired)
    attempts = orchestrator.last_unplanned_connect_removal_attempts
    assert [a["status"] for a in attempts] == ["ACCEPTED"]
    assert attempts[0]["removed_statements"] == [
        "airframe.structuralMount -> perceptionSystem.obstacleData"
    ]


def test_unplanned_connect_removal_rolls_back_on_sim_regression():
    """A load-bearing connect regresses the candidate simulation and rolls
    back — fail-loud is restored, nothing is silently severed."""
    from src.simulation.simulator import ScenarioResult

    connect = "connect airframe.structuralMount to perceptionSystem.obstacleData;"
    model, requirements, baseline = _s0_style_setup(connect)

    def _failing_sim(_text, name):
        return SimulationResult(model_name=name, scenario_results=[
            ScenarioResult(
                scenario_name="x", description="", tags=[], reachable=False,
                path=[], missing_nodes=["y"], unreachable_targets=["y"],
                issues=[],
            ),
        ])

    orchestrator = Orchestrator(
        _NoCallLLM(), max_iterations=1, quality_threshold=0.5,
    )
    closure = RefinementClosure(
        orchestrator, intelligence=ScriptedRefinementIntelligence(),
        simulation_runner=_failing_sim,
    )
    engine = closure._RefinementClosure__implementation

    repaired, _sim, accepted = engine._unplanned_connect_removal_pass(
        model, baseline, rule_score=0.9,
        requirements=requirements, dse_best_config=None,
    )

    assert accepted is False
    assert repaired is model
    attempts = orchestrator.last_unplanned_connect_removal_attempts
    assert attempts[0]["status"] == "REJECTED"
    assert attempts[0]["reason"] == "regression"
