"""Functional closure without surgical refinement falls back to full rewrite.

NO-SURGICAL@seed0 measured the old behavior: the closure pass printed
"surgical refinement disabled; functional gaps remain" and gave up with 2
budgeted passes and 0 attempts, so the arm measured whether closure repair
exists rather than surgical vs full-rewrite repair. The fallback keeps the
pass budget and the acceptance gates; only the candidate generator differs.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator, PrototypingState
from src.agents.refinement import ModelRevision, RefinementClosure
from src.agents.refinement_intelligence import ScriptedRefinementIntelligence
from src.llm.interface import MockLLM
from src.simulation.validator import SimulationResult
from src.sysml.lite_model import build_lite_model

_BROKEN = """package Closure {
    requirement def REQ_FUNC_007 { doc /* report health */ }
    part def Controller {
        satisfy requirement REQ_FUNC_007;
    }
    part controller : Controller;
}"""

_FIXED = _BROKEN.replace(
    "        satisfy requirement REQ_FUNC_007;",
    "        satisfy requirement REQ_FUNC_007;\n"
    "        action def reportHealth {}",
)


def _eval(total=0.9):
    return type("Evaluation", (), {
        "weighted_total": total, "issues": [], "recommendations": [],
        "criteria_scores": {},
    })()


def _harness(generate_responses, gap_audit):
    runtime = Orchestrator(
        llm=MockLLM(), max_iterations=1, quality_threshold=0.5,
        use_surgical_refinement=False,
    )
    runtime.state = PrototypingState(system_name="Closure",
                                     system_description="")
    intelligence = ScriptedRefinementIntelligence(
        generate=generate_responses,
        evaluate=[lambda payload: _eval()] * 4,
    )
    closure = RefinementClosure(
        runtime, intelligence=intelligence,
        simulation_runner=lambda _t, name: SimulationResult(model_name=name),
        verification_gap_audit=lambda _t, _n: [],
        functional_gap_audit=gap_audit,
    )
    return runtime, closure._RefinementClosure__implementation


def test_no_surgical_full_rewrite_closes():
    def gap_audit(text, _name):
        if "reportHealth" in text:
            return []
        return ["[VERIFY-GAP] REQ_FUNC_007 has no reachable report response"]

    fixed_model = build_lite_model(_FIXED, model_name="Closure")
    runtime, engine = _harness(
        generate_responses=[
            SimpleNamespace(success=True, output=fixed_model),
        ],
        gap_audit=gap_audit,
    )
    model = build_lite_model(_BROKEN, model_name="Closure")

    current, _score, _sim = engine._functional_closure_pass(
        model, SimulationResult(model_name="Closure"), 0.9,
        ["REQ_FUNC_007: report health"], None,
    )

    closure_record = engine.last_functional_closure
    assert closure_record["status"] == "CLOSED"
    assert closure_record["attempts"] == 1
    assert closure_record["accepted_repairs"] == 1
    assert [c.get("generator") for c in closure_record["repair_contexts"]] \
        == ["FULL_REWRITE"]
    assert [c.get("status") for c in closure_record["repair_contexts"]] \
        == ["ACCEPTED"]


def test_unusable_rewrite_leaves_open():
    runtime, engine = _harness(
        generate_responses=[
            SimpleNamespace(success=False, output=None),
            SimpleNamespace(success=False, output=None),
        ],
        gap_audit=lambda _t, _n: [
            "[VERIFY-GAP] REQ_FUNC_007 has no reachable report response"
        ],
    )
    model = build_lite_model(_BROKEN, model_name="Closure")

    engine._functional_closure_pass(
        model, SimulationResult(model_name="Closure"), 0.9,
        ["REQ_FUNC_007: report health"], None,
    )

    closure_record = engine.last_functional_closure
    assert closure_record["status"] == "OPEN"
    assert closure_record["attempts"] == 2
    assert all(
        c.get("reason") == "full_rewrite_unusable"
        for c in closure_record["repair_contexts"]
    )
