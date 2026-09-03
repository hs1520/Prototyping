"""Regression tests for the 2026-08-26 tier-1 semantic fixes.

Each pins the corrected behavior of a defect no prior test covered, which is
how all six survived a 1700-test green suite.
"""
from src.dse.domain_objective import mass_limit
from src.dse.physics_estimator import DesignInputs
from src.realization.closure import close_the_loop
from src.realization.realization_emitter import emit_realization_package
from src.simulation.behavioral_sim import (
    BehavioralScenarioResult,
    BehavioralSimResult,
    _compute_score,
)
from src.simulation.validator import SimulationResult

from .realization_fixtures import catalog


def _scenario(passed: bool, tags):
    return BehavioralScenarioResult(
        name="s", state_machine="SM", description="d", passed=passed,
        tags=list(tags),
    )


def test_mass_limit_tightest_bound():
    reqs = [
        "REQ-CONS-001: maximum take-off mass shall not exceed 25 kg.",
        "REQ-CONS-004: the all-up mass shall be limited to 4 kg.",
    ]
    rid, bound = mass_limit(reqs)
    assert bound == 4.0
    assert rid == "REQ-CONS-004"


def test_trigger_conflict_untestable():
    conflicted = _scenario(False, ["emergency", "trigger_conflict"])
    real_pass = _scenario(True, [])
    # excluded from the denominator: neither free PASS nor penalised FAIL
    assert _compute_score([conflicted, real_pass]) == 1.0
    assert _compute_score([conflicted]) == 1.0


def test_constraint_failure_scored():
    br = BehavioralSimResult(
        model_name="m",
        scenario_results=[_scenario(False, ["parametric"])],
        sim_score=0.0,
        extracted_sm_count=0,
    )
    sim = SimulationResult(model_name="m")
    sim.reachability_score = 1.0
    sim.behavioral_result = br
    # old predicate (extracted_sm_count == 0) discarded the failing sweep
    assert sim.combined_score < 1.0


def test_veto_gives_full_robustness():
    from src.dse.evaluator import DesignEvaluator, EvaluationResult

    ev = DesignEvaluator(quality_threshold=0.75)
    result = EvaluationResult(configuration_name="c")
    result.criteria_scores = {
        "safety_assurance": 0.20, "requirement_coverage": 0.9,
        "structural_completeness": 0.9, "interface_quality": 0.9,
    }
    result.weighted_total = 0.70
    result.issues = ["[VETO] safety_assurance=0.20 < 0.40 — floor"]
    assert ev.verdict_robustness(result) == 1.0


def test_emitter_no_unmet_satisfy():
    d = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], [
        "REQ-PERF-002: endurance at least 15 minutes.",
        "REQ-FUNC-003: transport a 2.5 kg payload with a hover throttle "
        "margin of at least 30 percent.",
    ], catalog())
    payload = [v for v in rep.per_requirement if v.family == "payload"]
    sysml, ok = emit_realization_package(rep)
    assert ok
    if payload and payload[0].met is False:
        assert "satisfy req_func_003;" not in sysml
    # payload has no realized attribute - no endurance assert in its name
    for line in sysml.splitlines():
        if "realizationCloses" in line and "0.3" in line:
            raise AssertionError(f"fabricated payload assert: {line}")
