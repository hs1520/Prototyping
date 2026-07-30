from types import SimpleNamespace

from src.prototyping.ag_quality_gate import (
    build_ag_non_degradation_report,
)
from src.prototyping.model_qualification import build_model_qualification


class _Syntax:
    @staticmethod
    def total_errors():
        return 0


class _Simulation:
    def __init__(self, structural, behavioral, *, connections=1):
        self.scenario_results = [
            SimpleNamespace(scenario_name=name, passed=passed)
            for name, passed in structural
        ]
        self.behavioral_result = SimpleNamespace(
            extracted_sm_count=1,
            scenario_results=[
                SimpleNamespace(name=name, passed=passed)
                for name, passed in behavioral
            ],
            failed_scenarios=lambda: [
                item
                for item in self.behavioral_result.scenario_results
                if not item.passed
            ],
        )
        self.num_parts = 2
        self.num_ports = 2
        self.num_connections = connections
        self.num_actions = 0
        self.reachability_score = 1.0

    def passed_scenarios(self):
        return [item for item in self.scenario_results if item.passed]


def test_ag_non_degradation_detects_a_lost_behavior_scenario():
    before = _Simulation([("structure", True)], [("behavior", True)])
    after = _Simulation([("structure", True)], [("behavior", False)])
    report = build_ag_non_degradation_report(before, after)
    assert report["status"] == "FAIL"
    assert report["lost_behavioral_scenarios"] == ["behavior"]


def test_ag_non_degradation_requires_topology_to_remain_unchanged():
    before = _Simulation([("structure", True)], [("behavior", True)])
    after = _Simulation(
        [("structure", True)], [("behavior", True)], connections=2
    )
    report = build_ag_non_degradation_report(before, after)
    assert report["status"] == "FAIL"
    assert report["topology_unchanged"] is False


def test_role_scenario_loss_is_advisory_when_frozen_paths_are_stable():
    before = _Simulation([("role_guess", True)], [("behavior", True)])
    after = _Simulation([("role_guess", False)], [("behavior", True)])
    fixed = {
        "total": 1,
        "results": [{
            "obligation_id": "STRUCT_REQ_SAFE_001_001",
            "status": "PASS",
        }],
    }
    before.structural_obligation_report = fixed
    after.structural_obligation_report = fixed

    report = build_ag_non_degradation_report(before, after)

    assert report["status"] == "PASS"
    assert report["role_scenarios_advisory"] is True
    assert report["lost_structural_scenarios"] == ["role_guess"]
    assert report["lost_requirement_structural_obligations"] == []


def test_qualification_fails_closed_when_expected_ag_binding_is_missing():
    simulation = _Simulation(
        [("structure", True)], [("behavior", True)]
    )
    digest = {
        "status": "PASS",
        "model_digest": "same",
        "simulation_source_model_digest": "same",
        "evaluation_source_model_digest": "same",
    }
    qualification = build_model_qualification(
        model_text="package Demo {}",
        requirements=[],
        syntax_result=_Syntax(),
        simulation_result=simulation,
        terminal_consistency=digest,
        ag_expected=True,
    )
    assert qualification["status"] == "NOT_QUALIFIED"
    assert "A_G_TERMINAL_BINDING" in qualification["failed_checks"]
    assert "A_G_NON_DEGRADATION" in qualification["failed_checks"]
