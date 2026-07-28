from dataclasses import dataclass

from src.prototyping.model_qualification import build_model_qualification
from src.prototyping.namespace_integrity import (
    check_user_namespace_integrity,
)
from src.simulation.syntax_checker import check_syntax


def test_same_scope_cross_kind_definition_collision_fails():
    model = """package P {
    attribute def FaultSignal;
    action def FaultSignal {}
}"""

    report = check_user_namespace_integrity(model)

    assert report["status"] == "FAIL"
    assert report["duplicate_members"] == [{
        "scope": "P",
        "name": "FaultSignal",
        "kinds": ["attribute def", "action def"],
        "count": 2,
    }]


def test_same_name_in_different_owning_scopes_is_legal():
    model = """package P {
    part def A { action def reset {} }
    part def B { action def reset {} }
}"""

    assert check_user_namespace_integrity(model)["status"] == "PASS"


def test_comments_and_documentation_do_not_create_false_members():
    model = """package P {
    doc /* action def FaultSignal {} */
    // attribute def FaultSignal;
    action def FaultSignal {}
}"""

    assert check_user_namespace_integrity(model)["status"] == "PASS"


@dataclass
class _Simulation:
    reachability_score: float = 1.0
    scenario_results: tuple = ()
    behavioral_result: object | None = None

    def passed_scenarios(self):
        return []


def test_namespace_collision_is_a_terminal_qualification_failure():
    model = """package P {
    requirement def REQ_SAFE_001 { doc /* safe */ }
    satisfy requirement REQ_SAFE_001;
    attribute def FaultSignal;
    action def FaultSignal {}
}"""

    qualification = build_model_qualification(
        model_text=model,
        requirements=("REQ-SAFE-001: safe",),
        syntax_result=check_syntax(model),
        simulation_result=_Simulation(),
        terminal_consistency={
            "status": "PASS",
            "model_digest": "same",
            "simulation_source_model_digest": "same",
            "evaluation_source_model_digest": "same",
        },
    )
    checks = {
        item["name"]: item for item in qualification["checks"]
    }

    assert qualification["status"] == "NOT_QUALIFIED"
    assert checks["SYSML_SYNTAX_AND_SEMANTICS"]["status"] == "PASS"
    assert checks["SYSML_SYNTAX_AND_SEMANTICS"]["evidence"][
        "warning_count"
    ] >= 1
    assert "namespace-distinguishability" in checks[
        "SYSML_SYNTAX_AND_SEMANTICS"
    ]["evidence"]["warning_codes"]
    assert checks["USER_NAMESPACE_INTEGRITY"]["status"] == "FAIL"
