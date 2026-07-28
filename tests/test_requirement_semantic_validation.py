from dataclasses import dataclass
from types import SimpleNamespace

from src.prototyping.model_qualification import build_model_qualification
from src.prototyping.requirement_semantics import (
    compile_requirement_semantic_obligations,
    validate_requirement_semantic_obligations,
)


_REQUIREMENT = (
    "REQ-FUNC-002: The system shall maintain at least 5 metres of "
    "separation while avoiding an obstacle."
)
_GOOD_MODEL = """package P {
    item def ObstacleData {
        attribute separation : Real;
    }
    port def ObstaclePort {
        in item payload : ObstacleData;
    }
    requirement def REQ_FUNC_002 {
        doc /* maintain obstacle separation */
    }
    part def Controller {
        in port obstacleData : ObstaclePort;
        attribute currentSeparation : Real =
            obstacleData.payload.separation;
        attribute minimumSeparation : Real = 5 [m];
        attribute avoidanceActivationDistance : Real = 8 [m];
        satisfy requirement REQ_FUNC_002;
        assert constraint keepSeparation {
            currentSeparation >= minimumSeparation
        }
        state def Avoidance {
            state nominal;
            state avoiding;
            transition beginAvoidance first nominal
                if currentSeparation <= avoidanceActivationDistance
                then avoiding;
        }
    }
}"""


def _obligations():
    return compile_requirement_semantic_obligations((_REQUIREMENT,))


def test_source_bound_sysml_constraint_passes_bounded_fidelity_gate():
    report = validate_requirement_semantic_obligations(
        _GOOD_MODEL,
        _obligations(),
        model_name="P",
    )

    assert report["status"] == "PASS"
    assert report["claim_boundary"] == (
        "MODEL_SEMANTIC_FIDELITY_NOT_PHYSICAL_PROOF"
    )
    result = report["results"][0]
    assert result["satisfying_owner"] == "Controller"
    assertion = result["candidate_evidence"][0]["assertions"][0]
    assert assertion["source_binding"] == (
        "currentSeparation = obstacleData.payload.separation"
    )
    assert assertion["numeric_bound"] == (5.0, "m")


def test_literal_placeholder_does_not_count_as_runtime_measurement():
    model = _GOOD_MODEL.replace(
        "obstacleData.payload.separation",
        "5 [m]",
    )

    report = validate_requirement_semantic_obligations(
        model,
        _obligations(),
        model_name="P",
    )

    assert report["status"] == "FAIL"
    assert any(
        "not bound to an input-port data feature" in issue
        for issue in report["results"][0]["issues"]
    )


def test_avoidance_that_starts_below_frozen_boundary_is_rejected_as_too_late():
    model = _GOOD_MODEL.replace(
        "avoidanceActivationDistance : Real = 8 [m]",
        "avoidanceActivationDistance : Real = 4 [m]",
    )

    report = validate_requirement_semantic_obligations(
        model,
        _obligations(),
        model_name="P",
    )

    assert report["status"] == "FAIL"
    assert any(
        "only after the frozen" in issue
        for issue in report["results"][0]["issues"]
    )


@dataclass
class _Simulation:
    reachability_score: float = 1.0
    scenario_results: tuple = ()
    behavioral_result: object | None = None

    def passed_scenarios(self):
        return []


def test_semantic_fidelity_failure_is_a_terminal_hard_gate():
    failed_report = validate_requirement_semantic_obligations(
        _GOOD_MODEL.replace(
            "obstacleData.payload.separation",
            "5 [m]",
        ),
        _obligations(),
        model_name="P",
    )
    qualification = build_model_qualification(
        model_text=_GOOD_MODEL,
        requirements=(_REQUIREMENT,),
        syntax_result=SimpleNamespace(total_errors=lambda: 0),
        simulation_result=_Simulation(),
        terminal_consistency={
            "status": "PASS",
            "model_digest": "same",
            "simulation_source_model_digest": "same",
            "evaluation_source_model_digest": "same",
        },
        semantic_fidelity_report=failed_report,
        semantic_fidelity_expected=True,
    )

    checks = {item["name"]: item for item in qualification["checks"]}
    assert qualification["status"] == "NOT_QUALIFIED"
    assert checks["REQUIREMENT_MODEL_SEMANTIC_FIDELITY"][
        "status"
    ] == "FAIL"


def test_no_supported_numeric_clause_is_not_overclaimed_or_failed():
    report = validate_requirement_semantic_obligations(
        _GOOD_MODEL,
        (),
        model_name="P",
    )
    qualification = build_model_qualification(
        model_text=_GOOD_MODEL,
        requirements=(_REQUIREMENT,),
        syntax_result=SimpleNamespace(total_errors=lambda: 0),
        simulation_result=_Simulation(),
        terminal_consistency={
            "status": "PASS",
            "model_digest": "same",
            "simulation_source_model_digest": "same",
            "evaluation_source_model_digest": "same",
        },
        semantic_fidelity_report=report,
    )

    checks = {item["name"]: item for item in qualification["checks"]}
    assert report["status"] == "UNVERIFIED"
    assert checks["REQUIREMENT_MODEL_SEMANTIC_FIDELITY"][
        "status"
    ] == "NOT_APPLICABLE"
