from dataclasses import dataclass
from types import SimpleNamespace

from src.prototyping.model_qualification import build_model_qualification
from src.prototyping.requirement_semantics import (
    SemanticBindingPlan,
    compile_requirement_semantic_obligations,
    materialize_semantic_bindings,
    validate_semantic_bindings,
    validate_requirement_semantic_obligations,
)
from src.simulation.syntax_checker import check_syntax


_REQUIREMENT = (
    "REQ-FUNC-002: The system shall maintain at least 5 metres of "
    "separation while avoiding an obstacle."
)
_GOOD_MODEL = """package P {
    private import ISQ::*;
    private import SI::*;
    item def ObstacleData {
        attribute separation : LengthValue;
    }
    port def ObstaclePort {
        in item payload : ObstacleData;
    }
    requirement def REQ_FUNC_002 {
        doc /* maintain obstacle separation */
    }
    part def Controller {
        in port obstacleData : ObstaclePort;
        attribute currentSeparation : LengthValue =
            obstacleData.payload.separation;
        attribute minimumSeparation : LengthValue = 5 [m];
        attribute avoidanceActivationDistance : LengthValue = 8 [m];
        satisfy requirement REQ_FUNC_002;
        state def Avoidance {
            state nominal;
            state avoiding {
                // PLAN-CONSTRAINT keepSeparation provenance=FROZEN_REQUIREMENT activation=STATE_ACTIVE verification=STATE_EXECUTION reference=Avoidance::avoiding
                assert constraint keepSeparation {
                    currentSeparation >= minimumSeparation
                }
            }
            transition beginAvoidance first nominal
                if currentSeparation <= avoidanceActivationDistance
                then avoiding;
        }
    }
}"""


def _obligations():
    return compile_requirement_semantic_obligations((_REQUIREMENT,))


def _binding():
    return SemanticBindingPlan.from_dict({
        "obligation_id": "SEM_REQ_FUNC_002_001",
        "requirement_id": "REQ_FUNC_002",
        "source": {
            "component": "Perception",
            "port": "obstacleData",
        },
        "target": {
            "component": "Controller",
            "port": "obstacleData",
            "runtime_attribute": "currentSeparation",
        },
        "payload": {
            "port_type": "ObstaclePort",
            "port_feature": "payload",
            "item_type": "ObstacleData",
            "item_feature": "separation",
            "value_type": "Real",
            "unit": "m",
        },
        "constraint": {
            "name": "keepSeparation",
            "threshold_attribute": "minimumSeparation",
        },
    })


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


def test_typed_binding_is_materialized_from_frozen_plan():
    incomplete = """package P {
        item def ObstacleData;
        port def ObstaclePort;
        requirement def REQ_FUNC_002;
        part def Perception {
            out port obstacleData : ObstaclePort;
        }
        part def Controller {
            in port obstacleData : ObstaclePort;
            attribute currentSeparation : Real = 5 [m];
            satisfy requirement REQ_FUNC_002;
        }
        part perception : Perception;
        part controller : Controller;
        connect perception.obstacleData to controller.obstacleData;
    }"""

    materialized, conformance = materialize_semantic_bindings(
        incomplete,
        (_binding(),),
        _obligations(),
    )
    report = validate_semantic_bindings(
        materialized,
        (_binding(),),
        _obligations(),
    )

    assert _binding().value_type == "LengthValue"
    assert conformance["status"] == "PASS"
    assert conformance["transaction_committed"] is True
    assert "private import ISQ::*;" in materialized
    assert "private import SI::*;" in materialized
    assert "attribute separation : LengthValue;" in materialized
    assert "in item payload : ObstacleData;" in materialized
    assert (
        "attribute currentSeparation : LengthValue = "
        "obstacleData.payload.separation;"
    ) in materialized
    assert (
        "attribute minimumSeparation : LengthValue = 5 [m];"
        in materialized
    )
    assert "assert constraint" not in materialized
    assert not check_syntax(
        materialized,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    ).has_errors
    assert report["status"] == "PASS"
    assert report["status"] == "PASS"


def test_semantic_materialization_rolls_back_on_payload_type_conflict():
    conflicting = _GOOD_MODEL.replace(
        "in item payload : ObstacleData;",
        "in item payload : WrongData;",
    )

    materialized, conformance = materialize_semantic_bindings(
        conflicting,
        (_binding(),),
        _obligations(),
    )

    assert materialized == conflicting
    assert conformance["status"] == "FAIL"
    assert conformance["transaction_committed"] is False
    assert any("carries WrongData" in issue for issue in conformance["issues"])


def test_semantic_materialization_reuses_compatible_initialized_features():
    already_materialized = _GOOD_MODEL.replace(
        "attribute separation : LengthValue;",
        "attribute separation : LengthValue = 0.0 [m];",
    ).replace(
        "in item payload : ObstacleData;",
        "item payload : ObstacleData;",
    )

    materialized, conformance = materialize_semantic_bindings(
        already_materialized,
        (_binding(),),
        _obligations(),
    )

    assert materialized == already_materialized
    assert conformance["status"] == "PASS"
    assert conformance["transaction_committed"] is True
    assert conformance["deterministic_changes"] == []
    assert conformance["input_model_digest"] == conformance[
        "output_model_digest"
    ]


def test_semantic_materialization_rejects_duplicate_feature_usages():
    duplicated = _GOOD_MODEL.replace(
        "attribute separation : LengthValue;",
        (
            "attribute separation : LengthValue = 0.0 [m];\n"
            "        attribute separation : LengthValue;"
        ),
    ).replace(
        "in item payload : ObstacleData;",
        (
            "item payload : ObstacleData;\n"
            "        in item payload : ObstacleData;"
        ),
    )

    materialized, conformance = materialize_semantic_bindings(
        duplicated,
        (_binding(),),
        _obligations(),
    )

    assert materialized == duplicated
    assert conformance["status"] == "FAIL"
    assert conformance["transaction_committed"] is False
    assert any("appears 2 times" in issue for issue in conformance["issues"])
    assert any(
        "must appear exactly once" in issue
        for issue in conformance["issues"]
    )


def test_semantic_materialization_canonicalizes_one_malformed_named_feature():
    malformed = _GOOD_MODEL.replace(
        "attribute separation : LengthValue;",
        "attribute separation : LengthValue [m];",
    )

    materialized, conformance = materialize_semantic_bindings(
        malformed,
        (_binding(),),
        _obligations(),
    )

    assert conformance["status"] == "PASS"
    assert "attribute separation : LengthValue [m];" not in materialized
    assert materialized.count("attribute separation : LengthValue;") == 1
    assert conformance["deterministic_changes"] == [
        "canonicalized item feature ObstacleData.separation"
    ]


def test_semantic_materialization_rejects_cross_kind_item_definition():
    conflicting = _GOOD_MODEL.replace(
        "item def ObstacleData",
        "part def ObstacleData",
    )

    materialized, conformance = materialize_semantic_bindings(
        conflicting,
        (_binding(),),
        _obligations(),
    )

    assert materialized == conflicting
    assert conformance["status"] == "FAIL"
    assert any(
        "must be item def" in issue
        for issue in conformance["issues"]
    )


def test_semantic_materialization_canonicalizes_plan_owned_attribute_defs():
    llm_serialized = _GOOD_MODEL.replace(
        "item def ObstacleData {\n"
        "        attribute separation : LengthValue;\n"
        "    }",
        "attribute def ObstacleData {\n"
        "        attribute separation : Real [m];\n"
        "    }",
    ).replace(
        "in item payload : ObstacleData;",
        "attribute payload : ObstacleData;",
    )

    materialized, conformance = materialize_semantic_bindings(
        llm_serialized,
        (_binding(),),
        _obligations(),
    )

    assert conformance["status"] == "PASS"
    assert conformance["transaction_committed"] is True
    assert "attribute def ObstacleData" not in materialized
    assert "item def ObstacleData" in materialized
    assert "attribute separation : LengthValue;" in materialized
    assert "attribute payload : ObstacleData;" not in materialized
    assert "in item payload : ObstacleData;" in materialized
    assert (
        "canonicalized definition kind ObstacleData: "
        "attribute def -> item def"
    ) in conformance["deterministic_changes"]
    assert (
        "canonicalized port payload ObstaclePort.payload"
    ) in conformance["deterministic_changes"]
    assert not check_syntax(
        materialized,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    ).has_errors


def test_avoidance_that_starts_below_frozen_boundary_is_rejected_as_too_late():
    model = _GOOD_MODEL.replace(
        "avoidanceActivationDistance : LengthValue = 8 [m]",
        "avoidanceActivationDistance : LengthValue = 4 [m]",
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
