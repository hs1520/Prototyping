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


def test_state_definition_and_assert_constraint_collision_fails():
    model = """package P {
    part def RecoveryPowerSupply {
        state def RecoveryPowerSupplyBehavior { state idle; }
        assert constraint RecoveryPowerSupplyBehavior { true }
    }
}"""

    report = check_user_namespace_integrity(model)

    assert report["status"] == "FAIL"
    assert report["duplicate_members"] == [{
        "scope": "P::RecoveryPowerSupply",
        "name": "RecoveryPowerSupplyBehavior",
        "kinds": ["state def", "assert constraint"],
        "count": 2,
    }]


def test_duplicate_attribute_usages_in_same_item_scope_fail():
    model = """package P {
    item def ObstacleData {
        attribute separation : Real = 0.0 [m];
        attribute separation : Real;
    }
}"""

    report = check_user_namespace_integrity(model)

    assert report["status"] == "FAIL"
    assert report["duplicate_members"] == [{
        "scope": "P::ObstacleData",
        "name": "separation",
        "kinds": ["attribute", "attribute"],
        "count": 2,
    }]


def test_duplicate_item_usages_in_same_port_scope_fail():
    model = """package P {
    item def ObstacleData;
    port def ObstaclePort {
        item payload : ObstacleData;
        in item payload : ObstacleData;
    }
}"""

    report = check_user_namespace_integrity(model)

    assert report["status"] == "FAIL"
    assert report["duplicate_members"] == [{
        "scope": "P::ObstaclePort",
        "name": "payload",
        "kinds": ["item", "item"],
        "count": 2,
    }]


def test_same_feature_name_in_different_definition_scopes_is_legal():
    model = """package P {
    item def A { attribute value : Real; }
    item def B { attribute value : Real; }
}"""

    assert check_user_namespace_integrity(model)["status"] == "PASS"


def test_same_scope_port_and_attribute_collision_fails():
    model = """package P {
    part def Latch {
        out port startupInhibitActive : InhibitPort;
        attribute startupInhibitActive : Boolean = false;
    }
}"""

    report = check_user_namespace_integrity(model)

    assert report["status"] == "FAIL"
    assert report["duplicate_members"] == [{
        "scope": "P::Latch",
        "name": "startupInhibitActive",
        "kinds": ["out port", "attribute"],
        "count": 2,
    }]


def test_same_port_name_in_different_definition_scopes_is_legal():
    model = """package P {
    part def Producer { out port status : StatusPort; }
    part def Consumer { in port status : StatusPort; }
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
    assert checks["SYSML_SYNTAX_AND_SEMANTICS"]["status"] == "FAIL"
    assert checks["SYSML_SYNTAX_AND_SEMANTICS"]["evidence"][
        "warning_count"
    ] >= 1
    assert "namespace-distinguishability" in checks[
        "SYSML_SYNTAX_AND_SEMANTICS"
    ]["evidence"]["warning_codes"]
    assert checks["USER_NAMESPACE_INTEGRITY"]["status"] == "FAIL"


def test_reserved_ag_identity_has_an_explicit_qualification_check():
    model = """package P {
    requirement def REQ_SAFE_001 { doc /* safe */ }
    satisfy requirement REQ_SAFE_001;
}"""
    reserved = {
        "artifact_role": "A_G_RESERVED_IDENTITY_CONFORMANCE",
        "status": "FAIL",
        "issues": [
            "RecoveryPowerSupply::RecoveryPowerSupplyBehavior: "
            "wrong-kind declarations: state def=1"
        ],
    }

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
        generation_plan_conformance={
            "status": "FAIL",
            "ag_reserved_identity_conformance": reserved,
        },
    )
    checks = {
        item["name"]: item for item in qualification["checks"]
    }

    assert checks["A_G_RESERVED_IDENTITY_CONFORMANCE"]["status"] == "FAIL"
    assert checks["A_G_RESERVED_IDENTITY_CONFORMANCE"]["evidence"] == reserved


def test_planned_event_identity_has_an_explicit_qualification_check():
    model = """package P {
    requirement def REQ_SAFE_001 { doc /* safe */ }
    satisfy requirement REQ_SAFE_001;
}"""
    event_symbols = {
        "artifact_role": "PLANNED_EVENT_SYMBOL_CONFORMANCE",
        "status": "FAIL",
        "issues": [
            "FaultSignal: wrong definition kinds: action def"
        ],
    }

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
        generation_plan_conformance={
            "status": "FAIL",
            "planned_event_symbol_conformance": event_symbols,
        },
    )
    checks = {
        item["name"]: item for item in qualification["checks"]
    }

    assert checks["PLANNED_EVENT_SYMBOL_CONFORMANCE"]["status"] == "FAIL"
    assert checks["PLANNED_EVENT_SYMBOL_CONFORMANCE"][
        "evidence"
    ] == event_symbols
