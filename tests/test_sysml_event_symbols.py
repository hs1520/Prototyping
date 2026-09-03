from src.prototyping.ag_behavior_plan import (
    BehaviorObligation,
    BehaviorObligationPlan,
    TransitionObligation,
    materialize_behavior_obligations,
)
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.event_symbols import (
    PlannedEventSymbol,
    check_planned_event_symbol_conformance,
    collect_planned_event_symbols,
    materialize_planned_event_symbols,
)
from src.prototyping.namespace_integrity import (
    check_user_namespace_integrity,
)
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    PlannedState,
    PlannedTransition,
    validate_planned_behaviors,
)
from src.simulation.syntax_checker import check_syntax


def _plan() -> BehaviorObligationPlan:
    return BehaviorObligationPlan((
        BehaviorObligation(
            requirement_id="REQ_SAFE_001",
            contract_id="ControllerContract",
            owner_def="Controller",
            owner_usage="controller",
            stable_behavior_id="ControllerBehavior",
            realization_kind="STATE_MACHINE",
            assumptions=("fault",),
            guarantees=("response",),
            initial_state="idle",
            transitions=(
                TransitionObligation(
                    source="idle",
                    trigger="FaultSignal",
                    target="responding",
                    action="setResponse",
                ),
            ),
        ),
    ))


def test_empty_action_reclassified():
    fragment = """action def FaultSignal {}
state def ControllerBehavior {
    entry; then idle;
    state idle;
    state responding { entry action setResponse; }
    transition realize1 first idle accept FaultSignal then responding;
}"""

    materialized, report = materialize_behavior_obligations(fragment, _plan())

    assert report["status"] == "PASS"
    assert materialized.count("item def FaultSignal;") == 1
    assert "action def FaultSignal" not in materialized
    assert "attribute def FaultSignal" not in materialized
    assert report["event_symbol_conformance"]["removed_declarations"] == [
        {"name": "FaultSignal", "kind": "action def"},
    ]


def test_missing_event_becomes_item():
    materialized, report = materialize_behavior_obligations("", _plan())

    assert report["status"] == "PASS"
    assert "item def FaultSignal;" in materialized
    assert "action def FaultSignal" not in materialized
    assert "attribute def FaultSignal" not in materialized


def test_existing_item_reused():
    fragment = """item def FaultSignal;
state def ControllerBehavior {
    entry; then idle;
    state idle;
    state responding { entry action setResponse; }
    transition realize1 first idle accept FaultSignal then responding;
}"""

    materialized, report = materialize_behavior_obligations(fragment, _plan())

    assert report["status"] == "PASS"
    assert materialized.count("item def FaultSignal") == 1
    assert "action def FaultSignal" not in materialized
    assert "attribute def FaultSignal" not in materialized


def test_extractor_reads_all_kinds():
    canonical = """package AG {
    action def FaultSignal {}
    requirement def SystemContract {
        doc /* bounded A/G system contract for REQ_SAFE_001 */
        attribute fault : Boolean;
        assume constraint a_fault { fault }
        require constraint g_observed { fault }
    }
}"""
    legacy = canonical.replace(
        "action def FaultSignal {}", "attribute def FaultSignal;"
    )
    item_typed = canonical.replace(
        "action def FaultSignal {}", "item def FaultSignal;"
    )

    assert extract_ag_graph(canonical).declared_event_signals == (
        "FaultSignal",
    )
    assert extract_ag_graph(legacy).declared_event_signals == ("FaultSignal",)
    assert extract_ag_graph(item_typed).declared_event_signals == (
        "FaultSignal",
    )


def test_single_item_no_warning():
    model = """package P {
    item def FaultSignal;
    part def Controller {
        state def ControllerBehavior {
            entry; then idle;
            state idle;
            state responding;
            transition respond first idle accept FaultSignal then responding;
        }
    }
}"""

    result = check_syntax(model, filter_stdlib_diagnostics=False)

    assert result.total_errors() == 0
    assert not [
        item for item in result.warnings
        if item.get("code") == "namespace-distinguishability"
    ]


def test_cross_kind_duplicate_canonicalized():
    model = """package P {
    item def FaultSignal;
    action def FaultSignal {}
    part def Controller {
        state def ControllerBehavior {
            entry; then idle;
            state idle;
            state responding;
            transition respond first idle accept FaultSignal then responding;
        }
    }
}"""
    symbols = (PlannedEventSymbol("FaultSignal", ("A_G::Controller",)),)

    materialized, report = materialize_planned_event_symbols(model, symbols)

    assert report["status"] == "PASS"
    assert materialized.count("item def FaultSignal;") == 1
    assert "action def FaultSignal" not in materialized
    assert check_planned_event_symbol_conformance(
        materialized, symbols
    )["status"] == "PASS"
    assert check_user_namespace_integrity(materialized)["status"] == "PASS"
    assert check_syntax(
        materialized, filter_stdlib_diagnostics=False
    ).total_errors() == 0


def test_nonempty_action_not_reclassified():
    model = """package P {
    action def FaultSignal {
        action performResponse;
    }
}"""
    symbols = (PlannedEventSymbol("FaultSignal", ("A_G::Controller",)),)

    materialized, report = materialize_planned_event_symbols(model, symbols)

    assert report["status"] == "FAIL"
    assert "action def FaultSignal" in materialized
    assert report["unsafe_conflicts"] == [
        "FaultSignal: non-empty action definition cannot be reclassified as "
        "an event type"
    ]


def test_specialized_item_preserved():
    model = """package P {
    item def BaseSignal;
    item def FaultSignal :> BaseSignal {
        attribute severity : Integer;
    }
}"""
    symbols = (PlannedEventSymbol("FaultSignal", ("A_G::Controller",)),)

    materialized, report = materialize_planned_event_symbols(model, symbols)

    assert report["status"] == "PASS"
    assert materialized.count("item def FaultSignal") == 1
    assert "item def FaultSignal :> BaseSignal {" in materialized
    assert "attribute severity : Integer;" in materialized


def test_port_def_drift_removed():
    model = """package P {
    private import ScalarValues::*;
    port def OverrideCommand {
        in item command;
    }
    item def OverrideCommand {
        attribute overrideActive : Boolean;
    }
    part def Controller {
        state def ControllerBehavior {
            entry; then idle;
            state idle;
            state responding;
            transition respond first idle accept OverrideCommand then responding;
        }
    }
}"""
    symbols = (PlannedEventSymbol("OverrideCommand", ("PLAN",)),)

    materialized, report = materialize_planned_event_symbols(model, symbols)

    assert report["status"] == "PASS"
    assert materialized.count("item def OverrideCommand") == 1
    assert "port def OverrideCommand" not in materialized
    assert "attribute overrideActive : Boolean;" in materialized
    assert {"name": "OverrideCommand", "kind": "port def"} in report[
        "removed_declarations"
    ]
    assert check_syntax(
        materialized, filter_stdlib_diagnostics=False
    ).total_errors() == 0


def test_unregistered_accept_rejected():
    model = """package P {
    item def RegisteredSignal;
    item def UnplannedSignal;
    state def ControllerBehavior {
        entry; then idle;
        state idle;
        state responding;
        transition respond first idle accept UnplannedSignal then responding;
    }
}"""
    report = check_planned_event_symbol_conformance(
        model,
        (PlannedEventSymbol("RegisteredSignal", ("PLAN",)),),
    )

    assert report["status"] == "FAIL"
    assert report["accept_bindings"] == [{
        "name": "UnplannedSignal",
        "line": 8,
        "expected_kind": "item def",
        "status": "FAIL",
        "issues": [
            "accept target is not present in the frozen event registry",
        ],
    }]


def test_accept_without_registry_fails():
    report = check_planned_event_symbol_conformance(
        """package P {
        item def FaultSignal;
        state def B {
            entry; then idle;
            state idle;
            state fault;
            transition t first idle accept FaultSignal then fault;
        }
    }""",
        (),
    )

    assert report["status"] == "FAIL"
    assert report["accept_bindings"][0]["status"] == "FAIL"


def test_port_accept_reference_rejected():
    class Port:
        name = "obstacleData"

    class Component:
        name = "Controller"
        ports = (Port(),)

    behavior = PlannedBehavior(
        owner="Controller",
        behavior_id="ControllerBehavior",
        initial_state="idle",
        states=(
            PlannedState("idle", "INITIAL"),
            PlannedState("responding", "RESPONSE", "respond"),
        ),
        transitions=(
            PlannedTransition(
                "respond",
                "idle",
                "responding",
                "ACCEPT",
                "obstacleData",
            ),
        ),
    )

    symbols = collect_planned_event_symbols(
        (behavior,), components=(Component(),)
    )
    assert [item.name for item in symbols] == ["obstacleData"]
    issues = validate_planned_behaviors(
        (behavior,),
        component_names={"Controller"},
        component_port_names={"Controller": {"obstacleData"}},
        requirements=(),
        state_active_constraints=(),
    )
    assert any(
        "not an owner port usage" in issue for issue in issues
    )
