from src.prototyping.ag_behavior_plan import (
    BehaviorObligation,
    BehaviorObligationPlan,
    TransitionObligation,
    materialize_behavior_obligations,
)
from src.prototyping.ag_extractor import extract_ag_graph
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


def test_behavior_materialization_reuses_existing_action_event_definition():
    fragment = """action def FaultSignal {}
state def ControllerBehavior {
    entry; then idle;
    state idle;
    state responding { entry action setResponse; }
    transition realize1 first idle accept FaultSignal then responding;
}"""

    materialized, report = materialize_behavior_obligations(fragment, _plan())

    assert report["status"] == "PASS"
    assert materialized.count("action def FaultSignal") == 1
    assert "attribute def FaultSignal" not in materialized


def test_missing_event_definition_is_materialized_as_an_action_definition():
    materialized, report = materialize_behavior_obligations("", _plan())

    assert report["status"] == "PASS"
    assert "action def FaultSignal {}" in materialized
    assert "attribute def FaultSignal" not in materialized


def test_ag_extractor_reads_canonical_and_legacy_event_definitions():
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

    assert extract_ag_graph(canonical).declared_event_signals == (
        "FaultSignal",
    )
    assert extract_ag_graph(legacy).declared_event_signals == ("FaultSignal",)


def test_one_action_event_definition_has_no_namespace_warning():
    model = """package P {
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

    result = check_syntax(model, filter_stdlib_diagnostics=False)

    assert result.total_errors() == 0
    assert not [
        item for item in result.warnings
        if item.get("code") == "namespace-distinguishability"
    ]
