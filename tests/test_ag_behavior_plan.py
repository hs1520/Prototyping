import pytest
from dataclasses import replace

from src.agents.design_agent import DesignAgent
from src.prototyping.ag_behavior_plan import (
    INVARIANT,
    STATE_MACHINE,
    BehaviorObligationPlan,
    check_owned_behavior_obligation_conformance,
    compile_behavior_obligation_plan,
    materialize_behavior_obligations,
)
from src.prototyping.ag_chains import (
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_008_CHAIN,
)
from src.prototyping.ag_decision import build_spec_from_decisions
from tests.test_option2_ag_decision import _BOUNDARY, _CORRECT
from src.simulation import extractor
from src.simulation.syntax_checker import check_syntax


def test_all_three_chains_compile_one_typed_obligation_per_component():
    plan = compile_behavior_obligation_plan((
        REQ_SAFE_004_CHAIN,
        REQ_SAFE_005_CHAIN,
        REQ_SAFE_008_CHAIN,
    ))
    assert plan.status == "PASS"
    assert len(plan.obligations) == 8
    assert {
        item.contract_id for item in plan.obligations
    } == {
        component.name
        for chain in (
            REQ_SAFE_004_CHAIN,
            REQ_SAFE_005_CHAIN,
            REQ_SAFE_008_CHAIN,
        )
        for component in chain.components
    }


def test_continuous_power_availability_is_an_invariant_not_a_fake_state():
    plan = compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,))
    power = next(
        item for item in plan.obligations
        if item.contract_id == "RecoveryPowerSupplyContract"
    )
    assert power.realization_kind == INVARIANT
    assert power.transitions == ()
    assert "recoveryActuationPowerAvailable" in power.invariant_expression


def test_timed_failsafe_plan_carries_complete_priority_arbitration_topology():
    plan = compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,))
    arbiter = next(
        item for item in plan.obligations
        if item.stable_behavior_id == "SafetyResponseArbitration"
    )
    targets = {item.target for item in arbiter.transitions}
    lower_members = (
        set(REQ_SAFE_005_CHAIN.priority.members)
        - {REQ_SAFE_005_CHAIN.priority.selected_response}
    )
    assert lower_members <= targets
    assert REQ_SAFE_005_CHAIN.priority.selected_response in targets
    selected = next(
        item for item in arbiter.transitions
        if item.target == REQ_SAFE_005_CHAIN.priority.selected_response
    )
    assert selected.guard == REQ_SAFE_005_CHAIN.priority.trigger
    assert {
        item.target for item in arbiter.transitions
        if item.guard == "not criticalPropulsionFailureDetected"
    } == lower_members


def test_llm_decided_spec_without_realization_paths_is_compiled_losslessly():
    decided = build_spec_from_decisions(_CORRECT, _BOUNDARY)
    assert any(
        not component.realization_paths
        for component in decided.components
        if component.trigger_signal is not None
    )
    plan = compile_behavior_obligation_plan((decided,))
    assert plan.status == "PASS"
    assert len(plan.obligations) == 3
    state_obligations = [
        item for item in plan.obligations
        if item.realization_kind == STATE_MACHINE
    ]
    assert len(state_obligations) == 2
    assert all(item.transitions for item in state_obligations)
    rendered = plan.render_for_prompt()
    assert "SafetyResponseArbitration" in rendered
    assert "RecoverySystemBehavior" in rendered


def test_lifecycle_roles_without_optional_paths_are_not_fake_invariants():
    sparse_specs = []
    for chain in (REQ_SAFE_004_CHAIN, REQ_SAFE_008_CHAIN):
        sparse_specs.append(replace(
            chain,
            components=tuple(
                replace(
                    component,
                    trigger_signal=None,
                    realization_paths=(),
                )
                for component in chain.components
            ),
        ))
    plan = compile_behavior_obligation_plan(sparse_specs)
    assert plan.status == "PASS"
    assert all(
        item.realization_kind == STATE_MACHINE
        for item in plan.obligations
    )
    by_contract = {
        item.contract_id: item for item in plan.obligations
    }
    assert len(
        by_contract["SelfTestStatusLatchContract"].transitions
    ) == 4
    assert by_contract[
        "SelfTestStatusLatchContract"
    ].initial_state == "poweredOff"
    assert len(
        by_contract["ReleaseCommandGatewayContract"].transitions
    ) == 3
    assert by_contract[
        "ReleaseCommandGatewayContract"
    ].initial_state == "awaitingAuthorisation"
    assert len(
        by_contract["PayloadLockMechanismContract"].transitions
    ) == 3
    assert by_contract[
        "PayloadLockMechanismContract"
    ].initial_state == "lockedUnpowered"


def test_step4_materializes_absent_exact_obligations_and_passes_gate():
    plan = compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,))
    fragment, report = materialize_behavior_obligations(
        "state def UnrelatedBehavior { state idle; }", plan
    )
    assert report["status"] == "PASS", report
    assert set(report["materialized"]) == {
        item.stable_behavior_id for item in plan.obligations
    }
    invariant = next(
        item for item in plan.obligations
        if item.realization_kind == INVARIANT
    )
    assert f"assert constraint {invariant.stable_behavior_id}" in fragment
    assert "state def RecoverySystemBehavior" in fragment


def test_step4_replaces_an_incorrect_exact_definition_without_a_duplicate():
    original = compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,))
    obligation = next(
        item for item in original.obligations
        if item.realization_kind == STATE_MACHINE
    )
    plan = BehaviorObligationPlan((obligation,))
    fragment = (
        f"state def {obligation.stable_behavior_id} {{\n"
        "    entry; then wrongInitial;\n"
        "    state wrongInitial;\n"
        "}"
    )
    materialized, report = materialize_behavior_obligations(fragment, plan)
    assert report["status"] == "PASS"
    assert report["materialized"] == []
    assert report["replaced_inconsistent"] == [
        obligation.stable_behavior_id
    ]
    assert materialized.count(
        f"state def {obligation.stable_behavior_id}"
    ) == 1
    assert "wrongInitial" not in materialized


def test_assembled_gate_requires_realization_inside_the_exact_owner():
    original = compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,))
    invariant = next(
        item for item in original.obligations
        if item.realization_kind == INVARIANT
    )
    plan = BehaviorObligationPlan((invariant,))
    misplaced, report = materialize_behavior_obligations("", plan)
    assert report["status"] == "PASS"
    model = (
        "package System {\n"
        f"{misplaced}\n"
        f"part def {invariant.owner_def} {{ }}\n"
        "}"
    )
    owned = check_owned_behavior_obligation_conformance(model, plan)
    assert owned["status"] == "FAIL"


def test_assembly_restores_a_rewritten_definition_inside_the_exact_owner():
    original = compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,))
    obligation = next(
        item for item in original.obligations
        if item.realization_kind == STATE_MACHINE
    )
    plan = BehaviorObligationPlan((obligation,))
    fragment, step4 = materialize_behavior_obligations("", plan)
    assert step4["status"] == "PASS"
    assembled = f"""
    package System {{
        part def {obligation.owner_def} {{
            state def {obligation.stable_behavior_id} {{
                entry; then wrongInitial;
                state wrongInitial;
            }}
        }}
    }}
    """
    restored, injected = DesignAgent._inject_missing_ag_obligation_defs(
        assembled, fragment, plan
    )
    assert any(item.startswith("replaced::") for item in injected)
    assert "wrongInitial" not in restored
    assert restored.count(
        f"state def {obligation.stable_behavior_id}"
    ) == 1
    owned = check_owned_behavior_obligation_conformance(restored, plan)
    assert owned["status"] == "PASS", owned


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for the materialization syntax gate",
)
def test_all_reviewed_obligations_materialize_without_unresolved_references():
    plan = compile_behavior_obligation_plan((
        REQ_SAFE_004_CHAIN,
        REQ_SAFE_005_CHAIN,
        REQ_SAFE_008_CHAIN,
    ))
    fragment, step4 = materialize_behavior_obligations("", plan)
    assert step4["status"] == "PASS"
    owners = []
    for obligation in plan.obligations:
        if obligation.owner_def not in owners:
            owners.append(obligation.owner_def)
    model = "package DeliveryUAV {\n" + "\n".join(
        f"    part def {owner} {{ }}"
        for owner in owners
    ) + "\n}"
    model, _injected = DesignAgent._inject_missing_ag_obligation_defs(
        model, fragment, plan
    )
    owned = check_owned_behavior_obligation_conformance(model, plan)
    assert owned["status"] == "PASS", owned
    syntax = check_syntax(
        model,
        fail_closed=True,
        filter_stdlib_diagnostics=True,
    )
    assert not syntax.has_errors, syntax.format_for_llm()
