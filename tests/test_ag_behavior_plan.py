from src.prototyping.ag_behavior_plan import (
    INVARIANT,
    STATE_MACHINE,
    compile_behavior_obligation_plan,
)
from src.prototyping.ag_chains import (
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_008_CHAIN,
)
from src.prototyping.ag_decision import build_spec_from_decisions
from tests.test_option2_ag_decision import _BOUNDARY, _CORRECT


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
