from src.prototyping.ag_behavior_plan import (
    compile_behavior_obligation_plan,
)
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.generation_plan import (
    ModelGenerationPlan,
    attach_ag_behavior_obligations,
)


def _base_plan(owner="SafetyResponseArbiter"):
    return ModelGenerationPlan.from_payload({
        "components": [
            {
                "name": owner,
                "responsibility": "arbitrate safety responses",
                "requirements": ["REQ_SAFE_005"],
                "ports": [
                    {
                        "name": "status",
                        "direction": "out",
                        "type": "CommandPort",
                        "external": True,
                    },
                ],
            },
            {
                "name": "RecoveryPowerSupply",
                "responsibility": "supply recovery power",
                "requirements": ["REQ_SAFE_005"],
                "ports": [
                    {
                        "name": "power",
                        "direction": "out",
                        "type": "PowerPort",
                        "external": True,
                    },
                ],
            },
            {
                "name": "RecoverySystem",
                "responsibility": "deploy recovery system",
                "requirements": ["REQ_SAFE_005"],
                "ports": [
                    {
                        "name": "command",
                        "direction": "in",
                        "type": "CommandPort",
                        "external": True,
                    },
                ],
            },
        ],
        "connections": [
            {
                "source": {
                    "component": owner,
                    "port": "status",
                },
                "target": {
                    "component": "RecoverySystem",
                    "port": "command",
                },
                "item_type": "CommandPort",
                "requirements": ["REQ_SAFE_005"],
            },
        ],
    }, requirements=["REQ-SAFE-005: deploy recovery system"])


def test_model_plan_v2_carries_typed_ag_behavior_obligations():
    behavior_plan = compile_behavior_obligation_plan(
        (REQ_SAFE_005_CHAIN,)
    )
    plan = attach_ag_behavior_obligations(_base_plan(), behavior_plan)
    assert plan.status == "PASS"
    assert plan.to_dict()["schema_version"] == "2.0"
    assert len(plan.behavior_obligations) == 3
    prompt = plan.render_for_prompt()
    assert "TYPED A/G BEHAVIOR OBLIGATIONS" in prompt
    assert "SafetyResponseArbitration" in prompt
    assert "RecoveryPowerSupplyBehavior" in prompt


def test_cross_validator_rejects_an_ag_owner_outside_model_plan():
    behavior_plan = compile_behavior_obligation_plan(
        (REQ_SAFE_005_CHAIN,)
    )
    plan = attach_ag_behavior_obligations(
        _base_plan(owner="DifferentArbiter"),
        behavior_plan,
    )
    assert plan.status == "INVALID"
    assert any(
        "SafetyResponseArbiter is absent" in issue
        for issue in plan.issues
    )


def test_model_plan_v2_round_trips_behavior_obligations():
    original = attach_ag_behavior_obligations(
        _base_plan(),
        compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,)),
    )
    restored = ModelGenerationPlan.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
