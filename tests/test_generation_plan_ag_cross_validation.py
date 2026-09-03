from src.prototyping.ag_behavior_plan import (
    compile_behavior_obligation_plan,
)
from src.prototyping.ag_chains import (
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_008_CHAIN,
)
from src.prototyping.generation_plan import (
    ComponentPlan,
    ConnectionPlan,
    ModelGenerationPlan,
    PortPlan,
    apply_generation_plan,
    attach_ag_behavior_obligations,
)
from src.prototyping.planned_behavior import PlannedBehavior, PlannedState
from src.prototyping.structural_obligations import (
    RequirementRealizationPlan,
    StructuralObligation,
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


def test_plan_v3_carries_obligations():
    behavior_plan = compile_behavior_obligation_plan(
        (REQ_SAFE_005_CHAIN,)
    )
    plan = attach_ag_behavior_obligations(_base_plan(), behavior_plan)
    assert plan.status == "PASS"
    assert plan.to_dict()["schema_version"] == "3.0"
    assert len(plan.behavior_obligations) == 3
    prompt = plan.render_for_prompt()
    assert "TYPED A/G BEHAVIOR OBLIGATIONS" in prompt
    assert "SafetyResponseArbitration" in prompt
    assert "RecoveryPowerSupplyBehavior" in prompt


def test_reject_owner_outside_plan():
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


def test_reject_reserved_identity_reuse():
    base = _base_plan()
    colliding = PlannedBehavior(
        owner="RecoveryPowerSupply",
        behavior_id="RecoveryPowerSupplyBehavior",
        initial_state="idle",
        states=(PlannedState("idle", "INITIAL"),),
        transitions=(),
        provenance="DESIGN_DECISION",
    )
    base = ModelGenerationPlan(
        **{
            **base.__dict__,
            "planned_behaviors": (colliding,),
        }
    )

    plan = attach_ag_behavior_obligations(
        base,
        compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,)),
    )

    assert plan.status == "INVALID"
    assert any(
        "collides with frozen A/G reserved identity" in issue
        for issue in plan.issues
    )


def test_allow_same_kind_identity():
    base = _base_plan()
    aligned = PlannedBehavior(
        owner="SafetyResponseArbiter",
        behavior_id="SafetyResponseArbitration",
        initial_state="idle",
        states=(PlannedState("idle", "INITIAL"),),
        transitions=(),
        provenance="DESIGN_DECISION",
    )
    base = ModelGenerationPlan(
        **{
            **base.__dict__,
            "planned_behaviors": (aligned,),
        }
    )

    plan = attach_ag_behavior_obligations(
        base,
        compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,)),
    )

    assert plan.status == "PASS", plan.issues


def test_invariant_restored_after_pass():
    behavior_plan = compile_behavior_obligation_plan(
        (REQ_SAFE_005_CHAIN,)
    )
    plan = attach_ag_behavior_obligations(_base_plan(), behavior_plan)
    invariant = next(
        item for item in behavior_plan.obligations
        if item.realization_kind == "INVARIANT"
    )
    owners = sorted({
        item.owner_def for item in behavior_plan.obligations
    })
    owner_defs = "\n".join(
        (
            f"part def {owner} {{\n"
            + (
                f"state def {invariant.stable_behavior_id} "
                "{ state idle; }\n"
                if owner == invariant.owner_def else ""
            )
            + "}"
        )
        for owner in owners
    )
    model = f"package P {{\n{owner_defs}\n}}"

    compiled, report = apply_generation_plan(model, plan)

    assert f"state def {invariant.stable_behavior_id}" not in compiled
    assert compiled.count(
        f"assert constraint {invariant.stable_behavior_id}"
    ) == 1
    assert report["ag_reserved_identity_conformance"]["status"] == "PASS"


def test_plan_v3_round_trips():
    original = attach_ag_behavior_obligations(
        _base_plan(),
        compile_behavior_obligation_plan((REQ_SAFE_005_CHAIN,)),
    )
    restored = ModelGenerationPlan.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_local_behavior_frozen_identity():
    plan = ModelGenerationPlan(
        components=(
            ComponentPlan(
                name="ReleaseCommandGateway",
                responsibility="Authorize payload release.",
                requirements=("REQ_SAFE_008",),
                ports=(
                    PortPlan(
                        name="releaseCommand",
                        direction="out",
                        port_type="CommandPort",
                        external=True,
                    ),
                ),
            ),
            ComponentPlan(
                name="PayloadLockMechanism",
                responsibility="Default to locked state upon power-on.",
                requirements=("REQ_SAFE_008",),
                ports=(
                    PortPlan(
                        name="releaseCommand",
                        direction="in",
                        port_type="CommandPort",
                        external=True,
                    ),
                ),
            ),
        ),
        connections=(
            ConnectionPlan(
                source_component="ReleaseCommandGateway",
                source_port="releaseCommand",
                target_component="PayloadLockMechanism",
                target_port="releaseCommand",
                item_type="CommandPort",
                requirements=("REQ_SAFE_008",),
            ),
        ),
        requirement_realizations=(
            RequirementRealizationPlan(
                requirement_id="REQ_SAFE_008",
                realization_kind="LOCAL_BEHAVIOR",
                trigger_concept="upon power-on",
                effect_concept="default to the mechanically locked state",
                connection_path=(),
                owner_component="PayloadLockMechanism",
                behavior_kind="STATE_DEF",
                behavior_name="LockStateBehavior",
            ),
        ),
        structural_obligations=(
            StructuralObligation(
                obligation_id="STRUCT_REQ_SAFE_008_001",
                requirement_id="REQ_SAFE_008",
                source_component="PayloadLockMechanism",
                target_component="PayloadLockMechanism",
                required_components=("PayloadLockMechanism",),
                required_connections=(),
                entry_kind="SOURCE_ANCHORED_LOCAL_BEHAVIOR",
                trigger_concept="upon power-on",
                effect_concept="default to the mechanically locked state",
                provenance="FROZEN_REQUIREMENT_REALIZATION",
                realization_kind="LOCAL_BEHAVIOR",
                behavior_kind="STATE_DEF",
                behavior_name="LockStateBehavior",
            ),
        ),
    )
    behavior_plan = compile_behavior_obligation_plan(
        (REQ_SAFE_008_CHAIN,)
    )
    canonical = next(
        item.stable_behavior_id
        for item in behavior_plan.obligations
        if item.owner_def == "PayloadLockMechanism"
    )

    reconciled = attach_ag_behavior_obligations(
        plan, behavior_plan
    )

    assert reconciled.status == "PASS"
    assert (
        reconciled.requirement_realizations[0].behavior_name
        == canonical
    )
    assert reconciled.structural_obligations[0].behavior_name == canonical
    assert reconciled.behavior_identity_reconciliations == (
        "REQ_SAFE_008::PayloadLockMechanism::LockStateBehavior -> "
        f"{canonical}",
    )
    assert ModelGenerationPlan.from_dict(
        reconciled.to_dict()
    ).to_dict() == reconciled.to_dict()
