"""Deterministic initial_state normalisation, handled by the parser instead of a
paid correction round.

On run3's first draw 12 of 12 planned machines wrote ``initial_state`` as
``Behavior::State`` and the validator fanned it into ~40 chained issues, costing
a 15-25k output-token rewrite. Follows the behavior_kind rule: derive rather
than re-ask, and refuse anything that is not the machine's own identity.
"""
from src.prototyping.generation_plan import ModelGenerationPlan
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    normalise_planned_behavior_identities,
    validate_planned_behaviors,
)


def _behavior(initial_state, states=None, owner="Mechanism"):
    return PlannedBehavior.from_dict({
        "owner": owner,
        "behavior_id": "ReleaseBehavior",
        "initial_state": initial_state,
        "states": states or [
            {"state_id": "Locked", "role": "INITIAL"},
            {"state_id": "Releasing", "role": "RESPONSE",
             "entry_action": "actuateRelease"},
        ],
        "transitions": [{
            "transition_id": "toReleasing", "source": "Locked",
            "target": "Releasing", "trigger_kind": "ACCEPT",
            "trigger": "DeliveryCoordinateSatisfied",
        }],
    })


def test_self_qualified_stripped():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("ReleaseBehavior::Locked")]
    )
    assert behaviors[0].initial_state == "Locked"
    assert len(audit) == 1
    assert "'ReleaseBehavior::Locked' -> 'Locked'" in audit[0]
    issues = validate_planned_behaviors(
        behaviors, component_names={"Mechanism"},
        component_port_names={"Mechanism": set()},
        requirements=(), state_active_constraints=(),
    )
    assert [i for i in issues if "initial_state" in i or "INITIAL" in i] == []


def test_owner_qualified_triple_stripped():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("Mechanism::ReleaseBehavior::Locked")]
    )
    assert behaviors[0].initial_state == "Locked"
    assert len(audit) == 1


def test_foreign_prefix_kept():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("SomeOtherBehavior::Locked")]
    )
    assert behaviors[0].initial_state == "SomeOtherBehavior::Locked"
    assert audit == ()
    issues = validate_planned_behaviors(
        behaviors, component_names={"Mechanism"},
        component_port_names={"Mechanism": set()},
        requirements=(), state_active_constraints=(),
    )
    assert any("initial_state" in i for i in issues)


def test_undeclared_suffix_kept():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("ReleaseBehavior::Unlatched")]
    )
    assert behaviors[0].initial_state == "ReleaseBehavior::Unlatched"
    assert audit == ()


def test_initial_role_derived():
    behaviors, audit = normalise_planned_behavior_identities([
        _behavior("ReleaseBehavior::Locked", states=[
            {"state_id": "Locked", "role": "NORMAL"},
            {"state_id": "Releasing", "role": "RESPONSE",
             "entry_action": "actuateRelease"},
        ])
    ])
    assert behaviors[0].initial_state == "Locked"
    roles = {s.state_id: s.role for s in behaviors[0].states}
    assert roles == {"Locked": "INITIAL", "Releasing": "RESPONSE"}
    assert len(audit) == 2
    assert "role INITIAL derived" in audit[1]


def test_contradicting_claim_not_fixed():
    behaviors, audit = normalise_planned_behavior_identities([
        _behavior("ReleaseBehavior::Locked", states=[
            {"state_id": "Locked", "role": "NORMAL"},
            {"state_id": "Releasing", "role": "INITIAL",
             "entry_action": "actuateRelease"},
        ])
    ])
    assert [line for line in audit if "role INITIAL" in line] == []
    issues = validate_planned_behaviors(
        behaviors, component_names={"Mechanism"},
        component_port_names={"Mechanism": set()},
        requirements=(), state_active_constraints=(),
    )
    assert any("role INITIAL" in i for i in issues)


def test_from_payload_records_audit():
    payload = {
        "components": [{
            "name": "Mechanism",
            "responsibility": "Releases the payload.",
            "requirements": ["REQ_FUNC_001"],
            "ports": [{"name": "cmd", "direction": "in",
                       "type": "CommandPort", "external": True}],
        }],
        "connections": [],
        "behaviors": [{
            "owner": "Mechanism",
            "behavior_id": "ReleaseBehavior",
            "initial_state": "ReleaseBehavior::Locked",
            "states": [
                {"state_id": "Locked", "role": "INITIAL"},
                {"state_id": "Releasing", "role": "RESPONSE",
                 "entry_action": "actuateRelease"},
            ],
            "transitions": [{
                "transition_id": "toReleasing", "source": "Locked",
                "target": "Releasing", "trigger_kind": "ACCEPT",
                "trigger": "DeliveryCoordinateSatisfied",
            }],
        }],
    }
    plan = ModelGenerationPlan.from_payload(
        payload, requirements=["REQ_FUNC_001: release the payload"],
    )

    behavior = plan.planned_behaviors[0]
    assert behavior.initial_state == "Locked"
    assert any(
        "self-qualified reference stripped" in line
        for line in plan.behavior_identity_reconciliations
    )
    assert not any("initial_state" in issue for issue in plan.issues)


def _realization_payload(behavior_name, behavior_kind="STATE_DEF"):
    return {
        "components": [{
            "name": "PayloadMechanism",
            "responsibility": "Releases the payload.",
            "requirements": ["REQ_SAFE_006"],
            "ports": [{"name": "cmd", "direction": "in",
                       "type": "CommandPort", "external": True}],
        }],
        "connections": [],
        "requirement_realizations": [{
            "requirement_id": "REQ_SAFE_006",
            "realization_kind": "LOCAL_BEHAVIOR",
            "owner_component": "PayloadMechanism",
            "behavior_name": behavior_name,
            "behavior_kind": behavior_kind,
            "trigger_concept": "delivery-abort condition is active",
            "effect_concept": "maintain the payload in the mechanically locked state",
        }],
        "behaviors": [{
            "owner": "PayloadMechanism",
            "behavior_id": "ReleaseBehavior",
            "initial_state": "Locked",
            "states": [
                {"state_id": "Locked", "role": "INITIAL",
                 "do_action": "maintainLock"},
                {"state_id": "Releasing", "role": "RESPONSE",
                 "entry_action": "actuateRelease"},
            ],
            "transitions": [{
                "transition_id": "toReleasing", "source": "Locked",
                "target": "Releasing", "trigger_kind": "ACCEPT",
                "trigger": "DeliveryCoordinateSatisfied",
                "guard": "not deliveryAbortConditionActive",
            }],
        }],
    }


def test_state_name_reconciles_to_machine():
    """s0v6: REQ_SAFE_006 recorded behavior_name 'Locked', a state of
    PayloadMechanism's one planned machine, so the plan froze the unsatisfiable
    obligation STATE_DEF PayloadMechanism.Locked and lost terminal qualification. A
    name that is exactly one machine's state now derives to that machine, audited.
    """
    plan = ModelGenerationPlan.from_payload(
        _realization_payload("Locked"),
        requirements=["REQ_SAFE_006: maintain the payload in the "
                      "mechanically locked state whenever a delivery-abort "
                      "condition is active"],
    )
    realization = plan.requirement_realizations[0]
    assert realization.behavior_name == "ReleaseBehavior"
    assert any(
        "'Locked' -> 'ReleaseBehavior'" in line
        for line in plan.behavior_identity_reconciliations
    )
    assert not any("is not a planned behavior" in i for i in plan.issues)


def test_novel_behavior_name_kept():
    """behaviors[] writes only the behaviours it names: run2's plan carries three
    STATE_DEF realizations whose machines Step 4 authors. Only a collision with a
    planned machine's own state reconciles.
    """
    plan = ModelGenerationPlan.from_payload(
        _realization_payload("SomethingNeverPlanned"),
        requirements=["REQ_SAFE_006: maintain the locked state"],
    )
    assert plan.requirement_realizations[0].behavior_name == (
        "SomethingNeverPlanned"
    )
    assert not any("is not a planned behavior" in i for i in plan.issues)


def test_action_def_not_cross_checked():
    """behaviors[] writes state machines only; an ACTION_DEF realization may name an
    action def it never declares (REQ_SAFE_005 deployBallisticRecoveryParachute).
    """
    plan = ModelGenerationPlan.from_payload(
        _realization_payload(
            "deployBallisticRecoveryParachute", behavior_kind="ACTION_DEF",
        ),
        requirements=["REQ_SAFE_006: maintain the locked state"],
    )
    assert not any("is not a planned behavior" in i for i in plan.issues)


def test_transition_endpoints_stripped():
    """Same category as initial_state, on the transition surface: Behavior::State with
    the machine's own identity as prefix and a declared state as suffix is a
    spelling, not another state. Foreign prefixes stay for the validator.
    """
    behaviors, audit = normalise_planned_behavior_identities([
        PlannedBehavior.from_dict({
            "owner": "Mechanism",
            "behavior_id": "ReleaseBehavior",
            "initial_state": "Locked",
            "states": [
                {"state_id": "Locked", "role": "INITIAL"},
                {"state_id": "Releasing", "role": "RESPONSE",
                 "entry_action": "actuateRelease"},
            ],
            "transitions": [{
                "transition_id": "toReleasing",
                "source": "ReleaseBehavior::Locked",
                "target": "Mechanism::ReleaseBehavior::Releasing",
                "trigger_kind": "ACCEPT",
                "trigger": "DeliveryCoordinateSatisfied",
            }, {
                "transition_id": "foreign",
                "source": "OtherBehavior::Locked",
                "target": "Releasing",
                "trigger_kind": "ACCEPT",
                "trigger": "SomeEvent",
            }],
        })
    ])
    transitions = {t.transition_id: t for t in behaviors[0].transitions}
    assert transitions["toReleasing"].source == "Locked"
    assert transitions["toReleasing"].target == "Releasing"
    assert transitions["foreign"].source == "OtherBehavior::Locked"
    assert sum("transition toReleasing" in line for line in audit) == 2


def test_shadowing_transition_renamed():
    """s0v15: transition_id armSystem sat beside entry_action armSystem in one
    machine, so the transition shadows the action def, `entry action x : armSystem`
    resolves to a usage and the zero-warning gate fails on usage-feature-typing.
    The transition id is unreferenced, so it is renamed.
    """
    behaviors, audit = normalise_planned_behavior_identities([
        PlannedBehavior.from_dict({
            "owner": "FlightController",
            "behavior_id": "PowerOnSelfTestBehavior",
            "initial_state": "PowerOn",
            "states": [
                {"state_id": "PowerOn", "role": "INITIAL",
                 "do_action": "performSelfTest"},
                {"state_id": "Armed", "role": "RESPONSE",
                 "entry_action": "armSystem"},
            ],
            "transitions": [{
                "transition_id": "armSystem", "source": "PowerOn",
                "target": "Armed", "trigger_kind": "ACCEPT",
                "trigger": "SelfTestPassed",
            }],
        })
    ])
    transition = behaviors[0].transitions[0]
    assert transition.transition_id == "armSystemTransition"
    assert any(
        "'armSystem' -> 'armSystemTransition'" in line for line in audit
    )
    behaviors2, audit2 = normalise_planned_behavior_identities([
        PlannedBehavior.from_dict({
            "owner": "FlightController",
            "behavior_id": "B",
            "initial_state": "S",
            "states": [
                {"state_id": "S", "role": "INITIAL"},
                {"state_id": "T", "role": "RESPONSE",
                 "entry_action": "doThing"},
            ],
            "transitions": [{
                "transition_id": "toT", "source": "S", "target": "T",
                "trigger_kind": "ACCEPT", "trigger": "Ev",
            }],
        })
    ])
    assert behaviors2[0].transitions[0].transition_id == "toT"
    assert not any("transition_id" in line for line in audit2)
