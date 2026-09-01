"""Deterministic initial_state normalisation — the first mechanical issue
category eaten by the parser instead of a paid correction round.

Measured on run3's first draw: 12 of 12 planned machines wrote
``initial_state`` in the qualified form ``Behavior::State`` and the validator
fanned it into ~40 chained issues, buying a 15-25k output-token full rewrite
for a decision a parser can make alone. The rule follows the behavior_kind
reconciliation: derive, don't re-ask — and refuse anything that is not the
machine's own identity.
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


def test_a_self_qualified_initial_state_is_stripped_and_audited():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("ReleaseBehavior::Locked")]
    )
    assert behaviors[0].initial_state == "Locked"
    assert len(audit) == 1
    assert "'ReleaseBehavior::Locked' -> 'Locked'" in audit[0]
    # the normalised behaviour validates clean — no chained issues remain
    issues = validate_planned_behaviors(
        behaviors, component_names={"Mechanism"},
        component_port_names={"Mechanism": set()},
        requirements=(), state_active_constraints=(),
    )
    assert [i for i in issues if "initial_state" in i or "INITIAL" in i] == []


def test_the_owner_qualified_triple_is_the_same_self_reference():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("Mechanism::ReleaseBehavior::Locked")]
    )
    assert behaviors[0].initial_state == "Locked"
    assert len(audit) == 1


def test_a_foreign_prefix_is_left_for_the_validator_to_refuse():
    """Another machine's name is not a spelling of THIS machine's state —
    stripping it would silently decide which behaviour the plan meant."""
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


def test_an_undeclared_suffix_is_left_for_the_validator_to_refuse():
    behaviors, audit = normalise_planned_behavior_identities(
        [_behavior("ReleaseBehavior::Unlatched")]
    )
    assert behaviors[0].initial_state == "ReleaseBehavior::Unlatched"
    assert audit == ()


def test_role_initial_is_derived_when_no_state_claims_it():
    """initial_state owns the fact; the role is its duplicate."""
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


def test_a_contradicting_initial_claim_is_not_reconciled():
    """A DIFFERENT state marked INITIAL is a real contradiction, not a
    spelling — deciding it here would launder a plan defect."""
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


def test_from_payload_normalises_and_records_the_reconciliation():
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


def test_a_state_named_as_behavior_name_reconciles_to_its_machine():
    """s0v6 anchor: REQ_SAFE_006 recorded behavior_name 'Locked' — a STATE
    of PayloadMechanism's one planned machine — and with no cross-check the
    plan froze the unsatisfiable obligation STATE_DEF PayloadMechanism.Locked
    and lost terminal qualification. behaviors[] is the sole writer: a name
    that is exactly one machine's state derives to that machine, audited."""
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


def test_a_novel_behavior_name_is_left_for_step_4_to_author():
    """behaviors[] is the sole writer of the behaviours it NAMES, not of
    all behaviours: run2's archived plan carries three STATE_DEF
    realizations whose machines Step 4 authors beyond the planned ones.
    Only a collision with a planned machine's own state reconciles."""
    plan = ModelGenerationPlan.from_payload(
        _realization_payload("SomethingNeverPlanned"),
        requirements=["REQ_SAFE_006: maintain the locked state"],
    )
    assert plan.requirement_realizations[0].behavior_name == (
        "SomethingNeverPlanned"
    )
    assert not any("is not a planned behavior" in i for i in plan.issues)


def test_an_action_def_realization_is_not_cross_checked_against_behaviors():
    """behaviors[] writes state machines and nothing else; an ACTION_DEF
    realization legitimately names an action def it never declares
    (measured: REQ_SAFE_005 deployBallisticRecoveryParachute)."""
    plan = ModelGenerationPlan.from_payload(
        _realization_payload(
            "deployBallisticRecoveryParachute", behavior_kind="ACTION_DEF",
        ),
        requirements=["REQ_SAFE_006: maintain the locked state"],
    )
    assert not any("is not a planned behavior" in i for i in plan.issues)
