from gazebo_poc.safety_precedence_evidence import evaluate_safety_precedence


def test_verified_needs_control_run():
    """Without a control run showing the responses would have fired, an inert
    arbiter and an arbitrating one look the same.
    """
    no_control = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=["initiateEmergencyLand", "initiateBatteryRtb"],
    )
    assert no_control.status == "inconclusive"
    assert "cannot be told apart from one that was never going to fire" in no_control.description

    inert_control = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=["initiateEmergencyLand"],
        control_fired_action_definitions=[],
    )
    assert inert_control.status == "inconclusive"
    assert "does not put the winner in competition" in inert_control.description


def test_parachute_wins_over_competitors():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=[
            "initiateArmingInhibit",
            "initiateEmergencyLand",
            "initiateBatteryRtb",
        ],
        # the same hazard state with the winning condition withheld fires the
        # competitors, so their silence above is suppression
        control_fired_action_definitions=[
            "initiateArmingInhibit",
            "initiateEmergencyLand",
        ],
    )

    assert evidence.status == "verified"
    assert evidence.control_actions_fired == (
        "initiateArmingInhibit", "initiateEmergencyLand",
    )
    assert "those responses were suppressed" in evidence.description
    assert evidence.winner_fired is True
    assert evidence.competing_actions_fired == ()


def test_competitor_firing_fails():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute", "initiateEmergencyLand"],
        winner_action_definition="deployParachute",
        competing_action_definitions=["initiateEmergencyLand"],
    )

    assert evidence.status == "failed"
    assert evidence.competing_actions_fired == ("initiateEmergencyLand",)


def test_winner_not_own_competitor():
    """The competing set comes from the arbiter's full action table, so it contains
    the winning action too.

    In run3 the parachute firing was then judged 'a competing response fired
    alongside the winner'.
    """
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployBallisticRecoveryParachute"],
        winner_action_definition="deployBallisticRecoveryParachute",
        competing_action_definitions=[
            "deployBallisticRecoveryParachute",
            "initiateBatteryRtb",
            "initiateEmergencyLand",
        ],
        control_fired_action_definitions=["initiateEmergencyLand"],
    )

    assert evidence.status == "verified"
    assert evidence.competing_actions_fired == ()
    assert evidence.declared_competing_actions == (
        "initiateBatteryRtb", "initiateEmergencyLand",
    )


def test_no_competitors_inconclusive():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=[],
    )

    assert evidence.status == "inconclusive"
