from gazebo_poc.safety_precedence_evidence import evaluate_safety_precedence


def test_a_verified_precedence_needs_a_control_that_actually_competed():
    """Silence is not precedence. Without a control run showing those responses
    WOULD have fired, an inert arbiter and a correctly arbitrating one are
    indistinguishable."""
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
        control_fired_action_definitions=[],      # nothing competed either way
    )
    assert inert_control.status == "inconclusive"
    assert "does not put the winner in competition" in inert_control.description


def test_parachute_wins_over_every_declared_competing_response():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=[
            "initiateArmingInhibit",
            "initiateEmergencyLand",
            "initiateBatteryRtb",
        ],
        # the same hazard state with only the winning condition withheld does
        # fire the competitors — so their silence above is suppression
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


def test_any_competing_response_firing_is_a_precedence_failure():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute", "initiateEmergencyLand"],
        winner_action_definition="deployParachute",
        competing_action_definitions=["initiateEmergencyLand"],
    )

    assert evidence.status == "failed"
    assert evidence.competing_actions_fired == ("initiateEmergencyLand",)


def test_the_winner_is_not_its_own_competitor():
    """The competing set is discovered from the generated arbiter's full
    action table, so it contains the winning action too. run3's first real
    parachute firing was judged 'a competing response fired alongside the
    winner' — the failure was manufactured out of the win itself."""
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployBallisticRecoveryParachute"],
        winner_action_definition="deployBallisticRecoveryParachute",
        competing_action_definitions=[
            "deployBallisticRecoveryParachute",   # the winner, as discovered
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


def test_no_declared_competitors_cannot_establish_precedence_over_all_others():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=[],
    )

    assert evidence.status == "inconclusive"
