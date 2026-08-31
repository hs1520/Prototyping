from gazebo_poc.safety_precedence_evidence import evaluate_safety_precedence


def test_parachute_wins_over_every_declared_competing_response():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=[
            "initiateArmingInhibit",
            "initiateEmergencyLand",
            "initiateBatteryRtb",
        ],
    )

    assert evidence.status == "verified"
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


def test_no_declared_competitors_cannot_establish_precedence_over_all_others():
    evidence = evaluate_safety_precedence(
        fired_action_definitions=["deployParachute"],
        winner_action_definition="deployParachute",
        competing_action_definitions=[],
    )

    assert evidence.status == "inconclusive"
