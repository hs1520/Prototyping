

def test_a_guarded_transition_is_counted_from_the_parse_not_the_spelling():
    """The safety sub-metric counts guarded transitions, and it used to do that
    with a pattern that required `if` immediately after the source state.

    `first X accept Sig if guard then Y` is legal SysML v2 and is exactly the
    shape the metric wants, but the intervening accept clause hid it, so every
    A/G chain scored 0 on this sub-metric. Both spellings now agree, because the
    count comes from the parser's own notion of a guard rather than from one
    way of writing one.
    """
    from src.dse.evaluator import DesignEvaluator, _GUARDED_TRANSITION
    from src.dse.design_space import DesignConfiguration
    from src.sysml.lite_model import build_lite_model

    def model_text(with_accept: bool) -> str:
        trigger = "accept FaultSignal " if with_accept else ""
        return (
            "package S {\n"
            "    requirement def REQ_SAFE_001 { doc /* stop on fault */ }\n"
            "    item def FaultSignal;\n"
            "    part def Monitor {\n"
            "        attribute faultDetected : Boolean;\n"
            "        state def MonitorBehavior {\n"
            "            entry; then Nominal;\n"
            "            state Nominal;\n"
            "            state Stopped;\n"
            f"            transition t1 first Nominal {trigger}"
            "if faultDetected then Stopped;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )

    plain, accepting = model_text(False), model_text(True)

    # the pattern alone: both spellings are one guarded transition
    assert len(_GUARDED_TRANSITION.findall(plain)) == 1
    assert len(_GUARDED_TRANSITION.findall(accepting)) == 1

    # and the scored result does not depend on which spelling was used
    scores = [
        DesignEvaluator().evaluate(
            DesignConfiguration({}),
            build_lite_model(text, model_name="S"),
            requirements=["REQ-SAFE-001: the monitor shall stop on a fault."],
        ).criteria_scores["safety_assurance"]
        for text in (plain, accepting)
    ]
    assert scores[0] == scores[1], (
        f"the same transition scored {scores} depending on its spelling"
    )
