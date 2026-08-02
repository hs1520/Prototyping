"""The provider-text normalization seam preserves its historical rule order."""

from src.sysml.text_normalization import NORMALIZATION_RULE_ORDER


def test_normalization_rule_order_is_explicit():
    assert NORMALIZATION_RULE_ORDER == {
        # `fix_c_style_negation` is appended, not interleaved: the two rules
        # before it are historical and their relative order is load-bearing,
        # while this one is new and independent of both.
        "syntax_gate": (
            "strip_readonly_keyword",
            "fix_keyword_item_names",
            "fix_c_style_negation",
        ),
        "design_semantics": (
            "fix_capability_semantics",
            "fix_safety_action_semantics",
        ),
        "design_post_assembly": (
            "fix_doc_syntax",
            "strip_invalid_requirement_attrs",
            "normalise_connect_syntax",
        ),
        "surgical_repair": ("strip_code_fences",),
        "ag_authored_planning": ("strip_ag_implementation",),
        "ag_terminal_binding": ("strip_named_item_definitions",),
    }


def test_c_style_negation_becomes_not_without_touching_inequality():
    """`!=` is legal SysML and the A/G emitter produces it; only negation moves."""
    from src.sysml.text_normalization import fix_c_style_negation

    assert fix_c_style_negation("if !sensorFailure") == "if not sensorFailure"
    assert fix_c_style_negation("if (!armed and !locked)") == (
        "if (not armed and not locked)"
    )
    unchanged = (
        "require constraint c { not triggered or "
        "selectedResponse != FlightSafetyResponses::otherSafetyResponseSelected }"
    )
    assert fix_c_style_negation(unchanged) == unchanged


def test_the_archived_run_that_lost_its_qualification_now_parses():
    """pilot_n6_20260802/seed-3/R0-CURRENT failed the hard gate on one `!`.

    Pinned against the archived model rather than a fixture, so the rule is
    tested on the text that actually defeated it.
    """
    import pathlib

    from src.simulation.syntax_checker import check_syntax
    from src.sysml.text_normalization import fix_c_style_negation

    archived = pathlib.Path(
        "examples/output/pilot_n6_20260802/seed-3/R0-CURRENT/shared_model_final.sysml"
    )
    if not archived.exists():          # evidence archives are optional in a checkout
        return
    source = archived.read_text(encoding="utf-8")
    assert check_syntax(source, fail_closed=True).total_errors() == 1
    assert check_syntax(
        fix_c_style_negation(source), fail_closed=True
    ).total_errors() == 0
