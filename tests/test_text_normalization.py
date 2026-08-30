"""The provider-text normalization seam preserves its historical rule order."""

from src.sysml.text_normalization import NORMALIZATION_RULE_ORDER


def test_normalization_rule_order_is_explicit():
    assert NORMALIZATION_RULE_ORDER == {
        # `fix_c_style_negation` is appended, not interleaved: the two rules
        # after `fix_doc_syntax` are historical and their relative order is
        # load-bearing. `fix_doc_syntax` leads (2026-08-30): a quoted doc body
        # is a hard parser error that survived all three LLM fix attempts, and
        # rewriting it first gives every later rule a parseable text.
        "syntax_gate": (
            "fix_doc_syntax",
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
        # Terminal-commit normalisation added 2026-08-29: inherited-port
        # redeclarations are inert but fail the zero-warning qualification
        # (run 00e4d333, ten of them).
        "terminal_commit": ("strip_redundant_inherited_ports",),
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


def test_fix_doc_syntax_rewrites_quoted_bodies_without_equals():
    """Measured 2026-08-30: `doc '...';` inside a port body was a parser error
    the three-attempt LLM syntax gate could not clear."""
    from src.sysml.text_normalization import fix_doc_syntax

    text = (
        "        in port remoteIdMonitor : RemoteIdPort {\n"
        "            doc 'Monitor remote ID broadcast; satisfies REQ_INTF_003';\n"
        "        }\n"
    )
    fixed, count = fix_doc_syntax(text)
    assert count == 1
    assert "doc /* Monitor remote ID broadcast; satisfies REQ_INTF_003 */" in fixed
    assert "'" not in fixed.split("doc /*")[1].split("*/")[0]

    double, count2 = fix_doc_syntax('doc "plain double";\n')
    assert count2 == 1 and 'doc /* plain double */' in double


def test_fix_doc_syntax_keeps_legacy_equals_form_and_inner_apostrophes():
    from src.sysml.text_normalization import fix_doc_syntax

    fixed, count = fix_doc_syntax('doc = "legacy form";')
    assert count == 1 and "doc /* legacy form */" in fixed

    fixed2, count2 = fix_doc_syntax('doc "the vehicle\'s remote ID";')
    assert count2 == 1
    assert "doc /* the vehicle's remote ID */" in fixed2

    # comment-form docs and comment closers inside bodies stay safe
    valid = "doc /* already valid */"
    assert fix_doc_syntax(valid) == (valid, 0)
    defused, _ = fix_doc_syntax('doc "sneaky */ closer";')
    assert "*/ closer" not in defused.split("doc /*", 1)[1].rsplit("*/", 1)[0]


def test_fix_doc_syntax_repairs_the_archived_defect_to_parseable_sysml():
    from src.simulation.syntax_checker import check_syntax
    from src.sysml.text_normalization import fix_doc_syntax

    model = (
        "package M {\n"
        "    port def RemoteIdPort;\n"
        "    part def SafetyMonitor {\n"
        "        in port remoteIdMonitor : RemoteIdPort {\n"
        "            doc 'Monitor remote ID broadcast; satisfies REQ_INTF_003';\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    assert check_syntax(model).has_errors
    fixed, count = fix_doc_syntax(model)
    assert count == 1
    assert not check_syntax(fixed).has_errors
