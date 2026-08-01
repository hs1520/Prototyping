"""The provider-text normalization seam preserves its historical rule order."""

from src.sysml.text_normalization import NORMALIZATION_RULE_ORDER


def test_normalization_rule_order_is_explicit():
    assert NORMALIZATION_RULE_ORDER == {
        "syntax_gate": (
            "strip_readonly_keyword",
            "fix_keyword_item_names",
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
