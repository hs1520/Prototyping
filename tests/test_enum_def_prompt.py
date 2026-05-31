"""
Smoke tests for Layer 1 + OPER category: enum def generation prompt rules.

These tests verify that:
- Prompt templates contain the rules needed to emit `enum def` for mode machines.
- REQUIREMENTS_COT_TEMPLATE extracts REQ-OPER requirements from descriptions.
- BEHAVIOR_TEMPLATE treats REQ-OPER as the primary mode machine trigger.
They do NOT test the extractor or simulator (those are Layer 2+).
"""

import pytest
from src.llm.chain_of_thought import (
    ARCHITECTURE_DECOMPOSITION_TEMPLATE,
    BEHAVIOR_TEMPLATE,
    INTEGRATION_TEMPLATE,
    PART_DEFINITIONS_TEMPLATE,
    REQUIREMENTS_COT_TEMPLATE,
    SYSML_EXPERT_SYSTEM_PROMPT,
)
from src.agents.requirements_agent import _VALID_CATEGORIES


class TestEnumDefInExpertPrompt:
    def test_enum_def_listed_as_key_construct(self):
        assert "enum def" in SYSML_EXPERT_SYSTEM_PROMPT

    def test_enum_def_explains_value_access(self):
        # LLM must know values are accessed as EnumName::Value
        assert "EnumName::Value" in SYSML_EXPERT_SYSTEM_PROMPT


class TestEnumDefInBehaviorTemplate:
    def test_mode_machine_rules_section_present(self):
        assert "MODE MACHINE RULES" in BEHAVIOR_TEMPLATE

    def test_owner_package_sentinel_documented(self):
        assert "// OWNER: package" in BEHAVIOR_TEMPLATE

    def test_attr_owner_hint_documented(self):
        assert "// ATTR OWNER:" in BEHAVIOR_TEMPLATE

    def test_enum_def_syntax_example_present(self):
        assert "enum def" in BEHAVIOR_TEMPLATE

    def test_enum_value_body_uses_plain_names(self):
        # Values inside enum def body must NOT use :: (e.g. FlightMode::IDLE)
        assert "plain identifiers" in BEHAVIOR_TEMPLATE

    def test_enum_equality_guard_allowed(self):
        # The carve-out for enum == must be explicit
        assert "ENUM GUARD EXCEPTION" in BEHAVIOR_TEMPLATE
        assert "== allowed" in BEHAVIOR_TEMPLATE or "IS allowed" in BEHAVIOR_TEMPLATE

    def test_enum_correct_guard_example_present(self):
        assert "FlightMode::HOVER" in BEHAVIOR_TEMPLATE

    def test_enum_wrong_guard_missing_prefix_documented(self):
        # Must warn about bare == HOVER (no type prefix)
        assert "missing EnumType:: prefix" in BEHAVIOR_TEMPLATE

    def test_numeric_equality_ban_still_present(self):
        # Numeric == ban must NOT have been removed
        assert "NEVER use `==` or `!=` for a numeric condition" in BEHAVIOR_TEMPLATE

    def test_do_not_use_integer_assignment_in_enum(self):
        assert "IDLE = 0" in BEHAVIOR_TEMPLATE  # listed as wrong form

    def test_do_not_use_qualified_values_in_enum_body(self):
        # FlightMode::IDLE inside enum def body is wrong
        assert "FlightMode::IDLE" in BEHAVIOR_TEMPLATE  # listed as wrong


class TestEnumDefInPartDefinitionsTemplate:
    def test_enum_typed_attribute_rule_present(self):
        assert "enum-typed mode attribute" in PART_DEFINITIONS_TEMPLATE

    def test_enum_mode_naming_convention_present(self):
        assert "FlightMode" in PART_DEFINITIONS_TEMPLATE


class TestOperCategory:
    """REQ-OPER category: requirements agent + prompt templates."""

    def test_oper_in_valid_categories(self):
        assert "OPER" in _VALID_CATEGORIES

    def test_requirements_template_has_oper_section(self):
        assert "OPERATIONAL MODES" in REQUIREMENTS_COT_TEMPLATE
        assert "OPER" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_format_example_present(self):
        assert "REQ-OPER-001" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_phase_arrow_format_in_example(self):
        assert "→" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_all_caps_phase_names_required(self):
        assert "ALL-CAPS" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_max_one_requirement_rule(self):
        assert "EXACTLY ONE" in REQUIREMENTS_COT_TEMPLATE or "at most ONE" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_zero_or_one_not_more(self):
        assert "0 or 1" in REQUIREMENTS_COT_TEMPLATE or "ZERO if" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_in_category_set_string(self):
        # The format rule must list OPER as a valid category
        assert "OPER" in REQUIREMENTS_COT_TEMPLATE

    def test_behavior_template_oper_is_primary_trigger(self):
        assert "REQ-OPER" in BEHAVIOR_TEMPLATE
        assert "PRIMARY" in BEHAVIOR_TEMPLATE or "primary" in BEHAVIOR_TEMPLATE

    def test_behavior_template_oper_rule_present(self):
        # BEHAVIOR_TEMPLATE rules section must mention OPER
        assert "OPER requirement" in BEHAVIOR_TEMPLATE

    def test_architecture_template_oper_mapping_present(self):
        assert "OPER" in ARCHITECTURE_DECOMPOSITION_TEMPLATE


class TestEnumDefInIntegrationTemplate:
    def test_owner_package_rule_present(self):
        assert "// OWNER: package" in INTEGRATION_TEMPLATE

    def test_package_scope_placement_rule_present(self):
        # Rule 1c: enum defs go at package level
        assert "package scope" in INTEGRATION_TEMPLATE or "package level" in INTEGRATION_TEMPLATE

    def test_attr_owner_injection_rule_present(self):
        assert "// ATTR OWNER:" in INTEGRATION_TEMPLATE

    def test_checklist_includes_enum_def(self):
        assert "enum defs from the behavioral fragment" in INTEGRATION_TEMPLATE

    def test_checklist_includes_attr_owner(self):
        assert "ATTR OWNER" in INTEGRATION_TEMPLATE
