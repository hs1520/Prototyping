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
    def test_enum_def_key_construct(self):
        assert "enum def" in SYSML_EXPERT_SYSTEM_PROMPT

    def test_enum_value_access(self):
        assert "EnumName::Value" in SYSML_EXPERT_SYSTEM_PROMPT


class TestEnumDefInBehaviorTemplate:
    def test_mode_machine_rules_present(self):
        assert "MODE MACHINE RULES" in BEHAVIOR_TEMPLATE

    def test_owner_package_sentinel(self):
        assert "// OWNER: package" in BEHAVIOR_TEMPLATE

    def test_attr_owner_hint(self):
        assert "// ATTR OWNER:" in BEHAVIOR_TEMPLATE

    def test_enum_def_syntax_example(self):
        assert "enum def" in BEHAVIOR_TEMPLATE

    def test_enum_body_plain_names(self):
        assert "plain identifiers" in BEHAVIOR_TEMPLATE

    def test_enum_equality_guard_allowed(self):
        assert "ENUM GUARD EXCEPTION" in BEHAVIOR_TEMPLATE
        assert "== allowed" in BEHAVIOR_TEMPLATE or "IS allowed" in BEHAVIOR_TEMPLATE

    def test_enum_guard_example(self):
        assert "FlightMode::HOVER" in BEHAVIOR_TEMPLATE

    def test_enum_guard_missing_prefix(self):
        assert "missing EnumType:: prefix" in BEHAVIOR_TEMPLATE

    def test_numeric_equality_ban(self):
        assert "NEVER use `==` or `!=` for a numeric condition" in BEHAVIOR_TEMPLATE

    def test_enum_no_integer_assignment(self):
        assert "IDLE = 0" in BEHAVIOR_TEMPLATE

    def test_enum_body_no_qualified_values(self):
        assert "FlightMode::IDLE" in BEHAVIOR_TEMPLATE


class TestEnumDefInPartDefinitionsTemplate:
    def test_step2_no_mode_attribute(self):
        # The mode attribute is injected by Step 4 via `// ATTR OWNER:`;
        # Step 2 does not declare it, to avoid enum-def duplication.
        assert "do NOT declare the mode attribute" in PART_DEFINITIONS_TEMPLATE

    def test_step2_defers_enum_def(self):
        assert "do NOT" in PART_DEFINITIONS_TEMPLATE
        assert "enum def" in PART_DEFINITIONS_TEMPLATE


class TestOperCategory:
    def test_oper_in_valid_categories(self):
        assert "OPER" in _VALID_CATEGORIES

    def test_requirements_oper_section(self):
        assert "OPERATIONAL MODES" in REQUIREMENTS_COT_TEMPLATE
        assert "OPER" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_format_example(self):
        assert "REQ-OPER-001" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_phase_arrow_format(self):
        assert "→" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_all_caps_phases(self):
        assert "ALL-CAPS" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_max_one_requirement(self):
        assert "EXACTLY ONE" in REQUIREMENTS_COT_TEMPLATE or "at most ONE" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_zero_or_one(self):
        assert "0 or 1" in REQUIREMENTS_COT_TEMPLATE or "ZERO if" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_in_category_set(self):
        assert "OPER" in REQUIREMENTS_COT_TEMPLATE

    def test_oper_primary_trigger(self):
        assert "REQ-OPER" in BEHAVIOR_TEMPLATE
        assert "PRIMARY" in BEHAVIOR_TEMPLATE or "primary" in BEHAVIOR_TEMPLATE

    def test_behavior_oper_rule(self):
        assert "OPER requirement" in BEHAVIOR_TEMPLATE

    def test_architecture_oper_mapping(self):
        assert "OPER" in ARCHITECTURE_DECOMPOSITION_TEMPLATE


class TestEnumDefInIntegrationTemplate:
    def test_owner_package_rule(self):
        assert "// OWNER: package" in INTEGRATION_TEMPLATE

    def test_package_scope_placement(self):
        assert "package scope" in INTEGRATION_TEMPLATE or "package level" in INTEGRATION_TEMPLATE

    def test_attr_owner_injection(self):
        assert "// ATTR OWNER:" in INTEGRATION_TEMPLATE

    def test_checklist_enum_def(self):
        assert "enum defs from the behavioral fragment" in INTEGRATION_TEMPLATE

    def test_checklist_attr_owner(self):
        assert "ATTR OWNER" in INTEGRATION_TEMPLATE
