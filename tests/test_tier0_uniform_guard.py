"""Every Tier 0 rewrite is adopted only on a strict fall in the error count.

Two of the five fixers were written that way from the start; three adopted their
edit unconditionally and were brought into line. These cases pin the resulting
contract and, equally, that bringing them into line did not change what Tier 0
produces -- each fixer still resolves the error class it targets, and a chain of
two still converges.
"""
from __future__ import annotations

import pytest

from src.agents.orchestrator import Orchestrator
from src.simulation.syntax_checker import check_syntax


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("Tier 0 must not call the LLM")


class _Model:
    def __init__(self) -> None:
        self.metadata: dict = {}
        self.name = "M"


# A guard names an attribute nothing declares -> one sema error -> ATTR-INJ.
_GUARD_ATTR_MISSING = """package M {
    part def Ctl {
        state def Modes {
            entry; then idle;
            state idle;
            state active;
            transition idle_to_active
                first idle
                if lowBattery
                then active;
        }
    }
}
"""

# An item named with a reserved word -> one parser error -> KW-FIX.
_RESERVED_ITEM_NAME = """package M {
    port def P { in item state : ScalarValues::Real; }
    part def C { port p : P; }
}
"""

_BOTH = _GUARD_ATTR_MISSING.replace(
    "part def Ctl {",
    "port def P { in item state : ScalarValues::Real; }\n    part def Ctl {",
)


def _tier0(text: str):
    orchestrator = Orchestrator(_NoCallLLM())
    return orchestrator._tier0_deterministic_fixes(
        text, _Model(), check_syntax(text)
    )


@pytest.mark.parametrize(
    "name,text,parser_errors,sema_errors",
    [
        ("attr_inj", _GUARD_ATTR_MISSING, 0, 1),
        ("kw_fix", _RESERVED_ITEM_NAME, 1, 0),
        ("both", _BOTH, 1, 1),
    ],
)
def test_tier0_resolves_each_error_class_without_the_llm(
    name, text, parser_errors, sema_errors
):
    before = check_syntax(text)
    assert len(before.parser_errors) == parser_errors, name
    assert len(before.sema_errors) == sema_errors, name

    fixed, result, _hints, resolved = _tier0(text)

    assert fixed != text, f"{name}: Tier 0 made no edit"
    assert resolved is True, f"{name}: Tier 0 did not resolve all errors"
    assert check_syntax(fixed).total_errors() == 0, name
    assert result.total_errors() == 0, name


def test_a_chain_of_two_fixers_still_converges_under_the_strict_rule():
    """The case a strict-improvement rule could plausibly break.

    KW-FIX runs before ATTR-INJ. If the first fixer's edit were discarded for
    failing to reduce the count on its own, the pair could no longer reach zero
    together. It does.
    """
    fixed, result, _hints, resolved = _tier0(_BOTH)
    assert resolved is True
    assert result.total_errors() == 0
    assert "attribute lowBattery" in fixed  # ATTR-INJ contributed
    assert check_syntax(fixed).total_errors() == 0
