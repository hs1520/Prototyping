"""Every Tier 0 rewrite is adopted only on a strict fall in the error count.

Two of the five fixers were written that way from the start; three adopted their
edit unconditionally and were brought into line. These cases pin the resulting
contract and, equally, that bringing them into line did not change what Tier 0
produces -- each fixer still resolves the error class it targets, and a chain of
two still converges.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator, PrototypingState
from src.agents.refinement import (
    ModelRevision,
    RefinementClosure,
    RefinementClosureRequest,
)
from src.agents.refinement_intelligence import ScriptedRefinementIntelligence
from src.simulation.validator import SimulationResult
from src.simulation.syntax_checker import check_syntax
from src.sysml.lite_model import build_lite_model


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("Tier 0 must not call the LLM")


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
    orchestrator = Orchestrator(
        _NoCallLLM(),
        max_iterations=1,
        quality_threshold=0.5,
        use_surgical_refinement=False,
    )
    orchestrator.state = PrototypingState(
        system_name="M",
        system_description="",
    )
    intelligence = ScriptedRefinementIntelligence(evaluate=(
        SimpleNamespace(
            weighted_total=1.0,
            issues=[],
            recommendations=[],
            criteria_scores={},
        ),
    ))
    orchestrator.refinement_closure = RefinementClosure(
        orchestrator,
        intelligence=intelligence,
        simulation_runner=lambda _text, name: SimulationResult(model_name=name),
        verification_gap_audit=lambda _text, _name: [],
    )
    outcome = orchestrator.refinement_closure.refine(
        RefinementClosureRequest(
            base=ModelRevision.capture(
                build_lite_model(text, model_name="M")
            ),
            requirements=(),
        )
    )
    return (
        outcome.revision.sysml,
        check_syntax(outcome.revision.sysml),
        (),
        outcome.evidence["syntax_error_count"] == 0,
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


# A guard names `y`, which nothing declares; `x` is declared but in a different
# part definition, so it is not visible from the guard. `x` is one edit away
# from `y`, which is enough for the Levenshtein fixer to propose it.
_TYPO_WHOSE_NEAREST_NAME_IS_OUT_OF_SCOPE = (
    "package M { part def A { attribute x : Real = 1.0; } "
    "part def B { state def S { entry; then a; state a; "
    "transition t first a if y > 1 then a; } } }"
)


def test_a_rewrite_that_fixes_nothing_is_rejected_even_though_it_looks_right():
    """The case the strict rule exists for, and the reason it is not cosmetic.

    Substituting `x` for `y` resolves nothing: the error merely moves from "no
    feature named y" to "no feature named x", because `x` is declared in another
    part. The error count is identical before and after, so the edit is refused.
    Without the rule it would be adopted, and a guard would silently come to
    reference a different variable in a different part -- a model that passes
    every syntactic check while meaning something else.
    """
    text = _TYPO_WHOSE_NEAREST_NAME_IS_OUT_OF_SCOPE
    before = check_syntax(text)
    assert len(before.sema_errors) == 1
    assert "No Feature named 'y' found" in before.sema_errors[0]["message"]

    # The fixer does propose the substitution ...
    import src.agents.refinement as refinement

    proposed = refinement.try_fix_sema_errors(text, before.sema_errors)
    assert proposed.fixed_text != text
    assert "if x >" in proposed.fixed_text
    # ... and it resolves nothing.
    assert check_syntax(proposed.fixed_text).total_errors() == before.total_errors()

    # Tier 0 therefore keeps the original identifier, and reaches zero errors by
    # declaring the attribute the guard actually names.
    fixed, result, _hints, resolved = _tier0(text)
    assert "if y >" in fixed
    assert "if x >" not in fixed
    assert resolved is True
    assert result.total_errors() == 0
