"""Warnings must never impersonate a failed compile.

Measured on an authoritative run: a model with 1 parser error and 46 warnings
had its parser error repaired in the first refinement iteration, yet every
subsequent iteration still scored syntactic_validity 0.000 — the 46 warnings
alone saturated the old linear formula to the same score as a hard parse
failure — and the "fails compilation" veto pinned the run at the cap while
its repair prompts demanded fixes for errors that did not exist.

The cure keeps warnings priced (the terminal qualification still demands zero
warnings) but separates the regimes: an error-free result is floored at 0.5,
above the 0.40 veto line, and the compilation veto itself now requires actual
errors regardless of the score's numeric value.
"""
from __future__ import annotations

from src.dse.design_space import DesignConfiguration
from src.dse.evaluator import DesignEvaluator
from src.simulation.syntax_checker import SyntaxCheckResult, _compute_score
from src.sysml.lite_model import build_lite_model

_MODEL = build_lite_model(
    """package M {
        requirement def REQ_FUNC_001 { doc /* navigate */ }
        part def Ctl {
            attribute speed : Real = 1.0;
            satisfy requirement REQ_FUNC_001;
        }
    }""",
    model_name="M",
)


def _warning(n: int) -> list[dict]:
    return [
        {"line": i, "col": 0, "message": f"warning {i}", "code": ""}
        for i in range(n)
    ]


def test_warnings_only_scores_floor_at_half_and_stay_monotonic():
    assert _compute_score(0, 0, 0) == 1.0
    assert _compute_score(0, 0, 3) == 0.85
    assert _compute_score(0, 0, 10) == 0.5
    # the archived shape: 46 warnings, zero errors — must NOT reach 0.0
    assert _compute_score(0, 0, 46) == 0.5
    # errors keep the compounding regime, including saturation
    assert _compute_score(1, 0, 46) == 0.0
    assert _compute_score(0, 1, 0) == 0.88
    assert _compute_score(4, 0, 0) == 0.0


def test_error_free_result_never_triggers_the_compilation_veto():
    heavy = SyntaxCheckResult(
        has_errors=False,
        warnings=_warning(46),
        score=_compute_score(0, 0, 46),
    )
    result = DesignEvaluator().evaluate(
        config=DesignConfiguration(name="t", parameters={}),
        model=_MODEL, syntax_result=heavy,
    )
    assert result.criteria_scores["syntactic_validity"] == 0.5
    assert not any(
        issue.startswith("[VETO] syntactic_validity") for issue in result.issues
    )


def test_real_errors_still_veto():
    broken = SyntaxCheckResult(
        has_errors=True,
        parser_errors=[{"line": 1, "col": 0, "message": "Unexpected '}'",
                        "code": ""}],
        warnings=_warning(46),
        score=_compute_score(1, 0, 46),
    )
    result = DesignEvaluator().evaluate(
        config=DesignConfiguration(name="t", parameters={}),
        model=_MODEL, syntax_result=broken,
    )
    assert result.criteria_scores["syntactic_validity"] == 0.0
    assert any(
        issue.startswith("[VETO] syntactic_validity") for issue in result.issues
    )


def test_veto_keys_on_errors_not_on_the_numeric_score():
    """Belt independent of the score formula: even a below-floor score must
    not assert failed compilation when no error exists."""
    weird = SyntaxCheckResult(
        has_errors=False, warnings=_warning(2), score=0.1,
    )
    result = DesignEvaluator().evaluate(
        config=DesignConfiguration(name="t", parameters={}),
        model=_MODEL, syntax_result=weird,
    )
    assert not any(
        issue.startswith("[VETO] syntactic_validity") for issue in result.issues
    )


def test_diagnostics_fallback_floors_warning_only_models():
    evaluator = DesignEvaluator()
    evaluator._cached_syntax_result = None
    score = evaluator._score_syntactic_validity(
        DesignConfiguration(name="t", parameters={}), _MODEL, None)
    assert score == 1.0  # fixture has no diagnostics at all
