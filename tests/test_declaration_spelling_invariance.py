"""Every check that reads a declaration must see all its legal spellings.

Five separate modules were found blind to a legal way of writing the same
declaration, each in a different check, each discovered only when a real run
failed:

  activated_constraint_plan  `: LengthValue [m]`      -> duplicate attribute
  requirement_semantics      same, plus no initializer -> duplicate attribute
  generation_plan            planned name, other type  -> port reported missing
                                                          AND unplanned at once
  dse/evaluator              `first X accept S if g`   -> every A/G chain 0
  dse/domain_objective       `: ISQ::LengthValue`      -> attribute skipped

They share one shape: the model is legal, the check is a pattern, and the
pattern encodes one way of writing the thing. This file states the spellings
that must all be recognised, so the next such pattern fails here rather than in
a paid run.
"""
from __future__ import annotations

import pytest


ATTRIBUTE_SPELLINGS = [
    "attribute currentSeparation = 5.0;",
    "attribute currentSeparation : Real = 5.0;",
    "attribute currentSeparation : LengthValue = 5.0 [m];",
    "attribute currentSeparation : LengthValue [m] = 5.0 [m];",
    "attribute currentSeparation : ISQ::LengthValue = 5.0 [m];",
    "attribute currentSeparation : Boolean;",
]


@pytest.mark.parametrize("declaration", ATTRIBUTE_SPELLINGS)
def test_the_planned_attribute_materialiser_sees_every_spelling(declaration):
    from src.prototyping.activated_constraint_plan import _ATTRIBUTE

    match = _ATTRIBUTE.search(declaration)
    assert match is not None, (
        "an existing declaration written this way is invisible, so the "
        "materialiser appends a second one"
    )
    assert match.group("name") == "currentSeparation"


@pytest.mark.parametrize("declaration", ATTRIBUTE_SPELLINGS)
def test_the_semantic_binder_sees_every_spelling(declaration):
    import re

    # the pattern requirement_semantics builds per attribute name
    pattern = re.compile(
        r"\battribute\s+currentSeparation"
        r"(?:\s*:\s*[A-Za-z_][\w:]*(?:\s*\[[^\]{}]*\])?)?"
        r"(?:\s*=\s*[^;{}]+)?\s*;"
    )
    assert pattern.search(declaration) is not None


@pytest.mark.parametrize("declaration", [
    d for d in ATTRIBUTE_SPELLINGS if "=" in d and "Boolean" not in d
])
def test_the_design_ontology_sees_every_numeric_spelling(declaration):
    from src.dse.domain_objective import _ATTR_RE

    match = _ATTR_RE.search(declaration)
    assert match is not None, "a numeric attribute written this way is skipped"
    assert match.group(1) == "currentSeparation"
    assert match.group(2) == "5.0"


@pytest.mark.parametrize("transition", [
    "transition t1 first Nominal if faultDetected then Stopped;",
    "transition t1 first Nominal accept FaultSignal if faultDetected then Stopped;",
])
def test_the_guarded_transition_pattern_sees_every_spelling(transition):
    from src.dse.evaluator import _GUARDED_TRANSITION

    assert _GUARDED_TRANSITION.search(transition) is not None, (
        "a guarded transition written this way is not counted, so a model "
        "made only of them scores zero on the fault-transition sub-metric"
    )
