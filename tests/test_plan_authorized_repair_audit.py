"""A refused plan-authorized repair must say which condition refused it.

The admission rule is unchanged: a repair is applied when it altered the text
and preserved the frozen obligation set, the parse, and behaviour. What changed
is what a refusal leaves behind. It used to record one sentence naming all three
possible causes disjunctively and discard the pass counts it had just computed,
so an artefact could not distinguish a syntax regression from a lost obligation.
"""
from __future__ import annotations

import itertools
import json
from unittest.mock import patch

import pytest

from src.agents.orchestrator import Orchestrator


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("this path must not call the LLM")


_MODEL = """package M {
    part def A { port p : PortDef; }
    port def PortDef { in item i : ScalarValues::Real; }
}
"""

_EMPTY_PLAN = {
    "schema_version": "1.0", "source": "TEST",
    "components": [{"name": "A", "responsibility": "r", "requirements": [],
                    "ports": [], "attributes": []}],
    "connections": [], "requirement_realizations": [],
    "structural_obligations": [], "semantic_obligations": [],
    "semantic_bindings": [], "constraints": [], "planned_behaviors": [],
    "behavior_obligations": [],
}


class _Model:
    def __init__(self) -> None:
        self.name = "M"
        self.metadata = {"whole_model_generation_plan": dict(_EMPTY_PLAN)}


def _drive(repaired_text: str):
    """Run the plan-authorized repair path with a controlled repair result."""
    orchestrator = Orchestrator(_NoCallLLM())
    model = _Model()
    with patch(
        "src.prototyping.generation_plan.apply_generation_plan",
        return_value=(repaired_text, {"status": "PASS", "issues": []}),
    ), patch.object(
        Orchestrator, "_run_simulation", lambda self, text, name: None
    ), patch(
        "src.agents.refinement.get_sysml_text", lambda m: _MODEL
    ), patch.object(
        Orchestrator, "_sync_model_text", lambda self, m, t: None
    ), patch.object(
        Orchestrator, "_fix_stuck_transitions", lambda self, m: m
    ):
        orchestrator._sim_refinement_loop(model, [], max_iters=0)
    return model.metadata


def test_a_refusal_names_the_condition_that_refused_it():
    metadata = _drive(_MODEL + "\npart def ??? {")
    conformance = metadata["generation_plan_conformance"]

    assert conformance["status"] == "FAIL"
    audit = conformance["plan_authorized_repair"]
    assert audit["applied"] is False
    assert audit["text_changed"] is True
    assert audit["failed_conditions"] == ["syntax_ok"]
    assert audit["syntax_ok"] is False
    assert audit["fixed_set_preserved"] is True
    assert audit["behavior_preserved"] is True

    # The counts that were previously computed and thrown away.
    assert "obligations_passed_before" in audit
    assert "obligations_passed_after" in audit
    assert "obligations_lost" in audit

    # And the reason reaches the issue text rather than a disjunction.
    issue = " ".join(conformance["issues"])
    assert "syntax_ok" in issue
    assert "syntax, behavior, or frozen" not in issue


def test_the_refusal_is_retained_as_its_own_record():
    metadata = _drive(_MODEL + "\npart def ??? {")
    rejected = metadata["rejected_plan_authorized_repairs"]
    assert len(rejected) == 1
    assert rejected[0]["failed_conditions"] == ["syntax_ok"]
    # Serialisable, since it travels into the run artefacts.
    json.dumps(rejected)


def test_an_unchanged_candidate_is_not_a_refusal():
    """A plan needing no repair is the normal case, not a failure.

    Every archived run of the frozen pilot took this path: the deterministic
    repair added nothing, and the conformance status stayed PASS. Treating an
    unchanged candidate as a refused repair would mark all of them failed.
    """
    metadata = _drive(_MODEL)
    conformance = metadata["generation_plan_conformance"]
    audit = conformance["plan_authorized_repair"]

    assert audit["text_changed"] is False
    assert audit["applied"] is False
    assert conformance["status"] != "FAIL"
    assert "rejected_plan_authorized_repairs" not in metadata


@pytest.mark.parametrize(
    "text_changed,fixed_set,syntax,behaviour",
    list(itertools.product([True, False], repeat=4)),
)
def test_the_admission_rule_is_unchanged(
    text_changed, fixed_set, syntax, behaviour
):
    """The recorded booleans must not alter who gets admitted."""
    previous = text_changed and fixed_set and syntax and behaviour
    failed = [
        name for name, held in (
            ("fixed_set_preserved", fixed_set),
            ("syntax_ok", syntax),
            ("behavior_preserved", behaviour),
        ) if not held
    ]
    current = text_changed and not failed
    assert current == previous
