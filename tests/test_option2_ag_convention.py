"""The convention spec must stay complete against the checker (design §6, §9).

Three measured Vertex runs stalled because the checker enforced rules that existed
only inside it and the deterministic emitter: an author was never told to emit a
realization dependency, to import the library defining ``DurationValue``, or to
name a response action ``set<Concept>``. Each cost a run to discover.

These tests make that class of defect structurally impossible: a checker diagnostic
code with no entry in ``ag_convention`` fails the suite, and so does a withheld
value that reaches the rendered authoring rules.
"""
from __future__ import annotations

import re

import pytest

from src.prototyping import ag_contracts, ag_convention
from src.prototyping.ag_convention import (
    ALL_OBLIGATIONS,
    CONVENTION,
    SPEC_VALUED,
    Obligation,
    render_authoring_rules,
    withheld_values,
)


def _checker_codes() -> set[str]:
    return {
        value for name, value in vars(ag_contracts).items()
        if name.startswith("CODE_") and isinstance(value, str)
    }


def _priority_obligation_names() -> set[str]:
    """The obligation names the priority check reports, read from its source.

    Read from source rather than hard-coded here: a name added to the checker's
    obligation tuple must show up as a missing spec entry, not silently pass.
    """
    source = open(ag_contracts.__file__).read()
    block = source[source.index("obligations = ("):source.index("unmet = [")]
    return set(re.findall(r'\(\s*"(\w+)"\s*,', block))


def test_every_checker_diagnostic_code_has_a_convention_entry():
    documented = {item.obligation_id for item in ALL_OBLIGATIONS}
    missing = _checker_codes() - documented
    assert not missing, (
        f"checker codes with no ag_convention entry: {sorted(missing)}. An "
        "obligation the author is never told is unsatisfiable by any author."
    )


def test_every_named_priority_obligation_has_a_convention_entry():
    documented = {item.obligation_id for item in ALL_OBLIGATIONS}
    missing = _priority_obligation_names() - documented
    assert not missing, (
        f"priority obligations with no ag_convention entry: {sorted(missing)}"
    )


def test_the_spec_documents_nothing_the_checker_does_not_enforce():
    """A stale rule is as harmful as a missing one — it instructs the author to
    satisfy something no longer checked."""
    enforced = _checker_codes() | _priority_obligation_names()
    # gate obligations have no A/G code; they are enforced by the syntax gate
    gate_ids = {item.obligation_id for item in ag_convention.GATE_OBLIGATIONS}
    stale = {item.obligation_id for item in ALL_OBLIGATIONS} - enforced - gate_ids
    assert not stale, f"ag_convention entries no longer enforced: {sorted(stale)}"


def test_rendered_rules_leak_no_withheld_value():
    rendered = render_authoring_rules()
    for value in withheld_values():
        assert value not in rendered
    # the reviewed answers themselves must never appear, however phrased
    for secret in ("CONTROLLED_BATTERY_LANDING", "COMMUNICATION_LOSS_SAFE_LANDING",
                   "LOW_BATTERY_RETURN_TO_BASE", "_SAFE005_PRIORITY_MEMBERS",
                   "_SOURCE_PATTERN_PROFILE"):
        assert secret not in rendered


def test_the_orchestrator_imports_cleanly_on_its_own():
    """`src.prototyping` imports the orchestrator, so a module-level import of an
    ag_* module from the orchestrator is circular whenever the orchestrator is
    imported first. The suite does not catch it because conftest imports in the
    other order — only a fresh interpreter entering through the orchestrator does.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import src.agents.orchestrator"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-800:]


def test_the_checker_holds_no_reviewed_answer():
    """The runtime checker must stay gold-blind (AG_CHECKER_VERSION ag-bounded-5).

    It documents PASS as meaning the graph is complete and *internally* compatible.
    It previously also compared against REQ_SAFE_005's reviewed response set,
    precedence ordering and winning response, and against a requirement-to-pattern
    table — so a gold-blind verdict depended on the very facts the LLM-authored arm
    measures, and the priority topology could only be recalled, never derived.
    Those comparisons belong to `ag_eval_semantics.priority_agreement`.
    """
    source = open(ag_contracts.__file__).read()
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for answer in (
        "CONTROLLED_BATTERY_LANDING",      # reviewed response-set members
        "COMMUNICATION_LOSS_SAFE_LANDING",
        "LOW_BATTERY_RETURN_TO_BASE",
        "PARACHUTE_DEPLOYMENT",           # the reviewed winning response
        "criticalPropulsionFailureDetected",  # the reviewed trigger
        "parachuteDeployed",              # the reviewed observation
        "RecoverySystemContract",         # reviewed element names
        "SafetyResponseArbiterContract",
        "RecoveryPowerSupplyContract",
        "_SOURCE_PATTERN_PROFILE",        # requirement-to-pattern table
    ):
        assert answer not in code, (
            f"{answer!r} is a reviewed answer for a specific chain; the runtime "
            "checker must not compare against it — score it in the evaluator"
        )


def test_every_convention_entry_actually_tells_the_author_what_to_do():
    for item in ALL_OBLIGATIONS:
        if item.category == CONVENTION:
            assert item.authoring_rule, item.obligation_id
            assert len(item.authoring_rule) > 40, (
                f"{item.obligation_id}: too terse to be actionable"
            )


def test_spec_valued_entries_record_what_is_withheld():
    for item in ALL_OBLIGATIONS:
        if item.category == SPEC_VALUED:
            assert item.withheld, item.obligation_id


def test_obligation_ids_are_unique():
    ids = [item.obligation_id for item in ALL_OBLIGATIONS]
    assert len(ids) == len(set(ids))


def test_a_convention_without_a_rule_is_rejected_at_construction():
    with pytest.raises(ValueError, match="must state its rule"):
        Obligation("X", CONVENTION)


def test_a_spec_valued_obligation_without_withheld_text_is_rejected():
    with pytest.raises(ValueError, match="must record what"):
        Obligation("X", SPEC_VALUED, authoring_rule="something")
