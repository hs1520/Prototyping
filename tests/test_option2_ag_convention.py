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


def test_published_syntax_agrees_with_the_emitter():
    """The deterministic emitter is the reference implementation of the notation,
    so a template published to an author must match what it actually emits.

    A rule once demanded guarded competing transitions while the transition
    template it published had no guard slot. The generator invented `guard <expr>`,
    which does not parse, and lost three of four feedback rounds to it.
    """
    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package

    emitted = emit_ag_package(REQ_SAFE_005_CHAIN)
    transitions = [
        line.strip() for line in emitted.splitlines()
        if line.strip().startswith("transition ")
    ]
    assert transitions, "emitter reference produced no transitions"
    guarded = [line for line in transitions if " if " in line]
    assert guarded, "expected the reference to contain a guarded transition"

    rendered = render_authoring_rules()
    # the guard keyword the emitter actually uses must be the one we publish
    assert "if <expr>" in rendered or "if <TRIGGER>" in rendered
    # and we must never publish a keyword the grammar does not have
    assert "guard <" not in rendered
    for line in guarded:
        assert " guard " not in line, (
            "emitter uses a guard keyword the published rules do not describe"
        )


def test_guard_concepts_are_declared_in_their_own_state_def():
    """Across every encoded chain the emitter declares each guard concept as an
    attribute of the state def that uses it — a state machine cannot see the
    attributes of the contract it realizes. The generator produced undeclared
    guard references until this was published, so the rules must state it and the
    reference must keep obeying it.
    """
    import re

    from src.prototyping import ag_chains
    from src.prototyping.ag_emitter import emit_ag_package

    chains = [
        getattr(ag_chains, name) for name in dir(ag_chains)
        if name.startswith("REQ_") and name.endswith("_CHAIN")
    ]
    assert chains, "no encoded chains found"
    checked = 0
    for chain in chains:
        for block in re.finditer(
            r"state def (\w+) \{(.*?)\n    \}", emit_ag_package(chain), re.S
        ):
            declared = set(re.findall(r"attribute (\w+)\s*:", block.group(2)))
            referenced = set(
                re.findall(r"\bif\s+(?:not\s+)?(\w+)", block.group(2))
            )
            assert referenced <= declared, (
                f"{block.group(1)} guards on undeclared "
                f"{sorted(referenced - declared)}"
            )
            checked += len(referenced)
    assert checked, "no guarded transitions exercised this invariant"

    assert "same `state def`" in render_authoring_rules()


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
    from src.prototyping import ag_chains

    # Strip comments AND docstrings: a line of prose explaining which answer was
    # removed is not the checker holding one, and counting it kept the debt below
    # looking open after it had been closed.
    code_lines, in_doc = [], False
    for line in open(ag_contracts.__file__).read().splitlines():
        stripped = line.strip()
        if in_doc:
            if '"""' in stripped:
                in_doc = False
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(('"""', 'r"""')):
            if stripped.count('"""') == 1:
                in_doc = True
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    # Derived from the chains rather than hand-listed. A hand-list only covers the
    # answers already thought of: the first version of this test named REQ_SAFE_005
    # literals and missed a per-requirement table of REQ_SAFE_004's and
    # REQ_SAFE_008's reviewed invariants sitting in the same module.
    def _chain_answers(chain) -> set[str]:
        found = {chain.system_contract, chain.source_requirement}
        found.update(component.name for component in chain.components)
        if chain.priority:
            found.update(chain.priority.members)
            found.add(chain.priority.response_set_id)
        found.update(item.invariant_id for item in chain.invariants)
        found.update(item.source_id for item in chain.invariants)
        return {item for item in found if item}

    timed, invariant_patterned = {"_SOURCE_PATTERN_PROFILE"}, set()
    for name in dir(ag_chains):
        if not (name.startswith("REQ_") and name.endswith("_CHAIN")):
            continue
        chain = getattr(ag_chains, name)
        target = (
            timed if chain.pattern == "TRIGGERED_TIMED_FAILSAFE_RESPONSE"
            else invariant_patterned
        )
        target |= _chain_answers(chain)
    answers = timed

    # Gold-blindness is complete for the TIMED-FAILSAFE path and NOT for the
    # invariant patterns, which still compare against reviewed invariant sets and
    # against per-chain state-machine shapes (exact state names, signals and
    # transition sets). Generalising those is a design task, not a refactor, so the
    # debt is recorded explicitly here rather than hidden: a PASS on REQ_SAFE_004 or
    # REQ_SAFE_008 is partly "reproduce the reviewed answer", and neither chain can
    # carry an honest generation-accuracy claim until this is closed.
    # See docs/R2_GENERATION_FINDINGS.md §6 (limitation).
    leaked = sorted(answer for answer in answers if answer in code)
    assert not leaked, (
        f"the runtime checker compares against reviewed answers: {leaked}. Those "
        "are what the LLM arms are measured on — score them in the evaluator."
    )
    # The debt is CLOSED: the invariant patterns' obligations are now derived from
    # each chain's own declared invariants, so no reviewed answer remains here for
    # any pattern.
    remaining = sorted(item for item in invariant_patterned if item in code)
    assert not remaining, remaining


def test_every_pattern_role_the_checker_demands_is_published():
    """The checker refuses an invariant set that leaves one of a pattern's roles
    unfilled. Which roles a pattern has is the pattern's definition — CONVENTION,
    like the notation itself — so it must be published; which concepts fill them
    stays the author's derivation.

    Two measured chains failed on exactly this: well-formed invariants that did not
    cover the roles, because no prompt had ever said the pattern had roles. The two
    tables are pinned to each other here so a role added to the checker cannot go
    unstated.
    """
    published = {
        item.pattern: item.roles
        for item in ag_convention.INVARIANT_ROLE_OBLIGATIONS
    }
    assert published == dict(ag_contracts.PATTERN_INVARIANT_ROLES), (
        "the checker's required roles and the published ones have diverged; a "
        "role the author is never told is unsatisfiable by any author"
    )
    rendered = ag_convention.render_invariant_role_rules()
    for pattern, roles in published.items():
        assert pattern in rendered
        for role in roles:
            assert role in rendered, f"{pattern} role {role!r} is not stated"


def test_the_role_rules_reach_the_authored_sysml_rules_too():
    """The obligation is the checker's, not one mode's: whichever way a package is
    produced, an unfilled role is INVARIANT_SEMANTICS_INVALID."""
    rendered = render_authoring_rules()
    for item in ag_convention.INVARIANT_ROLE_OBLIGATIONS:
        for role in item.roles:
            assert role in rendered, f"{item.pattern} role {role!r} unstated"


def test_the_published_role_shapes_match_the_reference_invariants():
    """Read off the reference implementation rather than assumed (defect class:
    a rule published from a guess about what the checker wanted).

    Every shape the rules describe must be one the encoded chains actually use, so
    an author following them writes invariants the checker can read roles from.
    """
    from src.prototyping import ag_chains
    from src.prototyping.ag_contracts import (
        _invariant_roles, _startup_inhibit_roles,
    )
    from src.prototyping.ag_emitter import emit_ag_package
    from src.prototyping.ag_extractor import extract_ag_graph

    derive = {
        "STARTUP_INHIBIT": _startup_inhibit_roles,
        "LOCKED_UNTIL_AUTHORISED_RELEASE": _invariant_roles,
    }
    exercised = set()
    for name in dir(ag_chains):
        if not (name.startswith("REQ_") and name.endswith("_CHAIN")):
            continue
        chain = getattr(ag_chains, name)
        if chain.pattern not in derive:
            continue
        graph = extract_ag_graph(emit_ag_package(chain))
        roles = derive[chain.pattern](graph)
        for role in ag_contracts.PATTERN_INVARIANT_ROLES[chain.pattern]:
            assert roles.get(role), (
                f"{chain.source_requirement}: the reference itself does not fill "
                f"the {role!r} role the rules publish"
            )
        exercised.add(chain.pattern)
    assert exercised == set(ag_contracts.PATTERN_INVARIANT_ROLES)


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


def test_core_tier_renders_a_strict_subset_of_the_full_rule_set():
    """Rule-set size is an experimental variable, so the tiers must actually
    partition the rules — a CORE arm that silently renders everything would make
    the ablation measure nothing."""
    full = render_authoring_rules()
    core = render_authoring_rules(tiers=(ag_convention.CORE,))
    full_rules = {line.split(". ", 1)[-1] for line in full.splitlines()}
    core_rules = {line.split(". ", 1)[-1] for line in core.splitlines()}
    assert core_rules < full_rules, "CORE must be a strict subset of the full set"
    assert core_rules, "CORE must not be empty"
    # the tiers together must account for every rendered rule, or an obligation
    # would be unstated in every arm
    refinement = render_authoring_rules(tiers=(ag_convention.REFINEMENT,))
    refinement_rules = {line.split(". ", 1)[-1] for line in refinement.splitlines()}
    assert core_rules | refinement_rules == full_rules
    assert not (core_rules & refinement_rules)
    # CORE must still carry what makes a package parse at all
    assert "private import ScalarValues::*;" in core


def test_obligation_ids_are_unique():
    ids = [item.obligation_id for item in ALL_OBLIGATIONS]
    assert len(ids) == len(set(ids))


def test_a_convention_without_a_rule_is_rejected_at_construction():
    with pytest.raises(ValueError, match="must state its rule"):
        Obligation("X", CONVENTION)


def test_a_spec_valued_obligation_without_withheld_text_is_rejected():
    with pytest.raises(ValueError, match="must record what"):
        Obligation("X", SPEC_VALUED, authoring_rule="something")
