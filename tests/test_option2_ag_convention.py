"""The convention spec stays complete against the checker (design §6, §9).

Three Vertex runs stalled on rules that existed only inside the checker and the
deterministic emitter: nothing told the author to emit a realization dependency,
import the library defining ``DurationValue``, or name a response action
``set<Concept>``. A checker diagnostic code with no ``ag_convention`` entry now
fails the suite, as does a withheld value that reaches the rendered authoring
rules.
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

    Hard-coding them here would let a name added to the checker's obligation tuple
    pass instead of showing up as a missing spec entry.
    """
    source = open(ag_contracts.__file__).read()
    block = source[source.index("obligations = ("):source.index("unmet = [")]
    return set(re.findall(r'\(\s*"(\w+)"\s*,', block))


def test_checker_codes_documented():
    documented = {item.obligation_id for item in ALL_OBLIGATIONS}
    missing = _checker_codes() - documented
    assert not missing, (
        f"checker codes with no ag_convention entry: {sorted(missing)}. An "
        "obligation the author is never told is unsatisfiable by any author."
    )


def test_priority_obligations_documented():
    documented = {item.obligation_id for item in ALL_OBLIGATIONS}
    missing = _priority_obligation_names() - documented
    assert not missing, (
        f"priority obligations with no ag_convention entry: {sorted(missing)}"
    )


def test_no_stale_convention_entries():
    enforced = _checker_codes() | _priority_obligation_names()
    # gate obligations have no A/G code; they are enforced by the syntax gate
    gate_ids = {item.obligation_id for item in ag_convention.GATE_OBLIGATIONS}
    stale = {item.obligation_id for item in ALL_OBLIGATIONS} - enforced - gate_ids
    assert not stale, f"ag_convention entries no longer enforced: {sorted(stale)}"


def test_rules_leak_no_withheld_value():
    rendered = render_authoring_rules()
    for value in withheld_values():
        assert value not in rendered
    for secret in ("CONTROLLED_BATTERY_LANDING", "COMMUNICATION_LOSS_SAFE_LANDING",
                   "LOW_BATTERY_RETURN_TO_BASE", "_SAFE005_PRIORITY_MEMBERS",
                   "_SOURCE_PATTERN_PROFILE"):
        assert secret not in rendered


def test_syntax_matches_emitter():
    """The deterministic emitter is the reference implementation of the notation, so a
    template published to an author matches what it emits.

    A rule once demanded guarded competing transitions while the transition template
    it published had no guard slot; the generator invented `guard <expr>`, which does
    not parse, and lost three of four feedback rounds to it.
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
    assert "if <expr>" in rendered or "if <TRIGGER>" in rendered
    assert "guard <" not in rendered
    for line in guarded:
        assert " guard " not in line, (
            "emitter uses a guard keyword the published rules do not describe"
        )


def test_trigger_rule_names_concept():
    """The checker matches lexically, so saying "compatible" alone leaves the author
    unable to infer the accept-signal name.
    """
    rendered = render_authoring_rules()
    assert "`<Concept>Signal`" in rendered
    assert "`SensorFailureReportedSignal`" in rendered
    assert "not `SensorFailureSignal`" in rendered


def test_extractor_conventions_published():
    rendered = render_authoring_rules()
    assert "discharge<Concept>__to__<ConsumerContract>" in rendered
    assert "inv__<invariant_id>__source__<source_id>__kind__<source_kind>" in rendered
    assert "INSIDE the system contract" in rendered
    assert "enum <MEMBER>;" in rendered
    assert "lifecycle/interface input" in rendered
    assert "one atomic Boolean identifier" in rendered
    assert "require constraint g_observed" in rendered
    assert "common source" in rendered
    assert "distinct power-loss event" in rendered


def test_guard_concepts_declared():
    """Across every encoded chain the emitter declares each guard concept as an
    attribute of the state def that uses it, because a state machine cannot see the
    attributes of the contract it realizes. The generator produced undeclared guard
    references until this was published, so the rules state it and the reference
    keeps obeying it.
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


def test_orchestrator_imports_alone():
    """`src.prototyping` imports the orchestrator, so a module-level import of an ag_*
    module from the orchestrator is circular when the orchestrator is imported
    first. conftest imports in the other order, so only a fresh interpreter entering
    through the orchestrator catches it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import src.agents.orchestrator"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-800:]


def test_checker_holds_no_answer():
    """The runtime checker stays gold-blind (AG_CHECKER_VERSION ag-bounded-5).

    PASS means the graph is complete and internally compatible. It once also compared
    against REQ_SAFE_005's reviewed response set, precedence ordering and winning
    response, and against a requirement-to-pattern table, so the verdict depended on
    the facts the LLM-authored arm measures and the priority topology could only be
    recalled. Those comparisons belong to `ag_eval_semantics.priority_agreement`.
    """
    from src.prototyping import ag_chains

    # Strip comments and docstrings: prose naming a removed answer is not the checker
    # holding one, and counting it kept the debt looking open after it had been
    # closed.
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
    # Derived from the chains rather than hand-listed: the first version named
    # REQ_SAFE_005 literals and missed a per-requirement table of REQ_SAFE_004's and
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

    # Gold-blindness is complete for the TIMED-FAILSAFE path but not for the invariant
    # patterns, which still compare against reviewed invariant sets and per-chain
    # state-machine shapes (state names, signals, transition sets). Generalising them
    # is a design task, not a refactor, so the debt is recorded here: a PASS on
    # REQ_SAFE_004 or REQ_SAFE_008 is partly "reproduce the reviewed answer", and
    # neither chain supports a generation-accuracy claim until this closes.
    # See docs/R2_GENERATION_FINDINGS.md §6 (limitation).
    leaked = sorted(answer for answer in answers if answer in code)
    assert not leaked, (
        f"the runtime checker compares against reviewed answers: {leaked}. Those "
        "are what the LLM arms are measured on — score them in the evaluator."
    )
    # Debt closed: the invariant patterns' obligations are derived from each chain's
    # own declared invariants, so no reviewed answer remains here for any pattern.
    remaining = sorted(item for item in invariant_patterned if item in code)
    assert not remaining, remaining


def test_pattern_roles_published():
    """The checker refuses an invariant set that leaves a pattern role unfilled.

    Which roles a pattern has is the pattern's definition, so it is published; which
    concepts fill them stays the author's derivation. Two measured chains failed
    with well-formed invariants that did not cover the roles because no prompt said
    the pattern had roles, so the two tables are pinned to each other here.
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


def test_role_rules_in_authoring_rules():
    rendered = render_authoring_rules()
    for item in ag_convention.INVARIANT_ROLE_OBLIGATIONS:
        for role in item.roles:
            assert role in rendered, f"{item.pattern} role {role!r} unstated"


def test_role_shapes_match_reference():
    """Read off the reference implementation rather than guessed at (defect class: a
    rule published from a guess about what the checker wanted).

    Every shape the rules describe is one the encoded chains use, so an author
    following them writes invariants the checker can read roles from.
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


def test_convention_entries_actionable():
    for item in ALL_OBLIGATIONS:
        if item.category == CONVENTION:
            assert item.authoring_rule, item.obligation_id
            assert len(item.authoring_rule) > 40, (
                f"{item.obligation_id}: too terse to be actionable"
            )


def test_spec_valued_record_withheld():
    for item in ALL_OBLIGATIONS:
        if item.category == SPEC_VALUED:
            assert item.withheld, item.obligation_id


def test_core_tier_strict_subset():
    """Rule-set size is an experimental variable, so the tiers partition the rules; a
    CORE arm that silently rendered everything would make the ablation measure
    nothing.
    """
    full = render_authoring_rules()
    core = render_authoring_rules(tiers=(ag_convention.CORE,))
    full_rules = {line.split(". ", 1)[-1] for line in full.splitlines()}
    core_rules = {line.split(". ", 1)[-1] for line in core.splitlines()}
    assert core_rules < full_rules, "CORE must be a strict subset of the full set"
    assert core_rules, "CORE must not be empty"
    # the tiers together cover every rendered rule, or an obligation is unstated
    # in every arm
    refinement = render_authoring_rules(tiers=(ag_convention.REFINEMENT,))
    refinement_rules = {line.split(". ", 1)[-1] for line in refinement.splitlines()}
    assert core_rules | refinement_rules == full_rules
    assert not (core_rules & refinement_rules)
    assert "private import ScalarValues::*;" in core


def test_obligation_ids_unique():
    ids = [item.obligation_id for item in ALL_OBLIGATIONS]
    assert len(ids) == len(set(ids))


def test_convention_without_rule_rejected():
    with pytest.raises(ValueError, match="must state its rule"):
        Obligation("X", CONVENTION)


def test_spec_valued_without_withheld():
    with pytest.raises(ValueError, match="must record what"):
        Obligation("X", SPEC_VALUED, authoring_rule="something")
