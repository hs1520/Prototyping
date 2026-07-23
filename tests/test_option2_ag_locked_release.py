"""Third candidate: current-source REQ_SAFE_008 plus derived lock constraints.

Exercises a third bounded safety pattern, LOCKED_UNTIL_AUTHORISED_RELEASE, whose
defining obligation is *default-safe*: the power-on (initial) state must be the
locked state, distinct from the guarded release state. The chain declares its
pattern in the committed model (``safety_pattern=`` on the system contract), which
is the authority the conformance checker dispatches on — three patterns (one
timed, two untimed) cannot be told apart by topology alone.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.prototyping.ag_gold_template import GOLD_STATUS_DRAFT, build_gold_draft
from src.prototyping.ag_assurance import (
    PatternCase,
    check_safety_pattern_conformance,
    route_failure_diagnostics,
)
from src.prototyping.ag_chains import (
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_008_CHAIN,
    select_ag_chains,
)
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph
from src.simulation.syntax_checker import check_syntax

_SRC = (
    "The payload-release actuator shall default to the mechanically locked "
    "state upon power-on, before any arming or flight authorisation."
)


def _model() -> str:
    return (
        "package Src { requirement def REQ_SAFE_008 { doc /* " + _SRC + " */ } }\n"
        + emit_ag_package(REQ_SAFE_008_CHAIN)
    )


def test_safe008_gold_uses_the_exact_current_repository_source():
    from examples.drone_system_v2 import DRONE_REQUIREMENTS

    current = next(
        item for item in DRONE_REQUIREMENTS
        if item.startswith("REQ-SAFE-008:")
    )
    assert _GOLD_SRC == current


def test_locked_release_chain_is_valid_sysml_and_passes_the_ag_trace():
    model = _model()
    syntax = check_syntax(model)
    assert syntax.has_errors is False
    assert syntax.score == 1.0

    report = check_ag_graph(extract_ag_graph(model, revision=1))
    assert report.verdict == "PASS", [d.code for d in report.diagnostics]
    # a guard/ordering invariant, not a timed chain: no timing composition
    assert report.timing["ok"] is None
    assert len(report.allocations) == 4
    # Lifecycle events are typed interface inputs, not permanent conjunctive
    # assumptions; only the command transaction predicates are discharged.
    assert len(report.discharge_edges) == 2


def test_pattern_is_taken_from_the_declared_model_annotation():
    graph = extract_ag_graph(_model(), revision=1)
    # the committed model is the authority for which pattern applies
    assert graph.system.declared_pattern == "LOCKED_UNTIL_AUTHORISED_RELEASE"
    report = check_ag_graph(graph)
    pattern = check_safety_pattern_conformance(graph, report)
    assert pattern["verdict"] == "PASS"
    assert {c["pattern"] for c in pattern["cases"]} == {
        "LOCKED_UNTIL_AUTHORISED_RELEASE"
    }
    # the locked default is present and distinct from the release state
    assert all(c["default_safe_present"] is True for c in pattern["cases"])
    assert all(c["timing_criterion_present"] is False for c in pattern["cases"])


def test_default_safe_obligation_is_specific_to_the_locked_release_pattern():
    """A power-on-released model fails LOCKED_UNTIL_AUTHORISED_RELEASE even though
    the same topology satisfies the untimed STARTUP_INHIBIT core — proof the third
    pattern is a genuine obligation, not a relabel of the second."""
    core = dict(
        trigger_present=True,
        reachable_response=True,
        entry_action_present=True,
        timing_criterion_present=False,
        invariant_preserved=True,
    )
    # default state is itself a release state (powers on released): the guard is
    # violated only for the locked-until-release pattern.
    locked = PatternCase(
        contract="c", pattern="LOCKED_UNTIL_AUTHORISED_RELEASE",
        default_safe_present=False, **core,
    )
    inhibit = PatternCase(
        contract="c", pattern="STARTUP_INHIBIT",
        default_safe_present=False, **core,
    )
    assert locked.status == "FAIL"
    assert inhibit.status == "PASS"
    # and with a genuine locked default the locked pattern passes
    assert PatternCase(
        contract="c", pattern="LOCKED_UNTIL_AUTHORISED_RELEASE",
        default_safe_present=True, **core,
    ).status == "PASS"


def test_three_chains_use_three_distinct_patterns():
    patterns = {
        REQ_SAFE_005_CHAIN.pattern,
        REQ_SAFE_004_CHAIN.pattern,
        REQ_SAFE_008_CHAIN.pattern,
    }
    assert patterns == {
        "TRIGGERED_TIMED_FAILSAFE_RESPONSE",
        "STARTUP_INHIBIT",
        "LOCKED_UNTIL_AUTHORISED_RELEASE",
    }
    chosen = select_ag_chains([
        "REQ-SAFE-005: deploy parachute",
        "REQ-SAFE-004: prevent arming",
        "REQ-SAFE-008: keep payload locked until authorised",
    ])
    assert {c.source_requirement for c in chosen} == {
        "REQ_SAFE_005", "REQ_SAFE_004", "REQ_SAFE_008",
    }


def test_locked_release_pattern_needs_real_topology_not_a_label():
    model = _model()
    broken = "\n".join(
        line for line in model.splitlines() if "dependency realize" not in line
    )
    report = check_ag_graph(extract_ag_graph(broken, revision=1))
    pattern = check_safety_pattern_conformance(
        extract_ag_graph(broken, revision=1), report
    )
    assert pattern["verdict"] == "FAIL"
    routes = route_failure_diagnostics(
        report.diagnostics, source_requirement="REQ_SAFE_008",
        realization_links=report.realization_links,
    )
    assert len(routes["failures"]) >= 1


def test_locked_release_checks_actions_authorisation_and_power_loss_relock():
    mutations = (
        (
            "state lockedUnpowered "
            "{ entry action setPayloadLockedForDeenergiseToLock; }",
            "state lockedUnpowered { entry action setPayloadUnlocked; }",
        ),
        (
            "accept AuthorisedReleaseCommandReceivedSignal "
            "then unlockedPowered;",
            "accept ReceivedReleaseCommandSignal then unlockedPowered;",
        ),
        (
            "accept PowerLostSignal then lockedUnpowered;",
            "accept PowerOnSignal then lockedUnpowered;",
        ),
    )
    for before, after in mutations:
        broken = _model().replace(before, after)
        graph = extract_ag_graph(broken, revision=1)
        report = check_ag_graph(graph)
        pattern = check_safety_pattern_conformance(
            graph, report
        )
        assert report.verdict == "FAIL"
        assert "PATTERN_TOPOLOGY_INCOMPLETE" in {
            diagnostic.code for diagnostic in report.diagnostics
        }
        assert pattern["verdict"] == "FAIL"
        mechanism = next(
            case for case in pattern["cases"]
            if case["contract"] == "PayloadLockMechanismContract"
        )
        assert mechanism["default_safe_present"] is False


def test_authorisation_history_is_guarded_and_cleared_per_transaction():
    mutations = (
        (
            "if authorisationDataValid then authorisationGranted;",
            "then authorisationGranted;",
        ),
        (
            "transition clearOnPowerLoss first authorisationGranted "
            "accept PowerLostSignal then awaitingAuthorisation;",
            "",
        ),
        (
            "transition clearOnNewPowerCycle first authorisationGranted "
            "accept PowerOnSignal then awaitingAuthorisation;",
            "",
        ),
    )
    for before, after in mutations:
        broken = _model().replace(before, after)
        graph = extract_ag_graph(broken, revision=1)
        pattern = check_safety_pattern_conformance(
            graph, check_ag_graph(graph)
        )
        gateway = next(
            case for case in pattern["cases"]
            if case["contract"] == "ReleaseCommandGatewayContract"
        )
        assert gateway["invariant_preserved"] is False
        assert pattern["verdict"] == "FAIL"


def test_locked_release_checker_rejects_extra_unauthorised_unlock_transition():
    approved = (
        "transition authorisedUnlock first lockedPowered "
        "accept AuthorisedReleaseCommandReceivedSignal then unlockedPowered;"
    )
    broken = _model().replace(
        approved,
        approved
        + "\n        transition unauthorisedUnlock first lockedPowered "
        "accept ReceivedReleaseCommandSignal then unlockedPowered;",
    )
    report = check_ag_graph(extract_ag_graph(broken, revision=1))
    assert report.verdict == "FAIL"
    assert "PATTERN_TOPOLOGY_INCOMPLETE" in {
        diagnostic.code for diagnostic in report.diagnostics
    }


def test_locked_release_checker_requires_states_and_deenergise_invariant():
    missing_state = _model().replace(
        "state lockedPowered { entry action maintainPayloadLocked; }",
        "state powered { entry action maintainPayloadLocked; }",
    )
    report = check_ag_graph(extract_ag_graph(missing_state, revision=1))
    assert report.verdict == "FAIL"
    assert "PATTERN_TOPOLOGY_INCOMPLETE" in {
        diagnostic.code for diagnostic in report.diagnostics
    }

    without_deenergise = "\n".join(
        line for line in _model().splitlines()
        if "inv__SAFE008_DEENERGISE_TO_LOCK" not in line
    )
    report = check_ag_graph(extract_ag_graph(without_deenergise, revision=1))
    assert report.verdict == "FAIL"
    assert "INVARIANT_SEMANTICS_INVALID" in {
        diagnostic.code for diagnostic in report.diagnostics
    }

    without_mechanism_guarantee = _model().replace(
        "require constraint g_deenergiseToLock { deenergiseToLock }",
        "",
    )
    report = check_ag_graph(
        extract_ag_graph(without_mechanism_guarantee, revision=1)
    )
    assert report.verdict == "FAIL"
    assert "PATTERN_TOPOLOGY_INCOMPLETE" in {
        diagnostic.code for diagnostic in report.diagnostics
    }


def test_locked_release_runtime_checker_requires_invariant_semantics_and_pattern():
    model = _model()
    without_invariants = "\n".join(
        line for line in model.splitlines()
        if "require constraint inv__" not in line
    )
    report = check_ag_graph(extract_ag_graph(without_invariants, revision=1))
    assert report.verdict == "FAIL"
    assert "INVARIANT_SEMANTICS_MISSING" in {
        diagnostic.code for diagnostic in report.diagnostics
    }

    wrong_pattern = model.replace(
        "safety_pattern=LOCKED_UNTIL_AUTHORISED_RELEASE",
        "safety_pattern=STARTUP_INHIBIT",
    )
    report = check_ag_graph(extract_ag_graph(wrong_pattern, revision=1))
    assert report.verdict == "FAIL"
    assert "PATTERN_DECLARATION_INCONSISTENT" in {
        diagnostic.code for diagnostic in report.diagnostics
    }


_GOLD_SRC = (
    "REQ-SAFE-008: The payload-release actuator shall default to the "
    "mechanically locked state upon power-on, before any arming or flight "
    "authorisation."
)
_DRAFT_FILE = Path("docs/gold/REQ_SAFE_008_ag_gold.draft.json")


def test_committed_gold_draft_is_regenerable_and_unfrozen():
    on_disk = json.loads(_DRAFT_FILE.read_text(encoding="utf-8"))
    assert on_disk["status"] == GOLD_STATUS_DRAFT
    assert on_disk["artifact_role"] == "EVALUATOR_GOLD"
    assert on_disk["chain_id"] == "REQ_SAFE_008"
    # the fully-specified chain leaves the reviewer no UNRESOLVED discharge edges
    assert all(e["by"] is not None for e in on_disk["discharge_edges"])
    # regenerable: the committed draft equals a fresh generation from the spec
    assert on_disk == build_gold_draft(REQ_SAFE_008_CHAIN, source_text=_GOLD_SRC)
