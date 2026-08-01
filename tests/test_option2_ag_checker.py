"""R2-BBAG bounded Assume-Guarantee checker — Increment 2 tests.

Covers the worked REQ_SAFE_005 chain (design §7), the validated SysML convention
(§6.2), and the required-test invariants (§16): one owner per component guarantee,
assumption discharge, incompatible units/timing detection, and honest failure
localisation (a genuine gap is not mislabelled circular).
"""
from __future__ import annotations

from src.prototyping.ag_contracts import (
    AGEdge,
    AGGraph,
    Assumption,
    Contract,
    Guarantee,
    READY,
    INCOMPLETE,
    check_ag_graph,
    CODE_ASSUMPTION_UNDISCHARGED,
    CODE_CIRCULAR_ASSUMPTION,
    CODE_GUARANTEE_NO_OWNER,
    CODE_TIMING_BUDGET_EXCEEDED,
    CODE_UNIT_INCOMPATIBLE,
    CODE_DECOMPOSITION_INSUFFICIENT,
)
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_emitter import emit_ag_package
from src.simulation.syntax_checker import check_syntax


# REQ_SAFE_005 — critical propulsion failure → parachute deployment (design §7).
REQ_SAFE_005_SYSML = (
    "package Source { requirement def REQ_SAFE_005 { doc /* source */ } }\n"
    + emit_ag_package(REQ_SAFE_005_CHAIN)
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def test_convention_passes_the_syside_gate():
    # §6.2: the bounded A/G convention must be valid SysML, not invented keywords.
    result = check_syntax(REQ_SAFE_005_SYSML)
    assert result.has_errors is False
    assert result.score == 1.0


def test_req_safe_005_chain_extracts_and_passes():
    graph = extract_ag_graph(REQ_SAFE_005_SYSML, revision=7)
    assert graph.system is not None
    assert graph.system.name == "SystemParachuteContract"
    assert graph.system.observation == "parachuteDeployed"
    assert {c.name for c in graph.components} == {
        "SafetyResponseArbiterContract",
        "RecoveryPowerSupplyContract",
        "RecoverySystemContract",
    }

    report = check_ag_graph(graph)
    assert report.verdict == "PASS", _codes(report)
    assert report.system_completeness == READY
    assert set(report.component_completeness.values()) == {READY}
    # Detection is the boundary event; 0.10 + 0.35 = 0.45 <= 0.50 s.
    assert report.timing["sum"] == 0.45
    assert report.timing["ok"] is True
    # every non-environment assumption is discharged; environment ones are marked
    assert report.discharge[
        "RecoverySystemContract.parachuteDeploymentCommand"
    ] == "discharged"
    assert report.discharge[
        "RecoverySystemContract.recoveryActuationPowerAvailable"
    ] == "discharged"


def test_compound_component_guarantee_is_incomplete_not_an_action_target():
    compound = REQ_SAFE_005_SYSML.replace(
        "require constraint g_parachuteResponseSelected "
        "{ parachuteResponseSelected }",
        "require constraint g_parachuteResponseSelected "
        "{ parachuteResponseSelected and airborne }",
    )
    report = check_ag_graph(extract_ag_graph(compound))
    assert (
        report.component_completeness["SafetyResponseArbiterContract"]
        == INCOMPLETE
    )
    assert "COMPONENT_GUARANTEE_NONATOMIC" in _codes(report)


def test_exactly_one_owner_per_component_guarantee():
    # §16: every component guarantee has exactly one responsible owner.
    graph = extract_ag_graph(REQ_SAFE_005_SYSML)
    owners = {c.name: c.owners for c in graph.components}
    assert all(len(value) == 1 for value in owners.values())
    assert CODE_GUARANTEE_NO_OWNER not in _codes(check_ag_graph(graph))


def test_additive_timing_budget_exceeded_is_detected():
    over = REQ_SAFE_005_SYSML.replace(
        "attribute latencyBudget : DurationValue = 0.35 [s];",
        "attribute latencyBudget : DurationValue = 0.45 [s];",
    )
    report = check_ag_graph(extract_ag_graph(over))
    assert report.verdict == "FAIL"
    assert CODE_TIMING_BUDGET_EXCEEDED in _codes(report)
    assert report.timing["sum"] == 0.55


def test_undischarged_assumption_is_localised_and_not_mislabelled_circular():
    # Remove the internal power guarantee required by RecoverySystem.
    broken = REQ_SAFE_005_SYSML.replace(
        "require constraint g_recoveryActuationPowerAvailable "
        "{ recoveryActuationPowerAvailable }", ""
    )
    report = check_ag_graph(extract_ag_graph(broken))
    assert report.verdict == "FAIL"
    assert CODE_ASSUMPTION_UNDISCHARGED in _codes(report)
    # A cascade behind an upstream gap must not be reported as a cycle (§16).
    assert CODE_CIRCULAR_ASSUMPTION not in _codes(report)
    assert report.discharge[
        "RecoverySystemContract.recoveryActuationPowerAvailable"
    ] == "undischarged"
    assert report.component_completeness["RecoveryPowerSupplyContract"] == INCOMPLETE


def test_missing_decomposition_owner_is_flagged():
    orphaned = REQ_SAFE_005_SYSML.replace(
        "    satisfy requirement recoverySystemContract : RecoverySystemContract by recoverySystem;\n",
        "",
    )
    report = check_ag_graph(extract_ag_graph(orphaned))
    assert report.verdict == "FAIL"
    assert CODE_GUARANTEE_NO_OWNER in _codes(report)


def test_genuine_circular_assumption_is_detected():
    # A needs B's guarantee and B needs A's guarantee; neither is environment.
    a = Contract(
        name="A", role="component",
        assumptions=(Assumption(concept="b_sig", expr="b_sig", kind="boolean"),),
        guarantees=(Guarantee(concept="a_sig", expr="a_sig", kind="boolean"),),
        element_id="A",
    )
    b = Contract(
        name="B", role="component",
        assumptions=(Assumption(concept="a_sig", expr="a_sig", kind="boolean"),),
        guarantees=(Guarantee(concept="b_sig", expr="b_sig", kind="boolean"),),
        element_id="B",
    )
    system = Contract(
        name="Sys", role="system",
        assumptions=(Assumption(concept="start", expr="start", kind="boolean",
                                is_environment=True, constraint_name="env_start"),),
        guarantees=(Guarantee(concept="a_sig", expr="a_sig", kind="boolean"),),
        observation="a_sig", element_id="Sys",
    )
    graph = AGGraph(
        system=system, components=(a, b),
        edges=(AGEdge("decomposes", "Sys", "A"), AGEdge("decomposes", "Sys", "B")),
        revision=1,
    )
    report = check_ag_graph(graph)
    assert report.verdict == "FAIL"
    assert CODE_CIRCULAR_ASSUMPTION in _codes(report)


def test_incompatible_timing_units_are_detected():
    system = Contract(
        name="Sys", role="system", timing_budget=0.5, timing_unit="s",
        assumptions=(Assumption("x", "x", "boolean", is_environment=True,
                                constraint_name="env_x"),),
        guarantees=(Guarantee("y", "y", "boolean"),),
        observation="y", element_id="Sys",
    )
    comp = Contract(
        name="C", role="component", timing_budget=100.0, timing_unit="ms",
        assumptions=(Assumption("x", "x", "boolean", is_environment=True,
                                constraint_name="env_x"),),
        guarantees=(Guarantee("y", "y", "boolean"),),
        element_id="C",
    )
    graph = AGGraph(system=system, components=(comp,),
                    edges=(AGEdge("decomposes", "Sys", "C"),), revision=2)
    report = check_ag_graph(graph)
    assert CODE_UNIT_INCOMPATIBLE in _codes(report)
    assert report.verdict == "FAIL"


def test_report_cites_source_revision_digest_and_is_regenerable():
    graph = extract_ag_graph(REQ_SAFE_005_SYSML, revision=42, model_digest="deadbeef")
    report = check_ag_graph(graph)
    d = report.to_dict()
    assert d["source_model_revision"] == 42
    assert d["source_model_digest"] == "deadbeef"
    assert d["checker_version"] == report.checker_version
    assert d["artifact_role"] == "RUNTIME_A_G_PREDICTION"
    # deterministic: same input → identical derived view
    assert check_ag_graph(extract_ag_graph(REQ_SAFE_005_SYSML, revision=42,
                                           model_digest="deadbeef")).to_dict() == d


def test_decomposition_insufficient_when_observation_unproduced():
    # System observes 'z' but no component produces it.
    system = Contract(
        name="Sys", role="system", observation="z",
        assumptions=(Assumption("start", "start", "boolean", is_environment=True,
                                constraint_name="env_start"),),
        guarantees=(Guarantee("z", "z", "boolean"),), element_id="Sys",
    )
    comp = Contract(
        name="C", role="component",
        assumptions=(Assumption("start", "start", "boolean", is_environment=True,
                                constraint_name="env_start"),),
        guarantees=(Guarantee("other", "other", "boolean"),), element_id="C",
    )
    graph = AGGraph(system=system, components=(comp,),
                    edges=(AGEdge("decomposes", "Sys", "C"),), revision=1)
    report = check_ag_graph(graph)
    assert CODE_DECOMPOSITION_INSUFFICIENT in _codes(report)


def test_concurrent_segments_compose_by_maximum_not_by_sum():
    """§18-Q5. Blanket addition is only sound for a serial chain.

    It was applied unconditionally, which was right for the one encoded timed
    chain by accident of its shape. Two 0.3 s responses running side by side
    occupy 0.3 s; calling that 0.6 s rejects a design that meets its deadline —
    conservative, but wrong, and the author had no way to say "these are
    concurrent" at all.
    """
    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package
    from dataclasses import replace

    # both contributing segments in one concurrent group; each fits the deadline
    # alone, their sum does not
    components = tuple(
        replace(item, latency_budget=0.35, timing_segment_group=1)
        if item.latency_budget is not None else item
        for item in REQ_SAFE_005_CHAIN.components
    )
    concurrent = replace(REQ_SAFE_005_CHAIN, components=components)
    report = check_ag_graph(extract_ag_graph(emit_ag_package(concurrent)))
    timing = report.timing
    assert timing["sum"] == 0.35, timing
    assert timing["ok"] is True
    assert timing["composition"]["structure"] == "serial_and_concurrent"
    assert "TIMING_BUDGET_EXCEEDED" not in {d.code for d in report.errors()}

    # the same budgets on the serial path (no group) must still be rejected
    serial = replace(REQ_SAFE_005_CHAIN, components=tuple(
        replace(item, latency_budget=0.35)
        if item.latency_budget is not None else item
        for item in REQ_SAFE_005_CHAIN.components
    ))
    serial_report = check_ag_graph(extract_ag_graph(emit_ag_package(serial)))
    assert serial_report.timing["sum"] == 0.7
    assert "TIMING_BUDGET_EXCEEDED" in {
        d.code for d in serial_report.errors()
    }


def test_a_declared_margin_is_deadline_that_is_not_apportioned():
    """§18-Q5, and the measured divergence behind it.

    Gold apportioned 0.1 + 0.35 = 0.45 of a 0.5 s deadline, holding 0.05 s back;
    the model used 0.2 + 0.3 = 0.5 and kept none. Both met the deadline, so the
    difference — how much reserve a safety response keeps — was invisible to the
    checker and looked like noise. Declared, it is checkable.
    """
    from dataclasses import replace

    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package

    # the reference apportions 0.45 of 0.5; a 0.05 s margin exactly fits
    fits = replace(REQ_SAFE_005_CHAIN, timing_margin=0.05)
    report = check_ag_graph(extract_ag_graph(emit_ag_package(fits)))
    assert report.timing["margin"] == 0.05
    assert report.timing["committed"] == 0.5
    assert report.timing["ok"] is True

    # ask for more reserve than the unspent deadline and it must fail, even
    # though the budgets alone are well inside the deadline
    overcommitted = replace(REQ_SAFE_005_CHAIN, timing_margin=0.1)
    failed = check_ag_graph(extract_ag_graph(emit_ag_package(overcommitted)))
    assert failed.timing["ok"] is False
    assert "TIMING_BUDGET_EXCEEDED" in {d.code for d in failed.errors()}
    message = next(
        d.message for d in failed.errors()
        if d.code == "TIMING_BUDGET_EXCEEDED"
    )
    assert "margin" in message, message


def test_the_composition_rules_are_published_to_the_author():
    """Every obligation in this file's checks must be stated somewhere an author
    reads, or it is unsatisfiable — the defect class this project met eight
    times."""
    from src.prototyping.ag_convention import (
        DECISION_FIELD_OBLIGATIONS, render_authoring_rules,
    )

    rules = render_authoring_rules()
    assert "timingSegmentGroup" in rules
    assert "timingMargin" in rules
    fields = {name for name, _rule in DECISION_FIELD_OBLIGATIONS}
    assert {"timing_segment_group", "timing_margin_seconds"} <= fields


def test_extraction_does_not_depend_on_how_the_model_is_spelled():
    """Why ag_extractor was left on patterns while other modules moved to Syside.

    Four regex-based checks elsewhere were found blind to a legal spelling — a
    unit suffix after a type, an `accept` clause before a guard, a port declared
    with the planned name but another type. The same failure here would corrupt
    every A/G verdict, so before converting 39 sites in the extraction core the
    question was asked as a measurement: does extraction actually depend on
    spelling?

    It does not, under equivalent rewrites that leave the model's meaning alone.
    Conversion was therefore declined as risk without evidence of need — and this
    pins the property so that stays true. A failure here is a reason to revisit
    that decision, not to relax the test.
    """
    import re

    from src.prototyping.ag_chains import (
        REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN, REQ_SAFE_008_CHAIN,
    )
    from src.prototyping.ag_emitter import emit_ag_package
    from src.prototyping.ag_extractor import extract_ag_graphs

    text = "\n".join(
        emit_ag_package(chain) for chain in
        (REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN, REQ_SAFE_008_CHAIN)
    )

    def fingerprint(source: str):
        return [
            (
                sorted(graph.source_requirement_ids),
                len(graph.components),
                len(graph.behaviors),
                sorted(
                    assumption.concept
                    for component in graph.components
                    for assumption in component.assumptions
                ),
            )
            for graph in extract_ag_graphs(source)
        ]

    baseline = fingerprint(text)
    assert baseline, "fixture must extract at least one chain"

    rewrites = {
        "extra whitespace around the colon": lambda s: s.replace(" : ", "  :  "),
        "constraint body on its own line": lambda s: re.sub(
            r"((?:assume|require) constraint \w+)\s*\{\s*([^}\n]+?)\s*\}",
            r"\1 {\n        \2\n    }", s,
        ),
        "transition wrapped after its name": lambda s: re.sub(
            r"(transition \w+) (first \w+)", r"\1\n            \2", s,
        ),
        "dependency wrapped before from": lambda s: re.sub(
            r"(dependency \w+) (from )", r"\1\n        \2", s,
        ),
    }
    for label, rewrite in rewrites.items():
        rewritten = rewrite(text)
        assert rewritten != text, f"rewrite {label!r} did not change the text"
        assert fingerprint(rewritten) == baseline, (
            f"extraction changed under {label!r}: the same model, spelled "
            "differently, produced a different A/G graph"
        )


def test_extraction_survives_every_meaning_preserving_rewrite():
    """The wider probe that the narrow one should have been.

    A first version of this check tried five rewrites, found no difference, and
    was used to justify leaving ag_extractor on patterns while seven other
    modules moved to the parser. That conclusion was wrong: it had simply not
    tried the rewrites that bite. Widening it found a real defect —
    `maxLatency : DurationValue [s] = 0.5 [s]` is legal, and the deadline
    vanished, so a timed chain was judged not to be a timed pattern at all.

    Rewrites here must PRESERVE MEANING. Removing `= true` from
    `timingSegmentRequired : Boolean = true` looks like a spelling change and is
    not: that value is the fact being read, and "not declared" is a third state.
    A probe that changes meaning reports a false blind spot, which is how the
    first pass nearly sent a correct distinction to be "fixed".
    """
    import re

    from src.prototyping.ag_chains import (
        REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN, REQ_SAFE_008_CHAIN,
    )
    from src.prototyping.ag_contracts import check_ag_graph
    from src.prototyping.ag_emitter import emit_ag_package
    from src.prototyping.ag_extractor import extract_ag_graphs

    text = "\n".join(
        emit_ag_package(chain) for chain in
        (REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN, REQ_SAFE_008_CHAIN)
    )

    def fingerprint(source: str):
        return [
            (
                sorted(graph.source_requirement_ids),
                check_ag_graph(graph).verdict,
                len(graph.components),
                len(graph.behaviors),
                getattr(graph.system, "timing_budget", None),
                tuple(sorted(
                    d.code for d in check_ag_graph(graph).diagnostics
                )),
            )
            for graph in extract_ag_graphs(source)
        ]

    baseline = fingerprint(text)
    assert baseline, "fixture must extract at least one chain"

    rewrites = {
        "qualified type name": lambda s: re.sub(
            r":\s*Boolean\s*;", ": ScalarValues::Boolean;", s),
        "unit suffix on the type": lambda s: re.sub(
            r":\s*DurationValue\s*=", ": DurationValue [s] =", s),
        "wide spacing around the colon": lambda s: s.replace(" : ", "   :   "),
        "constraint body on its own line": lambda s: re.sub(
            r"((?:assume|require) constraint \w+)\s*\{\s*([^}\n]+?)\s*\}",
            r"\1 {\n        \2\n    }", s),
        "transition wrapped after its name": lambda s: re.sub(
            r"(transition \w+) (first \w+)", r"\1\n            \2", s),
        "transition fully expanded": lambda s: re.sub(
            r"(first \w+) (accept \w+) (then|if)",
            r"\1\n                \2\n                \3", s),
        "dependency wrapped before from": lambda s: re.sub(
            r"(dependency \w+) (from )", r"\1\n        \2", s),
        "dependency from and to split": lambda s: re.sub(
            r"(from [\w:]+) (to )", r"\1\n        \2", s),
        "enum members on their own lines": lambda s: re.sub(
            r"(enum \w+;)", r"\n        \1", s),
        "satisfy wrapped": lambda s: re.sub(
            r"(satisfy requirement \w+) (: )", r"\1\n        \2", s),
        "CRLF line endings": lambda s: s.replace("\n", "\r\n"),
    }
    for label, rewrite in rewrites.items():
        rewritten = rewrite(text)
        assert rewritten != text, f"rewrite {label!r} did not change the text"
        assert fingerprint(rewritten) == baseline, (
            f"extraction changed under {label!r}: the same model, spelled "
            "differently, produced a different A/G graph"
        )
