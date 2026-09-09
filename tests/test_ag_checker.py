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


REQ_SAFE_005_SYSML = (
    "package Source { requirement def REQ_SAFE_005 { doc /* source */ } }\n"
    + emit_ag_package(REQ_SAFE_005_CHAIN)
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def test_convention_passes_syside_gate():
    result = check_syntax(REQ_SAFE_005_SYSML)
    assert result.has_errors is False
    assert result.score == 1.0


def test_req_safe_005_extracts_and_passes():
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
    assert report.timing["sum"] == 0.45
    assert report.timing["ok"] is True
    assert report.discharge[
        "RecoverySystemContract.parachuteDeploymentCommand"
    ] == "discharged"
    assert report.discharge[
        "RecoverySystemContract.recoveryActuationPowerAvailable"
    ] == "discharged"


def test_compound_guarantee_incomplete():
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


def test_one_owner_per_guarantee():
    graph = extract_ag_graph(REQ_SAFE_005_SYSML)
    owners = {c.name: c.owners for c in graph.components}
    assert all(len(value) == 1 for value in owners.values())
    assert CODE_GUARANTEE_NO_OWNER not in _codes(check_ag_graph(graph))


def test_timing_budget_exceeded():
    over = REQ_SAFE_005_SYSML.replace(
        "attribute latencyBudget : DurationValue = 0.35 [s];",
        "attribute latencyBudget : DurationValue = 0.45 [s];",
    )
    report = check_ag_graph(extract_ag_graph(over))
    assert report.verdict == "FAIL"
    assert CODE_TIMING_BUDGET_EXCEEDED in _codes(report)
    assert report.timing["sum"] == 0.55


def test_undischarged_not_circular():
    broken = REQ_SAFE_005_SYSML.replace(
        "require constraint g_recoveryActuationPowerAvailable "
        "{ recoveryActuationPowerAvailable }", ""
    )
    report = check_ag_graph(extract_ag_graph(broken))
    assert report.verdict == "FAIL"
    assert CODE_ASSUMPTION_UNDISCHARGED in _codes(report)
    # A cascade behind an upstream gap is not reported as a cycle (§16).
    assert CODE_CIRCULAR_ASSUMPTION not in _codes(report)
    assert report.discharge[
        "RecoverySystemContract.recoveryActuationPowerAvailable"
    ] == "undischarged"
    assert report.component_completeness["RecoveryPowerSupplyContract"] == INCOMPLETE


def test_missing_owner_flagged():
    orphaned = REQ_SAFE_005_SYSML.replace(
        "    satisfy requirement recoverySystemContract : RecoverySystemContract by recoverySystem;\n",
        "",
    )
    report = check_ag_graph(extract_ag_graph(orphaned))
    assert report.verdict == "FAIL"
    assert CODE_GUARANTEE_NO_OWNER in _codes(report)


def test_circular_assumption_detected():
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


def test_incompatible_timing_units():
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


def test_report_cites_revision():
    graph = extract_ag_graph(REQ_SAFE_005_SYSML, revision=42)
    report = check_ag_graph(graph)
    d = report.to_dict()
    assert d["source_model_revision"] == 42
    assert d["checker_version"] == report.checker_version
    assert d["artifact_role"] == "RUNTIME_A_G_PREDICTION"
    assert check_ag_graph(extract_ag_graph(REQ_SAFE_005_SYSML, revision=42)).to_dict() == d


def test_unproduced_observation_insufficient():
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


def test_concurrent_compose_by_max():
    """§18-Q5. Blanket addition is only sound for a serial chain.

    It was applied unconditionally, which happened to suit the one encoded timed
    chain. Two 0.3 s responses side by side occupy 0.3 s; calling that 0.6 s rejects
    a design that meets its deadline, and the author could not declare concurrency
    at all.
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


def test_declared_margin_not_apportioned():
    """§18-Q5: a declared margin is deadline that is not apportioned.

    Gold apportioned 0.1 + 0.35 = 0.45 of a 0.5 s deadline, holding 0.05 s back; the
    model used 0.2 + 0.3 = 0.5 and kept none. Both met the deadline, so the reserve
    difference was invisible to the checker until declared.
    """
    from dataclasses import replace

    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    from src.prototyping.ag_emitter import emit_ag_package

    fits = replace(REQ_SAFE_005_CHAIN, timing_margin=0.05)
    report = check_ag_graph(extract_ag_graph(emit_ag_package(fits)))
    assert report.timing["margin"] == 0.05
    assert report.timing["committed"] == 0.5
    assert report.timing["ok"] is True

    # reserve larger than the unspent deadline fails, even though the budgets
    # alone sit well inside the deadline
    overcommitted = replace(REQ_SAFE_005_CHAIN, timing_margin=0.1)
    failed = check_ag_graph(extract_ag_graph(emit_ag_package(overcommitted)))
    assert failed.timing["ok"] is False
    assert "TIMING_BUDGET_EXCEEDED" in {d.code for d in failed.errors()}
    message = next(
        d.message for d in failed.errors()
        if d.code == "TIMING_BUDGET_EXCEEDED"
    )
    assert "margin" in message, message


def test_composition_rules_published():
    """Every obligation checked here is stated somewhere an author reads, or it is
    unsatisfiable: the defect class this project met eight times.
    """
    from src.prototyping.ag_convention import (
        DECISION_FIELD_OBLIGATIONS, render_authoring_rules,
    )

    rules = render_authoring_rules()
    assert "timingSegmentGroup" in rules
    assert "timingMargin" in rules
    fields = {name for name, _rule in DECISION_FIELD_OBLIGATIONS}
    assert {"timing_segment_group", "timing_margin_seconds"} <= fields


def test_extraction_spelling_agnostic():
    """Why ag_extractor was left on patterns while other modules moved to Syside.

    Four regex-based checks elsewhere were blind to a legal spelling (a unit suffix
    after a type, an `accept` clause before a guard, a port declared with the
    planned name but another type), so before converting 39 sites in the extraction
    core the question was measured: does extraction depend on spelling? It does not,
    under rewrites that leave the model's meaning alone, so conversion was declined
    and this pins the property. A failure here is a reason to revisit that decision,
    not to relax the test.
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


def test_extraction_survives_rewrites():
    """The wider probe the narrow one should have been.

    A first version tried five rewrites, found no difference, and was used to
    justify leaving ag_extractor on patterns; it had not tried the rewrites that
    bite. Widening it found a real defect: `maxLatency : DurationValue [s] = 0.5 [s]`
    is legal and the deadline vanished, so a timed chain was judged not to be a
    timed pattern. Rewrites here preserve meaning: dropping `= true` from
    `timingSegmentRequired : Boolean = true` changes the fact being read, since
    "not declared" is a third state, and a meaning-changing probe reports a false
    blind spot.
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


def test_unnamed_transitions_extracted():
    """``transition first idle accept X then done;`` is legal SysML v2: the transition
    name is optional.

    The extractor required one and parsed zero transitions from this spelling, so a
    correct realisation read as no reachable trigger, no reachable response and no
    trigger-response path (pilot_n6_4bb7544 seed 1, R2-BBAG). The same model with
    and without transition names extracts the same transitions and reaches the same
    verdict.
    """
    import re as _re
    named = REQ_SAFE_005_SYSML
    unnamed = _re.sub(r"\btransition\s+\w+\s+first\b", "transition first", named)
    assert unnamed != named, "fixture must contain named transitions"
    assert "transition first" in unnamed

    g_named = extract_ag_graph(named, revision=7)
    g_unnamed = extract_ag_graph(unnamed, revision=7)
    named_transitions = {
        (b.name, t.source, t.trigger, t.target)
        for b in g_named.behaviors for t in b.transitions
    }
    unnamed_transitions = {
        (b.name, t.source, t.trigger, t.target)
        for b in g_unnamed.behaviors for t in b.transitions
    }
    assert named_transitions, "fixture must have transitions"
    assert unnamed_transitions == named_transitions

    report = check_ag_graph(g_unnamed)
    assert report.verdict == "PASS", _codes(report)
