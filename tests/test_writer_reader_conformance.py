"""Discover every SysML emitter and enforce its writer/reader contract.

Emitter discovery is deliberately source-driven.  The case functions below are
not a registry: their names are derived from the discovered qualified name, and
the coverage test fails when a new public emitter has no conformance obligation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pytest

from src.dse.physics_estimator import DesignInputs
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_planning import emit_ag_planning_package
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    PlannedState,
    PlannedTransition,
    emit_planned_behavior,
)
from src.simulation.connectivity_fixer import ConnectStmt, parse_connects
from src.simulation.port_fixer import PortAdd
from src.simulation.state_extractor import extract_state_machines
from src.simulation.syntax_checker import check_syntax
from src.simulation.transition_fixer import (
    TransitionStmt,
    build_state_machine_summary,
)
from src.sysml.lite_model import build_lite_model
from src.sysml.model import PartDefinition, SysMLModel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GOLDEN = ROOT / "tests" / "golden" / "writer_emitter_outputs.json"


@dataclass(frozen=True)
class EmitterCase:
    emit: Callable[[], str]
    parse_source: Callable[[str], str]
    readers: Mapping[str, Callable[[str], object]]
    expected: Mapping[str, object]


@dataclass(frozen=True)
class SpellingPerturbation:
    name: str
    apply: Callable[[str], str | None]


def _returns_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    annotation = ast.unparse(node.returns) if node.returns is not None else ""
    return "str" in annotation


def _discovered_emitters() -> tuple[str, ...]:
    """Find public SysML serializers; no author-maintained emitter list exists."""
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("emit_") and _returns_text(node):
                    found.append(f"{module}:{node.name}")
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if member.name in {"to_sysml", "to_sysml_text"} and _returns_text(member):
                        found.append(f"{module}:{node.name}.{member.name}")
    return tuple(found)


def _case_name(emitter: str) -> str:
    return "case__" + emitter.replace(".", "__").replace(":", "__")


def _text(value: object) -> str:
    if isinstance(value, tuple):
        emitted, ok = value
        assert ok is True
        return str(emitted)
    return str(value)


def _identity(text: str) -> str:
    return text


def _ag_facts(text: str) -> object:
    graph = extract_ag_graph(text)
    report = check_ag_graph(graph)

    def contract_facts(contract):
        if contract is None:
            return None
        return (
            contract.name,
            contract.role,
            tuple((item.concept, item.expr, item.kind) for item in contract.assumptions),
            tuple((item.concept, item.expr, item.kind) for item in contract.guarantees),
            contract.timing_budget,
            contract.timing_unit,
            contract.timing_segment_required,
            contract.timing_segment_group,
            contract.timing_margin,
            contract.observation,
            contract.owners,
            contract.source_requirement,
            contract.declared_pattern,
        )

    return (
        contract_facts(graph.system),
        tuple(contract_facts(component) for component in graph.components),
        tuple((edge.kind, edge.src, edge.dst, edge.subject) for edge in graph.edges),
        tuple(
            (
                behavior.name,
                behavior.initial_state,
                tuple(
                    (
                        transition.source,
                        transition.trigger,
                        transition.target,
                        None if transition.guard in {None, "true", "(true)"} else transition.guard,
                    )
                    for transition in behavior.transitions
                ),
                tuple(sorted(behavior.entry_actions.items())),
            )
            for behavior in graph.behaviors
        ),
        tuple(sorted(graph.verification_targets.items())),
        graph.source_requirement_ids,
        graph.priority,
        graph.invariants,
        graph.selected_model_elements,
        graph.declared_event_signals,
        report.verdict,
        tuple(sorted(item.code for item in report.diagnostics)),
    )


def _planning_facts(text: str) -> object:
    graph = extract_ag_graph(text)
    return (
        graph.system.name if graph.system else None,
        tuple(component.name for component in graph.components),
        tuple(behavior.name for behavior in graph.behaviors),
    )


def _behavior_facts(text: str) -> object:
    machines = extract_state_machines(text)
    return tuple(
        (
            machine.name,
            tuple(
                (state.name, state.entry_action, state.do_action)
                for state in machine.states
            ),
            tuple(
                (
                    transition.name,
                    transition.source,
                    transition.target,
                    transition.accept_trigger,
                    tuple(guard.raw for guard in transition.guards),
                )
                for transition in machine.transitions
            ),
        )
        for machine in machines
    )


def _lite_facts(text: str) -> object:
    model = build_lite_model(text)
    return (
        tuple(part.name for part in model.part_definitions),
        tuple(req.name for req in model.requirement_definitions),
    )


def _connect_facts(text: str) -> object:
    return tuple(item.key() for item in parse_connects(text))


def _transition_facts(text: str) -> object:
    summary = build_state_machine_summary(text, "ControllerBehavior")
    assert summary is not None
    return tuple(
        (item.name, item.source, item.accept_cmd, item.guard_raw, item.target)
        for item in summary.transitions
    )


def _port_facts(text: str) -> object:
    model = build_lite_model(text)
    part = next(item for item in model.part_definitions if item.name == "Device")
    return tuple(
        (
            port.name,
            port.direction,
            port.type_ref.name if port.type_ref is not None else None,
        )
        for port in part.ports
    )


def case__src__prototyping__ag_emitter__emit_ag_package() -> EmitterCase:
    expected = _ag_facts(emit_ag_package(REQ_SAFE_005_CHAIN))
    return EmitterCase(
        lambda: emit_ag_package(REQ_SAFE_005_CHAIN), _identity,
        {"ag_extractor": _ag_facts}, {"ag_extractor": expected},
    )


def case__src__prototyping__ag_planning__emit_ag_planning_package() -> EmitterCase:
    expected = _planning_facts(emit_ag_planning_package(REQ_SAFE_005_CHAIN))
    return EmitterCase(
        lambda: emit_ag_planning_package(REQ_SAFE_005_CHAIN), _identity,
        {"ag_extractor": _planning_facts}, {"ag_extractor": expected},
    )


def _planned_behavior() -> PlannedBehavior:
    return PlannedBehavior(
        owner="FlightController",
        behavior_id="ControllerBehavior",
        initial_state="Idle",
        states=(PlannedState("Idle", "INITIAL"), PlannedState("Responding", "RESPONSE")),
        transitions=(PlannedTransition("respond", "Idle", "Responding", "ACCEPT", "FaultSignal"),),
    )


def _wrap_behavior(fragment: str) -> str:
    return (
        "package P {\n    item def FaultSignal;\n"
        + "    action def respondToFault {}\n"
        + "\n".join(f"    {line}" for line in fragment.splitlines())
        + "\n}"
    )


def case__src__prototyping__planned_behavior__emit_planned_behavior() -> EmitterCase:
    rendered = _wrap_behavior(emit_planned_behavior(_planned_behavior()))
    return EmitterCase(
        lambda: emit_planned_behavior(_planned_behavior()), _wrap_behavior,
        {"state_extractor": _behavior_facts},
        {"state_extractor": _behavior_facts(rendered)},
    )


def _realization_text() -> str:
    from src.realization.closure import close_the_loop
    from src.realization.realization_emitter import emit_realization_package
    from tests.realization_fixtures import catalog

    design = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    report = close_the_loop(
        design, [], ["REQ-PERF-002: endurance at least 15 minutes."], catalog()
    )
    return _text(emit_realization_package(report))


def case__src__realization__realization_emitter__emit_realization_package() -> EmitterCase:
    rendered = _realization_text()
    return EmitterCase(
        _realization_text, _identity, {"lite_model": _lite_facts},
        {"lite_model": _lite_facts(rendered)},
    )


def _analysis_text() -> str:
    from src.dse.analysis_emitter import emit_endurance_analysis

    return _text(emit_endurance_analysis(DesignInputs(0.5, 5000, 4, 4, 0.13), 10.0))


def case__src__dse__analysis_emitter__emit_endurance_analysis() -> EmitterCase:
    rendered = _analysis_text()
    return EmitterCase(
        _analysis_text, _identity, {"lite_model": _lite_facts},
        {"lite_model": _lite_facts(rendered)},
    )


def _structured_model() -> SysMLModel:
    model = SysMLModel(name="ConformanceModel")
    model.part_definitions.append(PartDefinition(name="Controller"))
    return model


def case__src__sysml__model__SysMLModel__to_sysml_text() -> EmitterCase:
    rendered = _structured_model().to_sysml_text()
    return EmitterCase(
        lambda: _structured_model().to_sysml_text(), _identity,
        {"lite_model": _lite_facts}, {"lite_model": _lite_facts(rendered)},
    )


def _lite_model_text() -> str:
    return "package LiteConformance { part def Sensor; }"


def case__src__sysml__lite_model__SysMLLiteModel__to_sysml_text() -> EmitterCase:
    rendered = _lite_model_text()
    return EmitterCase(
        lambda: build_lite_model(rendered).to_sysml_text(), _identity,
        {"lite_model": _lite_facts}, {"lite_model": _lite_facts(rendered)},
    )


def _wrap_connect(fragment: str) -> str:
    return (
        "package P { port def DataPort; part def Source { out port data : DataPort; } "
        "part def Sink { in port data : DataPort; } part def System { "
        "part source : Source; part sink : Sink; " + fragment + " } }"
    )


def case__src__simulation__connectivity_fixer__ConnectStmt__to_sysml() -> EmitterCase:
    stmt = ConnectStmt("source", "data", "sink", "data")
    rendered = _wrap_connect(stmt.to_sysml())
    return EmitterCase(
        stmt.to_sysml, _wrap_connect, {"connectivity_reader": _connect_facts},
        {"connectivity_reader": _connect_facts(rendered)},
    )


def _wrap_transition(fragment: str) -> str:
    return (
        "package P { item def FaultSignal; state def ControllerBehavior { "
        "entry; then Idle; state Idle; state Responding; " + fragment + " } }"
    )


def case__src__simulation__transition_fixer__TransitionStmt__to_sysml() -> EmitterCase:
    stmt = TransitionStmt("respond", "Idle", "", "Responding", "FaultSignal")
    rendered = _wrap_transition(stmt.to_sysml())
    return EmitterCase(
        stmt.to_sysml, _wrap_transition, {"transition_reader": _transition_facts},
        {"transition_reader": _transition_facts(rendered)},
    )


def _wrap_port(fragment: str) -> str:
    return f"package P {{ port def DataPort; part def Device {{ {fragment} }} }}"


def case__src__simulation__port_fixer__PortAdd__to_sysml() -> EmitterCase:
    addition = PortAdd("Device", "in", "data", "DataPort")
    rendered = _wrap_port(addition.to_sysml())
    return EmitterCase(
        addition.to_sysml, _wrap_port, {"lite_model": _port_facts},
        {"lite_model": _port_facts(rendered)},
    )


DISCOVERED = _discovered_emitters()


def _unit_suffix_on_type(text: str) -> str | None:
    """Add the same unit already carried by the value to its declared type.

    This is meaning-preserving because both spellings declare seconds, and the
    numeric initializer and its unit are unchanged.  The precondition rejects
    mixed or absent units instead of guessing dimensional meaning.
    """
    pattern = re.compile(
        r"(attribute\s+\w+\s*:\s*DurationValue)(\s*=\s*-?[\d.]+\s*\[s\])"
    )
    match = pattern.search(text)
    if match is None:
        return None
    assert "[s]" in match.group(2) and "[" not in match.group(1)
    return text[:match.start()] + match.group(1) + " [s]" + match.group(2) + text[match.end():]


def _qualified_type(text: str) -> str | None:
    """Qualify a type through an import already present in the emitted package.

    `ScalarValues::Boolean` and imported `Boolean` resolve to the same library
    classifier.  Requiring the import before rewriting is the semantic proof.
    """
    if "private import ScalarValues::*;" not in text:
        return None
    match = re.search(r":\s*Boolean\s*;", text)
    if match is None:
        return None
    assert "ScalarValues::*" in text
    return text[:match.start()] + ": ScalarValues::Boolean;" + text[match.end():]


def _optional_true_guard(text: str) -> str | None:
    """Add a tautological optional guard to an event-triggered transition.

    `accept E then` and `accept E if true then` enable on exactly the same event;
    unlike removing an arbitrary accept clause, this cannot discard a trigger.
    """
    # Probe model-level emitters whose consumer is the Syside state reader.  The
    # repair-line parser is a legacy lexical gate and is covered at baseline,
    # but §3.3 forbids changing it here; its disagreement is reported separately.
    if "// OWNER:" not in text and "private import ScalarValues::*;" not in text:
        return None
    match = re.search(r"(accept\s+\w+)(\s+then\s+\w+;)", text)
    if match is None:
        return None
    assert "if" not in match.group(0)
    return text[:match.start()] + match.group(1) + " if true" + match.group(2) + text[match.end():]


def _redundant_initializer(text: str) -> str | None:
    """Initialize a Boolean already constrained to true by the same contract.

    The initializer is redundant only when the exact feature has a positive
    assume constraint in the same emitted text.  This precondition deliberately
    excludes `timingSegmentRequired`, where removing `= true` changes a fact.
    """
    for match in re.finditer(r"attribute\s+(\w+)\s*:\s*Boolean\s*;", text):
        name = match.group(1)
        constraint = re.compile(
            rf"assume\s+constraint\s+\w+\s*\{{\s*{re.escape(name)}\s*\}}"
        )
        if constraint.search(text) is None:
            continue
        assert name != "timingSegmentRequired"
        replacement = match.group(0)[:-1] + " = true;"
        return text[:match.start()] + replacement + text[match.end():]
    return None


PERTURBATIONS = (
    SpellingPerturbation("unit suffix added/removed", _unit_suffix_on_type),
    SpellingPerturbation("qualified/unqualified type", _qualified_type),
    SpellingPerturbation("optional clause present/absent", _optional_true_guard),
    SpellingPerturbation("initializer present/absent", _redundant_initializer),
)


def test_every_discovered_emitter_has_a_conformance_obligation():
    missing = [item for item in DISCOVERED if _case_name(item) not in globals()]
    assert not missing, f"discovered SysML emitters without obligations: {missing}"


@pytest.mark.parametrize("emitter", DISCOVERED)
def test_discovered_emitter_parses_and_round_trips_to_every_reader(emitter):
    factory = globals().get(_case_name(emitter))
    assert callable(factory), f"{emitter} has no conformance case"
    case = factory()
    emitted = case.emit()
    source = case.parse_source(emitted)
    syntax = check_syntax(source, fail_closed=True)
    assert not syntax.has_errors, (emitter, syntax.to_dict())
    for reader_name, reader in case.readers.items():
        assert reader(source) == case.expected[reader_name], reader_name


def _perturbation_cases():
    result = []
    for emitter in DISCOVERED:
        factory = globals().get(_case_name(emitter))
        if not callable(factory):
            continue
        emitted = factory().emit()
        for perturbation in PERTURBATIONS:
            if perturbation.apply(emitted) is not None:
                result.append((emitter, perturbation))
    return result


@pytest.mark.parametrize(
    ("emitter", "perturbation"),
    _perturbation_cases(),
    ids=lambda value: value.name if isinstance(value, SpellingPerturbation) else value,
)
def test_reader_facts_are_invariant_under_meaning_preserving_spelling(
    emitter, perturbation
):
    case = globals()[_case_name(emitter)]()
    baseline_text = case.parse_source(case.emit())
    perturbed_emission = perturbation.apply(case.emit())
    assert perturbed_emission is not None
    assert perturbed_emission != case.emit()
    perturbed_text = case.parse_source(perturbed_emission)
    syntax = check_syntax(perturbed_text, fail_closed=True)
    assert not syntax.has_errors, (emitter, perturbation.name, syntax.to_dict())
    for reader_name, reader in case.readers.items():
        assert reader(perturbed_text) == reader(baseline_text), (
            emitter,
            perturbation.name,
            reader_name,
        )


def test_every_required_spelling_class_is_exercised_on_emitter_output():
    exercised = {perturbation.name for _, perturbation in _perturbation_cases()}
    assert exercised == {item.name for item in PERTURBATIONS}


def test_legacy_transition_repair_reader_disagrees_with_syside_on_optional_guard():
    """Record, but do not repair, the reader disagreement found by this probe.

    P0-3 §3.3 says a reader change is a finding rather than a step in this task.
    The model-level Syside reader retains the transition; the repair-line reader
    silently drops the same legal construct when `accept` and `if` coexist.
    """
    stmt = TransitionStmt("respond", "Idle", "", "Responding", "FaultSignal")
    perturbed = stmt.to_sysml().replace(
        "accept FaultSignal\n", "accept FaultSignal if true\n"
    )
    source = _wrap_transition(perturbed)
    assert not check_syntax(source, fail_closed=True).has_errors
    assert len(extract_state_machines(source)[0].transitions) == 1
    summary = build_state_machine_summary(source, "ControllerBehavior")
    assert summary is not None
    assert summary.transitions == []


def test_discovered_emitter_outputs_match_byte_golden():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = {}
    for emitter in DISCOVERED:
        factory = globals().get(_case_name(emitter))
        assert callable(factory), f"{emitter} has no fixed golden input"
        payload = factory().emit().encode("utf-8")
        actual[emitter] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    assert actual == expected
