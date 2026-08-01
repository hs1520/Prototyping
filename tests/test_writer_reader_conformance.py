"""Discover every SysML emitter and enforce its writer/reader contract.

Emitter discovery is deliberately source-driven.  The case functions below are
not a registry: their names are derived from the discovered qualified name, and
the coverage test fails when a new public emitter has no conformance obligation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pytest

from src.dse.physics_estimator import DesignInputs
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
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


@dataclass(frozen=True)
class EmitterCase:
    emit: Callable[[], str]
    parse_source: Callable[[str], str]
    readers: Mapping[str, Callable[[str], object]]
    expected: Mapping[str, object]


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
    return (
        graph.system.name if graph.system else None,
        tuple(component.name for component in graph.components),
        tuple(behavior.name for behavior in graph.behaviors),
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
            tuple(machine.states),
            tuple(
                (transition.name, transition.source, transition.target)
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
