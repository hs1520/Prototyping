"""Discover every SysML emitter and enforce its writer/reader contract.

Emitter discovery is source-driven: the case functions below are named from the
discovered qualified name, and the coverage test fails when a new public emitter
has no conformance obligation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pytest

from src.dse.design_space import DesignConfiguration
from src.dse.diagnostics import diagnose
from src.dse.evaluator import DesignEvaluator
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
from src.sysml.model import FeatureDirection, PartDefinition, SysMLModel


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


def _wrap_ag(fragment: str) -> str:
    return (
        "package ConformanceInput { requirement def REQ_SAFE_005; "
        "part def EvidenceOwner { satisfy requirement REQ_SAFE_005; } }\n"
        + fragment
    )


def _ag_facts(text: str) -> object:
    graph = extract_ag_graph(text)

    def contract_facts(contract):
        if contract is None:
            return None
        return (
            contract.name,
            tuple(item.concept for item in contract.assumptions),
            tuple(item.concept for item in contract.guarantees),
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
        tuple(
            (edge.src, edge.dst)
            for edge in graph.edges
            if edge.kind == "decomposes"
        ),
    )


def _expected_ag_facts(spec) -> object:
    def component_facts(component):
        return (
            component.name,
            tuple(item.concept for item in component.assumptions),
            component.guarantees,
            component.latency_budget,
            "s" if component.latency_budget is not None else None,
            component.timing_segment_required,
            component.timing_segment_group,
            None,
            None,
            (component.owner_usage,),
            None,
            None,
        )

    system = (
        spec.system_contract,
        spec.system_assumptions,
        (spec.observation,),
        spec.deadline,
        "s" if spec.deadline is not None else None,
        None,
        None,
        spec.timing_margin,
        spec.observation,
        (),
        spec.source_requirement,
        spec.pattern,
    )
    return (
        system,
        tuple(component_facts(item) for item in spec.components),
        tuple((spec.system_contract, item.name) for item in spec.components),
    )


def _planning_facts(text: str) -> object:
    graph = extract_ag_graph(text)
    return (
        graph.system.name if graph.system else None,
        tuple(component.name for component in graph.components),
        tuple(behavior.name for behavior in graph.behaviors),
    )


def _expected_planning_facts(spec) -> object:
    return (
        spec.system_contract,
        tuple(item.name for item in spec.components),
        (),
    )


def _ag_attribute_facts(text: str) -> object:
    from src.prototyping import ag_extractor

    names = Counter(
        match.group(1) for match in ag_extractor._ATTR_RE.finditer(text)
    )
    names.update(
        match.group(1) for match in ag_extractor._BOOL_ATTR_RE.finditer(text)
    )
    return tuple(sorted(names.items()))


def _expected_ag_attribute_facts(spec, *, include_implementation: bool = True) -> object:
    names = Counter(dict.fromkeys((
        *spec.system_assumptions,
        *(spec.system_observation_concepts or (spec.observation,)),
        *spec.selected_model_elements,
    )).keys())
    if spec.deadline is not None:
        names["maxLatency"] += 1
    if spec.timing_margin is not None:
        names["timingMargin"] += 1
    for component in spec.components:
        names.update(dict.fromkeys((
            *(item.concept for item in component.assumptions),
            *component.interface_inputs,
            *component.guarantees,
        )).keys())
        if component.latency_budget is not None:
            names["latencyBudget"] += 1
        if component.timing_segment_group is not None:
            names["timingSegmentGroup"] += 1
        if component.timing_segment_required is not None:
            names["timingSegmentRequired"] += 1
    if spec.priority is not None:
        # The trigger is declared by both the auxiliary priority contract and
        # its arbitration behavior; selectedResponse belongs to the contract.
        names[spec.priority.trigger] += 1 + int(include_implementation)
        names["selectedResponse"] += 1
    return tuple(sorted(names.items()))


def _evaluator_ag_facts(text: str) -> object:
    model = build_lite_model(text)
    evaluator = DesignEvaluator()
    evaluator._syside_model = model._syside_model
    return evaluator._score_requirement_coverage(
        DesignConfiguration("writer-reader-conformance"), model, None
    )


def _diagnostics_ag_facts(text: str) -> object:
    issues, _ = diagnose(build_lite_model(text), None)
    return tuple(
        issue for issue in issues if issue.startswith("Untraced requirements:")
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


def _expected_behavior_facts(behavior: PlannedBehavior) -> object:
    return ((
        behavior.behavior_id,
        tuple(
            (state.state_id, state.entry_action, state.do_action)
            for state in behavior.states
        ),
        tuple(
            (
                transition.transition_id,
                transition.source,
                transition.target,
                transition.trigger if transition.trigger_kind == "ACCEPT" else None,
                (),
            )
            for transition in behavior.transitions
        ),
    ),)


def _lite_facts(text: str) -> object:
    model = build_lite_model(text)
    return (
        tuple(part.name for part in model.part_definitions),
        tuple(req.name for req in model.requirement_definitions),
    )


def _realization_facts(text: str) -> object:
    parts, requirements = _lite_facts(text)
    return (tuple(name for name in parts if name != "RealizedDesign"), requirements)


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
    return EmitterCase(
        lambda: emit_ag_package(REQ_SAFE_005_CHAIN), _wrap_ag,
        {
            "ag_extractor": _ag_facts,
            "ag_attribute_reader": _ag_attribute_facts,
            "dse_evaluator": _evaluator_ag_facts,
            "dse_diagnostics": _diagnostics_ag_facts,
        },
        {
            "ag_extractor": _expected_ag_facts(REQ_SAFE_005_CHAIN),
            "ag_attribute_reader": _expected_ag_attribute_facts(REQ_SAFE_005_CHAIN),
            "dse_evaluator": 1.0,
            "dse_diagnostics": (),
        },
    )


def case__src__prototyping__ag_planning__emit_ag_planning_package() -> EmitterCase:
    return EmitterCase(
        lambda: emit_ag_planning_package(REQ_SAFE_005_CHAIN), _wrap_ag,
        {
            "ag_extractor": _planning_facts,
            "ag_attribute_reader": _ag_attribute_facts,
            "dse_evaluator": _evaluator_ag_facts,
            "dse_diagnostics": _diagnostics_ag_facts,
        },
        {
            "ag_extractor": _expected_planning_facts(REQ_SAFE_005_CHAIN),
            "ag_attribute_reader": _expected_ag_attribute_facts(
                REQ_SAFE_005_CHAIN, include_implementation=False
            ),
            # Planning contains no state realization, so the SAFE implementation sub-score
            # is absent while satisfy coverage remains.
            "dse_evaluator": 0.6,
            "dse_diagnostics": (),
        },
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
    behavior = _planned_behavior()
    return EmitterCase(
        lambda: emit_planned_behavior(behavior), _wrap_behavior,
        {"state_extractor": _behavior_facts},
        {"state_extractor": _expected_behavior_facts(behavior)},
    )


def _realization_report():
    from src.realization.closure import close_the_loop
    from tests.realization_fixtures import catalog

    design = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    return close_the_loop(
        design, [], ["REQ-PERF-002: endurance at least 15 minutes."], catalog()
    )


def _realization_text(report=None) -> str:
    from src.realization.realization_emitter import emit_realization_package

    report = report or _realization_report()
    return _text(emit_realization_package(report))


def _identifier_from_input(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def case__src__realization__realization_emitter__emit_realization_package() -> EmitterCase:
    report = _realization_report()
    realized = report.chosen.rd
    expected_parts = tuple(_identifier_from_input(name) for name in (
        realized.combo.name,
        realized.pack.name,
        realized.frame.name,
        realized.integration_bundle.name,
    ))
    expected_requirements = tuple(
        item.req_id.replace("-", "_")
        for item in report.per_requirement
        if item.scope == "closure"
    )
    return EmitterCase(
        lambda: _realization_text(report), _identity,
        {"lite_model": _realization_facts},
        {"lite_model": (expected_parts, expected_requirements)},
    )


ANALYSIS_DESIGN = DesignInputs(0.5, 5000, 4, 4, 0.13)
ANALYSIS_TARGET = 10.0
ANALYSIS_REQUIREMENT = "REQ-PERF-002"
ANALYSIS_PART = "AnalyzedDesign"


def _analysis_text() -> str:
    from src.dse.analysis_emitter import emit_endurance_analysis

    return _text(emit_endurance_analysis(
        ANALYSIS_DESIGN,
        ANALYSIS_TARGET,
        satisfy_req=ANALYSIS_REQUIREMENT,
        part_name=ANALYSIS_PART,
    ))


def case__src__dse__analysis_emitter__emit_endurance_analysis() -> EmitterCase:
    return EmitterCase(
        _analysis_text, _identity, {"lite_model": _lite_facts},
        {"lite_model": ((ANALYSIS_PART,), (ANALYSIS_REQUIREMENT.replace("-", "_"),))},
    )


def _structured_model() -> SysMLModel:
    model = SysMLModel(name="ConformanceModel")
    model.part_definitions.append(PartDefinition(name="Controller"))
    return model


def case__src__sysml__model__SysMLModel__to_sysml_text() -> EmitterCase:
    model = _structured_model()
    return EmitterCase(
        model.to_sysml_text, _identity,
        {"lite_model": _lite_facts},
        {"lite_model": (
            tuple(item.name for item in model.part_definitions),
            tuple(item.name for item in model.requirement_definitions),
        )},
    )


def _lite_model_text() -> str:
    return "package LiteConformance { part def Sensor; }"


def case__src__sysml__lite_model__SysMLLiteModel__to_sysml_text() -> EmitterCase:
    rendered = _lite_model_text()
    return EmitterCase(
        lambda: build_lite_model(rendered).to_sysml_text(), _identity,
        {"lite_model": _lite_facts}, {"lite_model": (("Sensor",), ())},
    )


def _wrap_connect(fragment: str) -> str:
    return (
        "package P { port def DataPort; part def Source { out port data : DataPort; } "
        "part def Sink { in port data : DataPort; } part def System { "
        "part source : Source; part sink : Sink; " + fragment + " } }"
    )


def case__src__simulation__connectivity_fixer__ConnectStmt__to_sysml() -> EmitterCase:
    stmt = ConnectStmt("source", "data", "sink", "data")
    return EmitterCase(
        stmt.to_sysml, _wrap_connect, {"connectivity_reader": _connect_facts},
        {"connectivity_reader": (stmt.key(),)},
    )


def _wrap_transition(fragment: str) -> str:
    return (
        "package P { item def FaultSignal; state def ControllerBehavior { "
        "entry; then Idle; state Idle; state Responding; " + fragment + " } }"
    )


def case__src__simulation__transition_fixer__TransitionStmt__to_sysml() -> EmitterCase:
    stmt = TransitionStmt("respond", "Idle", "", "Responding", "FaultSignal")
    return EmitterCase(
        stmt.to_sysml, _wrap_transition, {"transition_reader": _transition_facts},
        {"transition_reader": ((
            stmt.name,
            stmt.source,
            stmt.accept_cmd,
            stmt.guard_raw,
            stmt.target,
        ),)},
    )


def _wrap_port(fragment: str) -> str:
    return f"package P {{ port def DataPort; part def Device {{ {fragment} }} }}"


def case__src__simulation__port_fixer__PortAdd__to_sysml() -> EmitterCase:
    addition = PortAdd("Device", "in", "data", "DataPort")
    return EmitterCase(
        addition.to_sysml, _wrap_port, {"lite_model": _port_facts},
        {"lite_model": ((addition.name, FeatureDirection.IN, addition.port_type),)},
    )


DISCOVERED = _discovered_emitters()


def _unit_suffix_on_type(text: str) -> str | None:
    """Add the unit already carried by the value to its declared type.

    Both spellings declare seconds and the numeric initializer is unchanged, so
    meaning is preserved. The precondition rejects mixed or absent units rather
    than guessing dimensional meaning.
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
    classifier; the required import is what makes the rewrite safe.
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

    `accept E then` and `accept E if true then` enable on the same event, so no
    trigger is discarded.
    """
    # Probe model-level emitters whose consumer is the Syside state reader. The
    # repair-line parser is a legacy lexical gate covered at baseline; §3.3 forbids
    # changing it here, so its disagreement is reported separately.
    if "// OWNER:" not in text and "private import ScalarValues::*;" not in text:
        return None
    match = re.search(r"(accept\s+\w+)(\s+then\s+\w+;)", text)
    if match is None:
        return None
    assert "if" not in match.group(0)
    return text[:match.start()] + match.group(1) + " if true" + match.group(2) + text[match.end():]


def _redundant_initializer(text: str) -> str | None:
    """Initialize a Boolean already constrained to true by the same contract.

    The initializer is redundant only when the same feature has a positive assume
    constraint in the emitted text; that precondition excludes
    `timingSegmentRequired`, where removing `= true` changes a fact.
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


def test_every_emitter_has_case():
    missing = [item for item in DISCOVERED if _case_name(item) not in globals()]
    assert not missing, f"discovered SysML emitters without obligations: {missing}"


@pytest.mark.parametrize("emitter", DISCOVERED)
def test_emitter_round_trips_readers(emitter):
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
def test_readers_invariant_to_spelling(
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


def test_all_spelling_classes_used():
    exercised = {perturbation.name for _, perturbation in _perturbation_cases()}
    assert exercised == {item.name for item in PERTURBATIONS}


def _perturbation_named(name: str) -> SpellingPerturbation:
    return next(item for item in PERTURBATIONS if item.name == name)


def _assert_ag_mutation_is_asymmetric_and_caught(
    perturbation_name: str,
) -> None:
    emitter = "src.prototyping.ag_emitter:emit_ag_package"
    case = globals()[_case_name(emitter)]()
    baseline = case.parse_source(case.emit())
    assert _ag_attribute_facts(baseline) == case.expected["ag_attribute_reader"]
    with pytest.raises(AssertionError):
        test_readers_invariant_to_spelling(
            emitter,
            _perturbation_named(perturbation_name),
        )


def test_unit_suffix_kills_mutation(monkeypatch):
    from src.prototyping import ag_extractor

    monkeypatch.setattr(ag_extractor, "_ATTR_RE", re.compile(
        r"\battribute\s+(\w+)\s*:\s*(\w+(?:::\w+)*)"
        r"\s*(?:=\s*(-?[\d.]+)"
        r"\s*(?:\[([^\]]+)\])?)?\s*;"
    ))
    _assert_ag_mutation_is_asymmetric_and_caught("unit suffix added/removed")


def test_qualified_type_kills_mutation(monkeypatch):
    from src.prototyping import ag_extractor

    monkeypatch.setattr(ag_extractor, "_ATTR_RE", re.compile(
        r"\battribute\s+(\w+)\s*:\s*(\w+)"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*(?:=\s*(-?[\d.]+)"
        r"\s*(?:\[([^\]]+)\])?)?\s*;"
    ))
    monkeypatch.setattr(ag_extractor, "_BOOL_ATTR_RE", re.compile(
        r"\battribute\s+(\w+)\s*:\s*Boolean"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*=\s*(true|false)\s*;",
        re.I,
    ))
    _assert_ag_mutation_is_asymmetric_and_caught("qualified/unqualified type")


def test_initializer_kills_mutation(monkeypatch):
    from src.prototyping import ag_extractor

    monkeypatch.setattr(ag_extractor, "_BOOL_ATTR_RE", re.compile(
        r"\battribute\s+((?!airborne\b)\w+)\s*:\s*(?:\w+::)*Boolean"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*=\s*(true|false)\s*;",
        re.I,
    ))
    _assert_ag_mutation_is_asymmetric_and_caught("initializer present/absent")


def test_optional_clause_kills_mutation(monkeypatch):
    original = extract_state_machines

    def optional_clause_blind(text: str):
        if re.search(r"accept\s+\w+\s+if\s+true", text):
            return []
        return original(text)

    monkeypatch.setattr(
        sys.modules[__name__], "extract_state_machines", optional_clause_blind
    )
    emitter = "src.prototyping.planned_behavior:emit_planned_behavior"
    case = globals()[_case_name(emitter)]()
    baseline = case.parse_source(case.emit())
    assert _behavior_facts(baseline) == case.expected["state_extractor"]
    with pytest.raises(AssertionError):
        test_readers_invariant_to_spelling(
            emitter,
            _perturbation_named("optional clause present/absent"),
        )


def test_repair_reader_disagrees():
    """Record, but do not repair, the reader disagreement found by this probe.

    P0-3 §3.3 treats a reader change as a finding, not a step here. The model-level
    Syside reader retains the transition; the repair-line reader drops the same
    legal construct when `accept` and `if` coexist.
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


def test_emitter_output_matches_golden():
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
