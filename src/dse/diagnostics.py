"""Diagnostic issue/recommendation generation for the DSE evaluation pipeline.

Extracted from DesignEvaluator._diagnose so the scoring class stays focused
on numeric metrics while this module owns the human-readable feedback.

Public API
----------
diagnose(model, dse_config, *, syside_attr_map, n_state_defs, syside_model)
    -> Tuple[List[str], List[str]]   (issues, recommendations)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .design_space import DesignConfiguration
from ..utils.sysml_text_utils import named_block_span
from .eval_helpers import (
    _HAS_NX,
    _has_numeric_unit_attr,
    _build_connection_graph,
    _build_port_type_map,
    _satisfied_req_ids,
    _sysml_text,
    nx,
)
from .model_facts import extract_dse_model_facts
from ..simulation.connectivity_fixer import parse_connects
from ..sysml.model import DiagnosticSeverity, SysMLModel


# A stakeholder requirement id, as distinct from an A/G contract definition.
# Both are `requirement def`; only the first is what the model is judged to
# cover. Single source; dse.evaluator imports this pattern.
_STAKEHOLDER_REQ = re.compile(r"^REQ[_-][A-Za-z]+[_-]\d+$", re.IGNORECASE)


def diagnose(
    model: SysMLModel,
    dse_config: Optional[DesignConfiguration],
    *,
    syside_attr_map: Optional[Dict[str, float]] = None,
    n_state_defs: Optional[int] = None,
    syside_model: Optional[Any] = None,
    syntax_result: Optional[Any] = None,
) -> Tuple[List[str], List[str]]:
    """Produce actionable issue + recommendation strings for *model*."""
    issues: List[str] = []
    recs:   List[str] = []
    text = _sysml_text(model)
    if syside_attr_map is None:
        syside_attr_map = {}

    # ── Untraced requirements ────────────────────────────────
    # Stakeholder requirements only. The bounded A/G layer also declares its
    # contracts as `requirement def` (the profile needs legal SysML constructs),
    # so counting every definition reported those contracts as untraced. Their
    # trace is the A/G graph, measured by ag_traceability, not a `satisfy` link.
    req_ids = {
        r.name for r in model.requirement_definitions
        if _STAKEHOLDER_REQ.match(r.name)
    } or {r.name for r in model.requirement_definitions}
    sat_ids = _satisfied_req_ids(model)
    untraced = sorted(req_ids - sat_ids)
    if untraced:
        issues.append(f"Untraced requirements: {', '.join(untraced)}")
        recs.append(
            "Add `satisfy requirement REQ_X_NNN;` inside the responsible part def "
            "for each untraced requirement."
        )

    # ── Parts without ports ──────────────────────────────────
    # Resolve `:>` specialization first: a retyped catalogue implementation
    # (`Catalog_X :> PropulsionSystem`) declares no ports of its own but
    # inherits its base's, so reporting it sent the surgical editor after
    # DSE-generated parts (10 variants plus the closure, all false positives).
    # The analysis closure and plan-declared passive parts are excluded too.
    from ..utils.sysml_text_utils import (
        part_def_bases,
        passive_components_in_text,
    )
    from .analysis_emitter import ANALYSIS_CLOSURE_DEF_NAME

    own_ports = {p.name: bool(p.ports) for p in model.part_definitions}
    type_bases = part_def_bases(text)
    passive = passive_components_in_text(text)

    def _has_ports_transitively(name: str) -> bool:
        seen: Set[str] = set()
        frontier = [name]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            if own_ports.get(current):
                return True
            frontier.extend(type_bases.get(current, ()))
        return False

    no_ports = [
        p.name for p in model.part_definitions
        if p.name != ANALYSIS_CLOSURE_DEF_NAME
        and p.name not in passive
        and not _has_ports_transitively(p.name)
    ]
    if no_ports:
        issues.append(f"Parts with no ports: {', '.join(no_ports)}")
        recs.append("Add at least one directed port (in/out/inout) to each part def.")

    no_attrs = [
        p.name for p in model.part_definitions
        if any(
            "_PERF_" in str(r) or "_CONS_" in str(r)
            for r in getattr(p, "satisfied_requirements", [])
        )
        and not _has_numeric_unit_attr(p, syside_attr_map)
    ]
    if no_attrs:
        issues.append(
            f"PERF/CONS parts missing numeric+unit attributes: {', '.join(no_attrs)}"
        )
        recs.append(
            "Add `attribute <name> : Real = <value> [<unit>];` to each PERF/CONS part def."
        )

    declared = {m.group(1) for m in re.finditer(r"\bpart\s+(\w+)\s*:\s*\w+\s*;", text)}
    in_connects = {
        g
        for stmt in parse_connects(text)
        for g in (stmt.src_inst, stmt.tgt_inst)
    }
    dangling = sorted(declared - in_connects)
    if dangling:
        issues.append(f"Dangling part usages (no connect): {', '.join(dangling)}")
        recs.append(
            "Add connect statements for all part usages so every instance "
            "participates in at least one data flow."
        )

    intf_boundary: Set[str] = set()
    for part in model.part_definitions:
        for rel in getattr(part, "satisfy_relationships", []):
            req_name = (rel.target.name if rel.target else "") or ""
            if "_INTF_" in req_name:
                intf_boundary.add(part.name)
                break

    generic_ports: List[str] = []
    for pname in intf_boundary:
        span = named_block_span(text, "part", pname)
        if span:
            generic_ports += re.findall(
                r"(?:in|out|inout)\s+port\s+(\w+)\s*:\s*(?:DataPort|RfPort|RFPort)\b",
                text[span[0] + 1:span[1]], re.IGNORECASE,
            )

    if generic_ports:
        issues.append(
            f"Generic port types on INTF parts ({len(generic_ports)}): "
            f"{', '.join(generic_ports[:5])}{'...' if len(generic_ports) > 5 else ''}"
        )
        recs.append(
            "Replace DataPort / RfPort on interface-boundary components with the "
            "protocol-specific signal type (e.g. MAVLinkSignal, CANSignal). "
            "Internal ports on non-INTF parts are exempt."
        )

    target_map: Dict[str, List[str]] = {}
    for stmt in parse_connects(text):
        src, sp, tgt, tp = (
            stmt.src_inst, stmt.src_port, stmt.tgt_inst, stmt.tgt_port
        )
        key = f"{tgt}.{tp}"
        target_map.setdefault(key, []).append(f"{src}.{sp}")
    fan_ins = {k: v for k, v in target_map.items() if len(v) > 1}
    if fan_ins:
        for tgt_key, sources in list(fan_ins.items())[:3]:
            issues.append(
                f"Fan-in on {tgt_key} from [{', '.join(sources)}]"
            )
        recs.append(
            "Each `in port` must receive from exactly one source.  "
            "Introduce an aggregator or voter part for many-to-one flows."
        )

    # ── Syside diagnostic errors ─────────────────────────────
    # Prefer the fresh syntax-gate result for the current text: the model
    # object can still carry parse diagnostics from a superseded build, and
    # reporting those next to a clean gate puts a contradictory "fix
    # compilation errors" instruction into refinement prompts.
    if syntax_result is not None:
        fresh_errors = (
            list(getattr(syntax_result, "parser_errors", ()) or ())
            + list(getattr(syntax_result, "sema_errors", ()) or ())
        )
        if fresh_errors:
            sample = "; ".join(
                str(item.get("message", item)) if isinstance(item, dict) else str(item)
                for item in fresh_errors[:3]
            )
            issues.append(
                f"Syside parse/semantic errors ({len(fresh_errors)}): {sample}"
                f"{'...' if len(fresh_errors) > 3 else ''}"
            )
            recs.append(
                "Fix compilation errors before refining the design — invalid SysML "
                "will be silently ignored by downstream tooling."
            )
    else:
        diag_errors = [
            d for d in model.diagnostics
            if d.severity == DiagnosticSeverity.ERROR
        ]
        if diag_errors:
            sample = "; ".join(d.message for d in diag_errors[:3])
            issues.append(
                f"Syside parse/semantic errors ({len(diag_errors)}): {sample}"
                f"{'...' if len(diag_errors) > 3 else ''}"
            )
            recs.append(
                "Fix compilation errors before refining the design — invalid SysML "
                "will be silently ignored by downstream tooling."
            )

    port_type_map = _build_port_type_map(model)
    for stmt in parse_connects(text):
        src_inst, src_port, tgt_inst, tgt_port = (
            stmt.src_inst, stmt.src_port, stmt.tgt_inst, stmt.tgt_port
        )
        src_type = port_type_map.get(src_port)
        tgt_type = port_type_map.get(tgt_port)
        if src_type and tgt_type:
            src_domain = "power" if "power" in src_type.lower() else "data"
            tgt_domain = "power" if "power" in tgt_type.lower() else "data"
            if src_domain != tgt_domain:
                issues.append(
                    f"Type domain mismatch: connect {src_inst}.{src_port} "
                    f"({src_type}) to {tgt_inst}.{tgt_port} ({tgt_type}) — "
                    f"power and data domains must not be mixed"
                )
            elif src_type != tgt_type:
                issues.append(
                    f"Port type mismatch: connect {src_inst}.{src_port} "
                    f"({src_type}) to {tgt_inst}.{tgt_port} ({tgt_type})"
                )
    if any("Type domain mismatch" in i or "Port type mismatch" in i for i in issues):
        recs.append(
            "Ensure connected ports share the same type.  Power ports "
            "(PowerPort) must only connect to other power ports; "
            "protocol signal ports must only connect to matching signal ports."
        )

    graph = _build_connection_graph(text)
    graph_nodes = graph.nodes() if callable(graph.nodes) else set(graph.nodes())
    if graph_nodes:
        comps = (
            list(nx.weakly_connected_components(graph))
            if _HAS_NX
            else graph.weakly_connected_components()
        )
        if len(comps) > 1:
            component_members = sorted(
                (sorted(str(member) for member in component) for component in comps),
                key=lambda members: (-len(members), members),
            )
            comp_sizes = [len(component) for component in component_members]
            issues.append(
                "UNTRACED_ADVISORY: connection graph has "
                f"{len(comps)} weakly connected components "
                f"(sizes: {comp_sizes}; members: {component_members}). "
                "Global connectedness is not a frozen requirement obligation."
            )
            recs.append(
                "Review each weak component against frozen structural obligations "
                "and declared external boundaries. Do not add a connection unless "
                "it has requirement or explicit design-decision provenance."
            )

        # Feedback cycles are not reported. A closed loop (command down + status
        # up, or sensor->controller->actuator->plant->sensor) is the normal
        # topology of a control system; self-loops are already excluded and
        # bidirectional 2-cycles are valid, so there is no bad cycle class to
        # flag. Reporting them added a non-actionable "verify" note and could
        # trigger extra refinement rounds. Connectivity defects are caught by the
        # disconnected-component check above and by the reachability simulation.

    safe_reqs = [r for r in model.requirement_definitions if "_SAFE_" in r.name]
    state_defs = n_state_defs if n_state_defs is not None else \
        len(re.findall(r"\bstate\s+def\s+\w+", text))
    if safe_reqs and state_defs == 0:
        ids = ", ".join(r.name for r in safe_reqs)
        issues.append(
            f"SAFE requirement(s) {ids} have no state-machine implementations"
        )
        recs.append(
            "For each SAFE requirement add a `state def` with at least one fault "
            "entry state, a fault transition, and an `action def emergencyXxx {{ }}`."
        )

    if dse_config:
        params = dse_config.parameters
        facts = extract_dse_model_facts(
            text, params, syside_model=syside_model
        )

        redundancy = str(params.get("redundancy_level", "none")).lower()
        if facts.redundancy is not None:
            if not facts.redundancy.has_composite_voting:
                issues.append(
                    f"MCTS: {redundancy} redundancy lacks voting logic — "
                    "channel transitions fire independently with no majority rule"
                )
                recs.append(
                    "Inside the redundancy state def, add a transition whose guard "
                    "combines ≥ 2 channel conditions, using the canonical SysML v2 "
                    "`first / if / then` keywords:\n"
                    "  transition majorityFail\n"
                    "      first Active\n"
                    "      if channelAFailed and channelBFailed\n"
                    "      then FailsafeActive;"
                )
            ungrounded = sorted(facts.redundancy.ungrounded_guards)
            if ungrounded:
                issues.append(
                    f"MCTS: {redundancy} redundancy uses ungrounded guards — "
                    f"{', '.join(ungrounded[:3])}"
                    f"{'...' if len(ungrounded) > 3 else ''} "
                    "(no in port or attribute produces these signals)"
                )
                recs.append(
                    "For each guard like `channelXFailed`, declare a producer in "
                    "the same part body, e.g.:\n"
                    "  in port channelAStatus : DataPort;\n"
                    "  attribute channelAFailed : Boolean = false;"
                )

        if facts.protocol is not None:
            protocol = facts.protocol
            protocol_id = protocol.signal_type.removesuffix("Signal")
            if protocol.has_definition and not protocol.has_body:
                issues.append(
                    f"MCTS: `port def {protocol.signal_type}` is an empty forward "
                    "declaration — protocol carries no payload type"
                )
                recs.append(
                    f"Replace `port def {protocol.signal_type};` with a proper "
                    "definition that references the protocol's payload item def, e.g.:\n"
                    f"  port def {protocol.signal_type} {{ inout item data : "
                    f"{protocol_id}Telemetry; }}"
                )
            if protocol.stale_generic_definitions:
                issues.append(
                    "Stale generic port defs remain: "
                    + ", ".join(protocol.stale_generic_definitions)
                )
                recs.append(
                    "Remove the unused `port def DataPort { ... }` / `port def "
                    f"RfPort {{ ... }}` block declarations — all usages now reference "
                    f"`{protocol.signal_type}`, so these are dead code."
                )
            if protocol.power_misuse_ports:
                issues.append(
                    f"MCTS: protocol type {protocol.signal_type} incorrectly applied "
                    "to power port(s): "
                    + ", ".join(protocol.power_misuse_ports)
                )
                recs.append(
                    "Restore PowerPort type on power-domain ports — they carry "
                    "electrical energy, not protocol data."
                )

        if facts.sensors is not None:
            expected = int(params.get("num_sensors", 0))
            instances = facts.sensors.instances
            if len(instances) < expected:
                issues.append(
                    f"MCTS: num_sensors={expected} but only {len(instances)} "
                    "sensor part usage(s) declared"
                )
                recs.append(
                    f"Add {expected - len(instances)} more "
                    "`part sensorUnitN : <SensorPartDef>;` declarations."
                )
            dangling = sorted(instances - facts.sensors.connected_instances)
            if dangling:
                issues.append(
                    "MCTS: redundant sensor instances are dangling (no connect): "
                    + ", ".join(dangling)
                )
                recs.append(
                    "Introduce an aggregator/voter part def and connect every "
                    "redundant sensor instance to it, e.g.:\n"
                    "  part def SensorVoter { in port a, b, c : DataPort; "
                    "out port voted : DataPort; }\n"
                    "  connect sensorUnit2.sensorStatus to sensorVoter.b;"
                )
            if not facts.sensors.has_aggregator:
                issues.append(
                    f"MCTS: {expected}-way sensor redundancy has no voter/aggregator "
                    "part — redundant readings are not combined"
                )
                recs.append(
                    "Add a Voter / Aggregator / FusionUnit part def that consumes "
                    "all redundant sensor outputs and produces a single combined output."
                )

        if facts.control is not None and not facts.control.has_behavior:
            frequency = float(params.get("control_frequency_hz", 0))
            issues.append(
                f"MCTS: control_frequency={frequency}Hz set but the controller "
                "part has no action def or state def to drive at that rate"
            )
            recs.append(
                "Inside the main controller part def, add a periodic action def "
                "or a state machine whose transitions reference `controlFrequency`."
            )

        distributed = params.get("distributed_control")
        controller_count = facts.controller_count or 0
        if distributed is True and controller_count < 3:
            issues.append(
                f"MCTS: distributed_control=True but only {controller_count} "
                "controller-class part def(s) — topology is not actually distributed"
            )
            recs.append(
                "Split control logic into ≥3 part defs whose names reflect their role "
                "(e.g., NavigationController, PayloadManager, PowerManager)."
            )
        elif distributed is False and controller_count > 2:
            issues.append(
                f"MCTS: distributed_control=False but {controller_count} "
                "`*Controller` part defs exist — topology contradicts the "
                "centralisation decision"
            )
            recs.append(
                "Consolidate controllers into one centralised controller part def, "
                "absorbing the responsibilities of the others."
            )
    return issues, recs
