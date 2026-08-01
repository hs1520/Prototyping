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
from .eval_helpers import (
    _HAS_NX,
    _has_numeric_unit_attr,
    _SENSOR_USAGE_RE,
    _build_connection_graph,
    _build_port_type_map,
    _satisfied_req_ids,
    _sysml_text,
    nx,
)
from ..sysml.model import DiagnosticSeverity, SysMLModel
from ..utils.sysml_text_utils import find_block_end
from ..utils.syside_utils import SYSIDE_OK as _SYSIDE_EVAL_OK, syside as _syside_eval


#: A stakeholder requirement id, as distinct from an A/G contract definition.
#: Both are `requirement def`; only the first is what the model is judged to
#: cover. Mirrors `_STAKEHOLDER_REQ` in dse.evaluator.
_STAKEHOLDER_REQ = re.compile(r"^REQ[_-][A-Za-z]+[_-]\d+$", re.IGNORECASE)


def _connections(model_text: str):
    """Every `connect a.x to b.y`, from the parser where Syside is available.

    Four sites here asked this with four patterns. Sharing one entry also means
    prose in a requirement `doc` comment is no longer read as a declaration.
    """
    from src.simulation.connectivity_fixer import parse_connects

    try:
        return parse_connects(model_text)
    except Exception:
        return []


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

    # ── Untraced requirements ────────────────────────────────────────────
    # Stakeholder requirements only. A model carrying the bounded A/G layer also
    # declares its contracts as `requirement def`, because the profile requires
    # legal SysML requirement constructs — so counting every definition reported
    # the assurance layer's own contracts as untraced requirements. Their trace
    # is the A/G graph, measured by ag_traceability, not a `satisfy` link.
    # Same denominator confusion the evaluator's coverage dimension had.
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

    # ── Parts without ports ──────────────────────────────────────────────
    no_ports = [p.name for p in model.part_definitions if not p.ports]
    if no_ports:
        issues.append(f"Parts with no ports: {', '.join(no_ports)}")
        recs.append("Add at least one directed port (in/out/inout) to each part def.")

    # ── PERF/CONS parts without numeric+unit attributes ──────────────────
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

    # ── Dangling part usages ─────────────────────────────────────────────
    declared = {m.group(1) for m in re.finditer(r"\bpart\s+(\w+)\s*:\s*\w+\s*;", text)}
    in_connects = {
        g
        for stmt in _connections(text)
        for g in (stmt.src_inst, stmt.tgt_inst)
    }
    dangling = sorted(declared - in_connects)
    if dangling:
        issues.append(f"Dangling part usages (no connect): {', '.join(dangling)}")
        recs.append(
            "Add connect statements for all part usages so every instance "
            "participates in at least one data flow."
        )

    # ── Residual DataPort / RfPort on INTF-boundary parts ───────────────
    intf_boundary: Set[str] = set()
    for part in model.part_definitions:
        for rel in getattr(part, "satisfy_relationships", []):
            req_name = (rel.target.name if rel.target else "") or ""
            if "_INTF_" in req_name:
                intf_boundary.add(part.name)
                break

    generic_ports: List[str] = []
    for pname in intf_boundary:
        block_m = re.search(
            rf"\bpart\s+def\s+{re.escape(pname)}\s*\{{(.*?)\n\s*\}}",
            text, re.DOTALL,
        )
        if block_m:
            generic_ports += re.findall(
                r"(?:in|out|inout)\s+port\s+(\w+)\s*:\s*(?:DataPort|RfPort|RFPort)\b",
                block_m.group(1), re.IGNORECASE,
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

    # ── Fan-in violations ────────────────────────────────────────────────
    target_map: Dict[str, List[str]] = {}
    for stmt in _connections(text):
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

    # ── Syside diagnostic errors ─────────────────────────────────────────
    # Prefer the fresh syntax-gate result for the current model text.  The
    # model object may still carry parse diagnostics from a superseded build
    # (for example from before deterministic guard-attribute injection), and
    # reporting those alongside a clean syntax gate injects a contradictory
    # "fix compilation errors" instruction into refinement prompts.
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

    # ── Connect type mismatches ──────────────────────────────────────────
    port_type_map = _build_port_type_map(model)
    for stmt in _connections(text):
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

    # ── Isolated sub-graphs ──────────────────────────────────────────────
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

        # NOTE: feedback cycles are intentionally NOT reported.  A closed loop
        # in the connection graph (command down + status up, or
        # sensor→controller→actuator→plant→sensor) is the normal, required
        # topology of a control system — not a defect.  The graph already
        # excludes self-loops, and bidirectional 2-cycles are legitimate, so
        # there is no structurally "bad" cycle class to flag.  Reporting them
        # only produced a non-actionable "verify" note that polluted the issue
        # list and could trigger unnecessary refinement rounds.  Genuine
        # connectivity defects are caught by the disconnected-component check
        # above and by the reachability simulation.

    # ── SAFE requirements without state machines ─────────────────────────
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

    # ── MCTS decision gaps ───────────────────────────────────────────────
    if dse_config:
        params = dse_config.parameters

        # Redundancy
        redundancy = str(params.get("redundancy_level", "none")).lower()
        if redundancy in ("triple", "dual"):
            n_channels = 3 if redundancy == "triple" else 2  # noqa: F841
            safety_pat = re.compile(
                r"\bpart\s+def\s+\w*(?:Safety|Monitor|Fault|Health)\w*\s*\{",
                re.IGNORECASE,
            )
            safety_match = safety_pat.search(text)
            safety_body = ""
            if safety_match:
                brace_open = text.index("{", safety_match.start())
                end = find_block_end(text, brace_open)
                if end != -1:
                    safety_body = text[brace_open:end]

            # Accept both the canonical SysML v2 form (`first X if <guard>
            # then Y;`) and the legacy form (`from X to Y when <guard>;`) —
            # mirrors _score_dse_fidelity so diagnosis and scoring agree.
            voting_canonical = re.compile(
                # the optional `accept` clause is legal and is what the A/G
                # profile writes; without it the guard is invisible here
                r"\btransition\s+\w+\s+first\s+\w+"
                r"(?:\s+accept\s+[^;]+?)?"
                r"\s+if\s+([^;]+?)\s+then\s+\w+\s*;",
                re.IGNORECASE | re.DOTALL,
            )
            voting_legacy = re.compile(
                r"\btransition\s+\w+\s+from\s+\w+\s+to\s+\w+\s+when\s+([^;]+);",
                re.IGNORECASE,
            )
            has_voting = any(
                " and " in m.group(1).lower() or " or " in m.group(1).lower()
                or len(re.findall(r"channel[a-z]\w*", m.group(1).lower())) >= 2
                for pat in (voting_canonical, voting_legacy)
                for m in pat.finditer(safety_body)
            )
            if not has_voting:
                issues.append(
                    f"MCTS: {redundancy} redundancy lacks voting logic — "
                    f"channel transitions fire independently with no majority rule"
                )
                recs.append(
                    f"Inside the redundancy state def, add a transition whose guard "
                    f"combines ≥ 2 channel conditions, using the canonical SysML v2 "
                    f"`first / if / then` keywords:\n"
                    f"  transition majorityFail\n"
                    f"      first Active\n"
                    f"      if channelAFailed and channelBFailed\n"
                    f"      then FailsafeActive;"
                )

            # Collect guard identifiers from both syntaxes; skip boolean
            # literals which need no producer (mirrors _score_dse_fidelity).
            _GUARD_LITERALS = {"true", "false"}
            guard_names = {
                m.group(1)
                for kw in (r"if", r"when")
                for m in re.finditer(rf"\b{kw}\s+(\w+)", safety_body, re.IGNORECASE)
                if m.group(1).lower() not in _GUARD_LITERALS
            }
            ungrounded = [
                g for g in guard_names
                if not re.search(
                    rf"\b(?:in\s+port|attribute)\s+{re.escape(g)}\b",
                    safety_body, re.IGNORECASE,
                )
            ]
            if ungrounded:
                issues.append(
                    f"MCTS: {redundancy} redundancy uses ungrounded guards — "
                    f"{', '.join(sorted(ungrounded)[:3])}"
                    f"{'...' if len(ungrounded) > 3 else ''} "
                    f"(no in port or attribute produces these signals)"
                )
                recs.append(
                    "For each guard like `channelXFailed`, declare a producer in the "
                    "same part body, e.g.:\n"
                    "  in port channelAStatus : DataPort;\n"
                    "  attribute channelAFailed : Boolean = false;"
                )

        # Communication protocol
        protocol = str(params.get("communication_protocol", "")).strip()
        if protocol and protocol.lower() not in ("none", ""):
            proto_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
            signal_type = f"{proto_id}Signal"

            has_def = bool(re.search(
                rf"port\s+def\s+{re.escape(signal_type)}\b", text, re.IGNORECASE,
            ))
            has_body = bool(re.search(
                rf"port\s+def\s+{re.escape(signal_type)}\s*\{{[^}}]*\}}",
                text, re.IGNORECASE,
            ))
            if has_def and not has_body:
                issues.append(
                    f"MCTS: `port def {signal_type}` is an empty forward declaration"
                    f" — protocol carries no payload type"
                )
                recs.append(
                    f"Replace `port def {signal_type};` with a proper definition that "
                    f"references the protocol's payload item def, e.g.:\n"
                    f"  port def {signal_type} {{ inout item data : {proto_id}Telemetry; }}"
                )

            leftover = re.findall(
                r"\bport\s+def\s+(DataPort|RfPort|RFPort|GenericPort)\s*\{",
                text, re.IGNORECASE,
            )
            if leftover:
                issues.append(
                    f"Stale generic port defs remain: "
                    f"{', '.join(sorted(set(leftover)))}"
                )
                recs.append(
                    "Remove the unused `port def DataPort { ... }` / `port def RfPort "
                    "{ ... }` block declarations — all usages now reference "
                    f"`{signal_type}`, so these are dead code."
                )

            power_misuse = re.findall(
                rf"\b(?:in|out|inout)\s+port\s+(\w*[Pp]ower\w*)\s*:\s*"
                rf"{re.escape(signal_type)}\b",
                text,
            )
            if power_misuse:
                issues.append(
                    f"MCTS: protocol type {signal_type} incorrectly applied to "
                    f"power port(s): {', '.join(power_misuse)}"
                )
                recs.append(
                    "Restore PowerPort type on power-domain ports — they carry "
                    "electrical energy, not protocol data."
                )

        # Sensor redundancy
        num_sensors = int(params.get("num_sensors", 0))
        if num_sensors > 1:
            instances = {m.group(1) for m in _SENSOR_USAGE_RE.finditer(text)}
            if len(instances) < num_sensors:
                issues.append(
                    f"MCTS: num_sensors={num_sensors} but only "
                    f"{len(instances)} sensor part usage(s) declared"
                )
                recs.append(
                    f"Add {num_sensors - len(instances)} more "
                    f"`part sensorUnitN : <SensorPartDef>;` declarations."
                )

            connected_parts: set = set()
            for stmt in _connections(text):
                connected_parts.add(stmt.src_inst)
                connected_parts.add(stmt.tgt_inst)
            dangling_sensors = sorted(instances - connected_parts)
            if dangling_sensors:
                issues.append(
                    f"MCTS: redundant sensor instances are dangling "
                    f"(no connect): {', '.join(dangling_sensors)}"
                )
                recs.append(
                    "Introduce an aggregator/voter part def and connect every "
                    "redundant sensor instance to it, e.g.:\n"
                    "  part def SensorVoter { in port a, b, c : DataPort; "
                    "out port voted : DataPort; }\n"
                    "  connect sensorUnit2.sensorStatus to sensorVoter.b;"
                )

            has_aggregator = bool(re.search(
                r"\bpart\s+def\s+\w*"
                r"(?:Aggregat|Voter|Fusion|Combiner|Arbiter|Merger|Selector)\w*",
                text, re.IGNORECASE,
            ))
            if not has_aggregator:
                issues.append(
                    f"MCTS: {num_sensors}-way sensor redundancy has no "
                    f"voter/aggregator part — redundant readings are not combined"
                )
                recs.append(
                    "Add a Voter / Aggregator / FusionUnit part def that consumes "
                    "all redundant sensor outputs and produces a single combined output."
                )

        # Control frequency
        freq = float(params.get("control_frequency_hz", 0))
        if freq > 0:
            ctrl_pat = re.compile(
                r"\bpart\s+def\s+\w*"
                r"(?:Controller|Flight|Autopilot|Nav|MainControl)\w*\s*\{",
                re.IGNORECASE,
            )
            ctrl_match = ctrl_pat.search(text)
            ctrl_body = ""
            if ctrl_match:
                brace_open = text.index("{", ctrl_match.start())
                end = find_block_end(text, brace_open)
                if end != -1:
                    ctrl_body = text[brace_open:end]
            has_behaviour = bool(
                re.search(r"\baction\s+def\s+\w+", ctrl_body, re.IGNORECASE)
                or re.search(r"\bstate\s+def\s+\w+", ctrl_body, re.IGNORECASE)
            )
            if not has_behaviour:
                issues.append(
                    f"MCTS: control_frequency={freq}Hz set but the controller "
                    f"part has no action def or state def to drive at that rate"
                )
                recs.append(
                    "Inside the main controller part def, add a periodic action "
                    "def or a state machine whose transitions reference "
                    "`controlFrequency`."
                )

        # Distributed control topology
        distributed = params.get("distributed_control")
        _diag_pd_cls = getattr(_syside_eval, "PartDefinition", None) if _SYSIDE_EVAL_OK else None
        if distributed is True:
            if syside_model is not None and _diag_pd_cls is not None:
                _dist_kws = ("controller", "manager", "module", "subsystem", "node")
                ctrl_count = len({
                    pd.name for pd in syside_model.nodes(_diag_pd_cls)
                    if any(kw in (pd.name or "").lower() for kw in _dist_kws)
                })
            else:
                ctrl_count = len(set(re.findall(
                    r"\bpart\s+def\s+(\w*"
                    r"(?:Controller|Manager|Module|Subsystem|Node)\w*)\s*\{",
                    text, re.IGNORECASE,
                )))
            if ctrl_count < 3:
                issues.append(
                    f"MCTS: distributed_control=True but only {ctrl_count} "
                    f"controller-class part def(s) — topology is not actually distributed"
                )
                recs.append(
                    "Split control logic into ≥3 part defs whose names reflect their "
                    "role (e.g., NavigationController, PayloadManager, PowerManager)."
                )
        elif distributed is False:
            if syside_model is not None and _diag_pd_cls is not None:
                ctrl_count = sum(
                    1 for pd in syside_model.nodes(_diag_pd_cls)
                    if "controller" in (pd.name or "").lower()
                )
            else:
                ctrl_count = len(re.findall(
                    r"\bpart\s+def\s+\w*Controller\w*\s*\{",
                    text, re.IGNORECASE,
                ))
            if ctrl_count > 2:
                issues.append(
                    f"MCTS: distributed_control=False but {ctrl_count} "
                    f"`*Controller` part defs exist — topology contradicts the "
                    f"centralisation decision"
                )
                recs.append(
                    "Consolidate controllers into one centralised controller part def, "
                    "absorbing the responsibilities of the others."
                )

    return issues, recs
