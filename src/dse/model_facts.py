"""One parser-backed view of the model facts used by DSE scoring and diagnosis.

The facts are policy-free: this module reports what is present in the committed
model, ``evaluator`` decides scores and ``diagnostics`` decides messages, which
keeps numerical policy out of the shared seam.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .eval_helpers import _SENSOR_USAGE_RE
from ..simulation.connectivity_fixer import ConnectStmt, parse_connects
from ..utils.sysml_text_utils import find_block_end
from ..utils.syside_utils import SYSIDE_OK, syside


@dataclass(frozen=True)
class RedundancyFacts:
    shape_count: int
    has_composite_voting: bool
    guard_names: frozenset[str]
    ungrounded_guards: frozenset[str]


@dataclass(frozen=True)
class ProtocolFacts:
    signal_type: str
    has_definition: bool
    has_body: bool
    item_linked: bool
    stale_generic_definitions: Tuple[str, ...]
    power_misuse_ports: Tuple[str, ...]


@dataclass(frozen=True)
class SensorFacts:
    instances: frozenset[str]
    connected_instances: frozenset[str]
    has_aggregator: bool


@dataclass(frozen=True)
class ControlFacts:
    frequency_in_controller: bool
    has_behavior: bool


@dataclass(frozen=True)
class DSEModelFacts:
    connections: Tuple[ConnectStmt, ...]
    redundancy: Optional[RedundancyFacts] = None
    protocol: Optional[ProtocolFacts] = None
    sensors: Optional[SensorFacts] = None
    control: Optional[ControlFacts] = None
    controller_count: Optional[int] = None


def _part_bodies_matching(text: str, name_pattern: str) -> str:
    pattern = re.compile(
        rf"\bpart\s+def\s+(\w*(?:{name_pattern})\w*)\s*\{{",
        re.IGNORECASE,
    )
    bodies: list[str] = []
    for match in pattern.finditer(text):
        opening = text.index("{", match.start())
        closing = find_block_end(text, opening)
        if closing != -1:
            bodies.append(text[opening:closing])
    return "\n".join(bodies)


def _connection_facts(model_text: str) -> Tuple[ConnectStmt, ...]:
    try:
        return tuple(parse_connects(model_text))
    except Exception:
        # DSE also runs on partially generated models: a parser failure means no
        # trustworthy connection evidence, and evaluation continues.
        return ()


def _redundancy_facts(text: str) -> RedundancyFacts:
    body = _part_bodies_matching(text, "Safety|Monitor|Fault|Health")
    channel_states = re.findall(
        r"\bstate\s+(Channel[A-Z]\w*|Primary\w*|Backup\w*|Standby\w*)",
        body,
        re.IGNORECASE,
    )
    channel_bools = re.findall(
        r"\battribute\s+(channel[A-Z]\w*Failed|primary\w*Failed|backup\w*Failed)"
        r"\s*:\s*Boolean",
        body,
        re.IGNORECASE,
    )
    voting_patterns = (
        re.compile(
            r"\btransition\b(?:\s+(?!first\b)\w+)?\s+first\s+\w+"
            r"(?:\s+accept\s+[^;]+?)?"
            r"\s+if\s+([^;]+?)\s+then\s+\w+\s*;",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\btransition\s+\w+\s+from\s+\w+\s+to\s+\w+"
            r"\s+when\s+([^;]+);",
            re.IGNORECASE,
        ),
    )
    guards = [
        match.group(1).lower()
        for pattern in voting_patterns
        for match in pattern.finditer(body)
    ]
    has_voting = any(
        " and " in guard
        or " or " in guard
        or len(re.findall(r"channel[a-z]\w*", guard)) >= 2
        for guard in guards
    )
    literals = {"true", "false"}
    guard_names = {
        match.group(1)
        for keyword in ("if", "when")
        for match in re.finditer(
            rf"\b{keyword}\s+(\w+)", body, re.IGNORECASE
        )
        if match.group(1).lower() not in literals
    }
    ungrounded = {
        guard
        for guard in guard_names
        if not re.search(
            rf"\b(?:in\s+port|attribute)\s+{re.escape(guard)}\b",
            body,
            re.IGNORECASE,
        )
    }
    return RedundancyFacts(
        shape_count=max(len(set(channel_states)), len(set(channel_bools))),
        has_composite_voting=has_voting,
        guard_names=frozenset(guard_names),
        ungrounded_guards=frozenset(ungrounded),
    )


def _protocol_facts(text: str, protocol: str) -> ProtocolFacts:
    protocol_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
    signal_type = f"{protocol_id}Signal"
    body_match = re.search(
        rf"port\s+def\s+{re.escape(signal_type)}\s*\{{([^}}]*)\}}",
        text,
        re.IGNORECASE,
    )
    body = body_match.group(1) if body_match else ""
    stale = tuple(sorted(set(re.findall(
        r"\bport\s+def\s+(DataPort|RfPort|RFPort|GenericPort)\s*\{",
        text,
        re.IGNORECASE,
    ))))
    power_misuse = tuple(re.findall(
        rf"\b(?:in|out|inout)\s+port\s+(\w*[Pp]ower\w*)\s*:\s*"
        rf"{re.escape(signal_type)}\b",
        text,
    ))
    return ProtocolFacts(
        signal_type=signal_type,
        has_definition=bool(re.search(
            rf"port\s+def\s+{re.escape(signal_type)}\b", text, re.IGNORECASE
        )),
        has_body=body_match is not None,
        item_linked=bool(re.search(
            rf"\bitem\s+\w+\s*:\s*\w*{re.escape(protocol_id)}\w*",
            body,
            re.IGNORECASE,
        )),
        stale_generic_definitions=stale,
        power_misuse_ports=power_misuse,
    )


def _controller_count(text: str, distributed: bool, syside_model: Any) -> int:
    if syside_model is not None and SYSIDE_OK:
        part_definition = getattr(syside, "PartDefinition", None)
        if part_definition is not None:
            if distributed:
                keywords = ("controller", "manager", "module", "subsystem", "node")
                return len({
                    part.name
                    for part in syside_model.nodes(part_definition)
                    if any(keyword in (part.name or "").lower() for keyword in keywords)
                })
            return sum(
                1
                for part in syside_model.nodes(part_definition)
                if "controller" in (part.name or "").lower()
            )
    if distributed:
        return len(set(re.findall(
            r"\bpart\s+def\s+(\w*(?:Controller|Manager|Module|Subsystem|Node)\w*)"
            r"\s*\{",
            text,
            re.IGNORECASE,
        )))
    return len(re.findall(
        r"\bpart\s+def\s+\w*Controller\w*\s*\{", text, re.IGNORECASE
    ))


def extract_dse_model_facts(
    model_text: str,
    parameters: Mapping[str, Any],
    *,
    syside_model: Any = None,
) -> DSEModelFacts:
    """Extract the shared observable facts for one selected DSE configuration."""
    text = model_text or ""
    connections = _connection_facts(text)

    redundancy = str(parameters.get("redundancy_level", "none")).lower()
    redundancy_facts = (
        _redundancy_facts(text) if redundancy in {"dual", "triple"} else None
    )

    protocol = str(parameters.get("communication_protocol", "")).strip()
    protocol_facts = (
        _protocol_facts(text, protocol)
        if protocol and protocol.lower() != "none"
        else None
    )

    sensor_facts: Optional[SensorFacts] = None
    if int(parameters.get("num_sensors", 0)) > 1:
        instances = frozenset(
            match.group(1) for match in _SENSOR_USAGE_RE.finditer(text)
        )
        connected_parts = frozenset(
            endpoint
            for statement in connections
            for endpoint in (statement.src_inst, statement.tgt_inst)
        )
        sensor_facts = SensorFacts(
            instances=instances,
            connected_instances=frozenset(instances & connected_parts),
            has_aggregator=bool(re.search(
                r"\bpart\s+def\s+\w*"
                r"(?:Aggregat|Voter|Fusion|Combiner|Arbiter|Merger|Selector)\w*",
                text,
                re.IGNORECASE,
            )),
        )

    control_facts: Optional[ControlFacts] = None
    frequency = float(parameters.get("control_frequency_hz", 0))
    if frequency > 0:
        body = _part_bodies_matching(
            text, "Controller|Flight|Autopilot|Nav|MainControl"
        )
        frequency_text = f"{frequency:.1f}".rstrip("0").rstrip(".")
        control_facts = ControlFacts(
            frequency_in_controller=bool(re.search(
                rf"controlFrequency\s*:\s*Real\s*=\s*{re.escape(frequency_text)}",
                body,
            )),
            has_behavior=bool(
                re.search(r"\baction\s+def\s+\w+", body, re.IGNORECASE)
                or re.search(r"\bstate\s+def\s+\w+", body, re.IGNORECASE)
            ),
        )

    distributed = parameters.get("distributed_control")
    count = (
        _controller_count(text, distributed, syside_model)
        if isinstance(distributed, bool)
        else None
    )
    return DSEModelFacts(
        connections=connections,
        redundancy=redundancy_facts,
        protocol=protocol_facts,
        sensors=sensor_facts,
        control=control_facts,
        controller_count=count,
    )
