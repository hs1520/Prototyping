"""Prompt-mandated safety interconnect wiring, declared once.

Both authoring prompts hard-mandate the SafetyMonitor interconnect (override
command, communication status, sensor status) whenever the corresponding
component roles are present — yet whether the step-1 typed plan actually
declared those ports was left to the model.  When a roll skipped them, plan
conformance flagged prompt-mandated structure as unplanned (measured: four
unplanned ports + two unplanned connections on ablation pilot 2 and on
authoritative run 219eb9bb, failing TYPED_GENERATION_PLAN_CONFORMANCE).

This module is the single source of truth for that wiring: the two prompt
blocks live here (spliced verbatim into the templates), and the accepted plan
payload is augmented from the same table (planned-by-construction, the
catalog-seed pattern), so the authoring mandate and the conformance gate can
never drift apart.  The assembly stage already materialises planned
connections deterministically, which makes the augmentation self-healing: a
roll that forgets the wiring gets it restored rather than flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class MandatedLink:
    """One mandated out→in interconnect between two component roles."""

    source_role: str
    target_role: str
    port_name: str
    port_type: str = "DataPort"


#: source drives target; the link fires only when BOTH roles resolve to
#: exactly one planned component (matching the prompts' conditionality).
MANDATED_LINKS: Tuple[MandatedLink, ...] = (
    MandatedLink("safety_monitor", "controller", "overrideCmd"),
    MandatedLink("communication", "safety_monitor", "commStatus"),
    MandatedLink("perception", "safety_monitor", "sensorStatus"),
)

#: Keyword sets mirror the prompts' own role phrasings ("SafetyMonitor (or
#: similar safety-enforcement component)", "main controller/autopilot",
#: "comms/link", "sensor/IMU/camera").
ROLE_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "safety_monitor": ("safetymonitor", "safetyenforcement"),
    "controller": ("controller", "autopilot"),
    "communication": ("communication", "comms"),
    "perception": ("perception", "sensor", "imu", "camera"),
}

#: The authoritative copy of the block inside ARCHITECTURE_DECOMPOSITION_TEMPLATE.
#: tests/test_mandated_wiring.py asserts the template contains it byte-for-byte
#: AND that every MANDATED_LINKS entry appears in it, so prompt, table, and
#: augmentation cannot drift apart silently.  (Runtime splicing was rejected:
#: llm/__init__ eagerly imports chain_of_thought while prototyping/__init__
#: eagerly imports provider_factory -> llm.interface, so an llm -> prototyping
#: module-level import closes a package-init cycle.)
PLAN_SIDE_RULES_BLOCK = """\
- Safety interconnect ports (MANDATORY when these component types appear):
    • If a SafetyMonitor (or similar safety-enforcement component) is listed:
        – The main controller/autopilot component MUST include `in overrideCmd` in its port list.
        – SafetyMonitor MUST include `out overrideCmd` in its port list.
    • If a CommunicationSystem (or comms/link component) is listed:
        – It MUST include `out commStatus` in its port list.
        – SafetyMonitor MUST include `in commStatus` in its port list.
    • If a PerceptionSystem (or sensor/IMU/camera component) is listed:
        – It MUST include `out sensorStatus` in its port list.
        – SafetyMonitor MUST include `in sensorStatus` in its port list.
"""

#: Rendered verbatim into PART_DEFINITIONS_TEMPLATE.
PART_SIDE_RULES_BLOCK = """\
- Safety interconnect ports (MANDATORY — add these whenever the component type is present):
    • If a SafetyMonitor part def is defined:
        – The main controller/autopilot part def MUST declare `in port overrideCmd : DataPort;`
        – SafetyMonitor MUST declare `out port overrideCmd : DataPort;`
    • If a CommunicationSystem part def is defined:
        – CommunicationSystem MUST declare `out port commStatus : DataPort;`
        – SafetyMonitor MUST declare `in port commStatus : DataPort;`
    • If a PerceptionSystem (or sensor/IMU) part def is defined:
        – PerceptionSystem MUST declare `out port sensorStatus : DataPort;`
        – SafetyMonitor MUST declare `in port sensorStatus : DataPort;`
"""


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _resolve_role(role: str, component_names: list[str]) -> tuple[str | None, bool]:
    """(unique component name, ambiguous?) for a role, by keyword match."""
    keywords = ROLE_KEYWORDS[role]
    matches = [
        name for name in component_names
        if any(keyword in _normalise(name) for keyword in keywords)
    ]
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def augment_architecture_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Plan the mandated wiring by construction on an LLM plan payload.

    Returns an augmented copy of the payload plus one advisory note per
    deterministic addition.  Never overwrites anything the model declared: a
    same-name port with a conflicting direction/type, an ambiguous role, or a
    passive endpoint skips that link with a note instead.  Idempotent.
    """
    notes: list[str] = []
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        return dict(payload), notes

    components = [
        dict(component) if isinstance(component, Mapping) else component
        for component in raw_components
    ]
    named = {
        component["name"]: component
        for component in components
        if isinstance(component, Mapping) and component.get("name")
    }
    component_names = list(named)

    raw_connections = payload.get("connections")
    connections = [
        dict(connection) if isinstance(connection, Mapping) else connection
        for connection in (raw_connections if isinstance(raw_connections, list) else [])
    ]

    def port_entry(component: Mapping[str, Any], port_name: str):
        for port in component.get("ports") or []:
            if isinstance(port, Mapping) and str(port.get("name")) == port_name:
                return port
        return None

    def ensure_port(component_name: str, port_name: str,
                    direction: str, port_type: str) -> bool | None:
        """True=added, False=already planned compatibly, None=conflict."""
        component = named[component_name]
        existing = port_entry(component, port_name)
        if existing is not None:
            compatible = (
                str(existing.get("direction")) == direction
                and str(existing.get("type") or "DataPort") == port_type
            )
            return False if compatible else None
        ports = list(component.get("ports") or [])
        ports.append({
            "name": port_name,
            "direction": direction,
            "type": port_type,
            "external": False,
        })
        component["ports"] = ports
        return True

    for link in MANDATED_LINKS:
        source, source_ambiguous = _resolve_role(link.source_role, component_names)
        target, target_ambiguous = _resolve_role(link.target_role, component_names)
        if source_ambiguous or target_ambiguous:
            notes.append(
                f"mandated wiring skipped ({link.port_name}): role "
                f"{'source' if source_ambiguous else 'target'} matches more "
                "than one planned component"
            )
            continue
        if source is None or target is None:
            continue   # role absent — the mandate's condition does not fire
        if named[source].get("passive") or named[target].get("passive"):
            notes.append(
                f"mandated wiring skipped ({link.port_name}): endpoint is "
                "declared passive"
            )
            continue

        source_state = ensure_port(source, link.port_name, "out", link.port_type)
        target_state = ensure_port(target, link.port_name, "in", link.port_type)
        if source_state is None or target_state is None:
            notes.append(
                f"mandated wiring conflict ({link.port_name}): a planned port "
                "of the same name has a different direction or type — model "
                "declaration left untouched"
            )
            continue

        already_connected = any(
            isinstance(connection, Mapping)
            and (connection.get("source") or {}).get("component") == source
            and (connection.get("source") or {}).get("port") == link.port_name
            and (connection.get("target") or {}).get("component") == target
            and (connection.get("target") or {}).get("port") == link.port_name
            for connection in connections
        )
        added_connection = False
        if not already_connected:
            connections.append({
                "source": {"component": source, "port": link.port_name},
                "target": {"component": target, "port": link.port_name},
                "item_type": link.port_type,
                "requirements": [],
            })
            added_connection = True

        if source_state or target_state or added_connection:
            notes.append(
                "mandated wiring planned by construction: "
                f"{source}.{link.port_name} -> {target}.{link.port_name} "
                f"({link.port_type}; prompt-mandated safety interconnect)"
            )

    augmented = dict(payload)
    augmented["components"] = components
    augmented["connections"] = connections
    return augmented, notes
