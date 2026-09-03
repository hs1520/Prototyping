"""scenarios.py

Scenario definitions for SysML v2 structural reachability simulation.

A Scenario specifies a named operational sequence that must be structurally
reachable in the port-connection graph:
  - entry_nodes:    starting parts (PartUsage names)
  - target_nodes:   all must be reachable from some entry node
  - required_nodes: every node here must lie on some entry->target path

Scenario generation strategy
─────────────────────────────
``auto_detect_scenarios`` builds scenarios from the BehavioralGraph itself,
using a priority-stacked classification of each PartNode:

  P1. Satisfy relationships  - satisfied_reqs carrying recognisable safety/
      comms/power requirement IDs win.
  P2. Port direction topology - only OUT ports is a source (-> "sensor");
      only IN ports is a sink (-> "actuator"); mixed IN+OUT is a hub
      (-> keywords decide sub-class, else "controller").
  P3. Keyword matching       - bucket keywords on usage + def names, used to
      sub-classify hubs and no-port parts.
  P4. Topology tiebreaker    - unnamed hubs are promoted to "controller".
  P5. "other" fallback       - remaining parts join a hub-fallback pool; the
      highest-degree hub there generates a generic connectivity scenario.

Isolated parts (no connect statements) are included so the validator catches
them as failing scenarios and reports them as design defects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .extractor import BehavioralGraph, PartNode


@dataclass
class Scenario:
    name: str
    description: str
    entry_nodes: List[str]
    target_nodes: List[str]
    required_nodes: List[str] = field(default_factory=list)
    forbidden_nodes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


DRONE_SCENARIOS: List[Scenario] = [
    Scenario(
        name="gps_to_flight_controller",
        description="GPS data must reach FlightController for navigation",
        entry_nodes=["CommunicationSuite"],
        target_nodes=["FlightController"],
        tags=["nominal", "navigation"],
    ),
    Scenario(
        name="sensor_to_obstacle_avoidance",
        description="Sensor data must reach ObstacleAvoidanceSystem",
        entry_nodes=["SensorUnit"],
        target_nodes=["ObstacleAvoidanceSystem"],
        tags=["nominal", "sensing"],
    ),
    Scenario(
        name="emergency_abort_path",
        description="Emergency command must propagate from SafetyMonitor to FlightController",
        entry_nodes=["SafetyMonitor"],
        target_nodes=["FlightController"],
        tags=["safety", "emergency"],
    ),
    Scenario(
        name="power_to_controller",
        description="Power must reach FlightController",
        entry_nodes=["PowerSystem"],
        target_nodes=["FlightController"],
        tags=["nominal", "power"],
    ),
    Scenario(
        name="battery_safety_monitor",
        description="Battery status from PowerSystem must reach SafetyMonitor",
        entry_nodes=["PowerSystem"],
        target_nodes=["SafetyMonitor"],
        tags=["safety", "power"],
    ),
]


_SAFETY_KEYWORDS   = {"safety", "monitor", "emergency", "fault", "watchdog",
                       "failsafe", "supervisor"}
_CONTROL_KEYWORDS  = {"controller", "control", "flight", "manager", "processor",
                       "cpu", "navigation", "autopilot", "planner"}
_SENSOR_KEYWORDS   = {"sensor", "camera", "lidar", "radar", "gps", "imu",
                       "perception", "suite", "voter", "fusion"}
_COMMS_KEYWORDS    = {"comm", "radio", "telemetry", "network", "bus",
                       "ethernet", "link", "antenna"}
_POWER_KEYWORDS    = {"power", "battery", "energy", "supply", "bms",
                       "propulsion", "motor", "esc", "thruster"}
_ACTUATOR_KEYWORDS = {"actuator", "servo", "pump", "payload", "arm",
                       "wheel", "gimbal", "mechanism"}
# Passive mechanical / structural parts. They carry no signal role, so they
# generate no telemetry/control/uplink scenarios: a match is classified
# "structure" and participates only in power-target scenarios.
_STRUCTURE_KEYWORDS = {"airframe", "chassis", "frame", "fuselage", "housing",
                       "enclosure", "structure", "hull"}

_REQ_SAFETY_KW = {"safe", "emerg", "fault", "watchdog"}
_REQ_COMMS_KW  = {"comm", "comms", "telemetry", "link"}
_REQ_POWER_KW  = {"power", "batt", "energy"}


# ---------------------------------------------------------------------------
# Production-oriented port-name lexicons
# ---------------------------------------------------------------------------
# A part's functional role follows from what it produces, so these tokens are
# matched only against out (and inout) port names. An in-port name describes
# the other party's output and does not classify this part: a controller's
# `in sensorData` does not make it a sensor.
#
# Ordered most-distinctive-first; the first category a part produces wins.
_PRODUCE_SAFETY = ("override", "recovery", "emergency", "failsafe",
                   "abort", "chute", "parachute")
_PRODUCE_COMMS  = ("telemetry", "gcs", "uplink", "downlink", "mavlink",
                   "remoteid", "broadcast", "comm", "radio", "antenna")
_PRODUCE_CTRL   = ("cmd", "command", "control", "actuator", "steer", "setpoint")
_PRODUCE_SENSOR = ("sensordata", "sensor", "nav", "gps", "gnss", "imu",
                   "lidar", "radar", "perception", "position", "attitude", "odom")
_PRODUCE_POWER  = ("power", "energy", "battery", "volt", "current",
                   "charge", "thermal")
# In-port command tokens: a pure consumer of commands is an actuator/effector.
_CONSUME_CMD    = ("cmd", "command", "release", "actuate", "drive", "throttle")

# Ordered production rules: (lexicon, role). Safety before comms before
# controller so override/recovery and gcs/telemetry names are not swallowed
# by the generic `cmd` controller token.
_PRODUCTION_RULES = (
    (_PRODUCE_SAFETY, "safety"),
    (_PRODUCE_COMMS,  "comms"),
    (_PRODUCE_CTRL,   "controller"),
    (_PRODUCE_SENSOR, "sensor"),
    (_PRODUCE_POWER,  "power"),
)


def _port_topology(node: PartNode, bg: BehavioralGraph) -> Tuple[bool, bool]:
    part_ports = [bg.ports[pid] for pid in node.port_ids if pid in bg.ports]
    has_out = any(p.direction in ("out", "inout") for p in part_ports)
    has_in  = any(p.direction in ("in",  "inout") for p in part_ports)
    return has_out, has_in


def _part_degrees(bg: BehavioralGraph) -> Dict[str, int]:
    """Compute the number of distinct connection endpoints (degree) per part.

    Used as a centrality proxy: the highest-degree part is the likely primary
    hub / controller.
    """
    degrees: Dict[str, int] = {p: 0 for p in bg.parts}
    for conn in bg.connections:
        sp = conn.source.split(".")[0] if "." in conn.source else conn.source
        tp = conn.target.split(".")[0] if "." in conn.target else conn.target
        if sp in degrees:
            degrees[sp] += 1
        if tp in degrees and tp != sp:
            degrees[tp] += 1
    return degrees


def _classify_node(node: PartNode, bg: BehavioralGraph) -> str:
    """Priority-stacked classification:

    P1. Keyword matching       - intent from usage + def names, checked first so
        "FlightController" is not filed as "sensor" by a port-direction quirk or
        as "safety" by the SAFE requirements a controller satisfies (s0v8).
    P2. Satisfy relationships  - requirement IDs with unambiguous domain
        fragments (safe/comms/power) classify parts whose names carry no signal.
        FUNC/PERF requirements are skipped; any part can satisfy them.
    P3. Port direction topology - structural role for anonymous/abbreviated
        names that carry no keyword signal:
          pure OUT  -> "sensor"
          pure IN   -> "actuator"
          mixed hub -> "controller"
    P4. "other"                - no topology, no keyword match. The caller
        promotes by degree in auto_detect_scenarios.

    Keywords come before topology because port directions can be incomplete (an
    LLM may drop IN ports on a controller) while names rarely misrepresent intent.
    """
    # ── P1: keyword matching (all parts, all categories) ─────────────────────
    # Name identity comes first. Satisfy-based classification ran before it, and
    # a controller carrying a satisfy REQ_SAFE_* link was classified "safety",
    # killing every controller-role scenario with MISSING_TARGET_ROLE. Part names
    # rarely misrepresent intent; requirement allocation crosses roles.
    for name in (node.id, node.def_name):
        low = name.lower()
        if any(k in low for k in _SAFETY_KEYWORDS):    return "safety"
        if any(k in low for k in _CONTROL_KEYWORDS):   return "controller"
        if any(k in low for k in _SENSOR_KEYWORDS):    return "sensor"
        if any(k in low for k in _COMMS_KEYWORDS):     return "comms"
        if any(k in low for k in _POWER_KEYWORDS):     return "power"
        if any(k in low for k in _ACTUATOR_KEYWORDS):  return "actuator"
        # Structure is checked last among keywords: only parts with no active role
        # and a structural name (airframe, chassis, ...) are passive bodies.
        if any(k in low for k in _STRUCTURE_KEYWORDS): return "structure"

    for req_name in node.satisfied_reqs:
        low = req_name.lower()
        if any(k in low for k in _REQ_SAFETY_KW): return "safety"
        if any(k in low for k in _REQ_COMMS_KW):  return "comms"
        if any(k in low for k in _REQ_POWER_KW):  return "power"
        # FUNC/PERF: fall through - any part type can satisfy these

    # ── P3: production-oriented classification (name carries no keyword) ──────
    # A part's role follows from what it produces, so the production lexicons
    # match against out (and inout) port names only. In-port names describe
    # consumed signals (the other party's output) and are excluded.
    part_ports = [bg.ports[pid] for pid in node.port_ids if pid in bg.ports]
    out_names = [
        p.port_name.lower() for p in part_ports
        if p.direction in ("out", "inout") and p.port_name
    ]
    for lexicon, role in _PRODUCTION_RULES:
        if any(tok in pn for pn in out_names for tok in lexicon):
            return role

    in_names = [
        p.port_name.lower() for p in part_ports
        if p.direction in ("in", "inout") and p.port_name
    ]
    if not out_names and any(tok in pn for pn in in_names for tok in _CONSUME_CMD):
        return "actuator"

    # ── P4: conservative direction topology (no semantic signal at all) ──────
    # Only pure sources / pure sinks get a definite role. A part with mixed
    # in+out ports but no production-semantic out-port name becomes "other",
    # since promoting it to controller inflated the scenario set (e.g. a power
    # part with `in cmd` + `out status`). auto_detect_scenarios promotes the
    # highest-degree "other" to controller only when no controller was found.
    has_out, has_in = _port_topology(node, bg)
    if node.port_ids:
        if has_out and not has_in:  return "sensor"
        if has_in  and not has_out: return "actuator"
        return "other"

    return "other"


def auto_detect_scenarios(bg: BehavioralGraph) -> List[Scenario]:
    """Generate scenarios from the PartNodes in *bg*.

    Classification is priority-stacked (satisfy -> topology -> keywords ->
    hub-fallback); "other" parts join a hub-fallback pool whose highest-degree
    member becomes a generic controller candidate. Isolated parts (no connect
    statements) are included so the validator catches them as failing scenarios.
    """
    classified = classify_parts_by_role(bg)

    # A part whose definition is declared passive (`// PLAN-PASSIVE`) joins no
    # scenario, including power-to-structure, because the planner recorded that
    # the body exchanges nothing. Undeclared structural parts keep the
    # conservative treatment below.
    passive_ids = {
        node.id for node in bg.parts.values()
        if node.def_name in getattr(bg, "passive_defs", set())
    }
    if passive_ids:
        classified = {
            role: [pid for pid in ids if pid not in passive_ids]
            for role, ids in classified.items()
        }

    controllers = classified.get("controller", [])
    sensors     = classified.get("sensor", [])
    safety      = classified.get("safety", [])
    comms       = classified.get("comms", [])
    power       = classified.get("power", [])
    actuators   = classified.get("actuator", [])
    structures  = classified.get("structure", [])

    scenarios: List[Scenario] = []
    seen: set = set()

    def _add(name, desc, src, tgt, req=None, tags=None):
        if name in seen:
            return
        seen.add(name)
        scenarios.append(Scenario(
            name=name,
            description=desc,
            entry_nodes=[src],
            target_nodes=[tgt],
            required_nodes=[req] if req else [],
            tags=tags or [],
        ))

    for src in sensors:
        for tgt in controllers:
            _add(f"{src}_to_{tgt}",
                 f"Sensor data path: {src} → {tgt}",
                 src, tgt, tags=["nominal", "sensing"])

    for src in safety:
        for tgt in controllers:
            _add(f"emergency_{src}_to_{tgt}",
                 f"Emergency override: {src} → {tgt}",
                 src, tgt, tags=["safety", "emergency"])

    for src in power:
        for tgt in safety:
            _add(f"power_{src}_to_{tgt}",
                 f"Battery status path: {src} → {tgt}",
                 src, tgt, tags=["safety", "power"])

    for src in power:
        for tgt in controllers:
            _add(f"power_{src}_to_{tgt}",
                 f"Power path: {src} → {tgt}",
                 src, tgt, tags=["nominal", "power"])

    for src in comms:
        for tgt in controllers:
            _add(f"uplink_{src}_to_{tgt}",
                 f"Command uplink: {src} → {tgt}",
                 src, tgt, tags=["nominal", "comms"])

    for src in controllers:
        for tgt in comms:
            _add(f"telemetry_{src}_to_{tgt}",
                 f"Telemetry downlink: {src} → {tgt}",
                 src, tgt, tags=["nominal", "comms"])

    for src in controllers:
        for tgt in actuators:
            _add(f"control_{src}_to_{tgt}",
                 f"Control path: {src} → {tgt}",
                 src, tgt, tags=["nominal", "actuation"])

    # Power -> Structure (a structural body may house powered avionics).
    # Structures participate only here, not as telemetry/control/uplink
    # endpoints, so a passive airframe seeds no spurious signal scenarios.
    for src in power:
        for tgt in structures:
            _add(f"power_{src}_to_{tgt}",
                 f"Power path to structure: {src} → {tgt}",
                 src, tgt, tags=["nominal", "power"])

    # ── Fallback: generic connectivity when all classification fails ──────────
    # Fires only when the cross-products above yielded nothing: every part ended
    # up as "other" and hub-promotion produced no pairings. Degree picks the hub,
    # and the scenarios are labelled "unclassified" so the validator report shows
    # them as degraded fallback paths.
    active_parts = [p for p in bg.parts if p not in passive_ids]
    if not scenarios and len(active_parts) >= 2:
        degrees = _part_degrees(bg)
        hub = max(active_parts, key=lambda p: degrees.get(p, 0))
        for other in active_parts:
            if other == hub:
                continue
            _add(f"connectivity_{other}_to_{hub}",
                 f"Generic connectivity (no controller detected): {other} → {hub}",
                 other, hub, tags=["generic", "unclassified"])

    return scenarios


def classify_parts_by_role(bg: BehavioralGraph) -> Dict[str, List[str]]:
    """Return deterministic semantic-role assignments for one model graph.

    Shared by adaptive internal scenarios and the fixed cross-configuration
    post-hoc suite so both use one classification implementation.
    """
    classified: Dict[str, List[str]] = {}
    for pname, pnode in bg.parts.items():
        role = _classify_node(pnode, bg)
        classified.setdefault(role, []).append(pname)
    for names in classified.values():
        names.sort()

    others = classified.get("other", [])
    if others and not classified.get("controller"):
        degrees = _part_degrees(bg)
        ranked = sorted(others, key=lambda p: (-degrees.get(p, 0), p))
        top_hub = ranked[0]
        if degrees.get(top_hub, 0) > 0:
            classified["controller"] = [top_hub]
            classified["other"] = ranked[1:]
    return classified


def select_scenarios(bg: BehavioralGraph,
                     predefined: Optional[List[Scenario]] = None
                     ) -> List[Scenario]:
    """Return the best scenario set for a given BehavioralGraph."""
    part_names = set(bg.parts.keys())
    seen_names: set = set()
    result: List[Scenario] = []

    if predefined:
        for s in predefined:
            if (any(e in part_names for e in s.entry_nodes) and
                    any(t in part_names for t in s.target_nodes)):
                result.append(s)
                seen_names.add(s.name)

    for s in auto_detect_scenarios(bg):
        if s.name not in seen_names:
            result.append(s)
            seen_names.add(s.name)

    return result
