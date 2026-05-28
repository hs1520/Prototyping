"""
scenarios.py

Scenario definitions for SysML v2 structural reachability simulation.

A Scenario specifies a named operational sequence that must be structurally
reachable in the port-connection graph:
  - entry_nodes:    starting parts (PartUsage names)
  - target_nodes:   all must be reachable from some entry node
  - required_nodes: every node here must lie on some entry→target path

Scenario generation strategy
─────────────────────────────
``auto_detect_scenarios`` builds scenarios from the BehavioralGraph itself,
using a **priority-stacked** classification of each PartNode:

  P1. Satisfy relationships  — if satisfied_reqs contains recognisable safety/
      comms/power requirement IDs, that category wins (strongest signal).
  P2. Port direction topology — a part with only OUT ports is structurally a
      source (→ "sensor"); only IN ports is a sink (→ "actuator"); mixed IN+OUT
      is a hub (→ keywords decide sub-class, else "controller").
  P3. Keyword matching       — existing bucket keywords on usage + def names,
      used to sub-classify hubs and no-port parts.
  P4. Topology tiebreaker    — unnamed hubs (both in and out ports, no keyword
      match) are promoted to "controller" so they always appear in scenarios.
  P5. "other" fallback       — remaining unclassified parts are added to a
      hub-fallback pool; the highest-degree hub in this pool generates a
      generic connectivity scenario so no part is silently dropped.

Isolated parts (no connect statements) are intentionally included so
the validator catches them as failing scenarios and reports them as
design defects.
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


# ---------------------------------------------------------------------------
# Drone reference scenarios (kept for backward compatibility / direct use)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Part classification keywords
# ---------------------------------------------------------------------------

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

# Requirement-name prefixes that map to a class unambiguously.
# (Only fragments that reliably indicate the *owning* part's role.)
_REQ_SAFETY_KW = {"safe", "emerg", "fault", "watchdog"}
_REQ_COMMS_KW  = {"comm", "comms", "telemetry", "link"}
_REQ_POWER_KW  = {"power", "batt", "energy"}


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _port_topology(node: PartNode, bg: BehavioralGraph) -> Tuple[bool, bool]:
    """Return (has_out, has_in) for *node*'s ports in *bg*."""
    part_ports = [bg.ports[pid] for pid in node.port_ids if pid in bg.ports]
    has_out = any(p.direction in ("out", "inout") for p in part_ports)
    has_in  = any(p.direction in ("in",  "inout") for p in part_ports)
    return has_out, has_in


def _part_degrees(bg: BehavioralGraph) -> Dict[str, int]:
    """
    Compute the number of distinct connection endpoints (degree) for each part.
    Used as a proxy for graph centrality — the highest-degree part is the
    most likely primary hub / controller.
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


# ---------------------------------------------------------------------------
# Priority-stacked node classification
# ---------------------------------------------------------------------------

def _classify_node(node: PartNode, bg: BehavioralGraph) -> str:
    """
    Priority-stacked classification:

    P1. Satisfy relationships  — requirement IDs with unambiguous domain
        fragments (safe/comms/power) map directly to a class.  FUNC/PERF
        requirements are skipped — any part can satisfy them.
    P2. Keyword matching       — semantic intent expressed in usage + def names,
        checked unconditionally BEFORE topology.  Ensures a well-named part
        (e.g. "FlightController") is never mis-classified as "sensor" just
        because syside missed some of its input ports.
    P3. Port direction topology — structural role for anonymous/abbreviated
        names that carry no keyword signal:
          pure OUT  → "sensor"
          pure IN   → "actuator"
          mixed hub → "controller"
    P4. "other"                — no topology, no keyword match.  Caller handles
        via degree-based promotion in auto_detect_scenarios.

    Rationale for keywords-before-topology:
      Port directions can be incomplete (LLM may drop IN ports on a controller);
      part names almost never misrepresent intent.  Topology is a strong signal
      only when the name gives no information (abbreviated or random).
    """
    # ── P1: satisfy relationships ────────────────────────────────────────────
    for req_name in node.satisfied_reqs:
        low = req_name.lower()
        if any(k in low for k in _REQ_SAFETY_KW): return "safety"
        if any(k in low for k in _REQ_COMMS_KW):  return "comms"
        if any(k in low for k in _REQ_POWER_KW):  return "power"
        # FUNC/PERF: fall through — any part type can satisfy these

    # ── P2: keyword matching (all parts, all categories) ─────────────────────
    for name in (node.id, node.def_name):
        low = name.lower()
        if any(k in low for k in _SAFETY_KEYWORDS):   return "safety"
        if any(k in low for k in _CONTROL_KEYWORDS):  return "controller"
        if any(k in low for k in _SENSOR_KEYWORDS):   return "sensor"
        if any(k in low for k in _COMMS_KEYWORDS):    return "comms"
        if any(k in low for k in _POWER_KEYWORDS):    return "power"
        if any(k in low for k in _ACTUATOR_KEYWORDS): return "actuator"

    # ── P3: port direction topology (anonymous parts only) ───────────────────
    has_out, has_in = _port_topology(node, bg)
    if node.port_ids:
        if has_out and not has_in: return "sensor"      # pure source
        if has_in  and not has_out: return "actuator"   # pure sink
        if has_out and has_in:      return "controller" # unnamed hub

    return "other"


# ---------------------------------------------------------------------------
# Model-driven scenario generation
# ---------------------------------------------------------------------------

def auto_detect_scenarios(bg: BehavioralGraph) -> List[Scenario]:
    """
    Generate plausible scenarios from the PartNodes in *bg*.

    Uses priority-stacked classification (satisfy → topology → keywords →
    hub-fallback).  Parts that end up as "other" are added to a hub-fallback
    pool; the highest-degree "other" part (most connections) is promoted to
    a generic controller candidate so no part is silently ignored.

    Isolated parts (no connect statements) are intentionally included so
    that the validator catches them as failing scenarios.
    """
    # ── Classify all parts ───────────────────────────────────────────────────
    classified: Dict[str, List[str]] = {}
    for pname, pnode in bg.parts.items():
        cls = _classify_node(pnode, bg)
        classified.setdefault(cls, []).append(pname)

    controllers = classified.get("controller", [])
    sensors     = classified.get("sensor", [])
    safety      = classified.get("safety", [])
    comms       = classified.get("comms", [])
    power       = classified.get("power", [])
    actuators   = classified.get("actuator", [])
    others      = classified.get("other", [])

    # ── Hub-fallback: promote highest-degree "other" part to controller ───────
    # Only fires when there are NO recognised controllers at all — prevents
    # inflating an already-adequate controller list with mystery parts.
    # "other" means: no ports (or direction=none) AND no keyword match.
    # Among these, the most-connected part is the best controller proxy.
    if others and not controllers:
        degrees = _part_degrees(bg)
        others_by_degree = sorted(others, key=lambda p: degrees.get(p, 0), reverse=True)
        top_hub = others_by_degree[0]
        if degrees.get(top_hub, 0) > 0:
            # Only promote if it actually has connections (real hub behaviour).
            # Truly isolated parts (degree 0) are not valid controller candidates.
            controllers = [top_hub]
            others = others_by_degree[1:]

    # ── Scenario cross-products ──────────────────────────────────────────────
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

    # Sensor → Controller
    for src in sensors:
        for tgt in controllers:
            _add(f"{src}_to_{tgt}",
                 f"Sensor data path: {src} → {tgt}",
                 src, tgt, tags=["nominal", "sensing"])

    # Safety → Controller (critical path)
    for src in safety:
        for tgt in controllers:
            _add(f"emergency_{src}_to_{tgt}",
                 f"Emergency override: {src} → {tgt}",
                 src, tgt, tags=["safety", "emergency"])

    # Power → Safety (battery monitoring)
    for src in power:
        for tgt in safety:
            _add(f"power_{src}_to_{tgt}",
                 f"Battery status path: {src} → {tgt}",
                 src, tgt, tags=["safety", "power"])

    # Power → Controller
    for src in power:
        for tgt in controllers:
            _add(f"power_{src}_to_{tgt}",
                 f"Power path: {src} → {tgt}",
                 src, tgt, tags=["nominal", "power"])

    # Comms → Controller (uplink)
    for src in comms:
        for tgt in controllers:
            _add(f"uplink_{src}_to_{tgt}",
                 f"Command uplink: {src} → {tgt}",
                 src, tgt, tags=["nominal", "comms"])

    # Controller → Comms (telemetry downlink)
    for src in controllers:
        for tgt in comms:
            _add(f"telemetry_{src}_to_{tgt}",
                 f"Telemetry downlink: {src} → {tgt}",
                 src, tgt, tags=["nominal", "comms"])

    # Controller → Actuator
    for src in controllers:
        for tgt in actuators:
            _add(f"control_{src}_to_{tgt}",
                 f"Control path: {src} → {tgt}",
                 src, tgt, tags=["nominal", "actuation"])

    # ── Fallback: generic connectivity when all classification fails ──────────
    # This fires only when the cross-products above yielded nothing — meaning
    # every part ended up as "other" and the hub-promotion above either didn't
    # fire or produced no pairings.  Using degree to pick the hub is better than
    # random dict ordering, but these scenarios are labelled "unclassified" so
    # the validator report makes clear they are degraded fallback paths, not
    # intentional design scenarios.
    if not scenarios and len(bg.parts) >= 2:
        degrees = _part_degrees(bg)
        hub = max(bg.parts, key=lambda p: degrees.get(p, 0))
        for other in bg.parts:
            if other == hub:
                continue
            _add(f"connectivity_{other}_to_{hub}",
                 f"Generic connectivity (no controller detected): {other} → {hub}",
                 other, hub, tags=["generic", "unclassified"])

    return scenarios


def select_scenarios(bg: BehavioralGraph,
                     predefined: Optional[List[Scenario]] = None
                     ) -> List[Scenario]:
    """
    Return the best scenario set for a given BehavioralGraph.

    Strategy:
      1. Filter *predefined* scenarios whose entry/target nodes exist in *bg*.
         (Useful when predefined scenarios use exact PartUsage names.)
      2. Always supplement with auto_detect_scenarios — it uses priority-stacked
         classification so it adapts to any generated model.
      3. Deduplicate by name (predefined take priority).
    """
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
