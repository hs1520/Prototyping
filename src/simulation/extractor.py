"""
extractor.py

Uses syside native API to build a BehavioralGraph directly from SysML v2 text.

Key design decisions vs. the old Syside_AST_Parser version:

  - Graph nodes use **PartUsage names** (instance names such as "sensorSuite"),
    not PartDefinition names ("SensorSuite").  SysML v2 connections are written
    against usage names, so this gives 1-to-1 correspondence with the model.
    Multiple instances of the same type (sensorUnit2, sensorUnit3) stay distinct.

  - PartNode carries a ``def_name`` field (the PartDefinition name, e.g.
    "SensorSuite") for keyword-based type classification in scenario generation.

  - Port directions come from syside's FeatureDirectionKind enum — no guessing.

  - Connection endpoints are decoded via ``chaining_features`` on the
    ``owned_reference_subsetting`` of each ConnectionUsage end feature.
    This is exact resolution; no string heuristics.

  - If syside is unavailable the function returns an empty BehavioralGraph so
    the rest of the pipeline degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.suppressed import record_suppressed

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


# ---------------------------------------------------------------------------
# Graph node / edge data classes
# ---------------------------------------------------------------------------

@dataclass
class PartNode:
    id: str                             # PartUsage name  (e.g. "sensorSuite")
    def_name: str                       # PartDefinition name (e.g. "SensorSuite")
    actions: List[str] = field(default_factory=list)
    port_ids: List[str] = field(default_factory=list)
    satisfied_reqs: List[str] = field(default_factory=list)


@dataclass
class PortNode:
    id: str                             # "usageName.portName"
    part_name: str                      # PartUsage name
    port_name: str
    direction: str                      # "in" | "out" | "inout" | "none"


@dataclass
class ActionNode:
    id: str
    owner_part: Optional[str]


@dataclass
class ConnectionEdge:
    source: str                         # "usageName.portName"
    target: str                         # "usageName.portName"
    kind: str                           # "connection" | "flow" | "binding"
    owner: str                          # owner name (PartUsage or package)


@dataclass
class BehavioralGraph:
    parts: Dict[str, PartNode] = field(default_factory=dict)
    ports: Dict[str, PortNode] = field(default_factory=dict)
    actions: Dict[str, ActionNode] = field(default_factory=dict)
    connections: List[ConnectionEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _port_dir(d) -> str:
    """Convert syside FeatureDirectionKind to 'in' | 'out' | 'inout' | 'none'."""
    name = getattr(d, "name", str(d)).lower()
    if "inout" in name:
        return "inout"
    if "out" in name:
        return "out"
    if "in" in name:
        return "in"
    return "none"


def _decode_conn_end(end_feature) -> Optional[str]:
    """
    Return 'usageName.portName' for one end of a ConnectionUsage.

    SysML v2 connect endpoints are encoded as:
      ConnectionUsage.owned_end_features[i]
        .owned_reference_subsetting
        .referenced_feature          ← anonymous Feature
        .chaining_features           ← [PartUsage, PortUsage]
    """
    try:
        rs = end_feature.owned_reference_subsetting
        if rs is None:
            return None
        rf = rs.referenced_feature
        cf = list(getattr(rf, "chaining_features", []))
        if cf:
            return ".".join(f.name for f in cf if f.name)
    except Exception as exc:
        record_suppressed("simulation.extractor.conn_end_decode", exc)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_behavioral_graph(sysml_text: str) -> BehavioralGraph:
    """
    Parse *sysml_text* with syside and return a BehavioralGraph.

    Returns an empty BehavioralGraph if syside is unavailable or parsing fails.
    """
    bg = BehavioralGraph()

    if not _SYSIDE_OK:
        return bg

    try:
        model, _diags = _syside.try_load_model(sysml_source=sysml_text)
    except Exception:
        return bg

    # ── Step 1: PartDefinition → port map + satisfy-requirement map ────────────
    # def_name → {port_name: direction_str}
    def_ports: Dict[str, Dict[str, str]] = {}
    # def_name → [req_name, ...]   (for classification via satisfied requirements)
    def_reqs: Dict[str, List[str]] = {}
    sat_cls = getattr(_syside, "SatisfyRequirementUsage", None)
    try:
        for pd in model.elements(_syside.PartDefinition):
            if not pd.name:
                continue
            ports: Dict[str, str] = {}
            for port in pd.owned_ports:
                if port.name:
                    ports[port.name] = _port_dir(port.direction)
            # Ports inherited via `:>` specialisation are absent from owned_ports
            # but live in inherited_features.  A part bound to a variant type
            # (e.g. `Variant :> Base`) must expose Base's ports or its connects
            # get pruned as "no port".  Filter to directioned PortUsages so the
            # standard-library derived ports (direction None) are excluded.
            _port_cls = getattr(_syside, "PortUsage", None)
            if _port_cls is not None:
                try:
                    for feat in pd.inherited_features:
                        if (
                            isinstance(feat, _port_cls)
                            and feat.name
                            and feat.name not in ports
                            and getattr(feat, "direction", None) is not None
                        ):
                            ports[feat.name] = _port_dir(feat.direction)
                except Exception as exc:
                    record_suppressed("simulation.extractor.inherited_ports", exc)
            def_ports[pd.name] = ports

            # Collect satisfy-requirement names for this PartDefinition
            reqs: List[str] = []
            if sat_cls is not None:
                for src_attr in ("owned_requirements", "owned_features"):
                    try:
                        for feat in getattr(pd, src_attr, []) or []:
                            if not isinstance(feat, sat_cls):
                                continue
                            rname = getattr(feat, "name", None)
                            if rname and rname not in reqs:
                                reqs.append(rname)
                    except Exception as exc:
                        record_suppressed("simulation.extractor.def_satisfy", exc)
            def_reqs[pd.name] = reqs
    except Exception as exc:
        record_suppressed("simulation.extractor.part_defs", exc)

    # ── Step 2: PartUsage → PartNode + PortNode ─────────────────────────────
    # Parts declared inside an `analysis def` (e.g. the DSE trade study's alt{i}Design
    # binding parts) are ANALYSIS scaffolding, not the system assembly — they have no
    # connects and would otherwise count as "isolated", deflating reachability.
    _analysis_cls = getattr(_syside, "AnalysisCaseDefinition", None)

    def _in_analysis_scope(elem) -> bool:
        # True if elem is scaffolding: inside an `analysis def` (trade-study alt parts) OR inside
        # the injected DSE analysis closure `part def DseDesignAnalysis` (its recommendedDesign /
        # design-point part is NOT a system component and must not count toward reachability).
        cur = elem
        for _ in range(12):
            o = getattr(cur, "owner", None)
            if o is None:
                return False
            if _analysis_cls is not None and isinstance(o, _analysis_cls):
                return True
            if getattr(o, "name", None) == "DseDesignAnalysis":
                return True
            cur = o
        return False

    try:
        for pu in model.elements(_syside.PartUsage):
            usage_name = pu.name
            if not usage_name:
                continue
            if _in_analysis_scope(pu):       # trade-study analysis parts ≠ system assembly
                continue

            defs = list(pu.definitions)
            def_name = defs[0].name if defs else usage_name

            port_ids: List[str] = []
            for port_name, direction in def_ports.get(def_name, {}).items():
                pid = f"{usage_name}.{port_name}"
                bg.ports[pid] = PortNode(
                    id=pid,
                    part_name=usage_name,
                    port_name=port_name,
                    direction=direction,
                )
                port_ids.append(pid)

            bg.parts[usage_name] = PartNode(
                id=usage_name,
                def_name=def_name,
                port_ids=port_ids,
                satisfied_reqs=list(def_reqs.get(def_name, [])),
            )
    except Exception as exc:
        record_suppressed("simulation.extractor.part_usages", exc)

    # ── Step 3: ConnectionUsage → ConnectionEdge ────────────────────────────
    try:
        for conn in model.elements(_syside.ConnectionUsage):
            ends = list(conn.owned_end_features)
            if len(ends) < 2:
                continue
            src = _decode_conn_end(ends[0])
            tgt = _decode_conn_end(ends[1])
            if not src or not tgt:
                continue
            owner_name = conn.owner.name if conn.owner else "__package__"
            bg.connections.append(ConnectionEdge(
                source=src,
                target=tgt,
                kind="connection",
                owner=owner_name,
            ))
    except Exception as exc:
        record_suppressed("simulation.extractor.connections", exc)

    return bg
