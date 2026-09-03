"""extractor.py

Uses the syside native API to build a BehavioralGraph from SysML v2 text.

Design decisions vs. the old Syside_AST_Parser version:

  - Graph nodes use PartUsage names ("sensorSuite"), not PartDefinition names
    ("SensorSuite"), because connections are written against usage names.
    Multiple instances of one type (sensorUnit2, sensorUnit3) stay distinct.

  - PartNode carries ``def_name`` (the PartDefinition name) for keyword-based
    type classification in scenario generation.

  - Port directions come from syside's FeatureDirectionKind enum.

  - Connection endpoints are resolved via ``chaining_features`` on the
    ``owned_reference_subsetting`` of each ConnectionUsage end feature.

  - Without syside the function returns an empty BehavioralGraph so the rest
    of the pipeline still runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional

from ..utils.suppressed import record_suppressed

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


@dataclass
class PartNode:
    id: str
    def_name: str
    actions: List[str] = field(default_factory=list)
    port_ids: List[str] = field(default_factory=list)
    satisfied_reqs: List[str] = field(default_factory=list)


@dataclass
class PortNode:
    id: str
    part_name: str
    port_name: str
    direction: str


@dataclass
class ActionNode:
    id: str
    owner_part: Optional[str]


@dataclass
class ConnectionEdge:
    source: str
    target: str
    kind: str
    owner: str


@dataclass
class BehavioralGraph:
    parts: Dict[str, PartNode] = field(default_factory=dict)
    ports: Dict[str, PortNode] = field(default_factory=dict)
    actions: Dict[str, ActionNode] = field(default_factory=dict)
    connections: List[ConnectionEdge] = field(default_factory=list)
    # PartDefinition names the model declares passive via a
    # `// PLAN-PASSIVE <PartDef>: <reason>` marker from the generation plan. A
    # passive body exchanges nothing, so scenario generation expects no path in
    # or out. Read from the text so the decision reaches the simulator directly.
    passive_defs: Set[str] = field(default_factory=set)


def _port_dir(d) -> str:
    name = getattr(d, "name", str(d)).lower()
    if "inout" in name:
        return "inout"
    if "out" in name:
        return "out"
    if "in" in name:
        return "in"
    return "none"


def _decode_conn_end(end_feature) -> Optional[str]:
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


def _qualified_name(element) -> str:
    qualified = getattr(element, "qualified_name", None)
    if qualified is not None:
        value = str(qualified)
        if value:
            return value
    return str(getattr(element, "name", "") or "")


def _owning_package_name(element) -> Optional[str]:
    current = element
    for _ in range(24):
        current = getattr(current, "owner", None)
        if current is None:
            return None
        if type(current).__name__ == "Package":
            return str(getattr(current, "name", "") or "") or None
    return None


def extract_behavioral_graph(
    sysml_text: str,
    *,
    root_package: Optional[str] = None,
) -> BehavioralGraph:
    """Parse *sysml_text* with syside and return a BehavioralGraph."""
    bg = BehavioralGraph()
    bg.passive_defs = {
        m.group(1)
        for m in re.finditer(
            r"(?m)^[ \t]*//\s*PLAN-PASSIVE\s+([A-Za-z_]\w*)\s*:", sysml_text or ""
        )
    }

    if not _SYSIDE_OK:
        return bg

    try:
        model, _diags = _syside.try_load_model(sysml_source=sysml_text)
    except Exception:
        return bg

    # A generated file may hold the system package plus auxiliary A/G,
    # verification, or analysis packages, which are not components of the
    # simulated system. Apply the requested scope only when that package is
    # present, so callers passing a display model_name keep their behaviour.
    available_packages = {
        str(getattr(package, "name", "") or "")
        for package in model.elements(_syside.Package)
    }
    scoped_package = (
        str(root_package)
        if root_package and str(root_package) in available_packages
        else None
    )

    def_ports: Dict[str, Dict[str, str]] = {}
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
            # Ports inherited via `:>` are absent from owned_ports but live in
            # inherited_features, and a part bound to `Variant :> Base` needs Base's
            # ports or its connects get pruned as "no port". Filter to directioned
            # PortUsages to exclude standard-library derived ports (direction None).
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
            definition_id = _qualified_name(pd)
            def_ports[definition_id] = ports

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
            def_reqs[definition_id] = reqs
    except Exception as exc:
        record_suppressed("simulation.extractor.part_defs", exc)

    # ── Step 2: PartUsage -> PartNode + PortNode ────────────────────────
    # Parts declared inside an `analysis def` (DSE trade-study alt{i}Design binding
    # parts) are scaffolding, not system assembly: they have no connects and would
    # count as isolated, deflating reachability.
    _analysis_cls = getattr(_syside, "AnalysisCaseDefinition", None)

    def _in_analysis_scope(elem) -> bool:
        # True if elem is scaffolding: inside an `analysis def` (trade-study alt parts)
        # or inside the injected `part def DseDesignAnalysis` closure, whose
        # recommendedDesign part is not a component and does not count for reachability.
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
            if (
                scoped_package is not None
                and _owning_package_name(pu) != scoped_package
            ):
                continue
            if _in_analysis_scope(pu):       # trade-study analysis parts != system assembly
                continue

            defs = list(pu.definitions)
            definition = defs[0] if defs else None
            def_name = (
                str(getattr(definition, "name", "") or usage_name)
                if definition is not None else usage_name
            )
            definition_id = (
                _qualified_name(definition)
                if definition is not None else usage_name
            )

            port_ids: List[str] = []
            for port_name, direction in def_ports.get(definition_id, {}).items():
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
                satisfied_reqs=list(def_reqs.get(definition_id, [])),
            )
    except Exception as exc:
        record_suppressed("simulation.extractor.part_usages", exc)

    try:
        for conn in model.elements(_syside.ConnectionUsage):
            if (
                scoped_package is not None
                and _owning_package_name(conn) != scoped_package
            ):
                continue
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
