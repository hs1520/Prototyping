"""exec_graph.py

Builds a networkx DiGraph from a BehavioralGraph.

Node attributes:
  - type:      "part" | "port"
  - direction: (ports only) "in" | "out" | "inout" | "none"
  - owner:     (ports) parent part usage name
  - def_name:  (parts) PartDefinition name for type-based classification

Edge kinds:
  - egress       Part -> OutPort  (signal leaves the part)
  - ingress      InPort -> Part   (signal enters the part)
  - has_port     Part ↔ Port for direction="none" (unknown - bidirectional)
  - connection   Port -> Port     (declared connect statement)

Signal flow topology:
  SourcePart ──egress──► OutPort ──connection──► InPort ──ingress──► TargetPart

Direction-based edges replace the old bidirectional "has_port / port_owned_by"
pair, under which any two parts sharing a connected port looked reachable both
ways. Ports with direction "none" get both edges as a fallback.
"""

from __future__ import annotations

from typing import List, Optional, Set

try:
    import networkx as nx
    _HAS_NX = True
except ModuleNotFoundError:
    _HAS_NX = False

from .extractor import BehavioralGraph


def build_exec_graph(bg: BehavioralGraph) -> "nx.DiGraph":
    """Convert BehavioralGraph -> networkx DiGraph with direction-aware edges."""
    if not _HAS_NX:
        raise ImportError(
            "networkx is required for simulation. Install with: pip install networkx"
        )

    G = nx.DiGraph()

    for pid, part in bg.parts.items():
        G.add_node(pid, type="part", def_name=part.def_name)

    for port_id, port in bg.ports.items():
        G.add_node(port_id, type="port", direction=port.direction, owner=port.part_name)

        d = port.direction
        if d == "out":
            G.add_edge(port.part_name, port_id, kind="egress")
        elif d == "in":
            G.add_edge(port_id, port.part_name, kind="ingress")
        elif d == "inout":
            G.add_edge(port.part_name, port_id, kind="egress")
            G.add_edge(port_id, port.part_name, kind="ingress")
        else:
            # Direction not declared - conservative: allow both
            G.add_edge(port.part_name, port_id, kind="has_port")
            G.add_edge(port_id, port.part_name, kind="has_port")

    for conn in bg.connections:
        src = conn.source
        tgt = conn.target

        if src not in G:
            G.add_node(src, type="port", direction="none",
                       owner=src.split(".")[0])
        if tgt not in G:
            G.add_node(tgt, type="port", direction="none",
                       owner=tgt.split(".")[0])

        G.add_edge(src, tgt, kind=conn.kind, owner=conn.owner)

    return G


def reachable_from(G: "nx.DiGraph", source: str) -> Set[str]:
    """Return all nodes reachable from *source* via directed edges."""
    if not _HAS_NX:
        return set()
    if source not in G:
        return set()
    return nx.descendants(G, source)


def shortest_path(G: "nx.DiGraph", source: str,
                  target: str) -> Optional[List[str]]:
    """Return shortest directed path or None."""
    if not _HAS_NX:
        return None
    try:
        return nx.shortest_path(G, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def all_simple_paths(G: "nx.DiGraph", source: str, target: str,
                     cutoff: int = 12) -> List[List[str]]:
    """Return all simple directed paths up to *cutoff* length."""
    if not _HAS_NX:
        return []
    try:
        return list(nx.all_simple_paths(G, source, target, cutoff=cutoff))
    except (nx.NodeNotFound, nx.NetworkXError):
        return []
