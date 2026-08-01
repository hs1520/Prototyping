"""Shared pure-function helpers for the DSE evaluation pipeline.

Kept separate so both evaluator.py and diagnostics.py can import without
creating a circular dependency.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from ..sysml.model import SysMLModel

try:
    import networkx as nx
    _HAS_NX = True
except ModuleNotFoundError:
    nx = None       # type: ignore
    _HAS_NX = False


#: LEXICAL heuristic: identifies a sensor by the wording of its type name.
#: No parser can answer "is this a sensor" — a sensor typed `ForwardUnit` is
#: invisible here, and that is a property of the check, not of the model.
_SENSOR_USAGE_RE = re.compile(
    r"\bpart\s+(\w+)\s*:\s*\w*"
    r"(?:Sensor|Perception|Detector|Camera|Lidar|IMU|GPS|Radar)\w*\s*;",
    re.IGNORECASE,
)


def _sysml_text(model: SysMLModel) -> str:
    """Return the stored SysML text from model metadata (reliable source)."""
    return (getattr(model, "metadata", None) or {}).get("last_sysml_text", "")


def _build_connection_graph(text: str):
    """
    Build a directed graph of part-instance connections from SysML text.

    Nodes are part instance names; edges represent connect statements.
    Self-loops are skipped.  Returns a networkx DiGraph when available,
    otherwise a pure-Python shim with the same interface.
    """
    from src.simulation.connectivity_fixer import parse_connects

    edge_list: List[Tuple[str, str]] = []
    try:
        statements = parse_connects(text)
    except Exception:
        statements = []
    for stmt in statements:
        if stmt.src_inst != stmt.tgt_inst:
            edge_list.append((stmt.src_inst, stmt.tgt_inst))

    if _HAS_NX:
        g = nx.DiGraph()
        g.add_edges_from(edge_list)
        return g

    class _PureDiGraph:
        def __init__(self, edges: List[Tuple[str, str]]) -> None:
            self._succ: Dict[str, Set[str]] = {}
            self._pred: Dict[str, Set[str]] = {}
            for u, v in edges:
                self._succ.setdefault(u, set()).add(v)
                self._succ.setdefault(v, set())
                self._pred.setdefault(v, set()).add(u)
                self._pred.setdefault(u, set())

        def nodes(self) -> Set[str]:
            return set(self._succ)

        def edges(self) -> List[Tuple[str, str]]:
            return [(u, v) for u, vs in self._succ.items() for v in vs]

        def weakly_connected_components(self) -> List[Set[str]]:
            visited: Set[str] = set()
            components: List[Set[str]] = []
            undirected: Dict[str, Set[str]] = {}
            for u, vs in self._succ.items():
                undirected.setdefault(u, set()).update(vs)
                for v in vs:
                    undirected.setdefault(v, set()).add(u)
            for start in undirected:
                if start in visited:
                    continue
                comp: Set[str] = set()
                stack = [start]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.add(n)
                    stack.extend(undirected.get(n, set()) - visited)
                components.append(comp)
            return components

    return _PureDiGraph(edge_list)


def _build_port_type_map(model: SysMLModel) -> Dict[str, str]:
    """Returns {port_name: type_ref_name} built from all PartDefinition.ports."""
    mapping: Dict[str, str] = {}
    for part in model.part_definitions:
        for port in part.ports:
            if port.type_ref and port.type_ref.name:
                mapping[port.name] = port.type_ref.name
    return mapping


def _satisfied_req_ids(model: SysMLModel) -> set:
    return {
        sr.target.name
        for part in model.part_definitions
        for sr in part.satisfy_relationships
        if sr.target and sr.target.name
    }
