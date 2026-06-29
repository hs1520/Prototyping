"""Deterministic port-direction fixer: widen direction-blocking ports so existing connects are
traversable ('connected but signal direction wrong' → fixed without LLM)."""
from __future__ import annotations

import networkx as nx

from src.simulation import extractor as ex
from src.simulation.direction_fixer import fix_signal_directions
from src.simulation.exec_graph import build_exec_graph

_BAD = """package P {
    port def DataPort;
    part def Motor { out port status : DataPort; }
    part def Frame { out port status : DataPort; }
    part def Drone { part motor : Motor; part airframe : Frame; connect motor.status to airframe.status; }
}"""


def _reaches(text, a, b):
    G = build_exec_graph(ex.extract_behavioral_graph(text))
    return a in G and b in G and nx.has_path(G, a, b)


def test_widens_blocking_target_port():
    # Frame.status is OUT → connect target can't ingress → motor unreachable to airframe
    assert not _reaches(_BAD, "motor", "airframe")
    out, n, names = fix_signal_directions(_BAD)
    assert n == 1 and names == ["Frame.status"]
    assert "inout port status" in out
    assert _reaches(out, "motor", "airframe")          # now traversable


def test_widens_blocking_source_port():
    bad = _BAD.replace("part def Motor { out port status",
                       "part def Motor { in port status")   # src IN → can't egress
    out, n, names = fix_signal_directions(bad)
    assert "Motor.status" in names


def test_noop_when_directions_already_valid():
    good = """package P {
        port def DataPort;
        part def A { out port s : DataPort; }
        part def B { in port s : DataPort; }
        part def Sys { part a : A; part b : B; connect a.s to b.s; }
    }"""
    out, n, names = fix_signal_directions(good)
    assert n == 0 and out == good                       # valid → untouched (non-breaking)


def test_missing_connect_adds_feedback_path():
    from src.simulation.direction_fixer import fix_missing_connects
    # payload→flightController unreachable: only the command connect exists (fc→payload); the
    # feedback connect (payload.payloadStatus → fc.payloadStatus) is missing but the ports match.
    model = """package P {
        port def DataPort;
        part def FC { out port payloadCmd : DataPort; in port payloadStatus : DataPort; }
        part def Pay { in port payloadCmd : DataPort; out port payloadStatus : DataPort; }
        part def Sys {
            part fc : FC; part pay : Pay;
            connect fc.payloadCmd to pay.payloadCmd;
        }
    }"""
    out, n, lines = fix_missing_connects(model, [{"src": "pay", "tgts": ["fc"]}])
    assert n == 1
    assert "connect pay.payloadStatus to fc.payloadStatus;" in out
    # now pay→fc reachable
    import networkx as nx
    from src.simulation import extractor as ex
    from src.simulation.exec_graph import build_exec_graph
    G = build_exec_graph(ex.extract_behavioral_graph(out))
    assert nx.has_path(G, "pay", "fc")


def test_missing_connect_noop_when_no_matching_ports():
    from src.simulation.direction_fixer import fix_missing_connects
    model = """package P {
        port def DataPort;
        part def A { out port x : DataPort; }
        part def B { in port y : DataPort; }
        part def Sys { part a : A; part b : B; }
    }"""
    out, n, _ = fix_missing_connects(model, [{"src": "a", "tgts": ["b"]}])
    assert n == 0 and out == model            # no same-name match → nothing fabricated
