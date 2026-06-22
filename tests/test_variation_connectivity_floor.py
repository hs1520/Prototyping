"""Connectivity-regression guard for the variation-DSE refinement path.

The resolved variation model arrives at Phase 4-5 fully wired (reachability 1.0).
A full LLM rewrite tends to drop `connect` statements on the converted components,
isolating them. `_iterative_refinement(..., connectivity_floor=True)` must reject any
candidate that sheds connects relative to the model it was refined from.
"""
from __future__ import annotations

import inspect

from src.agents.orchestrator import Orchestrator

_WIRED = """package Drone {
    part def A { out port o; }
    part def B { in port i; }
    part def Sys {
        part a : A;
        part b : B;
        connect a.o to b.i;
        connect b.i to a.o;
    }
}"""


def test_count_connects_counts_connect_statements():
    assert Orchestrator._count_connects(_WIRED) == 2
    assert Orchestrator._count_connects("package P {}") == 0


def test_connectivity_floor_defaults_off():
    # opt-in: existing (non-variation) paths keep their behaviour unchanged
    sig = inspect.signature(Orchestrator._iterative_refinement)
    assert sig.parameters["connectivity_floor"].default is False


def test_guard_logic_rejects_dropped_connects():
    # the decision the guard makes: a candidate with fewer connects is a regression
    cur = Orchestrator._count_connects(_WIRED)
    dropped = _WIRED.replace("        connect b.i to a.o;\n", "")
    assert Orchestrator._count_connects(dropped) < cur   # would be rejected
    assert Orchestrator._count_connects(_WIRED) >= cur   # preserved → allowed
