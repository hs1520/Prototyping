"""Semantic-surgery gates: refinement fixes the design, not the spec.

Invariant: the surgical merge does not add or drop requirement definitions and
does not shed satisfy links, the loss mode the connect gate also guards.
"""
from __future__ import annotations

from src.simulation.surgical_refiner import _gates_ok

_BASE = """package D {
    requirement def REQ_A {
        doc /* First requirement. */
    }
    requirement def REQ_B {
        doc /* Second requirement. */
    }
    part def Alpha {
        attribute x : Real = 1.0;
        satisfy requirement REQ_A;
        satisfy requirement REQ_B;
    }
}"""


def test_identical_model_passes_gates():
    ok, why = _gates_ok(_BASE, _BASE)
    assert ok, why


def test_dropping_requirement_rejected():
    merged = _BASE.replace(
        "    requirement def REQ_B {\n        doc /* Second requirement. */\n    }\n", ""
    ).replace("        satisfy requirement REQ_B;\n", "")
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "requirement def" in why


def test_adding_requirement_rejected():
    merged = _BASE.replace(
        "    part def Alpha {",
        "    requirement def REQ_NEW {\n        doc /* Invented by refinement. */\n    }\n"
        "    part def Alpha {",
    )
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "requirement def" in why


def test_shedding_satisfy_rejected():
    merged = _BASE.replace("        satisfy requirement REQ_B;\n", "")
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "satisfy" in why


def test_adding_anchors_allowed():
    merged = _BASE.replace(
        "        satisfy requirement REQ_B;",
        "        satisfy requirement REQ_B;\n"
        "        attribute powerOnLocked : Boolean = true;",
    )
    ok, why = _gates_ok(_BASE, merged)
    assert ok, why


def test_rewording_source_rejected():
    merged = _BASE.replace("First requirement.", "Weakened requirement.")
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "source text" in why


def test_replacing_satisfy_rejected():
    merged = _BASE.replace(
        "satisfy requirement REQ_B;", "satisfy requirement REQ_A;"
    )
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "satisfy" in why


def test_replacing_connect_rejected():
    base = """package D {
        port def SignalPort;
        part def Source { out port signalOut : SignalPort; }
        part def Sink { in port signalIn : SignalPort; }
        part sourceA : Source;
        part sourceB : Source;
        part sink : Sink;
        connect sourceA.signalOut to sink.signalIn;
    }"""
    merged = base.replace("sourceA.signalOut", "sourceB.signalOut")
    ok, why = _gates_ok(base, merged)
    assert not ok and "connect" in why
