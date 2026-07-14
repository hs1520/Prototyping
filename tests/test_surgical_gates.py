"""Semantic-surgery gates: refinement fixes the DESIGN, never the SPEC.

The surgical merge must not add or drop requirement definitions, and must not
shed satisfy links — the same silent-loss failure mode the connect gate guards.
"""
from __future__ import annotations

from src.agents.surgical_refiner import _gates_ok

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


def test_dropping_a_requirement_def_is_rejected():
    merged = _BASE.replace(
        "    requirement def REQ_B {\n        doc /* Second requirement. */\n    }\n", ""
    ).replace("        satisfy requirement REQ_B;\n", "")
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "requirement def" in why


def test_adding_a_requirement_def_is_rejected():
    merged = _BASE.replace(
        "    part def Alpha {",
        "    requirement def REQ_NEW {\n        doc /* Invented by refinement. */\n    }\n"
        "    part def Alpha {",
    )
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "requirement def" in why


def test_shedding_a_satisfy_link_is_rejected():
    merged = _BASE.replace("        satisfy requirement REQ_B;\n", "")
    ok, why = _gates_ok(_BASE, merged)
    assert not ok and "satisfy" in why


def test_adding_anchors_and_satisfy_links_is_allowed():
    # The verification-anchor pass ADDS attributes/guards/satisfies — must pass.
    merged = _BASE.replace(
        "        satisfy requirement REQ_B;",
        "        satisfy requirement REQ_B;\n"
        "        attribute powerOnLocked : Boolean = true;",
    )
    ok, why = _gates_ok(_BASE, merged)
    assert ok, why
