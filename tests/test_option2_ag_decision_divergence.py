"""Fact provenance and divergence — the measure the three pillars were blind to.

A full 3x3 pilot of each emitter-rendered mode produced pattern-conformance reports
differing only in the model digest. Both modes render through `ag_emitter`, so
conformance is guaranteed by the renderer rather than earned by the decisions, and
everything that actually differed was invisible.

The classification is what carries the meaning: a divergence on a fact the
requirement text fixes is a defect, while a divergence on a fact the text does not
contain is a different design decision. Reporting the second as an error would
punish a model for choosing where the requirement is silent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_decision_divergence import (
    DESIGNER_SUPPLIED,
    FACT_PROVENANCE,
    REQUIREMENT_DETERMINED,
    compute_divergence,
    format_divergence,
)
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_extractor import extract_ag_graph

_BASE = (
    "package Src { requirement def REQ_SAFE_005 { doc /* deploy the parachute "
    "within 0.5 s */ } }"
)
_REVIEWED = _BASE + "\n\n" + emit_ag_package(REQ_SAFE_005_CHAIN)


def _divergence(model_text: str) -> dict:
    return compute_divergence(extract_ag_graph(model_text), REQ_SAFE_005_CHAIN)


def test_the_reviewed_decomposition_diverges_from_itself_nowhere():
    """The deterministic emitter renders the reviewed spec verbatim, so every fact
    must match. A non-zero baseline would mean the comparison is reading the wrong
    field — which it was: selected_response sits under the arbitration topology,
    not at the top level, so reading it there reported a divergence for every
    model including this one."""
    out = _divergence(_REVIEWED)
    assert out["requirement_determined"]["diverged"] == 0
    assert out["designer_supplied"]["diverged"] == 0, (
        out["designer_supplied"]["diverged_facts"]
    )


def test_a_different_apportionment_is_a_choice_not_a_defect():
    """0.2 + 0.3 and 0.1 + 0.35 both meet a 0.5 s deadline; the requirement is
    silent on margin, so this must not be counted against the author."""
    rebudgeted = _REVIEWED.replace("0.1 [s]", "0.2 [s]").replace(
        "0.35 [s]", "0.3 [s]"
    )
    out = _divergence(rebudgeted)
    assert "latency_apportionment" in out["designer_supplied"]["diverged_facts"]
    assert out["requirement_determined"]["diverged"] == 0


def test_a_misread_deadline_is_a_defect():
    """The requirement states the deadline, so diverging on it is not a choice."""
    out = _divergence(_REVIEWED.replace("0.5 [s]", "0.9 [s]"))
    assert "deadline_seconds" in out["requirement_determined"]["diverged_facts"]


def test_a_misread_pattern_is_a_defect():
    out = _divergence(
        _REVIEWED.replace(
            "safety_pattern=TRIGGERED_TIMED_FAILSAFE_RESPONSE",
            "safety_pattern=STARTUP_INHIBIT",
        )
    )
    assert "safety_pattern" in out["requirement_determined"]["diverged_facts"]


def test_every_fact_carries_a_reviewable_rationale():
    """The classification is a judgement, so it must be stated rather than buried
    in the comparison code."""
    for fact, (provenance, rationale) in FACT_PROVENANCE.items():
        assert provenance in (REQUIREMENT_DETERMINED, DESIGNER_SUPPLIED), fact
        assert len(rationale) > 30, fact


def test_it_declares_it_is_not_an_accuracy_measure():
    out = _divergence(_REVIEWED)
    assert "NOT accuracy" in out["metric_interpretation"]
    table = format_divergence(out)
    assert "DEFECT" in table and "CHOICE" in table


def test_the_measure_separates_the_two_pilot_modes():
    """The evidence this was built for: identical on every pillar, different here.

    The deterministic pilot renders the reviewed spec, so it diverges nowhere; the
    decided pilot makes its own calls on every designer-supplied fact and none of
    the requirement-determined ones.
    """
    deterministic = Path(
        "examples/output/revised_pilot_20260723_freeze2/seed-0/R2-BBAG/"
        "shared_model_final.sysml"
    )
    decided = Path(
        "examples/output/decided_pilot_20260725/seed-0/R2-BBAG/"
        "shared_model_final.sysml"
    )
    if not (deterministic.exists() and decided.exists()):
        pytest.skip("pilot evidence directories are gitignored")

    left = _divergence(deterministic.read_text())
    right = _divergence(decided.read_text())
    assert left["designer_supplied"]["diverged"] == 0
    assert right["designer_supplied"]["diverged"] > 0
    # the model got every fact the requirement text fixes, in both modes
    assert left["requirement_determined"]["diverged"] == 0
    assert right["requirement_determined"]["diverged"] == 0
