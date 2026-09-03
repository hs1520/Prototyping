"""Fact provenance and divergence - the measure the three pillars miss.

A 3x3 pilot of each emitter-rendered mode produced pattern-conformance reports
differing only in the model digest: both render through `ag_emitter`, so
conformance comes from the renderer. The classification carries the meaning - a
divergence on a fact the requirement text fixes is a defect, one on a fact the
text omits is a different design decision.
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


def test_reviewed_spec_diverges_nowhere():
    """The deterministic emitter renders the reviewed spec verbatim, so every fact
    matches.

    A non-zero baseline means the comparison reads the wrong field:
    selected_response sits under the arbitration topology, not at the top level.
    """
    out = _divergence(_REVIEWED)
    assert out["requirement_determined"]["diverged"] == 0
    assert out["designer_supplied"]["diverged"] == 0, (
        out["designer_supplied"]["diverged_facts"]
    )


def test_apportionment_choice_not_defect():
    rebudgeted = _REVIEWED.replace("0.1 [s]", "0.2 [s]").replace(
        "0.35 [s]", "0.3 [s]"
    )
    out = _divergence(rebudgeted)
    assert "latency_apportionment" in out["designer_supplied"]["diverged_facts"]
    assert out["requirement_determined"]["diverged"] == 0


def test_misread_deadline_defect():
    out = _divergence(_REVIEWED.replace("0.5 [s]", "0.9 [s]"))
    assert "deadline_seconds" in out["requirement_determined"]["diverged_facts"]


def test_misread_pattern_defect():
    out = _divergence(
        _REVIEWED.replace(
            "safety_pattern=TRIGGERED_TIMED_FAILSAFE_RESPONSE",
            "safety_pattern=STARTUP_INHIBIT",
        )
    )
    assert "safety_pattern" in out["requirement_determined"]["diverged_facts"]


def test_every_fact_has_rationale():
    for fact, (provenance, rationale) in FACT_PROVENANCE.items():
        assert provenance in (REQUIREMENT_DETERMINED, DESIGNER_SUPPLIED), fact
        assert len(rationale) > 30, fact


def test_not_accuracy_measure():
    out = _divergence(_REVIEWED)
    assert "NOT accuracy" in out["metric_interpretation"]
    table = format_divergence(out)
    assert "DEFECT" in table and "CHOICE" in table


def test_measure_separates_pilot_modes():
    """Identical on every pillar, different here.

    The deterministic pilot renders the reviewed spec, so it diverges nowhere; the
    decided pilot diverges on designer-supplied facts only.
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
    assert left["requirement_determined"]["diverged"] == 0
    assert right["requirement_determined"]["diverged"] == 0
