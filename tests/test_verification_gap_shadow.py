"""The mid-loop verification-gap audit sees planned materialisation.

Planned attributes materialise only at the terminal enforcement, after every
LLM rewrite. Auditing the raw loop text instead flagged REQ_CONS_001 as
unanchored although its 120 m threshold attribute was in the plan, and a
surgical anchor pass was spent - and rejected - on a gap the terminal
materialisation closed. Pinned against the committed pilot artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agents.refinement import _RefinementEngine

_PILOT_RUNS = (
    Path(__file__).resolve().parents[1]
    / "experiments/ablation/results/20260829_120859_pilot/runs"
)


def _pilot_text_without_threshold() -> str:
    text = (_PILOT_RUNS / "FULL_seed0.final.sysml").read_text()
    stripped = "\n".join(
        line for line in text.splitlines() if "maxAltitude" not in line
    )
    assert stripped != text, "fixture must actually remove the anchor"
    return stripped


def _pilot_plan() -> dict:
    report = json.loads((_PILOT_RUNS / "FULL_seed0.report.json").read_text())
    return report["whole_model_generation_plan"]


def _engine(plan_payload) -> _RefinementEngine:
    engine = _RefinementEngine.__new__(_RefinementEngine)
    engine._verification_gap_audit = None
    engine._runtime = SimpleNamespace(
        _active_model_generation_plan=plan_payload,
        last_requirement_input={},
    )
    return engine


def test_planned_anchors_raise_no_gap():
    engine = _engine(_pilot_plan())
    issues = engine._verification_gap_issues(
        _pilot_text_without_threshold(), "AutonomousDrone"
    )
    assert issues == [], (
        "the plan materialises the threshold attribute at terminal commit; "
        f"flagging it mid-loop wastes an anchor pass: {issues}"
    )


def test_gap_reported_without_plan():
    engine = _engine(None)
    issues = engine._verification_gap_issues(
        _pilot_text_without_threshold(), "AutonomousDrone"
    )
    assert any("REQ_CONS_001" in issue for issue in issues), (
        "with no plan committed to closing it, the gap must stay visible"
    )


def test_broken_plan_falls_back_to_text():
    engine = _engine({"nonsense": True})
    text = _pilot_text_without_threshold()
    assert engine._planned_materialization_shadow(text) == text
