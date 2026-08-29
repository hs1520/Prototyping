"""The mid-loop verification-gap audit must see planned materialisation.

Planned attributes materialise only at the terminal enforcement (deliberately
after every LLM rewrite).  The ablation pilot showed the cost of auditing the
raw loop text instead: REQ_CONS_001's 120 m threshold attribute was part of
the plan all along, yet the audit flagged the requirement as unanchored and a
surgical anchor pass was spent — and rejected — on a gap the terminal
materialisation closed moments later.  These tests pin the shadow audit to
the committed pilot artifacts that exposed it.
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


def test_plan_committed_anchors_do_not_raise_gap_issues():
    engine = _engine(_pilot_plan())
    issues = engine._verification_gap_issues(
        _pilot_text_without_threshold(), "AutonomousDrone"
    )
    assert issues == [], (
        "the plan materialises the threshold attribute at terminal commit; "
        f"flagging it mid-loop wastes an anchor pass: {issues}"
    )


def test_without_a_plan_the_gap_is_still_reported():
    engine = _engine(None)
    issues = engine._verification_gap_issues(
        _pilot_text_without_threshold(), "AutonomousDrone"
    )
    assert any("REQ_CONS_001" in issue for issue in issues), (
        "with no plan committed to closing it, the gap must stay visible"
    )


def test_shadow_falls_back_to_the_raw_text_on_a_broken_plan():
    engine = _engine({"nonsense": True})
    text = _pilot_text_without_threshold()
    assert engine._planned_materialization_shadow(text) == text
