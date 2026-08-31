"""Layer 4: sent-command traceability grounded in the plan's route, pinned on
run3 — whose parachute response sends RecoveryCmdData through recoveryCmd,
the exact terminal leg of its own SAFE_005 causal path, and was rejected by
the CHUTE-substring check."""
from __future__ import annotations

import json
from pathlib import Path

from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

_BUNDLE = (
    Path(__file__).parent.parent
    / "examples" / "output" / "run3_authoritative_20260831"
)


def _linker(with_plan: bool) -> RequirementLinker:
    text = (_BUNDLE / "final_model.sysml").read_text()
    plan = None
    if with_plan:
        plan = json.loads((_BUNDLE / "run_report.json").read_text())[
            "whole_model_generation_plan"
        ]
    model = build_lite_model(text, model_name="AutonomousDrone")
    return RequirementLinker(model, plan_payload=plan)


def test_route_matched_response_is_not_a_traceability_mismatch():
    bundle = _linker(with_plan=True).compile_evidence()
    mismatched = {m.get("req_id") for m in bundle.traceability_mismatches}
    assert "REQ_SAFE_005" not in mismatched
    l2 = {s.req_id for s in bundle.test_specs if s.tier == "L2"}
    assert "REQ_SAFE_005" in l2


def test_without_a_plan_the_substring_check_still_governs():
    """No plan → no route identity → the old acceptance surface, unchanged.
    Documents that the widening is plan-fed, not a blanket leniency."""
    bundle = _linker(with_plan=False).compile_evidence()
    mismatched = {m.get("req_id") for m in bundle.traceability_mismatches}
    assert "REQ_SAFE_005" in mismatched


def test_route_widening_never_removes_a_real_mismatch():
    """SAFE_004's SENSOR-family mismatch is a different finding and must
    survive the route check untouched."""
    bundle = _linker(with_plan=True).compile_evidence()
    mismatched = {m.get("req_id") for m in bundle.traceability_mismatches}
    assert "REQ_SAFE_004" in mismatched
