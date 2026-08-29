"""Semantic fidelity follows the plan's evidence routing, never a fixed template.

Measured failure this pins (authoritative run 219eb9bb): three obligations
failed "no assert constraint expresses the frozen subject bound" although
their evidence existed — the MTOW and endurance asserts live in the binding's
declared target component (FlightController) while other parts carry the
satisfy links, and the range assert was removed by the pipeline's own
capability normaliser (a mission-end bound is not a runtime invariant) with
only a prose waiver left behind.  The checker now also scans the binding's
declared owner and recognises the normaliser's delegation marker; every
assertion criterion stays as strict, and delegation without a marker is
still a failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "examples"))

from drone_system_v2 import DRONE_REQUIREMENTS  # noqa: E402

from src.prototyping.requirement_semantics import (  # noqa: E402
    SemanticBindingPlan,
    compile_requirement_semantic_obligations,
    validate_requirement_semantic_obligations,
)
from src.sysml.text_normalization import fix_capability_semantics  # noqa: E402


def _fixture(run_id: str):
    run = _REPO / "examples/output/runs" / run_id
    if not run.exists():
        run = (
            _REPO / "experiments/ablation/results" / run_id
        )
    report = json.loads((run / "canonical_run.json").read_text()) \
        if (run / "canonical_run.json").exists() else None
    if report is not None:
        plan = report["pipeline_report"]["whole_model_generation_plan"]
        text = (run / "final_model.sysml").read_text()
    else:
        report = json.loads((run / "runs/FULL_seed0.report.json").read_text())
        plan = report["whole_model_generation_plan"]
        text = (run / "runs/FULL_seed0.final.sysml").read_text()
    bindings = [
        SemanticBindingPlan.from_dict(item)
        for item in plan["semantic_bindings"]
    ]
    return text, bindings


def _validate(text, bindings):
    return validate_requirement_semantic_obligations(
        text,
        compile_requirement_semantic_obligations(DRONE_REQUIREMENTS),
        model_name="AutonomousDrone",
        bindings=bindings,
    )


def test_archived_219eb9bb_passes_via_binding_owner_extension():
    # All three archived failures shared one root: the asserts live in the
    # binding's declared owner (FlightController) while other parts carry the
    # satisfy links, so the owner scan never reached them.
    text, bindings = _fixture("219eb9bb-4acd-4d20-b858-d1b1ae46d890")
    report = _validate(text, bindings)

    by_req = {item["requirement_id"]: item for item in report["results"]}
    for req in ("REQ_CONS_003", "REQ_PERF_002", "REQ_PERF_006"):
        assert by_req[req]["status"] == "PASS", req
        assert by_req[req]["satisfying_owner"] == "FlightController", req
    assert report["passed"] == 9
    assert report["delegated"] == 0
    assert report["status"] == "PASS"


def _without_range_assert(text: str) -> str:
    stripped = re.sub(
        r"[ \t]*assert\s+constraint\s+operationalRangeConstraint\s*"
        r"\{[^{}]*\}\s*\n",
        "",
        text,
    )
    assert stripped != text
    return stripped


def test_a_normaliser_waived_constraint_is_delegated_not_failed():
    # Remove the materialised range assert but keep the normaliser's waiver
    # pair (PLAN-CONSTRAINT marker + mission-end line) — the legacy marker
    # format must route the obligation to the forward-flight tier.
    text, bindings = _fixture("219eb9bb-4acd-4d20-b858-d1b1ae46d890")
    report = _validate(_without_range_assert(text), bindings)
    by_req = {item["requirement_id"]: item for item in report["results"]}
    assert by_req["REQ_PERF_006"]["status"] == "DELEGATED"
    assert by_req["REQ_PERF_006"]["delegated_to"] == "FORWARD_FLIGHT_FIDELITY"
    assert report["passed"] == 8
    assert report["delegated"] == 1
    assert report["status"] == "PASS"


def test_pilot3_model_still_passes_everything_inline():
    text, bindings = _fixture("20260829_153525_pilot3")
    report = _validate(text, bindings)
    assert report["status"] == "PASS"
    assert report["passed"] == 9
    assert report["delegated"] == 0


def test_delegation_requires_the_marker_not_just_a_missing_assert():
    text, bindings = _fixture("219eb9bb-4acd-4d20-b858-d1b1ae46d890")
    stripped = "\n".join(
        line for line in _without_range_assert(text).splitlines()
        if "mission-end" not in line
    )
    report = _validate(stripped, bindings)
    by_req = {item["requirement_id"]: item for item in report["results"]}
    assert by_req["REQ_PERF_006"]["status"] == "FAIL"
    assert report["status"] == "FAIL"


def test_binding_owner_extension_never_relaxes_the_threshold_check():
    text, bindings = _fixture("219eb9bb-4acd-4d20-b858-d1b1ae46d890")
    weakened = text.replace(
        "attribute maxTakeOffMassPayloadBattery : MassValue = 8 [kg];",
        "attribute maxTakeOffMassPayloadBattery : MassValue = 9 [kg];",
    )
    assert weakened != text
    report = _validate(weakened, bindings)
    by_req = {item["requirement_id"]: item for item in report["results"]}
    assert by_req["REQ_CONS_003"]["status"] == "FAIL"


def test_capability_normaliser_emits_the_structured_delegation_marker():
    model = """package P {
    part def Nav {
        attribute maxOperationalRange : LengthValue = 5 [km];
        assert constraint operationalRangeConstraint {
            currentOperationalRange >= minOperationalRange
        }
    }
}"""
    fixed, fixes = fix_capability_semantics(
        model, has_range_floor=True, has_range_ceiling=False
    )
    assert fixes >= 2   # the rename and the dropped invariant
    assert (
        "// DELEGATED-CONSTRAINT operationalRangeConstraint "
        "tier=FORWARD_FLIGHT_FIDELITY" in fixed
    )
    assert "mission-end" in fixed
    assert "assert constraint operationalRangeConstraint" not in fixed
