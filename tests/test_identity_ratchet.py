"""Identity ratchets: a harness spelling does not become load-bearing.

Pinned on run3's committed evidence bundle, two directions:

- Anti-coupling: renamed identities produce identical verification structure.
  PLAN-tier consumers survive alien renames; the SEMANTIC tier survives renames
  that keep term coverage and reports an alien rename as NOT MEASURED.
- Anti-laundering: lenient identity resolution still fails real defects: a
  guard-stripped model reads "resolved and unguarded", and a response sent off
  its plan route still mismatches.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from gazebo_poc.model_mission import ModelDrivenMission
from src.sitl.requirement_linker import RequirementLinker
from src.sysml.lite_model import build_lite_model

_BUNDLE = (
    Path(__file__).parent.parent
    / "examples" / "output" / "run3_authoritative_20260831"
)

# Alien renames: no stem shared with the originals, and no PARACHUTE/CHUTE/CMD
# substring a legacy matcher could latch onto.
_ALIEN = {
    "DeliveryCoordinateConditionSatisfied": "KryptonPulseNine",
    "PayloadReleaseCommand": "XylemBurstFour",
    "CriticalPropulsionSubsystemFailure": "QuasarDropSeven",
    "deployBallisticRecoveryParachute": "zephyrActionTwo",
    "releasePayload": "zephyrActionOne",
    "deliveryAbortConditionActive": "omegaFlagThree",
    "recoveryCmd": "gammaPortTwo",
    "RecoveryCmdData": "DeltaItemSix",
    "RecoveryCmdPort": "DeltaSixPort",
}


def _rename(text: str, renames: dict) -> str:
    for old, new in renames.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def _model_text() -> str:
    return (_BUNDLE / "final_model.sysml").read_text()


def _plan_payload() -> dict:
    return json.loads((_BUNDLE / "run_report.json").read_text())[
        "whole_model_generation_plan"
    ]


def _linker_shape(model_text: str, plan_payload: dict):
    model = build_lite_model(model_text, model_name="AutonomousDrone")
    bundle = RequirementLinker(model, plan_payload=plan_payload).compile_evidence()
    return (
        {s.req_id for s in bundle.test_specs if s.tier == "L2"},
        {m.get("req_id") for m in bundle.traceability_mismatches},
    )


def test_plan_tier_is_spelling_independent():
    """Anti-coupling, PLAN tier: rename model and plan consistently with alien
    identities; the L2 suite and mismatch set stay byte-identical to run3's shape.

    The gap this pin once held open ({FUNC_005, SAFE_006}: the AST synthesizer's
    keyword-family guard matching) was closed by the identity tier (plan identifiers
    -> semantic tags); a reappearing gap means a harness spelling is load-bearing
    again.
    """
    base = _linker_shape(_model_text(), _plan_payload())
    renamed_plan = json.loads(_rename(json.dumps(_plan_payload()), _ALIEN))
    renamed = _linker_shape(_rename(_model_text(), _ALIEN), renamed_plan)
    assert renamed == base


def test_semantic_tier_resolves_renames():
    baseline = ModelDrivenMission(model_text=_model_text()).guards_for_event(
        "DeliveryCoordinateSatisfied"
    )
    superset = _rename(_model_text(), {
        "DeliveryCoordinateConditionSatisfied":
            "VerifiedDeliveryCoordinateConditionSatisfiedEvent",
    })
    renamed = ModelDrivenMission(model_text=superset).guards_for_event(
        "DeliveryCoordinateSatisfied"
    )
    assert baseline and renamed == baseline


def test_alien_rename_reads_not_measured():
    """Anti-coupling, SEMANTIC tier limit: an alien rename is beyond term matching, so
    the only acceptable reading is None (not measured). () would repeat the run3
    false verdict with extra steps.
    """
    mission = ModelDrivenMission(model_text=_rename(_model_text(), _ALIEN))
    assert mission.guards_for_event("DeliveryCoordinateSatisfied") is None


def test_guard_strip_reads_unguarded():
    stripped = re.sub(
        r"(?m)^[ \t]*if not deliveryAbortConditionActive[ \t]*\n", "",
        _model_text(),
    )
    assert "if not deliveryAbortConditionActive" not in stripped
    mission = ModelDrivenMission(model_text=stripped)
    assert mission.guards_for_event("DeliveryCoordinateSatisfied") == ()


def test_off_route_send_mismatches():
    """Anti-laundering, layer 4: the route widening accepts only a send on the
    requirement's own causal-path leg. Redirect the parachute command to another
    existing port and the mismatch returns.
    """
    off_route = _model_text().replace(
        "send new RecoveryCmdData() to recoveryCmd;",
        "send new RecoveryCmdData() to overrideCmd;",
    )
    assert "to overrideCmd;" in off_route
    _l2, mismatches = _linker_shape(off_route, _plan_payload())
    assert "REQ_SAFE_005" in mismatches
