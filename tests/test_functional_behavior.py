"""Behavioural verification of FUNCTIONAL actuation requirements (roadmap ①): a functional
response (release/navigate/report/return) is verified only if a REACHABLE state actually
produces that action; otherwise behavior-absent (honestly flags missing actuation logic).
"""
from __future__ import annotations

from src.dse.functional_behavior import functional_behavior_status
from src.dse.safety_behavior import BEHAVIOR_ABSENT, BEHAVIORALLY_VERIFIED

_MODEL = """package D {
    requirement def REQ_FUNC_011 { doc /* release payload at the waypoint */ }
    requirement def REQ_FUNC_012 { doc /* navigate to the GPS waypoint */ }
    part def FC {
        action def ReleaseAction { }
        attribute pos : Real = 0.0;
        state def M {
            state cruising;
            state delivering { entry action releasePayload : ReleaseAction; }
            transition initial then cruising;
            transition d first cruising if pos > 0.5 then delivering;
        }
        satisfy requirement REQ_FUNC_011;
        satisfy requirement REQ_FUNC_012;
    }
}"""
_REQS = [
    "REQ-FUNC-011: release the payload at the delivery waypoint.",
    "REQ-FUNC-012: navigate to the GPS waypoint.",
]


def test_functional_response_verified_and_absent():
    st = functional_behavior_status(_MODEL, _REQS)
    assert st["REQ-FUNC-011"] == BEHAVIORALLY_VERIFIED   # 'delivering' produces releasePayload
    assert st["REQ-FUNC-012"] == BEHAVIOR_ABSENT         # no navigate action produced anywhere


def test_unreachable_action_is_absent():
    # make the action-bearing state unreachable (no transition into it) → response not reachable
    model = _MODEL.replace("transition d first cruising if pos > 0.5 then delivering;", "")
    st = functional_behavior_status(model, _REQS)
    assert st["REQ-FUNC-011"] == BEHAVIOR_ABSENT


def test_landing_action_does_not_satisfy_postflight_report_response():
    model = """package D {
        requirement def REQ_FUNC_008 {
            doc /* Transmit a post-flight health report after landing. */
        }
        part def FC {
            action def landNow { }
            action def transmitHealthReport { }
            state def M {
                state Cruise;
                state Land { entry action onLand : landNow; }
                transition initial then Cruise;
                transition finish first Cruise accept CmdToLand then Land;
            }
            satisfy requirement REQ_FUNC_008;
        }
    }"""

    st = functional_behavior_status(
        model,
        ["REQ-FUNC-008: Transmit a post-flight health report after landing."],
    )

    assert st["REQ-FUNC-008"] == BEHAVIOR_ABSENT
