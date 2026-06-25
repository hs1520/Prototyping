"""SAFE-class behavioural verification: upgrade safety `satisfy` (allocation) to real
three-state behavioural evidence — does the safety state machine actually REACH a fail-safe
state? verified / violated (named but unreachable / no safe state) / absent (no state machine).
"""
from __future__ import annotations

from src.dse.safety_behavior import (
    BEHAVIOR_ABSENT, BEHAVIORALLY_VERIFIED, BEHAVIORALLY_VIOLATED, safety_behavior_status,
)

# REQ_SAFE_001: state machine reaches failsafe → verified
# REQ_SAFE_002: state machine present but NO safe state reachable → violated
# REQ_SAFE_003: satisfying part has no state machine → absent
_MODEL = """package D {
    requirement def REQ_SAFE_001 { doc /* enter failsafe on fault */ }
    requirement def REQ_SAFE_002 { doc /* must reach a safe state */ }
    requirement def REQ_SAFE_003 { doc /* lock on abort */ }
    part def MonitorA {
        action def emergencyStop { }
        attribute fault : Real = 0.0;
        state def MonA {
            state nominal;
            state failsafe { entry action stop : emergencyStop; }
            transition initial then nominal;
            transition toFail first nominal if fault > 0.5 then failsafe;
        }
        satisfy requirement REQ_SAFE_001;
    }
    part def MonitorB {
        attribute mode : Real = 0.0;
        state def MonB {
            state running;
            state degraded;
            transition initial then running;
            transition d first running if mode > 0.5 then degraded;
        }
        satisfy requirement REQ_SAFE_002;
    }
    part def MonitorC {
        attribute z : Real = 0.0;
        satisfy requirement REQ_SAFE_003;
    }
}"""
_REQS = [
    "REQ-SAFE-001: the system shall enter a failsafe state on fault.",
    "REQ-SAFE-002: the system shall reach a safe state.",
    "REQ-SAFE-003: the system shall lock on abort.",
]


def test_three_state_safety_behaviour():
    st = safety_behavior_status(_MODEL, _REQS)
    assert st["REQ-SAFE-001"] == BEHAVIORALLY_VERIFIED   # failsafe reachable
    assert st["REQ-SAFE-002"] == BEHAVIORALLY_VIOLATED   # SM present, no safe state
    assert st["REQ-SAFE-003"] == BEHAVIOR_ABSENT         # no state machine on the part


def test_only_safety_requirements_classified():
    # a non-safety requirement is not pulled into the safety behavioural check
    model = _MODEL.replace("REQ_SAFE_001", "REQ_FUNC_020").replace(
        "REQ-SAFE-001", "REQ-FUNC-020")
    reqs = ["REQ-FUNC-020: navigate to a waypoint."] + _REQS[1:]
    st = safety_behavior_status(model, reqs)
    assert "REQ-FUNC-020" not in st                      # not a safety req → skipped


def test_dynamic_failed_downgrades_reachable_safety_to_violated():
    from src.dse.safety_behavior import (BEHAVIORALLY_VERIFIED, BEHAVIORALLY_VIOLATED,
                                         safety_behavior_status)
    # SAFE-001 is structurally reachable→verified; a dynamic 'failed' (guard never fires)
    # downgrades it to violated — the fake-safety catch structural reachability can't make.
    assert safety_behavior_status(_MODEL, _REQS)["REQ-SAFE-001"] == BEHAVIORALLY_VERIFIED
    st = safety_behavior_status(_MODEL, _REQS, dynamic_fire={"MonitorA": "failed"})
    assert st["REQ-SAFE-001"] == BEHAVIORALLY_VIOLATED


def test_dynamic_firing_does_not_oververify_without_safe_state():
    from src.dse.safety_behavior import BEHAVIORALLY_VIOLATED, safety_behavior_status
    # MonitorB has no fail-safe state → a spurious non-safe 'fired' must NOT verify it
    st = safety_behavior_status(_MODEL, _REQS, dynamic_fire={"MonitorB": "fired"})
    assert st["REQ-SAFE-002"] == BEHAVIORALLY_VIOLATED


def test_dynamic_fire_by_part_drives_guards():
    from src.dse.dynamic_behavior import dynamic_fire_by_part
    v = dynamic_fire_by_part(_MODEL)
    assert v.get("MonitorA") == "fired"     # fault>0.5 guard driven → transition fires
