"""Honest requirement-verification coverage classifier.

`satisfy` is allocation/intent, not proof — this classifies each requirement by the evidence
that ACTUALLY exists (injected assert / quantitative target / allocated-only), so a baseline
doesn't overclaim verification of functional/safety/interface behaviour it can't prove.
"""
from __future__ import annotations

from src.dse.requirement_coverage import (
    ALLOCATED_ONLY, ANALYSIS_VERIFIED, QUANTITATIVE, classify_requirement_coverage, coverage_summary,
)
from src.dse.safety_behavior import BEHAVIOR_ABSENT

_MODEL = """package D {
    requirement def REQ_PERF_002 { doc /* endurance >= 20 min */ }
    requirement def REQ_CONS_003 { doc /* MTOW <= 25 kg */ }
    requirement def REQ_PERF_003 { doc /* cruise speed >= 15 m/s */ }
    requirement def REQ_FUNC_005 { doc /* release payload at the waypoint when no abort */ }
    requirement def REQ_INTF_003 { doc /* broadcast remote ID per ASTM F3411 */ }
    part def Sys {
        attribute enduranceMin : Real = 21.0;
        assert constraint enduranceMeetsReq { 21.0 >= 20.0 }
        satisfy req_perf_002;
        assert constraint mtowWithinReq { 9.0 <= 25.0 }
        satisfy req_cons_003;
        satisfy requirement REQ_PERF_003;
        satisfy requirement REQ_FUNC_005;
        satisfy requirement REQ_INTF_003;
    }
}"""
_REQS = [
    "REQ-PERF-002: sustain flight for a minimum of 20 minutes at maximum rated payload.",
    "REQ-CONS-003: maximum take-off weight shall not exceed 25 kg.",
    "REQ-PERF-003: cruise speed of at least 15 m/s.",
    "REQ-FUNC-005: release the payload at the delivery waypoint when no abort is active.",
    "REQ-INTF-003: broadcast remote identification per ASTM F3411.",
]


def test_classifies_by_real_evidence():
    cov = classify_requirement_coverage(_MODEL, _REQS)
    assert cov["REQ-PERF-002"] == ANALYSIS_VERIFIED      # injected endurance assert
    assert cov["REQ-CONS-003"] == ANALYSIS_VERIFIED      # injected MTOW assert
    assert cov["REQ-PERF-003"] == QUANTITATIVE           # numeric target, no inline assert
    # FUNC-005 mentions "abort" → safety-related, but its part has no state machine → absent
    assert cov["REQ-FUNC-005"] == BEHAVIOR_ABSENT
    assert cov["REQ-INTF-003"] == ALLOCATED_ONLY         # protocol: satisfy only, unverified


def test_summary_counts_and_no_overclaim():
    cov = classify_requirement_coverage(_MODEL, _REQS)
    s = coverage_summary(cov)
    assert "2 analysis-verified" in s
    assert "allocated-only (intent, UNVERIFIED)" in s


def test_endurance_not_analysis_verified_without_the_assert():
    # remove the injected assert → endurance is no longer "verified", just allocated/quantitative
    model = _MODEL.replace("assert constraint enduranceMeetsReq { 21.0 >= 20.0 }", "")
    cov = classify_requirement_coverage(model, _REQS)
    assert cov["REQ-PERF-002"] == QUANTITATIVE           # numeric target remains, but no assert


def test_gazebo_upgrades_endurance_to_flight_verified():
    from src.dse.requirement_coverage import classify_requirement_coverage, FLIGHT_VERIFIED
    # a model where endurance is analysis-verified (assert present)
    model = """package D {
        requirement def REQ_PERF_002 { doc /* endurance */ }
        part def Drone { attribute enduranceMeetsReq : Boolean; satisfy requirement REQ_PERF_002; }
    }"""
    reqs = ["REQ-PERF-002: the system shall sustain flight for a minimum of 20 minutes."]
    base = classify_requirement_coverage(model, reqs)
    assert base["REQ-PERF-002"] == "analysis-verified"          # without Gazebo
    # Gazebo flew stably + datasheet endurance 35 ≥ 20 → flight-verified (upgrade)
    gv = {"status": "ok", "datasheet_endurance_min": 35.2}
    up = classify_requirement_coverage(model, reqs, gazebo=gv)
    assert up["REQ-PERF-002"] == FLIGHT_VERIFIED
    # but NOT if datasheet endurance falls short of the target
    short = classify_requirement_coverage(model, reqs, gazebo={"status": "ok", "datasheet_endurance_min": 12})
    assert short["REQ-PERF-002"] == "analysis-verified"
    # and NOT if the flight wasn't stable
    unstable = classify_requirement_coverage(model, reqs, gazebo={"status": "infeasible", "datasheet_endurance_min": 35})
    assert unstable["REQ-PERF-002"] == "analysis-verified"
