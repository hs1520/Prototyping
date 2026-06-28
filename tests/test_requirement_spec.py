"""Structured requirement extraction: free-text → controlled-vocabulary specs.

Replaces the scattered keyword heuristics with one auditable parse (LLM when available,
deterministic rule extractor as fallback). These lock the disambiguations that were bugs.
"""
from __future__ import annotations

import json

from src.dse.requirement_spec import (
    ALTITUDE, ENDURANCE, MASS_MTOW, PAYLOAD, RANGE, ReqSpec, _rule_extract,
    extract_requirements, max_spec, max_value,
)

_REQS = [
    "REQ-PERF-002: sustain flight for a minimum of 25 minutes at the maximum rated payload.",
    "REQ-FUNC-003: transport payloads with a gross mass of up to 2.5 kg.",
    "REQ-CONS-003: maximum take-off weight, including payload and battery, shall not exceed 25.0 kg.",
    "REQ-CONS-001: shall not exceed a flight altitude of 120 metres above ground level.",
    "REQ-FUNC-006: incorporate a waypoint within 1.0 second of command.",
]


def _q(specs, quantity):
    return [s for s in specs if s.quantity == quantity]


def test_rule_extract_classifies_each_quantity_correctly():
    specs = _rule_extract(_REQS)
    # endurance: minutes, >=, 25 (the 1.0-second latency is NOT endurance)
    assert _q(specs, ENDURANCE) == [ReqSpec("REQ-PERF-002", ENDURANCE, ">=", 25.0, "minutes")]
    # payload (carry) vs MTOW (take-off) split, both kg <=
    assert _q(specs, PAYLOAD) == [ReqSpec("REQ-FUNC-003", PAYLOAD, "<=", 2.5, "kg")]
    assert _q(specs, MASS_MTOW) == [ReqSpec("REQ-CONS-003", MASS_MTOW, "<=", 25.0, "kg")]
    # altitude is altitude (<=), NOT range — the metre-unit trap
    assert _q(specs, ALTITUDE) == [ReqSpec("REQ-CONS-001", ALTITUDE, "<=", 120.0, "metres")]
    assert _q(specs, RANGE) == []


def test_rule_extract_operational_range_and_km():
    assert max_spec(_rule_extract(["REQ-PERF-004: operational range of at least 10 km."]), RANGE).value == 10000.0
    # a sensor "detection range" is not operational flight range
    assert _q(_rule_extract(["REQ-FUNC-002: detection range of 15 metres."]), RANGE) == []


def test_query_helpers():
    specs = _rule_extract(_REQS)
    assert max_value(specs, ENDURANCE) == 25.0
    assert max_value(specs, PAYLOAD) == 2.5
    assert max_spec(specs, MASS_MTOW).req_id == "REQ-CONS-003"
    assert max_spec(specs, RANGE) is None


class _FakeLLM:
    """Returns a hand-authored JSON extraction (+ one invalid row to exercise validation)."""
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, prompt, system_prompt=None):
        self.calls += 1
        return "```json\n" + json.dumps(self.payload) + "\n```"


def test_extract_uses_llm_when_available_and_validates():
    reqs = ["REQ-XLLM-001: fly for half an hour carrying the parcel."]   # unique → no cache hit
    llm = _FakeLLM([
        {"req_id": "REQ-XLLM-001", "quantity": "endurance", "operator": ">=", "value": 30, "unit": "min"},
        {"req_id": "REQ-XLLM-001", "quantity": "bogus", "operator": ">=", "value": 1, "unit": "x"},  # dropped
    ])
    specs = extract_requirements(reqs, llm=llm)
    assert llm.calls == 1
    assert specs == [ReqSpec("REQ-XLLM-001", ENDURANCE, ">=", 30.0, "min")]   # invalid row filtered


def test_operator_filter_excludes_geofence_range_from_capability():
    # a "<=" operational-radius limit must NOT be claimed as a ">=" range capability
    specs = [
        ReqSpec("REQ-CONS-005", RANGE, "<=", 10000.0, "km"),   # geofence (max radius)
        ReqSpec("REQ-PERF-004", RANGE, ">=", 8000.0, "metres"),  # capability target
    ]
    assert max_spec(specs, RANGE, ">=").req_id == "REQ-PERF-004"   # capability only
    assert max_value(specs, ENDURANCE, ">=") == 0.0


def test_range_requirement_skips_geofence_via_llm(monkeypatch=None):
    # end-to-end: LLM extracts REQ-CONS-005 as range "<=" (max radius) → range_requirement
    # (capability, ">=") returns None, so no false "RangeM >= 10000 satisfy req_cons_005".
    from src.dse.domain_objective import range_requirement
    reqs = ["REQ-CONS-005Z: restrict operational flight radius to a maximum of 10.0 km."]  # unique
    llm = _FakeLLM([{"req_id": "REQ-CONS-005Z", "quantity": "range",
                     "operator": "<=", "value": 10000, "unit": "km"}])
    extract_requirements(reqs, llm=llm)              # prime cache with the LLM spec
    assert range_requirement(reqs) == (None, 0.0)    # geofence not a range capability


def test_extract_falls_back_to_rules_when_llm_raises():
    reqs = ["REQ-XFB-002: sustain flight for 25 minutes."]   # unique → no cache hit

    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("llm down")

    specs = extract_requirements(reqs, llm=_Boom())
    assert max_value(specs, ENDURANCE) == 25.0   # deterministic fallback


def test_enforce_requirement_text_restores_verbatim():
    from src.dse.requirement_spec import enforce_requirement_text
    # LLM-corrupted doc ("all calculations") gets overwritten with the canonical input text
    model = ("package D {\n"
             "  requirement def REQ_SAFE_005 { doc /* deploy parachute, taking precedence over "
             "all calculations. */ }\n"
             "  requirement def REQ_FUNC_003 { doc /* carry up to 9 kg. */ }\n"
             "  requirement def REQ_XXX_999 { doc /* unknown, leave as-is. */ }\n}")
    reqs = ["REQ-SAFE-005: deploy the parachute, taking precedence over all other safety responses.",
            "REQ-FUNC-003: transport payloads with a gross mass of up to 1.5 kg."]
    out = enforce_requirement_text(model, reqs)
    assert "all other safety responses" in out and "all calculations" not in out
    assert "up to 1.5 kg" in out and "up to 9 kg" not in out          # corrupted value restored
    assert "unknown, leave as-is" in out                              # unknown req untouched


def test_bind_current_payload_unvacuums_bound():
    from src.dse.requirement_spec import bind_current_payload
    m = ("part def P {\n  attribute maxPayloadMass_kg : Real = 1.5 [kg];\n"
         "  attribute currentPayloadMass_kg : Real = 0.0 [kg];\n"
         "  assert constraint payloadMassBound { currentPayloadMass_kg <= maxPayloadMass_kg }\n}")
    reqs = ["REQ-FUNC-003: transport payloads with a gross mass of up to 1.5 kg."]
    out = bind_current_payload(m, reqs)
    assert "currentPayloadMass_kg : Real = 1.5" in out          # bound to rated delivery payload
    assert "currentPayloadMass_kg : Real = 0.0" not in out      # vacuous placeholder gone
    # no payload requirement → untouched
    assert bind_current_payload(m, ["REQ-PERF-001: cruise at 15 m/s."]) == m
