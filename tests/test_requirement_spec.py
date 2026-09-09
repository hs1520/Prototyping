"""Structured requirement extraction: free-text -> controlled-vocabulary specs.

One parse replaces the scattered keyword heuristics (LLM when available, rule
extractor as fallback). These lock the disambiguations that were bugs.
"""
from __future__ import annotations

import json

from src.dse.requirement_spec import (
    ALTITUDE, ENDURANCE, MASS_MTOW, PAYLOAD, RANGE, SPEED, ReqSpec, _rule_extract,
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


def test_rule_extract_quantities():
    specs = _rule_extract(_REQS)
    assert _q(specs, ENDURANCE) == [ReqSpec("REQ-PERF-002", ENDURANCE, ">=", 25.0, "minutes")]
    assert _q(specs, PAYLOAD) == [ReqSpec("REQ-FUNC-003", PAYLOAD, "<=", 2.5, "kg")]
    assert _q(specs, MASS_MTOW) == [ReqSpec("REQ-CONS-003", MASS_MTOW, "<=", 25.0, "kg")]
    # altitude is altitude (<=), not range - the metre-unit trap
    assert _q(specs, ALTITUDE) == [ReqSpec("REQ-CONS-001", ALTITUDE, "<=", 120.0, "metres")]
    assert _q(specs, RANGE) == []


def test_rule_extract_range_km():
    assert max_spec(_rule_extract(["REQ-PERF-004: operational range of at least 10 km."]), RANGE).value == 10000.0
    # a sensor "detection range" is not operational flight range
    assert _q(_rule_extract(["REQ-FUNC-002: detection range of 15 metres."]), RANGE) == []


def test_rule_extract_skips_wind():
    cruise = _rule_extract(["REQ-PERF-003: cruise speed at least 15 m/s."])
    assert _q(cruise, SPEED) == [ReqSpec("REQ-PERF-003", SPEED, ">=", 15.0, "m/s")]
    wind = _rule_extract(["REQ-PERF-004: operate in wind conditions up to 15 m/s."])
    assert _q(wind, SPEED) == []


def test_max_airspeed_lower_bound():
    capability = _rule_extract([
        "REQ-PERF-003: The system shall achieve a maximum airspeed of 15 m/s in nil-wind level flight."
    ])
    assert _q(capability, SPEED) == [ReqSpec("REQ-PERF-003", SPEED, ">=", 15.0, "m/s")]

    limit = _rule_extract([
        "REQ-CONS-010: The system shall not exceed a maximum speed of 15 m/s near the depot."
    ])
    assert _q(limit, SPEED) == [ReqSpec("REQ-CONS-010", SPEED, "<=", 15.0, "m/s")]


def test_query_helpers():
    specs = _rule_extract(_REQS)
    assert max_value(specs, ENDURANCE) == 25.0
    assert max_value(specs, PAYLOAD) == 2.5
    assert max_spec(specs, MASS_MTOW).req_id == "REQ-CONS-003"
    assert max_spec(specs, RANGE) is None


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, prompt, system_prompt=None):
        self.calls += 1
        return "```json\n" + json.dumps(self.payload) + "\n```"


def test_extract_uses_llm():
    reqs = ["REQ-XLLM-001: fly for half an hour carrying the parcel."]   # unique -> no cache hit
    llm = _FakeLLM([
        {"req_id": "REQ-XLLM-001", "quantity": "endurance", "operator": ">=", "value": 30, "unit": "min"},
        {"req_id": "REQ-XLLM-001", "quantity": "bogus", "operator": ">=", "value": 1, "unit": "x"},
    ])
    specs = extract_requirements(reqs, llm=llm)
    assert llm.calls == 1
    assert specs == [ReqSpec("REQ-XLLM-001", ENDURANCE, ">=", 30.0, "min")]


def test_operator_filter_excludes_geofence():
    specs = [
        ReqSpec("REQ-CONS-005", RANGE, "<=", 10000.0, "km"),
        ReqSpec("REQ-PERF-004", RANGE, ">=", 8000.0, "metres"),
    ]
    assert max_spec(specs, RANGE, ">=").req_id == "REQ-PERF-004"
    assert max_value(specs, ENDURANCE, ">=") == 0.0


def test_range_skips_geofence_llm(monkeypatch=None):
    # LLM extracts REQ-CONS-005 as range "<=" (max radius) -> range_requirement
    # (capability, ">=") returns None, so no false satisfy link is emitted.
    from src.dse.domain_objective import range_requirement
    reqs = ["REQ-CONS-005Z: restrict operational flight radius to a maximum of 10.0 km."]
    llm = _FakeLLM([{"req_id": "REQ-CONS-005Z", "quantity": "range",
                     "operator": "<=", "value": 10000, "unit": "km"}])
    extract_requirements(reqs, llm=llm)
    assert range_requirement(reqs) == (None, 0.0)


def test_extract_falls_back_to_rules():
    reqs = ["REQ-XFB-002: sustain flight for 25 minutes."]   # unique -> no cache hit

    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("llm down")

    specs = extract_requirements(reqs, llm=_Boom())
    assert max_value(specs, ENDURANCE) == 25.0


def test_enforce_text_verbatim():
    from src.dse.requirement_spec import enforce_requirement_text
    model = ("package D {\n"
             "  requirement def REQ_SAFE_005 { doc /* deploy parachute, taking precedence over "
             "all calculations. */ }\n"
             "  requirement def REQ_FUNC_003 { doc /* carry up to 9 kg. */ }\n"
             "  requirement def REQ_XXX_999 { doc /* unknown, leave as-is. */ }\n}")
    reqs = ["REQ-SAFE-005: deploy the parachute, taking precedence over all other safety responses.",
            "REQ-FUNC-003: transport payloads with a gross mass of up to 1.5 kg."]
    out = enforce_requirement_text(model, reqs)
    assert "all other safety responses" in out and "all calculations" not in out
    assert "up to 1.5 kg" in out and "up to 9 kg" not in out
    assert "unknown, leave as-is" in out


def test_bind_current_payload():
    from src.dse.requirement_spec import bind_current_payload
    m = ("part def P {\n  attribute maxPayloadMass_kg : Real = 1.5 [kg];\n"
         "  attribute currentPayloadMass_kg : Real = 0.0 [kg];\n"
         "  assert constraint payloadMassBound { currentPayloadMass_kg <= maxPayloadMass_kg }\n}")
    reqs = ["REQ-FUNC-003: transport payloads with a gross mass of up to 1.5 kg."]
    out = bind_current_payload(m, reqs)
    assert "currentPayloadMass_kg : Real = 1.5" in out
    assert "currentPayloadMass_kg : Real = 0.0" not in out
    assert bind_current_payload(m, ["REQ-PERF-001: cruise at 15 m/s."]) == m
