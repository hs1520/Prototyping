"""Structured requirement extraction: free-text requirements → controlled-vocabulary specs.

ONE auditable parse replaces the scattered keyword heuristics (endurance_target /
max_rated_payload / range_requirement / mass-limit) that proved brittle — a length unit
can't tell range from altitude, "payload" appears in MTOW text, "second" isn't endurance.
An LLM does the semantic understanding when available (robust to phrasing; output pinned to
a controlled vocabulary + validated), and a deterministic rule extractor is the offline/test
fallback. Consumers query specs by quantity instead of re-grepping requirement text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── controlled vocabulary (canonical unit in the comment) ─────────────────────────────
ENDURANCE = "endurance"   # flight time,           minutes,  >=
MASS_MTOW = "mtow"        # max take-off/all-up mass, kg,     <=
PAYLOAD = "payload"       # rated carried payload,  kg,       <=
RANGE = "range"           # operational/flight range, metres, >=
SPEED = "speed"           # cruise/airspeed,        m/s,      >=
ALTITUDE = "altitude"     # vertical limit,         metres,   <=  (classified so it is NOT
                          #                                        mistaken for range)
QUANTITIES = (ENDURANCE, MASS_MTOW, PAYLOAD, RANGE, SPEED, ALTITUDE)
_OPERATORS = (">=", "<=", "==")


@dataclass(frozen=True)
class ReqSpec:
    req_id: str
    quantity: str         # one of QUANTITIES
    operator: str         # ">=", "<=", "=="
    value: float          # in the canonical unit: minutes / kg / metres / m·s⁻¹
    unit: str


# ── deterministic rule extractor (consolidates the former scattered keyword constants) ──
_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z/%°]+(?:\s*/\s*[A-Za-z]+)?)?")
_REQ_ID_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")
_TIME_UNIT_MIN = {"minute": 1.0, "minutes": 1.0, "min": 1.0, "mins": 1.0,
                  "hour": 60.0, "hours": 60.0, "hr": 60.0, "hrs": 60.0}
_MASS_UNIT_KG = {"kg": 1.0, "kgs": 1.0, "kilogram": 1.0, "kilograms": 1.0,
                 "gram": 0.001, "grams": 0.001, "g": 0.001}
_LEN_UNIT_M = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
               "km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
               "kilometer": 1000.0, "kilometers": 1000.0}
_SPEED_UNIT_MPS = {"m/s": 1.0, "mps": 1.0, "m / s": 1.0}
_PAYLOAD_CARRY = ("transport", "carry", "carries", "carrying", "lift", "cargo", "payload")
_MTOW_KW = ("takeoff", "take-off", "take off", "mtow", "all-up", "all up", "gross weight")
_RANGE_TERMS = ("operational range", "flight range", "maximum range", "max range",
                "mission radius", "operational radius", "ferry range")
_VERTICAL_TERMS = ("altitude", "height", "agl", "above ground", "ceiling", "vertical")
_SPEED_CAPABILITY_TERMS = (
    "cruise speed", "cruise at", "airspeed", "flight speed", "ground speed",
    "maximum speed", "max speed", "fly at", "fly with speed",
)
_CONDITION_SPEED_TERMS = ("wind", "gust", "headwind", "tailwind", "crosswind")
_CAPABILITY_VERBS = ("achieve", "attain", "reach", "provide", "support", "maintain")


def _operator_for(body: str, default: str = ">=") -> str:
    low = body.lower()
    upper_terms = (
        "at most", "maximum", "max ", "shall not exceed", "not exceed",
        "no more than", "up to", "within", "below", "less than", "under",
        "restrict", "limited to", "limit",
    )
    lower_terms = (
        "at least", "minimum", "min ", "no less than", "not less than",
        "greater than", "more than",
    )
    if any(t in low for t in lower_terms):
        return ">="
    if any(t in low for t in upper_terms):
        return "<="
    return default


def _first(body: str, unit_map) -> Optional[Tuple[float, str]]:
    for num, unit in _NUM_UNIT_RE.findall(body):
        mult = unit_map.get((unit or "").strip().lower())
        if mult is not None:
            return float(num) * mult, (unit or "").strip()
    return None


def _rule_extract(requirements) -> List[ReqSpec]:
    """Behaviour-preserving deterministic extractor: the consolidated keyword logic, now in
    one place producing structured specs (quantity + operator + canonical-unit value)."""
    specs: List[ReqSpec] = []
    for r in requirements or []:
        low = r.lower()
        m = _REQ_ID_RE.search(r)
        rid = m.group(0).replace("_", "-") if m else ""
        body = r.split(":", 1)[1] if ":" in r else r
        is_mtow = any(k in low for k in _MTOW_KW)
        is_carry = any(k in low for k in _PAYLOAD_CARRY)
        is_vertical = any(k in low for k in _VERTICAL_TERMS)
        is_range = any(k in low for k in _RANGE_TERMS) and not is_vertical

        op = _operator_for(body)
        t = _first(body, _TIME_UNIT_MIN)        # endurance (minutes; hours→minutes)
        if t:
            specs.append(ReqSpec(rid, ENDURANCE, op, t[0], t[1]))
        mss = _first(body, _MASS_UNIT_KG)        # MTOW (take-off) vs payload (carry)
        if mss and is_mtow:
            specs.append(ReqSpec(rid, MASS_MTOW, "<=", mss[0], mss[1]))
        elif mss and is_carry:
            specs.append(ReqSpec(rid, PAYLOAD, "<=", mss[0], mss[1]))
        ln = _first(body, _LEN_UNIT_M)           # operational range vs altitude
        if ln and is_range:
            range_op = op
            if ("maximum range" in low or "max range" in low) and not any(
                k in low for k in ("restrict", "radius", "geofence", "limited to", "limit")
            ):
                range_op = ">="
            specs.append(ReqSpec(rid, RANGE, range_op, ln[0], ln[1]))
        elif ln and is_vertical:
            specs.append(ReqSpec(rid, ALTITUDE, "<=", ln[0], ln[1]))
        sp = _first(body, _SPEED_UNIT_MPS)       # cruise/airspeed
        is_wind_condition = (
            any(k in low for k in _CONDITION_SPEED_TERMS)
            and not any(k in low for k in ("nil-wind", "nil wind", "no-wind", "no wind"))
        )
        is_speed_capability = (
            any(k in low for k in _SPEED_CAPABILITY_TERMS)
            and not is_wind_condition
        )
        if sp and is_speed_capability:
            speed_op = op
            if (
                any(term in low for term in ("maximum airspeed", "max airspeed", "maximum speed", "max speed"))
                and any(verb in low for verb in _CAPABILITY_VERBS)
                and not any(k in low for k in ("shall not exceed", "not exceed", "restrict", "limited to", "limit", "up to", "within"))
            ):
                speed_op = ">="
            specs.append(ReqSpec(rid, SPEED, speed_op, sp[0], sp[1]))
    return specs


# ── LLM extractor (robust to phrasing; output validated against the vocabulary) ─────────
_LLM_SYSTEM = "You convert systems-engineering requirements into structured quantitative specs."
_LLM_PROMPT = """Extract EVERY quantitative constraint from the requirements below as JSON.

Output ONLY a JSON array; each element:
  {{"req_id": "REQ-...", "quantity": <one of endurance|mtow|payload|range|speed|altitude>,
    "operator": <one of ">="|"<="|"==">, "value": <number in the canonical unit>, "unit": "<original unit>"}}

Canonical units (convert to these):
- endurance = flight time in MINUTES (hours→minutes). A response/latency time in SECONDS is NOT endurance — omit it.
- mtow      = maximum take-off / all-up / gross mass in kg.
- payload   = carried/transported payload mass in kg ("payload/carry/transport" mass, NOT take-off mass).
- range     = OPERATIONAL/flight range or radius in metres (km→metres). A maximum ALTITUDE limit is 'altitude', NOT 'range'.
- speed     = cruise/airspeed in m/s. "Achieve/attain/reach a maximum airspeed of X" is
              a capability target (">="); "shall not exceed / limited to / up to X" is a limit ("<=").
- altitude  = vertical limit in metres.
operator: take the requirement's ACTUAL bound — "at least / minimum / no less than" → ">=";
"at most / maximum of / shall not exceed / restrict to / up to / within" → "<=". Do NOT assume a
direction from the quantity (e.g. a 'restrict operational radius to a maximum of 10 km' is range with "<=").
Omit non-quantitative requirements. Output the JSON array only, no prose.

Requirements:
{reqs}"""


def _llm_extract(requirements, llm) -> Optional[List[ReqSpec]]:
    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(requirements or []))
    raw = str(llm.chat(_LLM_PROMPT.format(reqs=numbered), system_prompt=_LLM_SYSTEM))
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    specs: List[ReqSpec] = []
    for d in data if isinstance(data, list) else []:
        q = str(d.get("quantity", "")).strip().lower()
        op = str(d.get("operator", "")).strip()
        if q not in QUANTITIES or op not in _OPERATORS:
            continue                                  # validate against the controlled vocab
        try:
            specs.append(ReqSpec(str(d.get("req_id", "")), q, op, float(d["value"]),
                                 str(d.get("unit", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return specs or None


_CACHE: dict = {}


def extract_requirements(requirements, llm=None) -> List[ReqSpec]:
    """Structured specs for the requirements. Uses ``llm`` (an object with ``.chat(prompt,
    system_prompt=...)``) when given — robust to phrasing, validated against the vocabulary;
    falls back to the deterministic rule extractor on any failure or when no LLM. Memoised by
    requirements so the pipeline can prime once with the LLM and deep callers reuse it."""
    key = tuple(requirements or [])
    if key in _CACHE:
        return _CACHE[key]
    specs: Optional[List[ReqSpec]] = None
    if llm is not None:
        try:
            specs = _llm_extract(requirements, llm)
        except Exception:
            specs = None
    if not specs:
        specs = _rule_extract(requirements)
    _CACHE[key] = specs
    return specs


def max_value(specs: List[ReqSpec], quantity: str, operator: Optional[str] = None) -> float:
    vals = [s.value for s in specs if s.quantity == quantity
            and (operator is None or s.operator == operator)]
    return max(vals) if vals else 0.0


def max_spec(specs: List[ReqSpec], quantity: str, operator: Optional[str] = None) -> Optional[ReqSpec]:
    """Largest-value spec of ``quantity``. ``operator`` filters by direction so a metric
    clause only fires for the meaningful bound — e.g. range capability is a ">=" target, so
    a "<=" operational-radius/geofence requirement is NOT claimed satisfied by RangeM."""
    best: Optional[ReqSpec] = None
    for s in specs:
        if s.quantity == quantity and (operator is None or s.operator == operator) \
                and (best is None or s.value > best.value):
            best = s
    return best


_REQDOC_RE = re.compile(
    r"(requirement\s+def\s+(REQ[-_][A-Z]+[-_]\d+)\s*\{\s*doc\s*/\*)(.*?)(\*/)",
    re.DOTALL | re.IGNORECASE)


def enforce_requirement_text(model_text: str, requirements: List[str]) -> str:
    """Overwrite each ``requirement def`` doc body with the VERBATIM canonical requirement text
    from the input list — the LLM paraphrases doc strings during generation and can corrupt the
    meaning (e.g. 'all other safety responses' → 'all calculations'). This restores fidelity.
    Requirements unknown to the input list are left untouched."""
    canon = {}
    for r in requirements or []:
        m = _REQ_ID_RE.search(r)
        if not m:
            continue
        rid = m.group(0).upper().replace("_", "-")
        text = r.split(":", 1)[1].strip() if ":" in r else r.strip()
        canon[rid] = text

    def _sub(mo):
        rid = mo.group(2).upper().replace("_", "-")
        if rid not in canon:
            return mo.group(0)
        return f"{mo.group(1)} {canon[rid]} {mo.group(4)}"

    return _REQDOC_RE.sub(_sub, model_text)


_CUR_PAYLOAD_RE = re.compile(
    r"(current[A-Za-z]*[Pp]ayload[A-Za-z_]*\s*:\s*Real\s*=\s*)0\.0")


def bind_current_payload(model_text: str, requirements: List[str]) -> str:
    """Bind the LLM's placeholder ``current…Payload…Mass : Real = 0.0`` to the actual rated
    delivery payload (the max PAYLOAD spec), so the generated ``payloadMassBound`` constraint
    becomes a REAL check (rated payload ≤ declared capacity) instead of the vacuous 0.0 ≤ max.
    No payload requirement → model unchanged."""
    rated = max_value(extract_requirements(requirements), "payload")
    if not rated or rated <= 0:
        return model_text
    return _CUR_PAYLOAD_RE.sub(lambda m: f"{m.group(1)}{float(rated)}", model_text)
