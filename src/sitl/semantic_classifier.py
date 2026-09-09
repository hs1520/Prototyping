"""semantic_classifier.py

LLM 辅助的状态机语义分类器。

把模型里的每个 state def 喂给 LLM，取回一个语义标签（BATTERY_RTB、
GCS_LOSS、PARACHUTE_DEPLOY 等），供 RequirementLinker 按语义查 catalogue，
不依赖 req ID 精确匹配。

公共 API:
  SEMANTIC_TAGS         - 全部支持的标签集合
  classify_state_machines(model, llm) -> Dict[state_def_name, tag]
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, TYPE_CHECKING

from src.utils.sysml_text_utils import STATE_DEF_RE

if TYPE_CHECKING:
    from src.sysml.lite_model import SysMLLiteModel
    from src.llm.interface import LLMInterface


SEMANTIC_TAGS: List[str] = [
    "BATTERY_RTB",
    "BATTERY_LAND",
    "GCS_LOSS",
    "SENSOR_ARMING_INHIBIT",
    "PARACHUTE_DEPLOY",
    "PAYLOAD_ABORT_LOCK",
    "SENSOR_GROUND_ALERT",
    "ALTITUDE_FENCE",
    "CONTROL_LOOP_RATE",
    "MAVLINK_PROTOCOL",
    "MAX_SPEED",
    "RADIUS_FENCE",
    "RTCM_GPS",
    "UNKNOWN",
]

_TAG_DESCRIPTIONS = """\
BATTERY_RTB             — battery low triggers return-to-launch / RTL
BATTERY_LAND            — battery critical triggers emergency landing
GCS_LOSS                — ground-station heartbeat loss triggers fail-safe land/RTL
SENSOR_ARMING_INHIBIT   — sensor self-test failure blocks arming
PARACHUTE_DEPLOY        — propulsion / motor failure deploys recovery chute
PAYLOAD_ABORT_LOCK      — delivery-abort condition locks payload actuator
SENSOR_GROUND_ALERT     — sensor failure triggers alert/STATUSTEXT to ground station
ALTITUDE_FENCE          — altitude limit enforcement (geofence)
CONTROL_LOOP_RATE       — primary flight control loop frequency
MAVLINK_PROTOCOL        — MAVLink v2 serial protocol on GCS link
MAX_SPEED               — maximum horizontal airspeed / navigation speed limit
RADIUS_FENCE            — maximum operational radius / circular geofence
RTCM_GPS                — RTCM differential GNSS corrections for sub-metre accuracy
UNKNOWN                 — none of the above; cannot classify confidently
"""


_PROMPT_TEMPLATE = """\
Classify each SysML v2 state machine below into one of these semantic tags.
Return JSON only — a single object mapping each `state def` name to its tag.

Allowed tags (return EXACTLY one of these strings, nothing else):
{tag_descriptions}

Rules:
- Look at the state names, guard expressions (e.g. `batterySoc <= 25`), and
  entry actions (e.g. `initiateRtbSequence`).
- If a state machine clearly matches one tag, use it.
- If unsure or the machine is operational (not a fault response), use UNKNOWN.
- Do NOT invent new tags. Do NOT add commentary, do NOT wrap in markdown.

State machines to classify:
{state_machines}

Return ONLY a JSON object like:
{{"BatteryRtbMonitor": "BATTERY_RTB", "LinkLossMonitor": "GCS_LOSS", ...}}
"""


def classify_state_machines(
    model: "SysMLLiteModel",
    llm: "LLMInterface",
    verbose: bool = False,
) -> Dict[str, str]:
    """用 LLM 给模型里的每个 state def 打语义标签。"""
    snippets = _extract_state_def_snippets(model)
    if not snippets:
        return {}

    state_machines_text = "\n\n".join(snippets)
    prompt = _PROMPT_TEMPLATE.format(
        tag_descriptions=_TAG_DESCRIPTIONS,
        state_machines=state_machines_text,
    )

    try:
        raw = llm.chat(prompt, system_prompt="You are a SysML semantic classifier.")
    except Exception as e:
        if verbose:
            print(f"  [SEMANTIC] LLM call failed: {e}")
        return {}

    parsed = _parse_json_dict(raw)
    if not parsed:
        if verbose:
            print(f"  [SEMANTIC] JSON parse failed; raw[:200]={raw[:200]!r}")
        return {}

    valid_tags = set(SEMANTIC_TAGS)
    cleaned: Dict[str, str] = {}
    for name, tag in parsed.items():
        if not isinstance(tag, str):
            continue
        tag = tag.strip().upper()
        if tag in valid_tags and tag != "UNKNOWN":
            cleaned[name] = tag

    if verbose:
        print(f"  [SEMANTIC] classified {len(cleaned)}/{len(parsed)} state machines")
        for n, t in cleaned.items():
            print(f"    • {n} → {t}")

    return cleaned


# ArduPilot Copter 参数白名单，防止 LLM 幻觉
_ARDU_PARAM_WHITELIST: Dict[str, str] = {
    "WPNAV_SPEED":      "max horizontal nav speed in cm/s",
    "WPNAV_SPEED_DN":   "max descent speed in cm/s",
    "WPNAV_SPEED_UP":   "max climb speed in cm/s",
    "WPNAV_RADIUS":     "waypoint acceptance radius in cm",
    "FENCE_RADIUS":     "circular geofence radius in m",
    "FENCE_ALT_MAX":    "max altitude geofence in m",
    "FENCE_ENABLE":     "enable geofence (0/1)",
    "GPS_TYPE":         "GPS receiver type (0=none, 1=auto/UBLOX)",
    "GPS_INJECT_TO":    "RTCM injection target (127=all receivers)",
    "ANGLE_MAX":        "max lean angle in centidegrees",
    "BATT_CAPACITY":    "battery capacity in mAh",
    "BATT_FS_LOW_PCT":  "battery low failsafe threshold (%)",
    "BATT_FS_CRT_PCT":  "battery critical failsafe threshold (%)",
    "ARMING_CHECK":     "pre-arm checks bitmask",
    "FS_GCS_ENABLE":    "GCS failsafe enable (0/1)",
    "FS_GCS_TIMEOUT":   "GCS heartbeat loss timeout in seconds",
    "CHUTE_ENABLED":    "parachute enable (0/1)",
    "SCHED_LOOP_RATE":  "main loop rate in Hz",
    "SERIAL0_PROTOCOL": "serial 0 protocol (2=MAVLink v2)",
    "MOT_THST_MAX":     "max throttle (0.0-1.0)",
}

_REQ_PARAM_PROMPT = """\
You are mapping SysML v2 requirements to ArduPilot Copter parameters.
For each requirement below, identify the ArduPilot parameter that most directly implements it.

Allowed parameters (ONLY use names from this list):
{param_list}

Rules:
- Return ONLY a JSON object — no prose, no markdown fences.
- If a requirement has a numeric value (e.g. "15 m/s", "10 km"), include it
  in "value" converted to the parameter's unit (see descriptions above).
- If no parameter applies, set "tag" to "UNKNOWN".
- "tier" must be "L1" (static parameter check, no SITL needed).

Requirements to map:
{requirements}

Return format:
{{"REQ_ID": {{"param": "PARAM_NAME", "value": <number>, "tier": "L1"}}, ...}}
"""


def suggest_params_from_requirements(
    req_texts: Dict[str, str],
    llm: "LLMInterface",
    verbose: bool = False,
) -> Dict[str, Dict]:
    """
    Layer 3b：用 LLM 从需求文本直接推断 ArduPilot 参数。

    针对无状态机 guard、层1-3均未匹配的 FUNC/PERF/CONS/INTF 需求。
    输出 {req_id: {"param": str, "value": float, "tier": "L1"}}。
    结果通过白名单校验，防止 LLM 幻觉。

    Parameters
    ----------
    req_texts : {req_id: requirement_text}
    llm       : LLM interface
    """
    if not req_texts:
        return {}

    param_list = "\n".join(
        f"  {name:<25} — {desc}"
        for name, desc in _ARDU_PARAM_WHITELIST.items()
    )
    req_lines = "\n".join(
        f"  {rid}: {text}" for rid, text in req_texts.items()
    )
    prompt = _REQ_PARAM_PROMPT.format(
        param_list=param_list,
        requirements=req_lines,
    )

    try:
        raw = llm.chat(prompt, system_prompt="You are an ArduPilot parameter mapper.")
    except Exception as e:
        if verbose:
            print(f"  [REQ-PARAM] LLM call failed: {e}")
        return {}

    parsed = _parse_json_dict(raw)
    if not parsed:
        if verbose:
            print(f"  [REQ-PARAM] JSON parse failed; raw[:200]={raw[:200]!r}")
        return {}

    valid_params = set(_ARDU_PARAM_WHITELIST.keys())
    result: Dict[str, Dict] = {}
    for req_id, entry in parsed.items():
        if not isinstance(entry, dict):
            continue
        param = str(entry.get("param", "")).strip().upper()
        if param not in valid_params or param == "UNKNOWN":
            continue
        try:
            value = float(entry.get("value", 0))
        except (TypeError, ValueError):
            continue
        tier = str(entry.get("tier", "L1")).upper()
        if tier not in ("L1", "L2"):
            tier = "L1"
        result[req_id] = {"param": param, "value": value, "tier": tier}

    if verbose:
        print(f"  [REQ-PARAM] mapped {len(result)}/{len(req_texts)} requirements")
        for rid, entry in result.items():
            print(f"    • {rid} → {entry['param']}={entry['value']}")

    return result


def extract_req_texts_from_model(model: "SysMLLiteModel") -> Dict[str, str]:
    """从 SysML 原始文本里提取每个 requirement def 的 doc 注释文本。"""
    try:
        text = model.to_sysml_text()
    except Exception:
        return {}
    if not text:
        return {}

    result: Dict[str, str] = {}
    for m in re.finditer(
        r'\brequirement\s+def\s+(\w+)\s*\{[^}]*?/\*\s*(.*?)\s*\*/[^}]*?\}',
        text,
        re.DOTALL,
    ):
        req_id = m.group(1)
        doc_text = re.sub(r'\s+', ' ', m.group(2)).strip()
        result[req_id] = doc_text
    return result


def _extract_state_def_snippets(model: "SysMLLiteModel") -> List[str]:
    try:
        text = model.to_sysml_text()
    except Exception:
        return []
    if not text:
        return []

    snippets: List[str] = []
    for m in STATE_DEF_RE.finditer(text):
        name = m.group(1)
        brace_open = text.index('{', m.start())
        depth = 0
        end = brace_open
        for i in range(brace_open, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        body = text[brace_open:end + 1]
        snippets.append(f"state def {name} {body}")

    return snippets


def _parse_json_dict(raw: str) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = raw.find('{')
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            return None
        candidate = raw[start:end + 1]

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None
    return {str(k): str(v) for k, v in obj.items()}
