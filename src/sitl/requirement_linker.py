"""
requirement_linker.py

Maps SysML requirement IDs to ArduPilot parameters and SITL test specs.

核心设计原则（设计文档 §2.3）：
  以 Requirement ID 作为语义锚点，通过 satisfy_relationships + AST 状态机
  提取 guard 阈值，而非依赖属性名字符串。

流程：
  REQ_SAFE_001
    → 找到 satisfy 该需求的 part（e.g. SafetyMonitor）
    → 用 state_extractor 提取该 part 的状态机 guard threshold
    → 按 REQ_ID 的语义意图映射到 ArduPilot 参数
    → 数值来自 AST，不依赖属性名

这样无论 LLM 把阈值属性命名为 batteryRtbThreshold、rtbBatteryThreshold
还是 lowBatteryLevel，只要数值存在于 guard 里，都能正确提取。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from src.sysml.lite_model import SysMLLiteModel


# ---------------------------------------------------------------------------
# REQ_ID → ArduPilot 参数映射目录
#
# threshold_slot:
#   指定从哪个 guard 里取阈值。
#   格式: (operator, index)
#     operator: 匹配 GuardCondition.operator（"<", "<=", ">", ">="）
#     index:    同一 part 里同 operator 的第几个 guard（0-based，按阈值升序排）
#   None: 不需要从 guard 提取（静态固定值）
#
# ardu_params:
#   值为数字 → 直接使用
#   值为 "@guard" → 用 threshold_slot 提取到的值
#   值为 "@guard*N" → 提取值乘以 N
# ---------------------------------------------------------------------------

_REQ_CATALOGUE: Dict[str, Dict[str, Any]] = {

    # ── Safety: battery RTB — guard: batteryCharge <= X% ──────────────
    "REQ_SAFE_001": {
        "threshold_slot": ("<=", 0),   # 第一个 <= guard（较大阈值 = RTB）
        "ardu_params": {
            "BATT_FS_LOW_PCT": "@guard",
            "BATT_FS_LOW_ACT": 2,      # 2 = RTL
        },
        "sitl_test": {
            "tier": "L1",
            "inject": None,
            "verify": None,
            "notes": "Static param check — threshold from AST guard.",
        },
    },

    # ── Safety: battery critical landing — guard: batteryCharge < X% ──
    "REQ_SAFE_002": {
        "threshold_slot": ("<", 0),    # 第一个 < guard（较小阈值 = 紧急降落）
        "ardu_params": {
            "BATT_FS_CRT_PCT": "@guard",
            "BATT_FS_CRT_ACT": 1,      # 1 = Land
        },
        "sitl_test": {
            "tier": "L1",
            "inject": None,
            "verify": None,
            "notes": "Static param check — threshold from AST guard.",
        },
    },

    # ── Safety: GCS link-loss — guard: commLossTime > X s ─────────────
    "REQ_SAFE_003": {
        "threshold_slot": (">", 0),
        "ardu_params": {
            "FS_GCS_ENABLE":   1,
            "FS_GCS_TIMEOUT":  "@guard",
        },
        "sitl_test": {
            "tier": "L2",
            "inject": "disconnect_gcs",
            "verify": "wait_mode(LAND, timeout=15.0)",
            "notes": "Disable GCS heartbeat; wait for LAND mode.",
        },
    },

    # ── Safety: startup inhibit on sensor failure ──────────────────────
    "REQ_SAFE_004": {
        "threshold_slot": None,
        "ardu_params": {
            "ARMING_CHECK": 1,
        },
        "sitl_test": {
            "tier": "L2",
            "inject": "param set SIM_GPS_DISABLE 1",
            "verify": "assert_arm_rejected()",
            "notes": "Disable GPS sim, confirm arming is rejected.",
        },
    },

    # ── Safety: parachute deploy — guard: propulsionFailure == True ────
    "REQ_SAFE_005": {
        "threshold_slot": ("bool", 0),   # boolean guard
        "ardu_params": {
            "CHUTE_ENABLED":   1,
            "CHUTE_DELAY_MS":  "@guard_attr*1000",  # parachuteDeployTime 属性
        },
        "sitl_test": {
            "tier": "L2",
            "inject": "param set SIM_ENGINE_FAIL 1",
            "verify": "wait_for_mavlink(MAV_CMD_DO_PARACHUTE, timeout=5.0)",
            "notes": "Timing validated by behavioral_sim; SITL checks existence only.",
        },
    },

    # ── Safety: payload lock on delivery-abort ─────────────────────────
    "REQ_SAFE_006": {
        "threshold_slot": None,
        "ardu_params": {},
        "sitl_test": {
            "tier": "L2",
            "inject": "set_delivery_abort(True)",
            "verify": "assert_gripper_locked()",
            "notes": "Requires Gazebo gripper plugin.",
        },
    },

    # ── Interface: MAVLink v2 ──────────────────────────────────────────
    "REQ_INTF_001": {
        "threshold_slot": None,
        "ardu_params": {
            "SERIAL0_PROTOCOL": 2,
        },
        "sitl_test": {
            "tier": "L1",
            "inject": None,
            "verify": None,
            "notes": "Static param check — MAVLink v2.",
        },
    },

    # ── Constraint: max altitude ───────────────────────────────────────
    "REQ_CONS_001": {
        "threshold_slot": None,
        "ardu_params": {
            "FENCE_ENABLE":  1,
            "FENCE_ALT_MAX": "@attr:maxAltitude",   # 直接读属性（不是 guard）
        },
        "sitl_test": {
            "tier": "L1",
            "inject": None,
            "verify": None,
            "notes": "Static param check — altitude fence.",
        },
    },

    # ── Performance: control loop rate ────────────────────────────────
    "REQ_PERF_006": {
        "threshold_slot": None,
        "ardu_params": {
            "SCHED_LOOP_RATE": "@attr:controlFrequency",
        },
        "sitl_test": {
            "tier": "L1",
            "inject": None,
            "verify": None,
            "notes": "Static param check.",
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResolvedParam:
    req_id: str
    part_name: str
    param_name: str
    value: Any
    source: str = ""       # 描述值的来源，例如 "guard:<=:25.0" 或 "attr:maxAltitude"


@dataclass
class SITLTestSpec:
    req_id: str
    tier: str
    inject: Optional[str]
    verify: Optional[str]
    notes: str
    params: List[ResolvedParam] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RequirementLinker
# ---------------------------------------------------------------------------

class RequirementLinker:
    """
    通过 AST（satisfy_relationships + state_extractor guard）将 SysML
    需求映射到 ArduPilot 参数，不依赖属性名字符串。
    """

    def __init__(self, model: SysMLLiteModel) -> None:
        self._model = model
        # req_id → part_name
        self._satisfy_map: Dict[str, str] = self._build_satisfy_map()
        # part_name → List[GuardCondition]
        self._guard_map: Dict[str, List[Any]] = self._build_guard_map()
        # part_name → {attr_name: float}
        self._attr_map: Dict[str, Dict[str, float]] = self._build_attr_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_parm_file(self) -> str:
        resolved = self._resolve_all()
        lines = [
            "# Auto-generated by requirement_linker.py (AST-based)",
            f"# Source model: {self._model.name}",
            "",
        ]
        seen: Dict[str, str] = {}
        for rp in resolved:
            val = self._format_value(rp.value)
            comment = f"  # {rp.req_id} ({rp.part_name}) [{rp.source}]"
            key = rp.param_name
            if key not in seen:
                lines.append(f"{key:<30} {val}{comment}")
                seen[key] = val
            elif seen[key] != val:
                lines.append(f"# CONFLICT {key}: {seen[key]} vs {val} ({rp.req_id})")
        return "\n".join(lines) + "\n"

    def generate_test_specs(self) -> List[SITLTestSpec]:
        specs: List[SITLTestSpec] = []
        resolved_by_req: Dict[str, List[ResolvedParam]] = {}
        for rp in self._resolve_all():
            resolved_by_req.setdefault(rp.req_id, []).append(rp)

        for req_id in self._covered_req_ids():
            cat = _REQ_CATALOGUE.get(req_id)
            if cat is None:
                continue
            st = cat["sitl_test"]
            specs.append(SITLTestSpec(
                req_id=req_id,
                tier=st["tier"],
                inject=st.get("inject"),
                verify=st.get("verify"),
                notes=st.get("notes", ""),
                params=resolved_by_req.get(req_id, []),
            ))
        return specs

    def coverage_report(self) -> str:
        covered = self._covered_req_ids()
        catalogued = set(_REQ_CATALOGUE.keys())
        matched = covered & catalogued
        unmatched = covered - catalogued

        lines = [
            f"Coverage report — {self._model.name}",
            f"  Satisfied reqs in model : {len(covered)}",
            f"  Mapped to ArduPilot     : {len(matched)}",
            "",
            "Mapped requirements:",
        ]
        for r in sorted(matched):
            tier = _REQ_CATALOGUE[r]["sitl_test"]["tier"]
            lines.append(f"  ✓ {r:<20} [{tier}]")
        if unmatched:
            lines.append("\nSatisfied but no ArduPilot mapping (skipped):")
            for r in sorted(unmatched):
                lines.append(f"  – {r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build maps from AST
    # ------------------------------------------------------------------

    def _build_satisfy_map(self) -> Dict[str, str]:
        """req_id → part_name（满足该需求的部件名）"""
        m: Dict[str, str] = {}
        for part in self._model.part_definitions:
            for sat in part.satisfy_relationships:
                m[sat.target.name] = part.name
        return m

    def _build_guard_map(self) -> Dict[str, List[Any]]:
        """
        part_name → 该 part 的状态机 guard 列表。
        使用 state_extractor 提取，与 behavioral_sim 共享同一 AST 路径。
        """
        from src.simulation.state_extractor import extract_state_machines
        guard_map: Dict[str, List[Any]] = {}
        try:
            # extract_state_machines 接收 SysML 原始文本，不是 LiteModel 对象
            sysml_text = self._model.to_sysml_text()
            if not sysml_text:
                return guard_map
            state_machines = extract_state_machines(sysml_text)
            for sm in state_machines:
                for trans in sm.fault_transitions():
                    guard_map.setdefault(sm.owner_part, []).extend(trans.guards)
        except Exception:
            pass
        return guard_map

    def _build_attr_map(self) -> Dict[str, Dict[str, float]]:
        """part_name → {attr_name: float}，用于 @attr: 直接读属性的情况。"""
        result: Dict[str, Dict[str, float]] = {}
        for part in self._model.part_definitions:
            attrs: Dict[str, float] = {}
            for attr in part.attributes:
                val = self._parse_attr_value(attr)
                if val is not None:
                    attrs[attr.name] = val
            result[part.name] = attrs
        return result

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    def _covered_req_ids(self) -> set:
        return set(self._satisfy_map.keys())

    def _resolve_all(self) -> List[ResolvedParam]:
        results: List[ResolvedParam] = []
        for req_id in self._covered_req_ids():
            cat = _REQ_CATALOGUE.get(req_id)
            if cat is None:
                continue
            part_name = self._satisfy_map[req_id]
            slot = cat.get("threshold_slot")

            # 从 guard AST 提取阈值
            guard_val: Optional[float] = None
            guard_source = ""
            if slot is not None and slot != ("bool", 0):
                op, idx = slot
                guard_val, guard_source = self._extract_guard_threshold(
                    part_name, op, idx
                )

            # REQ_SAFE_005 特殊：CHUTE_DELAY_MS 从属性提取
            chute_delay = self._extract_chute_delay(part_name) if req_id == "REQ_SAFE_005" else None

            for param_name, raw_value in cat["ardu_params"].items():
                if isinstance(raw_value, str) and raw_value.startswith("@guard*"):
                    # "@guard_attr*1000" → 属性值 × 1000
                    val = (chute_delay * 1000) if chute_delay is not None else "<unresolved:chute_delay>"
                    src = f"attr:parachuteDeployTime*1000" if chute_delay else "unresolved"
                elif raw_value == "@guard":
                    val = guard_val if guard_val is not None else f"<unresolved:guard:{slot}>"
                    src = guard_source or "unresolved"
                elif isinstance(raw_value, str) and raw_value.startswith("@attr:"):
                    attr_name = raw_value[6:]
                    val, src = self._extract_attr(part_name, attr_name)
                else:
                    val = raw_value
                    src = "static"

                results.append(ResolvedParam(
                    req_id=req_id,
                    part_name=part_name,
                    param_name=param_name,
                    value=val,
                    source=src,
                ))
        return results

    def _extract_guard_threshold(
        self,
        part_name: str,
        operator: str,
        index: int,
    ) -> Tuple[Optional[float], str]:
        """
        在 part_name 的状态机 guard 里，找第 index 个 operator 匹配的
        guard，返回 (threshold, source_description)。
        按阈值升序排（小的先返回）。
        """
        guards = self._guard_map.get(part_name, [])
        matched = []
        for g in guards:
            if getattr(g, "operator", None) == operator:
                t = getattr(g, "threshold", None)
                if t is not None:
                    matched.append((t, g))

        matched.sort(key=lambda x: x[0])
        if index < len(matched):
            t, g = matched[index]
            attr = getattr(g, "attribute", "?")
            return t, f"guard:{operator}:{t} (attr:{attr})"
        return None, ""

    def _extract_chute_delay(self, part_name: str) -> Optional[float]:
        """
        在 part_name 的属性里，找 parachute/deploy 相关的时间属性。
        使用语义关键词匹配，不依赖精确属性名。
        """
        attrs = self._attr_map.get(part_name, {})
        keywords = ["parachute", "deploy", "chute"]
        for name, val in attrs.items():
            nl = name.lower()
            if any(k in nl for k in keywords) and 0.1 <= val <= 5.0:
                return val
        # fallback: 找值在 [0.1, 5.0] 范围内的时间属性（秒）
        time_keywords = ["time", "delay", "timeout"]
        for name, val in attrs.items():
            nl = name.lower()
            if any(k in nl for k in time_keywords) and 0.1 <= val <= 2.0:
                return val
        return None

    def _extract_attr(
        self,
        part_name: str,
        attr_name: str,
    ) -> Tuple[Any, str]:
        """
        在 part_name 的所有属性里，优先精确名字匹配，
        否则做宽松关键词匹配。
        """
        # 先找与 satisfy 该需求的 part，再向其他 part 扩展
        all_parts = [p for p in self._model.part_definitions if p.name == part_name]
        all_parts += [p for p in self._model.part_definitions if p.name != part_name]

        for part in all_parts:
            attrs = self._attr_map.get(part.name, {})
            # 精确匹配
            if attr_name in attrs:
                return attrs[attr_name], f"attr:{part.name}.{attr_name}"
            # 宽松：attr_name 的关键词子集
            keywords = re.sub(r'([A-Z])', r' \1', attr_name).lower().split()
            for aname, aval in attrs.items():
                aname_lower = re.sub(r'([A-Z])', r' \1', aname).lower()
                if all(k in aname_lower for k in keywords):
                    return aval, f"attr:{part.name}.{aname}(~{attr_name})"

        return f"<unresolved:attr:{attr_name}>", "unresolved"

    def _parse_attr_value(self, attr) -> Optional[float]:
        raw = getattr(attr, "default_value", None)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            if isinstance(raw, str):
                m = re.search(r"-?\d+\.?\d*", raw)
                if m:
                    return float(m.group())
        return None

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
