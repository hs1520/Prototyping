"""
requirement_linker.py

Maps SysML requirements to ArduPilot parameters and SITL test specs.

核心设计原则（重构后）：
  以 guard / attribute 内容作为语义锚点，不依赖 REQ ID 字符串。

  REQ ID 是 LLM 自由命名的标签，跨系统、跨运行不稳定。
  guard 内容（operator + variable keywords + threshold）才是不变的语义。

映射流程：
  part.satisfy(REQ_X)
    → 取该 part 的状态机 guard（e.g. batterySoc <= 25.0）
    → 按 guard 内容（operator=<= + var 含 battery）匹配 _CONTENT_CATALOGUE
    → 得到 ArduPilot 参数（BATT_FS_LOW_PCT = 25.0）
    → REQ ID 仅用于测试脚本的标签，不参与匹配逻辑

五层 Fallback（内容优先）：
  层1  guard/attr 内容匹配（_CONTENT_CATALOGUE）← 最健壮
  层2  AST 合成器（guard 变量名关键词）
  层3  LLM 状态机语义分类
  层3b LLM 需求文本直接推断
  层4  全部失败 → unmapped
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from src.sysml.lite_model import SysMLLiteModel
from src.sitl.sitl_specs import InjectSpec, VerifySpec

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


# ---------------------------------------------------------------------------
# 内容匹配器数据结构
# ---------------------------------------------------------------------------

@dataclass
class GuardMatcher:
    """
    按 guard 内容匹配（与 REQ ID 无关）。

    operators     : 匹配的比较运算符列表；["bool"] 代表 bool_true guard
    var_keywords  : guard 变量名（小写）必须包含其中任一关键词
    threshold_min : 阈值下界（comparison guard）
    threshold_max : 阈值上界（comparison guard）
    action_kws    : 可选，进一步用 entry action 名区分同类 guard（如区分
                    SENSOR_ARMING_INHIBIT vs SENSOR_GROUND_ALERT）
    """
    operators: List[str]
    var_keywords: List[str]
    threshold_min: float = float("-inf")
    threshold_max: float = float("inf")
    action_kws: List[str] = field(default_factory=list)


@dataclass
class AttrMatcher:
    """
    按 part 属性名内容匹配（与 REQ ID 无关）。

    attr_keywords : 属性名（小写）必须包含其中任一关键词
    part_keywords : 可选，part 名（小写）必须包含其中任一关键词（防止跨部件误匹配）
    multiplier    : 单位换算系数（如 km→m 用 1000，m/s→cm/s 用 100）
    """
    attr_keywords: List[str]
    part_keywords: List[str] = field(default_factory=list)
    multiplier: float = 1.0


@dataclass
class ContentEntry:
    """
    内容驱动的 Catalogue 条目。

    匹配键：guard 内容或 attr 内容，而非 REQ ID 字符串。

    ardu_params 值语义：
      "@guard"       → 匹配 guard 的阈值（comparison guard 时）
      "@attr_match"  → 匹配到的属性值（AttrMatcher 时）
      "@attr_match*N"→ 同上 × N
      "@attr:name"   → 按名字关键词搜索的属性（独立于 AttrMatcher）
      "@attr:name*N" → 同上 × N
      number         → 静态固定值
    """
    semantic_tag: str
    guard_matcher: Optional[GuardMatcher] = None
    attr_matcher: Optional[AttrMatcher] = None
    ardu_params: Dict[str, Any] = field(default_factory=dict)
    tier: str = "L1"
    inject: Optional[Any] = None
    verify: Optional[Any] = None
    notes: str = ""


def _noop_l1() -> Dict[str, Any]:
    return {
        "inject": InjectSpec(kind="noop"),
        "verify": VerifySpec(kind="noop"),
    }


# ---------------------------------------------------------------------------
# 内容驱动的 Catalogue（替代旧的 _REQ_CATALOGUE: Dict[str, ...]）
#
# 顺序重要：优先级高（更具体）的条目放前面。
# ---------------------------------------------------------------------------

_CONTENT_CATALOGUE: List[ContentEntry] = [

    # ── Safety: battery RTB（reaches X%）
    #    LLM 对"达到某阈值"倾向于用 <= 运算符
    ContentEntry(
        semantic_tag="BATTERY_RTB",
        guard_matcher=GuardMatcher(
            operators=["<="],
            var_keywords=["battery", "soc", "charge", "volt"],
            # threshold_min=20 区分 RTB（高阈值）vs LAND（低阈值）
            threshold_min=20.0,
        ),
        ardu_params={"BATT_FS_LOW_PCT": "@guard", "BATT_FS_LOW_ACT": 2},
        tier="L1",
        **_noop_l1(),
        notes="Battery-RTB guard (<=) → BATT_FS_LOW_PCT.",
    ),

    # ── Safety: battery critical landing（falls below X%）
    #    LLM 对"低于"倾向于用 < 运算符
    ContentEntry(
        semantic_tag="BATTERY_LAND",
        guard_matcher=GuardMatcher(
            operators=["<", "<="],
            var_keywords=["battery", "soc", "charge", "volt"],
            threshold_max=20.0,  # 低阈值 → critical land
        ),
        ardu_params={"BATT_FS_CRT_PCT": "@guard", "BATT_FS_CRT_ACT": 1},
        tier="L1",
        **_noop_l1(),
        notes="Battery-critical guard (<) → BATT_FS_CRT_PCT.",
    ),

    # ── Safety: GCS link-loss（commLossTime > X s）
    ContentEntry(
        semantic_tag="GCS_LOSS",
        guard_matcher=GuardMatcher(
            operators=[">", ">="],
            var_keywords=["comm", "gcs", "link", "heartbeat", "loss"],
        ),
        ardu_params={"FS_GCS_ENABLE": 1, "FS_GCS_TIMEOUT": "@guard"},
        tier="L2",
        inject=InjectSpec(kind="disconnect_gcs", pre_takeoff_m=5.0),
        verify=VerifySpec(
            kind="wait_mode",
            args={"mode": "LAND", "fallback": "RTL"},
            timeout=25.0,
        ),
        notes="GCS timeout guard → FS_GCS_TIMEOUT + L2 disconnect test.",
    ),

    # ── Safety: sensor arming inhibit（bool: sensorSelfTestFailed）
    #    action_kws 区分 "inhibit arming" vs "ground alert"
    ContentEntry(
        semantic_tag="SENSOR_ARMING_INHIBIT",
        guard_matcher=GuardMatcher(
            operators=["bool"],
            var_keywords=["sensor", "selftest", "post", "prearm"],
            action_kws=["inhibit", "arm", "block", "prevent"],
        ),
        ardu_params={"ARMING_CHECK": 1},
        tier="L2",
        inject=InjectSpec(
            kind="set_param",
            params={"GPS_TYPE": 0.0, "_settle_s": 5.0, "_pre_mode": "GUIDED"},
        ),
        verify=VerifySpec(kind="assert_arm_rejected", timeout=6.0),
        notes="Sensor self-test failure → ARMING_CHECK; verify arm rejected.",
    ),

    # ── Safety: sensor ground alert（bool: sensorSelfTestFailed，但 action 是 alert/report）
    ContentEntry(
        semantic_tag="SENSOR_GROUND_ALERT",
        guard_matcher=GuardMatcher(
            operators=["bool"],
            var_keywords=["sensor", "selftest", "post", "prearm"],
            action_kws=["alert", "report", "transmit", "notify", "ground"],
        ),
        ardu_params={},
        tier="L2",
        inject=InjectSpec(
            kind="set_param",
            params={"GPS_TYPE": 0.0, "_settle_s": 3.0, "_pre_mode": "GUIDED"},
        ),
        verify=VerifySpec(
            kind="assert_sensor_unhealthy",
            args={"sensor": "gps"},
            timeout=12.0,
        ),
        notes="Sensor failure → SYS_STATUS GPS health bit cleared.",
    ),

    # ── Safety: parachute deploy（bool: propulsionCriticalFailure）
    ContentEntry(
        semantic_tag="PARACHUTE_DEPLOY",
        guard_matcher=GuardMatcher(
            operators=["bool"],
            var_keywords=["propulsion", "engine", "motor", "thrust"],
        ),
        ardu_params={
            "CHUTE_ENABLED": 1,
            "CHUTE_TYPE": 10,          # 10 = servo-released parachute
            "CHUTE_DELAY_MS": "@attr:parachute*1000",
            # SERVO8=Parachute → Gazebo channel 7（避开 gimbal 占用的 SERVO9-11）
            "SERVO8_FUNCTION": 27,     # 27 = k_parachute
            "CHUTE_SERVO_ON": 2000,    # PWM 2000 → COMMAND 归一化 1.0 → ParachutePlugin(>0.9) 部署
            "CHUTE_SERVO_OFF": 1000,   # PWM 1000 → 归一化 0.0（安全位）
        },
        tier="L2",
        inject=InjectSpec(
            kind="mavlink_command",
            # MAV_CMD_DO_PARACHUTE (208): param1=2 → RELEASE
            # 直接命令释放；verify 读 SERVO8 PWM，不依赖 STATUSTEXT。
            # （SIM_ENGINE_FAIL 只断电机推力，不触发坠毁检测）
            params={"command": 208, "param1": 2, "_settle_s": 0.5},
            pre_takeoff_m=10.0,
        ),
        verify=VerifySpec(
            kind="assert_servo_pwm",
            args={"channel": 8, "target_pwm": 2000, "tol": 50},
            timeout=15.0,
        ),
        notes="MAV_CMD_DO_PARACHUTE RELEASE → SERVO_OUTPUT_RAW.servo8_raw≈2000.",
    ),

    # ── Safety: payload abort lock（bool: deliveryAbortConditionActive）
    ContentEntry(
        semantic_tag="PAYLOAD_ABORT_LOCK",
        guard_matcher=GuardMatcher(
            operators=["bool"],
            var_keywords=["payload", "abort", "delivery", "lock", "gripper"],
        ),
        ardu_params={
            "GRIP_ENABLE": 1,
            "GRIP_TYPE": 1,           # 1 = servo gripper
            # SERVO7=Gripper → Gazebo channel 6（避开 gimbal 占用的 SERVO9-11/channel 8-10）
            "SERVO7_FUNCTION": 28,    # 28 = Gripper
            "GRIP_RELEASE": 2000,     # PWM 2000 → COMMAND 归一化 1.0 → TriggeredPublisher 触发
            "GRIP_GRAB": 1000,        # PWM 1000 → 归一化 0.0（不触发）
        },
        tier="L2",
        inject=InjectSpec(
            kind="mavlink_command",
            # MAV_CMD_DO_GRIPPER (211): param1=gripper_id(0), param2=action(1=RELEASE)
            params={"command": 211, "param1": 0, "param2": 1, "_settle_s": 1.5},
            pre_takeoff_m=5.0,
        ),
        verify=VerifySpec(
            kind="assert_servo_pwm",
            args={"channel": 7, "target_pwm": 2000, "tol": 50},
            timeout=10.0,
        ),
        notes="Payload abort → MAV_CMD_DO_GRIPPER → SERVO_OUTPUT_RAW.servo7_raw≈2000.",
    ),

    # ── Constraint: max altitude（attr: maxAltitude）
    ContentEntry(
        semantic_tag="ALTITUDE_FENCE",
        attr_matcher=AttrMatcher(
            attr_keywords=["maxaltitude", "altitude", "maxalt", "alt"],
            part_keywords=["flight", "controller", "autopilot"],
        ),
        ardu_params={"FENCE_ENABLE": 1, "FENCE_ALT_MAX": "@attr_match"},
        tier="L1",
        **_noop_l1(),
        notes="maxAltitude attribute → FENCE_ALT_MAX.",
    ),

    # ── Performance: control loop rate（attr: controlFrequency）
    ContentEntry(
        semantic_tag="CONTROL_LOOP_RATE",
        attr_matcher=AttrMatcher(
            attr_keywords=["controlfreq", "looprate", "controlloop", "frequency",
                           "freq", "rate"],
            part_keywords=["flight", "controller", "autopilot"],
        ),
        ardu_params={"SCHED_LOOP_RATE": "@attr_match"},
        tier="L1",
        **_noop_l1(),
        notes="controlFrequency attribute → SCHED_LOOP_RATE.",
    ),

    # ── Interface: MAVLink protocol（attr: encryptionKeyLength 或 part 名含 comm）
    ContentEntry(
        semantic_tag="MAVLINK_PROTOCOL",
        attr_matcher=AttrMatcher(
            attr_keywords=["encryption", "protocol", "mavlink", "keylength",
                           "serial", "maxoperational"],
            part_keywords=["communication", "comm", "link", "gcs"],
        ),
        ardu_params={"SERIAL0_PROTOCOL": 2},
        tier="L1",
        **_noop_l1(),
        notes="CommunicationSystem → SERIAL0_PROTOCOL=2 (MAVLink v2).",
    ),

    # ── Performance: max airspeed（attr: maxAirspeed, unit m/s → cm/s ×100）
    ContentEntry(
        semantic_tag="MAX_SPEED",
        attr_matcher=AttrMatcher(
            attr_keywords=["maxairspeed", "airspeed", "maxspeed"],
            multiplier=100.0,   # m/s → cm/s
        ),
        ardu_params={"WPNAV_SPEED": "@attr_match"},
        tier="L1",
        **_noop_l1(),
        notes="maxAirspeed attribute (m/s) × 100 → WPNAV_SPEED (cm/s).",
    ),

    # ── Constraint: operational radius（attr: maxOperationalRadius, unit km → m ×1000）
    ContentEntry(
        semantic_tag="RADIUS_FENCE",
        attr_matcher=AttrMatcher(
            attr_keywords=["maxoperationalradius", "operationalradius",
                           "maxradius", "radius"],
            multiplier=1000.0,  # km → m
        ),
        ardu_params={"FENCE_RADIUS": "@attr_match"},
        tier="L1",
        **_noop_l1(),
        notes="maxOperationalRadius (km) × 1000 → FENCE_RADIUS (m).",
    ),

    # ── Interface: RTCM GNSS corrections（attr: 含 gps/gnss 的 part）
    ContentEntry(
        semantic_tag="RTCM_GPS",
        attr_matcher=AttrMatcher(
            attr_keywords=["gnss", "gps", "correction", "rtcm", "differential"],
            part_keywords=["perception", "navigation", "gnss", "gps", "sensor"],
        ),
        ardu_params={"GPS_INJECT_TO": 127},
        tier="L1",
        **_noop_l1(),
        notes="RTCM/GNSS part → GPS_INJECT_TO=127 (broadcast corrections).",
    ),
]


# ---------------------------------------------------------------------------
# 按 semantic_tag 索引（供 Layer 2 AST 合成 / Layer 3 LLM 语义标签使用）
# ---------------------------------------------------------------------------

_TAG_TO_ENTRY: Dict[str, ContentEntry] = {
    e.semantic_tag: e for e in _CONTENT_CATALOGUE
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
    inject: InjectSpec
    verify: VerifySpec
    notes: str
    params: List[ResolvedParam] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RequirementLinker
# ---------------------------------------------------------------------------

class RequirementLinker:
    """
    通过 AST（satisfy_relationships + state_extractor guard）将 SysML
    需求映射到 ArduPilot 参数，不依赖属性名字符串。

    可选传入 llm 启用语义标签分类，使得不在 _REQ_CATALOGUE key 里的 req ID
    也能通过状态机语义找到匹配的模板（三层 fallback：req ID → 语义标签 →
    unmapped）。
    """

    def __init__(
        self,
        model: SysMLLiteModel,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self._model = model
        self._llm = llm
        self._verbose = verbose
        # req_id → part_name
        self._satisfy_map: Dict[str, List[str]] = self._build_satisfy_map()
        # part_name → List[GuardCondition]
        self._guard_map: Dict[str, List[Any]] = self._build_guard_map()
        # part_name → {attr_name: float}
        self._attr_map: Dict[str, Dict[str, float]] = self._build_attr_map()
        # semantic_tag → ContentEntry（供 Layer 2/3 反查）
        self._tag_to_entry: Dict[str, ContentEntry] = _TAG_TO_ENTRY
        # part_name → List[SynthesizedSpec]（AST 合成，主层）
        self._ast_specs = self._build_ast_specs()
        # req_id → tag（LLM 状态机语义分类，第三层 fallback）
        self._semantic_map: Dict[str, str] = self._build_semantic_map()
        # req_id → {param, value, tier}（LLM 需求文本直接推断，第四层 fallback）
        self._direct_param_map: Dict[str, Dict] = self._build_direct_param_map()
        # req_id → (part_name, guard, ContentEntry)：独占 guard 分配
        # 同一 guard 只能分配给一个 req，防止多 REQ 共享 part 时全部映射到同一 guard
        self._guard_assignment: Dict[str, Any] = self._assign_guards_exclusive()
        # _lookup_catalogue 结果缓存：避免 _resolve_all 多次调用时重复打印 [CONTENT]
        self._catalogue_cache: Dict[str, Optional[Dict]] = {}

    def _assign_guards_exclusive(self) -> Dict[str, Any]:
        """
        预分配：将 part 的每个 guard 独占地分配给一个 req。

        问题根因：多个 REQ 共享同一 part（如 SafetyMonitor satisfy 5 个 SAFE REQ）时，
        若不加独占控制，对每个 REQ 调用 _match_by_content 都会返回相同的第一个匹配 guard，
        导致 BATT_FS_LOW_PCT=25 被写多次而 BATT_FS_CRT_PCT=15 从未被写。

        算法：
          按 req_id 排序（确定性），对每个 req 贪心地分配第一个未被占用的 guard。
          返回 {req_id: {"part": str, "guard": GuardCondition, "entry": ContentEntry}}
          或    {req_id: None}（无 guard 可分配）
        """
        assignment: Dict[str, Any] = {}
        claimed: set = set()  # (part_name, guard_attr, guard_op)

        for req_id in sorted(self._covered_req_ids()):
            part_names = self._satisfy_map.get(req_id, [])
            found = False
            for entry in _CONTENT_CATALOGUE:
                if entry.guard_matcher is None:
                    continue
                gm = entry.guard_matcher
                for pname in part_names:
                    for guard in self._guard_map.get(pname, []):
                        if not self._guard_satisfies(guard, gm, pname):
                            continue
                        gkey = (pname,
                                getattr(guard, "attribute", ""),
                                getattr(guard, "operator", ""))
                        if gkey in claimed:
                            continue
                        # 成功分配
                        claimed.add(gkey)
                        assignment[req_id] = {
                            "part":  pname,
                            "guard": guard,
                            "entry": entry,
                        }
                        found = True
                        break
                    if found:
                        break
                if found:
                    break

        return assignment

    def _build_ast_specs(self):
        """
        用 AST 合成器从 guard 变量名 + entry action 名直接推导
        InjectSpec/VerifySpec，建立 part_name → List[SynthesizedSpec] 索引。

        不依赖 LLM，<1ms，Phase 2 标准命名后接近 100% 覆盖。
        """
        try:
            from src.sitl.ast_synthesizer import synthesize_specs
            return synthesize_specs(self._model, verbose=self._verbose)
        except Exception as e:
            if self._verbose:
                print(f"  [AST-SYN] failed: {e}")
            return {}

    def _build_semantic_map(self) -> Dict[str, str]:
        """
        用 LLM 给状态机打语义标签，建立 req_id → tag 映射。

        核心逻辑：一个 part 可能有多个状态机（多个安全行为），不能简单地
        用 part_name → tag（后写的会覆盖前面）。正确做法是把每个 req 的
        guard 变量与该 part 的状态机 guard 做交叉匹配，选最接近的 tag。
        """
        if self._llm is None:
            return {}
        try:
            from src.sitl.semantic_classifier import classify_state_machines
        except ImportError:
            return {}
        try:
            sm_to_tag = classify_state_machines(
                self._model, self._llm, verbose=self._verbose
            )
        except Exception as e:
            if self._verbose:
                print(f"  [SEMANTIC] classify failed: {e}")
            return {}

        # 建立 state_def_name → (tag, owner_part, guard_vars) 的详细表
        try:
            from src.simulation.state_extractor import extract_state_machines
            text = self._model.to_sysml_text() or ""
            sms = extract_state_machines(text)
        except Exception:
            sms = []

        # sm_name → (owner_part, guard 涉及的所有属性名)
        sm_detail: Dict[str, Tuple[str, List[str]]] = {}
        for sm in sms:
            guard_attrs: List[str] = []
            for t in sm.fault_transitions():
                for g in t.guards:
                    attr = getattr(g, "attribute", None)
                    if attr:
                        guard_attrs.append(attr.lower())
            sm_detail[sm.name] = (sm.owner_part, guard_attrs)

        # 为每个 req 找最匹配的 tag
        # 策略：req 满足的所有 part 中，找 guard 变量与该 req 最接近的 sm
        req_tag: Dict[str, str] = {}
        for req_id in self._satisfy_map:
            part_names = self._satisfy_map[req_id]  # 现在是 list

            # 收集所有满足该 REQ 的 part 下的候选 sm
            candidate_sms = [
                (sm_name, sm_to_tag[sm_name], detail)
                for sm_name, detail in sm_detail.items()
                if detail[0] in part_names and sm_name in sm_to_tag
            ]
            if not candidate_sms:
                continue

            # 如果只有一个 sm，直接用
            if len(candidate_sms) == 1:
                req_tag[req_id] = candidate_sms[0][1]
                continue

            # 多个 sm：用 req_id 关键词与 sm guard 属性做最佳匹配
            # 把 req_id 拆成小写词（REQ_FUNC_007 → ["func","007"]）
            req_tokens = set(req_id.lower().replace("_", " ").split())

            best_sm, best_tag, best_score = None, None, -1
            for sm_name, tag, (owner, guard_attrs) in candidate_sms:
                # 先用 tag 关键词打分
                tag_tokens = set(tag.lower().replace("_", " ").split())
                score = len(req_tokens & tag_tokens & set(
                    ["battery","batt","gcs","comm","link","sensor","parachute",
                     "propulsion","engine","payload","abort"]
                ))
                # 如果 guard 属性明确出现在 req_id 里，加分
                for attr in guard_attrs:
                    if any(tok in attr for tok in req_tokens):
                        score += 2
                if score > best_score:
                    best_score, best_sm, best_tag = score, sm_name, tag

            # 只有打到分才采用，否则不误判
            if best_tag and best_score > 0:
                req_tag[req_id] = best_tag

        return req_tag

    def _build_direct_param_map(self) -> Dict[str, Dict]:
        """
        Layer 3b（第四层 fallback）：用 LLM 从需求文本直接推断 ArduPilot 参数。

        只处理层1-3均未命中的需求（FUNC/PERF/CONS/INTF 类）。
        输入：需求文本（从 SysML doc 注释提取）。
        输出：{req_id: {param: str, value: float, tier: str}}。
        """
        if self._llm is None:
            return {}
        try:
            from src.sitl.semantic_classifier import (
                suggest_params_from_requirements,
                extract_req_texts_from_model,
            )
        except ImportError:
            return {}

        # 找出层1-3均未命中的 req（有 satisfy 关系但无 catalogue/AST/LLM 匹配）
        unmapped_reqs: Dict[str, str] = {}
        all_req_texts = extract_req_texts_from_model(self._model)

        for req_id in self._covered_req_ids():
            # 层2：AST 合成命中
            parts = self._satisfy_map.get(req_id, [])
            ast_hit = any(self._ast_specs.get(p) for p in parts)
            if ast_hit:
                continue
            # 层3：LLM 语义标签命中
            if req_id in self._semantic_map:
                continue
            # 三层全未命中 → 送给 Layer 3b
            text = all_req_texts.get(req_id, "")
            if text:
                unmapped_reqs[req_id] = text

        if not unmapped_reqs:
            return {}

        result = suggest_params_from_requirements(
            unmapped_reqs, self._llm, verbose=self._verbose
        )
        return result

    # ------------------------------------------------------------------
    # 内容匹配核心方法（Layer 1 — 完全不依赖 REQ ID）
    # ------------------------------------------------------------------

    def _entry_to_dict(self, entry: ContentEntry,
                       guard_val: Optional[float] = None,
                       guard_src: str = "",
                       attr_val: Optional[float] = None,
                       attr_src: str = "") -> Dict[str, Any]:
        """
        将 ContentEntry 转换为 _resolve_all 期望的 dict 格式。

        guard_val / attr_val：匹配时已解析的数值，存入 _resolved_* 键，
        供 _resolve_all 跳过 threshold_slot 机制直接使用。
        """
        return {
            "semantic_tag":        entry.semantic_tag,
            "threshold_slot":      None,          # 内容匹配不再需要
            "ardu_params":         entry.ardu_params,
            "_resolved_guard_val": guard_val,
            "_resolved_guard_src": guard_src,
            "_resolved_attr_val":  attr_val,
            "_resolved_attr_src":  attr_src,
            "_attr_multiplier":    entry.attr_matcher.multiplier
                                   if entry.attr_matcher else 1.0,
            "sitl_test": {
                "tier":    entry.tier,
                "inject":  entry.inject or InjectSpec(kind="noop"),
                "verify":  entry.verify or VerifySpec(kind="noop"),
                "notes":   entry.notes,
            },
        }

    def _guard_satisfies(self, guard, gm: GuardMatcher,
                          part_name: str) -> bool:
        """判断一条 guard 是否匹配 GuardMatcher。"""
        kind = getattr(guard, "kind", "")
        var  = getattr(guard, "attribute", "").lower()

        if "bool" in gm.operators:
            # bool_true guard
            if kind != "bool_true":
                return False
            if not any(kw in var for kw in gm.var_keywords):
                return False
            # 可选：检查 entry action 关键词
            if gm.action_kws:
                action = self._find_guard_action(part_name, guard)
                if not action or not any(
                    kw in action.lower() for kw in gm.action_kws
                ):
                    return False
            return True

        # comparison guard
        if kind != "comparison":
            return False
        op = getattr(guard, "operator", "")
        if op not in gm.operators:
            return False
        if not any(kw in var for kw in gm.var_keywords):
            return False
        th = getattr(guard, "threshold", 0.0)
        if not (gm.threshold_min <= th <= gm.threshold_max):
            return False
        return True

    def _find_guard_action(self, part_name: str, guard) -> Optional[str]:
        """
        在 part 的状态机里找到包含该 guard 的转移，返回其 target state 的
        entry action 名。用于 GuardMatcher.action_kws 区分语义。
        """
        try:
            from src.simulation.state_extractor import extract_state_machines
            text = self._model.to_sysml_text() or ""
            for sm in extract_state_machines(text):
                if sm.owner_part != part_name:
                    continue
                for tr in sm.fault_transitions():
                    for g in tr.guards:
                        if (getattr(g, "attribute", "") ==
                                getattr(guard, "attribute", "") and
                                getattr(g, "operator", "") ==
                                getattr(guard, "operator", "")):
                            return sm.entry_action_for_state(tr.target or "")
        except Exception:
            pass
        return None

    def _match_by_content(self, req_id: str) -> Optional[Dict[str, Any]]:
        """
        Layer 1（内容驱动）：按 guard/attr 内容匹配，与 REQ ID 完全无关。

        guard 匹配优先使用 _guard_assignment 预分配结果（独占，防止多 REQ
        抢占同一 guard）。attr 匹配仍在运行时动态进行（属性不存在争用问题）。

        返回 _entry_to_dict(entry, ...) 格式的 dict，或 None。
        """
        part_names = self._satisfy_map.get(req_id)
        if not part_names:
            return None

        # ── Guard 匹配：使用独占预分配结果 ────────────────────────────
        assigned = getattr(self, "_guard_assignment", {}).get(req_id)
        if assigned is not None:
            pname = assigned["part"]
            guard = assigned["guard"]
            entry = assigned["entry"]
            th    = getattr(guard, "threshold", None)
            attr  = getattr(guard, "attribute", "?")
            op    = getattr(guard, "operator", "?")
            g_val = float(th) if th is not None else None
            g_src = f"guard:{op}:{th} (attr:{attr})"
            if self._verbose:
                print(f"  [CONTENT] {req_id} → {entry.semantic_tag} "
                      f"guard={attr!r} {op} {th} (part={pname})")
            return self._entry_to_dict(entry, guard_val=g_val, guard_src=g_src)

        # ── Attr 匹配：动态搜索（属性不存在争用）─────────────────────
        for entry in _CONTENT_CATALOGUE:
            if entry.attr_matcher is None:
                continue
            am = entry.attr_matcher
            for pname in part_names:
                if am.part_keywords and not any(
                    kw in pname.lower() for kw in am.part_keywords
                ):
                    continue
                attrs = self._attr_map.get(pname, {})
                for aname, aval in attrs.items():
                    if not any(kw in aname.lower() for kw in am.attr_keywords):
                        continue
                    resolved = (round(aval * am.multiplier)
                                if am.multiplier != 1.0 else aval)
                    src = (f"attr:{pname}.{aname}"
                           + (f"*{int(am.multiplier)}"
                              if am.multiplier != 1.0 else ""))
                    if self._verbose:
                        print(f"  [CONTENT] {req_id} → {entry.semantic_tag} "
                              f"attr={pname}.{aname}={aval}"
                              f"{'×'+str(int(am.multiplier)) if am.multiplier!=1.0 else ''}"
                              f"={resolved}")
                    return self._entry_to_dict(
                        entry, attr_val=resolved, attr_src=src
                    )

        return None

    # ------------------------------------------------------------------

    def _lookup_catalogue(self, req_id: str) -> Optional[Dict[str, Any]]:
        """
        五层 Fallback（内容优先，REQ ID 无关）：

          层1  _CONTENT_CATALOGUE 内容匹配（guard/attr）← 新，最健壮
          层2  AST 合成器（guard 变量名关键词）
          层3  LLM 状态机语义分类
          层3b LLM 需求文本直接推断
          层4  全部失败 → None
        """
        if req_id in self._catalogue_cache:
            return self._catalogue_cache[req_id]

        part_names = self._satisfy_map.get(req_id)
        if not part_names:
            self._catalogue_cache[req_id] = None
            return None

        result: Optional[Dict[str, Any]] = None

        # ── 层1：内容匹配（_CONTENT_CATALOGUE）────────────────────────
        result = self._match_by_content(req_id)

        # ── 层2：AST 合成（遍历所有 satisfying parts）──────────────────
        if result is None:
            for pname in part_names:
                ast_candidates = self._ast_specs.get(pname, [])
                if not ast_candidates:
                    continue
                best = self._best_ast_spec(req_id, pname, ast_candidates)
                if best is not None:
                    base_entry = self._tag_to_entry.get(best.tag)
                    if base_entry:
                        # Resolve the threshold from the guard the AST synthesizer
                        # matched (best.guard_var), so a `@guard` placeholder is
                        # filled here rather than leaking as <unresolved:guard>
                        # even though the model contains the guard.
                        g_val, g_src = self._threshold_for_guard_var(
                            pname, best.guard_var,
                            base_entry.guard_matcher.operators
                            if base_entry.guard_matcher else None,
                        )
                        if self._verbose:
                            print(f"  [AST-SYN] {req_id} → tag={best.tag} "
                                  f"guard={best.guard_var!r} (part={pname})"
                                  + (f" thr={g_val}" if g_val is not None else ""))
                        d = self._entry_to_dict(
                            base_entry, guard_val=g_val, guard_src=g_src
                        )
                        # AST fallback may infer a useful inject from guard/action names,
                        # but S4 fixed these catalogue tags to deterministic MAVLink
                        # conformance checks. Do not let older AST wait_statustext
                        # templates reintroduce run-to-run flaky verification.
                        verify = best.verify
                        if (
                            base_entry.verify is not None
                            and base_entry.verify.kind in {
                                "assert_servo_pwm",
                                "assert_sensor_unhealthy",
                                "assert_arm_rejected",
                                "wait_mode",
                            }
                        ):
                            verify = base_entry.verify
                        d["sitl_test"] = {
                            "tier":   base_entry.tier,
                            "inject": best.inject,
                            "verify": verify,
                            "notes":  base_entry.notes,
                        }
                        result = d
                        break

        # ── 层3：LLM 状态机语义标签 ────────────────────────────────────
        if result is None:
            tag = self._semantic_map.get(req_id)
            if tag and tag in self._tag_to_entry:
                if self._verbose:
                    print(f"  [SEMANTIC] {req_id} → tag={tag} (parts={part_names})")
                result = self._entry_to_dict(self._tag_to_entry[tag])

        # ── 层3b：LLM 需求文本直接推断 ─────────────────────────────────
        if result is None:
            direct = self._direct_param_map.get(req_id)
            if direct:
                param_name = direct["param"]
                value      = direct["value"]
                tier       = direct.get("tier", "L1")
                if self._verbose:
                    print(f"  [REQ-PARAM] {req_id} → {param_name}={value} [{tier}]")
                result = {
                    "semantic_tag":        f"LLM_DIRECT:{param_name}",
                    "threshold_slot":      None,
                    "ardu_params":         {param_name: value},
                    "_resolved_guard_val": None,
                    "_resolved_attr_val":  None,
                    "sitl_test": {
                        "tier":   tier,
                        "inject": InjectSpec(kind="noop"),
                        "verify": VerifySpec(kind="noop"),
                        "notes":  "LLM-suggested param from requirement text.",
                    },
                }

        self._catalogue_cache[req_id] = result
        return result

    def _best_ast_spec(self, req_id: str, part_name: str, candidates):
        """
        从同一 part 的多个 AST 合成结果中，选与该 req 的 guard 最匹配的那个。
        利用 _guard_map 里该 part 的 guard 属性集合做交叉比对。
        """
        if len(candidates) == 1:
            return candidates[0]

        # 用该 req part 的 guard 属性名做匹配
        part_guards = self._guard_map.get(part_name, [])
        part_guard_attrs = {
            getattr(g, "attribute", "").lower()
            for g in part_guards
        }

        best, best_score = None, -1
        for cand in candidates:
            score = 0
            if cand.guard_var.lower() in part_guard_attrs:
                score += 3   # guard 变量直接匹配
            # req_id 词元 vs candidate tag 词元
            req_tokens = set(req_id.lower().replace("_", " ").split())
            tag_tokens = set(cand.tag.lower().replace("_", " ").split())
            score += len(req_tokens & tag_tokens)
            if score > best_score:
                best_score, best = score, cand

        return best if best_score >= 0 else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # 这些参数在 SITL 里设置会破坏仿真（如 GPS、EKF），
    # 只做 L1 验证（检 resolved 值），不写进 .parm 文件
    _SITL_SKIP_PARAMS: set = {
        "SCHED_LOOP_RATE",  # 改调度频率会破坏 SITL GPS 仿真，让 SITL 用默认 400Hz
    }

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
            # 跳过未解析的值（ArduCopter 解析到非数字会行为异常）
            if isinstance(rp.value, str) and rp.value.startswith("<"):
                lines.append(f"# SKIPPED {rp.param_name}: {val}  [{rp.source}]")
                continue
            if isinstance(rp.value, str) and "@" in rp.value:
                lines.append(f"# SKIPPED {rp.param_name}: {val}  [{rp.source}]")
                continue
            # 跳过 SITL 不兼容参数（L1 仍验证 resolved 值，但不写进 .parm）
            if rp.param_name in self._SITL_SKIP_PARAMS:
                lines.append(f"# SITL-SKIP {rp.param_name}: {val}  (L1-validated, not loaded into SITL)")
                continue
            comment = f"  # {rp.req_id} ({rp.part_name}) [{rp.source}]"
            key = rp.param_name
            if key not in seen:
                lines.append(f"{key:<30} {val}{comment}")
                seen[key] = val
            elif seen[key] != val:
                lines.append(f"# CONFLICT {key}: {seen[key]} vs {val} ({rp.req_id})")

        # CHUTE_ALT_MIN=0：测试时不限制降落伞触发高度
        if "CHUTE_ENABLED" in seen and "CHUTE_ALT_MIN" not in seen:
            lines.append(f"{'CHUTE_ALT_MIN':<30} 0  # SITL test: disable alt threshold")

        return "\n".join(lines) + "\n"

    def generate_test_specs(self) -> List[SITLTestSpec]:
        specs: List[SITLTestSpec] = []
        resolved_by_req: Dict[str, List[ResolvedParam]] = {}
        for rp in self._resolve_all():
            resolved_by_req.setdefault(rp.req_id, []).append(rp)

        for req_id in self._covered_req_ids():
            cat = self._lookup_catalogue(req_id)
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

    # ------------------------------------------------------------------
    # SITL → LLM feedback
    # ------------------------------------------------------------------

    def unresolved_feedback(self) -> List[Dict[str, Any]]:
        """Turn every unresolved SITL parameter into an actionable model-fix
        instruction for the design LLM.

        After the AST-synthesis threshold fix, an unresolved parameter is a
        trustworthy "model defect" signal: the requirement matched a catalogue
        tag, but the model genuinely lacks the guard/attribute the tag needs to
        supply a value.  Each item names the satisfying part, what element is
        missing, and which ArduPilot parameter depends on it.

        Returns a list of dicts: {req_id, tag, kind, part, param, message}.
        """
        items: List[Dict[str, Any]] = []
        for spec in self.generate_test_specs():
            cat = self._lookup_catalogue(spec.req_id)
            tag = (cat or {}).get("semantic_tag", "")
            entry = self._tag_to_entry.get(tag)
            parts = self._satisfy_map.get(spec.req_id, [])
            part = parts[0] if parts else "<the satisfying part>"
            for p in spec.params:
                if not (isinstance(p.value, str) and p.value.startswith("<unresolved")):
                    continue
                kind = p.value[len("<unresolved:"):].rstrip(">")
                items.append({
                    "req_id":  spec.req_id,
                    "tag":     tag,
                    "kind":    kind,
                    "part":    part,
                    "param":   p.param_name,
                    "message": self._unresolved_message(
                        spec.req_id, tag, entry, kind, part, p.param_name
                    ),
                })
        return items

    @staticmethod
    def _unresolved_message(req_id, tag, entry, kind, part, param_name) -> str:
        """State what the model is missing and why it matters — WITHOUT
        prescribing SysML syntax.

        The generation prompt already teaches canonical SysML v2 transition/
        attribute syntax; re-teaching it here is redundant and risky (a
        hand-written fragment that drifts from the canonical form would
        actively mislead the LLM, like the earlier `readonly` mistake).  So the
        feedback gives only semantic facts — which part, what quantity must be
        monitored/declared, which ArduPilot parameter depends on it — and lets
        the LLM apply its own (prompt-grounded, syntax-gate-validated) code.
        """
        gm = getattr(entry, "guard_matcher", None) if entry else None
        am = getattr(entry, "attr_matcher", None) if entry else None

        if kind.startswith("guard") and gm is not None:
            kws = "/".join(gm.var_keywords[:3]) or "the monitored quantity"
            return (
                f"{req_id} ({tag}): the part `{part}` that satisfies this "
                f"requirement defines no state-machine guard that compares a "
                f"{kws} variable against a numeric threshold. ArduPilot parameter "
                f"{param_name} is derived from that threshold, so it cannot be "
                f"set. Add the missing threshold-based guard."
            )

        if kind.startswith("attr") or kind == "chute_delay":
            if am is not None and am.attr_keywords:
                kws = "/".join(am.attr_keywords[:3])
            elif kind.startswith("attr:"):
                kws = kind.split(":", 1)[1]
            elif kind == "chute_delay":
                kws = "parachute deploy-time"
            else:
                kws = "the required"
            return (
                f"{req_id} ({tag}): the part `{part}` that satisfies this "
                f"requirement declares no numeric attribute representing the "
                f"{kws} value. ArduPilot parameter {param_name} is derived from "
                f"it, so it cannot be set. Add the missing attribute."
            )

        return (
            f"{req_id} ({tag}): parameter {param_name} is unresolved — the model "
            f"is missing the guard/attribute it maps from on part `{part}`."
        )

    def coverage_report(self) -> str:
        covered = self._covered_req_ids()
        matched_content, matched_ast, matched_llm, matched_direct = (
            set(), set(), set(), set()
        )

        for r in covered:
            entry = self._lookup_catalogue(r)
            if entry is None:
                continue
            tag = entry.get("semantic_tag", "")
            if tag.startswith("LLM_DIRECT:"):
                matched_direct.add(r)
            elif r in self._semantic_map:
                matched_llm.add(r)
            elif any(self._ast_specs.get(p) for p in self._satisfy_map.get(r, [])):
                matched_ast.add(r)
            else:
                matched_content.add(r)

        matched   = matched_content | matched_ast | matched_llm | matched_direct
        unmatched = covered - matched

        lines = [
            f"Coverage report — {self._model.name}",
            f"  Satisfied reqs in model  : {len(covered)}",
            f"  Mapped via content match : {len(matched_content)}",
            f"  Mapped via AST synth     : {len(matched_ast)}",
            f"  Mapped via LLM tag       : {len(matched_llm)}",
            f"  Mapped via LLM req-text  : {len(matched_direct)}",
            f"  Mapped total             : {len(matched)}",
            "",
            "Mapped requirements:",
        ]
        for r in sorted(matched_content):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            tag  = entry.get("semantic_tag", "?") if entry else "?"
            lines.append(f"  ✓ {r:<22} [{tier}] (content:{tag})")
        for r in sorted(matched_ast):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            lines.append(f"  ✓ {r:<22} [{tier}] (ast-synth)")
        for r in sorted(matched_llm):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            tag  = self._semantic_map.get(r, "?")
            lines.append(f"  ✓ {r:<22} [{tier}] (llm:{tag})")
        for r in sorted(matched_direct):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            param = entry.get("semantic_tag", "?").replace("LLM_DIRECT:", "") if entry else "?"
            lines.append(f"  ✓ {r:<22} [{tier}] (llm-direct:{param})")
        if unmatched:
            lines.append("\nSatisfied but no mapping (skipped):")
            for r in sorted(unmatched):
                lines.append(f"  – {r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build maps from AST
    # ------------------------------------------------------------------

    def _build_satisfy_map(self) -> Dict[str, List[str]]:
        """
        req_id → [part_name, ...]

        SysML v2 允许多个 part 同时 satisfy 同一 REQ（联合满足、分层满足、冗余满足）。
        保留所有满足关系；后续解析按"谁有 guard 就用谁"选取，不靠迭代顺序。
        """
        m: Dict[str, List[str]] = {}
        for part in self._model.part_definitions:
            for sat in part.satisfy_relationships:
                m.setdefault(sat.target.name, []).append(part.name)
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

    def _build_syside_attr_map(self) -> Dict[str, Dict[str, float]]:
        """
        Walk the syside AST and evaluate every AttributeUsage expression,
        grouped by owner part name.  Returns {part_name: {attr_name: float}}.

        Handles arithmetic expressions and unit-bearing literals that
        _parse_attr_value (regex-based) cannot evaluate.
        Falls back to {} when syside is unavailable or parsing fails.
        """
        if not _SYSIDE_OK:
            return {}
        sysml_text = self._model.to_sysml_text() if self._model else ""
        if not sysml_text:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        try:
            model, _ = _syside.try_load_model(sysml_source=sysml_text)
            compiler = _syside.Compiler()
            for attr in model.nodes(_syside.AttributeUsage):
                try:
                    owner = attr.owner
                    if owner is None:
                        continue
                    part_name = getattr(owner, "name", None)
                    if not part_name:
                        continue
                    expr = attr.feature_value_expression
                    if expr is None:
                        continue
                    val, report = compiler.evaluate(expr)
                    if not report.fatal and val is not None:
                        result.setdefault(part_name, {})[attr.name] = float(val)
                except Exception:
                    pass
        except Exception:
            pass
        return result

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

        # Augment with syside-evaluated values: handles expressions like
        # `= 10.0 [m/s]` or `= mass * g` that _parse_attr_value regex misses.
        # Syside values take precedence when they successfully evaluate.
        for part_name, syside_attrs in self._build_syside_attr_map().items():
            result.setdefault(part_name, {}).update(syside_attrs)

        return result

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    def _covered_req_ids(self) -> set:
        return set(self._satisfy_map.keys())

    def _resolve_all(self) -> List[ResolvedParam]:
        """
        遍历所有已映射的 REQ，将 ardu_params 的 token 解析为实际数值。

        token 解析优先级（内容匹配后已预解析的值优先）：
          @guard       → cat["_resolved_guard_val"]（内容匹配已算好）
          @attr_match  → cat["_resolved_attr_val"]（内容匹配已算好）
          @attr:name   → 关键词搜索属性值
          @attr:name*N → 同上 × N
          @guard*N / @guard_attr*N → parachute delay 特殊处理
          number       → 直接使用
        """
        results: List[ResolvedParam] = []
        for req_id in self._covered_req_ids():
            cat = self._lookup_catalogue(req_id)
            if cat is None:
                continue

            part_names = self._satisfy_map[req_id]
            part_name  = part_names[0]

            # 从 cat 取出预解析值（内容匹配路径已解析，其他路径为 None）
            pre_guard_val = cat.get("_resolved_guard_val")
            pre_guard_src = cat.get("_resolved_guard_src") or "guard"
            pre_attr_val  = cat.get("_resolved_attr_val")
            pre_attr_src  = cat.get("_resolved_attr_src") or "attr"

            # parachute delay：按关键词搜索属性（兼容新旧路径）
            chute_delay: Optional[float] = None
            if "CHUTE_DELAY_MS" in cat.get("ardu_params", {}):
                for pname in part_names:
                    chute_delay = self._extract_chute_delay(pname)
                    if chute_delay is not None:
                        part_name = pname
                        break

            for param_name, raw_value in cat["ardu_params"].items():
                if raw_value == "@guard":
                    val = pre_guard_val if pre_guard_val is not None \
                          else f"<unresolved:guard>"
                    src = pre_guard_src

                elif raw_value == "@attr_match":
                    # AttrMatcher 已预解析（含单位换算）
                    val = pre_attr_val if pre_attr_val is not None \
                          else f"<unresolved:attr_match>"
                    src = pre_attr_src

                elif isinstance(raw_value, str) and (
                    raw_value.startswith("@guard*") or
                    raw_value.startswith("@guard_attr*")
                ):
                    # parachute: parachuteDeployTime × 1000
                    val = (round(chute_delay * 1000)
                           if chute_delay is not None
                           else "<unresolved:chute_delay>")
                    src = "attr:parachute*1000" if chute_delay else "unresolved"

                elif isinstance(raw_value, str) and raw_value.startswith("@attr:"):
                    # @attr:name 或 @attr:name*N
                    rest = raw_value[6:]
                    if "*" in rest:
                        attr_name, factor_str = rest.split("*", 1)
                        factor = float(factor_str)
                    else:
                        attr_name, factor = rest, 1.0
                    val, src = self._extract_attr(part_name, attr_name)
                    if isinstance(val, (int, float)) and factor != 1.0:
                        val = round(val * factor)
                        src = f"{src}*{int(factor)}"

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

    def _threshold_for_guard_var(
        self,
        part_name: str,
        guard_var: str,
        operators: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], str]:
        """Resolve the numeric threshold of the model guard whose attribute
        matches *guard_var* (the variable the AST synthesizer matched).

        Used by the AST-synthesis layer so `@guard` placeholders are filled
        from the guard that layer actually found — instead of leaking as
        `<unresolved:guard>` even though the model contains the guard.

        When *operators* is given (the tag's expected operators), a guard
        whose operator is in that set is preferred.  Returns (None, "") when
        no matching guard carries a numeric threshold (e.g. a boolean guard,
        which needs no threshold).
        """
        gv = (guard_var or "").lower()
        cands = [
            g for g in self._guard_map.get(part_name, [])
            if (getattr(g, "attribute", "") or "").lower() == gv
            and getattr(g, "threshold", None) is not None
        ]
        if not cands:
            return None, ""
        if operators:
            preferred = [g for g in cands if getattr(g, "operator", None) in operators]
            if preferred:
                cands = preferred
        g = cands[0]
        op = getattr(g, "operator", "?")
        th = float(getattr(g, "threshold"))
        return th, f"guard:{op}:{th} (attr:{getattr(g, 'attribute', '?')}, ast)"

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
