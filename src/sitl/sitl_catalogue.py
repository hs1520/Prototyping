"""
sitl_catalogue.py

内容驱动的 SITL 需求目录：guard/attr 内容匹配器与 _CONTENT_CATALOGUE 条目表。

从 requirement_linker.py 拆出的纯数据模块——匹配与分配逻辑仍在
RequirementLinker；本模块只声明"什么样的 guard/attr 内容对应哪些
ArduPilot 参数与 SITL 测试模板"。条目顺序编码优先级（更具体的在前）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.sitl.sitl_specs import InjectSpec, VerifySpec


# ---------------------------------------------------------------------------
# 内容匹配器数据结构
# ---------------------------------------------------------------------------

@dataclass
class GuardMatcher:
    """
    按 guard 内容匹配（与 REQ ID 无关）。

    operators     : 匹配的比较运算符列表；["bool"] 代表 bool_true guard；
                    "event" 代表 accept 事件驱动的转移（无 guard，按事件
                    类型名匹配 var_keywords —— 事件驱动与 guard 驱动是同一
                    行为语义的两种合法 SysML 写法，只认 guard 会漏掉前者）
    var_keywords  : guard 变量名/accept 事件名（小写）必须包含其中任一关键词
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
    allow_port_match : 属性不命中时允许按 port 名匹配（同一 attr_keywords）。
                    port 没有数值，只有 ardu_params 全为静态值的条目才允许
                    开启——接口/配置类需求（RTCM、MAVLink）在模型里合法地
                    只声明 port 不声明属性，只认属性会漏掉它们。
    """
    attr_keywords: List[str]
    part_keywords: List[str] = field(default_factory=list)
    multiplier: float = 1.0
    allow_port_match: bool = False


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

    req_text_kws：attr 匹配的需求文本相关性 gate。attr 匹配本身只看
    "satisfy 的 part 上有没有关键词属性"，会把 MTOW/温度/法规类需求错挂到
    维度无关的参数上（如 FENCE_ALT_MAX "验证"姿态 RMS）。非空时要求需求
    文本至少命中一个关键词，否则该条目对这条需求不匹配——落入诚实的
    no-mapping，而不是错误的 L1 PASS。需求无 doc 文本时不启用该 gate。

    req_text_exclude_kws：即使命中正向关键词，若需求明确描述另一个物理量
    （例如逆风下的 ground speed），也禁止该条目宣称验证成功。
    """
    semantic_tag: str
    guard_matcher: Optional[GuardMatcher] = None
    attr_matcher: Optional[AttrMatcher] = None
    ardu_params: Dict[str, Any] = field(default_factory=dict)
    tier: str = "L1"
    inject: Optional[Any] = None
    verify: Optional[Any] = None
    notes: str = ""
    req_text_kws: List[str] = field(default_factory=list)
    req_text_exclude_kws: List[str] = field(default_factory=list)
    # 可选的附加 L2 用例：主 tier 保持 L1（参数一致性证据不动），额外生成
    # 一条 L2 行为检查（如 FENCE 缩尺执法测试）。二者是同一需求的两类证据，
    # 用两个 tier 字段而不是改写主 tier，避免已有 L1 证据语义漂移。
    l2_inject: Optional[Any] = None
    l2_verify: Optional[Any] = None
    l2_notes: str = ""


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

    # ── Safety: GCS link-loss（commLossTime > X s）——动作按需求文本分档。
    #    lenient 条目（GCS_LOSS）保留为默认/歧义档，同时兼容 AST 合成与 LLM
    #    语义分类仍产出的 "GCS_LOSS" 标签；文本明确 land-only / return-only 时
    #    由 _preferred_semantic_tags 选到动作特定条目——需求要求原地降落却
    #    发生 RTL 不再靠 fallback 放行（之前 "land at current position" 被
    #    "模式已切换到 RTL" 判 PASS，验证了错误动作）。
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
        notes="GCS timeout guard → FS_GCS_TIMEOUT + L2 disconnect test (either failsafe action accepted).",
    ),

    # ── Safety: GCS link-loss → LAND at current position（需求明确要求降落）
    ContentEntry(
        semantic_tag="GCS_LOSS_LAND",
        guard_matcher=GuardMatcher(
            operators=[">", ">="],
            var_keywords=["comm", "gcs", "link", "heartbeat", "loss"],
        ),
        ardu_params={"FS_GCS_ENABLE": 5, "FS_GCS_TIMEOUT": "@guard"},  # 5 = Land
        tier="L2",
        inject=InjectSpec(kind="disconnect_gcs", params={"FS_GCS_ENABLE": 5},
                          pre_takeoff_m=5.0),
        verify=VerifySpec(kind="wait_mode", args={"mode": "LAND"}, timeout=25.0),
        notes="GCS loss → LAND at current position (FS_GCS_ENABLE=5); requirement mandates landing, RTL is NOT accepted.",
    ),

    # ── Safety: GCS link-loss → RTL（需求明确要求返航）
    ContentEntry(
        semantic_tag="GCS_LOSS_RTL",
        guard_matcher=GuardMatcher(
            operators=[">", ">="],
            var_keywords=["comm", "gcs", "link", "heartbeat", "loss"],
        ),
        ardu_params={"FS_GCS_ENABLE": 1, "FS_GCS_TIMEOUT": "@guard"},  # 1 = RTL
        tier="L2",
        inject=InjectSpec(kind="disconnect_gcs", params={"FS_GCS_ENABLE": 1},
                          pre_takeoff_m=5.0),
        verify=VerifySpec(kind="wait_mode", args={"mode": "RTL"}, timeout=25.0),
        notes="GCS loss → RTL (FS_GCS_ENABLE=1); requirement mandates return, LAND is NOT accepted.",
    ),

    # ── Safety: sensor arming inhibit（bool: sensorSelfTestFailed）
    #    action_kws 区分 "inhibit arming" vs "ground alert"
    ContentEntry(
        semantic_tag="SENSOR_ARMING_INHIBIT",
        guard_matcher=GuardMatcher(
            operators=["bool"],
            var_keywords=["sensor", "selftest", "post", "prearm"],
        ),
        ardu_params={"ARMING_CHECK": 1},
        tier="L2",
        inject=InjectSpec(
            kind="set_param",
            params={"SIM_GPS1_ENABLE": 0.0, "_settle_s": 5.0, "_pre_mode": "GUIDED"},
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
        ),
        ardu_params={},
        tier="L2",
        inject=InjectSpec(
            kind="set_param",
            # SIM_GPS1_ENABLE=0 cuts the simulated GPS feed at runtime (immediate),
            # so EKF marks GPS unhealthy and the SYS_STATUS GPS bit clears. GPS_TYPE=0
            # (previous) is a driver-config param that generally needs a reboot to take
            # effect, so it left GPS "healthy" at runtime → false negative.
            params={"SIM_GPS1_ENABLE": 0.0, "_settle_s": 5.0, "_pre_mode": "GUIDED"},
        ),
        verify=VerifySpec(
            kind="assert_sensor_unhealthy",
            args={"sensor": "gps"},
            timeout=20.0,
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
            "CHUTE_ALT_MIN": 0,         # SITL test: release is allowed at low native-SITL altitudes
        },
        tier="L2",
        inject=InjectSpec(
            kind="mavlink_command",
            # MAV_CMD_DO_PARACHUTE (208): param1=2 → RELEASE
            # 直接命令释放；verify 读 SERVO8 PWM，不依赖 STATUSTEXT。
            # （SIM_ENGINE_FAIL 只断电机推力，不触发坠毁检测）
            params={
                "command": 208,
                "param1": 2,
                "_settle_s": 0.5,
                # Native SITL reliably actuates the parachute when these are applied
                # after takeoff, immediately before MAV_CMD_DO_PARACHUTE. They are
                # duplicated from ardu_params intentionally: boot defaults support
                # generated scripts, while runtime setup supports recommended.parm
                # runs that only carry design-level params.
                "CHUTE_ENABLED": 1,
                "CHUTE_TYPE": 10,
                "CHUTE_ALT_MIN": 0,
                "CHUTE_SERVO_ON": 2000,
                "CHUTE_SERVO_OFF": 1000,
                "SERVO8_FUNCTION": 27,
            },
            pre_takeoff_m=10.0,
        ),
        verify=VerifySpec(
            kind="assert_servo_pwm",
            args={"channel": 8, "target_pwm": 2000, "tol": 50},
            timeout=15.0,
        ),
        notes="MAV_CMD_DO_PARACHUTE RELEASE → SERVO_OUTPUT_RAW.servo8_raw≈2000.",
    ),

    # ── Safety: payload abort lock（bool guard 或 accept 事件，如
    #    deliveryAbortConditionActive / accept AbortConditionActive）
    #    var_keywords 不含 "delivery"：accept 面开放后，release 触发器
    #    （DeliveryCoordinateSatisfied）也含 "delivery"，会被错误吸入；
    #    abort/lock/payload/gripper 足以覆盖 bool 与 event 两种写法。
    #    req_text 门挡掉同族 release 类需求（FUNC/PERF 的释放条款）。
    ContentEntry(
        semantic_tag="PAYLOAD_ABORT_LOCK",
        guard_matcher=GuardMatcher(
            operators=["bool", "event"],
            var_keywords=["payload", "abort", "lock", "gripper"],
        ),
        req_text_kws=["abort", "lock"],
        req_text_exclude_kws=["release the payload", "release actuation"],
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
            # MAV_CMD_DO_GRIPPER (211): param1=gripper_id(0), param2=action.
            # MAVLink GRIPPER_ACTIONS: 0=RELEASE, 1=GRAB. Requirement is payload-abort
            # → LOCK (hold the payload) = GRAB, so param2=1 is correct.
            params={
                "command": 211,
                "param1": 0,
                "param2": 1,
                "_settle_s": 1.5,
                "GRIP_ENABLE": 1,
                "GRIP_TYPE": 1,
                "SERVO7_FUNCTION": 28,
                "GRIP_RELEASE": 2000,
                "GRIP_GRAB": 1000,
            },
            pre_takeoff_m=5.0,
        ),
        verify=VerifySpec(
            kind="assert_servo_pwm",
            # abort → LOCK = GRAB → servo drives to GRIP_GRAB=1000 (NOT release 2000).
            args={"channel": 7, "target_pwm": 1000, "tol": 50},
            timeout=10.0,
        ),
        notes="Payload abort → MAV_CMD_DO_GRIPPER GRAB → SERVO_OUTPUT_RAW.servo7_raw≈1000 (lock).",
    ),

    # ── Safety: payload power-on default lock（accept PowerOnEvent → 默认锁定态）
    #    验证的是飞控栈侧的等价命题：夹爪已配置且上电后、未解锁前，
    #    SERVO7 输出停在锁定 PWM（GRIP_NEUTRAL=GRIP_GRAB=1000，经验证
    #    ArduPilot 上电即驱动 neutral 位），且从未出现 release PWM。
    #    机械默认态本身属检验域；这里主张的只是 flight-stack 侧配置+输出。
    ContentEntry(
        semantic_tag="PAYLOAD_POWERON_LOCK",
        guard_matcher=GuardMatcher(
            operators=["bool", "event"],
            var_keywords=["power", "poweron", "boot", "startup"],
        ),
        req_text_kws=["power-on", "power on", "default"],
        ardu_params={
            "GRIP_ENABLE": 1,
            "GRIP_TYPE": 1,
            "SERVO7_FUNCTION": 28,
            "GRIP_RELEASE": 2000,
            "GRIP_GRAB": 1000,
            "GRIP_NEUTRAL": 1000,   # neutral=locked：上电输出即锁定位
        },
        tier="L2",
        inject=InjectSpec(
            kind="set_param",
            # 不起飞、不解锁：检查的就是 power-on 且 before-arming 的输出。
            # 参数已由启动 .parm 装载（AP_Gripper 在 init 读 GRIP_ENABLE，
            # 运行时改写不生效——boot defaults 是唯一可靠路径）；这里只留
            # settle 让 SERVO_OUTPUT_RAW 流稳定。
            params={"_settle_s": 3.0},
            pre_takeoff_m=0.0,
        ),
        verify=VerifySpec(
            kind="assert_servo_pwm",
            args={"channel": 7, "target_pwm": 1000, "tol": 50},
            timeout=10.0,
        ),
        notes="Power-on (disarmed) gripper output at locked PWM (GRIP_NEUTRAL=GRAB=1000); release PWM never driven before arming.",
    ),

    # ── Constraint: max altitude（attr: maxAltitude）
    #    L1（主 tier）：模型值 → FENCE_ALT_MAX 参数一致性。
    #    附加 L2：缩尺执法测试——起飞后把 FENCE_ALT_MAX 压到机体之下，
    #    断言飞控栈立刻执行围栏动作（RTL；RTL 不可用时降落）。验证的是
    #    执法机制存在且被行使；120m 数值本身仍由 L1 主张（缩尺不重放数值）。
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
        req_text_kws=["altitude", "height", "ceiling", "above ground", "agl", "vertical"],
        l2_inject=InjectSpec(
            kind="set_param",
            # 起飞到 15m（force_arm_and_takeoff 会临时关围栏、到高度后恢复），
            # 再把 ALT_MAX 压到 10m → 即刻越界 → FENCE_ACTION=1（RTL）。
            # 经验证：ALT_MAX 写入后 <1s 内 "Max Alt fence breached" → RTL。
            params={"FENCE_ACTION": 1, "FENCE_ALT_MAX": 10, "FENCE_ENABLE": 1,
                    "_settle_s": 1.0},
            pre_takeoff_m=15.0,
        ),
        l2_verify=VerifySpec(
            kind="wait_mode",
            args={"mode": "RTL", "fallback": "LAND"},
            timeout=30.0,
        ),
        l2_notes=(
            "Scaled fence-enforcement check: lower FENCE_ALT_MAX below current "
            "altitude in flight; flight stack must execute the fence action "
            "(RTL/LAND). Mechanism evidence only — the 120 m value is claimed "
            "by the L1 param row, not by this scaled test."
        ),
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
        req_text_kws=["loop", "frequency", "hz", "control rate", "refresh"],
    ),

    # ── Interface: MAVLink protocol（attr 或 port: mavlinkTelemetry 等；
    #    通信类 part 合法地只声明 port —— allow_port_match 让配置级 L1
    #    证据（SERIAL0_PROTOCOL=2，全静态值）不依赖属性存在）
    ContentEntry(
        semantic_tag="MAVLINK_PROTOCOL",
        attr_matcher=AttrMatcher(
            attr_keywords=["encryption", "protocol", "mavlink", "keylength",
                           "serial", "maxoperational"],
            part_keywords=["communication", "comm", "link", "gcs"],
            allow_port_match=True,
        ),
        ardu_params={"SERIAL0_PROTOCOL": 2},
        tier="L1",
        **_noop_l1(),
        notes="CommunicationSystem → SERIAL0_PROTOCOL=2 (MAVLink v2).",
        req_text_kws=["mavlink", "protocol", "telemetry", "data link", "datalink",
                      "communication", "encrypted", "encryption", "gcs", "command link"],
        # A configured MAVLink port proves protocol availability only.  It does
        # not prove that a lifecycle-triggered report is produced, nor that its
        # deadline is met.  Those requirements need an executable behaviour
        # anchor (landing completion → report action), not SERIAL0_PROTOCOL.
        req_text_exclude_kws=[
            "post-flight", "post flight", "health report", "status report",
            "after landing", "landing completion",
        ],
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
        req_text_kws=["speed", "airspeed", "velocity", "m/s"],
        # A navigation speed setpoint does not verify wind-relative groundspeed.
        req_text_exclude_kws=["ground speed", "groundspeed", "headwind",
                              "tailwind"],
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
        req_text_kws=["radius", "range", "geofence", "boundary", "distance"],
    ),

    # ── Interface: RTCM GNSS corrections（attr 或 port: gnssCorrections 等；
    #    感知类 part 合法地只声明 port。GPS_INJECT_TO=127 是配置级证据——
    #    飞控栈已配置接收/转发 RTCM 注入；亚米精度按需求 V 标注仍归 RTK bench）
    ContentEntry(
        semantic_tag="RTCM_GPS",
        attr_matcher=AttrMatcher(
            attr_keywords=["gnss", "gps", "correction", "rtcm", "differential"],
            part_keywords=["perception", "navigation", "gnss", "gps", "sensor"],
            allow_port_match=True,
        ),
        ardu_params={"GPS_INJECT_TO": 127},
        tier="L1",
        **_noop_l1(),
        notes="RTCM/GNSS part → GPS_INJECT_TO=127 (accept/forward RTCM corrections); config-level evidence only.",
        req_text_kws=["rtcm", "gnss", "gps", "correction", "differential", "positioning"],
    ),
]


# ---------------------------------------------------------------------------
# 按 semantic_tag 索引（供 Layer 2 AST 合成 / Layer 3 LLM 语义标签使用）
# ---------------------------------------------------------------------------

_TAG_TO_ENTRY: Dict[str, ContentEntry] = {
    e.semantic_tag: e for e in _CONTENT_CATALOGUE
}
