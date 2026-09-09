"""sitl_catalogue.py

内容驱动的 SITL 需求目录：guard/attr 内容匹配器与 _CONTENT_CATALOGUE 条目表。

从 requirement_linker.py 拆出的纯数据模块，匹配与分配逻辑仍在
RequirementLinker；这里只声明哪种 guard/attr 内容对应哪些 ArduPilot 参数与
SITL 测试模板。条目顺序即优先级（更具体的在前）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.sitl.sitl_specs import InjectSpec, VerifySpec


@dataclass
class GuardMatcher:
    """按 guard 内容匹配（与 REQ ID 无关）。

    operators     : 匹配的比较运算符列表；["bool"] 代表 bool_true guard；
                    "event" 代表 accept 事件驱动的转移（无 guard，按事件
                    类型名匹配 var_keywords；事件驱动与 guard 驱动是同一
                    行为语义的两种 SysML 写法）
    var_keywords  : guard 变量名/accept 事件名（小写）需含其中任一关键词
    threshold_min : 阈值下界（comparison guard）
    threshold_max : 阈值上界（comparison guard）
    action_kws    : 可选，用 entry action 名区分同类 guard（如
                    SENSOR_ARMING_INHIBIT vs SENSOR_GROUND_ALERT）
    """
    operators: List[str]
    var_keywords: List[str]
    threshold_min: float = float("-inf")
    threshold_max: float = float("inf")
    action_kws: List[str] = field(default_factory=list)


@dataclass
class AttrMatcher:
    """按 part 属性名内容匹配（与 REQ ID 无关）。

    attr_keywords : 属性名（小写）必须包含其中任一关键词
    part_keywords : 可选，part 名（小写）必须包含其中任一关键词（防跨部件误匹配）
    multiplier    : 单位换算系数（如 km->m 用 1000，m/s->cm/s 用 100）
    allow_port_match : 属性不命中时按 port 名匹配（同一 attr_keywords）。port
                    没有数值，只有 ardu_params 全为静态值的条目才允许开启；
                    接口/配置类需求（RTCM、MAVLink）在模型里常常只声明 port
                    不声明属性。
    """
    attr_keywords: List[str]
    part_keywords: List[str] = field(default_factory=list)
    multiplier: float = 1.0
    allow_port_match: bool = False


@dataclass
class ContentEntry:
    """内容驱动的 Catalogue 条目。

    匹配键：guard 内容或 attr 内容，而非 REQ ID 字符串。

    ardu_params 值语义：
      "@guard"       -> 匹配 guard 的阈值（comparison guard 时）
      "@attr_match"  -> 匹配到的属性值（AttrMatcher 时）
      "@attr_match*N"-> 同上 x N
      "@attr:name"   -> 按名字关键词搜索的属性（独立于 AttrMatcher）
      "@attr:name*N" -> 同上 x N
      number         -> 静态固定值

    req_text_kws：attr 匹配的需求文本相关性 gate。attr 命中只说明 satisfy 的
    part 上有关键词属性，会把 MTOW/温度/法规类需求错挂到维度无关的参数上
    （如 FENCE_ALT_MAX "验证"姿态 RMS）。非空时要求需求文本至少命中一个关键词，
    否则落入 no-mapping 而不是错误的 L1 PASS。需求无 doc 文本时不启用该 gate。

    req_text_exclude_kws：即使命中正向关键词，若需求描述的是另一个物理量
    （如逆风下的 ground speed），该条目也不宣称验证成功。
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
    # 可选的附加 L2 用例：主 tier 仍是 L1（参数一致性证据不动），再加一条
    # L2 行为检查（如 FENCE 缩尺执法测试）。同一需求的两类证据分两个 tier
    # 字段，不改写主 tier，避免已有 L1 证据语义漂移。
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
            threshold_max=20.0,
        ),
        ardu_params={"BATT_FS_CRT_PCT": "@guard", "BATT_FS_CRT_ACT": 1},
        tier="L1",
        **_noop_l1(),
        notes="Battery-critical guard (<) → BATT_FS_CRT_PCT.",
    ),

    # ── Safety: GCS link-loss（commLossTime > X s），动作按需求文本分档。
    #    GCS_LOSS 是默认/歧义档，兼容 AST 合成与 LLM 语义分类产出的同名标签；
    #    文本明确 land-only / return-only 时由 _preferred_semantic_tags 选到
    #    动作特定条目（此前 "land at current position" 被 "已切到 RTL" 判 PASS）。
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

    ContentEntry(
        semantic_tag="GCS_LOSS_LAND",
        guard_matcher=GuardMatcher(
            operators=[">", ">="],
            var_keywords=["comm", "gcs", "link", "heartbeat", "loss"],
        ),
        ardu_params={"FS_GCS_ENABLE": 5, "FS_GCS_TIMEOUT": "@guard"},
        tier="L2",
        inject=InjectSpec(kind="disconnect_gcs", params={"FS_GCS_ENABLE": 5},
                          pre_takeoff_m=5.0),
        verify=VerifySpec(kind="wait_mode", args={"mode": "LAND"}, timeout=25.0),
        notes="GCS loss → LAND at current position (FS_GCS_ENABLE=5); requirement mandates landing, RTL is NOT accepted.",
    ),

    ContentEntry(
        semantic_tag="GCS_LOSS_RTL",
        guard_matcher=GuardMatcher(
            operators=[">", ">="],
            var_keywords=["comm", "gcs", "link", "heartbeat", "loss"],
        ),
        ardu_params={"FS_GCS_ENABLE": 1, "FS_GCS_TIMEOUT": "@guard"},
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
            # SIM_GPS1_ENABLE=0 cuts the simulated GPS feed at runtime (immediate), so
            # EKF marks GPS unhealthy and the SYS_STATUS GPS bit clears. GPS_TYPE=0
            # (previous) is driver config and generally needs a reboot, so it left GPS
            # "healthy" at runtime -> false negative.
            params={"SIM_GPS1_ENABLE": 0.0, "_settle_s": 5.0, "_pre_mode": "GUIDED"},
        ),
        verify=VerifySpec(
            kind="assert_sensor_unhealthy",
            args={"sensor": "gps"},
            timeout=20.0,
        ),
        notes="Sensor failure → SYS_STATUS GPS health bit cleared.",
    ),

    # ── Safety: parachute deploy（bool guard 或 accept 事件：
    #    propulsionCriticalFailure / accept CriticalPropulsionFailure）。
    #    run 44642597 把降落伞写成事件机，bool-only 匹配面让它落入 silent
    #    no-mapping；开放 accept 面后，错发命令的模型仍被同一道响应溯源门
    #    拦截（5af6c666 形态回归钉住）。
    ContentEntry(
        semantic_tag="PARACHUTE_DEPLOY",
        guard_matcher=GuardMatcher(
            operators=["bool", "event"],
            var_keywords=["propulsion", "engine", "motor", "thrust"],
        ),
        ardu_params={
            "CHUTE_ENABLED": 1,
            "CHUTE_TYPE": 10,
            "CHUTE_DELAY_MS": "@attr:parachute*1000",
            # SERVO8=Parachute -> Gazebo channel 7（避开 gimbal 占用的 SERVO9-11）
            "SERVO8_FUNCTION": 27,
            "CHUTE_SERVO_ON": 2000,
            "CHUTE_SERVO_OFF": 1000,
            "CHUTE_ALT_MIN": 0,         # release allowed at low native-SITL altitude
        },
        tier="L2",
        inject=InjectSpec(
            kind="mavlink_command",
            # MAV_CMD_DO_PARACHUTE (208): param1=2 -> RELEASE
            # 直接命令释放；verify 读 SERVO8 PWM，不依赖 STATUSTEXT。
            # （SIM_ENGINE_FAIL 只断电机推力，不触发坠毁检测）
            params={
                "command": 208,
                "param1": 2,
                "_settle_s": 0.5,
                # Applied after takeoff, immediately before MAV_CMD_DO_PARACHUTE, so native
                # SITL actuates the parachute. Duplicated from ardu_params: boot defaults
                # serve generated scripts, runtime setup serves recommended.parm runs that
                # carry only design-level params.
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
    #    deliveryAbortConditionActive / accept AbortConditionActive）。
    #    var_keywords 不含 "delivery"：release 触发器
    #    （DeliveryCoordinateSatisfied）也含该词，会被错误吸入；
    #    abort/lock/payload/gripper 已覆盖 bool 与 event 两种写法。
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
            "GRIP_TYPE": 1,
            # SERVO7=Gripper -> Gazebo channel 6（避开 gimbal 占用的 SERVO9-11/channel 8-10）
            "SERVO7_FUNCTION": 28,
            "GRIP_RELEASE": 2000,
            "GRIP_GRAB": 1000,
        },
        tier="L2",
        inject=InjectSpec(
            kind="mavlink_command",
            # MAV_CMD_DO_GRIPPER (211): param1=gripper_id(0), param2=action.
            # MAVLink GRIPPER_ACTIONS: 0=RELEASE, 1=GRAB；需求是 payload-abort -> LOCK
            # （保住载荷）= GRAB，所以 param2=1。
            # 前置 RELEASE 建立状态转换判别力：boot .parm 的 GRIP_NEUTRAL=1000
            # （POWERON_LOCK 条目）让上电输出即为锁定 PWM，只发 GRAB 断言终值会假绿；
            # 先 RELEASE 到 2000 再 abort-GRAB 回 1000，也贴合"abort 时回到锁定"的语义。
            params={
                "pre_command": 211,
                "pre_param1": 0,
                "pre_param2": 0,
                "_pre_settle_s": 2.0,
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
            args={"channel": 7, "target_pwm": 1000, "tol": 50},
            timeout=10.0,
        ),
        notes="Payload abort → MAV_CMD_DO_GRIPPER GRAB → SERVO_OUTPUT_RAW.servo7_raw≈1000 (lock).",
    ),

    # ── Safety: payload power-on default lock（accept PowerOnEvent -> 默认锁定态）
    #    验证飞控栈侧的等价命题：夹爪已配置，上电后未解锁前 SERVO7 输出停在
    #    锁定 PWM（GRIP_NEUTRAL=GRIP_GRAB=1000，ArduPilot 上电即驱动 neutral），
    #    且从未出现 release PWM。机械默认态属检验域，不在此主张。
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
            "GRIP_NEUTRAL": 1000,
        },
        tier="L2",
        inject=InjectSpec(
            kind="set_param",
            # 不起飞、不解锁：检查的就是 power-on 且 before-arming 的输出。参数已由
            # 启动 .parm 装载（AP_Gripper 在 init 读 GRIP_ENABLE，运行时改写不生效，
            # boot defaults 是唯一可靠路径）；这里只留 settle 让 SERVO_OUTPUT_RAW 稳定。
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
    #    L1（主 tier）：模型值 -> FENCE_ALT_MAX 参数一致性。
    #    附加 L2：缩尺执法测试，起飞后把 FENCE_ALT_MAX 压到机体之下，断言
    #    飞控栈立刻执行围栏动作（RTL；RTL 不可用时降落）。检的是执法机制被
    #    行使；120m 数值本身仍由 L1 主张。
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
            # 再把 ALT_MAX 压到 10m -> 即刻越界 -> FENCE_ACTION=1（RTL）。
            # 经验证：ALT_MAX 写入后 <1s 内 "Max Alt fence breached" -> RTL。
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
    #    通信类 part 可以只声明 port - allow_port_match 让配置级 L1 证据
    #    （SERIAL0_PROTOCOL=2，全静态值）不依赖属性存在）
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
        # 附加 L2：SERIAL0_PROTOCOL=2 只说明端口被要求用 MAVLink v2，不说明链路
        # 在说 v2、会应答、且不断线。这条 L2 读实际帧头 (0xFD)、发一条必须被应答的
        # 命令、并量 HEARTBEAT 最大间隔。加密那半句不在范围内，由 obligation 拆分
        # 单独承担。
        l2_inject=InjectSpec(kind="noop"),
        l2_verify=VerifySpec(
            kind="assert_mavlink_v2_link",
            args={"window_s": 8.0, "max_gap_s": 2.0},
            timeout=30.0,
        ),
        l2_notes=(
            "Wire-level MAVLink v2 conformance: v2 start-of-frame on received "
            "packets, an answered uplink command, and HEARTBEAT continuity. "
            "Channel encryption is explicitly NOT covered."
        ),
        req_text_kws=["mavlink", "protocol", "telemetry", "data link", "datalink",
                      "communication", "encrypted", "encryption", "gcs", "command link"],
        # A configured MAVLink port proves protocol availability only, not that a
        # lifecycle-triggered report is produced or that its deadline is met. Those
        # requirements need an executable behaviour anchor (landing completion ->
        # report action), not SERIAL0_PROTOCOL.
        req_text_exclude_kws=[
            "post-flight", "post flight", "health report", "status report",
            "after landing", "landing completion",
        ],
    ),

    # ── Functional: 航点改写并入活动航线的时限（attr: waypointModificationLatency）
    #    L1 无对应参数可比（"并入活动航线"不是任何一个 ArduPilot 参数），所以
    #    直接是 L2：上传初始任务 -> 上传改写后的任务 -> 从 MISSION_ACK（收到
    #    有效改写命令的时刻）计时到导航控制器发布的 POSITION_TARGET_GLOBAL_INT
    #    已带上新坐标。MISSION_ITEM_INT 回读只说明任务存储已更新。传输耗时单独
    #    报告并排除，需求的时钟从"收到命令"起算。
    ContentEntry(
        semantic_tag="WAYPOINT_MODIFICATION_LATENCY",
        attr_matcher=AttrMatcher(
            attr_keywords=["waypointmodification", "waypointupdate", "replan",
                           "missionupdate"],
            part_keywords=["flight", "controller", "autopilot", "navigation"],
        ),
        ardu_params={},
        tier="L2",
        inject=InjectSpec(kind="noop"),
        verify=VerifySpec(
            kind="assert_waypoint_update_latency",
            args={"max_latency_s": "@attr_match"},
            timeout=60.0,
        ),
        notes=("Revised active waypoint → POSITION_TARGET_GLOBAL_INT controller "
               "target, timed from MISSION_ACK; mission-storage read-back and "
               "transfer time are reported but do not establish active adoption."),
        req_text_kws=["waypoint", "flight plan", "mission", "revised", "modification"],
        req_text_exclude_kws=["cep", "circular error", "obstacle", "collision"],
    ),

    ContentEntry(
        semantic_tag="MAX_SPEED",
        attr_matcher=AttrMatcher(
            attr_keywords=["maxairspeed", "airspeed", "maxspeed"],
            multiplier=100.0,
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

    ContentEntry(
        semantic_tag="RADIUS_FENCE",
        attr_matcher=AttrMatcher(
            attr_keywords=["maxoperationalradius", "operationalradius",
                           "maxradius", "radius"],
            multiplier=1000.0,
        ),
        ardu_params={"FENCE_RADIUS": "@attr_match"},
        tier="L1",
        **_noop_l1(),
        notes="maxOperationalRadius (km) × 1000 → FENCE_RADIUS (m).",
        req_text_kws=["radius", "range", "geofence", "boundary", "distance"],
    ),

    # ── Interface: RTCM GNSS corrections（attr 或 port: gnssCorrections 等；
    #    感知类 part 可以只声明 port。GPS_INJECT_TO=127 是配置级证据：飞控栈
    #    已配置接收/转发 RTCM 注入；亚米精度按需求 V 标注归 RTK bench）
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


_TAG_TO_ENTRY: Dict[str, ContentEntry] = {
    e.semantic_tag: e for e in _CONTENT_CATALOGUE
}
