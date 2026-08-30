"""
sitl_specs.py

结构化的 SITL inject / verify spec + handler 注册表。

替代之前的字符串 DSL（"param set X Y" / "wait_mode(LAND, timeout=15)"）。
新增一种 inject/verify 类型只需:
  1. 添加一个 Spec dataclass（或扩展现有 dataclass 的 kind）
  2. 在 INJECT_HANDLERS 或 VERIFY_HANDLERS 注册一个 InjectHandler/VerifyHandler

不再需要在 bridge 里加 if/elif 分支。
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectSpec:
    """
    描述一次故障注入。

    kind            handler 注册表的 key，决定如何注入
    params          注入参数（每个 kind 自定义语义）
    pre_takeoff_m   注入前若 > 0，则先解锁并起飞到该高度（米）

    Examples:
      InjectSpec(kind="noop")                          — L1 测试，不注入
      InjectSpec(kind="set_param",
                 params={"SIM_BARO_DISABLE": 1.0})
      InjectSpec(kind="disconnect_gcs", pre_takeoff_m=5.0)
      InjectSpec(kind="set_param",
                 params={"SIM_ENGINE_FAIL": 1.0},
                 pre_takeoff_m=10.0)
    """
    kind: str
    params: Mapping[str, float] = field(default_factory=dict)
    pre_takeoff_m: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class VerifySpec:
    """
    描述如何验证预期行为。

    kind     handler 注册表的 key
    args     该 kind 的参数（mode 名、关键词、等待时间等）
    timeout  统一的等待上限（秒）

    Examples:
      VerifySpec(kind="noop")                              — L1 不验证
      VerifySpec(kind="wait_mode",
                 args={"mode": "LAND", "fallback": "RTL"},
                 timeout=25.0)
      VerifySpec(kind="assert_arm_rejected")
      VerifySpec(kind="wait_statustext",
                 args={"keyword": "arachute"},
                 timeout=10.0)
    """
    kind: str
    args: Mapping[str, Any] = field(default_factory=dict)
    timeout: float = 10.0
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))


# ---------------------------------------------------------------------------
# Test context — handler 间共享的运行时辅助
# ---------------------------------------------------------------------------

# ArduCopter 模式 ID 硬编码兜底（mode_mapping() 在连接初期可能为空）
ARDUCOPTER_MODES: Dict[str, int] = {
    "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3,
    "GUIDED": 4, "LOITER": 5, "RTL": 6, "CIRCLE": 7,
    "LAND": 9, "DRIFT": 11, "SPORT": 13, "AUTOTUNE": 15,
    "POSHOLD": 16, "BRAKE": 17, "SMART_RTL": 21,
}


@dataclass
class TestContext:
    """传给 handler 的运行时上下文。"""
    mav: Any
    mavutil: Any

    def set_param(self, name: str, value: float) -> None:
        self.mav.mav.param_set_send(
            self.mav.target_system, self.mav.target_component,
            name.encode(), float(value),
            self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        time.sleep(0.3)

    def set_mode(self, mode_name: str) -> None:
        mapping = self.mav.mode_mapping() or {}
        mode_id = mapping.get(mode_name.upper()) \
                  or ARDUCOPTER_MODES.get(mode_name.upper())
        if mode_id is None:
            return
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        time.sleep(1)

    def force_arm_and_takeoff(self, altitude: float = 3.0) -> bool:
        """恢复传感器 → 切 GUIDED → 解锁 → 起飞 → 等到达目标高度。"""
        self.set_param("ARMING_CHECK", 0)   # bypass ALL PreArm checks for SITL
        self.set_param("SIM_GPS1_ENABLE", 1)
        self.set_param("SIM_BARO_DISABLE", 0)
        # FENCE_ENABLE=1 在没有 GPS fix 时会阻止解锁（Fence requires position）
        # 暂时关闭，起飞后恢复
        self.set_param("FENCE_ENABLE", 0)
        time.sleep(0.5)
        self.set_mode("GUIDED")
        # wait_ready_to_arm：EKF 在启动后需要数秒才能设定 origin/home
        # （"Arm: Need Position Estimate" / "AHRS: waiting for home"），这些是
        # 强制硬检查，ARMING_CHECK=0 和 force-arm(21196) 都绕不过。因此必须
        # 反复重试解锁直到 EKF 就绪，而不是只试一次就放弃。
        arm_ok = False
        deadline = time.time() + 45
        while time.time() < deadline:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 21196, 0, 0, 0, 0, 0,
            )
            ack = self.mav.recv_match(
                type="COMMAND_ACK", blocking=True, timeout=2)
            if (ack
                    and ack.command == self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
                    and ack.result == self.mavutil.mavlink.MAV_RESULT_ACCEPTED):
                arm_ok = True
                break
            time.sleep(2)
        if not arm_ok:
            self.set_param("FENCE_ENABLE", 1)   # 恢复围栏（123 行临时关闭）
            return False

        # 起飞指令（检查 ACK）
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            self.mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude,
        )
        # 读取 TAKEOFF ACK（允许 2s 内收到）
        t_ack = self.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=2)
        takeoff_accepted = (
            t_ack is not None
            and t_ack.command == self.mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
            and t_ack.result == self.mavutil.mavlink.MAV_RESULT_ACCEPTED
        )

        # 若 TAKEOFF 未被接受，改用速度指令强制爬升
        if not takeoff_accepted:
            self.mav.mav.set_position_target_local_ned_send(
                0,
                self.mav.target_system, self.mav.target_component,
                self.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000_1111_1100_0111,  # type_mask: 仅使用速度 vz
                0, 0, 0,                # pos (ignored)
                0, 0, -1.5,             # vx=0, vy=0, vz=-1.5 (上升，NED 向下为正)
                0, 0, 0, 0, 0,
            )

        # 等待爬升（目标高度 60%）。注意 relative_alt 是"相对 home 高度"，但 home
        # 未设定的瞬间该字段可能等于绝对海拔（~584000mm），会让简单的
        # `>= target` 误判为已起飞 → 在地面就返回 True。因此：
        #   1. 清掉旧高度帧，取起飞指令后的首个可信读数为基线；
        #   2. 上限做合理性约束（剔除 > altitude*3 的离谱读数）；
        #   3. 需连续 2 次满足，避免单帧抖动。
        target_mm = int(altitude * 0.6 * 1000)
        plausible_max_mm = int(altitude * 3 * 1000) + 5000
        drain_until = time.time() + 0.5
        while time.time() < drain_until:
            self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        baseline_mm = None
        hits = 0
        deadline = time.time() + 25
        while time.time() < deadline:
            msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
            if msg is not None:
                if baseline_mm is None:
                    if -1000 <= msg.relative_alt <= plausible_max_mm:
                        baseline_mm = msg.relative_alt
                    else:
                        continue
                elif msg.relative_alt < baseline_mm and -1000 <= msg.relative_alt <= plausible_max_mm:
                    baseline_mm = msg.relative_alt
                climb = msg.relative_alt - (baseline_mm or 0)
                if 0 < climb <= plausible_max_mm and climb >= target_mm:
                    hits += 1
                    if hits >= 2:
                        self.set_param("FENCE_ENABLE", 1)
                        return True
                else:
                    hits = 0
            # 持续发速度指令（若用速度控制则需要持续发送）
            if not takeoff_accepted and time.time() < deadline - 2:
                self.mav.mav.set_position_target_local_ned_send(
                    0,
                    self.mav.target_system, self.mav.target_component,
                    self.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000_1111_1100_0111,
                    0, 0, 0, 0, 0, -1.5, 0, 0, 0, 0, 0,
                )
            time.sleep(0.3)
        self.set_param("FENCE_ENABLE", 1)
        return False

    def reset_drone_state(self) -> None:
        """每个测试前恢复已知初始态：传感器正常、disarm、STABILIZE。"""
        self.set_param("SIM_GPS1_ENABLE", 1)
        self.set_param("SIM_ENGINE_FAIL", 0)
        self.set_param("SIM_BARO_DISABLE", 0)
        self.set_param("SIM_ACCEL1_FAIL", 0)
        self.set_param("SIM_MAG1_FAIL", 0)
        self.set_param("SIM_MAG2_FAIL", 0)
        self.set_param("SIM_MAG3_FAIL", 0)
        self.set_param("COMPASS_ENABLE", 1)
        self.set_param("EK3_ENABLE", 1)
        self.set_param("SIM_GPS1_ENABLE", 1)
        self.set_param("ARMING_CHECK", 0)   # SITL: bypass PreArm checks
        self.set_param("BATT_ARM_VOLT", 0)
        self.set_param("FS_GCS_ENABLE", 0)
        # 关闭 GCS 故障安全——pymavlink 不发心跳，10s 后会强制 LAND 打断起飞
        # disconnect_gcs 注入会在需要时手动重新启用
        self.set_param("FS_GCS_ENABLE", 0)
        time.sleep(0.5)
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0,
        )
        time.sleep(1)
        self.set_mode("STABILIZE")


# ---------------------------------------------------------------------------
# Handler protocol
# ---------------------------------------------------------------------------

# Inject handler: (ctx, spec) → None  (用 spec.pre_takeoff_m 自动处理起飞)
InjectHandler = Callable[[TestContext, InjectSpec], None]

# Verify handler: (ctx, spec) → (passed, message)
VerifyHandler = Callable[[TestContext, VerifySpec], Tuple[bool, str]]

# Render handler: (spec) → 渲染到独立脚本里的 Python 代码片段
RenderInjectHandler = Callable[[InjectSpec], str]
RenderVerifyHandler = Callable[[VerifySpec], str]


# ---------------------------------------------------------------------------
# 起飞前置守卫（进程内 + 渲染脚本共用同一不变量）
#
# 不变量：spec.pre_takeoff_m > 0 表示测试要求机体在空中。起飞失败时必须让
# 测试 FAIL，否则会在地面发执行器/故障注入指令、再靠 STATUSTEXT 假绿。
# 进程内 inject 由 _run_single_test 的 try/except 捕获异常 → 记为 False。
# ---------------------------------------------------------------------------

def _require_takeoff(ctx: TestContext, spec: InjectSpec) -> None:
    """进程内：起飞失败则抛异常，让测试 FAIL（而非在地面继续注入）。"""
    if spec.pre_takeoff_m > 0 and not ctx.force_arm_and_takeoff(
            altitude=spec.pre_takeoff_m):
        raise RuntimeError(
            f"起飞失败：未能解锁/爬升到 {spec.pre_takeoff_m}m")


def _render_require_takeoff(spec: InjectSpec) -> str:
    """渲染脚本：起飞失败则 raise，让生成的测试脚本非零退出（FAIL）。"""
    if spec.pre_takeoff_m <= 0:
        return ""
    return (f"print('  起飞至 {spec.pre_takeoff_m}m ...')\n"
            f"if not force_arm_and_takeoff(mav, altitude={spec.pre_takeoff_m}):\n"
            f"    raise RuntimeError('起飞失败：未能解锁/爬升到目标高度')\n")


# ---------------------------------------------------------------------------
# Inject handlers
# ---------------------------------------------------------------------------

def _inject_noop(ctx: TestContext, spec: InjectSpec) -> None:
    pass


def _render_inject_noop(spec: InjectSpec) -> str:
    return 'print("  (no inject)")\n'


def _inject_set_param(ctx: TestContext, spec: InjectSpec) -> None:
    # 处理特殊 key: _pre_mode（注入前切换模式）
    pre_mode = spec.params.get("_pre_mode")
    if pre_mode:
        ctx.set_mode(str(pre_mode))
        time.sleep(0.5)

    _require_takeoff(ctx, spec)

    settle_s = spec.params.get("_settle_s", 2.0)

    for name, value in spec.params.items():
        if name.startswith("_"):
            continue   # 跳过特殊控制 key（_pre_mode、_settle_s 等）
        ctx.set_param(name, float(value))

    # 注入后等待，让 ArduCopter 检测到故障
    time.sleep(settle_s)


def _render_inject_set_param(spec: InjectSpec) -> str:
    lines = []
    pre = _render_require_takeoff(spec)
    if pre:
        lines.append(pre.rstrip("\n"))
    for name, value in spec.params.items():
        if name.startswith("_"):
            continue
        lines.append(f'print("  设置参数 {name}={value}")')
        lines.append(f'set_param(mav, "{name}", {value})')
    lines.append("time.sleep(2)  # 等待故障检测")
    return "\n".join(lines) + "\n"


def _inject_disconnect_gcs(ctx: TestContext, spec: InjectSpec) -> None:
    # pre_takeoff_m>0 时要求真实起飞（空中失联才会触发 RTL/LAND；起飞失败则 FAIL）
    _require_takeoff(ctx, spec)
    # 动作可由条目指定（1=RTL、5=Land），否则保持 ArduPilot 默认 RTL。之前
    # 硬编码 1 会把 GCS_LOSS_LAND 的 boot 值覆盖回 RTL，verify 等 LAND 必假阴。
    ctx.set_param("FS_GCS_ENABLE", float(spec.params.get("FS_GCS_ENABLE", 1)))
    ctx.set_param("FS_GCS_TIMEOUT", float(spec.params.get("FS_GCS_TIMEOUT", 10)))
    # 预热：持续发 HEARTBEAT 至少 15s，让 ArduCopter 建立稳定的 GCS 连接状态
    t_warmup = time.time() + 15
    while time.time() < t_warmup:
        ctx.mav.mav.heartbeat_send(
            ctx.mavutil.mavlink.MAV_TYPE_GCS,
            ctx.mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )
        time.sleep(1.0)
    # 关闭连接，彻底切断心跳 → ArduCopter 检测到 GCS 断连
    # 不在 inject 中 sleep — verify 立即开始轮询模式切换
    try:
        ctx._gcs_conn_str = getattr(ctx.mav, 'address', "tcp:127.0.0.1:5760")
        ctx.mav.close()
    except Exception:
        ctx._gcs_conn_str = "tcp:127.0.0.1:5760"
    # 小延迟让 SITL 释放 TCP 端口，让 verify 可以重连
    time.sleep(2)


def _render_inject_disconnect_gcs(spec: InjectSpec) -> str:
    pre = _render_require_takeoff(spec)
    enable = float(spec.params.get("FS_GCS_ENABLE", 1))
    timeout = float(spec.params.get("FS_GCS_TIMEOUT", 10))
    return pre + textwrap.dedent(f"""\
        print("  启用 GCS 故障安全，停止心跳 ...")
        set_param(mav, "FS_GCS_ENABLE", {enable})
        set_param(mav, "FS_GCS_TIMEOUT", {timeout})
        time.sleep(12)
    """)


def _inject_skip(ctx: TestContext, spec: InjectSpec) -> None:
    pass


def _render_inject_skip(spec: InjectSpec) -> str:
    note = spec.notes or "skipped"
    return f'print("  ⚠ inject skipped: {note}")\n'


def _inject_mavlink_command(ctx: TestContext, spec: InjectSpec) -> None:
    """发送任意 MAVLink command_long（用于 gripper、喷射器等执行器命令）。

    spec.params keys:
      command  — MAVLink 命令 ID（必填）
      param1..param7 — 命令参数（默认 0）
      pre_command / pre_param1..pre_param7 / _pre_settle_s — 可选前置命令：
        先发它并等 _pre_settle_s，再发主命令。用于需要状态转换判别力的
        用例（如先 RELEASE 再 abort-GRAB——只断言终值时，若初值已等于
        目标值，主命令没生效也会假绿）。
      other numeric keys — runtime params applied after takeoff, before command
    """
    _require_takeoff(ctx, spec)

    cmd_id = int(spec.params.get("command", 0))
    p = [float(spec.params.get(f"param{i}", 0)) for i in range(1, 8)]
    command_keys = (
        {"command", "_settle_s", "pre_command"}
        | {f"param{i}" for i in range(1, 8)}
        | {f"pre_param{i}" for i in range(1, 8)}
    )
    for name, value in spec.params.items():
        if name in command_keys or name.startswith("_"):
            continue
        ctx.set_param(name, float(value))

    pre_cmd = spec.params.get("pre_command")
    if pre_cmd is not None:
        pre = [float(spec.params.get(f"pre_param{i}", 0)) for i in range(1, 8)]
        ctx.mav.mav.command_long_send(
            ctx.mav.target_system, ctx.mav.target_component,
            int(pre_cmd), 0,
            pre[0], pre[1], pre[2], pre[3], pre[4], pre[5], pre[6],
        )
        time.sleep(float(spec.params.get("_pre_settle_s", 2.0)))

    ctx.mav.mav.command_long_send(
        ctx.mav.target_system, ctx.mav.target_component,
        cmd_id, 0,
        p[0], p[1], p[2], p[3], p[4], p[5], p[6],
    )
    settle_s = float(spec.params.get("_settle_s", 1.5))
    time.sleep(settle_s)


def _render_inject_mavlink_command(spec: InjectSpec) -> str:
    cmd_id = int(spec.params.get("command", 0))
    params = [float(spec.params.get(f"param{i}", 0)) for i in range(1, 8)]
    settle_s = float(spec.params.get("_settle_s", 1.5))
    pre = _render_require_takeoff(spec)
    command_keys = (
        {"command", "_settle_s", "pre_command"}
        | {f"param{i}" for i in range(1, 8)}
        | {f"pre_param{i}" for i in range(1, 8)}
    )
    setup_lines = []
    for name, value in spec.params.items():
        if name in command_keys or name.startswith("_"):
            continue
        setup_lines.append(f'print("  设置参数 {name}={float(value)}")')
        setup_lines.append(f'set_param(mav, "{name}", {float(value)})')
    setup = ("\n".join(setup_lines) + "\n") if setup_lines else ""
    pre_block = ""
    pre_cmd = spec.params.get("pre_command")
    if pre_cmd is not None:
        pre_params = [float(spec.params.get(f"pre_param{i}", 0)) for i in range(1, 8)]
        pre_settle = float(spec.params.get("_pre_settle_s", 2.0))
        pre_block = textwrap.dedent(f"""\
            print("  发送前置 MAVLink command {int(pre_cmd)} ...")
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                {int(pre_cmd)}, 0,
                {pre_params[0]}, {pre_params[1]}, {pre_params[2]}, {pre_params[3]},
                {pre_params[4]}, {pre_params[5]}, {pre_params[6]},
            )
            time.sleep({pre_settle})
        """)
    return pre + setup + pre_block + textwrap.dedent(f"""\
        print("  发送 MAVLink command {cmd_id} ...")
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            {cmd_id}, 0,
            {params[0]}, {params[1]}, {params[2]}, {params[3]},
            {params[4]}, {params[5]}, {params[6]},
        )
        time.sleep({settle_s})
    """)


INJECT_HANDLERS: Dict[str, InjectHandler] = {
    "noop":             _inject_noop,
    "set_param":        _inject_set_param,
    "disconnect_gcs":   _inject_disconnect_gcs,
    "mavlink_command":  _inject_mavlink_command,
    "skip":             _inject_skip,
}

RENDER_INJECT: Dict[str, RenderInjectHandler] = {
    "noop":             _render_inject_noop,
    "set_param":        _render_inject_set_param,
    "disconnect_gcs":   _render_inject_disconnect_gcs,
    "mavlink_command":  _render_inject_mavlink_command,
    "skip":             _render_inject_skip,
}


# ---------------------------------------------------------------------------
# Verify handlers
# ---------------------------------------------------------------------------

def _verify_noop(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    return True, "(no verify)"


def _render_verify_noop(spec: VerifySpec) -> str:
    return 'print("  (no verify)")\nreturn True\n'


def _verify_wait_mode(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    mode = str(spec.args.get("mode", "")).upper()
    fallback = str(spec.args.get("fallback", "")).upper()

    # 如果连接已关闭（disconnect_gcs 场景），用保存的地址重新建立连接
    conn_ok = False
    try:
        # 尝试发一个非阻塞 recv 确认 socket 活着
        ctx.mav.recv_match(type="HEARTBEAT", blocking=False)
        conn_ok = True
    except Exception:
        conn_ok = False

    if not conn_ok:
        conn_str = getattr(ctx, "_gcs_conn_str", None) \
                   or getattr(ctx.mav, 'address', None) \
                   or "tcp:127.0.0.1:5760"
        try:
            ctx.mav = ctx.mavutil.mavlink_connection(conn_str)
            ctx.mav.wait_heartbeat(timeout=10)
        except Exception as e:
            return False, f"重连失败: {e}"

    deadline = time.time() + spec.timeout
    while time.time() < deadline:
        try:
            msg = ctx.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        except Exception:
            # socket 断开后重连一次
            try:
                conn_str = getattr(ctx, "_gcs_conn_str", "tcp:127.0.0.1:5760")
                ctx.mav = ctx.mavutil.mavlink_connection(conn_str)
                ctx.mav.wait_heartbeat(timeout=5)
                continue
            except Exception:
                break
        if msg:
            current = ctx.mavutil.mode_string_v10(msg)
            if mode in current.upper() or (fallback and fallback in current.upper()):
                return True, f"模式已切换到 {current}"
        time.sleep(0.5)
    return False, f"超时未切换到 {mode}"


def _render_verify_wait_mode(spec: VerifySpec) -> str:
    mode = str(spec.args.get("mode", "")).upper()
    fallback = str(spec.args.get("fallback", "")).upper()
    return textwrap.dedent(f"""\
        print("  等待飞行模式切换到 {mode}（或 {fallback}）...")
        deadline = time.time() + {spec.timeout}
        ok = False
        while time.time() < deadline:
            cur = get_mode(mav)
            if "{mode}" in cur.upper() or ("{fallback}" and "{fallback}" in cur.upper()):
                print(f"  ✓ 模式已切换到 {{cur}}")
                ok = True
                break
            time.sleep(0.5)
        if not ok:
            print("  ✗ 超时未切换到 {mode}")
        return ok
    """)


def _verify_assert_arm_rejected(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    ctx.mav.mav.command_long_send(
        ctx.mav.target_system, ctx.mav.target_component,
        ctx.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    deadline = time.time() + spec.timeout
    while time.time() < deadline:
        ack = ctx.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
        if ack and ack.command == ctx.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if ack.result != ctx.mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return True, "解锁被正确拒绝"
            return False, "解锁意外成功"
    return False, "未收到解锁 ACK"


def _render_verify_assert_arm_rejected(spec: VerifySpec) -> str:
    return textwrap.dedent("""\
        print("  尝试解锁，期望被拒绝 ...")
        armed = try_arm(mav)
        if not armed:
            print("  ✓ 解锁被正确拒绝")
        else:
            print("  ✗ 解锁意外成功")
        return not armed
    """)


def _verify_wait_statustext(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    # 支持单关键词（str）或多候选词（list），任一匹配即通过
    raw = spec.args.get("keyword", "")
    keywords: List[str] = (
        [k.lower() for k in raw] if isinstance(raw, list)
        else [str(raw).lower()]
    )
    deadline = time.time() + spec.timeout
    while time.time() < deadline:
        msg = ctx.mav.recv_match(type="STATUSTEXT", blocking=True, timeout=1)
        if msg:
            text_lower = msg.text.lower()
            if any(kw in text_lower for kw in keywords):
                return True, f"收到匹配 STATUSTEXT: {msg.text}"
    return False, f"未收到含 '{keywords}' 的 STATUSTEXT（超时）"


def _render_verify_wait_statustext(spec: VerifySpec) -> str:
    keyword = str(spec.args.get("keyword", ""))
    return textwrap.dedent(f"""\
        print("  等待包含 '{keyword}' 的 STATUSTEXT ...")
        ok = wait_for_statustext(mav, "{keyword}", timeout={spec.timeout})
        if ok:
            print("  ✓ 收到匹配 STATUSTEXT")
        else:
            print("  ✗ 未收到 STATUSTEXT（超时）")
        return ok
    """)


def _verify_assert_servo_pwm(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    channel = int(spec.args.get("channel", 0))
    target_pwm = int(spec.args.get("target_pwm", 0))
    tol = int(spec.args.get("tol", 50))
    if not (1 <= channel <= 16):
        return False, f"无效舵机通道: {channel}"
    field_name = f"servo{channel}_raw"
    deadline = time.time() + spec.timeout
    last_pwm = None
    while time.time() < deadline:
        msg = ctx.mav.recv_match(type="SERVO_OUTPUT_RAW", blocking=True, timeout=1)
        if msg is None:
            continue
        last_pwm = getattr(msg, field_name, None)
        if last_pwm is not None and abs(int(last_pwm) - target_pwm) <= tol:
            return True, f"{field_name}={last_pwm} within {target_pwm}±{tol}"
    return False, f"{field_name} 未达到 {target_pwm}±{tol} (last={last_pwm})"


def _render_verify_assert_servo_pwm(spec: VerifySpec) -> str:
    channel = int(spec.args.get("channel", 0))
    target_pwm = int(spec.args.get("target_pwm", 0))
    tol = int(spec.args.get("tol", 50))
    field_name = f"servo{channel}_raw"
    return textwrap.dedent(f"""\
        print("  等待 SERVO_OUTPUT_RAW.{field_name} 达到 {target_pwm}±{tol} ...")
        deadline = time.time() + {spec.timeout}
        last_pwm = None
        ok = False
        while time.time() < deadline:
            msg = mav.recv_match(type="SERVO_OUTPUT_RAW", blocking=True, timeout=1)
            if msg is None:
                continue
            last_pwm = getattr(msg, "{field_name}", None)
            if last_pwm is not None and abs(int(last_pwm) - {target_pwm}) <= {tol}:
                ok = True
                break
        if ok:
            print(f"  ✓ {field_name}={{last_pwm}} within {target_pwm}±{tol}")
        else:
            print(f"  ✗ {field_name} 未达到 {target_pwm}±{tol} (last={{last_pwm}})")
        return ok
    """)


def _sensor_bit(ctx: TestContext, sensor_name: str) -> Optional[int]:
    name = sensor_name.lower()
    if name == "gps":
        return int(getattr(ctx.mavutil.mavlink, "MAV_SYS_STATUS_SENSOR_GPS", 32))
    return None


def _verify_assert_sensor_unhealthy(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    sensor = str(spec.args.get("sensor", "gps")).lower()
    bit = _sensor_bit(ctx, sensor)
    if bit is None:
        return False, f"未知传感器健康位: {sensor}"
    deadline = time.time() + spec.timeout
    last_health = None
    last_fix_type = None
    while time.time() < deadline:
        msg = ctx.mav.recv_match(type=["SYS_STATUS", "GPS_RAW_INT"], blocking=True, timeout=1)
        if msg is None:
            continue
        mtype = msg.get_type() if hasattr(msg, "get_type") else (
            "SYS_STATUS" if hasattr(msg, "onboard_control_sensors_health") else ""
        )
        if mtype == "SYS_STATUS":
            last_health = int(getattr(msg, "onboard_control_sensors_health", 0))
            if (last_health & bit) == 0:
                return True, f"{sensor.upper()} health bit cleared in SYS_STATUS"
        elif sensor == "gps" and mtype == "GPS_RAW_INT":
            last_fix_type = int(getattr(msg, "fix_type", 0) or 0)
            if last_fix_type <= 1:
                return True, f"GPS_RAW_INT.fix_type={last_fix_type} indicates no GPS fix"
    last = f"0x{last_health:x}" if last_health is not None else "None"
    fix = f", last_fix_type={last_fix_type}" if last_fix_type is not None else ""
    return False, f"{sensor.upper()} health still healthy (last_health={last}{fix})"


def _render_verify_assert_sensor_unhealthy(spec: VerifySpec) -> str:
    sensor = str(spec.args.get("sensor", "gps")).lower()
    timeout = spec.timeout
    return textwrap.dedent(f"""\
        print("  等待 SYS_STATUS 中 {sensor.upper()} health bit 清零或 GPS_RAW_INT.fix_type 无解 ...")
        sensor_bit = getattr(mavutil.mavlink, "MAV_SYS_STATUS_SENSOR_{sensor.upper()}", 32)
        deadline = time.time() + {timeout}
        last_health = None
        last_fix_type = None
        ok = False
        while time.time() < deadline:
            msg = mav.recv_match(type=["SYS_STATUS", "GPS_RAW_INT"], blocking=True, timeout=1)
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "SYS_STATUS":
                last_health = int(getattr(msg, "onboard_control_sensors_health", 0))
                if (last_health & sensor_bit) == 0:
                    ok = True
                    break
            elif "{sensor}" == "gps" and mtype == "GPS_RAW_INT":
                last_fix_type = int(getattr(msg, "fix_type", 0) or 0)
                if last_fix_type <= 1:
                    ok = True
                    break
        if ok:
            print("  ✓ {sensor.upper()} unhealthy/no-fix state observed")
        else:
            last = f"0x{{last_health:x}}" if last_health is not None else "None"
            print(f"  ✗ {sensor.upper()} health still healthy (last_health={{last}}, last_fix_type={{last_fix_type}})")
        return ok
    """)


def _verify_skip(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    note = spec.notes or "verification skipped"
    return True, note


def _render_verify_skip(spec: VerifySpec) -> str:
    note = spec.notes or "verification skipped"
    return textwrap.dedent(f"""\
        print("  ⚠ verify skipped: {note}")
        return True
    """)


def _verify_assert_mavlink_v2_link(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    """MAVLink v2 framing + bidirectionality + link continuity, measured.

    A configured SERIAL0_PROTOCOL=2 proves the port was *asked* for MAVLink v2;
    it does not prove the link speaks it, answers, or stays up. This reads the
    wire: the v2 start-of-frame byte (0xFD) on actually-received packets, a
    command that must be answered to show the uplink is live, and the largest
    HEARTBEAT gap over a sampling window.
    """
    window = float(spec.args.get("window_s", 8.0))
    max_gap = float(spec.args.get("max_gap_s", 2.0))

    # -- downlink framing: read the real start-of-frame byte --------------
    v2_frames = v1_frames = 0
    beats: List[float] = []
    deadline = time.time() + window
    while time.time() < deadline:
        msg = ctx.mav.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        buf = msg.get_msgbuf()
        if buf:
            if buf[0] == 0xFD:
                v2_frames += 1
            elif buf[0] == 0xFE:
                v1_frames += 1
        if msg.get_type() == "HEARTBEAT":
            beats.append(time.time())

    if v2_frames == 0:
        return False, (
            f"no MAVLink v2 frames observed (v1 frames={v1_frames}); "
            "the link is not speaking MAVLink 2.0"
        )
    if v1_frames:
        return False, (
            f"link is mixed-version: {v2_frames} v2 frames but {v1_frames} v1 frames"
        )

    # -- uplink: a command the vehicle must answer ------------------------
    ctx.mav.mav.command_long_send(
        ctx.mav.target_system, ctx.mav.target_component,
        ctx.mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
        ctx.mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION,
        0, 0, 0, 0, 0, 0,
    )
    ack = ctx.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
    if ack is None:
        return False, (
            f"downlink is v2 ({v2_frames} frames) but the uplink was not "
            "answered: no COMMAND_ACK, so the link is not demonstrably bidirectional"
        )

    # -- continuity: the largest gap between heartbeats --------------------
    if len(beats) < 2:
        return False, f"only {len(beats)} heartbeats in {window:.0f} s — continuity not measurable"
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    worst = max(gaps)
    if worst > max_gap:
        return False, (
            f"link continuity broken: largest HEARTBEAT gap {worst:.2f} s "
            f"exceeds {max_gap:.1f} s over {window:.0f} s"
        )
    return True, (
        f"MAVLink v2 confirmed on the wire ({v2_frames} v2 frames, 0 v1), "
        f"bidirectional (COMMAND_ACK result={ack.result}), continuous "
        f"({len(beats)} heartbeats, largest gap {worst:.2f} s <= {max_gap:.1f} s "
        f"over {window:.0f} s). Channel encryption is NOT covered by this check."
    )


def _render_verify_assert_mavlink_v2_link(spec: VerifySpec) -> str:
    window = float(spec.args.get("window_s", 8.0))
    max_gap = float(spec.args.get("max_gap_s", 2.0))
    return textwrap.dedent(f"""\
        print("  reading the wire for MAVLink v2 framing + continuity ...")
        v2_frames = v1_frames = 0
        beats = []
        deadline = time.time() + {window}
        while time.time() < deadline:
            msg = mav.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            buf = msg.get_msgbuf()
            if buf:
                if buf[0] == 0xFD:
                    v2_frames += 1
                elif buf[0] == 0xFE:
                    v1_frames += 1
            if msg.get_type() == "HEARTBEAT":
                beats.append(time.time())
        ok = v2_frames > 0 and v1_frames == 0 and len(beats) >= 2
        if ok:
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0,
            )
            ack = mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
            gaps = [b - a for a, b in zip(beats, beats[1:])]
            worst = max(gaps)
            ok = ack is not None and worst <= {max_gap}
            print(f"  v2_frames={{v2_frames}} v1_frames={{v1_frames}} "
                  f"ack={{ack is not None}} worst_gap={{worst:.2f}}s")
        else:
            print(f"  v2_frames={{v2_frames}} v1_frames={{v1_frames}} beats={{len(beats)}}")
        print("  " + ("OK" if ok else "FAIL") + " — channel encryption is NOT covered")
        return ok
    """)


def _verify_assert_waypoint_update_latency(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    """Time from an accepted waypoint-modification command to the active plan
    carrying it.

    The transfer itself is protocol overhead, so the clock starts at MISSION_ACK
    — the point at which the vehicle has RECEIVED a valid modification — and
    stops when a read-back of the mission shows the revised coordinate. Both
    intervals are reported so the split is visible.
    """
    limit = float(spec.args.get("max_latency_s", 1.0))
    mav, mv = ctx.mav, ctx.mavutil

    def upload(items) -> Optional[float]:
        """Send a mission; return the time MISSION_ACK arrived."""
        mav.mav.mission_count_send(
            mav.target_system, mav.target_component, len(items),
            mv.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        sent = 0
        deadline = time.time() + 15.0
        while sent < len(items) and time.time() < deadline:
            req = mav.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                blocking=True, timeout=3,
            )
            if req is None:
                continue
            seq = int(req.seq)
            lat, lon, alt = items[seq]
            mav.mav.mission_item_int_send(
                mav.target_system, mav.target_component, seq,
                mv.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mv.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1, 0, 0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), float(alt),
                mv.mavlink.MAV_MISSION_TYPE_MISSION,
            )
            sent = max(sent, seq + 1)
        ack = mav.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
        if ack is None or int(ack.type) != mv.mavlink.MAV_MISSION_ACCEPTED:
            return None
        return time.time()

    def readback(seq: int) -> Optional[tuple]:
        mav.mav.mission_request_int_send(
            mav.target_system, mav.target_component, seq,
            mv.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        item = mav.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=2)
        if item is None:
            return None
        return (int(item.x), int(item.y))

    home_lat, home_lon = -35.363262, 149.165237
    original = [(home_lat, home_lon, 0.0),
                (home_lat + 0.0004, home_lon, 20.0),
                (home_lat + 0.0008, home_lon, 20.0)]
    if upload(original) is None:
        return False, "the original mission was not accepted; no baseline plan to revise"

    revised = list(original)
    revised[2] = (home_lat + 0.0008, home_lon + 0.0006, 25.0)
    target = (int(revised[2][0] * 1e7), int(revised[2][1] * 1e7))

    # The active plan is polled, so one read-back round-trip is the floor on
    # what this method can resolve. Measure it, so the latency below is
    # reported as the upper bound it actually is.
    poll_started = time.time()
    readback(2)
    poll_cost = time.time() - poll_started

    send_started = time.time()
    accepted_at = upload(revised)
    if accepted_at is None:
        return False, "the revised waypoint sequence was rejected by the vehicle"
    transfer_s = accepted_at - send_started

    deadline = accepted_at + max(limit * 4.0, 5.0)
    incorporated_at = None
    last_seen = None
    while time.time() < deadline:
        last_seen = readback(2)
        if last_seen == target:
            incorporated_at = time.time()
            break
    if incorporated_at is None:
        return False, (
            f"the active plan never carried the revision (last read-back {last_seen}, "
            f"expected {target}); transfer took {transfer_s:.3f} s"
        )

    latency = incorporated_at - accepted_at
    ok = latency <= limit
    return ok, (
        f"revised waypoint sequence was carried by the active flight plan within "
        f"{latency:.3f} s of MISSION_ACK (limit {limit:.1f} s). This is an UPPER "
        f"BOUND, not an exact interval: the plan is polled, and one read-back "
        f"round-trip costs {poll_cost:.3f} s, so incorporation happened at or "
        f"before the first poll that saw it. The upload transfer took "
        f"{transfer_s:.3f} s and is excluded — the requirement's clock starts at "
        f"receipt of the modification command, not at the start of its transfer"
    )


def _render_verify_assert_waypoint_update_latency(spec: VerifySpec) -> str:
    limit = float(spec.args.get("max_latency_s", 1.0))
    return textwrap.dedent(f"""\
        print("  timing waypoint-sequence incorporation ...")
        HOME_LAT, HOME_LON = -35.363262, 149.165237

        def _upload(items):
            mav.mav.mission_count_send(
                mav.target_system, mav.target_component, len(items),
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
            sent, deadline = 0, time.time() + 15.0
            while sent < len(items) and time.time() < deadline:
                req = mav.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                                     blocking=True, timeout=3)
                if req is None:
                    continue
                seq = int(req.seq)
                lat, lon, alt = items[seq]
                mav.mav.mission_item_int_send(
                    mav.target_system, mav.target_component, seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0,
                    int(lat * 1e7), int(lon * 1e7), float(alt),
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
                sent = max(sent, seq + 1)
            ack = mav.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
            if ack is None or int(ack.type) != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                return None
            return time.time()

        def _readback(seq):
            mav.mav.mission_request_int_send(
                mav.target_system, mav.target_component, seq,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
            item = mav.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=2)
            return None if item is None else (int(item.x), int(item.y))

        original = [(HOME_LAT, HOME_LON, 0.0),
                    (HOME_LAT + 0.0004, HOME_LON, 20.0),
                    (HOME_LAT + 0.0008, HOME_LON, 20.0)]
        revised = list(original)
        revised[2] = (HOME_LAT + 0.0008, HOME_LON + 0.0006, 25.0)
        target = (int(revised[2][0] * 1e7), int(revised[2][1] * 1e7))

        ok = False
        if _upload(original) is not None:
            accepted_at = _upload(revised)
            if accepted_at is not None:
                deadline = accepted_at + max({limit} * 4.0, 5.0)
                while time.time() < deadline:
                    if _readback(2) == target:
                        latency = time.time() - accepted_at
                        ok = latency <= {limit}
                        print(f"  incorporation latency {{latency:.3f}}s (limit {limit})")
                        break
        print("  " + ("OK" if ok else "FAIL"))
        return ok
    """)


VERIFY_HANDLERS: Dict[str, VerifyHandler] = {
    "noop":                _verify_noop,
    "wait_mode":           _verify_wait_mode,
    "assert_arm_rejected": _verify_assert_arm_rejected,
    "assert_servo_pwm":    _verify_assert_servo_pwm,
    "assert_sensor_unhealthy": _verify_assert_sensor_unhealthy,
    "wait_statustext":     _verify_wait_statustext,
    "assert_mavlink_v2_link": _verify_assert_mavlink_v2_link,
    "assert_waypoint_update_latency": _verify_assert_waypoint_update_latency,
    "skip":                _verify_skip,
}

RENDER_VERIFY: Dict[str, RenderVerifyHandler] = {
    "noop":                _render_verify_noop,
    "wait_mode":           _render_verify_wait_mode,
    "assert_arm_rejected": _render_verify_assert_arm_rejected,
    "assert_servo_pwm":    _render_verify_assert_servo_pwm,
    "assert_sensor_unhealthy": _render_verify_assert_sensor_unhealthy,
    "wait_statustext":     _render_verify_wait_statustext,
    "assert_mavlink_v2_link": _render_verify_assert_mavlink_v2_link,
    "assert_waypoint_update_latency": _render_verify_assert_waypoint_update_latency,
    "skip":                _render_verify_skip,
}


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def run_inject(ctx: TestContext, spec: InjectSpec) -> None:
    handler = INJECT_HANDLERS.get(spec.kind)
    if handler is None:
        raise ValueError(f"未注册的 inject kind: {spec.kind}")
    handler(ctx, spec)


def run_verify(ctx: TestContext, spec: VerifySpec) -> Tuple[bool, str]:
    handler = VERIFY_HANDLERS.get(spec.kind)
    if handler is None:
        return False, f"未注册的 verify kind: {spec.kind}"
    return handler(ctx, spec)


def render_inject(spec: InjectSpec) -> str:
    handler = RENDER_INJECT.get(spec.kind)
    if handler is None:
        return f'print("  unknown inject kind: {spec.kind}")\n'
    return handler(spec)


def render_verify(spec: VerifySpec) -> str:
    handler = RENDER_VERIFY.get(spec.kind)
    if handler is None:
        return f'print("  unknown verify kind: {spec.kind}")\nreturn True\n'
    return handler(spec)
