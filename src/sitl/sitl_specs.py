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

import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
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
    params: Dict[str, float] = field(default_factory=dict)
    pre_takeoff_m: float = 0.0
    notes: str = ""


@dataclass
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
    args: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 10.0
    notes: str = ""


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
      other numeric keys — runtime params applied after takeoff, before command
    """
    _require_takeoff(ctx, spec)

    cmd_id = int(spec.params.get("command", 0))
    p = [float(spec.params.get(f"param{i}", 0)) for i in range(1, 8)]
    command_keys = {"command", "_settle_s"} | {f"param{i}" for i in range(1, 8)}
    for name, value in spec.params.items():
        if name in command_keys or name.startswith("_"):
            continue
        ctx.set_param(name, float(value))

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
    command_keys = {"command", "_settle_s"} | {f"param{i}" for i in range(1, 8)}
    setup_lines = []
    for name, value in spec.params.items():
        if name in command_keys or name.startswith("_"):
            continue
        setup_lines.append(f'print("  设置参数 {name}={float(value)}")')
        setup_lines.append(f'set_param(mav, "{name}", {float(value)})')
    setup = ("\n".join(setup_lines) + "\n") if setup_lines else ""
    return pre + setup + textwrap.dedent(f"""\
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


VERIFY_HANDLERS: Dict[str, VerifyHandler] = {
    "noop":                _verify_noop,
    "wait_mode":           _verify_wait_mode,
    "assert_arm_rejected": _verify_assert_arm_rejected,
    "assert_servo_pwm":    _verify_assert_servo_pwm,
    "assert_sensor_unhealthy": _verify_assert_sensor_unhealthy,
    "wait_statustext":     _verify_wait_statustext,
    "skip":                _verify_skip,
}

RENDER_VERIFY: Dict[str, RenderVerifyHandler] = {
    "noop":                _render_verify_noop,
    "wait_mode":           _render_verify_wait_mode,
    "assert_arm_rejected": _render_verify_assert_arm_rejected,
    "assert_servo_pwm":    _render_verify_assert_servo_pwm,
    "assert_sensor_unhealthy": _render_verify_assert_sensor_unhealthy,
    "wait_statustext":     _render_verify_wait_statustext,
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
