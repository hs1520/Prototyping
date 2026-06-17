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
        #   1. 取首个读数为基线，按相对基线的爬升量判断；
        #   2. 上限做合理性约束（剔除 > altitude*3 的离谱读数）；
        #   3. 需连续 2 次满足，避免单帧抖动。
        target_mm = int(altitude * 0.6 * 1000)
        plausible_max_mm = int(altitude * 3 * 1000) + 5000
        baseline_mm = None
        hits = 0
        deadline = time.time() + 25
        while time.time() < deadline:
            msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
            if msg is not None:
                if baseline_mm is None or msg.relative_alt < baseline_mm:
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

    if spec.pre_takeoff_m > 0:
        ctx.force_arm_and_takeoff(altitude=spec.pre_takeoff_m)

    settle_s = spec.params.get("_settle_s", 2.0)

    for name, value in spec.params.items():
        if name.startswith("_"):
            continue   # 跳过特殊控制 key（_pre_mode、_settle_s 等）
        ctx.set_param(name, float(value))

    # 注入后等待，让 ArduCopter 检测到故障
    time.sleep(settle_s)


def _render_inject_set_param(spec: InjectSpec) -> str:
    lines = []
    if spec.pre_takeoff_m > 0:
        lines.append(f"print('  起飞至 {spec.pre_takeoff_m}m ...')")
        lines.append(f"force_arm_and_takeoff(mav, altitude={spec.pre_takeoff_m})")
    for name, value in spec.params.items():
        if name.startswith("_"):
            continue
        lines.append(f'print("  设置参数 {name}={value}")')
        lines.append(f'set_param(mav, "{name}", {value})')
    lines.append("time.sleep(2)  # 等待故障检测")
    return "\n".join(lines) + "\n"


def _inject_disconnect_gcs(ctx: TestContext, spec: InjectSpec) -> None:
    # 解锁（不需要真正起飞，地面解锁足以触发 GCS 故障安全）
    if spec.pre_takeoff_m > 0:
        ctx.force_arm_and_takeoff(altitude=spec.pre_takeoff_m)
    ctx.set_param("FS_GCS_ENABLE", 1)
    ctx.set_param("FS_GCS_TIMEOUT", 10)
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
    pre = (f"print('  起飞至 {spec.pre_takeoff_m}m ...')\n"
           f"force_arm_and_takeoff(mav, altitude={spec.pre_takeoff_m})\n"
           if spec.pre_takeoff_m > 0 else "")
    return pre + textwrap.dedent("""\
        print("  启用 GCS 故障安全，停止心跳 ...")
        set_param(mav, "FS_GCS_ENABLE", 1)
        set_param(mav, "FS_GCS_TIMEOUT", 10)
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
    """
    if spec.pre_takeoff_m > 0:
        ctx.force_arm_and_takeoff(altitude=spec.pre_takeoff_m)

    cmd_id = int(spec.params.get("command", 0))
    p = [float(spec.params.get(f"param{i}", 0)) for i in range(1, 8)]

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
    pre = ""
    if spec.pre_takeoff_m > 0:
        # 起飞失败必须让测试 FAIL，否则会在地面发执行器指令、靠 STATUSTEXT 假绿
        pre = (f"print('  起飞至 {spec.pre_takeoff_m}m ...')\n"
               f"if not force_arm_and_takeoff(mav, altitude={spec.pre_takeoff_m}):\n"
               f"    raise RuntimeError('起飞失败：未能解锁/爬升到目标高度')\n")
    return pre + textwrap.dedent(f"""\
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
    "wait_statustext":     _verify_wait_statustext,
    "skip":                _verify_skip,
}

RENDER_VERIFY: Dict[str, RenderVerifyHandler] = {
    "noop":                _render_verify_noop,
    "wait_mode":           _render_verify_wait_mode,
    "assert_arm_rejected": _render_verify_assert_arm_rejected,
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
