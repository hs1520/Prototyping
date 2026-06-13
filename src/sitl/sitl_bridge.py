"""
sitl_bridge.py

Translates a SysMLLiteModel + platform_profile into:
  L1 — ArduPilot .parm parameter file  (static, no SITL needed)
  L2 — pymavlink test scripts           (requires running SITL)

Usage
-----
    from src.sitl.sitl_bridge import SITLBridge

    bridge = SITLBridge(model, output_dir="sitl_output")
    bridge.generate_l1()          # writes AutonomousDrone.parm
    bridge.generate_l2_scripts()  # writes test_*.py scripts
    bridge.run_l2(host="127.0.0.1", port=5760)  # execute against live SITL
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.sysml.lite_model import SysMLLiteModel
from src.sitl.requirement_linker import RequirementLinker, SITLTestSpec

# ArduCopter 二进制默认路径 — 服务器用 ~/ardupilot/，本地用 ~/PycharmProjects/ardupilot/
_SERVER_ARDUCOPTER = os.path.expanduser("~/ardupilot/build/sitl/bin/arducopter")
_LOCAL_ARDUCOPTER  = os.path.expanduser("~/PycharmProjects/ardupilot/build/sitl/bin/arducopter")
_DEFAULT_ARDUCOPTER = (
    _SERVER_ARDUCOPTER if os.path.exists(_SERVER_ARDUCOPTER) else _LOCAL_ARDUCOPTER
)


# ---------------------------------------------------------------------------
# Test result
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    req_id: str
    tier: str
    passed: bool
    message: str
    duration_s: float = 0.0


@dataclass
class BridgeReport:
    model_name: str
    parm_file: str
    l1_results: List[TestResult] = field(default_factory=list)
    l2_results: List[TestResult] = field(default_factory=list)

    def passed(self) -> bool:
        return all(r.passed for r in self.l1_results + self.l2_results)

    def summary(self) -> str:
        all_results = self.l1_results + self.l2_results
        n = len(all_results)
        ok = sum(1 for r in all_results if r.passed)
        lines = [
            f"SITL Bridge Report — {self.model_name}",
            f"  Passed: {ok}/{n}",
            "",
        ]
        for r in all_results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} [{r.tier}] {r.req_id:<20} {r.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SITLBridge
# ---------------------------------------------------------------------------

# ArduPilot Copter 预置 Platform Profile（可直接使用）
ARDUPILOT_COPTER_PROFILE: dict = {
    "platform": "ArduPilot Copter",
    "mode_vocabulary": [
        "CMD_STABILIZE", "CMD_GUIDED", "CMD_AUTO",
        "CMD_RTL",       "CMD_LAND",   "CMD_TAKEOFF",
        "CMD_LOITER",    "CMD_POSHOLD",
    ],
    "emergency_mode": "CMD_LAND",
    # 需要解锁+起飞后才能有意义测试的模式
    "requires_airborne": {"CMD_RTL", "CMD_LAND", "CMD_AUTO",
                          "CMD_LOITER", "CMD_POSHOLD"},
}


class SITLBridge:

    def __init__(
        self,
        model: SysMLLiteModel,
        output_dir: str = "sitl_output",
        connection_string: str = "tcp:127.0.0.1:5760",
        arducopter_bin: str = _DEFAULT_ARDUCOPTER,
        llm: Optional[Any] = None,
        platform_profile: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        """
        llm             : 可选 LLMInterface，启用语义标签分类。
        platform_profile: 可选 Platform Profile dict。
                          传入后 generate_l2_scripts() 额外生成 accept
                          模式切换测试（CMD_RTL → SET_MODE RTL 等）。
                          为 None 时跳过 accept 测试（S11 选项C）。
        """
        self._model = model
        self._output_dir = Path(output_dir)
        self._connection_string = connection_string
        self._arducopter_bin = arducopter_bin
        self._platform_profile = platform_profile or {}
        self._linker = RequirementLinker(model, llm=llm, verbose=verbose)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sitl_proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # SITL 进程管理
    # ------------------------------------------------------------------

    def launch_sitl(
        self,
        home: str = "51.4,-2.35,0,0",
        wait_s: float = 8.0,
    ) -> bool:
        """
        在后台启动 arducopter SITL（直接二进制，内置仿真器）。
        返回 True 表示进程已启动并监听 5760 端口。
        """
        if not Path(self._arducopter_bin).exists():
            print(f"  ✗ arducopter 二进制不存在: {self._arducopter_bin}")
            return False

        parm_path = self._output_dir / f"{self._model.name}.parm"
        if not parm_path.exists():
            self.generate_l1()

        cmd = [
            self._arducopter_bin,
            "--model", "+",
            "--home", home,
            "--wipe",                    # 清空 EEPROM，确保 --defaults 生效
            "--defaults", str(parm_path),
        ]
        self._sitl_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  ▶ ArduCopter SITL 启动中 (PID {self._sitl_proc.pid}) ...")
        time.sleep(wait_s)

        # 简单探测：尝试 TCP 连接
        import socket
        try:
            s = socket.create_connection(("127.0.0.1", 5760), timeout=3)
            s.close()
        except OSError:
            print(f"  ✗ SITL 启动超时，端口 5760 无响应")
            self.stop_sitl()
            return False

        # 等待 EKF 对齐 + GPS fix（arming 需要两者都就绪）
        try:
            from pymavlink import mavutil as _mu
            _mav = _mu.mavlink_connection("tcp:127.0.0.1:5760")
            _mav.wait_heartbeat(timeout=10)
            _mav.mav.request_data_stream_send(
                _mav.target_system, _mav.target_component,
                _mu.mavlink.MAV_DATA_STREAM_ALL, 4, 1,
            )
            # 等待 EKF IMU 对齐 + EKF origin 设置（两者都就绪才能解锁）
            # "EKF3 IMU0 origin set" 是 AHRS home 就绪的标志
            ekf_deadline = time.time() + 60
            aligned = 0
            origin_set = 0
            while time.time() < ekf_deadline:
                msg = _mav.recv_match(type="STATUSTEXT", blocking=True, timeout=2)
                if msg is None:
                    continue
                text = msg.text.lower()
                if "alignment complete" in text:
                    aligned += 1
                elif "origin set" in text:
                    origin_set += 1
                # 两个 IMU 对齐 + 至少一个 origin 设置好 → 可以解锁
                if aligned >= 2 and origin_set >= 1:
                    break
            _mav.close()
        except Exception:
            pass

        print(f"  ✓ SITL 已就绪，监听 tcp:127.0.0.1:5760")
        return True

    def stop_sitl(self) -> None:
        """停止后台 SITL 进程，等待端口完全释放后返回。"""
        if self._sitl_proc and self._sitl_proc.poll() is None:
            pid = self._sitl_proc.pid
            self._sitl_proc.terminate()
            try:
                self._sitl_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._sitl_proc.kill()
                self._sitl_proc.wait(timeout=3)
            print(f"  ■ ArduCopter SITL 已停止 (PID {pid})")
            self._sitl_proc = None

        # 等端口 5760 真正释放，再允许下一个 SITL 启动
        import socket
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                s = socket.create_connection(("127.0.0.1", 5760), timeout=0.5)
                s.close()
                time.sleep(0.5)   # 端口还在占用，继续等
            except OSError:
                break             # 端口已释放

    # ------------------------------------------------------------------
    # L1 — 参数文件生成
    # ------------------------------------------------------------------

    def generate_l1(self) -> Path:
        """生成 .parm 文件，返回文件路径。"""
        parm_content = self._linker.generate_parm_file()
        parm_path = self._output_dir / f"{self._model.name}.parm"
        parm_path.write_text(parm_content, encoding="utf-8")
        print(f"[L1] .parm 文件已写入: {parm_path}")
        return parm_path

    def validate_l1(self) -> List[TestResult]:
        """静态验证：检查所有 L1 参数是否已正确解析（无 <unresolved>）。"""
        results: List[TestResult] = []
        specs = [s for s in self._linker.generate_test_specs() if s.tier == "L1"]

        for spec in specs:
            unresolved = [
                p for p in spec.params
                if isinstance(p.value, str) and p.value.startswith("<unresolved")
            ]
            if unresolved:
                msg = f"未解析参数: {[p.param_name for p in unresolved]}"
                results.append(TestResult(spec.req_id, "L1", False, msg))
            else:
                param_strs = ", ".join(
                    f"{p.param_name}={p.value}" for p in spec.params
                )
                results.append(TestResult(spec.req_id, "L1", True, param_strs))

        return results

    # ------------------------------------------------------------------
    # L2 — pymavlink 测试脚本生成
    # ------------------------------------------------------------------

    def generate_l2_scripts(self) -> List[Path]:
        """
        为每个 L2 测试规格生成独立的 pymavlink 测试脚本。

        safety guard 测试：来自 RequirementLinker（电池/GCS/传感器等）。
        accept 模式切换测试：来自 platform_profile（CMD_RTL → SET_MODE RTL 等）。
        无 platform_profile 时跳过 accept 测试（S11 选项C）。
        """
        paths: List[Path] = []

        # ── Safety guard 测试（原有）────────────────────────────────────
        specs = [s for s in self._linker.generate_test_specs() if s.tier == "L2"]
        for spec in specs:
            code = self._render_test_script(spec)
            script_path = self._output_dir / f"test_{spec.req_id.lower()}.py"
            script_path.write_text(code, encoding="utf-8")
            paths.append(script_path)
            print(f"[L2] 脚本已生成: {script_path}")

        # ── Accept 模式切换测试（需要 platform_profile）────────────────
        if self._platform_profile:
            accept_paths = self._generate_accept_mode_scripts()
            paths.extend(accept_paths)

        return paths

    def _cmd_to_mavlink_mode(self, cmd_name: str) -> Optional[str]:
        """
        CMD_RTL → "RTL"，CMD_LAND → "LAND" 等。
        去掉 CMD_ 前缀；不在词汇表里的命令返回 None。
        """
        vocab = self._platform_profile.get("mode_vocabulary", [])
        if cmd_name not in vocab:
            return None
        return cmd_name[4:] if cmd_name.startswith("CMD_") else None

    def _generate_accept_mode_scripts(self) -> List[Path]:
        """
        扫描模型里 accept-triggered 状态机的 nominal 转移，
        对每个能映射到 MAVLink 模式的命令生成 L2 模式切换测试脚本。

        CMD_X → SET_MODE X → wait_mode(X)
        """
        try:
            from src.simulation.state_extractor import extract_state_machines
            from src.simulation.behavioral_sim import _classify_accept_transitions
        except ImportError:
            return []

        sysml_text = self._model.to_sysml_text() or ""
        state_machines = extract_state_machines(sysml_text)
        requires_airborne = self._platform_profile.get("requires_airborne", set())

        paths: List[Path] = []
        seen_modes: set = set()  # 每个 MAVLink 模式只生成一个测试

        for sm in state_machines:
            if not sm.has_accept_transitions():
                continue
            nominal_trs, _ = _classify_accept_transitions(sm)
            for tr in nominal_trs:
                cmd = tr.accept_trigger or ""
                mode = self._cmd_to_mavlink_mode(cmd)
                if not mode or mode in seen_modes:
                    continue
                seen_modes.add(mode)
                airborne = cmd in requires_airborne
                code = self._render_accept_mode_script(cmd, mode, airborne)
                script_path = self._output_dir / f"test_mode_{mode.lower()}.py"
                script_path.write_text(code, encoding="utf-8")
                paths.append(script_path)
                print(f"[L2-ACCEPT] 脚本已生成: {script_path}  ({cmd} → SET_MODE {mode})")

        return paths

    def _render_accept_mode_script(
        self, cmd_name: str, mavlink_mode: str, requires_airborne: bool
    ) -> str:
        """渲染一个 MAVLink 模式切换测试脚本。"""
        conn = self._connection_string
        out  = self._output_dir
        pre  = ""
        if requires_airborne:
            pre = (
                "    print('  ARM + TAKEOFF to 5m ...')\n"
                "    if not force_arm_and_takeoff(mav, altitude=5.0):\n"
                "        print('FAIL  takeoff failed')\n"
                "        sys.exit(1)\n"
            )

        return "\n".join([
            "#!/usr/bin/env python3",
            f'"""',
            f"Auto-generated SITL accept-mode test: {cmd_name} → SET_MODE {mavlink_mode}",
            f"",
            f"Verifies that the ArduPilot flight controller accepts the MAVLink",
            f"mode switch corresponding to SysML accept trigger {cmd_name!r}.",
            f"",
            f"Run:",
            f"    python {out}/test_mode_{mavlink_mode.lower()}.py",
            f"(requires ArduPilot SITL running on {conn})",
            f'"""',
            "",
            "import sys, time",
            "from pymavlink import mavutil",
            "",
            f'CONNECTION = "{conn}"',
            "",
            "",
            "def connect():",
            "    mav = mavutil.mavlink_connection(CONNECTION)",
            "    mav.wait_heartbeat(timeout=10)",
            "    return mav",
            "",
            "",
            "def set_param(mav, name, value):",
            "    mav.mav.param_set_send(",
            "        mav.target_system, mav.target_component,",
            "        name.encode(), float(value),",
            "        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,",
            "    )",
            "    time.sleep(0.3)",
            "",
            "",
            "def force_arm_and_takeoff(mav, altitude=5.0):",
            "    set_param(mav, 'SIM_GPS1_ENABLE', 1)",
            "    set_param(mav, 'FENCE_ENABLE', 0)",
            "    mapping = mav.mode_mapping() or {}",
            "    guided_id = mapping.get('GUIDED') or 4",
            "    mav.mav.set_mode_send(mav.target_system,",
            "        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, guided_id)",
            "    time.sleep(0.5)",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,",
            "        0, 1, 0, 0, 0, 0, 0, 0,",
            "    )",
            "    ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)",
            "    if not ack or ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:",
            "        return False",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,",
            "        0, 0, 0, 0, 0, 0, 0, altitude,",
            "    )",
            "    deadline = time.time() + 20",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='GLOBAL_POSITION_INT',",
            "                             blocking=True, timeout=1)",
            "        if msg and msg.relative_alt >= int(altitude * 0.6 * 1000):",
            "            return True",
            "        time.sleep(0.3)",
            "    return False",
            "",
            "",
            "def set_mode(mav, mode_name):",
            "    mapping = mav.mode_mapping() or {}",
            "    mode_id = mapping.get(mode_name.upper())",
            "    if mode_id is None:",
            "        return False",
            "    mav.mav.set_mode_send(mav.target_system,",
            "        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)",
            "    time.sleep(0.5)",
            "    return True",
            "",
            "",
            "def wait_mode(mav, mode, timeout=15.0):",
            "    deadline = time.time() + timeout",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=2)",
            "        if msg:",
            "            cur = mavutil.mode_string_v10(msg)",
            "            if mode.upper() in cur.upper():",
            "                return True",
            "        time.sleep(0.5)",
            "    return False",
            "",
            "",
            "def main():",
            "    mav = connect()",
            f"    print('Testing SysML accept trigger: {cmd_name} → SET_MODE {mavlink_mode}')",
            pre.rstrip(),
            f"    print('  Sending SET_MODE {mavlink_mode} ...')",
            f"    if not set_mode(mav, '{mavlink_mode}'):",
            f"        print('FAIL  {cmd_name}: mode {mavlink_mode} not in ArduPilot mapping')",
            "        sys.exit(1)",
            f"    ok = wait_mode(mav, '{mavlink_mode}', timeout=10.0)",
            "    if ok:",
            f"        print('PASS  {cmd_name} → {mavlink_mode}')",
            "        sys.exit(0)",
            "    else:",
            f"        print('FAIL  {cmd_name}: timeout waiting for {mavlink_mode}')",
            "        sys.exit(1)",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ])

    def _render_test_script(self, spec: SITLTestSpec) -> str:
        """根据 SITLTestSpec 渲染一个独立可执行的 pymavlink 测试脚本。"""
        inject_body = textwrap.indent(self._render_inject(spec), "    ")
        verify_body = textwrap.indent(self._render_verify(spec), "    ")
        req_id = spec.req_id
        conn = self._connection_string
        out = self._output_dir

        lines = [
            "#!/usr/bin/env python3",
            f'"""',
            f"Auto-generated SITL test: {req_id}",
            f"Tier  : {spec.tier}",
            f"Notes : {spec.notes}",
            f"",
            f"Run:",
            f"    python {out}/test_{req_id.lower()}.py",
            f"(requires ArduPilot SITL running on {conn})",
            f'"""',
            "",
            "import sys, time",
            "from pymavlink import mavutil",
            "",
            f'CONNECTION = "{conn}"',
            "TIMEOUT    = 30.0",
            "",
            "",
            "def connect():",
            "    print(f'Connecting to {CONNECTION} ...')",
            "    mav = mavutil.mavlink_connection(CONNECTION)",
            "    mav.wait_heartbeat(timeout=10)",
            "    print(f'Connected — system {mav.target_system} component {mav.target_component}')",
            "    return mav",
            "",
            "",
            "def set_param(mav, name: str, value: float):",
            "    mav.mav.param_set_send(",
            "        mav.target_system, mav.target_component,",
            "        name.encode(), value,",
            "        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,",
            "    )",
            "    time.sleep(0.3)",
            "",
            "",
            "def get_mode(mav) -> str:",
            "    msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=5)",
            "    if msg is None:",
            "        return 'UNKNOWN'",
            "    return mavutil.mode_string_v10(msg)",
            "",
            "",
            "def wait_mode(mav, mode: str, timeout: float = 15.0) -> bool:",
            "    deadline = time.time() + timeout",
            "    while time.time() < deadline:",
            "        current = get_mode(mav)",
            "        if mode.upper() in current.upper():",
            "            return True",
            "        time.sleep(0.5)",
            "    return False",
            "",
            "",
            "def try_arm(mav) -> bool:",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,",
            "        0, 1, 0, 0, 0, 0, 0, 0,",
            "    )",
            "    ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)",
            "    if ack is None:",
            "        return False",
            "    return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED",
            "",
            "",
            "def wait_for_command(mav, cmd_id: int, timeout: float = 10.0) -> bool:",
            "    deadline = time.time() + timeout",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='COMMAND_LONG', blocking=True, timeout=1)",
            "        if msg and msg.command == cmd_id:",
            "            return True",
            "    return False",
            "",
            "",
            "def wait_for_statustext(mav, keyword: str, timeout: float = 10.0) -> bool:",
            "    deadline = time.time() + timeout",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='STATUSTEXT', blocking=True, timeout=1)",
            "        if msg and keyword.lower() in msg.text.lower():",
            "            return True",
            "    return False",
            "",
            "",
            "def set_mode(mav, mode_name: str) -> bool:",
            "    mode_id = mav.mode_mapping().get(mode_name.upper())",
            "    if mode_id is None:",
            "        return False",
            "    mav.mav.set_mode_send(mav.target_system,",
            "        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)",
            "    return wait_mode(mav, mode_name, timeout=5.0)",
            "",
            "",
            "def force_arm_and_takeoff(mav, altitude: float = 3.0) -> bool:",
            "    _MODES = {'STABILIZE':0,'GUIDED':4,'LOITER':5,'RTL':6,'LAND':9}",
            "    set_param(mav, 'SIM_GPS1_ENABLE', 1)",
            "    set_param(mav, 'SIM_BARO_DISABLE', 0)",
            "    time.sleep(0.5)",
            "    print('  等待 GPS 定位...')",
            "    gps_deadline = time.time() + 15",
            "    while time.time() < gps_deadline:",
            "        gps = mav.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)",
            "        if gps and gps.fix_type >= 3:",
            "            break",
            "        time.sleep(0.5)",
            "    # 切 GUIDED 模式（硬编码 ID 兜底）",
            "    mapping = mav.mode_mapping() or {}",
            "    guided_id = mapping.get('GUIDED') or _MODES['GUIDED']",
            "    mav.mav.set_mode_send(mav.target_system,",
            "        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, guided_id)",
            "    time.sleep(1)",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,",
            "        0, 1, 21196, 0, 0, 0, 0, 0,",
            "    )",
            "    ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)",
            "    if ack is None or ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:",
            "        return False",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,",
            "        0, 0, 0, 0, 0, 0, 0, altitude,",
            "    )",
            "    deadline = time.time() + 20",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)",
            "        if msg and msg.relative_alt >= int(altitude * 0.7 * 1000):",
            "            return True",
            "        time.sleep(0.5)",
            "    return False",
            "",
            "",
            "# ── Inject ──────────────────────────────────────────────────────",
            "def inject(mav):",
            inject_body,
            "",
            "# ── Verify ──────────────────────────────────────────────────────",
            "def verify(mav) -> bool:",
            verify_body,
            "",
            "# ── Main ────────────────────────────────────────────────────────",
            "def main():",
            "    mav = connect()",
            "    print('Injecting fault condition ...')",
            "    inject(mav)",
            "    print('Verifying expected behavior ...')",
            "    result = verify(mav)",
            "    if result:",
            f"        print('PASS  {req_id}')",
            "        sys.exit(0)",
            "    else:",
            f"        print('FAIL  {req_id}')",
            "        sys.exit(1)",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
        return "\n".join(lines)

    def _render_inject(self, spec: SITLTestSpec) -> str:
        from src.sitl.sitl_specs import render_inject
        return render_inject(spec.inject)

    def _render_verify(self, spec: SITLTestSpec) -> str:
        from src.sitl.sitl_specs import render_verify
        return render_verify(spec.verify)

    # ------------------------------------------------------------------
    # L2 — 直接执行测试（连接 SITL）
    # ------------------------------------------------------------------

    def run_l2(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        per_test_sitl: bool = True,
        sitl_home: str = "51.4,-2.35,0,0",
    ) -> List[TestResult]:
        """
        执行所有 L2 测试，返回结果列表。

        host / port
          当 None（默认）时使用构造时传入的 connection_string。
          显式指定时覆盖连接目标（优先级高于 connection_string）。

        per_test_sitl=True (默认)
          每个测试启动一个独立 SITL 进程，跑完即停。彻底隔离测试间状态。
        per_test_sitl=False
          假设 SITL 已在外部启动并运行，所有测试共享同一进程。
        """
        try:
            from pymavlink import mavutil
        except ImportError:
            return [TestResult("ALL", "L2", False, "pymavlink 未安装")]

        specs = [s for s in self._linker.generate_test_specs() if s.tier == "L2"]
        results: List[TestResult] = []
        if host is not None and port is not None:
            conn_str = f"tcp:{host}:{port}"
        else:
            conn_str = self._connection_string

        for spec in specs:
            print(f"\n[L2] 运行测试 {spec.req_id} ...")
            t0 = time.time()

            launched_here = False
            if per_test_sitl:
                # 清理可能残留的 EEPROM，确保 --wipe + --defaults 干净加载
                eeprom = Path.cwd() / "eeprom.bin"
                if eeprom.exists():
                    try:
                        eeprom.unlink()
                    except OSError:
                        pass
                launched_here = self.launch_sitl(home=sitl_home)
                if not launched_here:
                    results.append(TestResult(
                        spec.req_id, "L2", False, "独立 SITL 启动失败",
                        time.time() - t0,
                    ))
                    continue

            try:
                result = self._run_single_test(spec, conn_str, mavutil)
                dt = time.time() - t0
                results.append(TestResult(spec.req_id, "L2", result[0], result[1], dt))
            except Exception as e:
                dt = time.time() - t0
                results.append(TestResult(spec.req_id, "L2", False, f"异常: {e}", dt))
            finally:
                if per_test_sitl and launched_here:
                    self.stop_sitl()

        return results

    def _run_single_test(
        self,
        spec: SITLTestSpec,
        conn_str: str,
        mavutil,
    ):
        """执行单个 L2 测试，返回 (passed, message)。所有 inject/verify 调度
        都通过 sitl_specs 注册表完成，无 if/elif 解释器。"""
        from src.sitl.sitl_specs import TestContext, run_inject, run_verify

        mav = mavutil.mavlink_connection(conn_str)
        mav.wait_heartbeat(timeout=10)

        # 请求所有遥测数据流（否则 GLOBAL_POSITION_INT 等消息不会被发送）
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
        )
        time.sleep(0.5)

        ctx = TestContext(mav=mav, mavutil=mavutil)

        # 每次测试前恢复已知初始状态，避免跨测试的状态污染
        ctx.reset_drone_state()

        # ── Inject ──────────────────────────────────────────────────────
        try:
            run_inject(ctx, spec.inject)
        except Exception as e:
            return False, f"inject 异常: {e}"

        # ── Verify ──────────────────────────────────────────────────────
        try:
            return run_verify(ctx, spec.verify)
        except Exception as e:
            return False, f"verify 异常: {e}"

    # ------------------------------------------------------------------
    # 生成完整报告
    # ------------------------------------------------------------------

    def generate_full_report(
        self,
        run_l2: bool = False,
        auto_launch_sitl: bool = False,
        sitl_home: str = "51.4,-2.35,0,0",
    ) -> BridgeReport:
        """
        生成完整报告。

        run_l2=False  — 只做 L1 静态验证 + 生成 L2 脚本（不连 SITL）
        run_l2=True   — 额外执行 L2 测试（需要 SITL 运行中）
        auto_launch_sitl=True — run_l2=True 时自动启动/停止 arducopter
        """
        parm_path = self.generate_l1()
        l1_results = self.validate_l1()
        self.generate_l2_scripts()

        l2_results: List[TestResult] = []
        if run_l2:
            # 每个测试独立启停 SITL（per_test_sitl=True 是 run_l2 默认）
            # auto_launch_sitl 仅决定是否启用 L2，实际启停由 run_l2 内部完成
            if auto_launch_sitl:
                l2_results = self.run_l2(sitl_home=sitl_home, per_test_sitl=True)
            else:
                # 调用方自行管理 SITL 进程，所有测试共享
                l2_results = self.run_l2(per_test_sitl=False)

        return BridgeReport(
            model_name=self._model.name,
            parm_file=str(parm_path),
            l1_results=l1_results,
            l2_results=l2_results,
        )
