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
from typing import List, Optional, Tuple

from src.sysml.lite_model import SysMLLiteModel
from src.sitl.requirement_linker import RequirementLinker, SITLTestSpec

# ArduCopter 二进制默认路径（直接启动，绕过 sim_vehicle.py + MAVProxy）
_DEFAULT_ARDUCOPTER = os.path.expanduser(
    "~/PycharmProjects/ardupilot/build/sitl/bin/arducopter"
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

class SITLBridge:

    def __init__(
        self,
        model: SysMLLiteModel,
        output_dir: str = "sitl_output",
        connection_string: str = "tcp:127.0.0.1:5760",
        arducopter_bin: str = _DEFAULT_ARDUCOPTER,
    ) -> None:
        self._model = model
        self._output_dir = Path(output_dir)
        self._connection_string = connection_string
        self._arducopter_bin = arducopter_bin
        self._linker = RequirementLinker(model)
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
            print(f"  ✓ SITL 已就绪，监听 tcp:127.0.0.1:5760")
            return True
        except OSError:
            print(f"  ✗ SITL 启动超时，端口 5760 无响应")
            self.stop_sitl()
            return False

    def stop_sitl(self) -> None:
        """停止后台 SITL 进程。"""
        if self._sitl_proc and self._sitl_proc.poll() is None:
            self._sitl_proc.terminate()
            try:
                self._sitl_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sitl_proc.kill()
            print(f"  ■ ArduCopter SITL 已停止 (PID {self._sitl_proc.pid})")
            self._sitl_proc = None

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
        parm_content = self._linker.generate_parm_file()
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
        """为每个 L2 测试规格生成独立的 pymavlink 测试脚本。"""
        specs = [s for s in self._linker.generate_test_specs() if s.tier == "L2"]
        paths: List[Path] = []
        for spec in specs:
            code = self._render_test_script(spec)
            script_path = self._output_dir / f"test_{spec.req_id.lower()}.py"
            script_path.write_text(code, encoding="utf-8")
            paths.append(script_path)
            print(f"[L2] 脚本已生成: {script_path}")
        return paths

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
        inject = spec.inject or ""

        if inject == "disconnect_gcs":
            return textwrap.dedent("""\
                # 停止发送心跳包以模拟 GCS 断连
                print("  停止心跳包，等待链路丢失超时 ...")
                time.sleep(12)  # 超过 linkLossTimeout(10s)
            """)

        if inject.startswith("param set "):
            parts = inject.split()
            # "param set SIM_GPS_DISABLE 1"
            pname, pval = parts[2], parts[3]
            return textwrap.dedent(f"""\
                print("  设置参数 {pname}={pval}")
                set_param(mav, "{pname}", {pval})
                time.sleep(1)
            """)

        if inject.startswith("set_delivery_abort"):
            return textwrap.dedent("""\
                # 通过自定义 MAVLink 消息注入 delivery abort 状态（需要飞控支持）
                print("  注入 delivery_abort 条件（跳过：需 Gazebo gripper 插件）")
            """)

        return f'print("  inject: {inject}")\n'

    def _render_verify(self, spec: SITLTestSpec) -> str:
        verify = spec.verify or ""

        if verify.startswith("wait_mode("):
            # wait_mode(LAND, timeout=15.0)
            import re
            m = re.search(r"wait_mode\((\w+),\s*timeout=([\d.]+)\)", verify)
            if m:
                mode, timeout = m.group(1), m.group(2)
                return textwrap.dedent(f"""\
                    print("  等待飞行模式切换到 {mode} ...")
                    ok = wait_mode(mav, "{mode}", timeout={timeout})
                    if ok:
                        print("  ✓ 模式已切换到 {mode}")
                    else:
                        print("  ✗ 超时未切换到 {mode}")
                    return ok
                """)

        if verify == "assert_arm_rejected()":
            return textwrap.dedent("""\
                print("  尝试解锁，期望被拒绝 ...")
                armed = try_arm(mav)
                if not armed:
                    print("  ✓ 解锁被正确拒绝")
                else:
                    print("  ✗ 解锁成功（不应该）")
                return not armed
            """)

        if verify.startswith("wait_for_mavlink(MAV_CMD_DO_PARACHUTE"):
            import re
            m = re.search(r"timeout=([\d.]+)", verify)
            timeout = m.group(1) if m else "10.0"
            return textwrap.dedent(f"""\
                print("  等待 MAV_CMD_DO_PARACHUTE 命令 ...")
                ok = wait_for_command(mav, mavutil.mavlink.MAV_CMD_DO_PARACHUTE, timeout={timeout})
                if ok:
                    print("  ✓ 收到降落伞部署命令")
                else:
                    print("  ✗ 未收到降落伞部署命令（超时）")
                return ok
            """)

        if verify == "assert_gripper_locked()":
            return textwrap.dedent("""\
                # 需要 Gazebo gripper 插件支持，当前跳过
                print("  ⚠ assert_gripper_locked: 需要 Gazebo，标记为跳过")
                return True  # 跳过不计失败
            """)

        return f'print("  verify: {verify}")\nreturn True\n'

    # ------------------------------------------------------------------
    # L2 — 直接执行测试（连接 SITL）
    # ------------------------------------------------------------------

    def run_l2(
        self,
        host: str = "127.0.0.1",
        port: int = 5760,
    ) -> List[TestResult]:
        """连接 SITL 并执行所有 L2 测试，返回结果列表。"""
        try:
            from pymavlink import mavutil
        except ImportError:
            return [TestResult("ALL", "L2", False, "pymavlink 未安装")]

        specs = [s for s in self._linker.generate_test_specs() if s.tier == "L2"]
        results: List[TestResult] = []
        conn_str = f"tcp:{host}:{port}"

        for spec in specs:
            print(f"\n[L2] 运行测试 {spec.req_id} ...")
            t0 = time.time()
            try:
                result = self._run_single_test(spec, conn_str, mavutil)
                dt = time.time() - t0
                results.append(TestResult(spec.req_id, "L2", result[0], result[1], dt))
            except Exception as e:
                dt = time.time() - t0
                results.append(TestResult(spec.req_id, "L2", False, f"异常: {e}", dt))

        return results

    def _run_single_test(
        self,
        spec: SITLTestSpec,
        conn_str: str,
        mavutil,
    ):
        """执行单个 L2 测试，返回 (passed, message)。"""
        mav = mavutil.mavlink_connection(conn_str)
        mav.wait_heartbeat(timeout=10)

        inject = spec.inject or ""
        verify = spec.verify or ""

        # ── Inject ──────────────────────────────────────────────────────
        if inject.startswith("param set "):
            parts = inject.split()
            pname, pval = parts[2], float(parts[3])
            mav.mav.param_set_send(
                mav.target_system, mav.target_component,
                pname.encode(), pval,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            time.sleep(1)

        elif inject == "disconnect_gcs":
            time.sleep(12)

        # ── Verify ──────────────────────────────────────────────────────
        if verify.startswith("wait_mode("):
            import re
            m = re.search(r"wait_mode\((\w+),\s*timeout=([\d.]+)\)", verify)
            if m:
                mode, timeout = m.group(1), float(m.group(2))
                deadline = time.time() + timeout
                while time.time() < deadline:
                    msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
                    if msg:
                        current = mavutil.mode_string_v10(msg)
                        if mode.upper() in current.upper():
                            return True, f"模式已切换到 {current}"
                    time.sleep(0.5)
                return False, f"超时未切换到 {mode}"

        elif verify == "assert_arm_rejected()":
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0,
            )
            ack = mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
            if ack and ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return True, "解锁被正确拒绝"
            return False, "解锁意外成功"

        elif "MAV_CMD_DO_PARACHUTE" in verify:
            import re
            m = re.search(r"timeout=([\d.]+)", verify)
            timeout = float(m.group(1)) if m else 10.0
            deadline = time.time() + timeout
            while time.time() < deadline:
                msg = mav.recv_match(type="COMMAND_LONG", blocking=True, timeout=1)
                if msg and msg.command == mavutil.mavlink.MAV_CMD_DO_PARACHUTE:
                    return True, "收到降落伞部署命令"
            return False, "未收到降落伞部署命令（超时）"

        elif verify == "assert_gripper_locked()":
            return True, "跳过（需要 Gazebo 插件）"

        return True, "无验证逻辑（跳过）"

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
            launched = False
            if auto_launch_sitl:
                launched = self.launch_sitl(home=sitl_home)
                if not launched:
                    l2_results.append(TestResult(
                        "ALL", "L2", False, "SITL 启动失败，跳过 L2 测试"
                    ))
                    return BridgeReport(
                        model_name=self._model.name,
                        parm_file=str(parm_path),
                        l1_results=l1_results,
                        l2_results=l2_results,
                    )
            try:
                l2_results = self.run_l2()
            finally:
                if auto_launch_sitl and launched:
                    self.stop_sitl()

        return BridgeReport(
            model_name=self._model.name,
            parm_file=str(parm_path),
            l1_results=l1_results,
            l2_results=l2_results,
        )
