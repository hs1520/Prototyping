"""sitl_bridge.py"""

from __future__ import annotations

import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from src.sysml.lite_model import SysMLLiteModel
from src.sitl.requirement_linker import (
    RequirementEvidenceBundle,
    RequirementLinker,
    SITLTestSpec,
)
from src.utils.ardupilot import default_arducopter_binary

_DEFAULT_ARDUCOPTER = default_arducopter_binary()

# Gazebo (headless_gazebo 镜像) - fdm_backend="gazebo" 时使用
# home 坐标要匹配 worlds/iris_runway.sdf 里的 <spherical_coordinates>（CMAC），
# 否则 NavSat 插件算出的 GPS 位置和 ArduPilot 的 home 假设不一致。
_GAZEBO_IMAGE = "headless_gazebo"
_GAZEBO_CONTAINER = "ai_prototyping_gazebo"
_GAZEBO_UDP_PORT = 9002
_GAZEBO_HOME = "-35.363262,149.165237,584,0"
_NATIVE_HOME = "51.4,-2.35,0,0"


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
    trace_results: List[TestResult] = field(default_factory=list)

    def passed(self) -> bool:
        return all(r.passed for r in self.l1_results + self.l2_results + self.trace_results)

    def safety_status(self) -> str:
        """Summarise executable L2 safety checks without hiding traceability blocks."""
        blocked = any(not r.passed for r in self.trace_results)
        if blocked and self.l2_results:
            return "PARTIAL"
        if blocked:
            return "BLOCKED"
        if not self.l2_results:
            return "NOT_RUN"
        if all(r.passed for r in self.l2_results):
            return "PASS"
        return "FAIL"

    def summary(self) -> str:
        executable_results = self.l1_results + self.l2_results
        all_results = executable_results + self.trace_results
        ok = sum(1 for r in executable_results if r.passed)
        trace_blocked = sum(1 for r in self.trace_results if not r.passed)
        l2_ok = sum(1 for r in self.l2_results if r.passed)
        lines = [
            f"SITL Bridge Report — {self.model_name}",
            f"  L1/L2 passed: {ok}/{len(executable_results)}",
            f"  L2 safety status: {self.safety_status()} ({l2_ok}/{len(self.l2_results)} executable passed)",
            f"  Traceability blocked: {trace_blocked}",
            "",
        ]
        for r in all_results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} [{r.tier}] {r.req_id:<20} {r.message}")
        return "\n".join(lines)


ARDUPILOT_COPTER_PROFILE: dict = {
    "platform": "ArduPilot Copter",
    "mode_vocabulary": [
        "CMD_STABILIZE", "CMD_GUIDED", "CMD_AUTO",
        "CMD_RTL",       "CMD_LAND",   "CMD_TAKEOFF",
        "CMD_LOITER",    "CMD_POSHOLD",
    ],
    "emergency_mode": "CMD_LAND",
    "requires_airborne": {"CMD_RTL", "CMD_LAND", "CMD_AUTO",
                          "CMD_LOITER", "CMD_POSHOLD"},
    # 基础 SITL 参数：--wipe 后需要显式设置，否则 motor matrix 无效
    "base_sitl_params": {
        "FRAME_CLASS":    1,
        "FRAME_TYPE":     1,
        "ARMING_CHECK":   0,
        "DISARM_DELAY":   0,
        # EK3_CHECK_SCALE 保持默认 100。设为 0 本意是放宽 EKF 健康检查，实则把
        # GPS 精度阈值收成 0 -> GPS 永远过不了检查 -> EKF 拒绝融合 GPS -> 无位置
        # 估计/home -> "Arm: Need Position Estimate" -> 无法解锁，需飞行的 L2 全部
        # 假绿。
        "EK3_CHECK_SCALE": 100,
    },
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
        fdm_backend: str = "native",
        speedup: float = 1.0,
    ) -> None:
        """llm             : 可选 LLMInterface，启用语义标签分类。
        platform_profile: 可选 Platform Profile dict。传入后
                          generate_l2_scripts() 额外生成 accept 模式切换
                          测试（CMD_RTL -> SET_MODE RTL 等）；None 时跳过
                          accept 测试（S11 选项C）。
        fdm_backend     : "native"（默认）用 ArduPilot 内置简化物理模型
                          （--model +）；"gazebo" 切到外部 FDM（--model
                          JSON），并在 launch_sitl() 时自动拉起
                          headless_gazebo 容器（夹爪/降落伞/真实姿态动力学
                          一类的需求验证才需要）。
        speedup         : SITL 仿真时钟相对墙钟的倍率（--speedup N）。仅
                          native FDM 支持；gazebo 与外部物理引擎锁步，单方面
                          加速无效或破坏时序，强制钉回 1。L2 的测量与判定
                          窗口读仿真时钟，判定严格度不随倍率漂移；渲染出的
                          独立 test_*.py 脚本仍按 1x 编写。
        """
        self._model = model
        self._output_dir = Path(output_dir)
        self._connection_string = connection_string
        self._arducopter_bin = arducopter_bin
        self._platform_profile = platform_profile or {}
        linker = RequirementLinker(
            model,
            llm=llm,
            verbose=verbose,
        )
        self._requirement_evidence = linker.compile_evidence()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sitl_proc: Optional[subprocess.Popen] = None
        if fdm_backend not in ("native", "gazebo"):
            raise ValueError(f"未知 fdm_backend: {fdm_backend!r}（应为 'native' 或 'gazebo'）")
        self._fdm_backend = fdm_backend
        self._gazebo_started_by_us = False
        speedup = float(speedup)
        if speedup < 1.0:
            raise ValueError(f"speedup 必须 ≥ 1.0，收到 {speedup}")
        if fdm_backend == "gazebo" and speedup != 1.0:
            print(
                f"  ⚠ fdm_backend='gazebo' 与外部物理引擎锁步，"
                f"speedup={speedup} 被钉回 1.0（加速 Gazebo 需要 world "
                "RTF 与 --speedup 匹配调整，见 speedup 采纳计划 B 段）"
            )
            speedup = 1.0
        self._speedup = speedup

    @property
    def requirement_evidence(self) -> RequirementEvidenceBundle:
        return self._requirement_evidence

    def _gazebo_container_running(self) -> bool:
        try:
            out = subprocess.run(
                ["docker", "ps", "--filter", f"name=^{_GAZEBO_CONTAINER}$",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            return _GAZEBO_CONTAINER in out.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _ensure_gazebo_running(self, wait_s: float = 6.0) -> bool:
        if self._gazebo_container_running():
            print(f"  ✓ Gazebo 容器 [{_GAZEBO_CONTAINER}] 已在运行，复用")
            return True

        print(f"  ▶ 拉起 Gazebo 容器 [{_GAZEBO_CONTAINER}] ...")
        try:
            subprocess.run(
                ["docker", "run", "--rm", "-d",
                 "--name", _GAZEBO_CONTAINER,
                 "-p", f"{_GAZEBO_UDP_PORT}:{_GAZEBO_UDP_PORT}/udp",
                 _GAZEBO_IMAGE],
                check=True, capture_output=True, text=True, timeout=30,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"  ✗ 启动 Gazebo 容器失败: {e}")
            return False

        self._gazebo_started_by_us = True
        time.sleep(wait_s)
        if not self._gazebo_container_running():
            print("  ✗ Gazebo 容器启动后未能保持运行")
            return False
        print("  ✓ Gazebo 容器已就绪")
        return True

    def _stop_gazebo(self) -> None:
        if not self._gazebo_started_by_us:
            return
        try:
            subprocess.run(
                ["docker", "stop", _GAZEBO_CONTAINER],
                capture_output=True, text=True, timeout=15,
            )
            print(f"  ■ Gazebo 容器 [{_GAZEBO_CONTAINER}] 已停止")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        self._gazebo_started_by_us = False

    def launch_sitl(
        self,
        home: Optional[str] = None,
        wait_s: float = 8.0,
    ) -> bool:
        """在后台启动 arducopter SITL。

        fdm_backend="native" 用 ArduPilot 内置简化物理模型（--model +）；
        "gazebo" 自动拉起 headless_gazebo 容器并切到外部 FDM（--model JSON），
        home 默认对齐 Gazebo world 的 spherical_coordinates（CMAC），避免 GPS
        原点和物理引擎不一致。返回 True 表示进程已启动并监听 5760 端口。
        """
        if not Path(self._arducopter_bin).exists():
            print(f"  ✗ arducopter 二进制不存在: {self._arducopter_bin}")
            return False

        use_gazebo = self._fdm_backend == "gazebo"
        if use_gazebo and not self._ensure_gazebo_running():
            return False

        if home is None:
            home = _GAZEBO_HOME if use_gazebo else _NATIVE_HOME

        parm_path = self._output_dir / f"{self._model.name}.parm"
        if not parm_path.exists():
            self.generate_l1()

        cmd = [
            self._arducopter_bin,
            "--model", "JSON" if use_gazebo else "+",
            "--home", home,
            "--wipe",                    # 清空 EEPROM，确保 --defaults 生效
            "--defaults", str(parm_path),
        ]
        if self._speedup != 1.0:
            # 构造器已保证 gazebo 后端到不了这里（speedup 被钉回 1）。
            cmd += ["--speedup", str(self._speedup)]
        self._sitl_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  ▶ ArduCopter SITL 启动中 (PID {self._sitl_proc.pid}) ...")
        time.sleep(wait_s)

        import socket
        try:
            s = socket.create_connection(("127.0.0.1", 5760), timeout=3)
            s.close()
        except OSError:
            print("  ✗ SITL 启动超时，端口 5760 无响应")
            self.stop_sitl()
            return False

        # 等飞机可解锁/可起飞再返回，否则顺序套件里后面的测试会偶发起飞失败。
        # 判据是 EKF_STATUS_REPORT 的 EKF_POS_HORIZ_ABS 位（EKF 已有绝对水平位置，
        # 即 home 已设、armable）；STATUSTEXT "origin set" 只播一次，连接晚了抓不到。
        # Gazebo JSON FDM 下没有 Gazebo 喂数据该位不会置位，因此也确认了 FDM 握手。
        ready = False
        try:
            from pymavlink import mavutil as _mu
            _mav = _mu.mavlink_connection("tcp:127.0.0.1:5760")
            _mav.wait_heartbeat(timeout=15)
            _mav.mav.request_data_stream_send(
                _mav.target_system, _mav.target_component,
                _mu.mavlink.MAV_DATA_STREAM_ALL, 5, 1,
            )
            _ABS = getattr(_mu.mavlink, "EKF_POS_HORIZ_ABS", 16)
            ekf_deadline = time.time() + 90
            while time.time() < ekf_deadline:
                msg = _mav.recv_match(
                    type="EKF_STATUS_REPORT", blocking=True, timeout=2)
                if msg is not None and (msg.flags & _ABS):
                    ready = True
                    break
            _mav.close()
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.bridge.ekf_probe", exc)
            pass

        if not ready:
            print("  ✗ SITL 启动后 EKF 未就绪（无绝对位置估计），放弃本次启动")
            self.stop_sitl()
            return False

        print("  ✓ SITL 已就绪（EKF 绝对位置已锁定），监听 tcp:127.0.0.1:5760")
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

        import socket
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                s = socket.create_connection(("127.0.0.1", 5760), timeout=0.5)
                s.close()
                time.sleep(0.5)
            except OSError:
                break

        self._stop_gazebo()

    def generate_l1(self) -> Path:
        """生成 .parm 文件，返回文件路径。"""
        parm_content = self._requirement_evidence.parm_file
        parm_path = self._output_dir / f"{self._model.name}.parm"

        base_params = self._platform_profile.get("base_sitl_params", {})
        if base_params:
            from .parameter_projection import merge_base_parameters

            parm_content = merge_base_parameters(parm_content, base_params)

        parm_path.write_text(parm_content, encoding="utf-8")
        print(f"[L1] .parm 文件已写入: {parm_path}")
        return parm_path

    def validate_l1(self) -> List[TestResult]:
        """静态验证：检查所有 L1 参数是否已正确解析（无 <unresolved>）。"""
        results: List[TestResult] = []
        specs = [s for s in self._requirement_evidence.test_specs if s.tier == "L1"]

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

    def validate_traceability(self) -> List[TestResult]:
        """Report requirement-text/tag mismatches that block trustworthy SITL tests."""
        return [
            TestResult(
                req_id=str(m.get("req_id", "?")),
                tier="TRACE",
                passed=False,
                message=str(m.get("message", "traceability mismatch")),
            )
            for m in self._requirement_evidence.traceability_mismatches
        ]

    def generate_l2_scripts(self) -> List[Path]:
        """为每个 L2 测试规格生成独立的 pymavlink 测试脚本。"""
        paths: List[Path] = []

        specs = [s for s in self._requirement_evidence.test_specs if s.tier == "L2"]
        for spec in specs:
            code = self._render_test_script(spec)
            script_path = self._output_dir / f"test_{spec.req_id.lower()}.py"
            script_path.write_text(code, encoding="utf-8")
            paths.append(script_path)
            print(f"[L2] 脚本已生成: {script_path}")

        if self._platform_profile:
            accept_paths = self._generate_accept_mode_scripts()
            paths.extend(accept_paths)

        return paths

    def _cmd_to_mavlink_mode(self, cmd_name: str) -> Optional[str]:
        vocab = self._platform_profile.get("mode_vocabulary", [])
        if cmd_name not in vocab:
            return None
        return cmd_name[4:] if cmd_name.startswith("CMD_") else None

    def _generate_accept_mode_scripts(self) -> List[Path]:
        try:
            from src.simulation.state_extractor import extract_state_machines
            from src.simulation.behavioral_sim import _classify_accept_transitions
        except ImportError:
            return []

        sysml_text = self._model.to_sysml_text() or ""
        state_machines = extract_state_machines(sysml_text)
        requires_airborne = self._platform_profile.get("requires_airborne", set())

        paths: List[Path] = []
        seen_modes: set = set()

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
        from .script_runtime import render_pymavlink_runtime

        conn = self._connection_string
        preflight = ""
        if requires_airborne:
            preflight = textwrap.indent(
                "print('  ARM + TAKEOFF to 5m ...')\n"
                "if not force_arm_and_takeoff(mav, altitude=5.0):\n"
                "    print('FAIL  takeoff failed')\n"
                "    sys.exit(1)\n",
                "    ",
            )
        header = (
            "#!/usr/bin/env python3\n"
            '"""\n'
            f"Auto-generated SITL accept-mode test: {cmd_name} → SET_MODE {mavlink_mode}\n\n"
            "Verifies that the ArduPilot flight controller accepts the MAVLink\n"
            f"mode switch corresponding to SysML accept trigger {cmd_name!r}.\n\n"
            "Run:\n"
            f"    python {self._output_dir}/test_mode_{mavlink_mode.lower()}.py\n"
            f"(requires ArduPilot SITL running on {conn})\n"
            '"""\n\n'
        )
        main = (
            "\ndef main():\n"
            "    mav = connect()\n"
            f"    print('Testing SysML accept trigger: {cmd_name} → SET_MODE {mavlink_mode}')\n"
            f"{preflight}"
            f"    print('  Sending SET_MODE {mavlink_mode} ...')\n"
            f"    ok = set_mode(mav, '{mavlink_mode}')\n"
            "    if ok:\n"
            f"        print('PASS  {cmd_name} → {mavlink_mode}')\n"
            "        sys.exit(0)\n"
            "    else:\n"
            f"        print('FAIL  {cmd_name}: mode switch to {mavlink_mode} failed')\n"
            "        sys.exit(1)\n\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
        return header + render_pymavlink_runtime(conn) + main

    def _render_test_script(self, spec: SITLTestSpec) -> str:
        from .script_runtime import render_pymavlink_runtime

        inject_body = textwrap.indent(self._render_inject(spec), "    ")
        verify_body = textwrap.indent(self._render_verify(spec), "    ")
        req_id = spec.req_id
        conn = self._connection_string
        header = (
            "#!/usr/bin/env python3\n"
            '"""\n'
            f"Auto-generated SITL test: {req_id}\n"
            f"Tier  : {spec.tier}\n"
            f"Notes : {spec.notes}\n\n"
            "Run:\n"
            f"    python {self._output_dir}/test_{req_id.lower()}.py\n"
            f"(requires ArduPilot SITL running on {conn})\n"
            '"""\n\n'
        )
        test_body = (
            "\n# ── Inject ──────────────────────────────────────────────────────\n"
            "def inject(mav):\n"
            f"{inject_body}\n"
            "# ── Verify ──────────────────────────────────────────────────────\n"
            "def verify(mav) -> bool:\n"
            f"{verify_body}\n"
            "# ── Main ────────────────────────────────────────────────────────\n"
            "def main():\n"
            "    mav = connect()\n"
            "    print('Injecting fault condition ...')\n"
            "    inject(mav)\n"
            "    print('Verifying expected behavior ...')\n"
            "    result = verify(mav)\n"
            "    if result:\n"
            f"        print('PASS  {req_id}')\n"
            "        sys.exit(0)\n"
            "    else:\n"
            f"        print('FAIL  {req_id}')\n"
            "        sys.exit(1)\n\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
        return header + render_pymavlink_runtime(conn) + test_body
    def _render_inject(self, spec: SITLTestSpec) -> str:
        from src.sitl.sitl_specs import render_inject
        return render_inject(spec.inject)

    def _render_verify(self, spec: SITLTestSpec) -> str:
        from src.sitl.sitl_specs import render_verify
        return render_verify(spec.verify)

    def run_l2(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        per_test_sitl: bool = True,
        sitl_home: Optional[str] = None,
    ) -> List[TestResult]:
        """执行所有 L2 测试，返回结果列表。"""
        try:
            from pymavlink import mavutil
        except ImportError:
            return [TestResult("ALL", "L2", False, "pymavlink 未安装")]

        specs = [s for s in self._requirement_evidence.test_specs if s.tier == "L2"]
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
                for attempt in range(2):
                    launched_here = self.launch_sitl(home=sitl_home)
                    if launched_here:
                        break
                    if attempt == 0:
                        print("  ↻ SITL 启动/EKF 就绪失败，重试一次 ...")
                        time.sleep(2.0)
                if not launched_here:
                    results.append(TestResult(
                        spec.req_id, "L2", False, "独立 SITL 启动失败",
                        time.time() - t0,
                    ))
                    continue

            try:
                # 新拉起的 SITL（--wipe 全新启动）机体已是干净初始态，无需
                # reset_drone_state；Gazebo 下 reset 里的 SIM_*/EKF 参数还会扰乱 FDM 喂入的
                # 传感器估计，导致起飞失败。只有共享 SITL（per_test_sitl=False，状态跨测试
                # 残留）才需要 reset。
                result = self._run_single_test(
                    spec, conn_str, mavutil, fresh_sitl=launched_here)
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
        fresh_sitl: bool = False,
    ):
        """执行单个 L2 测试，返回 (passed, message)。inject/verify 调度全走
        sitl_specs 注册表，无 if/elif 解释器。

        fresh_sitl=True 表示本测试独立拉起了 SITL（--wipe），机体已是干净初始态，
        跳过 reset_drone_state（Gazebo 下 reset 会扰乱 FDM 传感器估计，导致起飞
        失败）。
        """
        from src.sitl.sitl_specs import TestContext, run_inject, run_verify

        mav = mavutil.mavlink_connection(conn_str)
        mav.wait_heartbeat(timeout=10)

        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
        )
        time.sleep(0.5)

        ctx = TestContext(mav=mav, mavutil=mavutil, speedup=self._speedup)

        # 仅共享 SITL 时才需清理跨测试状态污染；新拉起的 SITL 已是干净态
        if not fresh_sitl:
            ctx.reset_drone_state()

        # ── Apply this test's resolved ArduPilot params to the running SITL ──
        # These carry the actuator wiring the verify depends on (parachute
        # SERVO8_FUNCTION=27 / CHUTE_SERVO_ON=2000, gripper SERVO7_FUNCTION=28 /
        # GRIP_RELEASE=2000). They live in the catalogue but not in the boot .parm,
        # so without them the release drives an unassigned servo channel and
        # SERVO_OUTPUT_RAW-based verifies read 0. Applied before inject (hence before
        # takeoff) so the servo latches at the release PWM when the fault fires;
        # non-numeric (unresolved) values are skipped.
        for rp in getattr(spec, "params", None) or []:
            val = getattr(rp, "value", None)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                try:
                    ctx.set_param(rp.param_name, float(val))
                except Exception as exc:
                    from src.utils.suppressed import record_suppressed
                    record_suppressed("sitl.bridge.apply_ardu_param", exc)
                    pass
        time.sleep(0.5)  # let SERVOx_FUNCTION re-evaluate before the fault

        try:
            run_inject(ctx, spec.inject)
        except Exception as e:
            return False, f"inject 异常: {e}"

        try:
            return run_verify(ctx, spec.verify)
        except Exception as e:
            return False, f"verify 异常: {e}"

    def generate_full_report(
        self,
        run_l2: bool = False,
        auto_launch_sitl: bool = False,
        sitl_home: Optional[str] = None,
    ) -> BridgeReport:
        """生成完整报告。"""
        parm_path = self.generate_l1()
        l1_results = self.validate_l1()
        trace_results = self.validate_traceability()
        self.generate_l2_scripts()

        l2_results: List[TestResult] = []
        if run_l2:
            if auto_launch_sitl:
                l2_results = self.run_l2(sitl_home=sitl_home, per_test_sitl=True)
            else:
                l2_results = self.run_l2(per_test_sitl=False)

        return BridgeReport(
            model_name=self._model.name,
            parm_file=str(parm_path),
            l1_results=l1_results,
            l2_results=l2_results,
            trace_results=trace_results,
        )
