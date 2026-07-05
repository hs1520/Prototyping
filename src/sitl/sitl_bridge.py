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

# Gazebo (headless_gazebo 镜像) — fdm_backend="gazebo" 时使用
# home 坐标必须匹配 worlds/iris_runway.sdf 里的 <spherical_coordinates>（CMAC），
# 否则 NavSat 插件算出来的 GPS 位置和 ArduPilot 的 home 假设不一致。
_GAZEBO_IMAGE = "headless_gazebo"
_GAZEBO_CONTAINER = "ai_prototyping_gazebo"
_GAZEBO_UDP_PORT = 9002
_GAZEBO_HOME = "-35.363262,149.165237,584,0"
_NATIVE_HOME = "51.4,-2.35,0,0"


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
    # 基础 SITL 参数：--wipe 后需要显式设置，否则 motor matrix 无效
    "base_sitl_params": {
        "FRAME_CLASS":    1,  # QUAD
        "FRAME_TYPE":     1,  # X-frame
        "ARMING_CHECK":   0,  # SITL 测试：跳过全部 PreArm 检查
        "DISARM_DELAY":   0,  # 禁止自动解除解锁（防止 EKF 健康检查触发自动缴械）
        # 注意：EK3_CHECK_SCALE 必须保持默认 100。曾误设为 0，本意是"放宽
        # EKF 健康检查"，实则把 GPS 精度阈值收成 0 → GPS 永远过不了检查 →
        # EKF 拒绝融合 GPS → 无位置估计/home → "Arm: Need Position Estimate"
        # → 无法解锁 → 起飞链路彻底失效（所有需飞行的 L2 全部假绿）。
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
    ) -> None:
        """
        llm             : 可选 LLMInterface，启用语义标签分类。
        platform_profile: 可选 Platform Profile dict。
                          传入后 generate_l2_scripts() 额外生成 accept
                          模式切换测试（CMD_RTL → SET_MODE RTL 等）。
                          为 None 时跳过 accept 测试（S11 选项C）。
        fdm_backend     : "native"（默认）使用 ArduPilot 内置简化物理模型
                          （--model +）；"gazebo" 切到外部 FDM
                          （--model JSON），并在 launch_sitl() 时自动拉起
                          headless_gazebo 容器（需要的需求验证，如夹爪/
                          降落伞/真实姿态动力学，才需要这个）。
        """
        self._model = model
        self._output_dir = Path(output_dir)
        self._connection_string = connection_string
        self._arducopter_bin = arducopter_bin
        self._platform_profile = platform_profile or {}
        self._linker = RequirementLinker(model, llm=llm, verbose=verbose)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sitl_proc: Optional[subprocess.Popen] = None
        if fdm_backend not in ("native", "gazebo"):
            raise ValueError(f"未知 fdm_backend: {fdm_backend!r}（应为 'native' 或 'gazebo'）")
        self._fdm_backend = fdm_backend
        self._gazebo_started_by_us = False

    # ------------------------------------------------------------------
    # SITL 进程管理
    # ------------------------------------------------------------------

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
        """
        若 headless_gazebo 容器未运行，自动 docker run 拉起一个。
        已经在跑的容器（不管是不是我们启动的）直接复用，不重复启动。
        """
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
            print(f"  ✗ Gazebo 容器启动后未能保持运行")
            return False
        print(f"  ✓ Gazebo 容器已就绪")
        return True

    def _stop_gazebo(self) -> None:
        """仅停止本实例自己拉起的 Gazebo 容器，不影响用户手动起的容器。"""
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
        """
        在后台启动 arducopter SITL。
        fdm_backend="native" 时用 ArduPilot 内置简化物理模型（--model +）；
        fdm_backend="gazebo" 时自动拉起 headless_gazebo 容器，
        并切到外部 FDM（--model JSON），home 默认对齐 Gazebo world 的
        spherical_coordinates（CMAC），避免 GPS 原点和物理引擎不一致。
        返回 True 表示进程已启动并监听 5760 端口。
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

        # 等待飞机真正"可解锁/可起飞"再返回，否则顺序套件里后面的测试会偶发
        # 起飞失败。判据用 EKF_STATUS_REPORT 的 EKF_POS_HORIZ_ABS 标志位——它
        # 直接表示"EKF 已有绝对水平位置"，即 home 已设、armable。比数 STATUSTEXT
        # ("origin set" 只播一次，连接晚了就永远抓不到）确定得多。在 Gazebo JSON
        # FDM 下，没有 Gazebo 喂数据该位永远不会置位，所以它也顺带确认了 FDM 握手。
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
            print(f"  ✗ SITL 启动后 EKF 未就绪（无绝对位置估计），放弃本次启动")
            self.stop_sitl()
            return False

        print(f"  ✓ SITL 已就绪（EKF 绝对位置已锁定），监听 tcp:127.0.0.1:5760")
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

        self._stop_gazebo()

    # ------------------------------------------------------------------
    # L1 — 参数文件生成
    # ------------------------------------------------------------------

    def generate_l1(self) -> Path:
        """生成 .parm 文件，返回文件路径。"""
        parm_content = self._linker.generate_parm_file()
        parm_path = self._output_dir / f"{self._model.name}.parm"

        # Append platform base_sitl_params (e.g. FRAME_CLASS, ARMING_CHECK)
        # that must be present after --wipe. 用"已定义的参数名集合"做精确判重，
        # 不能用子串匹配（`k not in parm_content` 会被注释或更长的同名子串误伤，
        # 例如把 base 参数静默丢弃 → EK3_CHECK_SCALE 这类关键项可能漏写）。
        base_params = self._platform_profile.get("base_sitl_params", {})
        if base_params:
            existing_names = set()
            for line in parm_content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                existing_names.add(stripped.split()[0])
            additions = [
                f"{k:<30} {v}  # base SITL param"
                for k, v in base_params.items()
                if k not in existing_names
            ]
            if additions:
                parm_content += "\n" + "\n".join(additions) + "\n"

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
            "    # wait_ready_to_arm：EKF 启动后需数秒设定 origin/home，重试直到就绪",
            "    armed = False",
            "    arm_deadline = time.time() + 45",
            "    while time.time() < arm_deadline:",
            "        mav.mav.command_long_send(",
            "            mav.target_system, mav.target_component,",
            "            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,",
            "            0, 1, 21196, 0, 0, 0, 0, 0,",
            "        )",
            "        ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)",
            "        if (ack",
            "                and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM",
            "                and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):",
            "            armed = True",
            "            break",
            "        time.sleep(2)",
            "    if not armed:",
            "        return False",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,",
            "        0, 0, 0, 0, 0, 0, 0, altitude,",
            "    )",
            "    # relative_alt 是相对 home 高度；home 未设的瞬间可能等于绝对海拔",
            "    # （~584000mm），会让 '>= target' 在地面误判。按基线相对爬升判断，",
            "    # 加合理性上限 + 连续 2 帧确认。",
            "    target_mm = int(altitude * 0.6 * 1000)",
            "    plausible_max_mm = int(altitude * 3 * 1000) + 5000",
            "    baseline_mm = None",
            "    hits = 0",
            "    deadline = time.time() + 30",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='GLOBAL_POSITION_INT',",
            "                             blocking=True, timeout=1)",
            "        if msg is not None:",
            "            if baseline_mm is None or msg.relative_alt < baseline_mm:",
            "                baseline_mm = msg.relative_alt",
            "            climb = msg.relative_alt - (baseline_mm or 0)",
            "            if 0 < climb <= plausible_max_mm and climb >= target_mm:",
            "                hits += 1",
            "                if hits >= 2:",
            "                    return True",
            "            else:",
            "                hits = 0",
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
            "    # wait_ready_to_arm：EKF 启动后需数秒设定 origin/home，期间会",
            "    # 报 'Arm: Need Position Estimate'，这是强制硬检查，ARMING_CHECK=0",
            "    # 和 force-arm(21196) 都绕不过。必须反复重试直到 EKF 就绪。",
            "    armed = False",
            "    arm_deadline = time.time() + 45",
            "    while time.time() < arm_deadline:",
            "        mav.mav.command_long_send(",
            "            mav.target_system, mav.target_component,",
            "            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,",
            "            0, 1, 21196, 0, 0, 0, 0, 0,",
            "        )",
            "        ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)",
            "        if (ack is not None",
            "                and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM",
            "                and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):",
            "            armed = True",
            "            break",
            "        time.sleep(2)",
            "    if not armed:",
            "        return False",
            "    mav.mav.command_long_send(",
            "        mav.target_system, mav.target_component,",
            "        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,",
            "        0, 0, 0, 0, 0, 0, 0, altitude,",
            "    )",
            "    # relative_alt 是相对 home 高度；home 未设的瞬间可能等于绝对海拔",
            "    # （~584000mm），会让 '>= target' 在地面误判。按基线相对爬升判断，",
            "    # 加合理性上限 + 连续 2 帧确认。",
            "    target_mm = int(altitude * 0.7 * 1000)",
            "    plausible_max_mm = int(altitude * 3 * 1000) + 5000",
            "    baseline_mm = None",
            "    hits = 0",
            "    deadline = time.time() + 30",
            "    while time.time() < deadline:",
            "        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)",
            "        if msg is not None:",
            "            if baseline_mm is None or msg.relative_alt < baseline_mm:",
            "                baseline_mm = msg.relative_alt",
            "            climb = msg.relative_alt - (baseline_mm or 0)",
            "            if 0 < climb <= plausible_max_mm and climb >= target_mm:",
            "                hits += 1",
            "                if hits >= 2:",
            "                    return True",
            "            else:",
            "                hits = 0",
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
        sitl_home: Optional[str] = None,
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
                # 本测试若是新拉起的 SITL（--wipe 全新启动），机体已是干净初始态，
                # 无需 reset_drone_state；而且在 Gazebo 下 reset 里的 SIM_*/EKF
                # 参数会扰乱 FDM 喂入的传感器估计，导致起飞失败。仅共享 SITL
                # （per_test_sitl=False，测试间状态会残留）时才需要 reset。
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
        """执行单个 L2 测试，返回 (passed, message)。所有 inject/verify 调度
        都通过 sitl_specs 注册表完成，无 if/elif 解释器。

        fresh_sitl=True 表示本测试是独立新拉起的 SITL（--wipe），机体已是
        干净初始态，跳过 reset_drone_state（在 Gazebo 下 reset 会扰乱 FDM
        传感器估计，导致起飞失败）。
        """
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

        # 仅共享 SITL 时才需清理跨测试状态污染；新拉起的 SITL 已是干净态
        if not fresh_sitl:
            ctx.reset_drone_state()

        # ── Apply this test's resolved ArduPilot params to the running SITL ──
        # These carry the actuator wiring the verify depends on (e.g. parachute
        # SERVO8_FUNCTION=27 / CHUTE_SERVO_ON=2000, gripper SERVO7_FUNCTION=28 /
        # GRIP_RELEASE=2000). They live in the catalogue but are NOT in the boot
        # .parm, so without this the release drives an unassigned servo channel and
        # SERVO_OUTPUT_RAW-based verifies read 0 (deterministic false negative).
        # Applied BEFORE inject (hence before takeoff), so the servo latches at the
        # release PWM when the fault fires. Non-numeric (unresolved) values skipped.
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
        sitl_home: Optional[str] = None,
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
