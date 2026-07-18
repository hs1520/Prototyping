"""
tests/test_connectivity_fixer.py

Unit tests for src/simulation/connectivity_fixer.py.

Run with:
    python tests/test_connectivity_fixer.py
"""

from __future__ import annotations

import sys

import src.simulation.connectivity_fixer as _mod
from src.simulation.connectivity_fixer import (
    build_port_directory,
    parse_connects,
    validate_connects,
    merge_connects,
    build_connectivity_prompt,
    extract_connect_lines,
)


# ---------------------------------------------------------------------------
# 测试模型:7 个 part def + 装配段
# ---------------------------------------------------------------------------

SYSML = """\
package Drone {
    port def DataPort { inout item d : DataItem; }
    port def PowerPort { inout item p : PowerItem; }

    part def FlightController {
        in port sensorData : DataPort;
        out port motorCmd : DataPort;
        in port powerIn : PowerPort;
    }
    part def Perception {
        out port sensorData : DataPort;
    }
    part def PowerSystem {
        out port powerOut : PowerPort;
    }
    part def Airframe {
        inout port physicalMount : DataPort;
    }

    part def DroneSystem {
        part flightController : FlightController;
        part perception : Perception;
        part powerSystem : PowerSystem;
        part airframe : Airframe;

        connect perception.sensorData to flightController.sensorData;
    }
}
"""

_PASS = 0
_FAIL = 0


def ok(name: str, cond: bool, msg: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        print(f"  PASS  {name}")
        _PASS += 1
    else:
        print(f"  FAIL  {name}  {msg}")
        _FAIL += 1
    # Enforce under pytest too (standalone still prints the running tally above).
    assert cond, f"{name}: {msg}"


# ---------------------------------------------------------------------------
# T1 — build_port_directory
# ---------------------------------------------------------------------------

def test_port_directory():
    print("T1  build_port_directory")
    d = build_port_directory(SYSML)
    ok("inst_fc",        d.has_instance("flightController"))
    ok("inst_type",      d.instance_type.get("flightController") == "FlightController")
    ok("fc_sensorData",  d.port("flightController", "sensorData").direction == "in")
    ok("fc_motorCmd",    d.port("flightController", "motorCmd").direction == "out")
    ok("fc_port_type",   d.port("flightController", "sensorData").port_type == "DataPort")
    ok("airframe_inout", d.port("airframe", "physicalMount").direction == "inout")
    ok("no_fake_port",   d.port("flightController", "nope") is None)


# ---------------------------------------------------------------------------
# T2 — parse_connects
# ---------------------------------------------------------------------------

def test_parse_connects():
    print("T2  parse_connects")
    conns = parse_connects(SYSML)
    ok("one_connect", len(conns) == 1, f"got {len(conns)}")
    c = conns[0]
    ok("connect_fields",
       c.src_inst == "perception" and c.src_port == "sensorData"
       and c.tgt_inst == "flightController" and c.tgt_port == "sensorData")


# ---------------------------------------------------------------------------
# T3 — validate: 合法连接被接受
# ---------------------------------------------------------------------------

def test_validate_accepts_valid():
    print("T3  validate — 合法连接接受")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    # flightController.motorCmd(out) → ??? 没有匹配的 in DataPort 目标,换一个:
    # powerSystem.powerOut(out:PowerPort) → flightController.powerIn(in:PowerPort) ✓
    v = validate_connects(
        ["connect powerSystem.powerOut to flightController.powerIn;"], d, existing)
    ok("accepted_one", len(v.accepted) == 1, f"acc={[s.to_sysml() for s in v.accepted]} rej={v.rejected}")
    ok("no_reject",    len(v.rejected) == 0, f"rej={v.rejected}")


# ---------------------------------------------------------------------------
# T4 — validate: 凭空造端口被拒(核心防线)
# ---------------------------------------------------------------------------

def test_validate_rejects_fabricated_port():
    print("T4  validate — 拒绝凭空造端口")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    # airframe 没有 telemetryOut 端口 —— 模拟旧回归
    v = validate_connects(
        ["connect airframe.telemetryOut to flightController.sensorData;"], d, existing)
    ok("nothing_accepted", len(v.accepted) == 0, f"acc={[s.to_sysml() for s in v.accepted]}")
    ok("rejected_reason",  any("凭空造端口" in r[1] for r in v.rejected), f"rej={v.rejected}")


# ---------------------------------------------------------------------------
# T5 — validate: 方向错误被拒
# ---------------------------------------------------------------------------

def test_validate_rejects_wrong_direction():
    print("T5  validate — 拒绝方向错误")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    # perception.sensorData 是 out;flightController.motorCmd 是 out → out→out 非法
    v = validate_connects(
        ["connect perception.sensorData to flightController.motorCmd;"], d, existing)
    ok("rejected", len(v.accepted) == 0 and len(v.rejected) == 1, f"acc={v.accepted} rej={v.rejected}")
    ok("dir_reason", any("方向" in r[1] for r in v.rejected), f"rej={v.rejected}")


# ---------------------------------------------------------------------------
# T6 — validate: 双源 in 端口被拒
# ---------------------------------------------------------------------------

def test_validate_rejects_dual_driver():
    print("T6  validate — 拒绝双源 in 端口")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    # flightController.sensorData(in) 已被 perception 驱动;再接一个源 → 双源
    # 需要另一个 out:DataPort 源。motorCmd 是 out:DataPort。
    v = validate_connects(
        ["connect flightController.motorCmd to flightController.sensorData;"], d, existing)
    ok("rejected",  len(v.accepted) == 0, f"acc={v.accepted}")
    ok("dual_reason", any("双源" in r[1] or "已被驱动" in r[1] for r in v.rejected), f"rej={v.rejected}")


# ---------------------------------------------------------------------------
# T7 — validate: 类型不匹配被拒
# ---------------------------------------------------------------------------

def test_validate_rejects_type_mismatch():
    print("T7  validate — 拒绝类型不匹配")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    # powerSystem.powerOut(PowerPort) → flightController.sensorData(in:DataPort) 类型不符
    v = validate_connects(
        ["connect powerSystem.powerOut to flightController.sensorData;"], d, existing)
    ok("rejected",   len(v.accepted) == 0, f"acc={v.accepted}")
    ok("type_reason", any("类型不匹配" in r[1] for r in v.rejected), f"rej={v.rejected}")


# ---------------------------------------------------------------------------
# T8 — validate: 重复连接被拒
# ---------------------------------------------------------------------------

def test_validate_rejects_duplicate():
    print("T8  validate — 拒绝重复")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect perception.sensorData to flightController.sensorData;"], d, existing)
    ok("rejected", len(v.accepted) == 0, f"acc={v.accepted}")
    ok("dup_reason", any("重复" in r[1] for r in v.rejected), f"rej={v.rejected}")


# ---------------------------------------------------------------------------
# T9 — merge_connects:插在最后一条 connect 之后
# ---------------------------------------------------------------------------

def test_merge_after_last_connect():
    print("T9  merge_connects — 插在已有 connect 后")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect powerSystem.powerOut to flightController.powerIn;"], d, existing)
    m = merge_connects(SYSML, v.accepted)
    ok("n_added",      m.n_added == 1)
    ok("text_has_new", "connect powerSystem.powerOut to flightController.powerIn;" in m.merged_text)
    # 原有 connect 仍在
    ok("text_keeps_old", "connect perception.sensorData to flightController.sensorData;" in m.merged_text)
    # 新连接仍在 DroneSystem 块内(在最后的 '}' 之前)
    assembly_close = m.merged_text.rfind("}")
    new_pos = m.merged_text.find("powerSystem.powerOut to flightController.powerIn")
    ok("inside_assembly", new_pos < assembly_close)


# ---------------------------------------------------------------------------
# T10 — extract_connect_lines:从带散文/围栏的返回里抽 connect
# ---------------------------------------------------------------------------

def test_extract_connect_lines():
    print("T10  extract_connect_lines")
    resp = (
        "Here are the connections:\n"
        "```sysml\n"
        "connect a.x to b.y;\n"
        "connect c.z to d.w;\n"
        "```\n"
        "These fix the issue.\n"
    )
    lines = extract_connect_lines(resp)
    ok("two_lines", len(lines) == 2, f"got {lines}")
    ok("clean",     all(l.startswith("connect ") and l.endswith(";") for l in lines))


# ---------------------------------------------------------------------------
# T11 — build_connectivity_prompt:含端口目录 + 硬规则,不含整模型
# ---------------------------------------------------------------------------

def test_build_prompt():
    print("T11  build_connectivity_prompt")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    failed = [{"name": "power_powerSystem_to_airframe", "src": "powerSystem", "tgts": ["airframe"]}]
    p = build_connectivity_prompt(d, existing, failed, isolated_parts=["airframe"])

    ok("has_directory",  "[PORT DIRECTORY]" in p)
    ok("lists_ports",    "powerOut(out:PowerPort)" in p)
    ok("has_existing",   "[EXISTING CONNECTIONS]" in p)
    ok("has_failed",     "airframe" in p)
    ok("has_isolated",   "ISOLATED PARTS" in p)
    ok("forbids_invent", "NEVER invent a new port" in p)
    ok("rule_single_src","only ONE source" in p)
    # prompt 应紧凑(端口目录式,而非整模型):远小于 SYSML 行数的若干倍
    ok("compact",        len(p.splitlines()) <= 60, f"lines={len(p.splitlines())}")


if __name__ == "__main__":
    test_port_directory()
    test_parse_connects()
    test_validate_accepts_valid()
    test_validate_rejects_fabricated_port()
    test_validate_rejects_wrong_direction()
    test_validate_rejects_dual_driver()
    test_validate_rejects_type_mismatch()
    test_validate_rejects_duplicate()
    test_merge_after_last_connect()
    test_extract_connect_lines()
    test_build_prompt()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
