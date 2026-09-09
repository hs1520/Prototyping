from __future__ import annotations

import sys

from src.simulation.connectivity_fixer import (
    build_port_directory,
    parse_connects,
    validate_connects,
    merge_connects,
    build_connectivity_prompt,
    extract_connect_lines,
    audit_connects,
)


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


def test_parse_connects():
    print("T2  parse_connects")
    conns = parse_connects(SYSML)
    ok("one_connect", len(conns) == 1, f"got {len(conns)}")
    c = conns[0]
    ok("connect_fields",
       c.src_inst == "perception" and c.src_port == "sensorData"
       and c.tgt_inst == "flightController" and c.tgt_port == "sensorData")


def test_parse_connects_ignores_prose():
    text = """package P {
        requirement def REQ_SAFE_001 {
            doc /* Example only: connect fake.out to fake.in; */
        }
        connect first.out to target.one;
        connect second.out to target.two;
        // connect commented.out to target.three;
    }"""

    assert [statement.to_sysml() for statement in parse_connects(text)] == [
        "connect first.out to target.one;",
        "connect second.out to target.two;",
    ]


def test_validate_accepts_valid():
    print("T3  validate — 合法连接接受")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect powerSystem.powerOut to flightController.powerIn;"], d, existing)
    ok("accepted_one", len(v.accepted) == 1, f"acc={[s.to_sysml() for s in v.accepted]} rej={v.rejected}")
    ok("no_reject",    len(v.rejected) == 0, f"rej={v.rejected}")


def test_validate_rejects_fabricated_port():
    print("T4  validate — 拒绝凭空造端口")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect airframe.telemetryOut to flightController.sensorData;"], d, existing)
    ok("nothing_accepted", len(v.accepted) == 0, f"acc={[s.to_sysml() for s in v.accepted]}")
    ok("rejected_reason",  any("凭空造端口" in r[1] for r in v.rejected), f"rej={v.rejected}")


def test_validate_rejects_wrong_direction():
    print("T5  validate — 拒绝方向错误")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect perception.sensorData to flightController.motorCmd;"], d, existing)
    ok("rejected", len(v.accepted) == 0 and len(v.rejected) == 1, f"acc={v.accepted} rej={v.rejected}")
    ok("dir_reason", any("方向" in r[1] for r in v.rejected), f"rej={v.rejected}")


def test_validate_rejects_dual_driver():
    print("T6  validate — 拒绝双源 in 端口")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect flightController.motorCmd to flightController.sensorData;"], d, existing)
    ok("rejected",  len(v.accepted) == 0, f"acc={v.accepted}")
    ok("dual_reason", any("双源" in r[1] or "已被驱动" in r[1] for r in v.rejected), f"rej={v.rejected}")


def test_validate_rejects_type_mismatch():
    print("T7  validate — 拒绝类型不匹配")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect powerSystem.powerOut to flightController.sensorData;"], d, existing)
    ok("rejected",   len(v.accepted) == 0, f"acc={v.accepted}")
    ok("type_reason", any("类型不匹配" in r[1] for r in v.rejected), f"rej={v.rejected}")


def test_validate_rejects_duplicate():
    print("T8  validate — 拒绝重复")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect perception.sensorData to flightController.sensorData;"], d, existing)
    ok("rejected", len(v.accepted) == 0, f"acc={v.accepted}")
    ok("dup_reason", any("重复" in r[1] for r in v.rejected), f"rej={v.rejected}")


def test_candidate_and_audit_agree():
    directory = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    invalid = (
        "connect airframe.telemetryOut to flightController.sensorData;",
        "connect perception.sensorData to flightController.motorCmd;",
        "connect powerSystem.powerOut to flightController.sensorData;",
    )
    for line in invalid:
        candidate = validate_connects([line], directory, existing)
        model = SYSML[:SYSML.rfind("}")] + f"    {line}\n" + SYSML[SYSML.rfind("}"):]
        audit = audit_connects(model)
        assert not candidate.accepted
        assert candidate.rejected
        assert audit.has_violations
        assert audit.violations[-1].stmt.to_sysml() == line


def test_merge_after_last_connect():
    print("T9  merge_connects — 插在已有 connect 后")
    d = build_port_directory(SYSML)
    existing = parse_connects(SYSML)
    v = validate_connects(
        ["connect powerSystem.powerOut to flightController.powerIn;"], d, existing)
    m = merge_connects(SYSML, v.accepted)
    ok("n_added",      m.n_added == 1)
    ok("text_has_new", "connect powerSystem.powerOut to flightController.powerIn;" in m.merged_text)
    ok("text_keeps_old", "connect perception.sensorData to flightController.sensorData;" in m.merged_text)
    assembly_close = m.merged_text.rfind("}")
    new_pos = m.merged_text.find("powerSystem.powerOut to flightController.powerIn")
    ok("inside_assembly", new_pos < assembly_close)


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
    # prompt 应紧凑(端口目录式,非整模型),行数远小于 SYSML 本身
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
