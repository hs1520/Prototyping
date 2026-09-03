from __future__ import annotations

import sys

from src.simulation.extractor import BehavioralGraph, PartNode, PortNode
from src.simulation.scenarios import _classify_node, auto_detect_scenarios


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


def _port(part: str, pname: str, direction: str) -> PortNode:
    return PortNode(id=f"{part}.{pname}", part_name=part, port_name=pname, direction=direction)


def _build_graph() -> BehavioralGraph:
    bg = BehavioralGraph()

    bg.parts["flightController"] = PartNode(
        id="flightController", def_name="FlightController",
        port_ids=["flightController.sensorData", "flightController.motorCmd"],
    )
    bg.ports["flightController.sensorData"] = _port("flightController", "sensorData", "in")
    bg.ports["flightController.motorCmd"]   = _port("flightController", "motorCmd", "out")

    bg.parts["powerSystem"] = PartNode(
        id="powerSystem", def_name="PowerSystem",
        port_ids=["powerSystem.powerOut"],
    )
    bg.ports["powerSystem.powerOut"] = _port("powerSystem", "powerOut", "out")

    bg.parts["communicationSystem"] = PartNode(
        id="communicationSystem", def_name="CommunicationSystem",
        port_ids=["communicationSystem.commStatus"],
    )
    bg.ports["communicationSystem.commStatus"] = _port("communicationSystem", "commStatus", "out")

    bg.parts["airframe"] = PartNode(
        id="airframe", def_name="Airframe",
        port_ids=["airframe.physicalMount"],
    )
    bg.ports["airframe.physicalMount"] = _port("airframe", "physicalMount", "inout")

    return bg


def test_classify_node():
    print("T1  _classify_node")
    bg = _build_graph()
    ok("fc_controller",
       _classify_node(bg.parts["flightController"], bg) == "controller")
    ok("power_power",
       _classify_node(bg.parts["powerSystem"], bg) == "power")
    ok("comms_comms",
       _classify_node(bg.parts["communicationSystem"], bg) == "comms")
    ok("airframe_structure",
       _classify_node(bg.parts["airframe"], bg) == "structure",
       f"got {_classify_node(bg.parts['airframe'], bg)}")


def test_single_inout_not_controller():
    print("T2  单 inout 且无结构关键词 → other(非 controller)")
    bg = BehavioralGraph()
    bg.parts["widgetX"] = PartNode(
        id="widgetX", def_name="WidgetX", port_ids=["widgetX.bus"])
    bg.ports["widgetX.bus"] = _port("widgetX", "bus", "inout")
    cls = _classify_node(bg.parts["widgetX"], bg)
    ok("not_controller", cls != "controller", f"got {cls}")
    ok("is_other",       cls == "other", f"got {cls}")


def test_no_spurious_airframe_scenarios():
    print("T3  airframe 不再派生 telemetry/control/uplink 场景")
    bg = _build_graph()
    scenarios = auto_detect_scenarios(bg)
    names = [s.name for s in scenarios]

    bad = [n for n in names
           if "airframe" in n and (
               n.startswith("telemetry_") or n.startswith("control_")
               or n.startswith("uplink_"))]
    ok("no_telemetry_control_uplink", bad == [], f"spurious: {bad}")

    airframe_as_src = [n for n in names if n.startswith(("telemetry_airframe", "control_airframe"))]
    ok("airframe_not_source", airframe_as_src == [], f"airframe src scenarios: {airframe_as_src}")


def test_airframe_power_target_only():
    print("T4  airframe 只作为 power 场景的目标")
    bg = _build_graph()
    scenarios = auto_detect_scenarios(bg)
    airframe_scen = [s for s in scenarios if "airframe" in s.name]
    ok("only_power_target",
       all(s.name.startswith("power_") and s.name.endswith("_to_airframe")
           for s in airframe_scen),
       f"airframe scenarios: {[s.name for s in airframe_scen]}")


if __name__ == "__main__":
    test_classify_node()
    test_single_inout_not_controller()
    test_no_spurious_airframe_scenarios()
    test_airframe_power_target_only()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)
