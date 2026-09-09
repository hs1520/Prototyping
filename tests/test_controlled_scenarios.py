from __future__ import annotations

from src.simulation.controlled_scenarios import evaluate_controlled_scenarios


CONNECTED = """package D {
    port def Signal;
    part def SensorUnit { out port data : Signal; }
    part def SafetyMonitor {
        in port battery : Signal;
        out port override : Signal;
    }
    part def PowerSystem { out port power : Signal; }
    part def CommunicationSystem {
        in port telemetry : Signal;
        out port command : Signal;
    }
    part def FlightController {
        in port sensorIn : Signal;
        in port safetyIn : Signal;
        in port powerIn : Signal;
        in port commandIn : Signal;
        out port telemetryOut : Signal;
        out port actuatorCmd : Signal;
    }
    part def PayloadActuator { in port command : Signal; }
    part def System {
        part sensor : SensorUnit;
        part safety : SafetyMonitor;
        part power : PowerSystem;
        part comms : CommunicationSystem;
        part controller : FlightController;
        part actuator : PayloadActuator;
        connect sensor.data to controller.sensorIn;
        connect safety.override to controller.safetyIn;
        connect power.power to safety.battery;
        connect power.power to controller.powerIn;
        connect comms.command to controller.commandIn;
        connect controller.telemetryOut to comms.telemetry;
        connect controller.actuatorCmd to actuator.command;
    }
}"""


def test_fixed_denominator_all_pass():
    report = evaluate_controlled_scenarios(CONNECTED, model_name="D")

    assert report["scenario_set_fixed"] is True
    assert report["scenario_count"] == 7
    assert report["counts"] == {"PASS": 7, "FAIL": 0}


def test_missing_role_fails_not_removed():
    report = evaluate_controlled_scenarios(
        CONNECTED.replace("part actuator : PayloadActuator;", ""),
        model_name="D",
    )

    assert report["scenario_count"] == 7
    row = next(
        item for item in report["results"]
        if item["scenario_id"] == "S07_CONTROLLER_TO_ACTUATOR"
    )
    assert row["status"] == "FAIL"
    assert row["failure_code"] == "MISSING_TARGET_ROLE"
