from src.dse.design_space import DesignConfiguration
from src.dse.evaluator import DesignEvaluator
from src.dse.model_facts import extract_dse_model_facts
from src.sysml.model import SysMLModel


_SENSOR_MODEL = """package P {
    port def DataPort;
    part def GPSUnit { out port data : DataPort; }
    part def Controller { in port first : DataPort; in port second : DataPort; }
    part gps1 : GPSUnit;
    part gps2 : GPSUnit;
    part controller : Controller;
    connect gps1.data to controller.first;
    connect gps2.data to controller.second;
}"""


def test_sensor_connections_are_one_shared_fact_for_dse_consumers():
    facts = extract_dse_model_facts(_SENSOR_MODEL, {"num_sensors": 2})

    assert facts.sensors is not None
    assert facts.sensors.instances == frozenset({"gps1", "gps2"})
    assert facts.sensors.connected_instances == frozenset({"gps1", "gps2"})


def test_sensor_fidelity_uses_connection_objects_without_old_regex_state():
    model = SysMLModel(name="P")
    model.metadata["last_sysml_text"] = _SENSOR_MODEL

    score = DesignEvaluator()._score_dse_fidelity(
        DesignConfiguration(name="candidate"),
        model,
        DesignConfiguration(name="selected", parameters={"num_sensors": 2}),
    )

    assert 0.0 <= score <= 1.0
