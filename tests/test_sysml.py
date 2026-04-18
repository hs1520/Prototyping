"""Tests for SysML v2 model representation."""

import pytest
from src.sysml.model import (
    Action,
    Attribute,
    Block,
    Connector,
    FeatureDirection,
    Multiplicity,
    Port,
    Requirement,
    SysMLModel,
)


class TestRequirement:
    def test_creation(self):
        req = Requirement(name="REQ_001", text="The system shall operate at 100Hz")
        assert req.name == "REQ_001"
        assert req.text == "The system shall operate at 100Hz"
        assert req.satisfaction_level == 0.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            Requirement(name="")

    def test_str_representation(self):
        req = Requirement(name="REQ_001", text="The system shall operate")
        sysml_str = str(req)
        assert "requirement" in sysml_str
        assert "REQ_001" in sysml_str


class TestPort:
    def test_creation(self):
        port = Port(name="dataOut", direction=FeatureDirection.OUT, port_type="DataPort")
        assert port.name == "dataOut"
        assert port.direction == FeatureDirection.OUT

    def test_str_representation(self):
        port = Port(name="sensorIn", direction=FeatureDirection.IN, port_type="SensorPort")
        s = str(port)
        assert "port" in s
        assert "sensorIn" in s

    def test_conjugated_port(self):
        port = Port(name="cmdIn", port_type="CmdPort", conjugated=True)
        s = str(port)
        assert "~" in s


class TestAttribute:
    def test_creation(self):
        attr = Attribute(
            name="samplingRate",
            attribute_type="Real",
            default_value=100.0,
            unit="Hz",
        )
        assert attr.name == "samplingRate"
        assert attr.default_value == 100.0
        assert attr.unit == "Hz"

    def test_str_representation(self):
        attr = Attribute(name="frequency", attribute_type="Real", default_value=50.0, unit="Hz")
        s = str(attr)
        assert "attribute" in s
        assert "frequency" in s
        assert "Real" in s


class TestBlock:
    def test_creation(self):
        block = Block(name="Controller", block_type="part def")
        assert block.name == "Controller"
        assert block.block_type == "part def"
        assert len(block.ports) == 0

    def test_add_port(self):
        block = Block(name="Sensor")
        port = Port(name="dataOut", direction=FeatureDirection.OUT)
        block.add_port(port)
        assert len(block.ports) == 1
        assert block.ports[0].name == "dataOut"

    def test_add_attribute(self):
        block = Block(name="Sensor")
        attr = Attribute(name="samplingRate", attribute_type="Real")
        block.add_attribute(attr)
        assert len(block.attributes) == 1

    def test_add_sub_part(self):
        parent = Block(name="System")
        child = Block(name="Subsystem")
        parent.add_sub_part(child)
        assert len(parent.sub_parts) == 1

    def test_str_representation(self):
        block = Block(name="Controller", short_description="Main controller")
        s = str(block)
        assert "part def" in s
        assert "Controller" in s

    def test_satisfies_and_refines_are_serialized(self):
        block = Block(name="Controller")
        block.add_satisfies("REQ_001")
        block.add_refinement("AbstractController")

        s = str(block)
        assert "satisfy REQ_001" in s
        assert "refines AbstractController" in s


class TestSysMLModel:
    def test_creation(self):
        model = SysMLModel(name="TestSystem", description="A test system")
        assert model.name == "TestSystem"
        assert len(model.requirements) == 0
        assert len(model.blocks) == 0

    def test_add_requirement(self):
        model = SysMLModel(name="TestSystem")
        req = Requirement(name="REQ_001", text="The system shall work")
        model.add_requirement(req)
        assert len(model.requirements) == 1

    def test_add_block(self):
        model = SysMLModel(name="TestSystem")
        block = Block(name="Controller")
        model.add_block(block)
        assert len(model.blocks) == 1

    def test_get_block_by_name(self):
        model = SysMLModel(name="TestSystem")
        block = Block(name="Controller")
        model.add_block(block)
        found = model.get_block_by_name("Controller")
        assert found is not None
        assert found.name == "Controller"

    def test_get_block_not_found(self):
        model = SysMLModel(name="TestSystem")
        assert model.get_block_by_name("NonExistent") is None

    def test_to_sysml_text(self):
        model = SysMLModel(name="DroneSystem", description="Autonomous drone")
        block = Block(name="FlightController")
        block.add_port(Port(name="sensorIn", direction=FeatureDirection.IN))
        model.add_block(block)
        req = Requirement(name="REQ_001", text="The drone shall fly")
        model.add_requirement(req)

        sysml_text = model.to_sysml_text()
        assert "package DroneSystem" in sysml_text
        assert "FlightController" in sysml_text
        assert "REQ_001" in sysml_text

    def test_get_summary(self):
        model = SysMLModel(name="TestSystem")
        model.add_block(Block(name="A"))
        model.add_block(Block(name="B"))
        model.add_requirement(Requirement(name="R1", text="req 1"))

        summary = model.get_summary()
        assert summary["blocks_count"] == 2
        assert summary["requirements_count"] == 1
        assert "A" in summary["blocks"]

    def test_to_sysml_text_includes_refinement_relations(self):
        model = SysMLModel(name="RefinedSystem")
        req = Requirement(name="REQ_001", text="The system shall respond quickly")
        req.derived_from.append("REQ_000")
        model.add_requirement(req)

        block = Block(name="Controller")
        block.add_satisfies("REQ_001")
        block.add_refinement("AbstractController")
        model.add_block(block)

        sysml_text = model.to_sysml_text()
        assert "satisfy REQ_001" in sysml_text
        assert "refines AbstractController" in sysml_text
        assert "derived from: REQ_000" in sysml_text

    def test_add_connector(self):
        model = SysMLModel(name="TestSystem")
        conn = Connector(
            name="conn1",
            source_block_id="Sensor",
            source_port_id="dataOut",
            target_block_id="Controller",
            target_port_id="sensorIn",
        )
        model.add_connector(conn)
        assert len(model.connectors) == 1

    def test_full_model_sysml_text(self):
        """Test a complete model serializes to valid SysML v2 text."""
        model = SysMLModel(name="ControlSystem")
        
        sensor = Block(name="Sensor", block_type="part def")
        sensor.add_port(Port(name="dataOut", direction=FeatureDirection.OUT))
        sensor.add_attribute(Attribute(name="rate", attribute_type="Real", default_value=100.0))
        model.add_block(sensor)

        ctrl = Block(name="Controller", block_type="part def")
        ctrl.add_port(Port(name="sensorIn", direction=FeatureDirection.IN))
        ctrl.add_port(Port(name="cmdOut", direction=FeatureDirection.OUT))
        model.add_block(ctrl)

        model.add_connector(Connector(
            name="s2c",
            source_block_id="Sensor",
            source_port_id="dataOut",
            target_block_id="Controller",
            target_port_id="sensorIn",
        ))

        sysml = model.to_sysml_text()
        assert sysml.startswith("package ControlSystem")
        assert sysml.endswith("}")
        assert "Sensor" in sysml
        assert "Controller" in sysml
