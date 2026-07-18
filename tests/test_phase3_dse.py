"""Tests for the DSE best-config injectors (dse_injectors module).

The scalar-DSE helpers this file also used to cover (_define_design_space /
_score_config_against_requirements) were removed with the legacy scalar MCTS.
"""
from __future__ import annotations

import sys
from types import ModuleType

for _name, _attrs in [
    ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
    ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
    ("syside", {}),
]:
    if _name not in sys.modules:
        try:  # prefer the real package — a stub here poisons later test files
            __import__(_name)
            continue
        except ImportError:
            pass
        _mod = ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod

from src.agents.orchestrator import Orchestrator
from src.agents.dse_injectors import apply_best_config_to_model
from src.dse.design_space import DesignConfiguration, ParameterType
from src.llm.interface import LLMResponse, Message, MockLLM
from src.sysml.model import (
    AttributeUsage, ElementRef, PartDefinition, PortUsage,
    FeatureDirection, SysMLModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_orchestrator() -> Orchestrator:
    return Orchestrator(llm=MockLLM())


def _make_model(part_names=("FlightController", "SensorArray")) -> SysMLModel:
    model = SysMLModel(name="DroneSystem", description="test model")
    for name in part_names:
        model.part_definitions.append(PartDefinition(name=name))
    return model


_REQUIREMENTS = [
    "REQ-FUNC-001: The drone shall navigate to waypoints autonomously.",
    "REQ-PERF-001: The drone shall update navigation state at 200 Hz.",
    "REQ-SAFE-001: The drone shall execute emergency landing when battery < 10% is detected.",
    "REQ-SAFE-002: The drone shall halt motors when sensor failure is detected.",
    "REQ-INTF-001: The drone shall exchange telemetry with ground station via MAVLink.",
    "REQ-CONS-001: The drone shall comply with EASA drone regulations.",
]


# ─────────────────────────────────────────────────────────────────────────────
# 方案 A: Requirements-driven design space
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyBestConfigToModel:

    def _model_with_freq_attr(self) -> SysMLModel:
        model = SysMLModel(name="DroneSystem", description="")
        part = PartDefinition(name="FlightController")
        attr = AttributeUsage(name="controlFrequency")
        attr.default_value = "100.0"
        attr.unit = "Hz"
        part.attributes.append(attr)
        port = PortUsage(name="commandIn")
        port.direction = FeatureDirection.IN
        part.ports.append(port)
        model.part_definitions.append(part)
        return model

    def test_frequency_attribute_updated(self):
        model = self._model_with_freq_attr()
        config = DesignConfiguration(
            name="best",
            parameters={"control_frequency_hz": 400.0},
        )
        apply_best_config_to_model(config, model)
        attr = model.part_definitions[0].attributes[0]
        assert float(attr.default_value) == 400.0

    def test_redundancy_written_to_model_description(self):
        model = self._model_with_freq_attr()
        config = DesignConfiguration(
            name="best",
            parameters={"redundancy_level": "dual"},
        )
        apply_best_config_to_model(config, model)
        assert "redundancy=dual" in model.description

    def test_redundancy_none_does_not_modify_description(self):
        model = self._model_with_freq_attr()
        original_desc = model.description
        config = DesignConfiguration(
            name="best",
            parameters={"redundancy_level": "none"},
        )
        apply_best_config_to_model(config, model)
        assert model.description == original_desc

    def test_protocol_assigned_to_undeclared_port_type(self):
        model = self._model_with_freq_attr()
        # Port has no type_ref
        assert model.part_definitions[0].ports[0].type_ref is None
        config = DesignConfiguration(
            name="best",
            parameters={"communication_protocol": "MAVLink"},
        )
        apply_best_config_to_model(config, model)
        port = model.part_definitions[0].ports[0]
        assert port.type_ref is not None
        assert "MAVLink" in port.type_ref.name

    def test_protocol_not_overwritten_when_already_set(self):
        model = self._model_with_freq_attr()
        model.part_definitions[0].ports[0].type_ref = ElementRef(name="ExistingProtocol")
        config = DesignConfiguration(
            name="best",
            parameters={"communication_protocol": "CAN"},
        )
        apply_best_config_to_model(config, model)
        # Should NOT overwrite existing type_ref
        assert model.part_definitions[0].ports[0].type_ref.name == "ExistingProtocol"

    def test_sensor_count_stored_in_model_metadata(self):
        model = self._model_with_freq_attr()
        config = DesignConfiguration(
            name="best",
            parameters={"num_sensors": 4},
        )
        apply_best_config_to_model(config, model)
        assert model.metadata.get("recommended_sensor_count") == 4
        assert model.metadata.get("dse_best_config") == "best"

    def test_empty_params_does_not_raise(self):
        model = self._model_with_freq_attr()
        config = DesignConfiguration(name="empty", parameters={})
        apply_best_config_to_model(config, model)  # must not raise
