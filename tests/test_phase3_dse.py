"""Tests for Phase 3 Design Space Exploration optimisations.

Covers:
  方案 A — requirements-driven _define_design_space()
  方案 B — _score_config_against_requirements()
  方案 C — _apply_best_config_to_model()
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _name, _attrs in [
    ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
    ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
    ("syside", {}),
]:
    if _name not in sys.modules:
        _mod = ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod

from src.agents.orchestrator import Orchestrator
from src.agents.mcts_injectors import apply_best_config_to_model
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

class TestDefineDesignSpaceRequirementsDriven:

    def test_redundancy_choices_scale_with_safe_count(self):
        orch = _make_orchestrator()
        model = _make_model()

        # 0 SAFE reqs → only "none"
        space_0 = orch._define_design_space(model, requirements=[])
        r0 = next(p for p in space_0.parameters if p.name == "redundancy_level")
        assert r0.choices == ["none"]

        # 1 SAFE req → ["none", "dual"]
        space_1 = orch._define_design_space(model, requirements=[
            "REQ-SAFE-001: The drone shall execute emergency landing when battery failure detected."
        ])
        r1 = next(p for p in space_1.parameters if p.name == "redundancy_level")
        assert r1.choices == ["none", "dual"]

        # 2+ SAFE reqs → ["none", "dual", "triple"]
        space_2 = orch._define_design_space(model, requirements=_REQUIREMENTS)
        r2 = next(p for p in space_2.parameters if p.name == "redundancy_level")
        assert "triple" in r2.choices

    def test_protocol_extracted_from_intf_requirements(self):
        orch = _make_orchestrator()
        model = _make_model()
        space = orch._define_design_space(model, requirements=_REQUIREMENTS)
        proto_param = next(p for p in space.parameters if p.name == "communication_protocol")
        choices_upper = [c.upper() for c in proto_param.choices]
        assert "MAVLINK" in choices_upper

    def test_frequency_range_derived_from_perf_requirements(self):
        orch = _make_orchestrator()
        model = _make_model()
        space = orch._define_design_space(model, requirements=_REQUIREMENTS)
        freq_param = next(p for p in space.parameters if p.name == "control_frequency_hz")
        # PERF req specifies 200 Hz → default should be ~200
        assert freq_param.default_value == 200.0
        assert freq_param.min_value <= 200.0 <= freq_param.max_value

    def test_frequency_defaults_when_no_perf_requirements(self):
        orch = _make_orchestrator()
        model = _make_model()
        space = orch._define_design_space(model, requirements=[
            "REQ-FUNC-001: The system shall operate correctly."
        ])
        freq_param = next(p for p in space.parameters if p.name == "control_frequency_hz")
        assert freq_param.default_value == 100.0   # fallback

    def test_distributed_control_default_true_for_large_models(self):
        orch = _make_orchestrator()
        # 5 parts → should default to distributed=True
        model = _make_model(("P1", "P2", "P3", "P4", "P5"))
        space = orch._define_design_space(model, requirements=[])
        dist_param = next(p for p in space.parameters if p.name == "distributed_control")
        assert dist_param.default_value is True

    def test_sensor_count_anchored_to_existing_sensor_parts(self):
        orch = _make_orchestrator()
        model = _make_model(("FlightController", "SensorUnit1", "SensorUnit2"))
        space = orch._define_design_space(model, requirements=[])
        sensor_param = next(p for p in space.parameters if p.name == "num_sensors")
        # 2 existing sensor-named parts → min choice ≥ 2
        assert min(sensor_param.choices) >= 2

    def test_all_five_parameters_always_present(self):
        orch = _make_orchestrator()
        model = _make_model()
        space = orch._define_design_space(model, requirements=_REQUIREMENTS)
        names = {p.name for p in space.parameters}
        assert names == {
            "redundancy_level", "communication_protocol",
            "control_frequency_hz", "distributed_control", "num_sensors",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 方案 B: Requirement-bound scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreConfigAgainstRequirements:

    def _config(self, **params) -> DesignConfiguration:
        return DesignConfiguration(name="test", parameters=params)

    def test_perf_score_high_when_frequency_meets_requirement(self):
        scores = Orchestrator._score_config_against_requirements(
            self._config(control_frequency_hz=200.0, redundancy_level="none",
                         num_sensors=2),
            _REQUIREMENTS,
        )
        # 200 Hz matches "200 Hz" requirement → ratio = 1.0
        assert scores["perf_satisfaction"] >= 0.99

    def test_perf_score_low_when_frequency_below_requirement(self):
        scores = Orchestrator._score_config_against_requirements(
            self._config(control_frequency_hz=50.0, redundancy_level="none",
                         num_sensors=2),
            _REQUIREMENTS,
        )
        assert scores["perf_satisfaction"] < 0.5

    def test_safety_score_improves_with_redundancy(self):
        base_reqs = _REQUIREMENTS  # 2 SAFE reqs
        score_none = Orchestrator._score_config_against_requirements(
            self._config(redundancy_level="none", control_frequency_hz=200.0, num_sensors=2),
            base_reqs,
        )
        score_dual = Orchestrator._score_config_against_requirements(
            self._config(redundancy_level="dual", control_frequency_hz=200.0, num_sensors=2),
            base_reqs,
        )
        score_triple = Orchestrator._score_config_against_requirements(
            self._config(redundancy_level="triple", control_frequency_hz=200.0, num_sensors=2),
            base_reqs,
        )
        assert score_none["safety_margin"] < score_dual["safety_margin"] <= score_triple["safety_margin"]

    def test_safety_score_full_when_no_safe_requirements(self):
        reqs_no_safe = [r for r in _REQUIREMENTS if "-SAFE-" not in r]
        scores = Orchestrator._score_config_against_requirements(
            self._config(redundancy_level="none", control_frequency_hz=100.0, num_sensors=2),
            reqs_no_safe,
        )
        assert scores["safety_margin"] == 1.0

    def test_protocol_match_score_high_when_protocol_in_intf_req(self):
        scores = Orchestrator._score_config_against_requirements(
            self._config(communication_protocol="MAVLink",
                         control_frequency_hz=200.0, num_sensors=2),
            _REQUIREMENTS,
        )
        assert scores["protocol_match"] == 1.0

    def test_protocol_match_score_low_when_protocol_absent(self):
        scores = Orchestrator._score_config_against_requirements(
            self._config(communication_protocol="SPI",
                         control_frequency_hz=200.0, num_sensors=2),
            _REQUIREMENTS,
        )
        assert scores["protocol_match"] < 0.5

    def test_simplicity_inversely_proportional_to_sensors(self):
        s_few = Orchestrator._score_config_against_requirements(
            self._config(num_sensors=1, control_frequency_hz=100.0), []
        )
        s_many = Orchestrator._score_config_against_requirements(
            self._config(num_sensors=6, control_frequency_hz=100.0), []
        )
        assert s_few["simplicity"] > s_many["simplicity"]

    def test_returns_dict_with_expected_keys(self):
        scores = Orchestrator._score_config_against_requirements(
            self._config(control_frequency_hz=100.0, redundancy_level="none", num_sensors=3),
            _REQUIREMENTS,
        )
        assert {"perf_satisfaction", "safety_margin", "protocol_match", "simplicity"}.issubset(
            scores.keys()
        )
        assert all(0.0 <= v <= 1.0 for v in scores.values())


# ─────────────────────────────────────────────────────────────────────────────
# 方案 C: Apply best config back to model
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
        assert model.metadata.get("mcts_best_config") == "best"

    def test_empty_params_does_not_raise(self):
        model = self._model_with_freq_attr()
        config = DesignConfiguration(name="empty", parameters={})
        apply_best_config_to_model(config, model)  # must not raise
