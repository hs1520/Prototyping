"""Tests for the SysML AST client/adapter integration."""

from __future__ import annotations

import pytest

from src.agents.design_agent import DesignAgent
from src.llm.interface import MockLLM
from src.sysml.ast_adapter import SysMLAstAdapter, SysMLAstMappingError
from src.sysml.ast_client import SysMLAstClientError
from src.sysml.model import Block, FeatureDirection, SysMLModel


class TestSysMLAstAdapter:
    def test_ast_to_model_maps_core_elements(self):
        adapter = SysMLAstAdapter()
        ast = {
            "schema_version": "1.0",
            "status": "ok",
            "ast": {
                "kind": "package",
                "name": "ControlSystem",
                "namespace": "pkg::control",
                "qualifiedName": "pkg::control::ControlSystem",
                "sourceUri": "file:///tmp/control.sysml",
                "confidence": 0.88,
                "imports": ["pkg::common"],
                "packages": ["pkg::control"],
                "requirements": [
                    {
                        "kind": "requirement",
                        "name": "REQ_001",
                        "text": "The system shall operate safely",
                        "sourceSpan": {"line": 1, "column": 1},
                    }
                ],
                "blocks": [
                    {
                        "kind": "part def",
                        "name": "FlightController",
                        "qualifiedName": "pkg::control::FlightController",
                        "isAbstract": False,
                        "ports": [
                            {
                                "kind": "port",
                                "name": "sensorIn",
                                "direction": "in",
                                "portType": "SensorPort",
                                "multiplicity": "1",
                            }
                        ],
                        "attributes": [
                            {
                                "kind": "attribute",
                                "name": "loopRate",
                                "attributeType": "Real",
                                "defaultValue": 100.0,
                                "unit": "Hz",
                            }
                        ],
                        "actions": [
                            {
                                "kind": "action",
                                "name": "stabilize",
                                "description": "Keep the vehicle stable",
                                "inputs": ["command"],
                                "outputs": ["thrust"],
                                "preconditions": ["battery_ok"],
                                "postconditions": ["stable"],
                            }
                        ],
                        "satisfies": ["REQ_001"],
                        "refines": ["AbstractFlightController"],
                        "generalizations": ["AbstractController"],
                        "constraints": ["loopRate > 0"],
                        "expressions": ["loopRate = 100"],
                        "sourceSpan": {"line": 10, "column": 1},
                    }
                ],
                "connectors": [
                    {
                        "kind": "connector",
                        "name": "sensorToController",
                        "sourceBlockId": "Sensor",
                        "sourcePortId": "dataOut",
                        "targetBlockId": "FlightController",
                        "targetPortId": "sensorIn",
                    }
                ],
                "constraints": ["model constraint"],
                "expressions": ["x + y"],
                "diagnostics": [{"level": "warning", "message": "partial parse"}],
            },
        }

        model = adapter.ast_to_model(ast)

        assert isinstance(model, SysMLModel)
        assert model.name == "ControlSystem"
        assert model.namespace == "pkg::control"
        assert model.qualified_name == "pkg::control::ControlSystem"
        assert model.packages == ["pkg::control"]
        assert model.imports == ["pkg::common"]
        assert model.confidence == 0.88
        assert model.ast_version == "1.0"
        assert len(model.parse_diagnostics) == 1
        assert model.requirements[0].name == "REQ_001"
        assert model.requirements[0].text == "The system shall operate safely"
        assert model.blocks[0].name == "FlightController"
        assert model.blocks[0].ports[0].name == "sensorIn"
        assert model.blocks[0].ports[0].direction == FeatureDirection.IN
        assert model.blocks[0].attributes[0].name == "loopRate"
        assert model.blocks[0].actions[0].name == "stabilize"
        assert model.blocks[0].satisfies == ["REQ_001"]
        assert model.blocks[0].refines == ["AbstractFlightController"]
        assert model.blocks[0].generalizations == ["AbstractController"]
        assert model.connectors[0].source_block_id == "Sensor"
        assert model.connectors[0].target_block_id == "FlightController"
        assert model.constraints == ["model constraint"]
        assert model.expressions == ["x + y"]
        assert model.metadata["ast_adapter"] == "python-json-ast-v1"

    def test_ast_to_model_can_merge_into_existing_model(self):
        adapter = SysMLAstAdapter()
        existing = SysMLModel(name="ControlSystem")
        existing.add_block(Block(name="ExistingBlock"))

        ast = {
            "schema_version": "1.0",
            "status": "ok",
            "ast": {"blocks": [{"name": "NewBlock"}]},
        }
        model = adapter.ast_to_model(ast, existing_model=existing)

        assert model is existing
        assert model.get_block_by_name("ExistingBlock") is not None
        assert model.get_block_by_name("NewBlock") is not None

    def test_ast_to_model_requires_finalized_envelope(self):
        adapter = SysMLAstAdapter()
        bad_ast = {"root": {"name": "LegacySchema"}}
        with pytest.raises(SysMLAstMappingError):
            adapter.ast_to_model(bad_ast)


class TestDesignAgentAstIntegration:
    def test_design_agent_can_consume_ast_payload(self):
        class FakeAdapter:
            def ast_to_model(self, ast_payload, existing_model=None):
                model = existing_model or SysMLModel(name="DemoSystem")
                model.add_package("pkg::demo")
                model.metadata["received_ast"] = ast_payload
                return model

        agent = DesignAgent(MockLLM(), ast_adapter=FakeAdapter())
        result = agent.run(
            {
                "system_name": "DemoSystem",
                "requirements": [],
                "sysml_ast": {
                    "schema_version": "1.0",
                    "status": "ok",
                    "ast": {"name": "DemoSystem"},
                },
            }
        )

        assert result.success
        assert result.output.metadata["received_ast"]["status"] == "ok"
        assert result.output.packages == ["pkg::demo"]

    def test_design_agent_raises_config_error_when_ast_client_missing(self):
        llm = MockLLM()
        llm.inject_response("```sysml\npackage DemoSystem {}\n```")
        agent = DesignAgent(llm)
        with pytest.raises(RuntimeError, match="AST_CONFIG_ERROR"):
            agent.run({"system_name": "DemoSystem", "requirements": ["REQ-001: test"]})

    def test_design_agent_raises_service_error(self):
        class _FailClient:
            def parse_text(self, *args, **kwargs):
                raise SysMLAstClientError("service unavailable")

        llm = MockLLM()
        llm.inject_response("```sysml\npackage DemoSystem {}\n```")
        agent = DesignAgent(llm, ast_client=_FailClient())
        with pytest.raises(RuntimeError, match="AST_SERVICE_ERROR"):
            agent.run({"system_name": "DemoSystem", "requirements": ["REQ-001: test"]})





