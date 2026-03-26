"""
Design Agent for MBSE prototyping.

Specializes in generating and refining SysML v2 design models
from requirements using Chain of Thought prompting and RAG.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface, Message
from ..rag.retriever import RAGRetriever
from ..sysml.model import (
    Action,
    Attribute,
    Block,
    Connector,
    FeatureDirection,
    Port,
    SysMLModel,
)


class DesignAgent(BaseAgent):
    """
    Agent responsible for architectural design generation.

    Capabilities:
    - Generate SysML v2 models from requirements
    - Refine designs based on evaluation feedback
    - Decompose system into components
    - Define interfaces between components
    - Ensure requirement satisfaction
    """

    SYSTEM_PROMPT = """You are an expert system architect specializing in Model Based
Systems Engineering (MBSE) and SysML v2. Generate detailed, well-structured SysML v2
models for cyber-physical systems.

Design principles to follow:
1. Modular: Separate concerns into distinct components
2. Hierarchical: Organize with clear parent-child relationships
3. Connected: Define explicit interfaces between components
4. Traceable: Link every design element to requirements
5. Verifiable: Include measurable attributes for all components

Always provide SysML v2 code in ```sysml blocks.
"""

    def __init__(
        self,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        super().__init__("DesignAgent", llm, rag_retriever)
        self.cot = ChainOfThoughtPrompter(llm)
        self.cot.system_prompt = self.SYSTEM_PROMPT

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Generate a SysML v2 design from requirements.

        Expected task keys:
        - system_name: str
        - requirements: List[str]
        - context: str (optional additional context)
        - existing_model: SysMLModel (optional, for refinement)
        - refinement_feedback: str (optional feedback for refinement)
        """
        system_name = task.get("system_name", "UnnamedSystem")
        requirements = task.get("requirements", [])
        context = task.get("context", "")
        existing_model = task.get("existing_model")
        feedback = task.get("refinement_feedback", "")

        # Augment context with RAG
        query = f"{system_name} {' '.join(requirements[:3])}"
        rag_context = self.get_augmented_context(query)
        if rag_context and context:
            context = f"{context}\n\n{rag_context}"
        elif rag_context:
            context = rag_context

        if existing_model and feedback:
            # Refinement mode
            cot_result = self.cot.refine_design(
                model_text=existing_model.to_sysml_text(),
                feedback=feedback,
                issues=[feedback],
            )
        else:
            # Generation mode
            cot_result = self.cot.generate_design(
                system_name=system_name,
                requirements=requirements,
                context=context,
            )

        # Build or update the SysML model
        model = existing_model or SysMLModel(
            name=system_name,
            description=f"Auto-generated design for {system_name}",
        )

        if cot_result.extracted_sysml:
            self._populate_model_from_sysml(model, cot_result.extracted_sysml)
        else:
            self._populate_model_heuristically(model, system_name, requirements)

        result = AgentResult(
            agent_name=self.name,
            success=True,
            output=model,
            reasoning=cot_result.final_answer,
            metadata={
                "sysml_extracted": cot_result.extracted_sysml is not None,
                "thought_steps": len(cot_result.thought_steps),
                "blocks_created": len(model.blocks),
            },
        )
        self.record_result(result)
        return result

    def _populate_model_from_sysml(
        self, model: SysMLModel, sysml_text: str
    ) -> None:
        """
        Parse SysML v2 text and populate the model.

        This is a simplified parser; a production system would use
        a full SysML v2 grammar parser.
        """
        import re

        # Extract part definitions
        part_def_pattern = r"part def\s+(\w+)\s*\{([^}]*)\}"
        for match in re.finditer(part_def_pattern, sysml_text, re.DOTALL):
            block_name = match.group(1)
            block_body = match.group(2)

            if model.get_block_by_name(block_name):
                continue  # Skip if already exists

            block = Block(name=block_name, block_type="part def")

            # Parse attributes
            attr_pattern = r"attribute\s+(\w+)\s*:\s*(\w+)(?:\s*=\s*([^;]+))?"
            for attr_match in re.finditer(attr_pattern, block_body):
                attr = Attribute(
                    name=attr_match.group(1),
                    attribute_type=attr_match.group(2),
                    default_value=attr_match.group(3).strip() if attr_match.group(3) else None,
                )
                block.add_attribute(attr)

            # Parse ports
            port_pattern = r"port\s+(\w+)\s*(?::\s*~?(\w+))?"
            for port_match in re.finditer(port_pattern, block_body):
                port = Port(
                    name=port_match.group(1),
                    port_type=port_match.group(2) or "",
                )
                block.add_port(port)

            model.add_block(block)

        # Extract connections (deduplicate)
        existing_conn_keys = {
            (c.source_block_id, c.source_port_id, c.target_block_id, c.target_port_id)
            for c in model.connectors
        }
        conn_pattern = r"connect\s+(\w+)\.(\w+)\s+to\s+(\w+)\.(\w+)"
        for conn_match in re.finditer(conn_pattern, sysml_text):
            key = (conn_match.group(1), conn_match.group(2), conn_match.group(3), conn_match.group(4))
            if key in existing_conn_keys:
                continue
            conn = Connector(
                name=f"conn_{conn_match.group(1)}_{conn_match.group(3)}",
                source_block_id=conn_match.group(1),
                source_port_id=conn_match.group(2),
                target_block_id=conn_match.group(3),
                target_port_id=conn_match.group(4),
            )
            model.add_connector(conn)
            existing_conn_keys.add(key)

    def _populate_model_heuristically(
        self,
        model: SysMLModel,
        system_name: str,
        requirements: List[str],
    ) -> None:
        """
        Create a basic model structure when SysML extraction fails.

        Uses heuristics to identify common CPS components from requirements.
        Skips adding components that already exist in the model.
        """
        # Common cyber-physical system components
        standard_components = [
            ("Controller", [
                Port(name="sensorIn", direction=FeatureDirection.IN),
                Port(name="commandOut", direction=FeatureDirection.OUT),
                Attribute(name="processingRate", attribute_type="Real", default_value="1000.0", unit="Hz"),
            ]),
            ("Sensor", [
                Port(name="dataOut", direction=FeatureDirection.OUT),
                Attribute(name="samplingRate", attribute_type="Real", default_value="100.0", unit="Hz"),
            ]),
            ("Actuator", [
                Port(name="commandIn", direction=FeatureDirection.IN),
                Attribute(name="responseTime", attribute_type="Real", default_value="10.0", unit="ms"),
            ]),
        ]

        for comp_name, elements in standard_components:
            if model.get_block_by_name(comp_name):
                continue  # Skip if already exists
            block = Block(name=comp_name, block_type="part def")
            for elem in elements:
                if isinstance(elem, Port):
                    block.add_port(elem)
                elif isinstance(elem, Attribute):
                    block.add_attribute(elem)
            model.add_block(block)

        # Add connection between sensor and controller (only if not already present)
        if len(model.blocks) >= 2:
            existing_conns = {
                (c.source_block_id, c.source_port_id, c.target_block_id, c.target_port_id)
                for c in model.connectors
            }
            key = ("Sensor", "dataOut", "Controller", "sensorIn")
            if key not in existing_conns:
                model.add_connector(Connector(
                    name="sensorToController",
                    source_block_id="Sensor",
                    source_port_id="dataOut",
                    target_block_id="Controller",
                    target_port_id="sensorIn",
                ))
