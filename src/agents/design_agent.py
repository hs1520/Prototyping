"""
Design Agent for MBSE prototyping.

Specializes in generating and refining SysML v2 design models
from requirements using Chain of Thought prompting and RAG.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import (
    PartDefinition,
    SatisfyRelationship,
    ElementRef,
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
        - refinement_issues: List[str] (optional structured issues for refinement)
        """
        system_name = task.get("system_name", "UnnamedSystem")
        requirements = task.get("requirements", [])
        context = task.get("context", "")
        existing_model = task.get("existing_model")
        feedback = task.get("refinement_feedback", "")
        refinement_issues = task.get("refinement_issues", [])

        # Augment context with RAG
        # TODO query需要改造 3条不科学
        query = f"{system_name} {' '.join(requirements[:3])}"
        rag_context = self.get_augmented_context(
            query,
            include_official_sysml=True,
            allowed_extensions=(".sysml",),
        )
        if rag_context and context:
            context = f"{context}\n\n{rag_context}"
        elif rag_context:
            context = rag_context

        if existing_model and feedback:
            # Refinement mode
            cot_result = self.cot.refine_design(
                model_text=existing_model.to_sysml_text(),
                feedback=feedback,
                issues=refinement_issues or [feedback],
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

        if not cot_result.extracted_sysml:
            raise RuntimeError("[SysML_EXTRACTION_ERROR] 未提取到SysML v2 design.")

        # TODO: 未来集成Syside AST Parser来解析和映射SysML文本到model
        # from ..sysml.Syside_AST_Parser import parse_sysml_to_model
        # model = parse_sysml_to_model(cot_result.extracted_sysml, model_name=system_name)

        self._apply_requirement_traceability(model, requirements)

        result = AgentResult(
            agent_name=self.name,
            success=True,
            output=model,
            reasoning=cot_result.final_answer,
            metadata={
                "sysml_extracted": cot_result.extracted_sysml is not None,
                "thought_steps": len(cot_result.thought_steps),
                "part_definitions_created": len(model.part_definitions),
            },
        )
        self.record_result(result)
        return result

    def _apply_requirement_traceability(self, model: SysMLModel, requirements: List[str]) -> None:
        """Ensure each requirement is allocated to at least one design component when possible."""
        if not requirements or not model.part_definitions:
            return

        controller_like = next(
            (part for part in model.part_definitions if "controller" in part.name.lower()),
            None,
        )

        def score_part(req_text: str, part: PartDefinition) -> int:
            req_tokens = set(re.findall(r"[A-Za-z0-9_]+", req_text.lower()))
            part_tokens = set(re.findall(r"[A-Za-z0-9_]+", part.name.lower()))
            part_tokens.update(token.lower() for attr in part.attributes for token in [attr.name])
            return len(req_tokens & part_tokens)

        for req_text in requirements:
            req_id_match = re.match(r"(REQ-\w+-\d+|REQ-\d+):\s*(.*)", req_text)
            req_id = req_id_match.group(1).replace("-", "_") if req_id_match else req_text.split(":", 1)[0].replace("-", "_")

            # Check if requirement is already satisfied
            if any(req_id in str(part.satisfy_relationships) for part in model.part_definitions):
                continue

            best_part = max(
                model.part_definitions,
                key=lambda part: score_part(req_text, part),
                default=None
            )
            chosen_part = best_part if best_part and score_part(req_text, best_part) > 0 else controller_like or model.part_definitions[0]
            chosen_part.add_satisfy(SatisfyRelationship(source=chosen_part.to_ref(), target=ElementRef(name=req_id)))
