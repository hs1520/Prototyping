"""
Design Agent for MBSE prototyping.

Specializes in generating and refining SysML v2 design models
from requirements using Chain of Thought prompting and RAG.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base_agent import AgentResult, BaseAgent
from ..config import Config
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.ast_adapter import SysMLAstAdapter, SysMLAstMappingError
from ..sysml.ast_client import SysMLAstClient, SysMLAstClientError
from ..sysml.model import (
    Block,
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
        ast_client: Optional[SysMLAstClient] = None,
        ast_adapter: Optional[SysMLAstAdapter] = None,
    ):
        super().__init__("DesignAgent", llm, rag_retriever)
        self.cot = ChainOfThoughtPrompter(llm)
        self.cot.system_prompt = self.SYSTEM_PROMPT
        self.ast_adapter = ast_adapter or SysMLAstAdapter()
        self.ast_client = ast_client
        if self.ast_client is None and Config.SYSML_AST_SERVICE_URL.strip():
            try:
                self.ast_client = SysMLAstClient()
            except SysMLAstClientError:
                self.ast_client = None

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
        ast_payload = task.get("sysml_ast")
        source_uri = task.get("source_uri", "")

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

        if ast_payload is None:
            if not cot_result.extracted_sysml:
                raise RuntimeError("[AST_INPUT_ERROR] 未提取到SysML v2 design.")
            if self.ast_client is None:
                raise RuntimeError(
                    "[AST_CONFIG_ERROR] DesignAgent runs in AST-only mode but no AST client is configured. "
                    "Set SYSML_AST_SERVICE_URL or inject ast_client."
                )
            try:
                ast_payload = self.ast_client.parse_text(
                    cot_result.extracted_sysml,
                    source_uri=source_uri,
                )
            except SysMLAstClientError as exc:
                raise RuntimeError(f"[AST_SERVICE_ERROR] SysML AST service failed: {exc}") from exc

        if ast_payload is None:
            raise RuntimeError("[AST_INPUT_ERROR] 未提取到SysML v2 design.")
        model = self._populate_model_from_ast(model, ast_payload)

        self._apply_requirement_traceability(model, requirements)

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

    def _populate_model_from_ast(
        self,
        model: SysMLModel,
        ast_payload: Dict[str, Any],
    ) -> SysMLModel:
        """Populate the model from a JSON AST payload."""
        try:
            mapped = self.ast_adapter.ast_to_model(ast_payload, existing_model=model)
        except SysMLAstMappingError as exc:
            raise RuntimeError(f"[AST_MAPPING_ERROR] SysML AST mapping failed: {exc}") from exc
        mapped.add_mapping_note("Populated via AST adapter")
        return mapped

    # AST-only mode: keep method name for compatibility with older call sites.
    def _populate_model_from_sysml(
        self, model: SysMLModel, sysml_text: str
    ) -> None:
        raise RuntimeError(
            "[AST_LEGACY_REMOVED] Regex parser path has been removed. "
            "Use AST payloads via sysml_ast or ast_client."
        )

    #TODO 存在问题 建议用Embedding做 同时当前无置信度/不可解释/没有一对多分配
    def _apply_requirement_traceability(self, model: SysMLModel, requirements: List[str]) -> None:
        """Ensure each requirement is allocated to at least one design block when possible."""
        if not requirements or not model.blocks:
            return

        controller_like = next(
            (block for block in model.blocks if "controller" in block.name.lower()),
            None,
        )

        def score_block(req_text: str, block: Block) -> int:
            req_tokens = set(re.findall(r"[A-Za-z0-9_]+", req_text.lower()))
            block_tokens = set(re.findall(r"[A-Za-z0-9_]+", block.name.lower()))
            block_tokens.update(token.lower() for port in block.ports for token in [port.name])
            block_tokens.update(token.lower() for attr in block.attributes for token in [attr.name])
            return len(req_tokens & block_tokens)

        for req_text in requirements:
            req_id_match = re.match(r"(REQ-\w+-\d+|REQ-\d+):\s*(.*)", req_text)
            req_id = req_id_match.group(1).replace("-", "_") if req_id_match else req_text.split(":", 1)[0].replace("-", "_")
            if any(req_id in block.satisfies for block in model.blocks):
                continue

            best_block = max(model.blocks, key=lambda block: score_block(req_text, block), default=None)
            chosen_block = best_block if best_block and score_block(req_text, best_block) > 0 else controller_like or model.blocks[0]
            chosen_block.add_satisfies(req_id)

    def _populate_model_heuristically(
        self,
        model: SysMLModel,
        system_name: str,
        requirements: List[str],
    ) -> None:
        raise RuntimeError("[AST_LEGACY_REMOVED] Heuristic parser fallback has been removed in AST-only mode.")

