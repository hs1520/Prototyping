"""
Requirements Agent for MBSE prototyping.

Specializes in extracting, structuring, and validating requirements
from natural language system descriptions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import Requirement, SysMLModel


class RequirementsAgent(BaseAgent):
    """
    Agent responsible for requirements engineering tasks.

    Capabilities:
    - Extract structured requirements from natural language
    - Validate requirement completeness and consistency
    - Identify requirement dependencies
    - Classify requirements (functional, non-functional, interface, constraint)
    """

    SYSTEM_PROMPT = """You are a requirements engineering expert specializing in
cyber-physical systems. Extract clear, measurable, and testable requirements
using the INCOSE format: "The [system] shall [action] [object] [condition]".

Classify each requirement as:
- FUNC: Functional requirement (what the system does)
- PERF: Performance requirement (how well it does it)
- SAFE: Safety requirement (fail-safe behavior)
- INTF: Interface requirement (connections to other systems)
- CONS: Constraint (limitations, standards, regulations)

Use the format: REQ-{category}-{number}: [text]
"""

    def __init__(
        self,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        super().__init__("RequirementsAgent", llm, rag_retriever)
        self.cot = ChainOfThoughtPrompter(llm)
        self.cot.system_prompt = self.SYSTEM_PROMPT

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Extract and structure requirements from a system description.

        Expected task keys:
        - system_description: str
        - system_name: str (optional)
        - existing_requirements: List[str] (optional)
        """
        description = task.get("system_description", "")
        if not description:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output=[],
                reasoning="No system description provided",
            )

        # Augment with RAG context
        context = self.get_augmented_context(description)

        # Use CoT to extract requirements
        cot_result = self.cot.extract_requirements(description, context)

        # Parse requirements from CoT response
        requirements = self._parse_requirements(cot_result.final_answer)

        result = AgentResult(
            agent_name=self.name,
            success=True,
            output=requirements,
            reasoning=cot_result.final_answer,
            metadata={
                "num_requirements": len(requirements),
                "thought_steps": len(cot_result.thought_steps),
            },
        )
        self.record_result(result)
        return result

    def validate_requirements(self, requirements: List[str]) -> Dict[str, Any]:
        """
        Validate a list of requirements for completeness and consistency.

        Returns a validation report.
        """
        issues: List[str] = []
        warnings: List[str] = []

        for req in requirements:
            # Check for measurability
            if not any(char.isdigit() for char in req):
                warnings.append(f"Requirement may not be measurable: '{req[:50]}...'")

            # Check for "shall" keyword
            if "shall" not in req.lower():
                issues.append(f"Requirement missing 'shall': '{req[:50]}...'")

            # Check for ambiguous terms
            ambiguous = ["as fast as possible", "user-friendly", "robust", "efficient"]
            for term in ambiguous:
                if term in req.lower():
                    warnings.append(
                        f"Ambiguous term '{term}' in requirement: '{req[:50]}...'"
                    )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_requirements": len(requirements),
        }

    def create_sysml_requirements(
        self, requirements: List[str], model: SysMLModel
    ) -> None:
        """Add extracted requirements to a SysML model."""
        for i, req_text in enumerate(requirements):
            # Extract ID if present
            id_match = re.match(r"(REQ-\w+-\d+|REQ-\d+):\s*(.*)", req_text)
            if id_match:
                req_id = id_match.group(1).replace("-", "_")
                req_content = id_match.group(2).strip()
            else:
                req_id = f"REQ_{i+1:03d}"
                req_content = req_text.strip()

            req = Requirement(
                name=req_id,
                text=req_content,
                short_description=req_content[:80],
            )
            model.add_requirement(req)

    def _parse_requirements(self, text: str) -> List[str]:
        """Parse requirements from LLM response text."""
        requirements: List[str] = []
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            # Match lines starting with REQ- or bullet points with "shall"
            if re.match(r"REQ-\w+-\d+:", line) or re.match(r"REQ-\d+:", line):
                requirements.append(line)
            elif line.startswith("-") and "shall" in line.lower():
                requirements.append(line.lstrip("- "))
            elif line.startswith("*") and "shall" in line.lower():
                requirements.append(line.lstrip("* "))

        return requirements if requirements else self._extract_shall_sentences(text)

    def _extract_shall_sentences(self, text: str) -> List[str]:
        """Extract sentences containing 'shall' as fallback."""
        sentences = re.split(r"[.!?]", text)
        return [
            s.strip() for s in sentences
            if "shall" in s.lower() and len(s.strip()) > 10
        ]
