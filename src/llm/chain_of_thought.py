"""
Chain of Thought (CoT) prompting module.

Implements structured Chain of Thought prompting techniques for guiding
LLMs through complex MBSE design reasoning tasks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interface import LLMInterface, Message


SYSML_EXPERT_SYSTEM_PROMPT = """You are an expert in Model Based Systems Engineering (MBSE) 
and SysML v2. Your role is to help design cyber-physical systems using SysML v2 notation.

When designing systems:
1. Always think step-by-step, showing your reasoning
2. Follow SysML v2 syntax precisely
3. Consider both functional and non-functional requirements
4. Ensure traceability between requirements and design elements
5. Design for reliability, safety, and maintainability

SysML v2 key constructs:
- `package`: top-level namespace
- `part def`: defines a block/component type
- `port`: connection point with optional direction (in/out/inout)
- `attribute`: value property with type and unit
- `requirement`: captures stakeholder needs
- `action`: behavior specification
- `connect`: links ports between parts
- `satisfy`: links design elements to requirements
"""

REQUIREMENTS_COT_TEMPLATE = """Analyze the following system description and extract structured requirements.

System Description: {description}

Think through this step by step:
1. What are the PRIMARY functional requirements (what the system MUST do)?
2. What are the NON-FUNCTIONAL requirements (performance, safety, reliability)?
3. What are the INTERFACE requirements (inputs/outputs, protocols)?
4. What CONSTRAINTS exist (physical, regulatory, resource)?

For each requirement, use this format:
- REQ-XXX: [Type] Description of the requirement

After listing requirements, identify dependencies between them.
"""

DESIGN_COT_TEMPLATE = """Design a SysML v2 model for the following system, satisfying these requirements.

System: {system_name}
Requirements:
{requirements}

Additional context: {context}

Think through the design step by step:
1. DECOMPOSITION: What are the main subsystems/components?
2. INTERFACES: What ports and data flows connect the components?
3. ATTRIBUTES: What key parameters define each component?
4. BEHAVIOR: What actions does each component perform?
5. TRACEABILITY: Which components satisfy which requirements?

Provide the complete SysML v2 model in a code block marked with ```sysml
"""

EVALUATION_COT_TEMPLATE = """Evaluate how well this SysML v2 model satisfies the requirements.

Model:
{model}

Requirements:
{requirements}

Evaluate step by step:
1. COMPLETENESS: Does the design address all requirements?
2. CONSISTENCY: Are there contradictions or gaps?
3. PERFORMANCE: Will the design meet performance requirements?
4. SAFETY: Are safety requirements satisfied?
5. OVERALL SCORE: Provide a score from 0.0 to 1.0

Provide scores in JSON format at the end:
```json
{{"completeness": 0.0, "consistency": 0.0, "performance": 0.0, "safety": 0.0, "overall": 0.0}}
```
"""

REFINEMENT_COT_TEMPLATE = """Refine the following SysML v2 model based on evaluation feedback.

Current Model:
{model}

Evaluation Feedback:
{feedback}

Specific issues to address: {issues}

Think through the refinements step by step:
1. What specific changes are needed to address each issue?
2. How do the changes affect other parts of the design?
3. What new elements need to be added?
4. What existing elements need to be modified?

Provide the refined SysML v2 model in a code block marked with ```sysml
"""


@dataclass
class ThoughtStep:
    """A single step in a Chain of Thought reasoning process."""
    step_number: int
    description: str
    content: str
    confidence: float = 1.0


@dataclass
class CoTResult:
    """Result of a Chain of Thought reasoning process."""
    final_answer: str
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_json: Optional[Dict[str, Any]] = None
    extracted_sysml: Optional[str] = None

    def get_scores(self) -> Optional[Dict[str, float]]:
        """Extract evaluation scores from the response."""
        if self.extracted_json:
            return self.extracted_json
        return None


class ChainOfThoughtPrompter:
    """
    Implements Chain of Thought prompting for MBSE design tasks.

    Supports multiple CoT patterns:
    - Zero-shot CoT: "Think step by step"
    - Few-shot CoT: Provide examples with reasoning
    - Tree of Thought: Explore multiple reasoning paths
    - Self-consistency: Generate multiple solutions and vote
    """

    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.system_prompt = SYSML_EXPERT_SYSTEM_PROMPT

    def extract_requirements(
        self,
        system_description: str,
        context: str = "",
    ) -> CoTResult:
        """
        Use CoT prompting to extract structured requirements from a description.
        """
        prompt = REQUIREMENTS_COT_TEMPLATE.format(
            description=system_description + (f"\n\nAdditional context: {context}" if context else "")
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.3)
        return self._parse_cot_response(response.content)

    def generate_design(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        temperature: float = 0.5,
    ) -> CoTResult:
        """
        Use CoT prompting to generate a SysML v2 design.
        """
        req_text = "\n".join(f"  - {r}" for r in requirements)
        prompt = DESIGN_COT_TEMPLATE.format(
            system_name=system_name,
            requirements=req_text,
            context=context or "None",
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=temperature)
        return self._parse_cot_response(response.content)

    def evaluate_design(
        self,
        model_text: str,
        requirements: List[str],
    ) -> CoTResult:
        """
        Use CoT prompting to evaluate a SysML v2 model against requirements.
        """
        req_text = "\n".join(f"  - {r}" for r in requirements)
        prompt = EVALUATION_COT_TEMPLATE.format(
            model=model_text,
            requirements=req_text,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.2)
        return self._parse_cot_response(response.content)

    def refine_design(
        self,
        model_text: str,
        feedback: str,
        issues: List[str],
    ) -> CoTResult:
        """
        Use CoT prompting to refine a design based on evaluation feedback.
        """
        issues_text = "\n".join(f"  - {i}" for i in issues)
        prompt = REFINEMENT_COT_TEMPLATE.format(
            model=model_text,
            feedback=feedback,
            issues=issues_text,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.4)
        return self._parse_cot_response(response.content)

    def self_consistency_generate(
        self,
        system_name: str,
        requirements: List[str],
        num_samples: int = 3,
    ) -> CoTResult:
        """
        Generate multiple design candidates and return the most consistent one.

        Implements the self-consistency CoT technique for more reliable results.
        """
        candidates = []
        for _ in range(num_samples):
            result = self.generate_design(
                system_name=system_name,
                requirements=requirements,
                temperature=0.7,
            )
            candidates.append(result)

        # Select the candidate with the most SysML content (heuristic for completeness)
        best = max(
            candidates,
            key=lambda r: len(r.extracted_sysml or ""),
        )
        best.metadata["num_candidates"] = num_samples
        best.metadata["consistency_method"] = "max_sysml_length"
        return best

    def _parse_cot_response(self, response_text: str) -> CoTResult:
        """Parse an LLM response to extract CoT steps, SysML, and JSON."""
        result = CoTResult(final_answer=response_text)

        # Extract SysML code blocks
        sysml_pattern = r"```sysml\n(.*?)```"
        sysml_matches = re.findall(sysml_pattern, response_text, re.DOTALL)
        if sysml_matches:
            result.extracted_sysml = sysml_matches[-1].strip()

        # Extract JSON code blocks
        json_pattern = r"```json\n(.*?)```"
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            try:
                result.extracted_json = json.loads(json_matches[-1].strip())
            except json.JSONDecodeError:
                pass

        # Extract numbered steps
        step_pattern = r"(\d+)\.\s+([A-Z][^:]+):\s*(.*?)(?=\n\d+\.|$)"
        step_matches = re.findall(step_pattern, response_text, re.DOTALL)
        for num, description, content in step_matches:
            result.thought_steps.append(
                ThoughtStep(
                    step_number=int(num),
                    description=description.strip(),
                    content=content.strip(),
                )
            )

        return result
