"""Requirements Agent for MBSE prototyping."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import RequirementDefinition, SysMLModel
from ..sysml.lite_model import SysMLLiteModel


_VALID_CATEGORIES = {"FUNC", "PERF", "SAFE", "INTF", "CONS", "OPER"}
_REQ_ID_RE = re.compile(r"^(REQ-([A-Z]+)-(\d+)):\s*(.+)$")
_MIN_REQ_LENGTH = 20


class RequirementsAgent(BaseAgent):
    """Agent responsible for requirements engineering tasks."""

    SYSTEM_PROMPT = """You are a senior requirements engineer specializing in cyber-physical systems.
Your task is to extract COMPLETE, ATOMIC, and VERIFIABLE requirements from a system description.

Core rules:
1. Requirements describe WHAT the system must accomplish, never HOW it should be implemented.
   - WRONG: "The system shall use a Kalman filter for localization."
   - RIGHT:  "The system shall maintain position accuracy within 1 m of the target coordinate."
2. Each requirement covers exactly ONE capability or constraint (atomic).
3. Every requirement must be independently verifiable — include a number, unit, threshold, or
   explicit boolean condition.
4. Use INCOSE format: "The [system] shall [action] [object] [condition]."
5. Classify using exactly these category codes:
   FUNC  — core functional capability (what the system does)
   PERF  — quantitative performance (speed, accuracy, capacity, endurance)
   SAFE  — fail-safe, fault-tolerance, and hazard-mitigation behavior
   INTF  — external connections: protocols, signals, data formats, standards
   CONS  — physical, regulatory, cost, or resource constraints
   OPER  — named sequential operational phases (mode machine); generate at most ONE
           REQ-OPER requirement only when the system has 3+ distinct named phases
6. Format each line as: REQ-<CATEGORY>-<NNN>: The system shall ...
   NNN is a zero-padded 3-digit number, restarting from 001 within each category.
7. SAFETY SEVERITY (SAFE requirements only): append a failure-condition severity
   tag classifying the WORST credible consequence if the requirement is not met,
   per DO-178C / ARP4754A:
     [SEV:Catastrophic] — loss of vehicle, fatalities, uncontrolled crash
     [SEV:Hazardous]    — severe injury, large damage, near-total loss of control
     [SEV:Major]        — significant degradation, recoverable damage, reduced safety margin
     [SEV:Minor]        — slight degradation, nuisance, minimal safety impact
     [SEV:No-effect]    — no safety consequence
   Example: REQ-SAFE-001: The system shall execute an autoland on dual-engine failure. [SEV:Catastrophic]
   Classify by SEVERITY OF CONSEQUENCE, not by likelihood. Only SAFE requirements get a tag.
"""

    def __init__(
        self,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        super().__init__("RequirementsAgent", llm, rag_retriever)
        self.cot = ChainOfThoughtPrompter(llm)

    _MIN_DESCRIPTION_LENGTH = 50

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Extract and structure requirements from a system description."""
        description = task.get("system_description", "").strip()
        system_name = task.get("system_name", "system").strip() or "system"
        existing_requirements: List[str] = task.get("existing_requirements", [])

        if not description:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output=[],
                reasoning="No system description provided.",
            )
        if len(description) < self._MIN_DESCRIPTION_LENGTH:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output=[],
                reasoning=(
                    f"System description too short ({len(description)} chars). "
                    f"Provide at least {self._MIN_DESCRIPTION_LENGTH} characters."
                ),
            )

        # Feed manual requirements as fixed anchors so the LLM returns one unified set;
        # two independent passes would both start numbering at 001 and collide.
        cot_result = self.cot.extract_requirements(
            description,
            system_name=system_name,
            fixed_requirements=existing_requirements or None,
            system_prompt=self.SYSTEM_PROMPT,
        )

        requirements = self._parse_requirements(cot_result.final_answer)

        # Fixed requirements own their IDs and wording. The LLM may copy one and append
        # a severity tag, which a text-only check reads as different and reinserts as a
        # duplicate ID. Merge first with manual requirements winning, then verify every
        # anchor survived verbatim.
        conflict_warnings: List[str] = []
        if existing_requirements:
            requirements, conflict_warnings = self.merge_requirements(
                requirements, existing_requirements
            )
            requirements = self._verify_fixed_requirements(
                requirements, existing_requirements
            )
        dependencies = self._parse_dependencies(cot_result.final_answer)
        counts = self._count_by_category(requirements)

        result = AgentResult(
            agent_name=self.name,
            success=len(requirements) > 0,
            output=requirements,
            reasoning=cot_result.final_answer,
            metadata={
                "num_requirements": len(requirements),
                "thought_steps": len(cot_result.thought_steps),
                "counts_by_category": counts,
                "dependencies": dependencies,
                "requirement_conflicts": conflict_warnings,
                "requirement_semantic_analysis": None,
            },
        )
        self.record_result(result)
        return result

    def validate_requirements(
        self,
        requirements: List[str],
    ) -> Dict[str, Any]:
        """Validate a list of requirements for format correctness, completeness, and design-readiness."""
        issues: List[str] = []
        warnings: List[str] = []
        seen_ids: Dict[str, int] = {}
        counts_by_category: Dict[str, int] = {cat: 0 for cat in _VALID_CATEGORIES}
        counts_by_category["UNKNOWN"] = 0

        _ambiguous = [
            "as fast as possible", "user-friendly", "robust", "efficient",
            "easy to use", "high quality", "adequate", "appropriate",
        ]
        _measurable_boolean_keywords = {"detect", "activate", "initiate", "trigger", "enable", "disable"}

        for req in requirements:
            snippet = req[:60] + ("..." if len(req) > 60 else "")

            if len(req.strip()) < _MIN_REQ_LENGTH:
                issues.append(f"Requirement too short (< {_MIN_REQ_LENGTH} chars): '{snippet}'")
                continue

            m = _REQ_ID_RE.match(req.strip())
            if not m:
                issues.append(f"Invalid REQ ID format (expected REQ-<CATEGORY>-<NNN>): '{snippet}'")
            else:
                req_id, category = m.group(1), m.group(2)

                if category not in _VALID_CATEGORIES:
                    issues.append(
                        f"Unknown category '{category}' in '{req_id}' "
                        f"(valid: {', '.join(sorted(_VALID_CATEGORIES))})"
                    )
                    counts_by_category["UNKNOWN"] += 1
                else:
                    counts_by_category[category] += 1

                if req_id in seen_ids:
                    issues.append(f"Duplicate REQ ID '{req_id}' (first at position {seen_ids[req_id]})")
                else:
                    seen_ids[req_id] = len(seen_ids)

            if "shall" not in req.lower():
                issues.append(f"Missing 'shall': '{snippet}'")
                continue

            if req.lower().count("shall") > 1:
                warnings.append(f"Compound requirement (multiple 'shall') — split into atomic reqs: '{snippet}'")

            # Measurability: check text after the first colon so REQ ID digits do not count
            text_part = req.split(":", 1)[1] if ":" in req else req
            text_lower = text_part.lower()
            has_digit = any(ch.isdigit() for ch in text_part)
            has_boolean_action = any(kw in text_lower for kw in _measurable_boolean_keywords)
            if not has_digit and not has_boolean_action:
                warnings.append(f"No measurable criterion (number, unit, or threshold): '{snippet}'")

            for term in _ambiguous:
                if term in text_lower:
                    warnings.append(f"Ambiguous term '{term}': '{snippet}'")

        if counts_by_category.get("FUNC", 0) == 0:
            warnings.append(
                "No FUNCTIONAL requirements found — the design will have no behavioral "
                "specification to implement."
            )
        if counts_by_category.get("SAFE", 0) == 0:
            warnings.append(
                "No SAFETY requirements found — consider fail-safe and fault-tolerance "
                "behaviors for this cyber-physical system."
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_requirements": len(requirements),
            "counts_by_category": counts_by_category,
            "requirement_semantic_analysis": None,
        }

    def merge_requirements(
        self,
        llm_requirements: List[str],
        manual_requirements: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Merge LLM-extracted and manual requirements, eliminating duplicates."""
        def _parse_id(req: str) -> Optional[str]:
            m = _REQ_ID_RE.match(req.strip())
            return m.group(1) if m else None

        def _normalize(req: str) -> str:
            m = _REQ_ID_RE.match(req.strip())
            text = m.group(4) if m else req
            # Severity is metadata, not identity: the extraction LLM may append it to an
            # anchor that was required verbatim.
            text = re.sub(r"\s*\[SEV\s*:[^\]]+\]\s*$", "", text,
                          flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", text.lower().strip())

        def _next_id(category: str, used: set) -> str:
            n = 1
            while True:
                candidate = f"REQ-{category}-{n:03d}"
                if candidate not in used:
                    return candidate
                n += 1

        all_used_ids: set = {_parse_id(r) for r in manual_requirements if _parse_id(r)}
        manual_ids = set(all_used_ids)
        seen_texts = {_normalize(r) for r in manual_requirements}

        merged: List[str] = list(manual_requirements)
        conflict_warnings: List[str] = []

        for req in llm_requirements:
            req_id = _parse_id(req)
            norm = _normalize(req)

            if norm in seen_texts:
                continue

            if req_id and req_id in all_used_ids:
                cat_match = re.match(r"REQ-([A-Z]+)-\d+", req_id)
                if cat_match:
                    new_id = _next_id(cat_match.group(1), all_used_ids)
                    new_req = req.replace(req_id + ":", new_id + ":", 1)
                    if req_id in manual_ids:
                        conflict_warnings.append(
                            f"REQ ID conflict: '{req_id}' already exists in manual requirements "
                            f"with different content. LLM requirement reassigned to '{new_id}'."
                        )
                    merged.append(new_req)
                    all_used_ids.add(new_id)
                    seen_texts.add(norm)
                continue

            merged.append(req)
            if req_id:
                all_used_ids.add(req_id)
            seen_texts.add(norm)

        return merged, conflict_warnings

    def create_sysml_requirements(
        self, requirements: List[str], model: SysMLModel
    ) -> None:
        """Add extracted requirements to a SysML model, skipping already-present IDs."""
        if isinstance(model, SysMLLiteModel):
            # The Lite model serializes its source text, not this cache, so an in-memory
            # addition would hide a missing requirement from evaluation while the final
            # SysML stays unchanged. The generation/refinement path emits the real block.
            return
        existing_ids = {r.name for r in model.requirement_definitions}
        for i, req_text in enumerate(requirements):
            id_match = re.match(r"(REQ-\w+-\d+|REQ-\d+):\s*(.*)", req_text)
            if id_match:
                req_id = id_match.group(1).replace("-", "_")
                req_content = id_match.group(2).strip()
            else:
                req_id = f"REQ_{i+1:03d}"
                req_content = req_text.strip()

            if req_id in existing_ids:
                continue
            existing_ids.add(req_id)

            req = RequirementDefinition(
                name=req_id,
                text=req_content,
                short_description=req_content[:80],
            )
            model.add_requirement_definition(req)

    @staticmethod
    def _verify_fixed_requirements(
        unified: List[str],
        fixed: List[str],
    ) -> List[str]:
        def _norm(r: str) -> str:
            # Strip the ID prefix so ID reassignment does not hide content-identical
            # requirements.
            body = r.split(":", 1)[-1] if ":" in r else r
            return re.sub(r"\s+", " ", body.lower().strip())

        unified_norms = {_norm(r) for r in unified}
        result = list(unified)
        for req in fixed:
            if _norm(req) not in unified_norms:
                result.append(req)
                unified_norms.add(_norm(req))
        return result

    def _parse_requirements(self, text: str) -> List[str]:
        requirements: List[str] = []
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if re.match(r"REQ-\w+-\d+:", line) or re.match(r"REQ-\d+:", line):
                requirements.append(line)
            elif line.startswith("-") and "shall" in line.lower():
                requirements.append(line.lstrip("- "))
            elif line.startswith("*") and "shall" in line.lower():
                requirements.append(line.lstrip("* "))

        return requirements if requirements else self._extract_shall_sentences(text)

    def _extract_shall_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]", text)
        return [
            s.strip() for s in sentences
            if "shall" in s.lower() and len(s.strip()) > 10
        ]

    def _parse_dependencies(self, text: str) -> List[Dict[str, str]]:
        dep_marker = re.search(r"(?i)\bdependencies?\b\s*:", text)
        search_text = text[dep_marker.start():] if dep_marker else ""
        if not search_text:
            return []
        pattern = re.compile(
            r"\b(REQ-[A-Z]+-\d+)\b\s+depends\s+on\s+\b(REQ-[A-Z]+-\d+)\b",
            re.IGNORECASE,
        )
        return [
            {"from": m.group(1).upper(), "to": m.group(2).upper()}
            for m in pattern.finditer(search_text)
        ]

    def _count_by_category(self, requirements: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {cat: 0 for cat in _VALID_CATEGORIES}
        counts["UNKNOWN"] = 0
        for req in requirements:
            m = _REQ_ID_RE.match(req.strip())
            if m:
                cat = m.group(2)
                counts[cat if cat in _VALID_CATEGORIES else "UNKNOWN"] += 1
        return counts
