"""
Requirements Agent for MBSE prototyping.

Specializes in extracting, structuring, and validating requirements
from natural language system descriptions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import RequirementDefinition, SysMLModel


_VALID_CATEGORIES = {"FUNC", "PERF", "SAFE", "INTF", "CONS", "OPER"}
_REQ_ID_RE = re.compile(r"^(REQ-([A-Z]+)-(\d+)):\s*(.+)$")
_MIN_REQ_LENGTH = 20


class RequirementsAgent(BaseAgent):
    """
    Agent responsible for requirements engineering tasks.

    Capabilities:
    - Extract structured requirements from natural language
    - Validate requirement completeness and consistency
    - Identify requirement dependencies
    - Classify requirements (functional, non-functional, interface, constraint)
    """

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
"""

    def __init__(
        self,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        super().__init__("RequirementsAgent", llm, rag_retriever)
        self.cot = ChainOfThoughtPrompter(llm)
        self.cot.system_prompt = self.SYSTEM_PROMPT

    _MIN_DESCRIPTION_LENGTH = 50

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Extract and structure requirements from a system description.

        Expected task keys:
        - system_description: str
        - system_name: str (optional) — used verbatim in "The <system_name> shall ..." format
        - existing_requirements: List[str] (optional) — already captured reqs; LLM avoids duplicating
        """
        description = task.get("system_description", "").strip()
        system_name = task.get("system_name", "system").strip() or "system"
        existing_requirements: List[str] = task.get("existing_requirements", [])

        # Pre-check: description quality
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

        # Build optional context from existing requirements so LLM avoids duplicating them
        context = ""
        if existing_requirements:
            lines = "\n".join(f"  {r}" for r in existing_requirements[:30])
            context = f"Already captured requirements (do not duplicate or contradict):\n{lines}"

        # CoT extraction
        cot_result = self.cot.extract_requirements(
            description,
            system_name=system_name,
            context=context,
        )

        requirements = self._parse_requirements(cot_result.final_answer)
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
            },
        )
        self.record_result(result)
        return result

    def validate_requirements(self, requirements: List[str]) -> Dict[str, Any]:
        """
        Validate a list of requirements for format correctness, completeness, and design-readiness.

        Checks (issues = blocking, warnings = advisory):
        - REQ ID format: must match REQ-<CATEGORY>-<NNN>
        - Category validity: must be one of FUNC | PERF | SAFE | INTF | CONS | OPER
        - Duplicate IDs within the list
        - Presence of "shall"
        - At least one measurable criterion (digit or boolean keyword)
        - Compound requirements (multiple "shall" in one line)
        - Ambiguous qualitative terms
        - Minimum text length

        Returns a dict usable by downstream phases:
          valid, issues, warnings, total_requirements, counts_by_category
        """
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

            # Minimum length
            if len(req.strip()) < _MIN_REQ_LENGTH:
                issues.append(f"Requirement too short (< {_MIN_REQ_LENGTH} chars): '{snippet}'")
                continue

            # REQ ID format
            m = _REQ_ID_RE.match(req.strip())
            if not m:
                issues.append(f"Invalid REQ ID format (expected REQ-<CATEGORY>-<NNN>): '{snippet}'")
            else:
                req_id, category, _num, _text = m.group(1), m.group(2), m.group(3), m.group(4)

                # Category validity
                if category not in _VALID_CATEGORIES:
                    issues.append(
                        f"Unknown category '{category}' in '{req_id}' "
                        f"(valid: {', '.join(sorted(_VALID_CATEGORIES))})"
                    )
                    counts_by_category["UNKNOWN"] += 1
                else:
                    counts_by_category[category] += 1

                # Duplicate ID
                if req_id in seen_ids:
                    issues.append(f"Duplicate REQ ID '{req_id}' (first at position {seen_ids[req_id]})")
                else:
                    seen_ids[req_id] = len(seen_ids)

            # "shall" presence
            if "shall" not in req.lower():
                issues.append(f"Missing 'shall': '{snippet}'")
                continue

            # Compound requirement (multiple "shall")
            if req.lower().count("shall") > 1:
                warnings.append(f"Compound requirement (multiple 'shall') — split into atomic reqs: '{snippet}'")

            # Measurability — check only the text AFTER the first colon to avoid hitting digits in the REQ ID
            text_part = req.split(":", 1)[1] if ":" in req else req
            text_lower = text_part.lower()
            has_digit = any(ch.isdigit() for ch in text_part)
            has_boolean_action = any(kw in text_lower for kw in _measurable_boolean_keywords)
            if not has_digit and not has_boolean_action:
                warnings.append(f"No measurable criterion (number, unit, or threshold): '{snippet}'")

            # Ambiguous terms (also on text part only)
            for term in _ambiguous:
                if term in text_lower:
                    warnings.append(f"Ambiguous term '{term}': '{snippet}'")

        # Category coverage: warn on missing critical categories
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
        }

    def merge_requirements(
        self,
        llm_requirements: List[str],
        manual_requirements: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Merge LLM-extracted and manual requirements, eliminating duplicates.

        Manual requirements always take precedence and are kept as-is.

        For each LLM requirement:
        - Dropped if its normalized text matches any already-accepted entry (true duplicate).
        - Dropped silently if same ID AND same text as a manual requirement.
        - Reassigned to the next available ID in its category if same ID but different text
          (ID collision with genuinely different content).
        - Accepted as-is if unique.

        Returns (merged_list, conflict_warnings).
        """
        def _parse_id(req: str) -> Optional[str]:
            m = _REQ_ID_RE.match(req.strip())
            return m.group(1) if m else None

        def _normalize(req: str) -> str:
            m = _REQ_ID_RE.match(req.strip())
            text = m.group(4) if m else req
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

            # Text-based dedup: same content already present (regardless of ID)
            if norm in seen_texts:
                continue

            if req_id and req_id in all_used_ids:
                # ID already taken — either by a manual req or an earlier LLM req
                cat_match = re.match(r"REQ-([A-Z]+)-\d+", req_id)
                if cat_match:
                    new_id = _next_id(cat_match.group(1), all_used_ids)
                    new_req = req.replace(req_id + ":", new_id + ":", 1)
                    if req_id in manual_ids:
                        # Explicit conflict with a manual requirement — warn
                        conflict_warnings.append(
                            f"REQ ID conflict: '{req_id}' already exists in manual requirements "
                            f"with different content. LLM requirement reassigned to '{new_id}'."
                        )
                    # LLM-internal ID collision (same ID used by two LLM reqs) — reassign silently
                    merged.append(new_req)
                    all_used_ids.add(new_id)
                    seen_texts.add(norm)
                # malformed / unrecognised category — skip
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

    def _parse_dependencies(self, text: str) -> List[Dict[str, str]]:
        """
        Extract dependency pairs from the Dependencies section of the LLM response.

        Looks for patterns like:
          REQ-FUNC-002 depends on REQ-INTF-001
        Returns a list of {"from": "REQ-FUNC-002", "to": "REQ-INTF-001"} dicts.
        """
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
        """Count requirements per category without running full validation."""
        counts: Dict[str, int] = {cat: 0 for cat in _VALID_CATEGORIES}
        counts["UNKNOWN"] = 0
        for req in requirements:
            m = _REQ_ID_RE.match(req.strip())
            if m:
                cat = m.group(2)
                counts[cat if cat in _VALID_CATEGORIES else "UNKNOWN"] += 1
        return counts
