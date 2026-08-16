"""Role-scoped authoring of one refinement candidate."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..llm.chain_of_thought import ChainOfThoughtPrompter, CoTResult


ContextRetriever = Callable[[str], str]


REFINEMENT_SYSTEM_PROMPT = """You are an expert MBSE architect refining an existing SysML v2 model for a cyber-physical system.

Your task is to FIX specific reported issues while preserving all valid structure.
Rules:
  - Do NOT restructure parts that already satisfy their requirements correctly.
  - Do NOT add new part defs unless explicitly required by an unaddressed requirement.
  - Every change must directly address one of the listed issues.
  - Output must be syntactically valid SysML v2.

Key constructs (same as generation):
  satisfy requirement <REQ_ID>;          — INSIDE a part def body; REQ_ID uses underscores
  connect <partA>.<portA> to <partB>.<portB>;       — SysML v2 dot notation

  // State machine: emergencyStop is a top-level action def in the part body,
  // referenced (not defined) inside the fault state's entry; transitions use
  // the canonical first/if/then keywords (NOT from/to/when, NOT `->`).
  action def emergencyStop { }
  state def <Name> {
      state nominal;
      state fault { entry action stop : emergencyStop; }
      entry; then nominal;
      transition <name>Fault first nominal if <condition> then fault;
  }

SATISFY SYNTAX RULE (critical):
  ✓ Inside a part def: satisfy requirement REQ_SAFE_001;
  ✗ Never:            satisfy REQ_SAFE_001 by HealthMonitor;   (causes parser scope errors)

REQUIREMENT USAGE RULE (critical):
  ✓ At package level: requirement def REQ_FUNC_001 { doc /* description */ }
  ✓ Inside a part def: satisfy requirement REQ_FUNC_001;
  ✗ NEVER write: requirement req : String = "...";        (not valid SysML v2)
  ✗ NEVER write: requirement REQ_001 : String = "...";   (not valid SysML v2)
  The `requirement` keyword is ONLY valid as part of `requirement def` at package level.
  To trace a requirement, use `satisfy requirement <ID>;` inside the owning part def body.

DOC COMMENT SYNTAX RULE (critical):
  ✓ requirement def REQ_FUNC_001 { doc /* The drone shall navigate ... */ }
  ✗ NEVER: requirement def REQ_FUNC_001 { doc = "The drone shall navigate ..."; }
  The `doc = "string"` form is NOT valid SysML v2 — use `doc /* text */` always.

UNIT SYNTAX RULE:
  ✓ Use simple identifiers: [m], [kg], [min], [m_s], [dB]
  ✗ Never use slashes:      [m/s], [mm/hr]   (breaks the parser)

STATE NAMING RULE: every state name must be unique across ALL state defs in the part.

SAFE requirement rule: satisfy links for SAFE requirements must target the
dedicated safety/monitoring/sensor part. Do NOT assign them to generic
structural containers (Airframe, Chassis, MainUnit).

Enclose the entire refined model in exactly one ```sysml code block. No prose after the block.
"""


@dataclass(frozen=True)
class RefinementRequest:
    existing_model: object
    feedback: str
    issues: Sequence[str] = ()
    skip_rag: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class RefinementOutcome:
    response: CoTResult
    metadata: Mapping[str, object]


class RefinementAuthoring:
    """Own source selection, retrieval, role prompt, and response extraction."""

    def __init__(
        self,
        prompter: ChainOfThoughtPrompter,
        retrieve_context: ContextRetriever,
    ) -> None:
        self._prompter = prompter
        self._retrieve_context = retrieve_context

    def refine(self, request: RefinementRequest) -> RefinementOutcome:
        issues = list(request.issues)
        if request.skip_rag:
            rag_context = ""
            query = ""
            if request.verbose:
                print("\n  [DEBUG] Refinement RAG skipped (skip_rag=True)")
        else:
            query = self._build_query(issues)
            rag_context = self._retrieve_context(query)
            if request.verbose:
                print(f"\n  [DEBUG] Refinement RAG query: {query!r}")

        source_metadata = (
            getattr(request.existing_model, "metadata", None) or {}
        )
        source_text = (
            source_metadata.get("last_sysml_text")
            or request.existing_model.to_sysml_text()
        )
        source_label = (
            "original LLM text"
            if source_metadata.get("last_sysml_text")
            else "to_sysml_text() fallback"
        )
        augmented_feedback = (
            f"{rag_context}\n\n---\n\n{request.feedback}"
            if rag_context else request.feedback
        )
        response = self._prompter.refine_design(
            model_text=source_text,
            feedback=augmented_feedback,
            issues=issues or [request.feedback],
            system_prompt=REFINEMENT_SYSTEM_PROMPT,
        )
        if request.verbose:
            print(f"\n  {'─'*60}")
            print("  [DEBUG] Refined SysML (LLM output, before parsing)")
            print(f"  {'─'*60}")
            print(
                response.extracted_sysml
                or "  ⚠ No ```sysml block extracted — raw answer:\n"
                + response.final_answer
            )
        return RefinementOutcome(
            response=response,
            metadata={
                "rag_skipped": request.skip_rag,
                "rag_query": query,
                "source": source_label,
            },
        )

    @staticmethod
    def _build_query(issues: Sequence[str]) -> str:
        if not issues:
            return "SysML v2 satisfy requirement traceability"

        issues_lower = " ".join(issues).lower()
        terms: list[str] = []
        if any(
            keyword in issues_lower
            for keyword in ("safe", "fault", "emergency", "shutdown")
        ):
            terms.append(
                "state def fault entry transition emergency action"
            )
        if any(
            keyword in issues_lower
            for keyword in ("port", "direction", "undirected")
        ):
            terms.append("port def direction in out inout")
        if any(
            keyword in issues_lower
            for keyword in ("satisfy", "untraced", "traceab")
        ):
            terms.append("satisfy requirement traceability link")
        if any(
            keyword in issues_lower
            for keyword in ("attribute", "numeric", "value", "unit")
        ):
            terms.append("attribute numeric value unit constraint")
        if any(
            keyword in issues_lower
            for keyword in ("connect", "interface", "consistency")
        ):
            terms.append("connect port interface")
        return " ".join(
            terms or ("SysML v2 refine design improve quality",)
        )

