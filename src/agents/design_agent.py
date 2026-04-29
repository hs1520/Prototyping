"""
Design Agent for MBSE prototyping.

Specializes in generating and refining SysML v2 design models
from requirements using Chain of Thought prompting and RAG.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import (
    FeatureDirection,
    PartDefinition,
    SatisfyRelationship,
    ElementRef,
    SysMLModel,
)
from src.sysml.Syside_AST_Parser import parse_sysml_to_model


# ─────────────────────────────────────────────────────────────────────────────
# Debug utility — prints a human-readable summary of a SysMLModel
# ─────────────────────────────────────────────────────────────────────────────

_DIR_LABELS = {
    FeatureDirection.IN:    "in",
    FeatureDirection.OUT:   "out",
    FeatureDirection.INOUT: "inout",
    FeatureDirection.NONE:  "—",
}


def _print_sysml_model_debug(
    model: SysMLModel,
    label: str = "SysMLModel",
    parse_diagnostics: Optional[List[Dict]] = None,
) -> None:
    """Print a structured human-readable summary of a parsed SysMLModel."""
    W = 70
    satisfy_total = sum(
        len(p.satisfy_relationships) for p in model.part_definitions
    )
    print(f"\n  ┌{'─' * (W - 2)}┐")
    header = f" {label}: {model.name}"
    stats = (
        f"Parts={len(model.part_definitions)}  "
        f"Reqs={len(model.requirement_definitions)}  "
        f"Satisfy={satisfy_total}"
    )
    print(f"  │{header:<{W - 2}}│")
    print(f"  │  {stats:<{W - 4}}│")
    print(f"  └{'─' * (W - 2)}┘")

    for part in model.part_definitions:
        print(f"  ■ {part.name}")

        # Ports
        if part.ports:
            port_strs = []
            for p in part.ports:
                d = _DIR_LABELS.get(
                    getattr(p, "direction", FeatureDirection.NONE),
                    "—",
                )
                port_strs.append(f"{p.name}({d})")
            print(f"      Ports  : {', '.join(port_strs)}")
        else:
            print(f"      Ports  : (none)")

        # Attributes
        if part.attributes:
            attr_strs = []
            for a in part.attributes:
                val = getattr(a, "default_value", None)
                unit = getattr(a, "unit", "")
                suffix = f" = {val}" if val is not None else ""
                suffix += f" [{unit}]" if unit else ""
                attr_strs.append(f"{a.name}{suffix}")
            print(f"      Attrs  : {', '.join(attr_strs)}")

        # Actions
        if part.actions:
            print(f"      Actions: {', '.join(a.name for a in part.actions)}")

        # State defs
        if part.nested_definitions:
            print(
                f"      States : "
                f"{', '.join(nd.name for nd in part.nested_definitions)}"
            )

        # Satisfy links
        if part.satisfy_relationships:
            targets = [
                sr.target.name
                for sr in part.satisfy_relationships
                if sr.target and sr.target.name
            ]
            print(f"      Satisfy: {', '.join(targets)}")

    # Requirement definitions
    if model.requirement_definitions:
        req_names = [r.name for r in model.requirement_definitions]
        # Wrap at ~60 chars
        line, lines = [], []
        for rn in req_names:
            line.append(rn)
            if sum(len(x) + 2 for x in line) > 56:
                lines.append(", ".join(line))
                line = []
        if line:
            lines.append(", ".join(line))
        print(f"  Requirements: {lines[0]}")
        for extra in lines[1:]:
            print(f"               {extra}")

    # Parse diagnostics
    diags = parse_diagnostics or []
    errors = [d for d in diags if "error" in str(d.get("severity", "")).lower()]
    warnings = [d for d in diags if "warn" in str(d.get("severity", "")).lower()]
    if errors:
        print(f"  ⚠ Parse errors ({len(errors)}):")
        for d in errors[:5]:
            span = d.get("source_span") or {}
            loc = ""
            if span:
                start = span.get("start", {})
                loc = f" [line {start.get('line', '?')}]"
            print(f"      {d['message']}{loc}")
    elif warnings:
        print(f"  ℹ Parse warnings: {len(warnings)}")
    else:
        print(f"  ✓ Parse diagnostics: none")


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

    REFINEMENT_SYSTEM_PROMPT = """You are an expert MBSE architect refining an existing SysML v2 model for a cyber-physical system.

Your task is to FIX specific reported issues while preserving all valid structure.
Rules:
  - Do NOT restructure parts that already satisfy their requirements correctly.
  - Do NOT add new part defs unless explicitly required by an unaddressed requirement.
  - Every change must directly address one of the listed issues.
  - Output must be syntactically valid SysML v2.

Key constructs (same as generation):
  satisfy <REQ_ID> by <partRef>;    — REQ_ID uses underscores, not hyphens
  connect <partA>::<portA> to <partB>::<portB>;
  state def <Name> { state nominal; state fault { entry; action def emergencyStop {} }
                     transition nominal -> fault when <condition>; }

SAFE requirement rule: satisfy links for SAFE requirements must target the
dedicated safety/monitoring/sensor part. Do NOT assign them to generic
structural containers (Airframe, Chassis, MainUnit).

Enclose the entire refined model in exactly one ```sysml code block. No prose after the block.
"""

    SYSTEM_PROMPT = """You are an expert MBSE architect generating SysML v2 prototype models for cyber-physical systems.

Your output must be syntactically valid SysML v2. Key constructs:
  package <Name> { ... }
  part def <Name> { port ...; attribute ...; action ...; }
  port def <Name> { ... }   or inline:  port <name> : <PortDef>;
  attribute <name> : <Type> = <value> [<unit>];
  action def <Name> { ... }
  state def <Name> { ... }
  satisfy <REQ_ID> by <partRef>;        — REQ_ID uses underscores, not hyphens
  connect <partA>::<portA> to <partB>::<portB>;

Requirement category → mandatory SysML construct:
  FUNC  → part def + action def (the functional behavior)
  PERF  → attribute with numeric value and SI unit (the measurable bound)
  SAFE  → state def with explicit fault-entry transition + emergency action def;
          satisfy link MUST target the dedicated safety/monitoring/sensor part
          (e.g., SafetyMonitor, FaultManager, SensorSuite, HealthMonitor).
          Do NOT put SAFE satisfy links on a generic structural container such
          as Airframe, Chassis, MainUnit, or Body — those are for CONS/FUNC.
  INTF  → port def with direction + connect usage linking two components
  CONS  → doc annotation or attribute constraint capturing the imposed limit

Every part def MUST have:
  • ≥ 1 port with direction (in / out / inout)
  • ≥ 1 attribute with numeric value and unit
  • ≥ 1 satisfy link to a requirement (using underscore form of the REQ ID)

Generate a complete, consistent initial prototype. Do not over-engineer.
Enclose the entire model in exactly one ```sysml code block. No prose after the block.
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
        verbose = task.get("verbose", False)

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

        is_refinement = bool(existing_model and feedback)

        if is_refinement:
            # Refinement mode — use a preservation-oriented system prompt
            if verbose:
                print(f"\n  {'─'*60}")
                print(f"  [DEBUG] Refinement Mode — Input feedback")
                print(f"  {'─'*60}")
                print(feedback)
                print(f"\n  [DEBUG] Existing model before refinement:")
                _print_sysml_model_debug(existing_model, label="Before Refinement")

            # Prefer the original LLM-generated SysML text stored at parse time.
            # Falling back to to_sysml_text() would give the LLM a reconstructed
            # (potentially degraded) version rather than the actual last good output.
            source_text = (
                (getattr(existing_model, "metadata", None) or {}).get("last_sysml_text")
                or existing_model.to_sysml_text()
            )
            source_label = (
                "original LLM text"
                if (getattr(existing_model, "metadata", None) or {}).get("last_sysml_text")
                else "to_sysml_text() fallback"
            )
            if verbose:
                print(f"  [DEBUG] Refinement source: {source_label}")

            original_system_prompt = self.cot.system_prompt
            self.cot.system_prompt = self.REFINEMENT_SYSTEM_PROMPT
            try:
                cot_result = self.cot.refine_design(
                    model_text=source_text,
                    feedback=feedback,
                    issues=refinement_issues or [feedback],
                )
            finally:
                self.cot.system_prompt = original_system_prompt

            if verbose:
                print(f"\n  {'─'*60}")
                print(f"  [DEBUG] Refined SysML (LLM output, before parsing)")
                print(f"  {'─'*60}")
                if cot_result.extracted_sysml:
                    print(cot_result.extracted_sysml)
                else:
                    print("  ⚠ No ```sysml block extracted — raw answer:")
                    print(cot_result.final_answer)

            generation_metadata = {}
        else:
            # Generation mode — multi-step pipeline
            cot_result, generation_metadata = self._multistep_generate(
                system_name=system_name,
                requirements=requirements,
                context=context,
                verbose=verbose,
            )

        # Build or update the SysML model
        model = existing_model or SysMLModel(
            name=system_name,
            description=f"Auto-generated design for {system_name}",
        )

        if not cot_result.extracted_sysml:
            raise RuntimeError("[SysML_EXTRACTION_ERROR] 未提取到SysML v2 design.")

        # Integrate Syside AST Parser: parse the extracted SysML text into a SysMLModel.
        parse_strict = task.get("parse_strict", False)
        fail_on_parse_error = task.get("fail_on_parse_error", False)

        parse_diagnostics = []
        parse_label = "After Refinement" if is_refinement else "After Initial Generation"
        try:
            parsed = parse_sysml_to_model(
                cot_result.extracted_sysml, model_name=system_name, strict=parse_strict
            )

            # Serialize diagnostics into lightweight dicts for metadata
            for d in getattr(parsed, "diagnostics", []):
                sev = getattr(d, "severity", None)
                try:
                    sev_str = sev.value if hasattr(sev, "value") else str(sev)
                except Exception:
                    sev_str = str(sev)
                msg = getattr(d, "message", str(d))
                span = getattr(d, "source_span", None)
                source_span = None
                if span is not None:
                    start = getattr(span, "start", None)
                    end = getattr(span, "end", None)
                    if start is not None and end is not None:
                        source_span = {
                            "start": {"line": getattr(start, "line", 0), "character": getattr(start, "character", 0)},
                            "end": {"line": getattr(end, "line", 0), "character": getattr(end, "character", 0)},
                        }
                parse_diagnostics.append({"severity": sev_str, "message": msg, "source_span": source_span})

            # Replace existing model with parsed model (replacement strategy)
            model = parsed

            # Persist the original LLM-generated SysML text so future refinement
            # rounds can use it directly instead of reconstructing via to_sysml_text().
            if not hasattr(model, "metadata") or model.metadata is None:
                object.__setattr__(model, "metadata", {})
            model.metadata["last_sysml_text"] = cot_result.extracted_sysml

            if verbose:
                print(f"\n  {'─'*60}")
                print(f"  [DEBUG] Parsed SysMLModel — {parse_label}")
                print(f"  {'─'*60}")
                _print_sysml_model_debug(model, label=parse_label, parse_diagnostics=parse_diagnostics)

        except Exception as e:
            # Parsing failed unexpectedly
            parse_error_msg = str(e)
            if verbose:
                print(f"\n  ✗ [DEBUG] Parse FAILED: {parse_error_msg}")
            metadata = {"parse_error": parse_error_msg, "parse_diagnostics": parse_diagnostics}
            result = AgentResult(
                agent_name=self.name,
                success=not fail_on_parse_error,
                output=model,
                reasoning=cot_result.final_answer,
                metadata={**metadata, **generation_metadata},
            )
            self.record_result(result)
            return result

        untraced = self._apply_requirement_traceability(model, requirements)

        result = AgentResult(
            agent_name=self.name,
            success=True,
            output=model,
            reasoning=cot_result.final_answer,
            metadata={
                "sysml_extracted": cot_result.extracted_sysml is not None,
                "thought_steps": len(cot_result.thought_steps),
                "part_definitions_created": len(model.part_definitions),
                "parse_diagnostics": parse_diagnostics,
                "untraced_requirements": untraced,
                **generation_metadata,
            },
        )
        self.record_result(result)
        return result

    _BEHAVIORAL_CATEGORIES = {"FUNC", "SAFE"}

    def _multistep_generate(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        verbose: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        4-step generation pipeline:
          1. Architecture Decomposition  (structured text plan)
          2. Part Definitions            (SysML structural fragment)
          3. Behavioral Model            (SysML behavioral fragment, FUNC/SAFE only)
          4. Integration / Assembly      (complete SysML package)

        Returns (final_CoTResult, generation_metadata).
        final_CoTResult.extracted_sysml is the assembled model.
        generation_metadata carries per-step diagnostics.
        """
        metadata: Dict[str, Any] = {
            "generation_steps_completed": 0,
            "degraded_steps": [],
        }

        # --- Step 1: Architecture Decomposition ---
        step1 = self.cot.decompose_architecture(
            system_name=system_name,
            requirements=requirements,
            context=context,
        )
        architecture_text = step1.final_answer
        metadata["generation_steps_completed"] = 1
        metadata["architecture_length"] = len(architecture_text)

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 1 — Architecture Decomposition")
            print(f"  {'─'*60}")
            print(architecture_text)

        # --- Step 2: Part Definitions (structural fragment) ---
        step2 = self.cot.generate_part_definitions(
            system_name=system_name,
            architecture=architecture_text,
            requirements=requirements,
            context=context,
        )
        if step2.extracted_sysml:
            parts_fragment = step2.extracted_sysml
        else:
            parts_fragment = step2.final_answer
            metadata["degraded_steps"].append(
                "step2_parts: no SysML code block extracted, falling back to raw text"
            )
        metadata["generation_steps_completed"] = 2
        metadata["parts_fragment_length"] = len(parts_fragment)

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 2 — Part Definitions (SysML fragment)")
            print(f"  {'─'*60}")
            print(parts_fragment)

        # --- Step 3: Behavioral Model (only if FUNC or SAFE requirements exist) ---
        behavioral_reqs = [
            r for r in requirements
            if any(f"-{cat}-" in r for cat in self._BEHAVIORAL_CATEGORIES)
        ]
        if behavioral_reqs:
            step3 = self.cot.generate_behavior(
                system_name=system_name,
                architecture=architecture_text,
                behavioral_requirements=behavioral_reqs,
                parts_fragment=parts_fragment,
                context=context,
            )
            if step3.extracted_sysml:
                behavior_fragment = step3.extracted_sysml
            else:
                behavior_fragment = step3.final_answer
                metadata["degraded_steps"].append(
                    "step3_behavior: no SysML code block extracted, falling back to raw text"
                )
            metadata["generation_steps_completed"] = 3
            metadata["behavior_fragment_length"] = len(behavior_fragment)

            if verbose:
                print(f"\n  {'─'*60}")
                print(f"  [DEBUG] Step 3 — Behavioral Model (SysML fragment)")
                print(f"  {'─'*60}")
                print(behavior_fragment)
        else:
            behavior_fragment = ""
            metadata["generation_steps_completed"] = 3
            metadata["behavior_fragment_length"] = 0

            if verbose:
                print(f"\n  [DEBUG] Step 3 — Behavioral Model: skipped "
                      f"(no FUNC/SAFE requirements)")

        # --- Step 4: Integration / Assembly ---
        step4 = self.cot.assemble_model(
            system_name=system_name,
            parts_fragment=parts_fragment,
            behavior_fragment=behavior_fragment,
            requirements=requirements,
        )
        metadata["generation_steps_completed"] = 4
        metadata["total_thought_steps"] = (
            len(step1.thought_steps)
            + len(step2.thought_steps)
            + len(step3.thought_steps if behavioral_reqs else [])
            + len(step4.thought_steps)
        )

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 4 — Assembled SysML Model (final LLM output)")
            print(f"  {'─'*60}")
            if step4.extracted_sysml:
                print(step4.extracted_sysml)
            else:
                print("  ⚠ No ```sysml block extracted — raw answer:")
                print(step4.final_answer)

        return step4, metadata

    def _apply_requirement_traceability(
        self, model: SysMLModel, requirements: List[str]
    ) -> List[str]:
        """
        Ensure each requirement is linked to the best-matching component.

        Returns a list of requirement IDs that could not be matched to any component
        (score == 0 after scanning all parts). These are NOT force-assigned to a
        fallback part — callers should surface them as warnings.
        """
        if not requirements or not model.part_definitions:
            return []

        # Collect IDs already satisfied by the parsed model (from AST)
        satisfied_ids: set = {
            sr.target.name
            for part in model.part_definitions
            for sr in part.satisfy_relationships
            if sr.target and sr.target.name
        }

        def _tokenize(text: str) -> set:
            """Split text into lowercase tokens, handling camelCase, PascalCase, and snake_case."""
            # Insert space before each uppercase letter to split camelCase/PascalCase,
            # then extract all alphanumeric words.
            spaced = re.sub(r"([A-Z])", r" \1", text)
            return set(re.findall(r"[a-z0-9]+", spaced.lower()))

        def _part_tokens(part: PartDefinition) -> set:
            """Collect semantic tokens from a part: name, ports, attrs, actions, doc."""
            tokens: set = _tokenize(part.name)
            for port in part.ports:
                tokens.update(_tokenize(port.name))
            for attr in part.attributes:
                tokens.update(_tokenize(attr.name))
            for action in part.actions:
                tokens.update(_tokenize(action.name))
            if part.short_description:
                tokens.update(_tokenize(part.short_description))
            return tokens

        # Pre-compute token sets once per part
        part_token_sets = {part.name: _part_tokens(part) for part in model.part_definitions}

        def _score(req_tokens: set, part: PartDefinition) -> int:
            return len(req_tokens & part_token_sets[part.name])

        untraced: List[str] = []

        for req_text in requirements:
            id_match = re.match(r"(REQ-\w+-\d+|REQ-\d+):\s*(.*)", req_text)
            if not id_match:
                continue  # malformed requirement, skip
            req_id = id_match.group(1).replace("-", "_")
            req_body = id_match.group(2)

            if req_id in satisfied_ids:
                continue  # already linked by the parsed model

            req_tokens = set(re.findall(r"[A-Za-z0-9_]+", req_body.lower()))
            scored = sorted(
                model.part_definitions,
                key=lambda p: _score(req_tokens, p),
                reverse=True,
            )
            best = scored[0] if scored else None
            best_score = _score(req_tokens, best) if best else 0

            if best_score == 0:
                # No semantic match found — record as untraced, do not force-assign
                untraced.append(req_id)
                continue

            best.add_satisfy(SatisfyRelationship(
                source=best.to_ref(),
                target=ElementRef(name=req_id),
            ))
            satisfied_ids.add(req_id)

        return untraced
