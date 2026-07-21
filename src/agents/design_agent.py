"""
Design Agent for MBSE prototyping.

Specializes in generating and refining SysML v2 design models
from requirements using Chain of Thought prompting and RAG.
"""

from __future__ import annotations

import dataclasses
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
from ..sysml.lite_model import SysMLLiteModel, build_lite_model
from ..utils.sysml_text_utils import find_block_end


class StructuralGenerationError(RuntimeError):
    """Initial generation failed, while retaining the rejected model evidence."""

    def __init__(
        self,
        message: str,
        *,
        original_model_text: str,
        candidate_model_text: str,
        diagnostics: List[Dict[str, str]],
        generation_metadata: Dict[str, Any],
        parser_metadata: Dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.original_model_text = original_model_text
        self.candidate_model_text = candidate_model_text
        self.diagnostics = diagnostics
        self.generation_metadata = generation_metadata
        self.parser_metadata = parser_metadata


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
    sysml_text: Optional[str] = None,
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
                # When Syside returns NONE (direction defined in typed port def body),
                # infer direction from port-name prefix as a display hint.
                if d == "—":
                    _pn = p.name.lower()
                    if _pn.startswith("in") or _pn.endswith("in"):
                        d = "in*"
                    elif _pn.startswith("out") or _pn.endswith("out"):
                        d = "out*"
                    elif _pn.startswith("inout") or "inout" in _pn:
                        d = "io*"
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

        # Nested definitions (action def / port def / attribute def captured by parser;
        # NOTE: state def is NOT captured by the Syside parser — state defs are preserved
        # in the SysML text but do not appear here)
        if part.nested_definitions:
            print(
                f"      NestedDefs: "
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

    # State defs — the Syside parser does NOT extract StateDefinition nodes into
    # nested_definitions, so we scan the raw SysML text to report them.
    if sysml_text:
        state_def_names = re.findall(r"\bstate\s+def\s+(\w+)", sysml_text)
        if state_def_names:
            print(f"  StateDefs (text): {', '.join(state_def_names)}")
        else:
            print(f"  StateDefs (text): (none)")

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
  satisfy requirement <REQ_ID>;          — INSIDE a part def body; REQ_ID uses underscores
  connect <partA>.<portA> to <partB>.<portB>;       — SysML v2 dot notation

  // State machine: emergencyStop is a top-level action def in the part body,
  // referenced (not defined) inside the fault state's entry; transitions use
  // the canonical first/if/then keywords (NOT from/to/when, NOT `->`).
  action def emergencyStop { }
  state def <Name> {
      state nominal;
      state fault { entry action stop : emergencyStop; }
      transition initial then nominal;
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

    SYSTEM_PROMPT = """You are an expert MBSE architect generating SysML v2 prototype models for cyber-physical systems.

Your output must be syntactically valid SysML v2. Key constructs:
  package <Name> { ... }
  part def <Name> { port ...; attribute ...; action ...; }
  port def <Name> { ... }   or inline:  port <name> : <PortDef>;
  attribute <name> : <Type> = <value> [<unit>];
  action def <Name> { ... }
  state def <Name> { ... }
  connect <partA>.<portA> to <partB>.<portB>;     — SysML v2 dot notation; do NOT use `::`

SATISFY LINK SYNTAX — critical, follow exactly:
  ✓ INSIDE a part def body:   satisfy requirement <REQ_ID>;
  ✗ NEVER write inside a part def:  satisfy <REQ_ID> by <PartName>;
  The "by" form is only valid at the package level with lowercase usage names,
  but it causes Syside parser scope errors — do NOT use it anywhere.

DOC COMMENT SYNTAX — use only `doc /* text */`, never `doc = "text"`:
  ✓  requirement def REQ_FUNC_001 { doc /* The system shall navigate ... */ }
  ✗  requirement def REQ_FUNC_001 { doc = "The system shall navigate ..."; }
  The string-assignment form is not valid SysML v2 and causes parser round-trip bugs.

REQUIREMENT USAGE SYNTAX — never use `requirement` as an attribute:
  ✓ At package level: requirement def REQ_FUNC_001 { doc /* ... */ }
  ✓ Inside a part def: satisfy requirement REQ_FUNC_001;
  ✗ NEVER: requirement req : String = "...";   (completely invalid SysML v2)

UNIT SYNTAX — use only simple identifiers in [...]:
  ✓  attribute speed : Real = 15.0 [m_s];   (use underscore for compound units)
  ✗  attribute speed : Real = 15.0 [m/s];   (slash inside [...] breaks the parser)
  ✗  attribute prot  : Real = 54.0 [IP];    (acronyms may confuse Syside)
  For unitless or non-SI quantities, omit the [...] block entirely.

STATE DEF NAMING — every state inside every state def must have a globally unique name:
  ✓  state BattNominal;  state BattCritical { ... }    (prefixed with context)
  ✗  state Nominal;      state Nominal;                 (duplicate name across defs)

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
  • ≥ 1 satisfy link:  satisfy requirement <REQ_ID>;   (REQ_ID uses underscores)

STANDARD SAFETY ATTRIBUTE NAMES — use these EXACT names when the concept applies.
Downstream SITL parameter mapping looks them up by name; non-standard names
silently break ArduPilot parameter generation:
  • batterySoc          — battery state-of-charge percent  (Real, [percent])
                           used by SafetyMonitor for battery RTB / land thresholds
  • commLossTime        — seconds since last GCS heartbeat  (Real, [s])
                           used by SafetyMonitor link-loss state machine
  • controlFrequency    — primary flight control loop rate  (Real, [Hz])
                           used by FlightController to set SCHED_LOOP_RATE
  • parachuteDeployTime — parachute actuation delay         (Real, [s])
                           used by SafetyMonitor → CHUTE_DELAY_MS
  • propulsionCriticalFailure — engine/motor failure flag    (Boolean)
                           used by SafetyMonitor parachute trigger guard
  • sensorSelfTestFailed      — POST sensor failure flag    (Boolean)
                           used by SafetyMonitor arming inhibit
  • deliveryAbortConditionActive — payload abort flag        (Boolean)
                           used by SafetyMonitor payload lock state machine

State machine guards MUST reference these standard attribute names by exact spelling.
Example:
  ✓  if batterySoc <= 25.0     ✗  if batteryChargeLevel <= 25.0
  ✓  if commLossTime > 10.0    ✗  if linkTimeout > 10.0

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

    def _run_refinement(
        self,
        existing_model,
        feedback: str,
        refinement_issues: List[str],
        skip_rag: bool,
        verbose: bool,
    ):
        """Refinement mode: repair an existing model from evaluator feedback.

        Prefers the original LLM-generated SysML text stored at parse time —
        falling back to to_sysml_text() would give the LLM a reconstructed
        (potentially degraded) version rather than the actual last good
        output.  Returns the CoT refinement result.
        """
        if skip_rag:
            # Syntax-fix calls skip RAG entirely — the corpus has no useful examples
            # for "fix undefined feature reference" errors, and the generic fallback
            # query ("SysML v2 refine design improve quality") adds noise.
            rag_context = ""
            if verbose:
                print(f"\n  [DEBUG] Refinement RAG skipped (skip_rag=True)")
        else:
            refinement_rag_query = self._build_refinement_query(refinement_issues)
            rag_context = self.get_augmented_context(
                refinement_rag_query,
                include_official_sysml=True,
                allowed_extensions=(".sysml",),
            )
            if verbose:
                print(f"\n  [DEBUG] Refinement RAG query: {refinement_rag_query!r}")

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Refinement Mode — Input feedback")
            print(f"  {'─'*60}")
            print(feedback)
            print(f"\n  [DEBUG] Existing model before refinement:")
            _print_sysml_model_debug(existing_model, label="Before Refinement")

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

        # Prepend RAG context to the feedback so the LLM sees relevant examples
        # before being asked to fix the issues.
        augmented_feedback = (
            f"{rag_context}\n\n---\n\n{feedback}" if rag_context else feedback
        )

        original_system_prompt = self.cot.system_prompt
        self.cot.system_prompt = self.REFINEMENT_SYSTEM_PROMPT
        try:
            cot_result = self.cot.refine_design(
                model_text=source_text,
                feedback=augmented_feedback,
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
        return cot_result

    def _apply_semantic_fixes(
        self,
        cot_result,
        requirements: List[str],
        generation_metadata: Dict[str, Any],
        verbose: bool,
    ):
        """Deterministic semantic cleanup for both generation and refinement.

        These are requirement-operator invariants, not stylistic guesses, so
        no LLM call is spent repairing them.  Mutates *generation_metadata*
        in place and returns the (possibly replaced) CoT result.
        """
        cleaned_sysml, capability_fixes = self._fix_capability_semantics(
            cot_result.extracted_sysml, requirements
        )
        cleaned_sysml, action_fixes = self._fix_safety_action_semantics(cleaned_sysml)
        cleaned_sysml, self_test_fixes = self._fix_self_test_behavior_semantics(
            cleaned_sysml, requirements
        )
        cleaned_sysml, ownership_fixes = self._fix_functional_satisfy_ownership(
            cleaned_sysml, requirements
        )
        if capability_fixes or action_fixes or self_test_fixes or ownership_fixes:
            cot_result = dataclasses.replace(cot_result, extracted_sysml=cleaned_sysml)
            generation_metadata["semantic_fixes"] = {
                "capability": capability_fixes,
                "safety_action": action_fixes,
                "self_test_behavior": self_test_fixes,
                "functional_satisfy_ownership": ownership_fixes,
            }
            if verbose:
                print(
                    f"\n  [DEBUG] Semantic consistency fixes: "
                    f"capability={capability_fixes}, safety_action={action_fixes}, "
                    f"self_test_behavior={self_test_fixes}, "
                    f"functional_satisfy_ownership={ownership_fixes}"
                )
        return cot_result

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
        # Legacy external-contract guidance inputs. The R0-CURRENT baseline never
        # sets these (the orchestrator supplies them only under the B1/B2
        # robustness options), so every downstream `if contract_bundle:` block —
        # and its function-local safety_patterns/contract_types import — is skipped
        # on the baseline path. This agent is therefore already decoupled from the
        # legacy contract layer for R0; the gated blocks are removed wholesale with
        # the Layer-2 excision after the R2-BBAG checker lands.
        contract_bundle = task.get("contract_bundle")
        pattern_bindings = tuple(task.get("pattern_bindings") or ())

        is_refinement = bool(existing_model and feedback)
        skip_rag = task.get("skip_rag", False)

        if is_refinement:
            cot_result = self._run_refinement(
                existing_model, feedback, refinement_issues, skip_rag, verbose
            )
            generation_metadata = {}
        else:
            # Generation mode — multi-step pipeline
            cot_result, generation_metadata = self._multistep_generate(
                system_name=system_name,
                requirements=requirements,
                context=context,
                verbose=verbose,
                platform_profile=task.get("platform_profile"),
                contract_bundle=contract_bundle,
                pattern_bindings=pattern_bindings,
            )

        if not cot_result.extracted_sysml:
            raise RuntimeError("[SysML_EXTRACTION_ERROR] 未提取到SysML v2 design.")

        # Deterministic semantic cleanup applies to both initial generation and
        # refinement.  These are requirement-operator invariants, not stylistic
        # guesses, so do not spend another LLM call repairing them.
        cot_result = self._apply_semantic_fixes(
            cot_result, requirements, generation_metadata, verbose
        )

        parse_label = "After Refinement" if is_refinement else "After Initial Generation"

        # Parse via syside native API (replaces Syside_AST_Parser)
        model = build_lite_model(cot_result.extracted_sysml, model_name=system_name)

        # Never advertise a structurally empty initial design as successful.
        # A single syntax-only repair is part of the common generation pipeline
        # (identical for B0/B1/B2), rather than a B2 semantic intervention.
        if not is_refinement and not model.part_definitions:
            parser_metadata = {
                key: value
                for key, value in dict(
                    getattr(model, "metadata", None) or {}
                ).items()
                if key != "last_sysml_text"
            }
            textual_parts = re.findall(
                r"\bpart\s+def\s+(\w+)\s*\{",
                cot_result.extracted_sysml,
            )
            restored_parts = generation_metadata.get("injected_part_defs", [])
            original_model_text = cot_result.extracted_sysml
            initial_diagnostics = [
                {
                    "severity": (
                        d.severity.value
                        if hasattr(d.severity, "value") else str(d.severity)
                    ),
                    "message": d.message,
                }
                for d in model.diagnostics
            ]

            if parser_metadata.get("syside_available") is False:
                raise StructuralGenerationError(
                    "[SYSIDE_UNAVAILABLE] Initial model validation requires the "
                    "Syside parser; activate the project environment before running.",
                    original_model_text=original_model_text,
                    candidate_model_text=original_model_text,
                    diagnostics=initial_diagnostics,
                    generation_metadata=dict(generation_metadata),
                    parser_metadata=parser_metadata,
                )

            issue_lines = [
                "The assembled model contains textual part definitions, but Syside "
                "recovered none. Repair syntax only and preserve every requirement, "
                "part, behavior, threshold, satisfy link, and connection. Return one "
                "complete SysML v2 package.",
            ]
            issue_lines.extend(
                f"{item['severity']}: {item['message']}"
                for item in initial_diagnostics[:12]
            )
            parse_error = parser_metadata.get("syside_parse_error")
            if parse_error:
                issue_lines.append(f"parser exception: {parse_error}")
            repair_feedback = "\n".join(issue_lines)
            repaired_result = self._run_refinement(
                model,
                repair_feedback,
                issue_lines,
                skip_rag=True,
                verbose=verbose,
            )
            if repaired_result.extracted_sysml:
                repaired_result = self._apply_semantic_fixes(
                    repaired_result, requirements, generation_metadata, verbose
                )
                repaired_model = build_lite_model(
                    repaired_result.extracted_sysml, model_name=system_name
                )
            else:
                repaired_model = None

            repaired_diagnostics = [
                {
                    "severity": (
                        d.severity.value
                        if hasattr(d.severity, "value") else str(d.severity)
                    ),
                    "message": d.message,
                }
                for d in (repaired_model.diagnostics if repaired_model else [])
            ]
            generation_metadata["initial_parse_repair"] = {
                "attempted": True,
                "successful": bool(
                    repaired_model and repaired_model.part_definitions
                ),
                "initial_textual_part_defs": len(textual_parts),
                "initial_diagnostics": initial_diagnostics,
                "repair_diagnostics": repaired_diagnostics,
            }
            if repaired_model and repaired_model.part_definitions:
                cot_result = repaired_result
                model = repaired_model
            else:
                candidate_text = (
                    repaired_result.extracted_sysml
                    if repaired_result.extracted_sysml else original_model_text
                )
                candidate_parser_metadata = {
                    key: value
                    for key, value in dict(
                        getattr(repaired_model, "metadata", None)
                        or parser_metadata
                    ).items()
                    if key != "last_sysml_text"
                }
                raise StructuralGenerationError(
                    "[STRUCTURAL_GENERATION_ERROR] Assembled model contains no "
                    "parseable part definitions after one syntax-only repair "
                    f"(textual_part_defs={len(textual_parts)}, "
                    f"restored={restored_parts}).",
                    original_model_text=original_model_text,
                    candidate_model_text=candidate_text,
                    diagnostics=repaired_diagnostics or initial_diagnostics,
                    generation_metadata=dict(generation_metadata),
                    parser_metadata=candidate_parser_metadata,
                )

        parse_diagnostics = [
            {"severity": d.severity.value if hasattr(d.severity, "value") else str(d.severity),
             "message": d.message}
            for d in model.diagnostics
        ]

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] SysMLLiteModel — {parse_label}")
            print(f"  {'─'*60}")
            _print_sysml_model_debug(
                model,
                label=parse_label,
                parse_diagnostics=parse_diagnostics,
                sysml_text=cot_result.extracted_sysml,
            )

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

    _BEHAVIORAL_CATEGORIES = {"FUNC", "SAFE", "OPER"}

    def _step1_architecture(
        self,
        system_name: str,
        requirements: List[str],
        context: str,
        metadata: Dict[str, Any],
        verbose: bool,
    ):
        """Step 1: architecture decomposition (plain structured text).

        RAG is intentionally skipped: the corpus is exclusively `.sysml`
        grammar examples and RAGRetriever wraps every retrieved snippet in
        ```sysml fences, which (1) is the wrong knowledge type for
        decomposition planning — we want domain subsystem patterns, not SysML
        syntax; (2) primes the LLM to emit code blocks against this step's
        "plain text only" instruction; and (3) wastes tokens — Steps 2-4
        retrieve targeted SysML examples for their own tasks.  The fence-strip
        guard below is retained as defence-in-depth.
        Returns ``(step1_result, architecture_text)``.
        """
        if verbose:
            print(f"\n  [DEBUG] Step 1 — RAG skipped (corpus is SysML-only, "
                  f"would prime LLM to emit code blocks)")
        step1 = self.cot.decompose_architecture(
            system_name=system_name,
            requirements=requirements,
            context=context,   # caller-supplied context only, no RAG
        )
        architecture_text = step1.final_answer
        metadata["step1_rag_skipped"] = True

        # Guard: Step 1 must produce plain text only — no SysML.
        # If the LLM appended a ```sysml (or generic ```) code block, strip
        # everything from the first fence onward.  The structured component
        # list always precedes any code block, so this keeps the useful part.
        _fence_pos = architecture_text.find("```")
        if _fence_pos != -1:
            architecture_text = architecture_text[:_fence_pos].rstrip()
            metadata["degraded_steps"].append(
                "step1_architecture: SysML code block found and stripped"
            )
            if verbose:
                print(
                    f"\n  [DEBUG] Step 1 — Stripped trailing SysML code block "
                    f"(kept {len(architecture_text)} chars of plain-text plan)"
                )

        metadata["generation_steps_completed"] = 1
        metadata["architecture_length"] = len(architecture_text)

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 1 — Architecture Decomposition")
            print(f"  {'─'*60}")
            print(architecture_text)
        return step1, architecture_text

    def _step2_parts(
        self,
        system_name: str,
        architecture_text: str,
        requirements: List[str],
        step_context,
        metadata: Dict[str, Any],
        verbose: bool,
        contract_bundle=None,
    ):
        """Step 2: part definitions (structural fragment) with one bounded
        retry — a structural fragment without a single part definition cannot
        be repaired meaningfully by the later assembly/refinement stages, so
        fail before spending calls on interfaces, behaviour, and assembly.
        Returns ``(step2_result, parts_fragment)``.
        """
        ctx2 = step_context("parts")
        semantic_guidance = ""
        if contract_bundle:
            from ..prototyping.safety_patterns import render_step_guidance
            semantic_guidance = render_step_guidance(
                contract_bundle, (), step="parts",
                req_ids=[item.split(":", 1)[0] for item in requirements],
            )
        step2 = self.cot.generate_part_definitions(
            system_name=system_name,
            architecture=architecture_text,
            requirements=requirements,
            context=ctx2,
            semantic_guidance=semantic_guidance,
        )
        if step2.extracted_sysml:
            parts_fragment = step2.extracted_sysml
        else:
            parts_fragment = step2.final_answer
            metadata["degraded_steps"].append(
                "step2_parts: no SysML code block extracted, falling back to raw text"
            )

        if not re.search(r"\bpart\s+def\s+\w+\s*\{", parts_fragment):
            metadata["step2_part_retries"] = 1
            retry_step2 = self.cot.generate_part_definitions(
                system_name=system_name,
                architecture=architecture_text,
                requirements=requirements,
                context=ctx2,
                semantic_guidance=semantic_guidance,
            )
            retry_fragment = (
                retry_step2.extracted_sysml or retry_step2.final_answer or ""
            )
            if not re.search(r"\bpart\s+def\s+\w+\s*\{", retry_fragment):
                raise RuntimeError(
                    "[STRUCTURAL_GENERATION_ERROR] Step 2 produced no part "
                    "definitions after one targeted retry."
                )
            step2 = retry_step2
            parts_fragment = retry_fragment
            metadata["degraded_steps"].append(
                "step2_parts: first response had no part defs; targeted retry accepted"
            )
        metadata["generation_steps_completed"] = 2
        metadata["parts_fragment_length"] = len(parts_fragment)
        if semantic_guidance:
            metadata.setdefault("semantic_guidance_by_step", {})[
                "parts"
            ] = semantic_guidance

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 2 — Part Definitions (SysML fragment)")
            print(f"  {'─'*60}")
            print(parts_fragment)
        return step2, parts_fragment

    def _step3_interfaces(
        self,
        system_name: str,
        architecture_text: str,
        parts_fragment: str,
        requirements: List[str],
        step_context,
        metadata: Dict[str, Any],
        verbose: bool,
        contract_bundle=None,
    ):
        """Step 3: interface & flow definitions (item def / typed port def).
        Returns ``(step3_result, interfaces_fragment)``; the fragment is empty
        when no code block was extracted (degraded, not fatal)."""
        intf_reqs = [r for r in requirements if "-INTF-" in r]
        ctx3 = step_context("interfaces")
        semantic_guidance = ""
        if contract_bundle:
            from ..prototyping.safety_patterns import render_step_guidance
            semantic_guidance = render_step_guidance(
                contract_bundle, (), step="interfaces",
                req_ids=[item.split(":", 1)[0] for item in requirements],
            )
        step3 = self.cot.generate_interfaces_and_flows(
            system_name=system_name,
            architecture=architecture_text,
            parts_fragment=parts_fragment,
            intf_requirements=intf_reqs,
            context=ctx3,
            semantic_guidance=semantic_guidance,
        )
        if step3.extracted_sysml:
            interfaces_fragment = step3.extracted_sysml
        else:
            interfaces_fragment = ""
            metadata["degraded_steps"].append(
                "step3_interfaces: no SysML code block extracted, skipping"
            )
        metadata["generation_steps_completed"] = 3
        metadata["interfaces_fragment_length"] = len(interfaces_fragment)
        if semantic_guidance:
            metadata.setdefault("semantic_guidance_by_step", {})[
                "interfaces"
            ] = semantic_guidance

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 3 — Interface & Flow Definitions (SysML fragment)")
            print(f"  {'─'*60}")
            if interfaces_fragment:
                print(interfaces_fragment)
            else:
                print("  (skipped — no code block extracted)")
        return step3, interfaces_fragment

    def _step4_behavior(
        self,
        system_name: str,
        architecture_text: str,
        parts_fragment: str,
        requirements: List[str],
        platform_profile,
        step_context,
        metadata: Dict[str, Any],
        verbose: bool,
        contract_bundle=None,
        pattern_bindings=(),
    ):
        """Step 4: behavioral model — runs only when FUNC/SAFE/OPER
        requirements exist (the RAG call is skipped entirely otherwise).
        Returns ``(step4_result_or_None, behavior_fragment)``."""
        behavioral_reqs = [
            r for r in requirements
            if any(f"-{cat}-" in r for cat in self._BEHAVIORAL_CATEGORIES)
        ]
        # A typed contract may identify a behavioral obligation outside the
        # ID-category heuristic (notably PERF timed actuation).  Add only those
        # explicitly READY obligations; do not turn arbitrary PERF prose into
        # state machines.
        if contract_bundle:
            from ..prototyping.contract_types import READY, contract_bundle_from_dict
            typed_bundle = contract_bundle_from_dict(contract_bundle)
            behavioral_contract_ids = {
                contract.req_id
                for contract in typed_bundle.contracts
                if contract.completeness == READY
                and any(
                    obligation.kind in {
                        "triggered_response", "timed_response",
                        "timed_actuation", "state_invariant",
                    }
                    for obligation in contract.obligations
                )
            }
            for requirement in requirements:
                req_id = requirement.split(":", 1)[0].upper().replace("-", "_")
                if req_id in behavioral_contract_ids and requirement not in behavioral_reqs:
                    behavioral_reqs.append(requirement)
        if not behavioral_reqs:
            metadata["generation_steps_completed"] = 4
            metadata["behavior_fragment_length"] = 0

            if verbose:
                print(f"\n  [DEBUG] Step 4 — Behavioral Model: skipped "
                      f"(no FUNC/SAFE requirements)")
            return None, ""

        ctx4 = step_context("behavior")
        contract_pattern_guidance = ""
        if contract_bundle:
            from ..prototyping.safety_patterns import render_step_guidance
            req_ids = [r.split(":", 1)[0] for r in behavioral_reqs]
            contract_pattern_guidance = render_step_guidance(
                contract_bundle, pattern_bindings, step="behavior", req_ids=req_ids
            )
        step4 = self.cot.generate_behavior(
            system_name=system_name,
            architecture=architecture_text,
            behavioral_requirements=behavioral_reqs,
            parts_fragment=parts_fragment,
            context=ctx4,
            platform_profile=platform_profile,
            contract_pattern_guidance=contract_pattern_guidance,
        )
        if step4.extracted_sysml:
            behavior_fragment = step4.extracted_sysml
        else:
            behavior_fragment = step4.final_answer
            metadata["degraded_steps"].append(
                "step4_behavior: no SysML code block extracted, falling back to raw text"
            )
        metadata["generation_steps_completed"] = 4
        metadata["behavior_fragment_length"] = len(behavior_fragment)
        if contract_pattern_guidance:
            metadata.setdefault("semantic_guidance_by_step", {})[
                "behavior"
            ] = contract_pattern_guidance
            metadata["contract_pattern_guidance"] = contract_pattern_guidance
            metadata["pattern_bindings_used"] = [
                getattr(binding, "pattern_id", str(binding))
                for binding in pattern_bindings
                if getattr(binding, "req_id", "") in {
                    req_id.upper().replace("-", "_") for req_id in req_ids
                }
            ]

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 4 — Behavioral Model (SysML fragment)")
            print(f"  {'─'*60}")
            print(behavior_fragment)
        return step4, behavior_fragment

    def _postprocess_assembly(
        self,
        step5,
        parts_fragment: str,
        interfaces_fragment: str,
        behavior_fragment: str,
        metadata: Dict[str, Any],
        verbose: bool,
    ):
        """Post-assembly deterministic repair chain.

        Step 5 is an integration call, not an authority to delete the earlier
        fragments: restore dropped part/state/item defs programmatically, fix
        `doc = "...";` syntax, strip invalid requirement attribute lines,
        normalise `connect a::b` to dot notation, and flag suspicious
        connects.  Returns the (possibly replaced) step5 result.
        """
        # --- fix invalid `doc = "string";` → `doc /* string */` ---
        if step5.extracted_sysml:
            fixed_text, n_doc_fixed = self._fix_doc_syntax(step5.extracted_sysml)
            if n_doc_fixed:
                metadata["fixed_doc_syntax"] = n_doc_fixed
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Fixed {n_doc_fixed} invalid "
                        f"`doc = \"...\";` → `doc /* ... */` occurrence(s)"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=fixed_text)

        # --- restore structural part defs the LLM dropped ---
        if parts_fragment and step5.extracted_sysml:
            assembled_text, injected_parts = self._inject_missing_part_defs(
                step5.extracted_sysml, parts_fragment
            )
            if injected_parts:
                metadata["injected_part_defs"] = injected_parts
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Programmatic injection: "
                        f"restored {len(injected_parts)} dropped part def(s): "
                        f"{', '.join(injected_parts)}"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=assembled_text)

        # --- inject any state defs the LLM dropped ---
        if behavior_fragment and step5.extracted_sysml:
            assembled_text, injected_states = self._inject_missing_state_defs(
                step5.extracted_sysml, behavior_fragment
            )
            if injected_states:
                metadata["injected_state_defs"] = injected_states
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Programmatic injection: "
                        f"restored {len(injected_states)} dropped state def(s): "
                        f"{', '.join(injected_states)}"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=assembled_text)

        # --- inject any item defs / typed port defs the LLM dropped ---
        if interfaces_fragment and step5.extracted_sysml:
            assembled_text, injected_items = self._inject_missing_item_defs(
                step5.extracted_sysml, interfaces_fragment
            )
            if injected_items:
                metadata["injected_item_defs"] = injected_items
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Programmatic injection: "
                        f"restored {len(injected_items)} dropped item/port def(s): "
                        f"{', '.join(injected_items)}"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=assembled_text)

        # --- strip invalid `requirement <name> : <Type> = "...";` lines ---
        if step5.extracted_sysml:
            cleaned_text, n_stripped = self._strip_invalid_requirement_attrs(
                step5.extracted_sysml
            )
            if n_stripped:
                metadata["stripped_invalid_req_attrs"] = n_stripped
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Stripped {n_stripped} invalid "
                        f"`requirement <name> : <Type> = \"...\";` line(s)"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=cleaned_text)

        # --- normalise `connect a::b to c::d;` → `connect a.b to c.d;` ---
        # SysML v2 connect uses dot notation only.  Despite the prompt explicitly
        # teaching `.`, LLMs occasionally emit `::` (treating it as a generic
        # member-access operator).  Normalising here keeps every downstream
        # consumer (evaluator, RAG, refinement prompt) on the canonical form.
        if step5.extracted_sysml:
            normalised, n_normalised = self._normalise_connect_syntax(
                step5.extracted_sysml
            )
            if n_normalised:
                metadata["normalised_connect_syntax"] = n_normalised
                if verbose:
                    print(
                        f"\n  [DEBUG] Step 5 — Normalised {n_normalised} non-canonical "
                        f"`connect a::b to c::d;` → `connect a.b to c.d;`"
                    )
                step5 = dataclasses.replace(step5, extracted_sysml=normalised)

        # --- connect semantic validation ---
        if step5.extracted_sysml:
            suspicious = self._validate_connections(step5.extracted_sysml)
            if suspicious:
                metadata["suspicious_connections"] = suspicious
                if verbose:
                    print(f"\n  [DEBUG] ⚠ Suspicious connect statements ({len(suspicious)}):")
                    for s in suspicious:
                        print(f"      {s['source_port']} → {s['target_port']}: {s['warning']}")
        return step5

    def _multistep_generate(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        verbose: bool = False,
        platform_profile=None,
        contract_bundle=None,
        pattern_bindings=(),
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        4-step generation pipeline:
          1. Architecture Decomposition  (structured text plan)
          2. Part Definitions            (SysML structural fragment)
          3. Behavioral Model            (SysML behavioral fragment, FUNC/SAFE only)
          4. Integration / Assembly      (complete SysML package)

        Each step issues its own RAG query targeting the SysML constructs most
        relevant to that step, rather than sharing one generic upfront query.

        Returns (final_CoTResult, generation_metadata).
        final_CoTResult.extracted_sysml is the assembled model.
        generation_metadata carries per-step diagnostics.
        """
        metadata: Dict[str, Any] = {
            "generation_steps_completed": 0,
            "degraded_steps": [],
        }

        # Helper: fetch RAG context for a step and merge with any caller-supplied context.
        def _step_context(step: str) -> str:
            query = self._build_step_query(step, system_name, requirements)
            rag = self.get_augmented_context(
                query,
                include_official_sysml=True,
                allowed_extensions=(".sysml",),
            )
            if verbose:
                print(f"\n  [DEBUG] RAG query [{step}]: {query!r}")
            parts = [p for p in (context, rag) if p]
            return "\n\n".join(parts)

        # --- Step 1: Architecture Decomposition ---
        step1, architecture_text = self._step1_architecture(
            system_name, requirements, context, metadata, verbose
        )

        # --- Step 2: Part Definitions (structural fragment) ---
        step2, parts_fragment = self._step2_parts(
            system_name, architecture_text, requirements, _step_context,
            metadata, verbose, contract_bundle,
        )

        # --- Step 3: Interface & Flow Definitions (item def / typed port def) ---
        step3, interfaces_fragment = self._step3_interfaces(
            system_name, architecture_text, parts_fragment, requirements,
            _step_context, metadata, verbose, contract_bundle,
        )

        # --- Step 4: Behavioral Model (only if FUNC or SAFE requirements exist) ---
        step4, behavior_fragment = self._step4_behavior(
            system_name, architecture_text, parts_fragment, requirements,
            platform_profile, _step_context, metadata, verbose,
            contract_bundle, pattern_bindings,
        )

        # --- Step 5: Integration / Assembly ---
        # No separate RAG call for assembly — the prompt focuses on wiring together
        # the fragments already produced, not on new SysML constructs.
        assembly_guidance = ""
        if contract_bundle:
            from ..prototyping.safety_patterns import render_step_guidance
            assembly_guidance = render_step_guidance(
                contract_bundle, (), step="assembly",
                req_ids=[item.split(":", 1)[0] for item in requirements],
            )
        step5 = self.cot.assemble_model(
            system_name=system_name,
            parts_fragment=parts_fragment,
            interfaces_fragment=interfaces_fragment,
            behavior_fragment=behavior_fragment,
            requirements=requirements,
            semantic_guidance=assembly_guidance,
        )
        if assembly_guidance:
            metadata.setdefault("semantic_guidance_by_step", {})[
                "assembly"
            ] = assembly_guidance
        metadata["generation_steps_completed"] = 5
        metadata["total_thought_steps"] = (
            len(step1.thought_steps)
            + len(step2.thought_steps)
            + len(step3.thought_steps)
            + len(step4.thought_steps if step4 else [])
            + len(step5.thought_steps)
        )

        # --- Post-assembly: deterministic repair chain ---
        step5 = self._postprocess_assembly(
            step5, parts_fragment, interfaces_fragment, behavior_fragment,
            metadata, verbose,
        )

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 5 — Assembled SysML Model (final LLM output)")
            print(f"  {'─'*60}")
            if step5.extracted_sysml:
                print(step5.extracted_sysml)
            else:
                print("  ⚠ No ```sysml block extracted — raw answer:")
                print(step5.final_answer)

        return step5, metadata

    # ──────────────────────────────────────────────────────────────────────
    # Part-def preservation (Step 2 → Step 5 structural safety net)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _inject_missing_part_defs(
        cls,
        assembled: str,
        parts_fragment: str,
    ) -> Tuple[str, List[str]]:
        """Restore any Step-2 ``part def`` blocks omitted by Step 5.

        The structural fragment is the authoritative architecture produced by
        the dedicated part-generation call.  Assembly may enrich those blocks,
        but it must not silently delete them.  Only entirely missing named part
        definitions are injected; existing assembled definitions are untouched.
        """
        if not assembled or not parts_fragment:
            return assembled, []

        part_start_re = re.compile(r"\bpart\s+def\s+(\w+)\s*\{")
        extracted: List[Tuple[str, str]] = []
        cursor = 0
        while True:
            match = part_start_re.search(parts_fragment, cursor)
            if not match:
                break
            brace_pos = parts_fragment.index("{", match.start())
            end = find_block_end(parts_fragment, brace_pos)
            if end == -1:
                cursor = match.end()
                continue
            extracted.append((match.group(1), parts_fragment[match.start():end + 1]))
            cursor = end + 1

        missing = [
            (name, block)
            for name, block in extracted
            if not re.search(
                r"\bpart\s+def\s+" + re.escape(name) + r"\b",
                assembled,
            )
        ]
        if not missing:
            return assembled, []

        # SysML v2 permits both ordinary and quoted package names.  Vertex often
        # quotes human-readable names (e.g. ``package 'Drone System' {``), so
        # both forms must be valid deterministic injection points.
        package_match = re.search(
            r"\bpackage\s+(?:\w+|'[^']+')\s*\{",
            assembled,
        )
        if not package_match:
            return assembled, []

        injection = "\n    // (part defs restored from Step 2 by pipeline)\n"
        for _, block in missing:
            injection += "\n".join(
                "    " + line if line.strip() else line
                for line in block.splitlines()
            )
            injection += "\n\n"

        result = (
            assembled[:package_match.end()]
            + injection
            + assembled[package_match.end():]
        )
        return result, [name for name, _ in missing]

    # ──────────────────────────────────────────────────────────────────────
    # Programmatic state-def injection (Step 4 safety net)
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Connect semantic validation (Step 5 safety check)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_connections(
        assembled_sysml: str,
    ) -> List[Dict[str, str]]:
        """Check every connect statement for port-name semantic consistency.

        A connection is flagged as *suspicious* when the source port name and
        target port name share no domain tokens after stripping directional
        suffixes (``In`` / ``Out`` / ``Inout``).  False positives are possible
        for valid cross-domain connections (e.g. ``flightDataOut → releaseCmdIn``)
        so results are treated as warnings, not errors.

        Returns a list of dicts with keys: source_port, target_port, warning.
        """
        _DIR_SUFFIX = re.compile(r"(?:In|Out|Inout)$", re.IGNORECASE)
        # "status" kept intentionally: batteryStatusOut / battStatusIn share "status"
        # so removing it avoids false positives on batt ≠ battery abbreviation pairs.
        _STOP_TOKENS = {"data", "port", "signal", "link", "bus", "cmd",
                        "out", "in", "inout", "io"}

        def _domain_tokens(port_name: str) -> set:
            """Split camelCase/PascalCase port name → lowercase domain tokens."""
            # Strip directional suffix first
            stem = _DIR_SUFFIX.sub("", port_name)
            # Split on camelCase boundaries
            words = re.sub(r"([A-Z])", r" \1", stem).lower().split()
            return {w for w in words if len(w) > 2 and w not in _STOP_TOKENS}

        # Parse all connect statements.  SysML v2 uses dot notation only —
        # `connect partA.portA to partB.portB;`.  The `::` form is a deviation
        # (it is the namespace-qualified-name operator, not a connect endpoint
        # selector), so we do NOT accept it here; downstream syntax sanitisers
        # should normalise any stray `::` to `.` before this point.
        connect_re = re.compile(
            r"\bconnect\s+"
            r"(\w+)\.(\w+)\s+to\s+"
            r"(\w+)\.(\w+)\s*;",
            re.IGNORECASE,
        )
        suspicious: List[Dict[str, str]] = []

        # Collect all parsed connections for fan-in analysis
        connections: List[tuple] = []  # (src_part, src_port, tgt_part, tgt_port)
        for m in connect_re.finditer(assembled_sysml):
            src_part = m.group(1) or ""
            src_port = m.group(2) or ""
            tgt_part = m.group(3) or ""
            tgt_port = m.group(4) or ""
            if not src_port or not tgt_port:
                continue
            connections.append((src_part, src_port, tgt_part, tgt_port))

        # ── Semantic domain-token check ────────────────────────────────────
        for src_part, src_port, tgt_part, tgt_port in connections:
            src_tokens = _domain_tokens(src_port)
            tgt_tokens = _domain_tokens(tgt_port)
            # Only flag if BOTH ports have meaningful tokens and NO overlap
            if src_tokens and tgt_tokens and not (src_tokens & tgt_tokens):
                suspicious.append({
                    "source_port": f"{src_part}::{src_port}",
                    "target_port": f"{tgt_part}::{tgt_port}",
                    "warning": (
                        f"No shared domain tokens: "
                        f"{{{', '.join(sorted(src_tokens))}}} ↔ "
                        f"{{{', '.join(sorted(tgt_tokens))}}}"
                    ),
                })

        # ── Fan-in check: multiple sources → same (part, port) ────────────
        from collections import defaultdict
        target_map: Dict[str, List[str]] = defaultdict(list)
        for src_part, src_port, tgt_part, tgt_port in connections:
            tgt_key = f"{tgt_part}::{tgt_port}"
            target_map[tgt_key].append(f"{src_part}::{src_port}")

        for tgt_key, sources in target_map.items():
            if len(sources) > 1:
                suspicious.append({
                    "source_port": " + ".join(sources),
                    "target_port": tgt_key,
                    "warning": (
                        f"Fan-in: {len(sources)} sources connected to the same "
                        f"input port — route through an aggregator instead"
                    ),
                })

        return suspicious

    @classmethod
    def _inject_missing_state_defs(
        cls,
        assembled: str,
        behavior_fragment: str,
    ) -> Tuple[str, List[str]]:
        """Check whether any state defs from the behavioral fragment were dropped by
        the Step 4 LLM and, if so, inject them into their owning part def.

        The behavioral fragment uses ``// OWNER: <PartName>`` comments to annotate
        ownership.  For state defs without that annotation we fall back to a
        heuristic: state defs whose name contains "Safety", "Fault", "Startup",
        "Batt", "Comm", "Impact", or "Sep" are assigned to the first part whose
        name contains "Safety" or "Monitor" or "Fault".

        Returns:
            (possibly_modified_assembled, list_of_injected_state_def_names)
        """
        # ── 1. Extract state def blocks from behavior_fragment ──────────────
        # Pattern: optional "// OWNER: X" line, then "state def Name { ... }"
        owner_re = re.compile(r"//\s*OWNER:\s*(\w+)", re.IGNORECASE)
        state_start_re = re.compile(r"\bstate\s+def\s+(\w+)\s*\{")

        # Walk behavior_fragment, collecting (owner, state_def_name, full_block)
        behavior_state_defs: List[Tuple[Optional[str], str, str]] = []
        i = 0
        last_owner: Optional[str] = None
        while i < len(behavior_fragment):
            # Check for OWNER comment
            m_owner = owner_re.match(behavior_fragment, i)
            if m_owner:
                last_owner = m_owner.group(1)
                i = m_owner.end()
                continue

            # Check for state def
            m_state = state_start_re.match(behavior_fragment, i)
            if m_state:
                name = m_state.group(1)
                brace_pos = behavior_fragment.index("{", m_state.start())
                end = find_block_end(behavior_fragment, brace_pos)
                if end != -1:
                    block = behavior_fragment[m_state.start(): end + 1]
                    behavior_state_defs.append((last_owner, name, block))
                    i = end + 1
                    last_owner = None  # consumed
                    continue

            # Reset owner tracking when a blank line or other content appears
            if behavior_fragment[i] == "\n":
                # Only reset owner if we've moved past whitespace without hitting a state def
                pass
            i += 1

        if not behavior_state_defs:
            return assembled, []

        # ── 2. Identify which state defs are missing from assembled ─────────
        _SAFETY_HEURISTIC = re.compile(
            r"Safety|Fault|Startup|Batt|Comm|Impact|Sep|Landing|Separation",
            re.IGNORECASE,
        )

        # Find part def name that looks like a safety/monitor part (heuristic fallback)
        _monitor_re = re.compile(
            r"\bpart\s+def\s+(\w*(?:Safety|Monitor|Fault|Health)\w*)\s*\{",
            re.IGNORECASE,
        )
        monitor_match = _monitor_re.search(assembled)
        default_safety_part = monitor_match.group(1) if monitor_match else None

        injected_names: List[str] = []
        result = assembled

        for owner, name, block in behavior_state_defs:
            # Check if this state def already appears in the assembled model
            if re.search(r"\bstate\s+def\s+" + re.escape(name) + r"\b", result):
                continue  # already present — nothing to do

            # Determine target part
            target_part = owner
            if target_part is None:
                if _SAFETY_HEURISTIC.search(name):
                    target_part = default_safety_part
            if target_part is None:
                continue  # cannot determine owner — skip

            # Find the part def block for target_part in assembled
            part_def_re = re.compile(
                r"\bpart\s+def\s+" + re.escape(target_part) + r"\s*\{"
            )
            m_part = part_def_re.search(result)
            if not m_part:
                continue  # part not found — skip

            brace_open = result.index("{", m_part.start())
            closing = find_block_end(result, brace_open)
            if closing == -1:
                continue

            # Indent the block by 4 spaces and inject before the closing }
            indented = "\n".join(
                "        " + line if line.strip() else line
                for line in block.splitlines()
            )
            result = (
                result[:closing]
                + "\n        // (injected by pipeline)\n"
                + indented
                + "\n    "
                + result[closing:]
            )
            injected_names.append(name)

        return result, injected_names

    # ──────────────────────────────────────────────────────────────────────
    # Item-def / port-def injection (Step 3 safety net)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _inject_missing_item_defs(
        cls,
        assembled: str,
        interfaces_fragment: str,
    ) -> Tuple[str, List[str]]:
        """Check whether any item defs / typed port defs from the interfaces
        fragment were dropped by the Step 5 LLM and, if so, inject them at the
        package level.

        Item defs and port defs are package-level declarations and must appear
        at the top of the package body, before any ``part def`` blocks.  The
        injection point is right after the ``package Name {`` opening brace.

        Returns:
            (possibly_modified_assembled, list_of_injected_def_labels)
        """
        if not interfaces_fragment or not assembled:
            return assembled, []

        # ── 1. Extract item def / port def blocks from interfaces_fragment ─
        # Handles both `item def Name { ... }` and `port def Name { ... }`.
        # Semi-colon form (`item def Name;`) is also captured.
        def_start_re = re.compile(r"\b(item|port)\s+def\s+(\w+)\s*([{;])")

        extracted: List[Tuple[str, str, str]] = []  # (kind, name, full_block)
        i = 0
        while i < len(interfaces_fragment):
            m = def_start_re.search(interfaces_fragment, i)
            if not m:
                break
            kind = m.group(1)   # "item" or "port"
            name = m.group(2)
            sentinel = m.group(3)

            if sentinel == ";":
                block = interfaces_fragment[m.start():m.end()]
                extracted.append((kind, name, block))
                i = m.end()
            else:  # "{"
                brace_pos = interfaces_fragment.index("{", m.start())
                end = find_block_end(interfaces_fragment, brace_pos)
                if end != -1:
                    block = interfaces_fragment[m.start():end + 1]
                    extracted.append((kind, name, block))
                    i = end + 1
                else:
                    i = m.end()

        if not extracted:
            return assembled, []

        # ── 2. Identify which defs are absent from the assembled text ───────
        to_inject: List[str] = []
        injected_labels: List[str] = []

        for kind, name, block in extracted:
            pattern = rf"\b{re.escape(kind)}\s+def\s+{re.escape(name)}\b"
            if re.search(pattern, assembled):
                continue  # already present — nothing to do
            to_inject.append(block)
            injected_labels.append(f"{kind} def {name}")

        if not to_inject:
            return assembled, []

        # ── 3. Find injection point: right after `package Name {` ───────────
        pkg_open_re = re.compile(r"\bpackage\s+(?:\w+|'[^']+')\s*\{")
        m_pkg = pkg_open_re.search(assembled)
        inject_pos = m_pkg.end() if m_pkg else 0

        # ── 4. Build indented injection block and splice in ──────────────────
        injection = "\n    // (item defs / port defs injected by pipeline)\n"
        for block in to_inject:
            indented = "\n".join(
                "    " + line if line.strip() else line
                for line in block.splitlines()
            )
            injection += indented + "\n\n"

        result = assembled[:inject_pos] + injection + assembled[inject_pos:]
        return result, injected_labels

    # ──────────────────────────────────────────────────────────────────────
    # doc = "string" → doc /* string */ syntax normaliser
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_connect_syntax(sysml_text: str) -> Tuple[str, int]:
        """Convert `connect a::b to c::d;` → `connect a.b to c.d;`.

        SysML v2 uses dot notation for connect endpoints (verified against the
        official SysML-v2-release-src/examples corpus).  The `::` operator is
        for namespace-qualified names (`Package::Element`), not feature access
        in connect statements.  LLMs sometimes emit the `::` form anyway;
        normalising here ensures every downstream consumer sees the canonical
        SysML v2 syntax.

        Only `::` occurrences inside `connect ... to ...;` are touched — any
        other use (e.g. `Package::Element` qualified names) is preserved.

        Returns (normalised_text, count_of_substitutions).
        """
        # Match a complete connect statement that uses `::` on either side.
        # The capture groups isolate part / port pieces so we can rewrite with `.`.
        connect_pat = re.compile(
            r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)::(\w+)\s*;",
            re.IGNORECASE,
        )

        # Also handle the asymmetric forms (one side `::`, the other `.`).
        connect_mixed_left = re.compile(
            r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)\.(\w+)\s*;",
            re.IGNORECASE,
        )
        connect_mixed_right = re.compile(
            r"\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)::(\w+)\s*;",
            re.IGNORECASE,
        )

        count = 0

        def _rewrite(m: re.Match) -> str:  # type: ignore[type-arg]
            return f"connect {m.group(1)}.{m.group(2)} to {m.group(3)}.{m.group(4)};"

        for pat in (connect_pat, connect_mixed_left, connect_mixed_right):
            n = len(pat.findall(sysml_text))
            if n:
                sysml_text = pat.sub(_rewrite, sysml_text)
                count += n

        return sysml_text, count

    @staticmethod
    def _fix_doc_syntax(sysml_text: str) -> Tuple[str, int]:
        """Convert invalid ``doc = "string";`` to valid ``doc /* string */``.

        The correct SysML v2 doc-comment syntax is ``doc /* text */``.
        LLMs sometimes generate ``doc = "text";`` (or ``doc = "text"``) which
        is not valid SysML v2 — it is parsed by Syside as a feature-usage
        named ``doc`` of type String, causing round-trip serialization issues
        (the feature leaks into ``top_level_usages`` as a spurious
        ``requirement req : String = "..."`` line).

        Returns:
            (fixed_text, count_of_substitutions)
        """
        # Match: optional leading whitespace, `doc`, optional whitespace,
        # `=`, optional whitespace, a double-quoted string, optional `;`
        doc_eq_re = re.compile(
            r'\bdoc\s*=\s*"((?:[^"\\]|\\.)*)"[ \t]*;?',
        )
        count = len(doc_eq_re.findall(sysml_text))
        fixed = doc_eq_re.sub(lambda m: f'doc /* {m.group(1)} */', sysml_text)
        return fixed, count

    # ──────────────────────────────────────────────────────────────────────
    # Invalid requirement-attribute cleanup (post-assembly sanitiser)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_invalid_requirement_attrs(sysml_text: str) -> Tuple[str, int]:
        """Remove invalid ``requirement <name> : <Type> = "...";`` lines.

        These are not valid SysML v2.  The refinement LLM sometimes produces
        them when it tries to "document" a requirement inline — but the correct
        construct is ``satisfy requirement REQ_ID;`` inside a part def, or
        ``requirement def REQ_ID { doc /* ... */ }`` at package level.

        Only lines that match the specific bogus pattern are removed; all valid
        ``requirement def`` and ``satisfy requirement`` constructs are preserved.

        Returns:
            (cleaned_text, count_of_removed_lines)
        """
        # Matches: optional leading whitespace, `requirement`, identifier,
        # colon, type name, equals, a double-quoted string, semicolon.
        # Does NOT match `requirement def` (the `def` keyword breaks the pattern).
        invalid_re = re.compile(
            r"^[ \t]*requirement[ \t]+(?!def\b)\w+[ \t]*:[ \t]*\w+[ \t]*"
            r"=[ \t]*\"[^\"]*\"[ \t]*;[ \t]*$",
            re.MULTILINE,
        )
        matches = invalid_re.findall(sysml_text)
        cleaned = invalid_re.sub("", sysml_text)
        return cleaned, len(matches)

    @staticmethod
    def _fix_capability_semantics(
        sysml_text: str, requirements: List[str]
    ) -> Tuple[str, int]:
        """Repair range-floor naming and remove invalid always-on invariants.

        Operational range is a mission-end capability.  When the structured
        requirement is a lower bound, ``currentRange <= maxRange`` verifies the
        opposite property, while ``currentRange >= minRange`` is false at
        startup.  Keep the design target as ``min*Range`` and leave evaluation
        to the forward-flight fidelity tier.
        """
        try:
            from ..dse.requirement_spec import RANGE, extract_requirements
            specs = extract_requirements(requirements)
            has_floor = any(s.quantity == RANGE and s.operator == ">=" for s in specs)
            has_ceiling = any(s.quantity == RANGE and s.operator == "<=" for s in specs)
        except Exception:
            return sysml_text, 0
        if not has_floor or has_ceiling:
            return sysml_text, 0

        fixes = 0

        def _rename(match: re.Match) -> str:
            nonlocal fixes
            fixes += 1
            token = match.group(0)
            return ("min" if token.startswith("max") else "Min") + token[3:]

        result = re.sub(
            r"\b(?:max|Max)(?:Operational)?Range\b",
            _rename,
            sysml_text,
        )

        constraint_re = re.compile(
            r"(?ms)^(?P<indent>[ \t]*)assert\s+constraint\s+\w+\s*\{"
            r"(?P<body>[^{}]*(?:current\w*Range|distance\w*)[^{}]*)\}\s*"
        )

        def _drop_constraint(match: re.Match) -> str:
            nonlocal fixes
            body = match.group("body")
            is_mission_range = re.search(
                r"\b(?:current(?:Operational)?Range|distanceTravelled)\b",
                body,
                re.IGNORECASE,
            ) and re.search(
                r"\b(?:min|max)(?:Operational)?Range\b",
                body,
                re.IGNORECASE,
            )
            if not is_mission_range:
                return match.group(0)
            fixes += 1
            return (
                f"{match.group('indent')}// Operational range is a mission-end "
                "capability evaluated by forward-flight fidelity, not an invariant.\n"
            )

        return constraint_re.sub(_drop_constraint, result), fixes

    @staticmethod
    def _fix_safety_action_semantics(sysml_text: str) -> Tuple[str, int]:
        """Prevent a parachute action from sending a flight-mode LAND command."""
        action_re = re.compile(
            r"(?P<head>action\s+def\s+\w*(?:parachute|chute)\w*\s*\{)"
            r"(?P<body>[^{}]*)(?P<tail>\})",
            re.IGNORECASE,
        )
        fixes = 0

        def _fix_action(match: re.Match) -> str:
            nonlocal fixes
            body, n = re.subn(
                r"\bsend\s+CMD_(?:LAND|RTL|AUTO|GUIDED|LOITER|POSHOLD)\s*\(\)",
                "send CMD_PARACHUTE()",
                match.group("body"),
                flags=re.IGNORECASE,
            )
            fixes += n
            return match.group("head") + body + match.group("tail")

        result = action_re.sub(_fix_action, sysml_text)
        if fixes and not re.search(r"\baction\s+def\s+CMD_PARACHUTE\b", result):
            first_part = re.search(r"(?m)^[ \t]*part\s+def\s+", result)
            if first_part:
                result = (
                    result[:first_part.start()]
                    + "    action def CMD_PARACHUTE { }\n\n"
                    + result[first_part.start():]
                )
                fixes += 1
        return result, fixes

    @staticmethod
    def _fix_self_test_behavior_semantics(
        sysml_text: str,
        requirements: List[str],
    ) -> Tuple[str, int]:
        """Make a generated self-test phase produce an executable response.

        A bare ``state PhaseSelfTest;`` proves only that a phase name exists.
        When a FUNC requirement explicitly mandates an automated self-test,
        attach an entry action to that already-generated state. Existing
        self-test actions are reused; a minimal declaration is added only when
        the model has none.
        """
        if not any(
            "func" in req.lower()
            and any(k in req.lower() for k in (
                "self-test", "self test", "self-check", "self check"
            ))
            for req in requirements
        ):
            return sysml_text, 0

        part_start_re = re.compile(r"\bpart\s+def\s+(\w+)\s*\{")
        state_re = re.compile(
            r"\bstate\s+(\w*(?:SelfTest|SelfCheck)\w*)\s*;",
            re.IGNORECASE,
        )
        result = sysml_text
        cursor = 0
        while True:
            part_match = part_start_re.search(result, cursor)
            if not part_match:
                return result, 0
            brace_pos = result.index("{", part_match.start())
            part_end = find_block_end(result, brace_pos)
            if part_end == -1:
                cursor = part_match.end()
                continue
            block = result[part_match.start():part_end + 1]
            state_match = state_re.search(block)
            if not state_match:
                cursor = part_end + 1
                continue

            action_match = re.search(
                r"\baction\s+def\s+(\w*(?:SelfTest|SelfCheck)\w*)\b",
                block,
                re.IGNORECASE,
            )
            action_name = (
                action_match.group(1) if action_match
                else "performAutomatedSelfTest"
            )
            state_name = state_match.group(1)
            absolute_start = part_match.start() + state_match.start()
            absolute_end = part_match.start() + state_match.end()
            replacement = (
                f"state {state_name} {{\n"
                f"                entry action runSelfTest : {action_name};\n"
                "            }"
            )
            result = result[:absolute_start] + replacement + result[absolute_end:]
            fixes = 1

            if action_match is None:
                # Re-find the owner block after the state expansion and add the
                # declaration immediately inside it.
                owner_match = re.search(
                    r"\bpart\s+def\s+" + re.escape(part_match.group(1)) + r"\s*\{",
                    result,
                )
                if owner_match:
                    owner_open = result.index("{", owner_match.start())
                    result = (
                        result[:owner_open + 1]
                        + f"\n        action def {action_name} {{ }}\n"
                        + result[owner_open + 1:]
                    )
                    fixes += 1
            return result, fixes

    @staticmethod
    def _fix_functional_satisfy_ownership(
        sysml_text: str,
        requirements: List[str],
    ) -> Tuple[str, int]:
        """Align sequencing FUNC satisfy links with their state-machine owner.

        The integration LLM occasionally places a system-level satisfy link on
        a monitoring part even though the dedicated executable state machine is
        owned by another part.  The verification matrix then correctly refuses
        to credit that unrelated owner's behavior, and a later LLM closure tends
        to add a duplicate machine that regresses simulation.  For narrowly
        recognisable sequencing families, relocate the existing satisfy usage to
        the part that already owns the matching state machine.  No requirement
        definition or behavior is invented.
        """
        family_patterns = (
            (
                ("self-test", "self test", "self-check", "self check"),
                re.compile(
                    r"\bstate(?:\s+def)?\s+\w*(?:SelfTest|SelfCheck)\w*\b",
                    re.IGNORECASE,
                ),
            ),
            (
                ("health report", "post-flight", "post flight"),
                re.compile(
                    r"\bstate\s+def\s+\w*(?:HealthReport|Reporting)\w*\b",
                    re.IGNORECASE,
                ),
            ),
            (
                ("waypoint-modification", "waypoint modification", "revised waypoint"),
                re.compile(
                    r"\bstate\s+def\s+\w*(?:WaypointRevision|WaypointUpdate)\w*\b",
                    re.IGNORECASE,
                ),
            ),
        )

        req_families: List[Tuple[str, re.Pattern]] = []
        for requirement in requirements:
            match = re.match(r"(REQ[-_]FUNC[-_]\d+)\s*:\s*(.*)", requirement)
            if not match:
                continue
            req_id = match.group(1).replace("-", "_")
            body_low = match.group(2).lower()
            for keywords, state_pattern in family_patterns:
                if any(keyword in body_low for keyword in keywords):
                    req_families.append((req_id, state_pattern))
                    break

        result = sysml_text
        fixes = 0
        part_start_re = re.compile(r"\bpart\s+def\s+(\w+)\s*\{")

        for req_id, state_pattern in req_families:
            target_name: Optional[str] = None
            cursor = 0
            while True:
                part_match = part_start_re.search(result, cursor)
                if not part_match:
                    break
                brace_pos = result.index("{", part_match.start())
                end = find_block_end(result, brace_pos)
                if end == -1:
                    cursor = part_match.end()
                    continue
                block = result[part_match.start():end + 1]
                if state_pattern.search(block):
                    target_name = part_match.group(1)
                    break
                cursor = end + 1

            if target_name is None:
                continue

            satisfy_re = re.compile(
                r"(?m)^[ \t]*satisfy\s+(?:requirement\s+)?"
                + re.escape(req_id)
                + r"\s*;[ \t]*\n?",
                re.IGNORECASE,
            )

            # If the only occurrence is already in the correct block, preserve
            # the model byte-for-byte. Otherwise relocate to keep exactly one
            # satisfy usage, as required by the integration contract.
            target_match = re.search(
                r"\bpart\s+def\s+" + re.escape(target_name) + r"\s*\{",
                result,
            )
            if not target_match:
                continue
            target_end = find_block_end(result, result.index("{", target_match.start()))
            target_block = result[target_match.start():target_end + 1]
            occurrences = list(satisfy_re.finditer(result))
            if len(occurrences) == 1 and satisfy_re.search(target_block):
                continue

            result = satisfy_re.sub("", result)
            target_match = re.search(
                r"\bpart\s+def\s+" + re.escape(target_name) + r"\s*\{",
                result,
            )
            if not target_match:
                continue
            brace_pos = result.index("{", target_match.start())
            target_end = find_block_end(result, brace_pos)
            if target_end == -1:
                continue
            result = (
                result[:target_end]
                + f"\n        satisfy requirement {req_id};\n    "
                + result[target_end:]
            )
            fixes += 1

        return result, fixes

    # ──────────────────────────────────────────────────────────────────────
    # RAG query builders
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_step_query(
        step: str,
        system_name: str,
        requirements: List[str],
    ) -> str:
        """Build a RAG query tailored to the knowledge needed by each generation step.

        Step 1 (architecture): domain decomposition — what components does this
            type of system typically have?
        Step 2 (parts): structural SysML constructs — part def, port, attribute.
        Step 3 (behavior): behavioral SysML constructs — state def, fault
            transition, action def.  Queries are further shaped by the SAFE and
            FUNC requirement bodies so the retriever finds fault-handling examples.
        """
        def _req_keywords(reqs: List[str], n: int = 3) -> str:
            """Extract meaningful words from requirement bodies (skip stop words)."""
            _STOP = {"the", "shall", "must", "will", "system", "that", "with",
                     "from", "into", "when", "than", "this", "have", "been"}
            words: List[str] = []
            for r in reqs[:n]:
                body = r.split(":", 1)[-1].strip() if ":" in r else r
                for w in re.findall(r"[A-Za-z]{4,}", body):
                    if w.lower() not in _STOP:
                        words.append(w)
            # Deduplicate while preserving order
            seen: set = set()
            unique = []
            for w in words:
                lw = w.lower()
                if lw not in seen:
                    seen.add(lw)
                    unique.append(w)
            return " ".join(unique[:12])

        safe_reqs = [r for r in requirements if "-SAFE-" in r]
        func_reqs = [r for r in requirements if "-FUNC-" in r]
        intf_reqs = [r for r in requirements if "-INTF-" in r]
        perf_reqs = [r for r in requirements if "-PERF-" in r]

        if step == "architecture":
            # NOTE: Step 1 no longer triggers RAG retrieval (see
            # `_multistep_generate`).  This branch is preserved for backward
            # compatibility with external callers / tests that may still invoke
            # `_build_step_query("architecture", ...)` directly, but it is dead
            # code on the main pipeline path.
            func_kw = _req_keywords(func_reqs)
            return (
                f"{system_name} system architecture decomposition components subsystems "
                f"{func_kw}"
            ).strip()

        if step == "parts":
            # Structural SysML constructs shaped by PERF and INTF requirements.
            # PERF → numeric attributes with SI units.
            # INTF → port def with direction, connect statement.
            perf_kw = _req_keywords(perf_reqs)
            intf_kw = _req_keywords(intf_reqs)
            return (
                f"part def port attribute direction numeric value unit "
                f"{perf_kw} {intf_kw} {system_name}"
            ).strip()

        if step == "interfaces":
            # Interface/flow SysML constructs shaped by INTF requirements.
            # INTF → item def (payload type) + typed port def.
            intf_kw = _req_keywords(intf_reqs)
            return (
                f"item def flow port def typed signal protocol interface "
                f"{intf_kw} {system_name}"
            ).strip()

        if step == "behavior":
            # Behavioral SysML constructs shaped by SAFE, FUNC, and OPER requirements.
            # SAFE → state def with fault-entry transition + emergency action.
            # FUNC → action def capturing the functional behaviour.
            # OPER → enum def + mode machine state def with enum-equality transitions.
            oper_reqs = [r for r in requirements if "-OPER-" in r]
            safe_kw = _req_keywords(safe_reqs)
            func_kw = _req_keywords(func_reqs, n=2)
            if oper_reqs:
                oper_kw = _req_keywords(oper_reqs, n=1)
                return (
                    f"enum def mode machine operational phase transition state def "
                    f"fault entry emergency action "
                    f"{oper_kw} {safe_kw} {func_kw}"
                ).strip()
            return (
                f"state def fault entry transition emergency action "
                f"{safe_kw} {func_kw}"
            ).strip()

        # Fallback (e.g. "assembly" step — no dedicated RAG call, but keeps the
        # helper usable if called externally).
        return f"{system_name} SysML v2 satisfy connect package"

    @staticmethod
    def _build_refinement_query(issues: List[str]) -> str:
        """Build a RAG query that targets the SysML constructs needed to fix
        the specific issues reported by the evaluator.

        The query is assembled by matching known issue patterns to SysML
        vocabulary, so the retriever returns relevant syntax examples rather
        than generic system-name hits.
        """
        if not issues:
            return "SysML v2 satisfy requirement traceability"

        issues_lower = " ".join(issues).lower()
        terms: List[str] = []

        if any(kw in issues_lower for kw in ("safe", "fault", "emergency", "shutdown")):
            terms.append("state def fault entry transition emergency action")
        if any(kw in issues_lower for kw in ("port", "direction", "undirected")):
            terms.append("port def direction in out inout")
        if any(kw in issues_lower for kw in ("satisfy", "untraced", "traceab")):
            terms.append("satisfy requirement traceability link")
        if any(kw in issues_lower for kw in ("attribute", "numeric", "value", "unit")):
            terms.append("attribute numeric value unit constraint")
        if any(kw in issues_lower for kw in ("connect", "interface", "consistency")):
            terms.append("connect port interface")

        # If no pattern matched, use a generic improvement query
        if not terms:
            terms.append("SysML v2 refine design improve quality")

        return " ".join(terms)

    def _apply_requirement_traceability(
        self, model: SysMLModel, requirements: List[str]
    ) -> List[str]:
        """
        Ensure each requirement is linked to the best-matching component.

        Returns requirement IDs that remain untraced. ``SysMLLiteModel`` keeps
        the emitted SysML text as its source of truth, so missing links are never
        fabricated in its extracted in-memory cache; they must be repaired by a
        subsequent text-producing refinement. The legacy mutable ``SysMLModel``
        path may still add a strongly matched relationship because it serializes
        that relationship back into the model text.
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

            # Tightened threshold (was: best_score == 0).  A 1-token overlap is
            # a coincidence (e.g., the word "system" matching everywhere), not a
            # real semantic match — don't manufacture a satisfy link from it,
            # because that inflates the requirement_satisfaction dimension.
            # Require ≥ 2 shared domain tokens, OR a tie-breaking margin of 2
            # over the second-best part.
            second_best_score = (
                _score(req_tokens, scored[1]) if len(scored) > 1 else 0
            )
            strong_match = best_score >= 2
            unambiguous = (best_score - second_best_score) >= 2

            if best_score == 0 or not (strong_match or unambiguous):
                # Either no match, or only a weak coincidental overlap.
                # Record as untraced; the refinement loop will surface this so
                # the LLM can add an explicit satisfy link in the right part.
                untraced.append(req_id)
                continue

            if isinstance(model, SysMLLiteModel):
                # LitePartDef intentionally has no mutating add_satisfy API.
                # Updating only its extracted cache would make the evaluator see
                # a relationship absent from model.to_sysml_text(), creating
                # false traceability that disappears in post-hoc evaluation.
                untraced.append(req_id)
                continue

            best.add_satisfy(SatisfyRelationship(
                source=best.to_ref(),
                target=ElementRef(name=req_id),
            ))
            satisfied_ids.add(req_id)

        return untraced
