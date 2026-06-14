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

        is_refinement = bool(existing_model and feedback)
        skip_rag = task.get("skip_rag", False)

        if is_refinement:
            # Syntax-fix calls skip RAG entirely — the corpus has no useful examples
            # for "fix undefined feature reference" errors, and the generic fallback
            # query ("SysML v2 refine design improve quality") adds noise.
            if skip_rag:
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

            generation_metadata = {}
        else:
            # Generation mode — multi-step pipeline
            cot_result, generation_metadata = self._multistep_generate(
                system_name=system_name,
                requirements=requirements,
                context=context,
                verbose=verbose,
                platform_profile=task.get("platform_profile"),
            )

        if not cot_result.extracted_sysml:
            raise RuntimeError("[SysML_EXTRACTION_ERROR] 未提取到SysML v2 design.")

        parse_label = "After Refinement" if is_refinement else "After Initial Generation"

        # Parse via syside native API (replaces Syside_AST_Parser)
        model = build_lite_model(cot_result.extracted_sysml, model_name=system_name)

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

    def _multistep_generate(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        verbose: bool = False,
        platform_profile=None,
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

        # --- Step 1: Architecture Decomposition (RAG intentionally skipped) ---
        # Rationale: the RAG corpus is exclusively `.sysml` grammar examples,
        # and `RAGRetriever` wraps every retrieved snippet in ```sysml fences
        # before prepending it to the prompt.  That directly contradicts this
        # step's "plain structured text only — no SysML" instruction:
        #
        #   1. The retrieval is the wrong knowledge type for decomposition
        #      planning (we want domain subsystem patterns, not SysML syntax).
        #   2. The fenced examples prime the LLM to emit ```sysml code blocks,
        #      which then have to be fence-stripped post-hoc.
        #   3. The token cost is wasted — Step 2/3/4 already retrieve targeted
        #      SysML examples for their respective generation tasks.
        #
        # Domain decomposition draws on the LLM's pretrained knowledge of
        # cyber-physical system architecture instead.  The fence-strip guard
        # below is retained as defence-in-depth.
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

        # --- Step 2: Part Definitions (structural fragment) ---
        ctx2 = _step_context("parts")
        step2 = self.cot.generate_part_definitions(
            system_name=system_name,
            architecture=architecture_text,
            requirements=requirements,
            context=ctx2,
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

        # --- Step 3: Interface & Flow Definitions (item def / typed port def) ---
        intf_reqs = [r for r in requirements if "-INTF-" in r]
        ctx3 = _step_context("interfaces")
        step3 = self.cot.generate_interfaces_and_flows(
            system_name=system_name,
            architecture=architecture_text,
            parts_fragment=parts_fragment,
            intf_requirements=intf_reqs,
            context=ctx3,
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

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Step 3 — Interface & Flow Definitions (SysML fragment)")
            print(f"  {'─'*60}")
            if interfaces_fragment:
                print(interfaces_fragment)
            else:
                print("  (skipped — no code block extracted)")

        # --- Step 4: Behavioral Model (only if FUNC or SAFE requirements exist) ---
        behavioral_reqs = [
            r for r in requirements
            if any(f"-{cat}-" in r for cat in self._BEHAVIORAL_CATEGORIES)
        ]
        if behavioral_reqs:
            ctx4 = _step_context("behavior")
            step4 = self.cot.generate_behavior(
                system_name=system_name,
                architecture=architecture_text,
                behavioral_requirements=behavioral_reqs,
                parts_fragment=parts_fragment,
                context=ctx4,
                platform_profile=platform_profile,
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

            if verbose:
                print(f"\n  {'─'*60}")
                print(f"  [DEBUG] Step 4 — Behavioral Model (SysML fragment)")
                print(f"  {'─'*60}")
                print(behavior_fragment)
        else:
            behavior_fragment = ""
            step4 = None
            metadata["generation_steps_completed"] = 4
            metadata["behavior_fragment_length"] = 0

            if verbose:
                print(f"\n  [DEBUG] Step 4 — Behavioral Model: skipped "
                      f"(no FUNC/SAFE requirements)")

        # --- Step 5: Integration / Assembly ---
        # No separate RAG call for assembly — the prompt focuses on wiring together
        # the fragments already produced, not on new SysML constructs.
        step5 = self.cot.assemble_model(
            system_name=system_name,
            parts_fragment=parts_fragment,
            interfaces_fragment=interfaces_fragment,
            behavior_fragment=behavior_fragment,
            requirements=requirements,
        )
        metadata["generation_steps_completed"] = 5
        metadata["total_thought_steps"] = (
            len(step1.thought_steps)
            + len(step2.thought_steps)
            + len(step3.thought_steps)
            + len(step4.thought_steps if step4 else [])
            + len(step5.thought_steps)
        )

        # --- Post-assembly: fix invalid `doc = "string";` → `doc /* string */` ---
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

        # --- Post-assembly: inject any state defs the LLM dropped ---
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

        # --- Post-assembly: inject any item defs / typed port defs the LLM dropped ---
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

        # --- Post-assembly: strip invalid `requirement <name> : <Type> = "...";` lines ---
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

        # --- Post-assembly: normalise `connect a::b to c::d;` → `connect a.b to c.d;` ---
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

        # --- Post-assembly: connect semantic validation ---
        if step5.extracted_sysml:
            suspicious = self._validate_connections(step5.extracted_sysml)
            if suspicious:
                metadata["suspicious_connections"] = suspicious
                if verbose:
                    print(f"\n  [DEBUG] ⚠ Suspicious connect statements ({len(suspicious)}):")
                    for s in suspicious:
                        print(f"      {s['source_port']} → {s['target_port']}: {s['warning']}")

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
        pkg_open_re = re.compile(r"\bpackage\s+\w+\s*\{")
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

            best.add_satisfy(SatisfyRelationship(
                source=best.to_ref(),
                target=ElementRef(name=req_id),
            ))
            satisfied_ids.add(req_id)

        return untraced
