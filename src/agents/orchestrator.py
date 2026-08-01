"""
Multi-agent Orchestrator for MBSE prototyping.

Coordinates the different specialized agents to implement the
full rapid prototyping pipeline, combining forward and backward
inference with design space exploration.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .design_agent import DEFAULT_MAXIMUM_PLAN_ATTEMPTS, DesignAgent
from .requirements_agent import RequirementsAgent
from ..dse.design_space import DesignConfiguration, DesignParameter, DesignSpace, ParameterType
from ..dse.evaluator import DesignEvaluator
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import ElementRef, SysMLModel
from ..sysml.lite_model import SysMLLiteModel, build_lite_model

# Accept both model types wherever SysMLModel is checked
_SysMLModelTypes = (SysMLModel, SysMLLiteModel)
from ..simulation.validator import SimulationValidator, SimulationResult
from ..simulation.syntax_checker import check_syntax, SyntaxCheckResult
from ..simulation.levenshtein_fixer import (
    try_fix_sema_errors,
    format_hints_for_llm,
)
from ..simulation.error_localizer import (
    extract_error_context,
    merge_fixed_chunk,
    build_fix_prompt,
    strip_code_fences,
)
from ..simulation.connectivity_fixer import (
    build_port_directory,
    parse_connects,
    validate_connects,
    merge_connects,
    build_connectivity_prompt,
    extract_connect_lines,
)
from ..simulation.transition_fixer import (
    build_state_machine_summary,
    build_transition_prompt,
    extract_transition_lines,
    validate_transitions,
    merge_transitions,
)
from ..simulation.port_fixer import (
    collect_port_defs,
    build_port_fix_prompt,
    extract_port_additions,
    validate_port_additions,
    merge_port_additions,
)
from ..simulation.connect_auditor import audit_connects, AuditResult
from ..utils.sysml_text_utils import find_block_end, get_sysml_text
from .dse_injectors import (
    apply_best_config_to_model as _apply_best_config_to_model,
    apply_inject_attrs_to_sysml_text as _apply_inject_attrs_to_sysml_text,
    build_dse_design_constraints as _build_dse_design_constraints,
)

# System prompt for surgical LLM syntax fixes
_SURGICAL_FIX_SYSTEM = (
    "You are a SysML v2 syntax expert. "
    "Fix only the specified errors in the given code block. "
    "Return only the corrected code — no explanations, no markdown fences."
)

# System prompt for surgical connectivity fixes (connect statements only)
_CONNECTIVITY_FIX_SYSTEM = (
    "You are a SysML v2 connectivity expert. "
    "You add ONLY `connect a.x to b.y;` statements using existing ports. "
    "You never invent ports and never output anything but connect statements."
)


def _public_realization(realization):
    if not realization:
        return None
    return {k: v for k, v in realization.items() if not k.startswith("_")}

# System prompt for surgical transition source fixes (transition statements only)
_TRANSITION_FIX_SYSTEM = (
    "You are a SysML v2 state machine expert. "
    "You fix transition source states so a mode machine can traverse its full chain. "
    "You return ONLY corrected `transition ... ;` statements — never any other text."
)

# System prompt for surgical port additions (port declarations only)
_PORT_FIX_SYSTEM = (
    "You are a SysML v2 port architect. "
    "You add the minimum set of port declarations so unreachable signal paths "
    "become connectable. You return ONLY lines in the form "
    "`<PartDefName>: <direction> port <portName> : <PortType>;` — never any other text."
)

# Domain-detection keyword sets (ordered most-specific first)
_DRONE_KWS      = {"drone", "uav", "aerial", "quadcopter", "rotor", "flight", "autopilot"}
_INDUSTRIAL_KWS = {"factory", "plc", "industrial", "cnc", "conveyor", "scada", "fieldbus"}
_ROBOT_KWS      = {"ros2", "ros ", "manipulator", "mobile robot"}

# Scenario-name tag prefixes to strip when recovering the real entry instance.
_SCEN_TAG_PREFIXES = (
    "power_", "emergency_", "uplink_", "telemetry_",
    "control_", "connectivity_",
)


def _fix_keyword_item_names(sysml_text: str) -> str:
    """
    Quote SysML reserved words used as item-usage names inside port/part defs.

    Pattern:  (in|out|inout) item <keyword> :
    Fix:      (in|out|inout) item '<keyword>' :

    syside rejects bare reserved words like `message`, `flow`, `connect`,
    `accept`, `send`, `loop`, `state` as item-usage names (parser error
    "Unexpected 'item'").  Quoting them makes the syntax valid.
    """
    _SYSML_KW = re.compile(
        r'\b(in|out|inout)\s+item\s+'
        r'(message|flow|connect|accept|send|loop|state|item|if|then|first|else)\s*:',
        re.IGNORECASE,
    )
    return _SYSML_KW.sub(lambda m: f"{m.group(1)} item '{m.group(2)}' :", sysml_text)


_READONLY_ATTR_RE = re.compile(r'\breadonly\s+(attribute\b)')


def _strip_readonly_keyword(sysml_text: str) -> str:
    """
    Drop the `readonly` modifier before `attribute`.

    syside's parser rejects `readonly attribute X : ...` ("Unexpected
    identifier"), and the official SysML v2 corpus never uses `readonly`
    — constants are plain `attribute name : T = value [unit];`.  The
    design-limit vs runtime-state distinction is carried by naming
    convention (max/min/limit vs current*) and assert constraints, not by
    this keyword.  The generation prompt no longer teaches `readonly`;
    this is a deterministic safety net for residual LLM emissions so they
    cost no syntax-gate LLM round.
    """
    return _READONLY_ATTR_RE.sub(r'\1', sysml_text)


_GUARD_VAR_RE = re.compile(
    r'\bif\s+(\w+)\s*(?:[<>=!]+|$)',
)
_BOOL_GUARD_RE = re.compile(
    r'\bif\s+(\w+)\s*\n',
)
_PART_DEF_BLOCK_RE = re.compile(
    r'\bpart\s+def\s+(\w+)\s*\{'
)
_ATTR_DECL_RE = re.compile(
    r'\battribute\s+(\w+)\s*:'
)


def _inject_missing_guard_attrs(
    sysml_text: str,
    sema_errors: List[Dict],
) -> tuple:
    """
    For each sema error "No Feature named 'X' found", check whether X appears
    as a guard variable in a state machine inside a part def.  If so, inject
    `attribute X : Real = 0.0;` (or Boolean = false) into that part def.

    Returns (fixed_text, n_injected).
    """
    # Collect missing names from sema errors
    missing: set = set()
    for e in sema_errors:
        m = re.search(r"No Feature named '(\w+)' found", e.get("message", ""))
        if m:
            missing.add(m.group(1))

    if not missing:
        return sysml_text, 0

    # For each missing name, check it appears in a guard (if X ...) in the text
    guard_vars = set()
    for name in missing:
        # Match `if <name>` (boolean) or `if <name> <op>` (comparison)
        if re.search(rf'\bif\s+{re.escape(name)}\b', sysml_text):
            guard_vars.add(name)

    if not guard_vars:
        return sysml_text, 0

    # Find all part def blocks and inject missing attrs into the correct one
    # Strategy: inject into every part def that contains a state def referencing
    # the missing variable (simplest: inject into the first part def that
    # contains the guard reference in its block)
    text = sysml_text
    n_injected = 0

    # Build part def blocks index: name → (block_start, block_end)
    blocks = []
    for m in _PART_DEF_BLOCK_RE.finditer(text):
        part_name = m.group(1)
        brace = text.index('{', m.start())
        end = find_block_end(text, brace)
        if end != -1:
            blocks.append((part_name, brace, end))

    # For each guard var, find the part def block that contains it
    injections: List[tuple] = []   # (insert_pos, attr_line, var_name)
    already_injected: set = set()

    for var in sorted(guard_vars):   # sorted for deterministic order
        for part_name, brace, end in blocks:
            body = text[brace + 1: end]
            if not re.search(rf'\bif\s+{re.escape(var)}\b', body):
                continue
            # Check if attribute already declared in this block
            existing = {m.group(1) for m in _ATTR_DECL_RE.finditer(body)}
            if var in existing:
                continue
            key = (part_name, var)
            if key in already_injected:
                continue

            # Determine type and safe default from guard context
            # Boolean: bare `if flagName` (no operator follows)
            if re.search(rf'\bif\s+{re.escape(var)}\s*\n', body) or \
               re.search(rf'\bif\s+{re.escape(var)}\s*then\b', body):
                attr_line = f"        attribute {var} : Boolean = false;"
            else:
                # Numeric: infer safe initial from guard threshold
                th_match = re.search(
                    rf'\bif\s+{re.escape(var)}\s*([<>]=?)\s*([\d.]+)', body
                )
                if th_match:
                    op, th = th_match.group(1), float(th_match.group(2))
                    # Start well on the safe side of the threshold
                    default = th * 3.0 + 10.0 if op in ('<', '<=') else 0.0
                    default = round(default, 1)
                else:
                    default = 0.0
                attr_line = f"        attribute {var} : Real = {default};"

            # Insert after opening brace of the part def
            injections.append((brace + 1, attr_line, var))
            already_injected.add(key)
            break   # inject into first matching part def only

    if not injections:
        return text, 0

    # Apply in reverse order so earlier offsets stay valid
    for insert_pos, attr_line, var in sorted(injections, reverse=True):
        text = text[:insert_pos] + f"\n{attr_line}" + text[insert_pos:]
        n_injected += 1

    return text, n_injected


def _scenario_src_instance(scenario_name: str) -> str:
    """Recover the entry instance name from a tagged scenario name."""
    base = scenario_name.split("_to_")[0] if "_to_" in scenario_name else scenario_name
    for p in _SCEN_TAG_PREFIXES:
        if base.startswith(p):
            return base[len(p):]
    return base


def _chat_json(llm, prompt: str) -> Dict[str, Any]:
    """One LLM call → parsed JSON object.

    When the provider supports it (LLMInterface), a low-temperature answer that
    fails to parse is retried at escalating temperatures before giving up;
    duck-typed LLMs fall back to a single call.  Raises on unparseable output —
    callers already catch and skip.
    """
    import json

    def _extract(raw: str) -> Dict[str, Any]:
        raw = str(raw).replace("```json", "").replace("```", "").strip()
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])

    escalate = getattr(llm, "chat_with_escalation", None)
    if callable(escalate):
        content, _ok = escalate(
            prompt, validate=lambda c: isinstance(_extract(c), dict)
        )
        return _extract(content)
    return _extract(str(llm.chat(prompt)))


@dataclass
class PrototypingState:
    """Tracks the current state of the prototyping session."""
    system_name: str
    system_description: str
    requirements: List[str] = field(default_factory=list)
    current_model: Optional[SysMLModel] = None
    design_space: Optional[DesignSpace] = None
    iteration: int = 0
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)



from .exploration import ExplorationMixin
from .reporting import ReportingMixin

from .refinement import RefinementMixin

from .ag_assurance_source import AGAssuranceMixin
from .pipeline_state import PipelineStateMixin

from .collaboration import CollaborationMixin
from .generation_pipeline import GenerationPipelineMixin

from .initialization import InitializationMixin
from .requirements_design import RequirementsDesignMixin

class Orchestrator(InitializationMixin, RequirementsDesignMixin, GenerationPipelineMixin, CollaborationMixin, PipelineStateMixin, AGAssuranceMixin, RefinementMixin, ExplorationMixin, ReportingMixin):
    """
    Multi-agent orchestrator for AI-assisted MBSE rapid prototyping.

    Implements a cyclic design process:
    1. Requirements extraction (RequirementsAgent)
    2. Initial design generation (DesignAgent)
    3. Design space definition and exploration (MCTS)
    4. Design evaluation and scoring
    5. Design refinement based on feedback (DesignAgent)
    6. Repeat until quality threshold is met
    """


    def prototype(
        self,
        system_name: str,
        system_description: str,
        additional_requirements: Optional[List[str]] = None,
        mcts_iterations: int = 50,
        mcts_seed: Optional[int] = None,
        mcts_patience: Optional[int] = 15,
        parse_strict: Optional[bool] = None,
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: generate a validated model then run DSE (MCTS).
        Convenience wrapper — calls generate() then explore().
        """
        gen_result = self.generate(
            system_name=system_name,
            system_description=system_description,
            additional_requirements=additional_requirements,
            parse_strict=parse_strict,
            frozen_requirements=frozen_requirements,
        )
        return self.explore(
            generate_result=gen_result,
            mcts_iterations=mcts_iterations,
            mcts_seed=mcts_seed,
            mcts_patience=mcts_patience,
        )

    # ---------------------------------------------------------------------- #
    #  Stage 1 — Generate a validated SysML v2 model (no DSE)                #
    # ---------------------------------------------------------------------- #

    def generate(
        self,
        system_name: str,
        system_description: str,
        additional_requirements: Optional[List[str]] = None,
        parse_strict: Optional[bool] = None,
        platform_profile: Optional[Dict[str, Any]] = None,
        frozen_requirements: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate through board-activated, topic-dependent knowledge sources."""
        from ..prototyping.blackboard import RecordType
        from ..prototyping.controller import BlackboardController
        from ..utils.suppressed import reset_suppressed
        from .pipeline_records import GenerationContext

        reset_suppressed()
        self._init_pipeline_state()
        self.last_verification_anchor_attempts = []
        self.last_requirement_input = {}
        self._active_model_generation_plan = None
        self.state = PrototypingState(
            system_name=system_name,
            system_description=system_description,
        )
        print(f"\n{'='*60}")
        print(f"[generate]  {system_name}")
        print(f"{'='*60}\n")

        context = GenerationContext(
            system_name=system_name,
            system_description=system_description,
            additional_requirements=list(additional_requirements or ()),
            parse_strict=parse_strict,
            platform_profile=platform_profile,
            frozen_requirements=frozen_requirements,
        )
        self._runtime_board.publish_typed(
            RecordType.CONTROL,
            "pipeline.generation.context",
            "Orchestrator",
            {"record_schema": "GenerationContext", "system_name": system_name},
            context,
        )
        self._runtime_board.publish(
            RecordType.SOURCE,
            "pipeline.request",
            "Orchestrator",
            {"system_name": system_name},
        )
        controller = BlackboardController(self._runtime_board)
        for source in self._generation_sources(context):
            controller.register(source)
        controller.run()
        context.result["control_agenda"] = controller.agenda()
        return context.result
