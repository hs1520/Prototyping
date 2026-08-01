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

class Orchestrator(CollaborationMixin, PipelineStateMixin, AGAssuranceMixin, RefinementMixin, ExplorationMixin, ReportingMixin):
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

    def __init__(
        self,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
        quality_threshold: float = 0.75,
        max_iterations: int = 3,
        rule_weight: float = 0.6,
        llm_weight: float = 0.4,
        verbose: bool = False,
        use_variation_dse: bool = False,
        use_surgical_refinement: bool = True,
        realization_inject: bool = False,
        estimator_calibration: bool = True,
        phase9_hifi: Optional[str] = None,
        revised_experiment_arm: Optional[Any] = None,
        task_session_max_turns: int = 12,
        task_session_max_tokens: int = 600000,
        r2_generation_mode: Optional[str] = None,
        r2_authored_syntax_max_attempts: int = 3,
        maximum_ag_repair_attempts: int = 3,
        maximum_plan_attempts: int = DEFAULT_MAXIMUM_PLAN_ATTEMPTS,
    ):
        self._init_pipeline_state()
        self.llm = llm
        self.rag = rag_retriever
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.rule_weight = rule_weight
        self.llm_weight = llm_weight
        self.verbose = verbose
        # "Replace" mode — the LLM declares variation points in the model and the
        # DSE explores them (run_variation_dse).  Default False = catalog path:
        # bilevel MO-MCTS + inner BO over the operator space (_explore_bilevel).
        # The legacy scalar MCTS path was removed.
        self.use_variation_dse = use_variation_dse
        self.realization_inject = realization_inject
        self.last_recommended_design = None
        self.last_pareto_designs = []
        self.last_recommended_bindings = {}
        self.last_recommended_realizable = None
        self.last_realizable_front_count = None
        self.last_recommended_by = None
        self.last_recommended_estimator_feasible = None
        self.last_recommendation_status = None
        self.last_recommendable_front_count = None
        self.last_exploratory_design = None
        self.last_exploratory_pareto_alternatives = []
        self.last_constraint_counts = {}
        self.last_search_coverage = {}
        # Where the explored variation space came from: the mandatory deterministic
        # catalog architecture seed, optional LLM additions, a pre-existing model
        # space, or None (not run yet). Reported, never hidden.
        self.last_variation_proposal_source = None
        # Final, model-fixable functional verification closure. Unlike the
        # advisory general anchor pass, this is recomputed after every final
        # refinement path and exposed to authoritative publication gates.
        self.last_functional_closure = None
        self.last_verification_anchor_attempts: List[Dict[str, Any]] = []
        # Stimulus/envelope/response/oracle contracts from requirement intake.
        # High-fidelity runners consume these instead of reinterpreting prose
        # with unrelated hard-coded thresholds.
        self.last_requirement_semantic_analysis = None
        self.last_requirement_input: Dict[str, Any] = {}
        from ..prototyping.experiment_arms import RevisedExperimentArm
        self.revised_experiment_arm = (
            RevisedExperimentArm.parse(revised_experiment_arm)
            if revised_experiment_arm is not None else None
        )
        if (
            self.revised_experiment_arm is not None
            and not self.revised_experiment_arm.implemented
        ):
            raise NotImplementedError(
                f"revised experiment arm {self.revised_experiment_arm.value} "
                "is reserved but not implemented"
            )
        self.blackboard = None
        self.context_builder = None
        self.task_sessions = None
        from ..prototyping.experiment_arms import (
            R2_DETERMINISTIC_GENERATION_MODE,
            R2_GENERATION_MODES,
        )
        self.r2_generation_mode = (
            r2_generation_mode or R2_DETERMINISTIC_GENERATION_MODE
        )
        if self.r2_generation_mode not in R2_GENERATION_MODES:
            raise ValueError(
                f"unknown r2_generation_mode {self.r2_generation_mode!r}; "
                f"must be one of {R2_GENERATION_MODES}"
            )
        self.r2_authored_syntax_max_attempts = int(
            r2_authored_syntax_max_attempts
        )
        if self.r2_authored_syntax_max_attempts <= 0:
            raise ValueError(
                "r2_authored_syntax_max_attempts must be positive"
            )
        self.maximum_ag_repair_attempts = int(maximum_ag_repair_attempts)
        if self.maximum_ag_repair_attempts <= 0:
            raise ValueError("maximum_ag_repair_attempts must be positive")
        self.last_ag_authoring_attempts: List[Dict[str, Any]] = []
        self.ag_input_dispositions: Dict[str, Dict[str, Any]] = {}
        # R2 freezes its A/G engineering decisions before DesignAgent starts.
        # The same plan guides architecture/parts/interfaces/behaviour/assembly
        # without materialising fictional implementation elements.
        self._active_ag_generation_plan: Optional[Dict[str, Any]] = None
        self._active_model_generation_plan: Optional[Dict[str, Any]] = None
        self.last_ag_binding_report: Optional[Dict[str, Any]] = None
        self.last_ag_non_degradation: Optional[Dict[str, Any]] = None
        self.task_session_max_turns = int(task_session_max_turns)
        self.task_session_max_tokens = int(task_session_max_tokens)
        if self.task_session_max_turns <= 0 or self.task_session_max_tokens <= 0:
            raise ValueError("task-session turn/token budgets must be positive")
        # F1: catalog-grid estimator calibration, applied ONLY around the search
        # (the injected SysML calc defs and Phase 8's estimator_value column keep
        # the documented textbook constants). Provenance recorded, never hidden.
        self.use_estimator_calibration = estimator_calibration
        self.last_estimator_calibration = None
        # Phase 9 is default-OFF.  A published authority requires the dedicated
        # realization driver, which freezes the Phase 8 result and then executes
        # Gazebo -> SITL in one locked run bundle.  Explicit modes remain a
        # best-effort developer seam and NEVER upgrade datasheet CLOSED.
        self.phase9_hifi = phase9_hifi
        # Refinement asks the LLM for block-level replacements first (surgical:
        # untouched blocks cannot lose connects, output ~10× smaller) and only
        # falls back to the legacy whole-model rewrite when no valid merge is
        # produced. Disable to force the legacy path (tests / A-B comparison).
        self.use_surgical_refinement = use_surgical_refinement

        # Initialize specialized agents
        self.requirements_agent = RequirementsAgent(llm, rag_retriever)
        self.design_agent = DesignAgent(
            llm, rag_retriever, maximum_plan_attempts=maximum_plan_attempts
        )
        # Pass quality_threshold so the evaluator's veto cap is consistent with
        # the orchestrator's refinement gate (cap = threshold − 0.05).
        self.evaluator = DesignEvaluator(quality_threshold=quality_threshold)
        self.cot = ChainOfThoughtPrompter(llm)
        self.sim_validator = SimulationValidator()

        self.state: Optional[PrototypingState] = None

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
        """
        Generate a validated SysML v2 model without Design Space Exploration.

        Pipeline
        ────────
        Phase 1  Requirements Extraction
        Phase 2  Initial Design Generation
        Phase 3  Iterative Refinement  (syntax gate → eval → LLM, no MCTS)
        Phase 4  Behavioral Reachability Simulation

        The returned dict can be passed directly to explore() when DSE is
        needed later, or used as a standalone result.

        Returns
        -------
        {
          system_name, requirements,
          model, model_sysml, model_summary,
          final_score, iterations, evaluation_history,
          simulation_result,
        }
        """
        from ..utils.suppressed import reset_suppressed
        reset_suppressed()
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

        # ── Phase 1: Requirements Extraction ─────────────────────────────────
        print("Phase 1: Requirements Input")
        print("-" * 40)
        if frozen_requirements is not None:
            if additional_requirements:
                raise ValueError(
                    "frozen_requirements and additional_requirements are mutually "
                    "exclusive in a controlled run"
                )
            requirements = self._use_frozen_requirements(frozen_requirements)
            print(f"  ✓ Loaded {len(requirements)} frozen requirements")
        else:
            requirements = self._extract_requirements(
                system_name, system_description, additional_requirements or []
            )
            print(f"  ✓ Extracted {len(requirements)} requirements")
        self.state.requirements = requirements
        # Board first: A/G planning makes real LLM decisions, so it is an Agent
        # task and needs a typed task, an envelope, and an archived session like
        # every other one. Freezing the plan before the board existed left those
        # turns unrecorded, which §5.3 does not permit.
        self._open_collaboration_board(system_name, requirements)
        self._active_ag_generation_plan = self._prepare_ag_guided_generation(
            requirements
        )
        self._prepare_design_handoff(system_name, requirements)
        print()

        # ── Phase 2: Initial Design Generation ───────────────────────────────
        print("Phase 2: Initial Design Generation")
        print("-" * 40)
        model = self._generate_initial_design(
            system_name, requirements,
            parse_strict=parse_strict,
            platform_profile=platform_profile,
        )
        self.state.current_model = model
        raw_model_plan = (
            getattr(model, "metadata", None) or {}
        ).get("whole_model_generation_plan")
        if isinstance(raw_model_plan, Mapping):
            self._active_model_generation_plan = dict(raw_model_plan)
        print(f"  ✓ Generated model with {len(model.part_definitions)} part definitions\n")

        # ── Phase 3: Iterative Refinement (no MCTS) ───────────────────────────
        print("Phase 3: Iterative Refinement")
        print("-" * 40)
        final_model, final_score, final_sim = self._iterative_refinement(
            model, requirements, dse_best_config=None
        )

        # ── Phase 3.5: SITL-L1 refinement (only when targeting a platform) ────
        # Feed unresolved ArduPilot-parameter mappings (= model genuinely
        # missing a guard/attribute a requirement needs) back to the design
        # LLM.  Cheap & deterministic (no SITL process launch); L2 stays
        # terminal.
        if platform_profile is not None:
            final_model, final_score, final_sim = self._sitl_refinement_loop(
                final_model, requirements, final_score, final_sim, max_iters=2
            )

        # Terminal model mutation: runs after ordinary and optional SITL-L1
        # refinement so no later LLM rewrite can overwrite functional closure.
        final_model, final_score, final_sim = self._functional_closure_pass(
            final_model,
            final_sim,
            final_score,
            requirements,
            dse_best_config=None,
            max_iters=2,
        )
        pre_terminal_score = final_score

        # First close the ordinary generated model, then snapshot it.  Terminal
        # A/G binding is evaluated as a separate transaction so it cannot hide
        # damage to the executable architecture behind an aggregate score.
        self._restore_generation_plan_metadata(final_model)
        pre_ag_sysml, generation_plan_conformance = (
            self._enforce_terminal_generation_plan(
                final_model, get_sysml_text(final_model)
            )
        )
        pre_ag_sim = self._run_simulation(pre_ag_sysml, system_name)
        final_sysml = self._reconcile_guided_ag_contract_layer(
            pre_ag_sysml, requirements
        )
        self._commit_terminal_model(final_sysml, producer="Orchestrator.generate")
        collaboration_artifacts = self._build_collaboration_artifacts(final_sysml)
        final_sysml = collaboration_artifacts.pop(
            "_terminal_model_sysml", final_sysml
        )
        final_model, final_score, final_sim, terminal_consistency = (
            self._synchronize_terminal_snapshot(
                final_model,
                final_sysml,
                requirements,
                prior_score=pre_terminal_score,
                dse_best_config=None,
            )
        )
        self.last_ag_non_degradation = None
        if self._active_ag_generation_plan is not None:
            from ..prototyping.ag_quality_gate import (
                build_ag_non_degradation_report,
            )
            self.last_ag_non_degradation = build_ag_non_degradation_report(
                pre_ag_sim, final_sim
            )
        from ..prototyping.model_qualification import build_model_qualification
        structural_obligation_report = (
            self._validate_terminal_structural_obligations(
                final_model,
                final_sysml,
                system_name,
            )
        )
        semantic_fidelity_report = (
            self._validate_terminal_semantic_obligations(
                final_model,
                final_sysml,
                system_name,
            )
        )
        model_qualification = build_model_qualification(
            model_text=final_sysml,
            requirements=requirements,
            syntax_result=check_syntax(
                final_sysml,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            ),
            simulation_result=final_sim,
            terminal_consistency=terminal_consistency,
            structural_obligation_report=structural_obligation_report,
            semantic_fidelity_report=semantic_fidelity_report,
            generation_plan_conformance=generation_plan_conformance,
            ag_contract_graph=collaboration_artifacts.get("ag_contract_graph"),
            pattern_conformance_report=collaboration_artifacts.get(
                "pattern_conformance_report"
            ),
            ag_binding_report=self.last_ag_binding_report,
            ag_non_degradation=self.last_ag_non_degradation,
            ag_expected=self._active_ag_generation_plan is not None,
            generation_plan_expected=(
                self._active_model_generation_plan is not None
            ),
            semantic_fidelity_expected=bool(
                (semantic_fidelity_report or {}).get("total")
            ),
        )
        self.state.current_model = final_model
        print(f"  ✓ Final design score: {final_score:.3f}\n")

        # ── Phase 4: Behavioral Reachability Simulation ───────────────────────
        # Simulation already ran in Phase 3 — reuse the result, no duplicate run.
        print("Phase 4: Behavioral Reachability Simulation", flush=True)
        print("-" * 40)
        self._print_final_sim(final_sim)

        # ── Summary ───────────────────────────────────────────────────────────
        sim_warnings = (getattr(final_model, "metadata", None) or {}).get("sim_warnings", "")
        print(f"{'='*60}")
        print("Generation Complete!")
        print(f"  Final score:              {final_score:.3f}")
        print(
            f"  Model qualification:      "
            f"{model_qualification['status']}"
        )
        if final_sim.requirement_reachability_score is not None:
            print(
                "  Requirement reachability: "
                f"{final_sim.requirement_reachability_score:.3f} "
                f"({final_sim.requirement_scenarios_passed}/"
                f"{final_sim.requirement_scenarios_total} frozen paths)"
            )
            print(
                "  Advisory role scenarios: "
                f"{final_sim.reachability_score:.3f} "
                f"({len(final_sim.passed_scenarios())}/"
                f"{len(final_sim.scenario_results)} scenarios)"
            )
        else:
            print(
                f"  Simulation reachability:  "
                f"{final_sim.reachability_score:.3f} "
                f"({len(final_sim.passed_scenarios())}/"
                f"{len(final_sim.scenario_results)} scenarios)"
            )
        print(f"  Part definitions: {len(final_model.part_definitions)}")
        print(f"  Requirements:     {len(requirements)}")
        if sim_warnings:
            print()
            for line in sim_warnings.splitlines():
                print(f"  {line}")
        ledger = getattr(self.llm, "ledger", None)
        if ledger is not None and getattr(ledger, "calls", 0):
            print(f"  LLM usage:        {ledger.summary()}")
        if self.verbose:
            from ..utils.suppressed import suppressed_summary
            summary = suppressed_summary()
            if summary:
                print(f"  suppressed:       {summary}")
        print(f"{'='*60}\n")

        return {
            "system_name":        system_name,
            "requirements":       requirements,
            "model":              final_model,
            "model_sysml":        final_sysml,
            "model_summary":      final_model.get_summary(),
            "final_score":        final_score,
            "iterations":         self.state.iteration,
            "evaluation_history": self.state.evaluation_history,
            "simulation_result":  final_sim,
            "terminal_consistency": terminal_consistency,
            "model_qualification": model_qualification,
            "model_acceptance_status": model_qualification["status"],
            "whole_model_generation_plan": final_model.metadata.get(
                "whole_model_generation_plan"
            ),
            "generation_plan_conformance": final_model.metadata.get(
                "generation_plan_conformance"
            ),
            "step1_plan_attempts": final_model.metadata.get(
                "step1_plan_attempts"
            ),
            "step1_plan_retries": final_model.metadata.get(
                "step1_plan_retries", 0
            ),
            "structural_obligation_report": structural_obligation_report,
            "semantic_fidelity_report": semantic_fidelity_report,
            "ag_binding_report": self.last_ag_binding_report,
            "ag_non_degradation": self.last_ag_non_degradation,
            "functional_closure": dict(self.last_functional_closure or {}),
            "verification_anchor_attempts": list(
                self.last_verification_anchor_attempts
            ),
            "requirement_semantic_analysis": dict(
                self.last_requirement_semantic_analysis or {}
            ),
            "requirement_input": dict(self.last_requirement_input or {}),
            **collaboration_artifacts,
            "platform_profile":   platform_profile,
            "llm_usage":          ledger.as_dict() if ledger is not None else None,
        }

    # ---------------------------------------------------------------------- #
    #  Stage 2 — Design Space Exploration on a validated model                #
    # ---------------------------------------------------------------------- #








    def _extract_requirements(
        self,
        system_name: str,
        description: str,
        additional: List[str],
    ) -> List[str]:
        """Phase 1: Extract requirements using the RequirementsAgent."""
        result = self.requirements_agent.run({
            "system_description": description,
            "system_name": system_name,
            # Seed existing_requirements with manually provided ones so the LLM
            # is aware of them and avoids generating near-duplicates from the start.
            "existing_requirements": additional,
        })

        if not result.success and not result.output:
            print(f"  ✗ Requirements extraction failed: {result.reasoning}")

        # RequirementsAgent now returns a unified set (fixed anchors + new additions)
        # with consistent IDs in one pass — no separate merge step needed.
        requirements = result.output if result.output else list(additional)

        # Validate unified set
        validation = self.requirements_agent.validate_requirements(requirements)
        self.last_requirement_semantic_analysis = validation.get(
            "requirement_semantic_analysis"
        )

        from ..prototyping.requirement_inputs import build_frozen_requirement_set
        try:
            input_artifact = build_frozen_requirement_set(
                requirements,
                name=f"{system_name}-llm-extracted",
                source="llm_extracted",
            )
            self.last_requirement_input = {
                **input_artifact,
                "mode": "llm_extracted",
                "frozen": False,
                "fixed_anchor_count": len(additional),
            }
        except ValueError as exc:
            self.last_requirement_input = {
                "mode": "llm_extracted",
                "frozen": False,
                "requirement_set_digest": None,
                "validation_error": str(exc),
            }

        if validation["issues"]:
            for issue in validation["issues"][:5]:
                print(f"  ✗ {issue}")
        if validation["warnings"]:
            for warning in validation["warnings"][:3]:
                print(f"  ⚠ {warning}")

        # Category breakdown — prefer metadata already computed in run(), fall back to validation
        counts = result.metadata.get("counts_by_category") or validation.get("counts_by_category", {})
        category_summary = ", ".join(
            f"{cat}={n}" for cat, n in sorted(counts.items()) if n > 0
        )
        if category_summary:
            print(f"  Categories: {category_summary}")

        # Surface dependency info if found
        dependencies = result.metadata.get("dependencies", [])
        if dependencies:
            print(f"  Dependencies: {len(dependencies)} pair(s) identified")

        if self.verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] Phase 1 — Full Requirements List ({len(requirements)})")
            print(f"  {'─'*60}")
            for i, req in enumerate(requirements, 1):
                print(f"  [{i:>2}] {req}")

        return requirements

    def _use_frozen_requirements(self, frozen: Any) -> List[str]:
        """Load an immutable requirement artifact without an LLM extraction call."""
        from ..prototyping.requirement_inputs import resolve_frozen_requirement_set

        requirements, artifact = resolve_frozen_requirement_set(frozen)
        validation = self.requirements_agent.validate_requirements(requirements)
        if validation["issues"]:
            raise ValueError(
                "frozen requirements failed deterministic validation: "
                + "; ".join(validation["issues"])
            )
        self.last_requirement_semantic_analysis = validation.get(
            "requirement_semantic_analysis"
        )
        self.last_requirement_input = {
            **artifact,
            "mode": "frozen",
            "frozen": True,
        }
        return requirements

    def _generate_initial_design(
        self,
        system_name: str,
        requirements: List[str],
        parse_strict: Optional[bool] = None,
        platform_profile: Optional[Dict[str, Any]] = None,
    ) -> SysMLModel:
        """Phase 2: Generate initial SysML v2 design."""
        task = {
            "system_name": system_name,
            "requirements": requirements,
            "parse_strict": (parse_strict if parse_strict is not None else False),
            "verbose": self.verbose,
            "platform_profile": platform_profile,
        }
        handoff = self._design_handoff()
        if handoff is not None:
            _board_task, envelope, _session = self._design_handoff_objects(
                handoff
            )
            task["context"] = envelope.render_for_prompt()
        if self._active_ag_generation_plan is not None:
            task["semantic_guidance_by_step"] = dict(
                self._active_ag_generation_plan["guidance_by_step"]
            )
            behavior_plan = self._active_ag_generation_plan.get(
                "behavior_plan"
            )
            if behavior_plan is not None:
                task["ag_behavior_obligation_plan"] = behavior_plan.to_dict()
        observer_id = None
        add_observer = getattr(self.llm, "add_call_observer", None)
        remove_observer = getattr(self.llm, "remove_call_observer", None)
        if handoff is not None and callable(add_observer):
            _board_task, _envelope, session = self._design_handoff_objects(
                handoff
            )
            handoff.observer_error_count_before = len(
                getattr(self.llm, "call_observer_errors", ())
            )

            def archive_call(event: Mapping[str, Any]) -> None:
                handoff.captured_llm_calls += 1
                self._archive_provider_call(session, event)
                self._publish_generation_fragment(handoff, event)

            observer_id = add_observer(archive_call)
        try:
            result = self.design_agent.run(task)
        except Exception as exc:
            self._reject_design_handoff(
                f"{type(exc).__name__}: {exc}", producer="DesignAgent"
            )
            raise
        finally:
            if observer_id is not None and callable(remove_observer):
                remove_observer(observer_id)
        if handoff is not None and observer_id is not None:
            before = int(handoff.observer_error_count_before)
            handoff.observer_errors = list(
                getattr(self.llm, "call_observer_errors", ())[before:]
            )
        if result.success and isinstance(result.output, _SysMLModelTypes):
            model = result.output
        else:
            model = build_lite_model("", model_name=system_name)

        untraced = result.metadata.get("untraced_requirements", [])
        if untraced:
            print(f"  ⚠ {len(untraced)} requirement(s) could not be matched to any component: "
                  f"{', '.join(untraced)}")

        self.requirements_agent.create_sysml_requirements(requirements, model)
        model = self._materialize_guided_ag_contracts(model, system_name)
        self._finalize_design_handoff(result, model)
        return model





















    #: Topic carrying per-step generation drafts. Deliberately its own topic so
    #: it is easy to see — and easy to refuse. `ContextBuilder` accepts only
    #: `requirements.authoritative` and `design.ag_generation_plan` as design
    #: sources, so a draft can never become the input a later stage builds from.
    GENERATION_FRAGMENT_TOPIC = "generation.fragment"























    # ------------------------------------------------------------------















    # ------------------------------------------------------------------
    # Phase 3.5: SITL-L1 refinement loop
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # Simulation inner refinement loop
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # Connect audit step
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------








    @staticmethod
    def _build_refinement_feedback(
        eval_result: Any,
        cot_feedback: str,
        persistent_issues: Optional[List[str]] = None,
        mcts_constraints: str = "",
        sim_issues: Optional[List[str]] = None,
    ) -> str:
        """Combine evaluator issues, simulation failures, and LLM feedback into
        a refinement-oriented prompt section.

        Args:
            eval_result:        Rule-based evaluation result (issues + recommendations).
            cot_feedback:       LLM chain-of-thought final answer (may be empty).
            persistent_issues:  Issues that have appeared in more than one iteration.
            mcts_constraints:   Architectural decisions from MCTS (non-negotiable).
            sim_issues:         Behavioral simulation failures from SimulationValidator.
        """
        lines = []

        # MCTS decisions come first — they are non-negotiable architectural constraints
        if mcts_constraints:
            lines.append(mcts_constraints)
            lines.append("")

        lines.append("Refinement targets:")
        for issue in eval_result.issues:
            lines.append(f"- {issue}")
        for rec in eval_result.recommendations:
            lines.append(f"- {rec}")

        # Simulation failures: these are structural connectivity gaps found by
        # running the port-connection graph against operational scenarios.
        if sim_issues:
            lines.append("")
            lines.append(
                "Behavioral simulation failures (port-connection reachability check):\n"
                "  The following operational scenarios have no directed signal path in the model.\n"
                "  Add `connect <source_part>::<port> to <target_part>::<port>;` statements\n"
                "  to establish the missing paths."
            )
            for iss in sim_issues:
                lines.append(f"- [SIM] {iss}")

        if persistent_issues:
            lines.append("")
            lines.append(
                "Persistent issues (appeared in multiple iterations — escalate priority):"
            )
            for iss in persistent_issues:
                lines.append(f"- [PERSISTENT] {iss}")
        if cot_feedback:
            lines.append("")
            lines.append("LLM evaluation summary:")
            lines.append(cot_feedback)
        return "\n".join(lines)
