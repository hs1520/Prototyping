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

class Orchestrator(RefinementMixin, ExplorationMixin, ReportingMixin):
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
        self._active_design_handoff = None
        self._active_ag_planning_handoff = None
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
        handoff = self._active_design_handoff
        if handoff is not None:
            task["context"] = handoff["envelope"].render_for_prompt()
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
            session = handoff["session"]
            handoff["observer_error_count_before"] = len(
                getattr(self.llm, "call_observer_errors", ())
            )

            def archive_call(event: Mapping[str, Any]) -> None:
                handoff["captured_llm_calls"] += 1
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
            before = int(handoff.get("observer_error_count_before", 0))
            handoff["observer_errors"] = list(
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

    @staticmethod
    def _requirement_planning_model(requirements: List[str]) -> str:
        """Render immutable requirement inputs for pre-generation A/G planning."""
        from ..utils.req_id import normalise_req_id

        definitions: List[str] = []
        for raw in requirements:
            match = re.match(r"\s*([^:]+)\s*:\s*(.*)", str(raw), re.DOTALL)
            if not match:
                continue
            req_id = normalise_req_id(match.group(1))
            body = match.group(2).strip().replace("*/", "* /")
            definitions.append(
                f"    requirement def {req_id} {{ doc /* {body} */ }}"
            )
        return (
            "package AGPlanningInputs {\n"
            + "\n".join(definitions)
            + "\n}\n"
        )

    @staticmethod
    def _ag_guidance_for_specs(
        specs: List[Any],
        packages: List[str],
        *,
        authored_mode: bool = False,
        behavior_plan: Optional[Any] = None,
    ) -> Dict[str, str]:
        """Compile one frozen A/G plan into the five DesignAgent prompt seams."""
        from ..prototyping.ag_extractor import extract_ag_graph

        component_lines: List[str] = []
        interface_lines: List[str] = []
        behavior_lines: List[str] = []
        assembly_lines: List[str] = []
        for spec, package_text in zip(specs, packages):
            component_lines.append(
                f"- {spec.source_requirement}: system contract "
                f"{spec.system_contract}, observation `{spec.observation}`, "
                f"pattern {spec.pattern}."
            )
            for component in spec.components:
                component_lines.append(
                    f"  - REQUIRED component {component.owner_def} "
                    f"(usage {component.owner_usage}) owns "
                    f"{component.name} and guarantees "
                    f"{', '.join(component.guarantees)}."
                )
                assumptions = [
                    (
                        f"{item.concept} (environment)"
                        if item.environment else f"{item.concept} (internal)"
                    )
                    for item in component.assumptions
                ]
                interface_lines.append(
                    f"- {component.owner_def}: consumes "
                    f"{', '.join(assumptions) or '(none)'}; produces "
                    f"{', '.join(component.guarantees)}; behavior trigger "
                    f"{component.trigger_signal or '(continuous invariant)'}."
                )
                if not authored_mode:
                    paths = component.realization_paths or ()
                    behavior_lines.extend(
                        f"- {component.behavior}: {path.source} --"
                        f"{path.trigger or 'continuous'}--> {path.target}; "
                        f"entry action `{path.action}`"
                        + (f"; guard `{path.guard}`." if path.guard else ".")
                        for path in paths
                    )
            if authored_mode:
                # The approved boundary above supplies only components/interfaces.
                # Behavior guidance must come from the package the LLM just
                # authored, never from the reviewed candidate's realization paths.
                graph = extract_ag_graph(package_text)
                for behavior in graph.behaviors:
                    for transition in behavior.transitions:
                        action = behavior.entry_actions.get(
                            transition.target, "(no entry action)"
                        )
                        behavior_lines.append(
                            f"- {behavior.name}: {transition.source} --"
                            f"{transition.trigger}--> {transition.target}; "
                            f"entry action `{action}`"
                            + (
                                f"; guard `{transition.guard}`."
                                if transition.guard else "."
                            )
                        )
            assembly_lines.append(
                f"- Implement the frozen decisions for "
                f"{spec.source_requirement} in the real system package. "
                f"Do not emit planning package `{spec.package}`: the terminal "
                "binder will create the assurance package only after it can "
                "resolve the generated owners and behaviors."
            )

        prefix = (
            "\nA/G-GUIDED GENERATION PLAN (frozen before model generation):\n"
            "These are required design inputs, not optional examples. Generate "
            "the ordinary architecture around them and do not create competing "
            "owners, response concepts, or timing origins.\n"
        )
        behavior_guidance = (
            behavior_plan.render_for_prompt()
            if behavior_plan is not None else "\n".join(behavior_lines)
        )
        return {
            "architecture": prefix + "\n".join(component_lines),
            "parts": prefix + "\n".join(component_lines),
            "interfaces": prefix + "\n".join(interface_lines),
            "behavior": prefix + behavior_guidance,
            "assembly": prefix + "\n".join(assembly_lines),
        }

    def _prepare_ag_guided_generation(
        self, requirements: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Freeze R2 A/G decisions before any architecture or behavior is generated."""
        from ..prototyping.experiment_arms import RevisedExperimentArm

        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return None

        from ..prototyping.ag_chains import select_ag_chains

        selected = list(select_ag_chains(requirements))
        if not selected:
            raise RuntimeError(
                "R2-BBAG failed closed: no bounded A/G chain was selected"
            )
        self.last_ag_authoring_attempts = []
        self.ag_input_dispositions = {}
        planning_session = self._open_ag_planning_session(requirements)
        try:
            return self._compile_ag_generation_plan(selected, requirements)
        finally:
            self._close_ag_planning_session(planning_session)

    def _open_ag_planning_session(self, requirements: List[str]) -> Optional[Any]:
        """Open the board task/session that archives the A/G decision turns.

        Without this the decision conversation would be the only LLM work in the
        run whose transcript is not application-owned — and §5.3 requires every
        result to record what the model actually saw.
        """
        if (
            self.blackboard is None
            or self.context_builder is None
            or self.task_sessions is None
        ):
            return None
        from ..prototyping.blackboard import TaskStatus

        source = None
        for record in self.blackboard.records(topic="requirements.authoritative"):
            source = record
        if source is None:
            return None
        state = getattr(self, "state", None)
        system_name = (
            state.system_name if state is not None and state.system_name
            else "System"
        )
        task = self.blackboard.create_task(
            "AG_GENERATION_PLANNING",
            "AGPlanningAgent",
            required_topics=("requirements.authoritative",),
        )
        self.blackboard.transition_task(task.task_id, TaskStatus.ACTIVE)
        envelope = self.context_builder.build_ag_planning_context(
            task_id=task.task_id,
            system_name=system_name,
            source_record_ids=(source.record_id,),
        )
        session = self.task_sessions.open(
            task_id=task.task_id,
            agent_role="AGPlanningAgent",
            base_model_revision=task.base_model_revision,
            base_model_digest=task.base_model_digest,
            context_envelope_id=envelope.envelope_id,
            max_turns=self.task_session_max_turns,
            max_tokens=self.task_session_max_tokens,
        )
        observer_id = None
        add_observer = getattr(self.llm, "add_call_observer", None)
        if callable(add_observer):
            def archive_call(event: Mapping[str, Any]) -> None:
                self._archive_provider_call(session, event)

            observer_id = add_observer(archive_call)
        self._active_ag_planning_handoff = {
            "task": task, "envelope": envelope, "session": session,
            "observer_id": observer_id,
        }
        return self._active_ag_planning_handoff

    def _close_ag_planning_session(self, handoff: Optional[Any]) -> None:
        """Publish the typed planning result and close the session."""
        if not handoff:
            return
        from ..prototyping.blackboard import RecordType, TaskStatus
        from ..prototyping.task_session import SessionStatus

        remove_observer = getattr(self.llm, "remove_call_observer", None)
        if handoff.get("observer_id") is not None and callable(remove_observer):
            remove_observer(handoff["observer_id"])
        task, session, envelope = (
            handoff["task"], handoff["session"], handoff["envelope"]
        )
        record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.ag_planning.result",
            "AGPlanningAgent",
            {
                "success": True,
                "context_envelope_id": envelope.envelope_id,
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
                "included_record_ids": list(envelope.included_record_ids),
                "decision_attempts": len(self.last_ag_authoring_attempts),
                "accepted_status": "ACCEPTED",
            },
            task_id=task.task_id,
            session_id=session.session_id,
        )
        if task.status is TaskStatus.ACTIVE:
            self.blackboard.transition_task(
                task.task_id,
                TaskStatus.COMPLETED,
                producer="AGPlanningAgent",
                result_record_ids=(record.record_id,),
            )
        if session.status is SessionStatus.OPEN:
            session.close(
                SessionStatus.COMPLETED,
                output_record_ids=(record.record_id,),
            )
        self._active_ag_planning_handoff = None

    def _compile_ag_generation_plan(
        self, selected: List[Any], requirements: List[str]
    ) -> Dict[str, Any]:
        from ..prototyping.experiment_arms import (
            R2_DETERMINISTIC_GENERATION_MODE,
            R2_LLM_AUTHORED_GENERATION_MODE,
            R2_LLM_DECIDED_GENERATION_MODE,
        )
        from ..prototyping.ag_behavior_plan import (
            compile_behavior_obligation_plan,
        )
        from ..prototyping.ag_emitter import emit_ag_package
        from ..prototyping.ag_extractor import extract_ag_graph
        from ..prototyping.ag_planning import (
            check_ag_planning_graph,
            emit_ag_planning_package,
            strip_ag_implementation,
        )

        planning_model = self._requirement_planning_model(requirements)
        planned_specs: List[Any] = []
        planning_packages: List[str] = []
        guidance_packages: List[str] = []
        planning_reports: List[Dict[str, Any]] = []
        for reviewed_spec in selected:
            if self.r2_generation_mode == R2_DETERMINISTIC_GENERATION_MODE:
                planned_spec = reviewed_spec
                guidance_package = emit_ag_package(planned_spec)
                planning_package = emit_ag_planning_package(planned_spec)
            elif self.r2_generation_mode == R2_LLM_DECIDED_GENERATION_MODE:
                planned_spec = self._generate_llm_decided_ag_spec(
                    reviewed_spec,
                    planning_model,
                    generation_stage=True,
                )
                guidance_package = emit_ag_package(planned_spec)
                planning_package = emit_ag_planning_package(planned_spec)
            elif self.r2_generation_mode == R2_LLM_AUTHORED_GENERATION_MODE:
                guidance_package, _gate = (
                    self._generate_llm_authored_ag_until_quality_valid(
                        reviewed_spec,
                        planning_model,
                        generation_stage=True,
                    )
                )
                # Free-form authored packages remain the authority; the reviewed
                # boundary is used only to compile non-gold architecture prompts.
                planned_spec = reviewed_spec
                planning_package = strip_ag_implementation(
                    guidance_package, planned_spec
                )
            else:  # constructor validation should make this unreachable
                raise RuntimeError(
                    f"unsupported R2 generation mode {self.r2_generation_mode}"
                )
            gate = check_syntax(
                planning_package,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            if gate.has_errors:
                raise RuntimeError(
                    f"R2-BBAG pre-generation package for "
                    f"{reviewed_spec.source_requirement} failed syntax: "
                    f"{gate.short_summary()}"
                )
            planning_report = check_ag_planning_graph(
                extract_ag_graph(f"{planning_model}\n{planning_package}")
            )
            if planning_report.verdict != "PASS":
                raise RuntimeError(
                    f"R2-BBAG pre-generation planning graph for "
                    f"{reviewed_spec.source_requirement} failed closed: "
                    f"{self._format_ag_diagnostics(planning_report)}"
                )
            planned_specs.append(planned_spec)
            planning_packages.append(planning_package)
            guidance_packages.append(guidance_package)
            planning_reports.append(planning_report.to_dict())

        authored_mode = (
            self.r2_generation_mode == R2_LLM_AUTHORED_GENERATION_MODE
        )
        # Free-form authored packages have a separate authority boundary.  They
        # cannot be losslessly reconstructed from the reviewed candidate spec,
        # so ModelPlan-v2 attachment is limited to deterministic/decided modes.
        behavior_plan = (
            None if authored_mode
            else compile_behavior_obligation_plan(planned_specs)
        )
        if behavior_plan is not None and behavior_plan.status != "PASS":
            raise RuntimeError(
                "R2-BBAG typed behavior obligation compilation failed closed: "
                f"{behavior_plan.status}"
            )
        return {
            "stage": "PRE_GENERATION_A_G_PLANNING",
            "mode": self.r2_generation_mode,
            "source_requirements": [
                spec.source_requirement for spec in planned_specs
            ],
            "specs": planned_specs,
            "packages": planning_packages,
            "guidance_packages": guidance_packages,
            "planning_reports": planning_reports,
            "behavior_plan": behavior_plan,
            "behavior_plan_artifact": (
                behavior_plan.to_dict()
                if behavior_plan is not None else None
            ),
            "guidance_by_step": self._ag_guidance_for_specs(
                planned_specs,
                guidance_packages,
                authored_mode=authored_mode,
                behavior_plan=behavior_plan,
            ),
        }

    @staticmethod
    def _without_named_package(model_text: str, package_name: str) -> str:
        pattern = re.compile(
            rf"\bpackage\s+{re.escape(package_name)}\s*\{{"
        )
        text = str(model_text)
        while True:
            match = pattern.search(text)
            if match is None:
                return text
            brace = text.find("{", match.start())
            end = find_block_end(text, brace)
            if end == -1:
                return text
            text = text[:match.start()] + text[end + 1:]

    def _materialize_guided_ag_contracts(
        self, model: SysMLModel, system_name: str
    ) -> SysMLModel:
        """Attach planning provenance without adding shadow implementation."""
        if self._active_ag_generation_plan is None:
            return model
        # The planning package may have been echoed by the assembly LLM even
        # though it is not an implementation artifact.  Remove it before the
        # first Blackboard commit, otherwise immutable-requirement protection
        # correctly freezes provisional contract bodies and prevents terminal
        # binding from replacing them.
        original_metadata = dict(getattr(model, "metadata", None) or {})
        original_text = get_sysml_text(model)
        sanitized_text = original_text
        for spec in self._active_ag_generation_plan["specs"]:
            sanitized_text = self._without_named_package(
                sanitized_text, spec.package
            )
        if sanitized_text != original_text:
            model = build_lite_model(sanitized_text, model_name=system_name)
            model.metadata.update(original_metadata)
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        model.metadata.update({
            "last_sysml_text": sanitized_text,
            "ag_guided_generation": True,
            "ag_generation_stage": "PLANNING_INPUT_ONLY",
            "ag_generation_mode": self.r2_generation_mode,
            "ag_guided_requirements": list(
                self._active_ag_generation_plan["source_requirements"]
            ),
            "ag_planning_reports": list(
                self._active_ag_generation_plan["planning_reports"]
            ),
        })
        return model

    def _reconcile_guided_ag_contract_layer(
        self, model_text: str, requirements: List[str]
    ) -> str:
        """Bind frozen contracts to real terminal owners and behaviors."""
        from ..prototyping.experiment_arms import RevisedExperimentArm

        self.last_ag_binding_report = None
        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return model_text
        if self._active_ag_generation_plan is None:
            # Compatibility for callers that exercise the low-level seam without
            # running generate(); production generate() always has a plan.
            return self._apply_ag_contract_layer(model_text, requirements)
        from ..prototyping.ag_binding import bind_ag_contracts_to_model

        state = getattr(self, "state", None)
        system_package = (
            state.system_name
            if state is not None and state.system_name
            else "System"
        )
        result = bind_ag_contracts_to_model(
            model_text,
            self._active_ag_generation_plan["specs"],
            system_package=system_package,
            behavior_plan=self._active_ag_generation_plan.get(
                "behavior_plan"
            ),
        )
        self.last_ag_binding_report = result.report.to_dict()
        return result.model_text

    def _open_collaboration_board(
        self, system_name: str, requirements: List[str]
    ) -> bool:
        """Open the board and publish the authoritative source, before any task.

        A/G planning makes real LLM decisions, so under §5.3 it is an Agent task
        and needs a board, a typed task, and an archived session like any other.
        That is only possible if the board exists before the plan is frozen —
        hence this step is separate from, and earlier than, the design handoff.

        Returns False for an arm that does not use the blackboard at all.
        """
        from ..prototyping.experiment_arms import RevisedExperimentArm  # noqa: F401
        from ..prototyping.blackboard import Blackboard, RecordType
        from ..prototyping.context_builder import ContextBuilder
        from ..prototyping.task_session import TaskSessionRegistry

        arm = self.revised_experiment_arm
        self._active_design_handoff = None
        self._active_ag_planning_handoff = None
        if arm is None or not arm.uses_blackboard:
            self.blackboard = None
            self.context_builder = None
            self.task_sessions = None
            return False

        self.blackboard = Blackboard(system_name)
        self.context_builder = ContextBuilder(self.blackboard)
        self.task_sessions = TaskSessionRegistry()
        self.blackboard.publish(
            RecordType.SOURCE,
            "requirements.authoritative",
            "RequirementsAgent",
            {
                "requirements": list(requirements),
                "requirement_input_mode": self.last_requirement_input.get("mode"),
                "requirement_set_digest": self.last_requirement_input.get(
                    "requirement_set_digest"
                ),
                "gold_access": False,
            },
        )
        return True

    def _prepare_design_handoff(
        self, system_name: str, requirements: List[str]
    ) -> None:
        """Create the first typed RequirementsAgent -> DesignAgent handoff.

        This is the R1-BBCTX intervention.  The envelope is derived only from
        run-time source/model records; evaluator gold has no API into it.
        """
        from ..prototyping.blackboard import RecordType, TaskStatus

        # Direct callers (and any path that did not open the board first) still
        # get a board here; generate() opens it earlier so A/G planning is a
        # board-mediated Agent task rather than work done before the board exists.
        if self.blackboard is None:
            if not self._open_collaboration_board(system_name, requirements):
                return
        source_record = None
        for record in self.blackboard.records(topic="requirements.authoritative"):
            source_record = record
        if source_record is None:
            source_record = self.blackboard.publish(
                RecordType.SOURCE,
                "requirements.authoritative",
                "RequirementsAgent",
                {
                    "requirements": list(requirements),
                    "requirement_input_mode": self.last_requirement_input.get(
                        "mode"
                    ),
                    "requirement_set_digest": self.last_requirement_input.get(
                        "requirement_set_digest"
                    ),
                    "gold_access": False,
                },
            )
        source_record_ids = [source_record.record_id]
        if self._active_ag_generation_plan is not None:
            plan_record = self.blackboard.publish(
                RecordType.SOURCE,
                "design.ag_generation_plan",
                "AGPlanningAgent",
                {
                    "stage": self._active_ag_generation_plan["stage"],
                    "mode": self._active_ag_generation_plan["mode"],
                    "source_requirements": list(
                        self._active_ag_generation_plan[
                            "source_requirements"
                        ]
                    ),
                    "package_names": [
                        spec.package
                        for spec in self._active_ag_generation_plan["specs"]
                    ],
                    "guidance_steps": sorted(
                        self._active_ag_generation_plan[
                            "guidance_by_step"
                        ]
                    ),
                    "gold_access": False,
                },
            )
            source_record_ids.append(plan_record.record_id)
        design_task = self.blackboard.create_task(
            "INITIAL_MODEL_GENERATION",
            "DesignAgent",
            required_topics=tuple(
                ["requirements.authoritative"]
                + (
                    ["design.ag_generation_plan"]
                    if self._active_ag_generation_plan is not None else []
                )
            ),
        )
        self.blackboard.transition_task(design_task.task_id, TaskStatus.ACTIVE)
        envelope = self.context_builder.build_design_context(
            task_id=design_task.task_id,
            system_name=system_name,
            source_record_ids=tuple(source_record_ids),
        )
        session = self.task_sessions.open(
            task_id=design_task.task_id,
            agent_role="DesignAgent",
            base_model_revision=design_task.base_model_revision,
            base_model_digest=design_task.base_model_digest,
            context_envelope_id=envelope.envelope_id,
            max_turns=self.task_session_max_turns,
            max_tokens=self.task_session_max_tokens,
        )
        self._active_design_handoff = {
            "task": design_task,
            "envelope": envelope,
            "session": session,
            "captured_llm_calls": 0,
        }

    def _finalize_design_handoff(self, result: Any, model: SysMLModel) -> None:
        handoff = self._active_design_handoff
        if handoff is None:
            return
        from ..prototyping.blackboard import RecordType, TaskStatus, text_digest
        from ..prototyping.task_session import SessionStatus

        task = handoff["task"]
        session = handoff["session"]
        model_text = get_sysml_text(model)
        session.assert_current(
            self.blackboard.current_revision,
            self.blackboard.current_model.model_digest,
        )
        reasoning = str(getattr(result, "reasoning", "") or "")
        observer_errors = list(handoff.get("observer_errors", ()))
        if observer_errors or session.status is not SessionStatus.OPEN:
            detail = "; ".join(observer_errors) or session.status.value
            self._reject_design_handoff(
                f"incomplete DesignAgent transcript: {detail}",
                producer="Orchestrator",
            )
            raise RuntimeError(
                "R1-BBCTX rejected the DesignAgent result because its bounded "
                f"transcript was incomplete: {detail}"
            )
        if handoff.get("captured_llm_calls", 0) == 0:
            envelope_text = handoff["envelope"].render_for_prompt()
            try:
                self._append_session_message(
                    session,
                    "user",
                    envelope_text,
                    token_count=max(1, len(envelope_text) // 4),
                )
                self._append_session_message(
                    session,
                    "assistant",
                    reasoning or "DesignAgent returned a model candidate.",
                    token_count=max(1, len(reasoning) // 4) if reasoning else 1,
                )
            except Exception as exc:
                self._reject_design_handoff(
                    f"{type(exc).__name__}: {exc}", producer="Orchestrator"
                )
                raise
        success = bool(getattr(result, "success", False))
        result_record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.design.result",
            "DesignAgent",
            {
                "success": success,
                "reasoning": reasoning,
                "model_digest": text_digest(model_text),
                "input_model_revision": task.base_model_revision,
                "input_model_digest": task.base_model_digest,
                "context_envelope_id": handoff["envelope"].envelope_id,
                "context_envelope_digest": handoff["envelope"].envelope_digest,
                "transcript_digest": session.transcript_digest,
                "included_record_ids": list(
                    handoff["envelope"].included_record_ids
                ),
                "accepted_status": "ACCEPTED" if success else "REJECTED",
            },
            task_id=task.task_id,
            session_id=session.session_id,
        )
        task_status = TaskStatus.COMPLETED if success else TaskStatus.REJECTED
        session_status = SessionStatus.COMPLETED if success else SessionStatus.REJECTED
        self.blackboard.transition_task(
            task.task_id,
            task_status,
            producer="DesignAgent",
            result_record_ids=(result_record.record_id,),
        )
        session.close(session_status, output_record_ids=(result_record.record_id,))
        committed = self.blackboard.commit_model(
            model_text,
            base_revision=task.base_model_revision,
            base_digest=task.base_model_digest,
            producer="DesignAgent" if success else "OrchestratorFallback",
            task_id=task.task_id if success else None,
            session_id=session.session_id,
        )
        self.task_sessions.stale_after_commit(
            committed.revision, committed.model_digest
        )
        self._active_design_handoff = None

    def _reject_design_handoff(self, reason: str, *, producer: str) -> None:
        handoff = self._active_design_handoff
        if handoff is None:
            return
        from ..prototyping.blackboard import RecordType, TaskStatus
        from ..prototyping.task_session import SessionStatus

        task = handoff["task"]
        session = handoff["session"]
        record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.design.rejected",
            producer,
            {"success": False, "reason": str(reason)},
            task_id=task.task_id,
            session_id=session.session_id,
        )
        if task.status is TaskStatus.ACTIVE:
            self.blackboard.transition_task(
                task.task_id,
                TaskStatus.REJECTED,
                producer=producer,
                result_record_ids=(record.record_id,),
            )
        if session.status is SessionStatus.OPEN:
            session.close(
                SessionStatus.REJECTED,
                output_record_ids=(record.record_id,),
            )
        elif record.record_id not in session.output_record_ids:
            session.output_record_ids.append(record.record_id)
        self._active_design_handoff = None

    def _run_verification_handoff(self) -> Optional[Dict[str, Any]]:
        """Second board-mediated handoff: DesignAgent -> VerificationAgent (§15).

        The VerificationAgent knowledge source consumes the committed model the
        DesignAgent produced (as the relevant requirement-def context) plus the
        authoritative requirements, and publishes a typed per-requirement
        verification plan. Deterministic (no LLM, no gold); its purpose is to make
        the §13 coordination metrics' handoff/role denominators greater than one —
        two migrated handoffs instead of the illustrative single one.

        The authoritative requirements are re-affirmed at the terminal revision so
        the envelope references a current-revision SOURCE record (the design-time
        source record is pinned to an earlier revision and would read as stale).
        """
        if (
            self.blackboard is None
            or self.context_builder is None
            or self.task_sessions is None
        ):
            return None
        from ..prototyping.blackboard import RecordType, TaskStatus
        from ..prototyping.task_session import SessionStatus
        from ..prototyping.verification_planning import plan_verification

        source = None
        for record in self.blackboard.records(topic="requirements.authoritative"):
            source = record
        if source is None:
            return None
        reaffirmed = self.blackboard.publish(
            RecordType.SOURCE,
            "requirements.authoritative",
            "RequirementsAgent",
            {
                "requirements": list(source.payload.get("requirements", ())),
                "requirement_input_mode": source.payload.get(
                    "requirement_input_mode"
                ),
                "requirement_set_digest": source.payload.get(
                    "requirement_set_digest"
                ),
                "gold_access": False,
                "reaffirmed_for": "verification_planning",
            },
        )
        task = self.blackboard.create_task(
            "VERIFICATION_PLANNING",
            "VerificationAgent",
            required_topics=("requirements.authoritative",),
        )
        self.blackboard.transition_task(task.task_id, TaskStatus.ACTIVE)
        envelope = self.context_builder.build_verification_context(
            task_id=task.task_id,
            source_record_ids=(reaffirmed.record_id,),
        )
        session = self.task_sessions.open(
            task_id=task.task_id,
            agent_role="VerificationAgent",
            base_model_revision=task.base_model_revision,
            base_model_digest=task.base_model_digest,
            context_envelope_id=envelope.envelope_id,
            max_turns=self.task_session_max_turns,
            max_tokens=self.task_session_max_tokens,
        )
        plan = plan_verification(envelope.model_context)
        envelope_text = envelope.render_for_prompt()
        self._append_session_message(
            session,
            "user", envelope_text, token_count=max(1, len(envelope_text) // 4)
        )
        summary = (
            f"Planned verification for {plan['planned']} requirement(s); "
            f"tiers {plan['tier_histogram']}."
        )
        self._append_session_message(
            session,
            "assistant",
            summary,
            token_count=max(1, len(summary) // 4),
        )
        result_record = self.blackboard.publish(
            RecordType.RESULT,
            "agent.verification.result",
            "VerificationAgent",
            {
                "success": True,
                "context_envelope_id": envelope.envelope_id,
                "context_envelope_digest": envelope.envelope_digest,
                "transcript_digest": session.transcript_digest,
                "included_record_ids": list(envelope.included_record_ids),
                "verification_plan": plan,
                "accepted_status": "ACCEPTED",
            },
            task_id=task.task_id,
            session_id=session.session_id,
        )
        self.blackboard.transition_task(
            task.task_id,
            TaskStatus.COMPLETED,
            producer="VerificationAgent",
            result_record_ids=(result_record.record_id,),
        )
        session.close(
            SessionStatus.COMPLETED, output_record_ids=(result_record.record_id,)
        )
        return plan

    def _commit_terminal_model(self, model_text: str, *, producer: str) -> None:
        if self.blackboard is None:
            return
        from ..prototyping.blackboard import text_digest

        if self.blackboard.current_model.model_digest == text_digest(model_text):
            return
        committed = self.blackboard.commit_model(
            model_text,
            base_revision=self.blackboard.current_revision,
            base_digest=self.blackboard.current_model.model_digest,
            producer=producer,
        )
        self.task_sessions.stale_after_commit(
            committed.revision, committed.model_digest
        )

    def _synchronize_terminal_snapshot(
        self,
        model: SysMLModel,
        model_text: str,
        requirements: List[str],
        *,
        prior_score: float,
        dse_best_config: Optional[DesignConfiguration],
    ) -> tuple[SysMLModel, float, SimulationResult, Dict[str, Any]]:
        """Recompute every terminal verdict from the exact returned SysML text.

        Refinement helpers may accept a partially improving deterministic edit
        after the last scored iteration, and the A/G assurance layer may make a
        final bounded repair after ordinary refinement.  Consequently, reusing
        an earlier score or simulation can pair evidence from revision N with
        model text from revision N+1.  This gate makes the returned model text
        the single source of truth and records its digest on all derived
        evidence.
        """
        from ..prototyping.blackboard import text_digest

        model_digest = text_digest(model_text)
        model_name = getattr(model, "name", None) or (
            self.state.system_name if self.state is not None else "System"
        )
        # Reparse the exact terminal text instead of retaining structural
        # collections from a pre-reconciliation model object.
        terminal_model = build_lite_model(model_text, model_name=model_name)
        prior_metadata = dict(getattr(model, "metadata", None) or {})
        terminal_model.metadata.update(prior_metadata)
        self._sync_model_text(terminal_model, model_text)
        syntax_result = check_syntax(model_text)
        sim_result = self._run_simulation(model_text, model_name)
        evaluation = self.evaluator.evaluate(
            config=DesignConfiguration(
                name="terminal_snapshot",
                parameters={},
            ),
            model=terminal_model,
            dse_config=dse_best_config,
            syntax_result=syntax_result,
            sim_result=sim_result,
            requirements=requirements,
        )
        final_score = evaluation.weighted_total
        consistency = {
            "schema_version": "1.0",
            "status": "PASS",
            "model_digest": model_digest,
            "simulation_source_model_digest": model_digest,
            "evaluation_source_model_digest": model_digest,
            "score_kind": "DETERMINISTIC_TERMINAL_RULE_SCORE",
            "final_score": final_score,
            "pre_terminal_iteration_score": prior_score,
            "syntax_error_count": syntax_result.total_errors(),
            "simulation": {
                "reachability_score": sim_result.reachability_score,
                "scenarios_passed": len(sim_result.passed_scenarios()),
                "scenarios_total": len(sim_result.scenario_results),
                "requirement_reachability_score": (
                    sim_result.requirement_reachability_score
                ),
                "requirement_scenarios_passed": (
                    sim_result.requirement_scenarios_passed
                ),
                "requirement_scenarios_total": (
                    sim_result.requirement_scenarios_total
                ),
                "role_scenarios_advisory": (
                    sim_result.requirement_reachability_score is not None
                ),
                "advisory_diagnostic": (
                    sim_result.advisory_structural_evidence()
                ),
            },
        }
        terminal_model.metadata["terminal_consistency"] = consistency
        return terminal_model, final_score, sim_result, consistency

    def _enforce_terminal_generation_plan(
        self,
        model: SysMLModel,
        model_text: str,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Re-materialise and check the typed plan before the terminal commit."""
        metadata = dict(getattr(model, "metadata", None) or {})
        raw_plan = metadata.get("whole_model_generation_plan")
        if (
            not isinstance(raw_plan, Mapping)
            and isinstance(self._active_model_generation_plan, Mapping)
        ):
            raw_plan = self._active_model_generation_plan
            model.metadata["whole_model_generation_plan"] = dict(raw_plan)
        if not isinstance(raw_plan, Mapping):
            return model_text, None
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        from ..prototyping.generation_plan import (
            PLAN_APPLICATION_HISTORY_KEY,
            ModelGenerationPlan,
            append_plan_application_history,
            apply_generation_plan,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        previous_conformance = metadata.get(
            "generation_plan_conformance"
        )
        planned_text, conformance = apply_generation_plan(model_text, plan)
        history = metadata.get(PLAN_APPLICATION_HISTORY_KEY)
        if not isinstance(history, list):
            history = []
        if isinstance(previous_conformance, Mapping):
            archived = previous_conformance.get(
                PLAN_APPLICATION_HISTORY_KEY
            )
            if not history and isinstance(archived, list):
                history = [
                    dict(item)
                    for item in archived
                    if isinstance(item, Mapping)
                ]
        model.metadata[PLAN_APPLICATION_HISTORY_KEY] = history
        history = append_plan_application_history(
            model.metadata,
            conformance,
            stage="TERMINAL",
        )
        conformance[PLAN_APPLICATION_HISTORY_KEY] = history
        legacy_semantic_history: list[dict[str, Any]] = []
        if isinstance(previous_conformance, Mapping):
            legacy_semantic_history.extend(
                dict(item)
                for item in (
                    previous_conformance.get(
                        "semantic_binding_materialization_history"
                    ) or ()
                )
                if isinstance(item, Mapping)
            )
        legacy_semantic_history.extend(
            item for item in history
            if item.get("semantic_changes")
        )
        conformance["semantic_binding_materialization_history"] = (
            legacy_semantic_history
        )
        if plan.behavior_obligations:
            from ..prototyping.ag_behavior_plan import (
                BehaviorObligationPlan,
                materialize_owned_behavior_obligations,
            )

            behavior_plan = BehaviorObligationPlan(
                plan.behavior_obligations
            )
            planned_text, behavior_gate = (
                materialize_owned_behavior_obligations(
                    planned_text,
                    behavior_plan,
                    event_symbols=plan.planned_event_symbols,
                )
            )
            conformance["behavior_obligation_conformance"] = behavior_gate
            conformance["restored_behavior_elements"] = (
                list(behavior_gate["materialized"])
                + list(behavior_gate["replaced_inconsistent"])
            )
            if behavior_gate["status"] != "PASS":
                conformance["status"] = "FAIL"
        model.metadata["generation_plan_conformance"] = conformance
        return planned_text, conformance

    def _validate_terminal_structural_obligations(
        self,
        model: SysMLModel,
        model_text: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check the terminal model against its frozen requirement paths."""
        metadata = dict(getattr(model, "metadata", None) or {})
        raw_plan = metadata.get("whole_model_generation_plan")
        if not isinstance(raw_plan, Mapping):
            raw_plan = getattr(
                self,
                "_active_model_generation_plan",
                None,
            )
        if not isinstance(raw_plan, Mapping):
            return None
        from ..prototyping.generation_plan import ModelGenerationPlan
        from ..prototyping.structural_obligations import (
            validate_structural_obligations,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        return validate_structural_obligations(
            model_text,
            plan.structural_obligations,
            model_name=model_name,
        )

    def _validate_terminal_semantic_obligations(
        self,
        model: SysMLModel,
        model_text: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check source-derived numeric semantics on the terminal revision."""
        metadata = dict(getattr(model, "metadata", None) or {})
        raw_plan = metadata.get("whole_model_generation_plan")
        if not isinstance(raw_plan, Mapping):
            raw_plan = getattr(
                self,
                "_active_model_generation_plan",
                None,
            )
        if not isinstance(raw_plan, Mapping):
            return None
        from ..prototyping.generation_plan import ModelGenerationPlan
        from ..prototyping.requirement_semantics import (
            validate_requirement_semantic_obligations,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        report = validate_requirement_semantic_obligations(
            model_text,
            plan.semantic_obligations,
            model_name=model_name,
            bindings=plan.semantic_bindings,
        )
        model.metadata["semantic_fidelity_report"] = report
        return report

    def _restore_generation_plan_metadata(self, model: SysMLModel) -> None:
        """Keep the frozen typed plan across reparsing/refinement objects."""
        if not isinstance(self._active_model_generation_plan, Mapping):
            return
        if getattr(model, "metadata", None) is None:
            model.metadata = {}
        model.metadata.setdefault(
            "whole_model_generation_plan",
            dict(self._active_model_generation_plan),
        )

    #: Topic carrying per-step generation drafts. Deliberately its own topic so
    #: it is easy to see — and easy to refuse. `ContextBuilder` accepts only
    #: `requirements.authoritative` and `design.ag_generation_plan` as design
    #: sources, so a draft can never become the input a later stage builds from.
    GENERATION_FRAGMENT_TOPIC = "generation.fragment"

    def _publish_generation_fragment(
        self, handoff: Mapping[str, Any], event: Mapping[str, Any]
    ) -> None:
        """Record one generation step's draft on the board — ARCHIVAL ONLY.

        Before this, the board saw nothing between the DesignAgent task opening
        and the finished model being committed: the five intermediate drafts
        existed only as session-transcript turns, with no topic, so no knowledge
        source could subscribe to them and no coordination metric covered them.

        It is emphatically not a second authority. Element identity is carried
        by the validated, digest-bound `ModelGenerationPlan` and enforced at
        terminal compilation; reading names out of a raw draft would be the
        external-JSON-authority design §1 deliberately removed. Hence
        `authority: NONE_ARCHIVAL_ONLY`, and a ContextBuilder that refuses this
        topic as a source.
        """
        if self.blackboard is None:
            return
        from ..prototyping.blackboard import RecordType, text_digest

        response = event.get("response", {}) or {}
        fragment = str(response.get("content", ""))
        task = handoff.get("task")
        session = handoff.get("session")
        self.blackboard.publish(
            RecordType.ANALYSIS,
            self.GENERATION_FRAGMENT_TOPIC,
            "DesignAgent",
            {
                "artifact_role": "GENERATION_DRAFT_FRAGMENT",
                "authority": "NONE_ARCHIVAL_ONLY",
                "measurement_boundary": "INTERVENTION",
                "stage": event.get("label"),
                "call_index": int(handoff.get("captured_llm_calls", 0)),
                "conversation_id": event.get("conversation_id"),
                # membership of a conversation, not position in it: the opening
                # turn has offset 0 and is still part of one
                "multi_turn": event.get("conversation_id") is not None,
                "fragment_digest": text_digest(fragment),
                "fragment_chars": len(fragment),
                "fragment": fragment,
                "completion_tokens": int(
                    response.get("completion_tokens", 0) or 0
                ),
            },
            task_id=getattr(task, "task_id", None),
            session_id=getattr(session, "session_id", None),
        )

    def _archive_provider_call(self, session: Any, event: Mapping[str, Any]) -> None:
        """Archive one provider call into a task session, each turn charged once.

        A multi-turn call resends its earlier turns, so the provider's
        ``prompt_tokens`` covers content this session has already recorded.
        Charging that figure per call makes the session budget grow
        quadratically while the transcript grows linearly — the budget would
        then measure resends rather than accumulated context, and §13 defines
        session growth as what the session accumulates.  Each turn is therefore
        charged for its own content once; the real (cumulative, billed) provider
        cost stays in the TokenLedger, where cost belongs.
        """
        offset = int(event.get("new_message_offset", 0) or 0)
        for message in list(event.get("messages", ()))[offset:]:
            content = str(message.get("content", ""))
            self._append_session_message(
                session,
                str(message.get("role", "user")),
                content,
                token_count=max(1, len(content) // 4),
            )
        response = event.get("response", {})
        reply = str(response.get("content", ""))
        completion_tokens = int(response.get("completion_tokens", 0) or 0)
        self._append_session_message(
            session,
            "assistant",
            reply,
            token_count=completion_tokens or max(1, len(reply) // 4),
        )

    def _append_session_message(
        self,
        session: Any,
        role: str,
        content: str,
        *,
        token_count: int = 0,
    ) -> None:
        """Archive a turn with the model/event position at which it was used."""
        stamp: Dict[str, Any] = {}
        if self.blackboard is not None:
            stamp = {
                "model_revision": self.blackboard.current_revision,
                "model_digest": self.blackboard.current_model.model_digest,
                "board_sequence": self.blackboard.event_sequence,
            }
        session.append(
            role,
            content,
            token_count=token_count,
            **stamp,
        )

    def _build_collaboration_artifacts(
        self, model_text: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.revised_experiment_arm is None:
            return {}
        from ..prototyping.experiment_arms import (
            RevisedExperimentArm,
            revised_arm_metadata,
        )

        result: Dict[str, Any] = {
            # report the mode this orchestrator actually ran, not a fixed default
            "revised_experiment": revised_arm_metadata(
                self.revised_experiment_arm, self.r2_generation_mode
            )
        }
        if self.revised_experiment_arm is RevisedExperimentArm.SEMANTIC_ASSURANCE:
            authoring_attempts = list(self.last_ag_authoring_attempts)
            result["ag_authoring_attempts"] = {
                "schema_version": "1.0",
                "artifact_role": "R2_AUTHORED_QUALITY_ATTEMPTS",
                "feedback_policy": (
                    "ONE_FREEFORM_THEN_ADAPTIVE_GOLD_BLIND_STRUCTURED"
                ),
                "maximum_attempts_per_chain": (
                    self.r2_authored_syntax_max_attempts
                ),
                # These are intervention-level regeneration attempts. Provider
                # transport retries in llm_usage.retries are a different metric.
                "attempt_count": len(authoring_attempts),
                "authoring_retry_count": sum(
                    1
                    for item in authoring_attempts
                    if int(item.get("attempt", 1)) > 1
                ),
                "syntax_feedback_retry_count": sum(
                    item.get("feedback_received_scope") == "SYNTAX_ONLY"
                    for item in authoring_attempts
                ),
                "ag_feedback_retry_count": sum(
                    item.get("feedback_received_scope") == "AG_SEMANTIC"
                    for item in authoring_attempts
                ),
                "freeform_sysml_attempt_count": sum(
                    item.get("generation_strategy") == "FREEFORM_SYSML"
                    for item in authoring_attempts
                ),
                "structured_emitter_attempt_count": sum(
                    item.get("generation_strategy")
                    == "STRUCTURED_DECISIONS_DETERMINISTIC_EMITTER"
                    for item in authoring_attempts
                ),
                "retryable_decision_rejection_count": sum(
                    item.get("decision_failure_disposition")
                    == "RETRYABLE_VALIDATION_ERROR"
                    for item in authoring_attempts
                ),
                "architecture_input_required_count": sum(
                    item.get("decision_failure_disposition")
                    == "NEEDS_ARCHITECTURE_INPUT"
                    for item in authoring_attempts
                ),
                "attempts": authoring_attempts,
            }
        if self.blackboard is not None:
            from ..prototyping.blackboard import RecordType, text_digest

            if model_text is not None and (
                text_digest(model_text)
                != self.blackboard.current_model.model_digest
            ):
                raise ValueError(
                    "collaboration artifact input does not match the committed "
                    "Blackboard model revision/digest"
                )
            # Explicit current-revision activation fact. Controller preconditions
            # never use stale topics from an earlier model revision.
            terminal_records = [
                record
                for record in self.blackboard.records(
                    topic="model.terminal.ready"
                )
                if record.model_revision == self.blackboard.current_revision
                and record.model_digest
                == self.blackboard.current_model.model_digest
            ]
            if not terminal_records:
                design_results = self.blackboard.records(
                    topic="agent.design.result"
                )
                self.blackboard.publish(
                    RecordType.CONTROL,
                    "model.terminal.ready",
                    "Orchestrator",
                    {
                        "model_revision": self.blackboard.current_revision,
                        "model_digest": self.blackboard.current_model.model_digest,
                        "upstream_design_result_record_id": (
                            design_results[-1].record_id
                            if design_results else None
                        ),
                    },
                )
            # Event-driven control: the Blackboard Controller opportunistically
            # activates each registered downstream knowledge source once the board
            # satisfies current-revision typed preconditions. Verification
            # planning consumes the terminal-model fact and publishes its result;
            # R2 assurance then consumes both. Runs before the snapshot so the
            # complete agenda and typed outputs are captured.
            from ..prototyping.controller import (
                BlackboardController,
                KnowledgeSource,
            )

            controller = BlackboardController(self.blackboard)
            has_authoritative_source = bool(
                self.blackboard.records(topic="requirements.authoritative")
            )
            if has_authoritative_source:
                controller.register(KnowledgeSource(
                    name="verification_planning",
                    agent_role="VerificationAgent",
                    precondition_topics=("model.terminal.ready",),
                    activate=self._run_verification_handoff,
                    output_topics=("agent.verification.result",),
                ))
            if (
                self.revised_experiment_arm
                is RevisedExperimentArm.SEMANTIC_ASSURANCE
                and model_text is not None
            ):
                assurance_preconditions = ["model.terminal.ready"]
                if has_authoritative_source:
                    assurance_preconditions.append(
                        "agent.verification.result"
                    )
                controller.register(KnowledgeSource(
                    name="ag_semantic_assurance",
                    agent_role="AssuranceAgent",
                    precondition_topics=tuple(assurance_preconditions),
                    activate=lambda: self._build_ag_trace(model_text),
                    output_topics=("analysis.ag_trace",),
                ))
            for activation in controller.run():
                if (
                    activation["knowledge_source"] == "verification_planning"
                    and activation["result"] is not None
                ):
                    result["verification_plan"] = activation["result"]
                if activation["knowledge_source"] == "ag_semantic_assurance":
                    assurance = activation["result"]
                    result.update(assurance)
                    result["revised_experiment"].update({
                        "runtime_assurance_status": (
                            "PASS"
                            if assurance["ag_contract_graph"]["verdict"] == "PASS"
                            and assurance["pattern_conformance_report"]["verdict"]
                            == "PASS"
                            else "FAILED_OR_INCOMPLETE"
                        ),
                        "formal_ag_proof": False,
                        "physical_verification": False,
                    })
            result["control_agenda"] = controller.agenda()
            result["collaboration"] = {
                "blackboard": self.blackboard.snapshot(),
                "contexts": self.context_builder.snapshot(),
                "task_sessions": self.task_sessions.snapshot(
                    include_messages=True
                ),
            }
        return result

    def _apply_ag_contract_layer(
        self, model_text: str, requirements: List[str]
    ) -> str:
        """Merge the reviewed bounded A/G contract layer into the model (R2 only).

        A/G-aware generation (Stage 2-3, §12): under `R2-BBAG`, the reviewed
        decomposition for each selected requirement chain is emitted as valid SysML
        and merged so the committed model carries the contracts. R0/R1 models must
        never carry them (R1 is coordination-only). R2 is fail-closed: a missing
        selected chain or failed syntax gate aborts rather than silently running R1.
        """
        from ..prototyping.experiment_arms import RevisedExperimentArm

        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return model_text
        self.last_ag_authoring_attempts = []
        self.ag_input_dispositions = {}
        try:
            from ..prototyping.ag_chains import select_ag_chains
            from ..prototyping.ag_emitter import emit_ag_package
            from ..prototyping.experiment_arms import (
                R2_LLM_AUTHORED_GENERATION_MODE,
                R2_LLM_DECIDED_GENERATION_MODE,
            )

            specs = select_ag_chains(requirements)
            if not specs:
                raise ValueError(
                    "R2-BBAG MVP requires the reviewed REQ_SAFE_005 chain"
                )
            for spec in specs:
                if not re.search(
                    rf"\brequirement\s+def\s+{re.escape(spec.source_requirement)}\b",
                    model_text,
                ):
                    raise ValueError(
                        f"committed base model is missing authoritative "
                        f"{spec.source_requirement}; A/G emission cannot invent it"
                    )

            # The generated base model has already passed the syntax gate.
            # Re-check it here so a later terminal mutation cannot smuggle a new
            # parser/reference error into R2.  This gate uses the same narrowly
            # final source policy. No state-machine spelling is allowlisted.
            base_gate = check_syntax(
                model_text,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            if base_gate.has_errors or base_gate.warnings:
                raise RuntimeError(
                    "committed base model failed the strict shared syntax gate: "
                    f"{base_gate.short_summary()} (score={base_gate.score:.3f})"
                )

            # Produce one A/G package per selected chain, then gate each raw
            # package with *no* diagnostic filtering. In DETERMINISTIC_SPEC_EMITTER
            # mode the reviewed spec is rendered; in LLM_AUTHORED_AG mode the LLM
            # authors it from the requirement + approved architecture. Either way an
            # A/G syntax/reference defect must fail closed rather than be attributed
            # to the base model's known standard-library diagnostics — so an LLM
            # authoring error is honestly a failed R2 run, not a silent downgrade.
            llm_authored = (
                self.r2_generation_mode == R2_LLM_AUTHORED_GENERATION_MODE
            )
            llm_decided = (
                self.r2_generation_mode == R2_LLM_DECIDED_GENERATION_MODE
            )
            packages: list[str] = []
            for spec in specs:
                if llm_authored:
                    package_text, package_gate = (
                        self._generate_llm_authored_ag_until_quality_valid(
                            spec, model_text
                        )
                    )
                elif llm_decided:
                    # the LLM decides; the emitter renders, so the notation is
                    # conformant by construction and only the decisions are judged
                    package_text = emit_ag_package(
                        self._generate_llm_decided_ag_spec(spec, model_text)
                    )
                    package_gate = check_syntax(
                        package_text,
                        fail_closed=True,
                        filter_stdlib_diagnostics=False,
                    )
                else:
                    package_text = emit_ag_package(spec)
                    package_gate = check_syntax(
                        package_text,
                        fail_closed=True,
                        filter_stdlib_diagnostics=False,
                    )
                # Gate on ERRORS. The score is also depressed by warnings, and an
                # authored package legitimately earns benign ones (naming a state
                # `done` shadows a stdlib member). Rejecting on score threw away
                # valid packages and fed back "SYNTAX ERRORS" for a model that had
                # none. The strict score check survives only for the fully
                # deterministic path, where every identifier comes from a reviewed
                # spec: once the emitter renders concepts the MODEL chose, that
                # assumption no longer holds, and a benign warning would cost the
                # whole arm.
                if package_gate.has_errors or (
                    not llm_authored
                    and not llm_decided
                    and package_gate.score != 1.0
                ):
                    raise RuntimeError(
                        f"{spec.source_requirement} A/G package failed the raw "
                        f"syntax gate: {package_gate.short_summary()} "
                        f"(score={package_gate.score:.3f}); diagnostics: "
                        + " | ".join(
                            self._syntax_gate_diagnostic_lines(package_gate)
                        )
                    )
                packages.append(package_text)

            merged = model_text.rstrip() + "\n\n" + "\n\n".join(packages) + "\n"
            merged_gate = check_syntax(
                merged,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            if merged_gate.has_errors or merged_gate.warnings:
                raise RuntimeError(
                    "R2-BBAG A/G contract layer failed the merged syntax gate: "
                    f"{merged_gate.short_summary()} "
                    f"(score={merged_gate.score:.3f})"
                )
            print(
                f"  [R2-BBAG] merged {self.r2_generation_mode} A/G contract layer "
                f"for {len(specs)} selected chain(s)"
            )
            return merged
        except Exception as exc:
            raise RuntimeError(f"R2-BBAG failed closed: {exc}") from exc

    @staticmethod
    def _syntax_gate_diagnostic_lines(
        gate: SyntaxCheckResult,
        *,
        limit: int = 8,
    ) -> List[str]:
        """Bounded, actionable raw-gate diagnostics for logs and exceptions."""
        lines: List[str] = []
        for kind, items in (
            ("parser", gate.parser_errors),
            ("semantic", gate.sema_errors),
        ):
            for item in items[:max(0, limit - len(lines))]:
                lines.append(
                    f"{kind} L{item.get('line', '?')}: "
                    f"{item.get('message', 'unknown error')}"
                )
        omitted = gate.total_errors() - len(lines)
        if omitted > 0:
            lines.append(f"... {omitted} additional diagnostic(s) omitted")
        return lines

    def _generate_llm_authored_ag_until_quality_valid(
        self,
        spec,
        model_text: str,
        *,
        generation_stage: bool = False,
    ) -> Tuple[str, SyntaxCheckResult]:
        """Bounded, pre-commit syntax + gold-blind A/G-guided generation.

        The total authoring budget stays fixed. Syntax-invalid candidates receive
        only parser/reference feedback. Once syntax is valid, the same runtime
        checker and pattern profile used after commit provide semantic feedback.
        No evaluator gold is consulted. The first full PASS wins; otherwise the
        best syntax-valid candidate is returned for normal post-commit routing and
        surgical repair instead of discarding all generated engineering work.
        """
        from ..prototyping.ag_assurance import (
            check_safety_pattern_conformance,
        )
        from ..prototyping.ag_contracts import check_ag_graph
        from ..prototyping.ag_decision import extract_runtime_response_catalog
        from ..prototyping.ag_emitter import emit_ag_package
        from ..prototyping.ag_extractor import extract_ag_graph

        feedback: Optional[Dict[str, Any]] = None
        last_gate: Optional[SyntaxCheckResult] = None
        best: Optional[
            Tuple[Tuple[int, int, int], str, SyntaxCheckResult, int]
        ] = None
        total_attempts = self.r2_authored_syntax_max_attempts
        # One free-form call preserves the authored-SysML intervention. All
        # remaining calls are structured decisions rendered by the deterministic
        # emitter. They are a maximum, not a quota: PASS stops immediately, while
        # missing architecture facts stop without asking the model to invent them.
        structured_reserved = total_attempts >= 2
        freeform_attempts = 1
        latest_semantic_diagnostics = ""
        terminal_quality_disposition: Optional[str] = None
        runtime_response_catalog = (
            {"entries": [], "source": "pre_generation_design_decision"}
            if generation_stage
            else extract_runtime_response_catalog(model_text)
        )

        def evaluate_candidate(
            package_text: str,
            gate: SyntaxCheckResult,
            attempt_record: Dict[str, Any],
        ) -> Tuple[bool, str]:
            nonlocal best
            candidate_model = (
                model_text.rstrip() + "\n\n" + package_text.rstrip() + "\n"
            )
            graph = extract_ag_graph(candidate_model)
            report = check_ag_graph(graph)
            pattern = check_safety_pattern_conformance(graph, report)
            attempt_record["ag_verdict"] = report.verdict
            attempt_record["pattern_verdict"] = pattern["verdict"]
            attempt_record["ag_diagnostic_codes"] = [
                item.code for item in report.diagnostics
            ]
            verdict_rank = {"FAIL": 0, "INCOMPLETE": 1, "PASS": 2}
            rank = (
                verdict_rank.get(report.verdict, 0),
                int(pattern["verdict"] == "PASS"),
                -len(report.diagnostics),
            )
            if best is None or rank > best[0]:
                best = (
                    rank,
                    package_text,
                    gate,
                    len(self.last_ag_authoring_attempts) - 1,
                )
            pattern_lines = [
                f"- (PATTERN_CASE) {item.get('contract')} "
                f"{item.get('pattern')}: "
                f"{item.get('reason') or item.get('status')}"
                for item in pattern.get("cases", ())
                if item.get("status") != "PASS"
            ]
            diagnostics = self._format_ag_diagnostics(report)
            if pattern_lines:
                diagnostics += "\n" + "\n".join(pattern_lines)
            return (
                report.verdict == "PASS" and pattern["verdict"] == "PASS",
                diagnostics,
            )

        for attempt in range(1, freeform_attempts + 1):
            package_text = self._generate_llm_authored_ag_package(
                spec,
                model_text,
                feedback=feedback,
                generation_stage=generation_stage,
            )
            gate = check_syntax(
                package_text,
                fail_closed=True,
                filter_stdlib_diagnostics=False,
            )
            last_gate = gate
            syntax_ok = not gate.has_errors and not gate.warnings
            attempt_record = {
                "schema_version": "1.0",
                "artifact_role": "R2_AUTHORED_QUALITY_ATTEMPT",
                "source_requirement": spec.source_requirement,
                "attempt": attempt,
                "maximum_attempts": total_attempts,
                "generation_strategy": "FREEFORM_SYSML",
                "feedback_received_scope": (
                    str(feedback.get("feedback_scope"))
                    if feedback else "NONE"
                ),
                "syntax_ok": syntax_ok,
                "ag_verdict": None,
                "pattern_verdict": None,
                "selected": False,
                "parser_error_count": len(gate.parser_errors),
                "semantic_reference_error_count": len(gate.sema_errors),
                "syntax_summary": gate.short_summary(),
                "diagnostics": self._syntax_gate_diagnostic_lines(gate),
                # Rejected candidates otherwise disappear before a model commit.
                # Retain them as intervention audit, never as model authority.
                "package_text": package_text,
                "response_catalog": runtime_response_catalog,
            }
            self.last_ag_authoring_attempts.append(attempt_record)
            if not syntax_ok:
                print(
                    f"  [R2-BBAG] {spec.source_requirement} authored package "
                    f"failed syntax/reference gate on attempt {attempt}/"
                    f"{total_attempts}",
                    flush=True,
                )
                feedback = {
                    "feedback_scope": "SYNTAX_ONLY",
                    "diagnostics": gate.format_for_llm(),
                    "previous_package": package_text,
                }
                continue

            passed, diagnostics = evaluate_candidate(
                package_text, gate, attempt_record
            )
            latest_semantic_diagnostics = diagnostics
            if passed:
                attempt_record["selected"] = True
                attempt_record["stop_reason"] = "first_gold_blind_quality_pass"
                return package_text, gate

            print(
                f"  [R2-BBAG] {spec.source_requirement} authored package "
                f"passed syntax but failed gold-blind A/G quality on attempt "
                f"{attempt}/{total_attempts}: "
                f"A/G={attempt_record['ag_verdict']}, "
                f"pattern={attempt_record['pattern_verdict']}",
                flush=True,
            )
            feedback = {
                "feedback_scope": "AG_SEMANTIC",
                "diagnostics": diagnostics,
                "previous_package": package_text,
            }
        if structured_reserved:
            from ..prototyping.ag_decision import ArchitectureInputRequired

            structured_budget = total_attempts - freeform_attempts
            structured_calls = 0
            structured_feedback = latest_semantic_diagnostics

            def record_structured_decision(event: Dict[str, Any]) -> None:
                nonlocal structured_calls
                structured_calls += 1
                disposition = event.get("decision_failure_disposition")
                record: Dict[str, Any] = {
                    "schema_version": "1.0",
                    "artifact_role": "R2_AUTHORED_QUALITY_ATTEMPT",
                    "source_requirement": spec.source_requirement,
                    "attempt": freeform_attempts + structured_calls,
                    "maximum_attempts": total_attempts,
                    "generation_strategy": (
                        "STRUCTURED_DECISIONS_DETERMINISTIC_EMITTER"
                    ),
                    "feedback_received_scope": (
                        "DECISION_VALIDATION"
                        if int(event.get("decision_attempt", 1)) > 1
                        else (
                            "AG_SEMANTIC"
                            if structured_feedback
                            else "STRUCTURED_FALLBACK"
                        )
                    ),
                    # No SysML exists until a decision object validates and the
                    # emitter runs. `None` is deliberate; false would conflate a
                    # decision rejection with a raw syntax failure.
                    "syntax_ok": None,
                    "ag_verdict": None,
                    "pattern_verdict": None,
                    "selected": False,
                    "parser_error_count": 0,
                    "semantic_reference_error_count": 0,
                    "diagnostics": (
                        [event["decision_error"]]
                        if event.get("decision_error") else []
                    ),
                    "package_text": "",
                    "response_catalog": runtime_response_catalog,
                    **event,
                }
                if disposition == "NEEDS_ARCHITECTURE_INPUT":
                    record["stop_reason"] = "needs_architecture_input"
                self.last_ag_authoring_attempts.append(record)

            while structured_calls < structured_budget:
                calls_before = structured_calls
                try:
                    structured_spec = self._generate_llm_decided_ag_spec(
                        spec,
                        model_text,
                        max_decision_attempts=structured_budget - structured_calls,
                        quality_feedback=structured_feedback,
                        decision_attempt_observer=record_structured_decision,
                        stop_on_architecture_input_gap=True,
                        generation_stage=generation_stage,
                    )
                except ArchitectureInputRequired:
                    terminal_quality_disposition = "NEEDS_ARCHITECTURE_INPUT"
                    self.ag_input_dispositions[spec.source_requirement] = {
                        "disposition": "NEEDS_ARCHITECTURE_INPUT",
                        "reason": (
                            self.last_ag_authoring_attempts[-1].get(
                                "decision_error"
                            )
                            if self.last_ag_authoring_attempts else None
                        ),
                    }
                    break
                except Exception as exc:
                    # Validation attempts have already been recorded individually.
                    # Preserve an unexpected pre-validation failure as well.
                    if structured_calls == calls_before:
                        record_structured_decision({
                            "decision_attempt": 1,
                            "decision_validation_status": "ERROR",
                            "decision_failure_disposition": (
                                "RETRYABLE_VALIDATION_ERROR"
                            ),
                            "decision_error": f"{type(exc).__name__}: {exc}",
                            "decision_response": "",
                        })
                    self.last_ag_authoring_attempts[-1]["syntax_summary"] = (
                        "structured decision generation failed"
                    )
                    break

                attempt_record = self.last_ag_authoring_attempts[-1]
                package_text = emit_ag_package(structured_spec)
                gate = check_syntax(
                    package_text,
                    fail_closed=True,
                    filter_stdlib_diagnostics=False,
                )
                last_gate = gate
                attempt_record.update({
                    "syntax_ok": not gate.has_errors and not gate.warnings,
                    "parser_error_count": len(gate.parser_errors),
                    "semantic_reference_error_count": len(gate.sema_errors),
                    "syntax_summary": gate.short_summary(),
                    "diagnostics": self._syntax_gate_diagnostic_lines(gate),
                    "package_text": package_text,
                })
                if gate.has_errors or gate.warnings:
                    structured_feedback = gate.format_for_llm()
                    continue

                passed, diagnostics = evaluate_candidate(
                    package_text, gate, attempt_record
                )
                latest_semantic_diagnostics = diagnostics
                if passed:
                    attempt_record["selected"] = True
                    attempt_record["stop_reason"] = (
                        "structured_gold_blind_quality_pass"
                    )
                    return package_text, gate
                # A valid decision set can still fail a gold-blind semantic gate.
                # If one bounded call remains, feed those named defects into it.
                structured_feedback = diagnostics
        if best is not None:
            _rank, package_text, gate, record_index = best
            self.last_ag_authoring_attempts[record_index]["selected"] = True
            self.last_ag_authoring_attempts[record_index]["stop_reason"] = (
                (
                    "needs_architecture_input_best_syntax_valid_candidate"
                    if terminal_quality_disposition
                    == "NEEDS_ARCHITECTURE_INPUT"
                    else "quality_budget_exhausted_best_syntax_valid_candidate"
                )
            )
            self.last_ag_authoring_attempts[record_index][
                "terminal_quality_disposition"
            ] = terminal_quality_disposition
            return package_text, gate
        if terminal_quality_disposition == "NEEDS_ARCHITECTURE_INPUT":
            raise RuntimeError(
                f"{spec.source_requirement} A/G authoring stopped after "
                f"{len(self.last_ag_authoring_attempts)} provider call(s): "
                "NEEDS_ARCHITECTURE_INPUT; no syntax-valid candidate was "
                "available for post-commit assurance"
            )
        assert last_gate is not None
        raise RuntimeError(
            f"{spec.source_requirement} A/G package failed the raw syntax gate "
            f"after {self.r2_authored_syntax_max_attempts} bounded authoring attempts: "
            f"{last_gate.short_summary()} (score={last_gate.score:.3f}); "
            "diagnostics: "
            + " | ".join(self._syntax_gate_diagnostic_lines(last_gate))
        )

    def _generate_llm_authored_ag_package(
        self,
        spec,
        model_text: str,
        feedback: Optional[Dict[str, Any]] = None,
        *,
        generation_stage: bool = False,
    ) -> str:
        """Author one chain's bounded A/G SysML package with the LLM.

        LLM_AUTHORED_AG mode, setup (C). The LLM is given the stakeholder
        requirement and the COMPLETE approved architecture from the frozen boundary
        — components, owners, and each component's interfaces (the concepts it
        consumes and produces) — plus the bounded SysML v2 convention. It is NOT
        given the evaluator gold, nor the reviewed discharge wiring / timing /
        priority / invariant facts: it must DERIVE the discharge edges, the
        realizing behaviour, and those safety facts itself. So a decomposition the
        LLM gets wrong yields a real generation accuracy below 1.0, and the A/G
        assurance then detects and (partly) repairs the residual defects — the
        robustness enhancement this measures. The output is gated and traced by the
        same pipeline; a malformed package fails the run closed.
        """
        match = re.search(
            rf"requirement\s+def\s+{re.escape(spec.source_requirement)}\b[^{{]*\{{"
            r"(.*?)\}",
            model_text,
            re.DOTALL,
        )
        requirement_body = (match.group(1).strip() if match else "").strip()
        architecture_blocks = []
        for comp in spec.components:
            assumptions = list(dict.fromkeys(
                f"{a.concept} "
                f"[{'ENVIRONMENT' if a.environment else 'INTERNAL'}]"
                for a in comp.assumptions
            ))
            assumption_concepts = {a.concept for a in comp.assumptions}
            lifecycle_inputs = list(dict.fromkeys(
                item for item in comp.interface_inputs
                if item not in assumption_concepts
            ))
            architecture_blocks.append(
                f"  - component `{comp.name}` (part `{comp.owner_usage}` : "
                f"{comp.owner_def})\n"
                f"      contract assumptions (consumes): "
                f"{', '.join(assumptions) or '(none)'}\n"
                f"      lifecycle/interface inputs (behavior triggers only; "
                f"do NOT turn these into assumptions unless also listed above): "
                f"{', '.join(lifecycle_inputs) or '(none)'}\n"
                f"      produces (its guarantees): {', '.join(comp.guarantees)}"
            )
        architecture = "\n".join(architecture_blocks)
        from ..prototyping.ag_decision import extract_runtime_response_catalog

        response_catalog = (
            {"entries": [], "source": "pre_generation_design_decision"}
            if generation_stage
            else extract_runtime_response_catalog(model_text)
        )
        response_entries = response_catalog.get("entries", ())
        response_catalog_text = "\n".join(
            f"  - `{item['response_id']}` from `{item['source_id']}` "
            f"({item['source_kind']})"
            for item in response_entries
        ) or "  (no provenance-backed competing responses found)"

        # On a feedback iteration the LLM is shown its previous attempt plus the
        # A/G checker's diagnostics and asked to regenerate a complete, corrected
        # package — the check gates every round, so the model converges toward a
        # verified, internally-complete decomposition (robustness enhancement).
        feedback_section = ""
        if feedback:
            diagnostics = str(feedback.get("diagnostics") or "").strip()
            previous = str(feedback.get("previous_package") or "").strip()
            if feedback.get("feedback_scope") == "SYNTAX_ONLY":
                feedback_section = (
                    "Your PREVIOUS attempt failed only the raw SysML "
                    "syntax/reference gate. No A/G semantic checker result is being "
                    "given to you. Regenerate the COMPLETE package, correcting the "
                    "listed notation/reference errors while preserving its "
                    "engineering decisions:\n"
                    f"{diagnostics}\n\n"
                    f"Your previous attempt was:\n{previous}\n\n"
                )
            else:
                feedback_section = (
                    "Your PREVIOUS attempt was checked and failed with these A/G "
                    "defects. Regenerate the COMPLETE package, fixing ALL of them "
                    f"while keeping what was already correct:\n{diagnostics}\n\n"
                    f"Your previous attempt was:\n{previous}\n\n"
                )
        # deferred: src.prototyping imports the orchestrator, so a module-level
        # import here is circular whenever the orchestrator is imported first
        from ..prototyping.ag_convention import render_authoring_rules

        system_prompt = (
            "You are a systems engineer authoring a bounded Assume-Guarantee "
            "decomposition in SysML v2.\n"
            "Rules the toolchain enforces (these are the notation's rules — the "
            "engineering content is still yours to derive):\n"
            f"{render_authoring_rules()}\n"
            "Output ONLY the SysML package."
        )
        prompt = (
            "Author the bounded A/G contract package for this requirement.\n\n"
            f"Stakeholder requirement {spec.source_requirement}:\n"
            f"{requirement_body}\n\n"
            "Approved component architecture — use exactly these components, "
            "owners, classified contract assumptions, lifecycle/interface inputs, "
            "and produced guarantees. You must DERIVE yourself: which producer "
            "discharges each INTERNAL contract assumption, and each "
            f"component's realizing behaviour:\n{architecture}\n\n"
            + (
                "This is pre-generation planning: no behavior exists yet. For a "
                "timed priority pattern, define the concrete competing safety "
                "response ids now as architecture decisions, give every member "
                "source_kind STUDENT_DERIVED_DESIGN_CONSTRAINT and a stable "
                "source_id, and use exactly that vocabulary in the realizing "
                "behavior you author. Do not use ports, commands, Boolean "
                "guarantees, or placeholders as selectable responses.\n\n"
                if generation_stage
                else
                "Runtime response catalog extracted from the committed model's "
                "existing arbiter entry actions. For a timed priority pattern, "
                "the response-set members must be exactly these response ids; do "
                "not use ports, commands, Boolean guarantees, or invented "
                "placeholders as responses. Copy each entry's source_kind/source_id "
                "into the required member provenance docs:\n"
                f"{response_catalog_text}\n\n"
            )
            + f"System contract: {spec.system_contract}, decomposing to the "
            f"components above; system observed guarantee: {spec.observation}. "
            "Write that observation as the distinct constraint "
            "`require constraint g_observed { <observation> }`; do not use an "
            "invariant or a differently named constraint as its substitute. Its "
            "first member MUST be the provenance line in the form given by rule 5 "
            f"for {spec.source_requirement} — you choose the safety_pattern and the "
            "timing_origin.\n"
            "For each component author its Boolean attributes, an assume constraint "
            "only for each listed contract assumption (name an ENVIRONMENT "
            "assumption `env_<concept>` and an INTERNAL assumption `a_<concept>`), "
            "and one atomic require constraint for each produced guarantee. Use "
            "lifecycle/interface inputs as behavior triggers without inventing "
            "assumptions. Also author the owning part and satisfy, realizing "
            "state-machine behavior, and decompose/realize/discharge dependencies "
            "so every INTERNAL assumption is discharged by the upstream component "
            f"that produces it.\n\n{feedback_section}"
            f"Wrap everything in `package {spec.package} "
            "{ ... }` and output ONLY that package."
        )
        raw = str(self.llm.chat(prompt, system_prompt=system_prompt))
        text = raw.replace("```sysml", "").replace("```", "").strip()
        index = text.find(f"package {spec.package}")
        if index == -1:
            index = text.find("package ")
        return (text[index:] if index != -1 else text).strip()

    @staticmethod
    def _format_ag_diagnostics(report) -> str:
        """Render the A/G checker's diagnostics as an LLM fix-request list."""
        lines = []
        for diagnostic in report.diagnostics:
            where = (
                f" [contract: {diagnostic.contract}]"
                if getattr(diagnostic, "contract", None) else ""
            )
            lines.append(f"- ({diagnostic.code}) {diagnostic.message}{where}")
        return "\n".join(lines) or "(no diagnostics)"

    def _generate_llm_decided_ag_spec(
        self,
        spec,
        model_text: str,
        max_decision_attempts: int = 3,
        quality_feedback: str = "",
        decision_attempt_observer: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
        stop_on_architecture_input_gap: bool = True,
        generation_stage: bool = False,
    ):
        """Ask the LLM for the engineering decisions and assemble the emitter spec.

        LLM_DECIDED_SPEC mode. The model never writes SysML: it returns which
        pattern the requirement instantiates, what starts the timing, how the
        deadline divides, which producer discharges each assumption, and how the
        responses are ordered. `ag_decision` validates that against the frozen
        architecture boundary and builds the spec; `ag_emitter` renders it.

        The reviewed answers stay withheld exactly as in the authored mode — the
        boundary supplies components, ownership and interfaces, nothing more — so
        agreement with gold remains a real accuracy measure. What changes is that a
        wrong decision now yields a well-formed model that is wrong, instead of an
        unparseable one whose engineering cannot be scored at all.
        """
        from ..prototyping.ag_decision import (
            INVARIANT_SOURCE_KINDS as KNOWN_INVARIANT_SOURCE_KINDS,
            KNOWN_PATTERNS as KNOWN_AG_PATTERNS,
            ArchitectureInputRequired,
            DecisionError,
            DecisionFailureDisposition,
            build_spec_from_decisions,
            classify_decision_failure,
            extract_runtime_response_catalog,
            extract_decisions,
        )
        from ..prototyping.ag_convention import (
            render_decision_field_rules,
            render_invariant_role_rules,
        )
        from ..prototyping.architecture_boundary import (
            build_architecture_boundary_draft,
        )

        boundary = build_architecture_boundary_draft(spec)
        if not generation_stage:
            boundary["response_catalog"] = extract_runtime_response_catalog(
                model_text
            )
        architecture = "\n".join(
            f"  - component_id: {item['component_id']}\n"
            f"      consumes: "
            f"{', '.join(item['interfaces']['consumes']) or '(none)'}\n"
            f"      produces: {', '.join(item['interfaces']['produces'])}"
            for item in boundary["components"]
        )
        response_catalog = "\n".join(
            f"  - {item['response_id']} "
            f"(source_kind={item['source_kind']}, "
            f"source_id={item['source_id']})"
            for item in (boundary.get("response_catalog") or {}).get(
                "entries", ()
            )
        ) or "  (none)"
        match = re.search(
            rf"requirement\s+def\s+{re.escape(spec.source_requirement)}\b[^{{]*\{{"
            r"(.*?)\}",
            model_text,
            re.DOTALL,
        )
        requirement_body = (match.group(1).strip() if match else "").strip()
        system_prompt = (
            "You are a systems engineer making the decisions behind a bounded "
            "Assume-Guarantee decomposition. You do NOT write SysML — a renderer "
            "does that. Return ONE JSON object, nothing else, with exactly these "
            "keys:\n"
            '{\n'
            '  "safety_pattern": one of '
            f'{list(KNOWN_AG_PATTERNS)},\n'
            '  "timing_origin": the assumption concept that starts the deadline,\n'
            '  "deadline_seconds": number or null,\n'
            '  "observation": what the system as a whole guarantees — a concept, '
            'or a bounded expression over concepts using not/and/or. Every '
            'concept it asserts positively must be one the architecture below '
            'PRODUCES: the decomposition has to support the observation, so an '
            'observation naming a concept no component produces is unsupported '
            'however well it paraphrases the requirement,\n'
            '  "system_assumptions": [concepts the system assumes of its '
            'environment],\n'
            '  "components": [ { "component_id": from the architecture below,\n'
            '      "latency_budget_seconds": number or null,\n'
            '      "timing_segment_required": true/false,\n'
            '      "assumptions": [ { "concept": ...,\n'
            '          "discharged_by": the component_id that produces it, or '
            'null if it is an environment input } ],\n'
            '      "lifecycle_events": [ concepts this component consumes as typed '
            'EVENTS rather than assumes — power-on, power-loss and the like. A '
            'component that ASSUMES its power-on event is not safe by default, it '
            'is safe once that event happens to have occurred; a default-safe '
            'component therefore lists its events here and assumes nothing ] } ],\n'
            '  "priority": null, or for a timed failsafe { "response_set_id": ..., '
            '"members": [...], "selected_response": ... },\n'
            '  "invariants": [] for a timed failsafe, or for an invariant pattern '
            '(STARTUP_INHIBIT / LOCKED_UNTIL_AUTHORISED_RELEASE) one entry per '
            'obligation:\n'
            '      { "invariant_id": bare identifier,\n'
            '        "antecedent": [ { "concept": ..., "negated": true/false } ],\n'
            '        "consequent": [ { "concept": ..., "negated": true/false } ],\n'
            f'        "source_kind": one of {list(KNOWN_INVARIANT_SOURCE_KINDS)} }}\n'
            '      Each side is a conjunction of possibly-negated concepts, read as '
            '"whenever the antecedent holds, the consequent must hold". Use '
            'STAKEHOLDER when the requirement states the obligation and '
            'STUDENT_DERIVED_DESIGN_CONSTRAINT when you inferred it.\n'
            '}\n'
            "Two of those fields carry obligations the checker will hold you to:\n"
            f"{render_decision_field_rules()}\n"
            "Each invariant pattern is defined by the roles its invariants fill; "
            "an invariant set that leaves a role unfilled has not stated the "
            "pattern. Which concepts fill the roles is yours to derive from the "
            "requirement — the roles themselves are the pattern:\n"
            f"{render_invariant_role_rules()}\n"
            "Decide these yourself from the requirement: the pattern, the timing "
            "origin, the deadline and how it divides across components, which "
            "producer discharges each assumption, the response ordering, and — for "
            "an invariant pattern — the invariants that must always hold. An "
            "invariant pattern carries no deadline and no priority; a timed "
            "failsafe carries no invariants."
        )
        prompt = (
            f"Requirement {spec.source_requirement}:\n{requirement_body}\n\n"
            "Approved architecture — use exactly these components; you may not "
            f"add or rename any:\n{architecture}\n\n"
            + (
                "This decision happens before behavior generation. For a timed "
                "priority decision, define the concrete competing safety response "
                "ids now; they become the frozen vocabulary that downstream "
                "behavior must implement. Do not substitute interface signals, "
                "Boolean guarantees, commands, or placeholders. Give the set a "
                "stable response_set_id and decide which member wins.\n\n"
                if generation_stage
                else
                "Provenance-backed safety response catalog extracted from existing "
                "arbiter behavior. For a timed priority decision, "
                "`priority.members` must contain exactly these response ids. Do "
                "not substitute interface signals/guarantees and do not invent "
                "placeholders. You still decide which member wins and therefore "
                f"define the ordering:\n{response_catalog}\n\n"
            )
            + (
                "A previous free-form SysML candidate was rejected by the "
                "gold-blind runtime checker. Correct these internal consistency "
                "defects in your structured decisions:\n"
                f"{quality_feedback}\n\n"
                if quality_feedback.strip()
                else ""
            )
            + "Return only the JSON decision object."
        )
        # Decisions are small and structured, so a validation failure is worth
        # feeding back rather than discarding the run: the validator says exactly
        # what is incoherent, and the model only has to repair that field. The
        # validator still decides — a decision set that never becomes coherent
        # fails closed, it is never patched here.
        #
        # The retry is a real multi-turn conversation (§2 reasoning continuity):
        # the rejected decision object stays in the transcript as the model's own
        # assistant turn, so "keep everything that was already valid" refers to
        # something it can actually see. The turns are application-owned, so what
        # each turn saw remains reproducible and digest-recordable.
        from ..llm.interface import Conversation

        conversation = Conversation(self.llm, system_prompt=system_prompt)
        attempt_prompt = prompt
        last: Optional[DecisionError] = None
        for attempt in range(max_decision_attempts):
            raw = str(conversation.send(attempt_prompt))
            try:
                decided_spec = build_spec_from_decisions(
                    extract_decisions(raw), boundary
                )
                if generation_stage:
                    # Validate the compiled artifact, not only the JSON schema.
                    # This catches cross-field/emitter inconsistencies before the
                    # ordinary architecture spends four more provider calls.
                    from ..prototyping.ag_extractor import extract_ag_graph
                    from ..prototyping.ag_planning import (
                        check_ag_planning_graph,
                        emit_ag_planning_package,
                    )

                    # Pre-generation can validate contract decomposition,
                    # discharge, sufficiency, and timing. Ownership, executable
                    # behavior, and safety topology are terminal obligations.
                    candidate_package = emit_ag_planning_package(decided_spec)
                    candidate_graph = extract_ag_graph(
                        f"{model_text}\n{candidate_package}"
                    )
                    candidate_report = check_ag_planning_graph(
                        candidate_graph
                    )
                    if candidate_report.verdict != "PASS":
                        raise DecisionError(
                            "compiled A/G candidate failed the gold-blind "
                            "pre-generation gate: "
                            + self._format_ag_diagnostics(candidate_report)
                        )
                if decision_attempt_observer is not None:
                    decision_attempt_observer({
                        "decision_attempt": attempt + 1,
                        "decision_validation_status": "VALID",
                        "decision_failure_disposition": None,
                        "decision_error": None,
                        "decision_response": raw,
                    })
                return decided_spec
            except DecisionError as exc:
                last = exc
                disposition = classify_decision_failure(exc)
                if decision_attempt_observer is not None:
                    decision_attempt_observer({
                        "decision_attempt": attempt + 1,
                        "decision_validation_status": "REJECTED",
                        "decision_failure_disposition": disposition.value,
                        "decision_error": f"{type(exc).__name__}: {exc}",
                        "decision_response": raw,
                    })
                if (
                    stop_on_architecture_input_gap
                    and disposition
                    is DecisionFailureDisposition.NEEDS_ARCHITECTURE_INPUT
                ):
                    raise ArchitectureInputRequired(exc) from exc
                # The requirement, architecture and schema are already in the
                # conversation, so the follow-up turn carries only what is new.
                # Repasting the whole prompt would re-state them as if unsaid.
                attempt_prompt = (
                    f"Your previous decisions were rejected: {exc}\n"
                    "Return the corrected JSON decision object, keeping everything "
                    "that was already valid."
                )
        raise RuntimeError(
            f"{spec.source_requirement} LLM decisions failed closed after "
            f"{max_decision_attempts} attempts: {last}"
        )

    def _author_llm_ag_with_feedback(
        self, spec, model_text: str, max_iterations: int = 4
    ) -> Dict[str, Any]:
        """Iteratively author a chain's A/G with the LLM under A/G-check feedback.

        The LLM authors the bounded A/G; the committed-convention checker verifies
        the merged model; on a non-PASS verdict the diagnostics + the previous
        attempt are fed back and the LLM regenerates. The check gates every round,
        so the model converges toward a verified, internally-complete decomposition
        (verdict PASS, guarantees realised, assumptions discharged, patterns
        conformant) — the robustness enhancement. The check is the oracle, so this
        is sound; it maximises internal completeness/verifiability, not correctness
        against a reviewed gold (that remains the gold's job). Bounded by
        ``max_iterations``; returns the lowest-error attempt with its history.
        """
        from ..prototyping.ag_contracts import check_ag_graph
        from ..prototyping.ag_extractor import extract_ag_graph

        authored = self._generate_llm_authored_ag_package(spec, model_text)
        best_package = authored
        best_errors: Optional[int] = None
        best_verdict: Optional[str] = None
        best_codes: Dict[str, int] = {}
        history: List[Dict[str, Any]] = []
        # Every attempt is retained: a rejected round is the evidence for *why*
        # the loop did or did not converge, and a syntax-rejected package is
        # otherwise unrecoverable (it never reaches the returned model).
        attempts: List[str] = []
        for iteration in range(max_iterations):
            attempts.append(authored)
            merged = model_text.rstrip() + "\n\n" + authored + "\n"
            gate = check_syntax(
                authored, fail_closed=True, filter_stdlib_diagnostics=False
            )
            # errors only — a warning-depressed score is not a syntax failure, and
            # rejecting on it discarded valid packages and fed back a defect list
            # for a model that had none, destabilising the next round
            if gate.has_errors or gate.warnings:
                history.append({
                    "iteration": iteration, "syntax_ok": False,
                    "verdict": None, "error_count": None,
                    "syntax_summary": gate.short_summary()[:120],
                    "syntax_diagnostics": [
                        str(item)[:200]
                        for item in (*gate.parser_errors, *gate.sema_errors)[:8]
                    ],
                })
                feedback = {
                    "diagnostics": f"SYNTAX ERRORS: {gate.short_summary()}",
                    "previous_package": authored,
                }
            else:
                report = check_ag_graph(extract_ag_graph(merged))
                errors = report.errors()
                error_count = len(errors)
                codes: Dict[str, int] = {}
                for diagnostic in errors:
                    codes[diagnostic.code] = codes.get(diagnostic.code, 0) + 1
                history.append({
                    "iteration": iteration, "syntax_ok": True,
                    "verdict": report.verdict, "error_count": error_count,
                    "error_codes": codes,
                })
                if best_errors is None or error_count < best_errors:
                    best_package, best_errors = authored, error_count
                    best_verdict, best_codes = report.verdict, codes
                if report.verdict == "PASS":
                    break
                feedback = {
                    "diagnostics": self._format_ag_diagnostics(report),
                    "previous_package": authored,
                }
            if iteration == max_iterations - 1:
                break
            authored = self._generate_llm_authored_ag_package(
                spec, model_text, feedback=feedback
            )
        return {
            "final_package": best_package,
            "final_merged": model_text.rstrip() + "\n\n" + best_package + "\n",
            # The delivered package is the lowest-error attempt, which is not
            # necessarily the last one (a later round can regress, even to invalid
            # syntax). Report the verdict of what is actually returned.
            "final_verdict": best_verdict,
            "final_error_count": best_errors,
            "final_error_codes": best_codes,
            "history": history,
            "attempts": attempts,
        }

    def _build_ag_trace(self, model_text: str) -> Dict[str, Any]:
        """R2-BBAG A/G intervention: extract and check the bounded A/G graph from
        the committed model and publish the diagnostics as a typed ANALYSIS record.

        The committed SysML model is the sole authority (§6.2): the graph is read
        out of the model, never supplied from JSON, so a model that carries no A/G
        contracts yields an honest INCOMPLETE trace rather than fabricated content.
        This is intervention evidence, not gold-scored accuracy — the derived view
        records ``evaluation_ready=False`` until the independent evaluator lands.
        """
        from ..prototyping.ag_extractor import (
            extract_ag_graph,
            extract_ag_graphs,
        )
        from ..prototyping.ag_repair import attempt_dependency_closed_ag_repair
        from ..prototyping.blackboard import text_digest

        if text_digest(model_text) != self.blackboard.current_model.model_digest:
            raise ValueError(
                "A/G extraction input does not match the committed Blackboard "
                "model revision/digest"
            )
        graphs = extract_ag_graphs(
            self.blackboard.current_model.model_text,
            revision=self.blackboard.current_revision,
            model_digest=self.blackboard.current_model.model_digest,
        )
        if len(graphs) > 1:
            # Several selected chains co-exist (e.g. the drone co-selects
            # REQ_SAFE_004 and REQ_SAFE_005): each is an independent A/G
            # decomposition and must be checked on its own graph.
            return self._build_multichain_ag_trace(graphs)

        maximum_repair_attempts = self.maximum_ag_repair_attempts
        repair_attempts = 0
        analysis_round = 0
        analysis_history: list[dict[str, Any]] = []

        while True:
            revision = self.blackboard.current_revision
            digest = self.blackboard.current_model.model_digest
            current_text = self.blackboard.current_model.model_text
            graph = extract_ag_graph(
                current_text, revision=revision, model_digest=digest
            )
            report, pattern, failures, analysis_record, repair_candidates = (
                self._run_ag_analysis_round(
                    graph,
                    analysis_round=analysis_round,
                    repair_attempts=repair_attempts,
                    maximum_repair_attempts=maximum_repair_attempts,
                    allow_repair=True,
                )
            )
            analysis_history.append({
                "analysis_round": analysis_round,
                "source_model_revision": revision,
                "source_model_digest": digest,
                "verdict": report.verdict,
                "pattern_verdict": pattern["verdict"],
                "failure_ids": [
                    item["failure_id"] for item in failures["failures"]
                ],
            })
            if not repair_candidates:
                break
            accepted = False
            candidate_queue = [
                (repair_candidate, 0)
                for repair_candidate in repair_candidates
            ]
            candidate_index = 0
            while candidate_index < len(candidate_queue):
                repair_candidate, retry_index = candidate_queue[candidate_index]
                if repair_attempts >= maximum_repair_attempts:
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="DEFERRED",
                            reason="automatic_repair_budget_exhausted",
                        )
                    break
                candidate_index += 1
                decision = attempt_dependency_closed_ag_repair(
                    llm=self.llm,
                    board=self.blackboard,
                    context_builder=self.context_builder,
                    sessions=self.task_sessions,
                    failure_record_id=repair_candidate[0],
                    analysis_record_id=repair_candidate[1],
                )
                repair_attempts += 1
                if decision.status == "ACCEPTED":
                    accepted = True
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="SUPERSEDED",
                            reason="superseded_by_committed_repair",
                        )
                    break
                if (
                    retry_index == 0
                    and repair_attempts < maximum_repair_attempts
                ):
                    retry = self._build_ag_repair_retry_candidate(
                        repair_candidate, retry_index=1
                    )
                    candidate_queue.insert(candidate_index, (retry, 1))
            if not accepted:
                break
            analysis_round += 1

        failures["analysis_history"] = analysis_history
        repair_decisions = [
            dict(item.payload)
            for item in self.blackboard.records(topic="repair.decision")
        ]
        return {
            "ag_contract_graph": report.to_dict(),
            "pattern_conformance_report": pattern,
            "failure_diagnostics": failures,
            "repair_decisions": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_REPAIR_DECISIONS",
                "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                "measurement_boundary": "INTERVENTION",
                "decisions": repair_decisions,
            },
            "_terminal_model_sysml": self.blackboard.current_model.model_text,
        }

    def _run_ag_analysis_round(
        self,
        graph,
        *,
        analysis_round: int,
        repair_attempts: int,
        maximum_repair_attempts: int,
        allow_repair: bool,
    ):
        """Run one A/G analysis round for a single chain graph.

        Checks the graph, publishes the typed ``analysis.ag_trace`` and
        ``analysis.pattern_conformance`` records, routes every diagnostic to a
        typed failure, and dispatches repair/blocked tasks. Returns
        ``(report, pattern, failures, analysis_record, repair_candidate)``.

        ``allow_repair`` gates dependency-closed surgical repair. Every named
        repairable obligation receives its own candidate. A committed patch may
        supersede the rest, but a rejected patch cannot silently suppress the
        next obligation in the fixed run-level budget.
        """
        from ..prototyping.ag_assurance import (
            FailureRoute,
            check_safety_pattern_conformance,
            route_failure_diagnostics,
        )
        from ..prototyping.ag_contracts import check_ag_graph
        from ..prototyping.blackboard import RecordType, TaskStatus

        revision = graph.revision
        digest = graph.model_digest
        report = check_ag_graph(graph)
        analysis_record = self.blackboard.publish(
            RecordType.ANALYSIS,
            "analysis.ag_trace",
            "AGChecker",
            {
                "analysis_round": analysis_round,
                "verdict": report.verdict,
                "system_completeness": report.system_completeness,
                "component_completeness": dict(report.component_completeness),
                "diagnostic_codes": [d.code for d in report.diagnostics],
                "diagnostics": [d.as_dict() for d in report.diagnostics],
                "checker_version": report.checker_version,
                "evaluation_ready": False,
            },
        )
        pattern = check_safety_pattern_conformance(graph, report)
        pattern["analysis_round"] = analysis_round
        self.blackboard.publish(
            RecordType.ANALYSIS,
            "analysis.pattern_conformance",
            "SafetyPatternChecker",
            pattern,
        )
        # Pattern conformance is already a first-class report. Do not synthesize a
        # generic PATTERN_NONCONFORMANT repair target: it collapses missing
        # invariants, contract vocabulary, and behavior topology into one code and
        # consequently authorizes a behavior-only repair for defects outside that
        # slice. Itemized checker diagnostics (for example
        # PATTERN_TOPOLOGY_INCOMPLETE) remain routable.
        routed_diags = list(report.diagnostics)
        failures = route_failure_diagnostics(
            routed_diags,
            source_requirement=report.source_requirement,
            realization_links=report.realization_links,
        )
        self._apply_ag_input_disposition(
            failures, str(report.source_requirement or "")
        )
        failures.update({
            "source_model_revision": revision,
            "source_model_digest": digest,
            "analysis_record_id": analysis_record.record_id,
            "analysis_round": analysis_round,
        })
        repair_candidates = []
        for failure in failures["failures"]:
            failure["failure_id"] = (
                f"{analysis_record.record_id}:{failure['failure_id']}"
            )
            failure["analysis_round"] = analysis_round
            failure_record = self.blackboard.publish(
                RecordType.ANALYSIS,
                "diagnostic.failure",
                "AGFailureRouter",
                failure,
            )
            if failure.get("repair_authorized"):
                if not allow_repair or repair_attempts >= maximum_repair_attempts:
                    blocked_task = self.blackboard.create_task(
                        "A_G_SURGICAL_REPAIR",
                        "RepairAgent",
                        required_topics=(
                            "analysis.ag_trace", "diagnostic.failure"
                        ),
                    )
                    blocked = self.blackboard.publish(
                        RecordType.RESULT,
                        "repair.decision",
                        "AGRepairController",
                        {
                            "failure_id": failure["failure_id"],
                            "status": "BLOCKED",
                            # `multi_chain_auto_repair_out_of_scope` used to be
                            # reported here whenever several chains coexisted.
                            # Repair now runs per chain as a fixpoint, so that
                            # exemption no longer exists and the only honest
                            # reasons left are budget and an explicitly disabled
                            # run. Archived artifacts still carry the old string.
                            "reason": (
                                "automatic_repair_budget_exhausted"
                                if allow_repair
                                else "automatic_repair_disabled_for_this_run"
                            ),
                            "base_model_revision": revision,
                            "base_model_digest": digest,
                            "committed_model_revision": None,
                            "target_diagnostic_removed": False,
                            "regression_free": False,
                            "whole_model_fallback_used": False,
                            "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                            "measurement_boundary": "INTERVENTION",
                        },
                        task_id=blocked_task.task_id,
                    )
                    self.blackboard.transition_task(
                        blocked_task.task_id,
                        TaskStatus.BLOCKED,
                        producer="AGRepairController",
                        result_record_ids=(blocked.record_id,),
                    )
                else:
                    repair_task = self.blackboard.create_task(
                        "A_G_SURGICAL_REPAIR",
                        "RepairAgent",
                        required_topics=(
                            "analysis.ag_trace", "diagnostic.failure"
                        ),
                    )
                    self.blackboard.publish(
                        RecordType.CONTROL,
                        "repair.routed",
                        "AGFailureRouter",
                        {
                            "failure_id": failure.get("failure_id"),
                            "failure_record_id": failure_record.record_id,
                            "analysis_record_id": analysis_record.record_id,
                            "repair_task_id": repair_task.task_id,
                            "whole_model_fallback_allowed": False,
                        },
                        task_id=repair_task.task_id,
                    )
                    repair_candidates.append((
                        failure_record.record_id,
                        analysis_record.record_id,
                        repair_task.task_id,
                    ))
            elif (
                failure.get("route")
                == FailureRoute.UPSTREAM_INTEGRATION_REPAIR.value
            ):
                blocked_task = self.blackboard.create_task(
                    "A_G_UPSTREAM_INTEGRATION_REPAIR",
                    "ArchitectureAgent",
                    required_topics=("analysis.ag_trace", "diagnostic.failure"),
                )
                blocked = self.blackboard.publish(
                    RecordType.RESULT,
                    "repair.decision",
                    "AGRepairController",
                    {
                        "failure_id": failure["failure_id"],
                        "status": "BLOCKED",
                        "reason": (
                            "bounded_mvp_has_no_authorised_upstream_"
                            "decomposition_repair"
                        ),
                        "base_model_revision": revision,
                        "base_model_digest": digest,
                        "committed_model_revision": None,
                        "target_diagnostic_removed": False,
                        "regression_free": False,
                        "whole_model_fallback_used": False,
                        "producing_stage": "R2_FAILURE_ROUTING",
                        "measurement_boundary": "INTERVENTION",
                    },
                    task_id=blocked_task.task_id,
                )
                self.blackboard.transition_task(
                    blocked_task.task_id,
                    TaskStatus.BLOCKED,
                    producer="AGRepairController",
                    result_record_ids=(blocked.record_id,),
                )
        return report, pattern, failures, analysis_record, repair_candidates

    def _apply_ag_input_disposition(
        self,
        failures: Dict[str, Any],
        source_requirement: str,
    ) -> None:
        """Make an upstream input gap dominate dependent local repair routes."""
        from ..prototyping.ag_assurance import FailureRoute

        input_gap = self.ag_input_dispositions.get(source_requirement)
        if input_gap:
            for failure in failures["failures"]:
                if failure.get("diagnostic_code") != (
                    "PRIORITY_TOPOLOGY_INCOMPLETE"
                ):
                    continue
                # The local wiring symptom is not independently repairable once
                # this run has established that its response vocabulary is absent.
                # Carry that upstream disposition forward instead of allowing the
                # RepairAgent to invent a response Signal in a behavior-only slice.
                failure.update({
                    "classification": "CONTRACT_INCOMPLETENESS",
                    "route": FailureRoute.CLARIFICATION_OR_BLOCKED.value,
                    "repair_authorized": False,
                    "routing_basis": (
                        "UPSTREAM_RESPONSE_CATALOG_INPUT_DISPOSITION"
                    ),
                    "input_disposition": dict(input_gap),
                })

    def _close_unattempted_ag_repair_candidate(
        self,
        candidate,
        *,
        status: str,
        reason: str,
    ) -> None:
        """Give an unattempted candidate an explicit terminal disposition.

        Blackboard has no SUPERSEDED/DEFERRED task state, so the task transitions
        to BLOCKED while the repair decision carries the semantically precise
        status. This prevents orphan PENDING tasks without misreporting an LLM
        attempt.
        """
        from ..prototyping.blackboard import RecordType, TaskStatus

        failure_record_id, _analysis_record_id, task_id = candidate
        task = self.blackboard.task(task_id)
        if task.status is not TaskStatus.PENDING:
            return
        failure = self.blackboard.record(failure_record_id).payload
        decision = self.blackboard.publish(
            RecordType.RESULT,
            "repair.decision",
            "AGRepairController",
            {
                "failure_id": failure.get("failure_id"),
                "status": status,
                "reason": reason,
                "base_model_revision": task.base_model_revision,
                "base_model_digest": task.base_model_digest,
                "committed_model_revision": None,
                "target_diagnostic_removed": False,
                "regression_free": False,
                "whole_model_fallback_used": False,
                "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                "measurement_boundary": "INTERVENTION",
            },
            task_id=task_id,
        )
        self.blackboard.transition_task(
            task_id,
            TaskStatus.BLOCKED,
            producer="AGRepairController",
            result_record_ids=(decision.record_id,),
        )

    def _build_ag_repair_retry_candidate(self, candidate, *, retry_index: int):
        """Route one bounded retry with the previous gate's actionable feedback."""
        from ..prototyping.blackboard import RecordType

        failure_record_id, analysis_record_id, previous_task_id = candidate
        original = dict(self.blackboard.record(failure_record_id).payload)
        previous_decisions = self.blackboard.records(
            topic="repair.decision", task_id=previous_task_id
        )
        previous = (
            dict(previous_decisions[-1].payload) if previous_decisions else {}
        )
        gate = previous.get("gate") or {}
        audit = previous.get("audit") or {}
        feedback_parts = [
            f"previous_status={previous.get('status')}",
            f"previous_reason={previous.get('reason')}",
        ]
        if gate:
            feedback_parts.extend([
                f"target_removed={gate.get('target_removed')}",
                f"regression_free={gate.get('regression_free')}",
                f"behavior_preserved={gate.get('behavior_preserved')}",
                f"new_diagnostics={gate.get('new_diagnostics') or []}",
            ])
        if audit:
            feedback_parts.append(
                f"merge_rejection_reasons="
                f"{audit.get('rejection_reasons') or []}"
            )
        feedback = "; ".join(feedback_parts)
        original_id = str(original.get("failure_id"))
        original.update({
            "failure_id": f"{original_id}:retry-{retry_index}",
            "original_failure_id": original_id,
            "repair_retry_index": retry_index,
            "previous_repair_feedback": feedback,
            "message": (
                f"{original.get('message') or original.get('diagnostic_code')}. "
                f"Previous bounded repair was rejected: {feedback}. Correct that "
                "specific failure while preserving the valid slice."
            ),
        })
        retry_failure = self.blackboard.publish(
            RecordType.ANALYSIS,
            "diagnostic.failure",
            "AGRepairController",
            original,
        )
        repair_task = self.blackboard.create_task(
            "A_G_SURGICAL_REPAIR",
            "RepairAgent",
            required_topics=("analysis.ag_trace", "diagnostic.failure"),
        )
        self.blackboard.publish(
            RecordType.CONTROL,
            "repair.routed",
            "AGRepairController",
            {
                "failure_id": original["failure_id"],
                "failure_record_id": retry_failure.record_id,
                "analysis_record_id": analysis_record_id,
                "repair_task_id": repair_task.task_id,
                "retry_index": retry_index,
                "previous_repair_task_id": previous_task_id,
                "whole_model_fallback_allowed": False,
            },
            task_id=repair_task.task_id,
        )
        return (
            retry_failure.record_id,
            analysis_record_id,
            repair_task.task_id,
        )

    def _build_multichain_ag_trace(self, graphs) -> Dict[str, Any]:
        """Aggregate independent per-chain A/G traces (several selected chains).

        Each chain is checked on its own graph and publishes its own typed
        ``analysis.ag_trace`` record — one assurance case per source requirement.
        The run-level verdict is the conjunction: PASS only when every chain is
        PASS.

        **Repair runs here too, under one fixed run-level budget, as a fixpoint.**
        It used to be disabled outright on the grounds that surgical repair is a
        single-chain capability. The repair itself is — it edits one dependency-
        closed slice — but disabling it for multi-chain runs had a consequence
        nobody had stated: every pilot selects three chains, so the Increment 3
        exit gate ("one authorised model-semantic failure is automatically routed,
        attempted, and rechecked against the committed revision") was never
        exercised by any evidence run, and every archived repair decision read
        `multi_chain_auto_repair_out_of_scope` — not a repair that failed, a repair
        that never ran.

        What made it more than a flag is staleness: an accepted repair commits a
        new revision, which invalidates every OTHER chain's graph. So each round
        re-extracts all chains from the current committed revision, analyses them,
        attempts authorized repairs in deterministic chain/diagnostic order, and
        — if one is accepted — starts a new round. A rejected candidate no longer
        aborts the run or leaves other chains' tasks PENDING. The accumulators are
        rebuilt per round so returned artifacts describe the terminal revision,
        while `analysis_history` keeps every round. Once the fixed budget is spent,
        each remaining candidate receives an explicit DEFERRED disposition.
        """
        from ..prototyping.ag_extractor import extract_ag_graphs
        from ..prototyping.ag_repair import attempt_dependency_closed_ag_repair

        severity = {"PASS": 0, "INCOMPLETE": 1, "FAIL": 2}
        maximum_repair_attempts = self.maximum_ag_repair_attempts
        repair_attempts = 0
        analysis_round = 0
        analysis_history: list[dict[str, Any]] = []
        chains: list[dict[str, Any]] = []
        pattern_cases: list[dict[str, Any]] = []
        pattern_per_chain: list[dict[str, Any]] = []
        all_failures: list[dict[str, Any]] = []
        aggregate_verdict = "PASS"
        pattern_verdict = "PASS"
        checker_version: Optional[str] = None

        while True:
            # per round: the artifacts must describe ONE revision, so anything
            # accumulated from a superseded revision is discarded
            chains = []
            pattern_cases = []
            pattern_per_chain = []
            all_failures = []
            aggregate_verdict = "PASS"
            pattern_verdict = "PASS"
            candidate_groups: list[list[tuple]] = []

            for graph in graphs:
                report, pattern, failures, _record, chain_candidate = (
                    self._run_ag_analysis_round(
                        graph,
                        analysis_round=analysis_round,
                        repair_attempts=repair_attempts,
                        maximum_repair_attempts=maximum_repair_attempts,
                        allow_repair=True,
                    )
                )
                checker_version = report.checker_version
                chains.append(report.to_dict())
                pattern_cases.extend(pattern.get("cases", ()))
                pattern_per_chain.append({
                    "source_requirement": report.source_requirement,
                    "verdict": pattern["verdict"],
                    "cases": list(pattern.get("cases", ())),
                })
                all_failures.extend(failures["failures"])
                analysis_history.append({
                    "analysis_round": analysis_round,
                    "source_requirement": report.source_requirement,
                    "source_model_revision": graph.revision,
                    "source_model_digest": graph.model_digest,
                    "verdict": report.verdict,
                    "pattern_verdict": pattern["verdict"],
                    "failure_ids": [
                        item["failure_id"] for item in failures["failures"]
                    ],
                })
                if severity[report.verdict] > severity[aggregate_verdict]:
                    aggregate_verdict = report.verdict
                if pattern["verdict"] != "PASS":
                    pattern_verdict = "FAIL"
                if chain_candidate:
                    candidate_groups.append(list(chain_candidate))

            if not candidate_groups:
                break
            # Fair deterministic ordering: attempt the first named obligation
            # from every failing chain before a second obligation from any one
            # chain. A candidate's one bounded feedback retry still stays adjacent
            # to that candidate, so the retry sees an unchanged base revision.
            candidates: list[tuple] = []
            for obligation_index in range(
                max(len(group) for group in candidate_groups)
            ):
                for group in candidate_groups:
                    if obligation_index < len(group):
                        candidates.append(group[obligation_index])
            accepted = False
            candidate_queue = [(candidate, 0) for candidate in candidates]
            candidate_index = 0
            while candidate_index < len(candidate_queue):
                candidate, retry_index = candidate_queue[candidate_index]
                if repair_attempts >= maximum_repair_attempts:
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="DEFERRED",
                            reason="automatic_repair_budget_exhausted",
                        )
                    break
                candidate_index += 1
                decision = attempt_dependency_closed_ag_repair(
                    llm=self.llm,
                    board=self.blackboard,
                    context_builder=self.context_builder,
                    sessions=self.task_sessions,
                    failure_record_id=candidate[0],
                    analysis_record_id=candidate[1],
                )
                repair_attempts += 1
                if decision.status == "ACCEPTED":
                    accepted = True
                    for pending, _retry in candidate_queue[candidate_index:]:
                        self._close_unattempted_ag_repair_candidate(
                            pending,
                            status="SUPERSEDED",
                            reason="superseded_by_committed_repair",
                        )
                    break
                if (
                    retry_index == 0
                    and repair_attempts < maximum_repair_attempts
                ):
                    retry = self._build_ag_repair_retry_candidate(
                        candidate, retry_index=1
                    )
                    candidate_queue.insert(candidate_index, (retry, 1))
            if not accepted:
                break
            # the commit superseded every chain's graph: re-extract, then re-check
            analysis_round += 1
            graphs = extract_ag_graphs(
                self.blackboard.current_model.model_text,
                revision=self.blackboard.current_revision,
                model_digest=self.blackboard.current_model.model_digest,
            )

        revision = self.blackboard.current_revision
        digest = self.blackboard.current_model.model_digest
        repair_decisions = [
            dict(item.payload)
            for item in self.blackboard.records(topic="repair.decision")
        ]
        return {
            "ag_contract_graph": {
                "artifact_role": "RUNTIME_A_G_PREDICTION",
                "producing_stage": "R2_COMPOSITIONAL_TRACE",
                "measurement_boundary": "INTERVENTION",
                "experiment_namespace": "BLACKBOARD_AG_V1",
                "configuration": "R2-BBAG",
                "checker_version": checker_version,
                "source_model_revision": revision,
                "source_model_digest": digest,
                "verdict": aggregate_verdict,
                "multi_chain": True,
                "chain_count": len(chains),
                "source_requirements": [
                    c.get("source_requirement") for c in chains
                ],
                "chains": chains,
            },
            "pattern_conformance_report": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_PATTERN_CONFORMANCE",
                "producing_stage": "R2_PATTERN_CONFORMANCE",
                "measurement_boundary": "INTERVENTION",
                "multi_chain": True,
                "verdict": pattern_verdict,
                "cases": pattern_cases,
                "per_chain": pattern_per_chain,
            },
            "failure_diagnostics": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_FAILURE_ROUTING",
                "producing_stage": "R2_FAILURE_ROUTING",
                "measurement_boundary": "INTERVENTION",
                "multi_chain": True,
                "failures": all_failures,
                "analysis_history": analysis_history,
                "source_model_revision": revision,
                "source_model_digest": digest,
            },
            "repair_decisions": {
                "schema_version": "1.0",
                "artifact_role": "INTERVENTION_REPAIR_DECISIONS",
                "producing_stage": "R2_DEPENDENCY_CLOSED_REPAIR",
                "measurement_boundary": "INTERVENTION",
                "decisions": repair_decisions,
            },
            "_terminal_model_sysml": self.blackboard.current_model.model_text,
        }






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
