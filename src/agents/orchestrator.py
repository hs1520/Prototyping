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

from .design_agent import DesignAgent
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



class Orchestrator:
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
        task_session_max_tokens: int = 150000,
        r2_generation_mode: Optional[str] = None,
        r2_authored_syntax_max_attempts: int = 3,
        maximum_ag_repair_attempts: int = 3,
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
        self.design_agent = DesignAgent(llm, rag_retriever)
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
        model_qualification = build_model_qualification(
            model_text=final_sysml,
            requirements=requirements,
            syntax_result=check_syntax(
                final_sysml,
                fail_closed=True,
                filter_stdlib_diagnostics=True,
            ),
            simulation_result=final_sim,
            terminal_consistency=terminal_consistency,
            structural_obligation_report=structural_obligation_report,
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
        print(f"  Simulation reachability:  {final_sim.reachability_score:.3f} "
              f"({len(final_sim.passed_scenarios())}/{len(final_sim.scenario_results)} scenarios)")
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
            "structural_obligation_report": structural_obligation_report,
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

    def _reset_exploration_state(
        self, system_name: str, requirements: List[str], model: SysMLModel
    ) -> None:
        """Reset run-scoped recommendation metadata and session state.

        Reusing an Orchestrator must never let a prior run's physical design
        leak into a new authority decision.
        """
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

        # Re-initialise state for this exploration session
        self.state = PrototypingState(
            system_name=system_name,
            system_description="",
        )
        self.state.requirements  = requirements
        self.state.current_model = model

    def _phase8_realization(
        self, final_model: SysMLModel, requirements: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Phase 8: realization meet-in-the-middle artifact.

        Builds the realization report for the recommended design, or an
        explicit NO_RECOMMENDABLE_DESIGN verdict when the recommendation gate
        rejected every Pareto member.  When ``realization_inject`` is on, the
        RealizationPackage is injected into the final model text (best-effort,
        never fatal).  Returns the realization dict or None when skipped.
        """
        realization = None
        if self.use_variation_dse and getattr(self, "last_recommended_design", None):
            realization = self._realization_artifact(
                self.last_recommended_design,
                getattr(self, "last_pareto_designs", []) or [],
                requirements,
            )
        elif getattr(self, "last_recommendation_status", None) == "NO_RECOMMENDABLE_DESIGN":
            from ..realization.matcher import mapping_policy
            realization = {
                "verdict": "NO_RECOMMENDABLE_DESIGN",
                "chosen": None,
                "per_requirement": [],
                "forward_flight_ok": None,
                "rank_preservation": {},
                "failed_checks": [{
                    "name": "recommendation_gate",
                    "passed": False,
                    "detail": (
                        "no estimator-feasible, mapping-compliant, Phase8-closable "
                        "Pareto member"
                    ),
                }],
                "resize_note": "",
                "realization_model_sysml": "",
                "mapping_policy": mapping_policy(),
                "summary": (
                    "NO RECOMMENDABLE DESIGN — exploratory Pareto retained, but no "
                    "candidate passed estimator feasibility + catalog mapping + Phase 8 closure"
                ),
            }
        if realization:
            print(f"  ✓ {realization['summary']}")
            if self.realization_inject and realization.get("realization_model_sysml"):
                try:
                    from ..realization.realization_emitter import inject_realization_analysis
                    base_text = get_sysml_text(final_model)
                    injected, ok = inject_realization_analysis(base_text, realization["_report"])
                    if ok:
                        self._sync_model_text(final_model, injected)
                        print("  [realization] injected RealizationPackage into final model")
                except Exception as e:
                    print(f"  ⚠ realization model injection skipped ({e})")
        else:
            print("  ⚠ no recommended design / realization skipped")
        return realization

    def _print_explore_summary(
        self, final_model: SysMLModel, final_score: float, final_sim, best_config
    ) -> None:
        """Always-visible end-of-exploration summary (score, sim, LLM usage)."""
        sim_warnings = (getattr(final_model, "metadata", None) or {}).get("sim_warnings", "")
        print(f"{'='*60}")
        print("Exploration Complete!")
        print(f"  Final score:              {final_score:.3f}")
        print(f"  Simulation reachability:  {final_sim.reachability_score:.3f} "
              f"({len(final_sim.passed_scenarios())}/{len(final_sim.scenario_results)} scenarios)")
        print(f"  Part definitions: {len(final_model.part_definitions)}")
        print(f"  Best config:      {best_config.parameters}")
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

    def explore(
        self,
        generate_result: Dict[str, Any],
        mcts_iterations: int = 50,
        mcts_seed: Optional[int] = None,
        mcts_patience: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """
        Run multi-objective Design Space Exploration on a validated model.

        Takes the output of generate() as input.  Explores either the
        LLM-declared variation space (``use_variation_dse``) or the catalog
        operator space via the bilevel MO-MCTS + inner BO, applies the winning
        configuration to the model, then runs a final refinement pass to
        implement those architectural decisions.

        Pipeline
        ────────
        Phase 3  DSE (variation or catalog bilevel) + operator application
        Phase 4-5  Iterative Refinement (with DSE constraints in prompt)
        Phase 6  Behavioral Reachability Simulation

        Parameters
        ----------
        generate_result   Dict returned by generate().
        mcts_iterations   Retained for API compatibility — the bilevel search
                          uses its own budget (run_bilevel_dse ``iterations``).
        mcts_seed         Random seed (None = 0 for the catalog path).
        mcts_patience     Retained for API compatibility (unused).

        Returns
        -------
        Full result dict — superset of generate_result — with updated
        model/score/sim fields plus DSE-specific fields:
        {design_space_summary, design_space_parameters,
         best_config, pareto_alternatives, …}
        """
        model        = generate_result["model"]
        requirements = generate_result["requirements"]
        system_name  = generate_result["system_name"]

        if (
            self.revised_experiment_arm is not None
            and self.revised_experiment_arm.uses_blackboard
            and self.blackboard is None
        ):
            raise ValueError(
                "R1-BBCTX explore() must continue on the Orchestrator that "
                "created the revisioned generate() workspace"
            )

        self._reset_exploration_state(system_name, requirements, model)

        print(f"\n{'='*60}")
        print(f"[explore]  {system_name}")
        print(f"{'='*60}\n")

        # ── Phase 3: DSE ──────────────────────────────────────────────────────
        print("Phase 3: Design Space Exploration")
        print("-" * 40)
        if self.use_variation_dse:
            # "Replace" mode: LLM declares variation points, DSE explores + resolves them.
            design_space, best_config, pareto_front = self._explore_variations(
                model, requirements, mcts_seed
            )
            self.state.design_space = design_space
        else:
            # Catalog path: bilevel MO-MCTS (outer architecture operators) +
            # inner BO (control frequency).  The legacy scalar MCTS is retired.
            design_space, best_config, pareto_front = self._explore_bilevel(
                model, requirements, random_seed=mcts_seed
            )
            self.state.design_space = design_space

            # Lightweight injections (freq attr, doc annotation).
            _apply_best_config_to_model(best_config, model)
            _apply_inject_attrs_to_sysml_text(model)

            # Redundancy + protocol structures via valid-by-construction operator
            # merges (syntax-gated; replaced all regex text injection).
            from ..dse.operator_applicator import apply_architecture
            applied = apply_architecture(model, best_config)
            if applied:
                print(f"  [bilevel-DSE] applied via operators: {applied}")
        self._print_exploration_summary(design_space, best_config, pareto_front)

        # ── Phase 4-5: Refinement with DSE constraints ────────────────────────
        print("Phase 4-5: Iterative Refinement (DSE-grounded)")
        print("-" * 40)
        final_model, final_score, final_sim = self._iterative_refinement(
            model, requirements, dse_best_config=best_config,
            connectivity_floor=self.use_variation_dse,
        )
        final_model, final_score, final_sim = self._functional_closure_pass(
            final_model,
            final_sim,
            final_score,
            requirements,
            dse_best_config=best_config,
            max_iters=2,
        )
        self.state.current_model = final_model
        print(f"  ✓ Final design score: {final_score:.3f}\n")

        # ── Phase 6: Simulation ───────────────────────────────────────────────
        # Simulation already ran in Phase 4-5 — reuse the result.
        print("Phase 6: Behavioral Reachability Simulation", flush=True)
        print("-" * 40)
        self._print_final_sim(final_sim)

        # ── Phase 7: DSE→SITL verification artifact (layer 0+1, deterministic) ──
        # Builds SysML v2 verification cases for the quantified requirements + maps
        # settable families to ArduPilot .parm with a static L1 range check. No SITL
        # launch; never mutates the final model — purely an added artifact.
        print("Phase 7: DSE→SITL Verification (cases + L1)", flush=True)
        print("-" * 40)
        verification_artifact = self._dse_verification_artifact(
            get_sysml_text(final_model), requirements
        )
        if verification_artifact:
            print(f"  ✓ {verification_artifact['summary']}")
        else:
            print("  ⚠ no quantified requirements — verification skipped")

        # ── Phase 8: Realization meet-in-the-middle (deterministic, non-mutating by default) ──
        print("Phase 8: Realization (meet-in-the-middle)", flush=True)
        print("-" * 40)
        realization = self._phase8_realization(final_model, requirements)

        # ── Phase 9: opt-in high-fidelity closure (native SITL / Gazebo) ──────────
        # OFF by default: real flight needs Docker/arducopter and minutes per run,
        # so it is not on the default explore() path. When enabled it auto-connects
        # the Phase 8 recommendation to the high-fidelity feasibility runners and
        # attaches a summary. Best-effort; never mutates the model; never upgrades
        # datasheet CLOSED (SITL/Gazebo verify feasibility/dynamics, not endurance).
        hifi = None
        if getattr(self, "phase9_hifi", None):
            print("Phase 9: High-fidelity closure (native SITL / Gazebo)", flush=True)
            print("-" * 40)
            if getattr(self, "last_recommended_design", None):
                hifi = self._phase9_hifi_artifact(
                    self.phase9_hifi,
                    self.last_recommended_design,
                    get_sysml_text(final_model),
                    requirements,
                )
            if hifi:
                print(f"  ✓ {hifi['summary']}")
            else:
                print("  ⚠ no recommended design / high-fidelity closure skipped")

        # Snapshot the ordinary generated model before terminal A/G binding.
        pre_terminal_score = final_score
        self._restore_generation_plan_metadata(final_model)
        pre_ag_sysml, generation_plan_conformance = (
            self._enforce_terminal_generation_plan(
                final_model, get_sysml_text(final_model)
            )
        )
        pre_ag_sim = self._run_simulation(
            pre_ag_sysml, self.state.system_name
        )
        final_sysml = self._reconcile_guided_ag_contract_layer(
            pre_ag_sysml, requirements
        )
        self._commit_terminal_model(final_sysml, producer="Orchestrator.explore")
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
                dse_best_config=best_config,
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
                self.state.system_name,
            )
        )
        model_qualification = build_model_qualification(
            model_text=final_sysml,
            requirements=requirements,
            syntax_result=check_syntax(
                final_sysml,
                fail_closed=True,
                filter_stdlib_diagnostics=True,
            ),
            simulation_result=final_sim,
            terminal_consistency=terminal_consistency,
            structural_obligation_report=structural_obligation_report,
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
        )
        self.state.current_model = final_model

        # ── Summary ───────────────────────────────────────────────────────────
        self._print_explore_summary(final_model, final_score, final_sim, best_config)
        print(
            f"  Model qualification:      "
            f"{model_qualification['status']}"
        )
        ledger = getattr(self.llm, "ledger", None)

        # Merge evaluation histories: generate phase first, then explore phase.
        # **generate_result would overwrite with generate-only history if we
        # relied on dict spreading alone, so we concatenate explicitly.
        combined_history = (
            generate_result.get("evaluation_history", [])
            + self.state.evaluation_history
        )
        return {
            # ── Fields inherited / updated from generate() ────────────────────
            **generate_result,
            **collaboration_artifacts,
            "model":              final_model,
            "model_sysml":        final_sysml,
            "model_summary":      final_model.get_summary(),
            "final_score":        final_score,
            "iterations":         self.state.iteration,
            "evaluation_history": combined_history,
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
            "structural_obligation_report": structural_obligation_report,
            "ag_binding_report": self.last_ag_binding_report,
            "ag_non_degradation": self.last_ag_non_degradation,
            "functional_closure": dict(self.last_functional_closure or {}),
            "verification_anchor_attempts": list(
                self.last_verification_anchor_attempts
            ),
            # ── DSE-specific fields ───────────────────────────────────────────
            "design_space_summary": design_space.get_summary(),
            "design_space_parameters": [
                {
                    "name":        p.name,
                    "type":        p.param_type.value,
                    "choices":     p.choices,
                    "description": p.description,
                }
                for p in design_space.parameters
            ],
            "best_config": best_config.parameters,
            "pareto_alternatives": [
                {
                    "name":          c.name,
                    "parameters":    dict(c.parameters),
                    "scores":        dict(c.scores),
                    "overall_score": round(c.overall_score, 4),
                }
                for c in pareto_front
            ],
            # Diagnostic estimator front, explicitly separate from the official
            # constrained Pareto above. It can contain designs that fail mapping or
            # Phase 8 and must never be presented as recommendations.
            "exploratory_pareto_alternatives": list(
                getattr(self, "last_exploratory_pareto_alternatives", []) or []
            ),
            "dse_constraint_counts": dict(
                getattr(self, "last_constraint_counts", {}) or {}
            ),
            "dse_search_coverage": dict(
                getattr(self, "last_search_coverage", {}) or {}
            ),
            "recommended_by": getattr(self, "last_recommended_by", None),
            "recommended_estimator_feasible": getattr(self, "last_recommended_estimator_feasible", None),
            "recommendation_status": getattr(self, "last_recommendation_status", None),
            "recommendable_front_count": getattr(self, "last_recommendable_front_count", None),
            "variation_proposal_source": getattr(self, "last_variation_proposal_source", None),
            "estimator_calibration": getattr(self, "last_estimator_calibration", None),
            "dse_verification": verification_artifact,
            "realization": _public_realization(realization),
            "phase9_hifi": hifi,
            "llm_usage": ledger.as_dict() if ledger is not None else None,
        }

    @staticmethod
    def _dse_verification_artifact(model_sysml: str, requirements: List[str]):
        """Layer 0+1 DSE→SITL artifact: SysML v2 verification cases + settable-family
        L1. Deterministic, no flight, non-mutating. Returns None when there is nothing
        to verify or on any failure (the artifact never breaks the pipeline)."""
        try:
            from ..sitl.dse_verification import build_dse_verification
            rep = build_dse_verification(model_sysml, requirements)
            if not rep.verification_cases and not rep.parm_lines:
                return None
            return {
                "verification_cases": rep.verification_cases,
                "parm_lines": rep.parm_lines,
                "l1_ok": rep.l1_ok,
                "l1_results": [vars(r) for r in rep.l1_results],
                "verification_model_sysml": rep.verification_model,
                "summary": rep.summary(),
            }
        except Exception as e:
            print(f"  ⚠ DSE verification skipped ({e})")
            return None

    @staticmethod
    def _realization_artifact(design, pareto_designs, requirements):
        """Phase 8 realization artifact. Best-effort; never breaks the pipeline."""
        try:
            from ..realization.closure import close_the_loop
            from ..realization.matcher import mapping_policy
            from ..realization.realization_emitter import emit_realization_package

            report = close_the_loop(design, pareto_designs, requirements)
            sysml, _ok = emit_realization_package(report)
            chosen = None
            if report.chosen is not None:
                c = report.chosen
                chosen = {
                    "combo": c.rd.combo.name,
                    "pack": c.rd.pack.name,
                    "frame": c.rd.frame.name,
                    "rotor_count": c.rd.rotor_count,
                    "rotor_radius_m": c.rd.combo.prop_diameter_in * 0.0254 / 2.0,
                    "battery_capacity_mah": c.rd.pack.capacity_mah,
                    "battery_cells": c.rd.pack.cells,
                    "pack_nominal_voltage_v": c.metrics.pack_voltage_v,
                    "motor_curve_voltage_v": c.rd.combo.voltage_v,
                    "voltage_ratio": c.metrics.voltage_ratio,
                    "derated_max_thrust_per_motor_g": (
                        c.metrics.derated_max_thrust_per_motor_g
                    ),
                    "integration_bundle": c.rd.integration_bundle.name,
                    "integration_mass_g": c.rd.integration_bundle.mass_g,
                    "integration_components": list(c.rd.integration_bundle.components),
                    "total_mass_kg": c.metrics.total_mass_kg,
                    "endurance_min": c.metrics.endurance_min,
                    "total_hover_current_a": c.metrics.total_hover_current_a,
                    "hover_throttle": c.metrics.hover_throttle,
                    "twr": c.metrics.twr_max,
                    "cost": c.metrics.cost,
                    "cost_axis": "mass",
                    "distance": c.distance,
                    "design_drift": [vars(d) for d in getattr(c, "design_drift", ())],
                }
            closure_families = sorted({v.family for v in report.per_requirement
                                       if getattr(v, "scope", "closure") == "closure"})
            forward_families = sorted({v.family for v in report.per_requirement
                                       if getattr(v, "scope", "closure") == "forward_flight"})
            deferred_families = sorted({v.family for v in report.per_requirement
                                        if getattr(v, "scope", "closure") == "deferred"})
            closure_text = ", ".join(closure_families) if closure_families else "none"
            forward_text = ", ".join(forward_families) if forward_families else "none"
            deferred_text = ", ".join(deferred_families) if deferred_families else "none"
            if report.verdict in ("CLOSED", "CLOSED_AFTER_RESIZE"):
                summary = (
                    f"MEET-IN-THE-MIDDLE CLOSED — realizable + {closure_text} closed; "
                    "speed/range evaluated separately by lumped forward-flight fidelity "
                    f"(datasheet closure families: {closure_text}; "
                    f"forward_flight families: {forward_text}; "
                    f"deferred families: {deferred_text})"
                )
            else:
                summary = (
                    "REALIZATION GAP — top-down and bottom-up have not met for closure "
                    f"verdict families: {closure_text}; forward_flight families: {forward_text}; "
                    f"deferred families: {deferred_text}"
                )
            return {
                "verdict": report.verdict,
                "chosen": chosen,
                "per_requirement": [vars(v) for v in report.per_requirement],
                "forward_flight_ok": report.forward_flight_ok,
                "rank_preservation": dict(report.rank_preservation),
                "failed_checks": [vars(c) for c in report.failed_checks],
                "resize_note": report.resize_note,
                "realization_model_sysml": sysml,
                "summary": summary,
                "mapping_policy": mapping_policy(),
                "_report": report,
            }
        except Exception as e:
            print(f"  ⚠ realization skipped ({e})")
            return None

    @staticmethod
    def _phase9_hifi_artifact(mode, design, model_text, requirements):
        """Phase 9 opt-in high-fidelity closure. Best-effort; never breaks the
        pipeline; never upgrades datasheet CLOSED (SITL/Gazebo verify feasibility/
        dynamics, not endurance). Environment absence is reported, not faked."""
        try:
            from .phase9_hifi import VALID_MODES, run_hifi_closure

            if str(mode).strip().lower() not in VALID_MODES:
                print(f"  ⚠ phase9_hifi mode {mode!r} not in {VALID_MODES}; skipped")
                return None
            result = run_hifi_closure(mode, design, model_text, requirements)
            parts = []
            for layer in result.get("layers", []):
                name = layer.get("layer")
                if layer.get("status") == "skipped":
                    parts.append(f"{name}=skipped({layer.get('reason')})")
                elif name == "sitl":
                    parts.append(f"sitl(flight={layer.get('flight_passed')}, "
                                 f"safety={layer.get('safety_status')})")
                elif name == "gazebo":
                    parts.append(f"gazebo({layer.get('gazebo_status')})")
                else:
                    parts.append(f"{name}={layer.get('status')}")
            result["summary"] = (
                "PHASE 9 HIGH-FIDELITY — " + "; ".join(parts) + " "
                "[verifies feasibility/dynamics only; datasheet CLOSED unchanged; "
                "endurance never validated by SITL/Gazebo]"
            )
            return result
        except Exception as e:
            print(f"  ⚠ phase 9 high-fidelity closure skipped ({e})")
            return None

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
                for message in event.get("messages", ()):
                    self._append_session_message(
                        session,
                        str(message.get("role", "user")),
                        str(message.get("content", "")),
                    )
                response = event.get("response", {})
                prompt_tokens = int(response.get("prompt_tokens", 0) or 0)
                completion_tokens = int(
                    response.get("completion_tokens", 0) or 0
                )
                if prompt_tokens + completion_tokens == 0:
                    prompt_chars = sum(
                        len(str(item.get("content", "")))
                        for item in event.get("messages", ())
                    )
                    prompt_tokens = max(1, prompt_chars // 4)
                    completion_tokens = max(
                        1, len(str(response.get("content", ""))) // 4
                    )
                self._append_session_message(
                    session,
                    "assistant",
                    str(response.get("content", "")),
                    token_count=prompt_tokens + completion_tokens,
                )

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
        from ..prototyping.experiment_arms import (
            R2_DETERMINISTIC_GENERATION_MODE,
            R2_LLM_AUTHORED_GENERATION_MODE,
            R2_LLM_DECIDED_GENERATION_MODE,
            RevisedExperimentArm,
        )

        if self.revised_experiment_arm is not RevisedExperimentArm.SEMANTIC_ASSURANCE:
            return None

        from ..prototyping.ag_chains import select_ag_chains
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

        selected = list(select_ag_chains(requirements))
        if not selected:
            raise RuntimeError(
                "R2-BBAG failed closed: no bounded A/G chain was selected"
            )
        self.last_ag_authoring_attempts = []
        self.ag_input_dispositions = {}
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

    def _merge_planned_ag_packages(self, model_text: str) -> str:
        plan = self._active_ag_generation_plan
        if plan is None:
            return model_text
        merged = str(model_text)
        for spec, package_text in zip(plan["specs"], plan["packages"]):
            merged = self._without_named_package(merged, spec.package).rstrip()
            merged += "\n\n" + package_text.strip() + "\n"
        return merged

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

    def _prepare_design_handoff(
        self, system_name: str, requirements: List[str]
    ) -> None:
        """Create the first typed RequirementsAgent -> DesignAgent handoff.

        This is the R1-BBCTX intervention.  The envelope is derived only from
        run-time source/model records; evaluator gold has no API into it.
        """
        arm = self.revised_experiment_arm
        if arm is None or not arm.uses_blackboard:
            self.blackboard = None
            self.context_builder = None
            self.task_sessions = None
            self._active_design_handoff = None
            return

        from ..prototyping.blackboard import Blackboard, RecordType, TaskStatus
        from ..prototyping.context_builder import ContextBuilder
        from ..prototyping.task_session import TaskSessionRegistry

        self.blackboard = Blackboard(system_name)
        self.context_builder = ContextBuilder(self.blackboard)
        self.task_sessions = TaskSessionRegistry()
        source_record = self.blackboard.publish(
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
        from ..prototyping.generation_plan import (
            ModelGenerationPlan,
            apply_generation_plan,
        )

        plan = ModelGenerationPlan.from_dict(raw_plan)
        planned_text, conformance = apply_generation_plan(model_text, plan)
        if plan.behavior_obligations:
            from ..prototyping.ag_behavior_plan import (
                BehaviorObligationPlan,
                check_owned_behavior_obligation_conformance,
                materialize_behavior_obligations,
            )

            behavior_plan = BehaviorObligationPlan(
                plan.behavior_obligations
            )
            canonical_fragment, _step4_gate = (
                materialize_behavior_obligations("", behavior_plan)
            )
            planned_text, restored = (
                self.design_agent._inject_missing_ag_obligation_defs(
                    planned_text,
                    canonical_fragment,
                    behavior_plan,
                )
            )
            behavior_gate = (
                check_owned_behavior_obligation_conformance(
                    planned_text, behavior_plan
                )
            )
            conformance["behavior_obligation_conformance"] = behavior_gate
            conformance["restored_behavior_elements"] = restored
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

            # The generated base model has already passed the legacy syntax gate.
            # Re-check it here so a later terminal mutation cannot smuggle a new
            # parser/reference error into R2.  This gate uses the same narrowly
            # scoped standard-library diagnostic allowlist as R0/R1: Syside's
            # single-source loader reports official implicit/standard constructs
            # such as ``transition initial`` and SI unit symbols as unresolved.
            # Those diagnostics are not caused by the A/G intervention.
            base_gate = check_syntax(
                model_text,
                fail_closed=True,
                filter_stdlib_diagnostics=True,
            )
            # Errors only. A warning is not a syntax failure, and gating on the
            # score made one warning enough to fail an entire arm closed on a model
            # with zero parser and zero sema errors — a measured seed lost its R2
            # evidence to exactly that, its diagnostic reading "failed the shared
            # syntax gate: ✓ no syntax errors (score=0.950)". The same defect was
            # already removed from the A/G authoring loop (b9b5cdc); this copy of
            # the gate survived it.
            if base_gate.has_errors:
                raise RuntimeError(
                    "committed base model failed the shared syntax gate: "
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
                filter_stdlib_diagnostics=True,
            )
            if merged_gate.has_errors or (
                not llm_authored and merged_gate.score != 1.0
            ):
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
                filter_stdlib_diagnostics=True,
            )
            last_gate = gate
            syntax_ok = not gate.has_errors
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
                    filter_stdlib_diagnostics=True,
                )
                last_gate = gate
                attempt_record.update({
                    "syntax_ok": not gate.has_errors,
                    "parser_error_count": len(gate.parser_errors),
                    "semantic_reference_error_count": len(gate.sema_errors),
                    "syntax_summary": gate.short_summary(),
                    "diagnostics": self._syntax_gate_diagnostic_lines(gate),
                    "package_text": package_text,
                })
                if gate.has_errors:
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
        attempt_prompt = prompt
        last: Optional[DecisionError] = None
        for attempt in range(max_decision_attempts):
            raw = str(self.llm.chat(attempt_prompt, system_prompt=system_prompt))
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
                attempt_prompt = (
                    f"{prompt}\n\nYour previous decisions were rejected: {exc}\n"
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
                authored, fail_closed=True, filter_stdlib_diagnostics=True
            )
            # errors only — a warning-depressed score is not a syntax failure, and
            # rejecting on it discarded valid packages and fed back a defect list
            # for a model that had none, destabilising the next round
            if gate.has_errors:
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

    def _explore_bilevel(
        self,
        model: SysMLModel,
        requirements: List[str],
        random_seed: Optional[int] = None,
    ) -> Tuple[DesignSpace, DesignConfiguration, List[DesignConfiguration]]:
        """Phase 3 (catalog path): bilevel MO-MCTS + inner BO over the operator space.

        Outer MO-MCTS explores the catalog architecture operators (redundancy /
        topology / sensing / protocol); the inner BO tunes control_frequency_hz per
        architecture.  Returns (DesignSpace, best_config, pareto_front) with the
        same shapes the variation path produces.  On failure returns an empty
        design space + empty config so downstream refinement still runs.
        """
        from ..dse.pipeline_adapter import run_bilevel_dse, _to_design_configuration

        try:
            result = run_bilevel_dse(
                model, requirements,
                random_seed=random_seed if random_seed is not None else 0,
                score_quality=True,
            )
        except Exception as e:  # DSE failure must not break the pipeline
            print(f"  [bilevel-DSE] failed ({e}); continuing without DSE decisions")
            return (
                DesignSpace(name=f"{model.name}_CatalogSpace"),
                DesignConfiguration(name="dse_failed", parameters={}),
                [],
            )

        best_config = result.best_config

        # Report-facing design space: the catalog operator space the outer MCTS explored.
        ds = DesignSpace(name=f"{model.name}_CatalogSpace")
        for pname, choices in (
            ("redundancy_level", ["none", "dual", "triple"]),
            ("num_sensors", [1, 2, 3]),
            ("distributed_control", [False, True]),
            ("communication_protocol", ["MAVLink", "CAN", "Ethernet"]),
        ):
            ds.add_parameter(DesignParameter(
                name=pname,
                param_type=ParameterType.CATEGORICAL,
                default_value=choices[0],
                choices=list(choices),
                description="catalog operator variation point",
            ))
        pareto_front = [
            DesignConfiguration(
                name=f"alt{i}",
                parameters=dict(_to_design_configuration(state).parameters),
                scores=dict(objectives),
            )
            for i, (state, objectives) in enumerate(result.pareto_front)
        ]
        for cfg in pareto_front:
            ds.add_configuration(cfg)

        print(f"  [bilevel-DSE] recommended: {best_config.parameters}")
        print(f"     mandated_redundancy={result.mandated_redundancy}  "
              f"robustness={result.recommendation_robustness:.0%}  "
              f"front={len(result.pareto_front)}")
        if result.real_quality:
            dims = "  ".join(f"{k}={v:.2f}" for k, v in result.real_quality.items())
            print(f"     real design-quality (DesignEvaluator): {dims}")
        for note in result.notes:
            print(f"     note: {note}")
        return ds, best_config, pareto_front

    def _introduce_variations(self, model: SysMLModel, requirements: List[str]) -> SysMLModel:
        """Surgically convert connected components into variation points.

        CODE does the structural surgery (preserving the host's connects, variants
        specialise the host type so ports/connects stay valid); the LLM is asked
        ONLY for variant CONTENT per component (distinguishing attrs + rationale +
        linked requirement). This avoids the lossy whole-model rewrite that dropped
        connects and left variation points isolated.
        """
        from ..dse.variation_introducer import connected_components, introduce_variation
        from ..dse.variation_parser import admitted, parse_variation_points

        # Reset per-run provenance before every return path.  Reusing an
        # Orchestrator must never leak the previous run's proposal source.
        self.last_variation_proposal_source = None
        text = get_sysml_text(model)
        existing_points = admitted(parse_variation_points(text))[0]
        if any("catalog architecture seed" in p.rationale.lower()
               for p in existing_points):
            self.last_variation_proposal_source = "model-existing"
            return model  # already carries the mandatory catalog seed

        # A catalog architecture is a coupled tuple (rotor count, rotor radius,
        # battery cells), not three independently interchangeable values.  Seed that
        # complete, evidence-backed tuple space BEFORE asking the LLM for additional
        # variation points.  Apart from guaranteeing that catalog-realizable outer
        # designs are searched, making this point the first/canonical declarer keeps
        # normalize_variation_ownership() from stripping its architecture fields in
        # favour of a partial LLM proposal.
        max_points = 6
        introduced: List[str] = []
        catalog_seeded = False
        seed = self._catalog_seed_variation(text, requirements)
        if seed is not None:
            usage, type_name, rationale, reqs, variants = seed
            new_text, ok = introduce_variation(
                text, usage, type_name, variants, rationale, reqs
            )
            if ok:
                text = new_text
                introduced.append(usage)
                catalog_seeded = True
                print("  [variation-DSE] injected mandatory evidence-backed catalog "
                      f"architecture seed on '{usage}' ({len(variants)} architectures)")

        # A model may already contain objective variation points supplied upstream.
        # Do not generate additional LLM points in that case, but do still add the
        # mandatory catalog architecture seed above.  This closes the old early-
        # return hole without mutating the intent of the pre-existing space.
        if existing_points:
            if catalog_seeded:
                self.last_variation_proposal_source = "model-existing+catalog-seed"
            else:
                from ..dse.domain_objective import objective_families
                self.last_variation_proposal_source = (
                    "model-existing+catalog-seed-unavailable"
                    if objective_families(requirements) else "model-existing"
                )
            if introduced:
                self._sync_model_text(model, text)
                print("  [variation-DSE] added mandatory catalog architecture seed "
                      "beside pre-existing variation points")
            return model

        # Scan ALL remaining connected components: requirement-driven proposal
        # skips components that don't drive a quantified requirement, so we keep
        # looking to find the ones that DO.  The seeded host is no longer returned
        # by connected_components(), which also gives the coupled architecture tuple
        # one unambiguous outer-loop owner.  Cap kept modest — each extra point
        # multiplies the combinatorial space (and costs one LLM call), and the search
        # budget scales with the point count downstream.
        llm_introduced: List[str] = []
        for usage, type_name in connected_components(text):
            if len(introduced) >= max_points:
                break
            spec = self._propose_variants(usage, type_name, requirements)
            if spec is None:
                continue
            rationale, reqs, variants = spec
            new_text, ok = introduce_variation(
                text, usage, type_name, variants, rationale, reqs
            )
            if ok:
                text = new_text
                introduced.append(usage)
                llm_introduced.append(usage)

        if catalog_seeded and llm_introduced:
            self.last_variation_proposal_source = "catalog-seed+llm"
        elif catalog_seeded:
            self.last_variation_proposal_source = "catalog-seed"
        elif llm_introduced:
            # This is permitted only when no catalog seed is applicable (for example,
            # requirements without a quantified objective family).  A quantified
            # catalog-controlled run reports the unavailable state below instead of
            # silently representing an LLM-only space as catalog-complete.
            self.last_variation_proposal_source = "llm"
        else:
            from ..dse.domain_objective import objective_families
            if objective_families(requirements):
                self.last_variation_proposal_source = "catalog-seed-unavailable"
                print("  [variation-DSE] mandatory catalog architecture seed could not "
                      "form at least two legal, non-degenerate variants")
            else:
                self.last_variation_proposal_source = "catalog-seed-not-required"

        if introduced:
            self._sync_model_text(model, text)
            print(f"  [variation-DSE] surgically introduced {len(introduced)} variation "
                  f"point(s) (connects preserved): {introduced}")
        return model

    def _propose_variants(self, usage: str, type_name: str, requirements: List[str]):
        """Propose variant content for one component as SITL-SETTABLE DESIGN INPUTS.

        When the requirements carry quantified targets this is REQUIREMENT-DRIVEN: the
        LLM judges whether the component materially drives a quantified requirement
        (returns None if not → skip non-discriminating components) and emits, for the
        DESIGN INPUTS the component controls (battery / mass / rotor / cruise speed),
        4-6 variants spanning a real trade-off. The DSE scores these by feeding them
        through the physics estimator; SITL reproduces them by simulation (calibration).
        Emergent results (endurance/range) are NOT declared — they're derived.
        Falls back to the generic free-form prompt when no quantified targets exist.
        Returns (rationale, [req_ids], [VariantSpec]) or None.
        """
        from ..dse.domain_objective import (
            DESIGN_FIELD_ATTR,
            objective_families,
            requirement_targets,
            within_requirement_bounds,
        )
        from ..dse.variation_introducer import VariantSpec
        from ..realization.matcher import (
            catalog_design_domain, variant_design_is_catalog_admissible,
        )

        targets = requirement_targets(requirements)
        perf_fams = objective_families(requirements)
        if not targets or not perf_fams:
            return self._propose_variants_generic(usage, type_name, requirements)

        quant = "\n".join(
            f"{rid}: " + ", ".join(f"{f}>={t}" for f, t in fts)
            for rid, fts in targets.items()
        )
        design_keys = ", ".join(k for k in DESIGN_FIELD_ATTR if k != "battery_capacity_mah")
        catalog_domain = catalog_design_domain()
        prompt = (
            f"Component '{usage}' (type {type_name}). Quantified requirements:\n"
            f"{quant}\n\n"
            "Design inputs (SITL-settable) that determine performance:\n"
            f"{design_keys}\n\n"
            f"Does '{usage}' MATERIALLY drive any of these quantified requirements? "
            "Reply JSON ONLY.\n"
            'If NOT: {"relevant": false}\n'
            'If YES: {"relevant": true, "rationale": "<one line>", '
            '"satisfies": ["REQ-..."], "variants": [{"name": "<id>", '
            '"design": {"<design_input>": <number>}}]}\n'
            "design keys MUST be from the list above, and ONLY the inputs THIS component "
            "controls (e.g. a propulsion unit sets rotor_count/rotor_radius_m, a battery "
            "sets battery_cells, an airframe sets mass_kg). Battery CAPACITY is optimized "
            "internally by the inner layer — do NOT declare battery_capacity_mah. Give 4-6 "
            "variants spanning a real trade-off (more rotors / bigger rotor radius → more "
            "lift but heavier; more battery_cells → more power but heavier). satisfies must "
            "be a subset of the ids above. Catalog-controlled values MUST lie in this "
            f"evidence-backed domain (never invent another voltage family): {catalog_domain}"
        )
        try:
            data = _chat_json(self.llm, prompt)
            if not data.get("relevant", False):
                return None
            rationale = str(data.get("rationale", "quantified design trade-off"))
            valid = set(targets)
            reqs = [r for r in (str(x).replace("_", "-") for x in data.get("satisfies", []))
                    if r in valid]
            if not reqs:
                return None
            variants = []
            for v in data.get("variants", []):
                name = re.sub(r"\W", "", str(v.get("name", "")))
                if not name:
                    continue
                design = v.get("design", {}) or {}
                # objective bound filter: a variant must respect the quantified bounds of
                # the requirements it satisfies (e.g. payload ≤ the payload-mass limit), so
                # out-of-spec implementations never enter the variant library.
                if not within_requirement_bounds(design, reqs, requirements):
                    continue
                if not variant_design_is_catalog_admissible(design):
                    continue
                attr_lines = []
                for field, val in design.items():
                    field = str(field).strip().lower()
                    if field == "battery_capacity_mah":
                        continue  # inner-BO optimized, never variant-declared
                    if field in DESIGN_FIELD_ATTR and isinstance(val, (int, float)):
                        attr_lines.append(
                            f"attribute {DESIGN_FIELD_ATTR[field]} : Real = {float(val)};"
                        )
                if not attr_lines:
                    continue
                vtype = f"{name.capitalize()}{usage.capitalize()}Impl"
                variants.append(VariantSpec(name=name, type_name=vtype, attrs=" ".join(attr_lines)))
            return (rationale, reqs, variants) if len(variants) >= 2 else None
        except Exception as e:
            print(f"  [variation-DSE] variant proposal for '{usage}' skipped ({e})")
            return None

    def _propose_variants_generic(self, usage: str, type_name: str, requirements: List[str]):
        """Free-form variant proposal (used when requirements carry no quantified
        targets — the domain objective then can't discriminate anyway, so the DSE
        scores via the generic design-quality dims)."""
        from ..dse.variation_introducer import VariantSpec

        prompt = (
            f"Component '{usage}' (type {type_name}) has a genuine implementation "
            "trade-off. Propose 4-6 alternative implementations as JSON ONLY:\n"
            '{"rationale": "<one line>", "satisfies": ["REQ-..."], '
            '"variants": [{"name": "<id>", "attributes": {"<attrNameWithUnit>": <number>}}]}\n'
            "Attribute names must carry the unit (cruiseSpeedMps, maxHoverTimeMinutes, "
            "massKg, rangeM, ...). Pick satisfies from the requirements. JSON only.\n\n"
            "Requirements:\n" + "\n".join(requirements)
        )
        try:
            data = _chat_json(self.llm, prompt)
            rationale = str(data.get("rationale", "design trade-off"))
            reqs = [str(r) for r in data.get("satisfies", []) if r]
            variants = []
            for v in data.get("variants", []):
                name = re.sub(r"\W", "", str(v.get("name", "")))
                if not name:
                    continue
                attrs = " ".join(
                    f"attribute {re.sub(r'[^A-Za-z0-9_]', '', str(k))} : Real = {float(val)};"
                    for k, val in (v.get("attributes", {}) or {}).items()
                    if isinstance(val, (int, float))
                )
                vtype = f"{name.capitalize()}{usage.capitalize()}Impl"
                variants.append(VariantSpec(name=name, type_name=vtype, attrs=attrs))
            return (rationale, reqs, variants) if len(variants) >= 2 else None
        except Exception as e:
            print(f"  [variation-DSE] variant proposal for '{usage}' skipped ({e})")
            return None

    @staticmethod
    def _catalog_seed_variation(model_text: str, requirements: List[str]):
        """Mandatory deterministic catalog architecture seed for the outer loop.

        Applies when the requirements carry quantified objective families — i.e. the
        objective DSE has something real to optimize — and the
        model has a connected component whose name matches a DESIGN_ONTOLOGY concern
        (propulsion/airframe/power). The variant set is a standard engineering
        rotor-count/radius/cells trade (more disk area → endurance ↑ but mass ↑),
        restricted to the predeclared evidence-backed catalog domain. This constrains
        implementability and search coverage, not the objective score or the winner.
        Returns (usage, type_name, rationale, req_ids, variants) or None.
        """
        from ..dse.domain_objective import (
            DESIGN_FIELD_ATTR,
            objective_families,
            requirement_targets,
            within_requirement_bounds,
        )
        from ..dse.variation_introducer import VariantSpec, connected_components

        fams = set(objective_families(requirements))
        if not fams:
            return None
        req_ids = sorted(
            rid for rid, targets in requirement_targets(requirements).items()
            if any(f in fams for f, _ in targets)
        )
        if not req_ids:
            return None

        # Preference-ordered concern keywords: the propulsion-ish component is the
        # natural owner of rotor variants; airframe and power are acceptable hosts.
        concern_order = ("propuls", "rotor", "motor", "prop", "lift",
                         "airframe", "frame", "power", "batter", "energy")
        best = None
        for usage, type_name in connected_components(model_text):
            blob = f"{usage} {type_name}".lower()
            rank = next((i for i, k in enumerate(concern_order) if k in blob), None)
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, usage, type_name)
        if best is None:
            return None
        _, usage, type_name = best

        # Generate the seed from compatible frame + voltage-specific
        # motor/prop-curve + pack evidence, rather than generic architectures.
        from ..realization.matcher import catalog_design_domain
        designs = []
        for arch in catalog_design_domain()["architectures"]:
            diameter_in = arch["rotor_radius_m"] * 2.0 / 0.0254
            name = (
                f"catalog_r{arch['rotor_count']}_p{diameter_in:g}_"
                f"c{arch['battery_cells']}"
            ).replace(".", "p")
            designs.append((name, arch))
        variants = []
        for name, design in designs:
            if not within_requirement_bounds(design, req_ids, requirements):
                continue
            attrs = " ".join(
                f"attribute {DESIGN_FIELD_ATTR[field]} : Real = {float(val)};"
                for field, val in design.items()
            )
            vtype = f"{name.capitalize()}{usage.capitalize()}Impl"
            variants.append(VariantSpec(name=name, type_name=vtype, attrs=attrs))
        if len(variants) < 2:
            return None
        rationale = ("mandatory deterministic catalog architecture seed: evidence-backed "
                     "coupled rotor-count/radius/cell architectures")
        return usage, type_name, rationale, req_ids, variants

    # Compatibility for callers outside the orchestrator that used the old private
    # helper name.  Its semantics are now a mandatory architecture seed, not a
    # last-resort fallback.
    _fallback_variation = _catalog_seed_variation

    def _explore_variations(
        self, model: SysMLModel, requirements: List[str], seed: Optional[int],
    ) -> Tuple[DesignSpace, DesignConfiguration, List[DesignConfiguration]]:
        """Variation-DSE path: introduce variation points, explore them, resolve the
        recommendation into the model. Falls back to catalog bilevel if none admitted."""
        from ..dse.variation_dse import run_variation_dse
        from ..dse.variation_parser import admitted, parse_variation_points

        pre_variation_text = get_sysml_text(model)
        model = self._introduce_variations(model, requirements)
        introduced_text = get_sysml_text(model)
        # Prime the structured requirement extraction with the LLM (robust to phrasing;
        # output validated against the controlled vocabulary). Memoised, so the deterministic
        # DSE/analysis callers downstream reuse it. Best-effort — falls back to rules.
        try:
            from ..dse.requirement_spec import extract_requirements
            extract_requirements(requirements, llm=self.llm)
        except Exception as exc:
            from ..utils.suppressed import record_suppressed
            record_suppressed("orchestrator.requirement_spec_prime", exc)
            pass
        realizability = None
        realization_rank = None
        recommendability = None
        capacity_options = None
        try:
            from ..realization.matcher import catalog_capacity_options, match
            realizability = lambda di: bool(match(di, requirements))
            from ..realization.closure import close_the_loop

            recommendability = lambda di: close_the_loop(
                di, [], requirements
            ).verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
            capacity_options = lambda di: catalog_capacity_options(di, requirements)

            def _realization_rank(di):
                rep = close_the_loop(di, [], requirements)
                closes = 1.0 if rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"} else 0.0
                endurance = rep.chosen.metrics.endurance_min if rep.chosen is not None else 0.0
                return closes * 1_000_000.0 + endurance

            realization_rank = _realization_rank
        except Exception as e:
            print(f"  [variation-DSE] realizability-aware recommendation disabled ({e})")
        calibration = None
        self.last_estimator_calibration = None
        if getattr(self, "use_estimator_calibration", True):
            try:
                from ..realization.estimator_calibration import (
                    catalog_rank_check, fit_from_catalog,
                )
                calibration = fit_from_catalog()
                self.last_estimator_calibration = {
                    **calibration.as_dict(),
                    "rank_check": catalog_rank_check(fit=calibration),
                    "scope": ("search-only: SysML calc defs and Phase 8 "
                              "estimator_value keep the textbook constants"),
                }
                print("  [variation-DSE] estimator calibrated on catalog grid: "
                      f"fom_eff={calibration.fom_eff:.3f}, "
                      f"ED={calibration.energy_density_wh_kg:.0f} Wh/kg, "
                      f"endurance MAPE {calibration.endurance_mape_before:.0%}→"
                      f"{calibration.endurance_mape_after:.0%} "
                      f"(n={calibration.n_points})")
            except Exception as e:
                print(f"  [variation-DSE] estimator calibration skipped ({e})")
                calibration = None
        from contextlib import nullcontext

        from ..dse.physics_estimator import calibrated
        cal_ctx = calibrated(**calibration.overrides()) if calibration else nullcontext()
        with cal_ctx:
            res = run_variation_dse(
                model, requirements=requirements, random_seed=seed or 0,
                realizability=realizability,
                realization_rank=realization_rank,
                recommendability=recommendability,
                capacity_options=capacity_options,
            )
        if res is None:
            print("  [variation-DSE] no admissible variation space; "
                  "falling back to catalog bilevel DSE")
            return self._explore_bilevel(model, requirements, random_seed=seed)

        # build a design space from the admitted points (for the result/report)
        pts, _ = admitted(parse_variation_points(introduced_text))
        ds = DesignSpace(name=f"{model.name}_VariationSpace")
        for p in pts:
            ds.add_parameter(DesignParameter(
                name=p.point_id,
                param_type=ParameterType.CATEGORICAL,
                default_value=p.variant_names[0],
                choices=p.variant_names,
                description=(p.rationale or "")[:120],
            ))
        best_config = DesignConfiguration(
            name=("variation_recommended" if res.recommendation_status == "RECOMMENDED"
                  else "no_recommendable_design"),
            parameters=dict(res.recommended_choices),
        )
        pareto_front = [
            DesignConfiguration(name=f"alt{i}", parameters=dict(s), scores=dict(o))
            for i, (s, o) in enumerate(res.pareto_front)
        ]
        exploratory_front = [
            {
                "name": f"exploratory_alt{i}",
                "parameters": dict(state),
                "scores": dict(objectives),
            }
            for i, (state, objectives) in enumerate(res.exploratory_pareto_front)
        ]
        # Record the exploration into the design space so the summary reports the
        # real Pareto-front size (not 0): the front members carry multi-objective
        # scores, so get_pareto_front re-derives the same non-dominated set.
        for cfg in pareto_front:
            ds.add_configuration(cfg)
        ds.objective_weights = {
            "iterations_run": float(res.evaluated),
            "configurations_evaluated": float(res.evaluated),
            "early_stopped": 0.0,
        }
        # resolve the recommendation into the model (concrete, variations bound)
        if not hasattr(model, "metadata") or model.metadata is None:
            object.__setattr__(model, "metadata", {})
        # Wire an Automator-evaluable analysis closure into the recommended model: the
        # endurance constraint references the CHOSEN variants' design attributes
        # (closes the bare-attribute gap). Best-effort — never break the pipeline.
        # A variation declaration is an exploration model, not an implemented system.
        # If no official recommendation exists, restore the pre-DSE system for downstream
        # refinement/simulation; otherwise variant declarations look like live, dangling
        # part usages and create false connectivity failures. Pareto data remains in the
        # structured result and is deliberately not injected into the authority model.
        concrete = (
            res.concrete_model
            if res.recommendation_status == "RECOMMENDED"
            else pre_variation_text
        )
        if res.recommendation_status == "RECOMMENDED":
            try:
                from ..dse.analysis_emitter import inject_endurance_analysis, inject_trade_study
                injected, ok = inject_endurance_analysis(
                    concrete, requirements, capacity_mah=res.recommended_capacity_mah,
                    design=res.recommended_design)
                if ok:
                    concrete = injected
                    print("  [variation-DSE] injected Automator-evaluable endurance analysis closure")
                # also present the Pareto front as a SysML trade study over real alternatives
                ts, ok_ts = inject_trade_study(
                    concrete, [d for d, _ in res.pareto_designs], requirements,
                    recommended=res.recommended_design, bindings=res.pareto_bindings)
                if ok_ts:
                    concrete = ts
                    print(f"  [variation-DSE] injected DesignTradeStudy ({len(res.pareto_designs)} alternatives)")
            except Exception as exc:
                from ..utils.suppressed import record_suppressed
                record_suppressed("orchestrator.variation_analysis_injection", exc)
        model.metadata["last_sysml_text"] = concrete
        # expose the recommended design for opt-in high-fidelity (Gazebo) verification downstream
        self.last_recommended_design = res.recommended_design
        self.last_pareto_designs = res.pareto_designs
        self.last_recommended_bindings = res.recommended_bindings
        self.last_recommended_realizable = res.recommended_realizable
        self.last_realizable_front_count = res.realizable_front_count
        self.last_recommended_by = res.recommended_by
        self.last_recommended_estimator_feasible = res.recommended_estimator_feasible
        self.last_recommendation_status = res.recommendation_status
        self.last_recommendable_front_count = res.recommendable_front_count
        self.last_exploratory_design = res.exploratory_design
        self.last_exploratory_pareto_alternatives = exploratory_front
        self.last_constraint_counts = {
            "evaluated": res.evaluated,
            "estimator_feasible": res.estimator_feasible_count,
            "catalog_mapping_compliant": res.mapping_compliant_count,
            "phase8_closable": res.phase8_closable_count,
            "constraint_feasible": res.constraint_feasible_count,
            "official_pareto": len(res.pareto_front),
            "exploratory_pareto": len(res.exploratory_pareto_front),
        }
        self.last_search_coverage = {
            "mode": res.coverage_mode,
            "evaluated": res.evaluated,
            "search_space_size": res.search_space_size,
        }
        if res.recommendation_status == "RECOMMENDED":
            print(f"  [variation-DSE] explored {res.admitted_points} → recommended {res.recommended_choices}")
        else:
            print(
                f"  [variation-DSE] explored {res.admitted_points} → "
                f"{res.recommendation_status}; exploratory best={res.exploratory_choices}"
            )
        if res.recommended_capacity_mah is not None:
            print(f"  [variation-DSE] inner BO sized battery → {res.recommended_capacity_mah:.0f} mAh")
        if res.real_quality:
            dims = {k: round(v, 2) for k, v in res.real_quality.items()}
            print(f"     real quality: {dims}")
        for note in res.notes:
            print(f"     note: {note}")
        return ds, best_config, pareto_front

    @staticmethod
    def _print_exploration_summary(
        design_space: DesignSpace,
        best_config: DesignConfiguration,
        pareto_front: List[DesignConfiguration],
    ) -> None:
        """Print a transparent breakdown of MCTS exploration results."""
        summary = design_space.get_summary()
        diagnostics = design_space.objective_weights or {}
        iters = int(diagnostics.get("iterations_run", 0))
        early = bool(diagnostics.get("early_stopped", 0))
        # variation path records the true count in diagnostics (its configs aren't
        # all stored on the space); scalar path falls back to the summary count.
        evaluated = int(diagnostics.get("configurations_evaluated",
                                        summary['configurations_evaluated']))
        # size and the top-N list below must come from the SAME source, else they
        # contradict (e.g. "size 0" above "top 2").
        pareto_size = len(pareto_front)

        print(f"  ✓ Explored {evaluated} configurations "
              f"in {iters} iteration(s){' (early-stopped)' if early else ''}")
        print(f"  ✓ Pareto front size: {pareto_size}")

        # Show top-3 Pareto candidates
        top = pareto_front[:3]
        if top:
            print(f"  ✓ Pareto front (top {len(top)}):")
            for i, cfg in enumerate(top, 1):
                marker = "★" if cfg.id == best_config.id else " "
                params = ", ".join(f"{k}={v}" for k, v in cfg.parameters.items())
                scores = ", ".join(f"{k}={v:.2f}" for k, v in cfg.scores.items())
                print(f"     {marker} [{i}] overall={cfg.overall_score:.3f}  "
                      f"({scores})")
                print(f"          params: {params}")

        best_params = {k: v for k, v in best_config.parameters.items()}
        if best_config.name == "no_recommendable_design":
            print("  ⚠ No official configuration applied; Pareto results are exploratory\n")
        else:
            print(f"  ✓ Best config applied to model: {best_params}\n")

    # ------------------------------------------------------------------

    @staticmethod
    def _count_connects(model_text: str) -> int:
        """Number of `connect a.p to b.q` statements in the model text."""
        return len(re.findall(r"\bconnect\b", model_text, re.IGNORECASE))

    def _verification_gap_issues(self, sysml_text: str, model_name: str) -> List[str]:
        """Static verification-readiness audit (best-effort, no LLM).

        Projects the final verification matrix's `unassigned` set from the model
        text alone (see ``verification_audit``); any failure returns [] so the
        audit can never break the refinement loop.
        """
        try:
            from .verification_audit import verification_gap_issues
            return verification_gap_issues(
                sysml_text,
                model_name,
                allowed_req_ids=self._active_requirement_ids(),
            )
        except Exception:
            return []

    def _active_requirement_ids(self) -> Optional[set[str]]:
        """IDs admitted by Phase 1/frozen input; None only before input exists."""
        source_digests = (self.last_requirement_input or {}).get(
            "source_digests"
        )
        if not isinstance(source_digests, Mapping):
            return None
        return {str(req_id) for req_id in source_digests}

    def _functional_verification_gap_issues(
        self, sysml_text: str, model_name: str
    ) -> List[str]:
        """Functional subset of model-fixable verification gaps (fail closed)."""
        try:
            from .verification_audit import functional_verification_gap_issues
            return functional_verification_gap_issues(
                sysml_text, model_name, strict=True,
                allowed_req_ids=self._active_requirement_ids(),
            )
        except Exception as exc:
            raise RuntimeError(
                "functional verification audit failed; refusing to mark closure"
            ) from exc

    @staticmethod
    def _gap_req_ids(issues: List[str]) -> List[str]:
        return sorted(set(re.findall(
            r"\bREQ[_-]FUNC[_-]\d+\b", "\n".join(map(str, issues)), re.IGNORECASE
        )))

    def _functional_closure_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
        max_iters: int = 2,
    ) -> tuple[SysMLModel, float, Any]:
        """Close model-fixable FUNC gaps with dedicated, validated LLM surgery.

        This pass always runs after the ordinary refinement path, including when
        the quality loop exhausted its iteration budget. Each accepted edit must
        strictly reduce the functional gap ID set and must not regress syntax,
        simulation, or rule score. Remaining gaps are explicit terminal state,
        not a silently successful generation.
        """
        current = current_model
        current_sim = sim_result
        current_score = rule_score
        text = get_sysml_text(current)
        model_name = getattr(current, "name", None) or (
            self.state.system_name if self.state is not None else "System"
        )
        gaps = self._functional_verification_gap_issues(text, model_name)
        initial_ids = self._gap_req_ids(gaps)
        attempts = 0
        accepted = 0
        repair_contexts: List[Dict[str, Any]] = []

        if not gaps:
            self.last_functional_closure = {
                "status": "CLOSED",
                "initial_gap_req_ids": [],
                "remaining_gap_req_ids": [],
                "attempts": 0,
                "accepted_repairs": 0,
                "repair_contexts": [],
            }
            return current, current_score, current_sim

        print(f"\n  {'─'*62}", flush=True)
        print(
            f"  ▶  Functional closure  ({len(initial_ids)} gap(s), "
            f"max {max_iters} targeted pass{'es' if max_iters != 1 else ''})",
            flush=True,
        )

        if not self.use_surgical_refinement:
            print("  └─ ⚠ surgical refinement disabled; functional gaps remain", flush=True)
        else:
            from .surgical_refiner import (
                SurgicalAudit,
                attempt_surgical_refinement,
                build_dependency_closed_context,
            )
            from .verification_audit import behavioral_result_regressed

            for idx in range(max_iters):
                if not gaps:
                    break
                attempts += 1
                before_ids = set(self._gap_req_ids(gaps))
                print(
                    f"  │  Pass {idx + 1}/{max_iters}: targeted repair for "
                    f"{', '.join(sorted(before_ids))}",
                    flush=True,
                )
                full_text = get_sysml_text(current)
                context = build_dependency_closed_context(
                    full_text,
                    gaps,
                    allowed_req_ids=self._active_requirement_ids(),
                )
                if context is None:
                    repair_contexts.append({
                        "pass": idx + 1,
                        "status": "BLOCKED",
                        "reason": "dependency_closed_context_unresolved",
                        "target_req_ids": sorted(before_ids),
                        "llm_invoked": False,
                    })
                    print(
                        "  │    ⚠ dependency-closed owner context unresolved; "
                        "LLM not called",
                        flush=True,
                    )
                    continue
                surgical_audit = SurgicalAudit()
                repaired = attempt_surgical_refinement(
                    llm=self.llm,
                    model_text=full_text,
                    issues=gaps,
                    feedback=(
                        "This is the terminal functional-closure pass. Repair the "
                        "complete trigger -> reachable response entry action -> timing "
                        "constraint chain for every listed FUNC requirement."
                    ),
                    verbose=self.verbose,
                    audit=surgical_audit,
                    context_slice=context,
                )
                context_record = {
                    "pass": idx + 1,
                    "context": context.to_dict(),
                    "surgical_audit": surgical_audit.to_dict(),
                    "status": "CANDIDATE" if repaired is not None else "REJECTED",
                }
                repair_contexts.append(context_record)
                if repaired is None:
                    print("  │    ⚠ no syntax-safe surgical result", flush=True)
                    continue

                candidate = build_lite_model(
                    repaired.merged_text, model_name=model_name
                )
                self._restore_generation_plan_metadata(candidate)
                cand_syntax = check_syntax(repaired.merged_text)
                cand_sim = self._run_simulation(repaired.merged_text, model_name)
                cand_eval = self.evaluator.evaluate(
                    config=DesignConfiguration(
                        name=f"functional_closure_{idx + 1}", parameters={}
                    ),
                    model=candidate,
                    dse_config=dse_best_config,
                    syntax_result=cand_syntax,
                    sim_result=cand_sim,
                    requirements=requirements,
                )
                remaining = self._functional_verification_gap_issues(
                    repaired.merged_text, model_name
                )
                after_ids = set(self._gap_req_ids(remaining))
                progress = after_ids < before_ids
                regressed = (
                    cand_syntax.has_errors
                    or len(cand_sim.failed_scenarios())
                    > len(current_sim.failed_scenarios())
                    or behavioral_result_regressed(current_sim, cand_sim)
                    or cand_eval.weighted_total < current_score - 0.05
                )
                if progress and not regressed:
                    current = candidate
                    current_sim = cand_sim
                    current_score = cand_eval.weighted_total
                    gaps = remaining
                    accepted += 1
                    context_record["status"] = "ACCEPTED"
                    print(
                        f"  │    ✓ accepted: {len(before_ids)} -> "
                        f"{len(after_ids)} functional gap(s)",
                        flush=True,
                    )
                else:
                    why = (
                        "regression" if regressed
                        else "no functional-gap reduction"
                    )
                    context_record["status"] = "REJECTED"
                    context_record["post_merge_reason"] = why
                    print(f"  │    ⚠ rejected: {why}", flush=True)

        remaining_ids = self._gap_req_ids(gaps)
        status = "CLOSED" if not remaining_ids else "OPEN"
        self.last_functional_closure = {
            "status": status,
            "initial_gap_req_ids": initial_ids,
            "remaining_gap_req_ids": remaining_ids,
            "attempts": attempts,
            "accepted_repairs": accepted,
            "repair_contexts": repair_contexts,
        }
        if remaining_ids:
            print(
                "  └─ ✗ functional closure OPEN: " + ", ".join(remaining_ids),
                flush=True,
            )
        else:
            print("  └─ ✓ functional closure CLOSED", flush=True)
        print(f"  {'─'*62}", flush=True)
        return current, current_score, current_sim

    def _verification_anchor_pass(
        self,
        current_model: SysMLModel,
        sim_result: Any,
        rule_score: float,
        verify_gaps: List[str],
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
    ) -> tuple[SysMLModel, Any, bool]:
        """Run the one bounded verification-anchor pass on any clean exit path.

        Previously this lived only inside the "quality already clean" branch.
        When reachability was repaired by ``_sim_refinement_loop``, the method
        returned immediately and skipped anchoring altogether.  Keeping the pass
        in one helper makes both paths use identical syntax/simulation/score
        gates and preserves the one-pass bound.
        """
        if not verify_gaps or not self.use_surgical_refinement:
            return current_model, sim_result, False

        print(f"  ~ Quality met, but {len(verify_gaps)} requirement(s) "
              f"would be UNASSIGNED in the verification matrix — "
              f"one surgical anchor pass", flush=True)
        from .surgical_refiner import (
            SurgicalAudit,
            attempt_surgical_refinement,
            build_dependency_closed_context,
        )
        full_text = get_sysml_text(current_model)
        context = build_dependency_closed_context(
            full_text,
            verify_gaps,
            allowed_req_ids=self._active_requirement_ids(),
        )
        if context is None:
            self.last_verification_anchor_attempts.append({
                "status": "BLOCKED",
                "reason": "dependency_closed_context_unresolved",
                "target_req_ids": self._gap_req_ids(verify_gaps),
                "llm_invoked": False,
            })
            print(
                "  ⚠ Anchor pass blocked: dependency-closed owner context "
                "could not be resolved",
                flush=True,
            )
            return current_model, sim_result, False
        surgical_audit = SurgicalAudit()
        anchored = attempt_surgical_refinement(
            llm=self.llm,
            model_text=full_text,
            issues=verify_gaps,
            verbose=self.verbose,
            audit=surgical_audit,
            context_slice=context,
        )
        attempt_record = {
            "status": "CANDIDATE" if anchored is not None else "REJECTED",
            "context": context.to_dict(),
            "surgical_audit": surgical_audit.to_dict(),
        }
        self.last_verification_anchor_attempts.append(attempt_record)
        if anchored is None:
            print("  ⚠ Anchor pass not applicable (LLM output failed "
                  "the surgical gates)", flush=True)
            return current_model, sim_result, False

        anchor_model = build_lite_model(
            anchored.merged_text, model_name=current_model.name)
        self._restore_generation_plan_metadata(anchor_model)
        anchor_sim = self._run_simulation(
            anchored.merged_text, current_model.name)
        anchor_eval = self.evaluator.evaluate(
            config=DesignConfiguration(name="anchor_pass", parameters={}),
            model=anchor_model,
            dse_config=dse_best_config,
            syntax_result=check_syntax(anchored.merged_text),
            sim_result=anchor_sim,
            requirements=requirements,
        )
        remaining = self._verification_gap_issues(
            anchored.merged_text, current_model.name)
        from .verification_audit import behavioral_result_regressed
        regressed = (
            bool(anchor_sim.failed_scenarios())
            or behavioral_result_regressed(sim_result, anchor_sim)
            or anchor_eval.weighted_total < rule_score - 0.05
        )
        if not regressed and len(remaining) < len(verify_gaps):
            attempt_record["status"] = "ACCEPTED"
            print(f"  ✓ Anchor pass accepted: verification gaps "
                  f"{len(verify_gaps)} → {len(remaining)}", flush=True)
            return anchor_model, anchor_sim, True

        attempt_record["status"] = "REJECTED"
        attempt_record["post_merge_reason"] = (
            "regression" if regressed
            else "no_verification_gap_reduction"
        )
        print("  ⚠ Anchor pass rejected (no gap reduction or "
              "regression) — keeping the original model", flush=True)
        return current_model, sim_result, False

    def _blended_iteration_score(
        self,
        rule_score: float,
        eval_result,
        current_model: SysMLModel,
        requirements: List[str],
    ):
        """Blend rule-based and LLM evaluation scores for one iteration.

        The LLM evaluation is skipped when the rule score already meets the
        quality threshold or a [VETO] fired — the blended score could not
        change the outcome in either case.  Returns
        ``(score, llm_overall, cot_eval, veto_fired)``.
        """
        veto_fired = any(
            str(iss).startswith("[VETO]") for iss in eval_result.issues
        )
        if rule_score >= self.quality_threshold or veto_fired:
            return rule_score, None, None, veto_fired

        cot_eval = self.cot.evaluate_design(
            model_text=current_model.to_sysml_text(),
            requirements=requirements,
        )
        cot_scores = cot_eval.get_scores() or {}
        llm_overall = cot_scores.get("overall", None)
        if llm_overall is not None:
            score = round(
                self.rule_weight * rule_score + self.llm_weight * float(llm_overall),
                4,
            )
        else:
            score = rule_score
        return score, llm_overall, cot_eval, veto_fired

    def _early_exit_gates(
        self,
        sim_result,
        syntax_result,
        requirements: List[str],
        model: Optional[SysMLModel] = None,
    ):
        """Hard gates that must all pass before the quality-threshold early
        exit: behavioral state machines, scenario reachability, sema errors.
        A high rule-score can coexist with state-machine failures or
        connectivity gaps — those must be resolved first.
        Returns ``(behavioral_ok, reachability_ok, sema_ok)``."""
        _safe_reqs = [r for r in requirements if "-SAFE-" in r or "SAFE" in r.upper()[:10]]
        _br = sim_result.behavioral_result
        behavioral_ok = (
            _br is None
            or (
                _br.extracted_sm_count == 0
                and not _safe_reqs
            )
            or (
                _br.extracted_sm_count > 0
                and _br.sim_score >= 1.0
            )
        )
        structural_report = None
        if model is not None:
            structural_report = self._validate_terminal_structural_obligations(
                model,
                get_sysml_text(model),
                model.name,
            )
        reachability_ok = (
            structural_report.get("status") == "PASS"
            if structural_report is not None
            else not sim_result.failed_scenarios()
        )
        sema_ok = syntax_result is None or not syntax_result.has_errors
        return behavioral_ok, reachability_ok, sema_ok

    def _resolve_after_forced_fix(
        self,
        current_model: SysMLModel,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration],
        iteration: int,
        score: float,
    ):
        """Re-check the model after the forced sim fix pass.

        When the fix cleared every failed scenario, the score is re-evaluated
        with the fixed sim (plus one bounded verification-anchor pass) and
        ``resolved`` is True — the caller returns immediately.  Otherwise
        everything is passed back unchanged for the escalation path.
        Returns ``(resolved, model, score, sim_result)``.
        """
        _sysml_after = get_sysml_text(current_model)
        sim_result = self._run_simulation(_sysml_after, current_model.name)
        if sim_result.failed_scenarios():
            return False, current_model, score, sim_result

        print(f"  └─ Simulation fully resolved ✓", flush=True)
        # Re-evaluate with the fixed sim so the returned score
        # reflects the model's true post-fix quality.
        eval_after = self.evaluator.evaluate(
            config=DesignConfiguration(
                name=f"iteration_{iteration}_fixed",
                parameters={},
            ),
            model=current_model,
            dse_config=dse_best_config,
            syntax_result=check_syntax(_sysml_after),
            sim_result=sim_result,
            requirements=requirements,
        )
        score = eval_after.weighted_total
        post_fix_gaps = self._verification_gap_issues(
            _sysml_after, current_model.name)
        current_model, sim_result, anchor_accepted = (
            self._verification_anchor_pass(
                current_model=current_model,
                sim_result=sim_result,
                rule_score=score,
                verify_gaps=post_fix_gaps,
                requirements=requirements,
                dse_best_config=dse_best_config,
            )
        )
        if anchor_accepted:
            score = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}_fixed_anchor",
                    parameters={},
                ),
                model=current_model,
                dse_config=dse_best_config,
                syntax_result=check_syntax(
                    get_sysml_text(current_model)),
                sim_result=sim_result,
                requirements=requirements,
            ).weighted_total
        return True, current_model, score, sim_result

    def _generate_refinement_candidate(
        self,
        current_model: SysMLModel,
        current_sysml: str,
        eval_result,
        refinement_feedback: str,
        requirements: List[str],
    ) -> Optional[SysMLModel]:
        """Surgical refinement first: the LLM returns only the blocks it
        changes; the merge is syntax-gated and cannot shed connects on
        untouched components (prevention, not the after-the-fact rejection
        the full rewrite needs).  Falls back to the legacy whole-model
        rewrite on any failure.  Returns the candidate model or None."""
        if self.use_surgical_refinement:
            from .surgical_refiner import (
                SurgicalAudit,
                attempt_surgical_refinement,
            )
            surgical = attempt_surgical_refinement(
                llm=self.llm,
                model_text=current_sysml,
                issues=eval_result.issues + eval_result.recommendations,
                feedback=refinement_feedback,
                verbose=self.verbose,
                audit=SurgicalAudit(),
            )
            if surgical is not None:
                print(f"  ✓ Surgical refinement: {surgical.summary()}", flush=True)
                candidate = build_lite_model(
                    surgical.merged_text, model_name=current_model.name
                )
                raw_plan = (
                    getattr(current_model, "metadata", None) or {}
                ).get("whole_model_generation_plan")
                if isinstance(raw_plan, Mapping):
                    if getattr(candidate, "metadata", None) is None:
                        candidate.metadata = {}
                    candidate.metadata["whole_model_generation_plan"] = dict(
                        raw_plan
                    )
                return candidate
            print("  ⚠ Surgical refinement not applicable — "
                  "falling back to full rewrite", flush=True)

        refine_result = self.design_agent.run({
            "system_name": current_model.name,
            "requirements": requirements,
            "existing_model": current_model,
            "refinement_feedback": refinement_feedback,
            "refinement_issues": eval_result.issues + eval_result.recommendations,
            "verbose": self.verbose,
        })
        if (refine_result.success
                and isinstance(refine_result.output, _SysMLModelTypes)):
            candidate = refine_result.output
            raw_plan = (
                getattr(current_model, "metadata", None) or {}
            ).get("whole_model_generation_plan")
            if (
                isinstance(raw_plan, Mapping)
                and "whole_model_generation_plan"
                not in (getattr(candidate, "metadata", None) or {})
            ):
                if getattr(candidate, "metadata", None) is None:
                    candidate.metadata = {}
                candidate.metadata["whole_model_generation_plan"] = dict(
                    raw_plan
                )
            return candidate
        return None

    def _accept_refinement_candidate(
        self,
        *,
        candidate: SysMLModel,
        current_sysml: str,
        rule_score: float,
        dse_best_config: Optional[DesignConfiguration],
        requirements: List[str],
        connectivity_floor: bool,
    ) -> Optional[SysMLModel]:
        """P0 regression prevention + connectivity floor.

        The candidate is evaluated with the SAME inputs as rule_score (sim +
        syntax + mcts_config) — omitting sim_result makes
        behavioral_verification fall back to 1.0 and omitting mcts_config
        changes the weight denominator; both bias the comparison toward
        accepting the candidate.  check_syntax and _run_simulation are local
        (no LLM cost).  Under ``connectivity_floor`` any candidate that sheds
        connect statements relative to the model it was refined from is
        rejected — the resolved variation model arrives fully wired and a
        full LLM rewrite tends to drop connects on converted components.
        Returns the accepted candidate (after the simulation inner loop) or
        None when rejected."""
        cand_sysml = get_sysml_text(candidate)
        raw_plan = (
            getattr(candidate, "metadata", None) or {}
        ).get("whole_model_generation_plan")
        if isinstance(raw_plan, Mapping):
            from ..prototyping.generation_plan import (
                ModelGenerationPlan,
                apply_generation_plan,
            )

            planned_candidate, conformance = apply_generation_plan(
                cand_sysml,
                ModelGenerationPlan.from_dict(raw_plan),
            )
            if conformance.get("status") != "PASS":
                print(
                    "  ⚠ Refinement violates the frozen typed structure "
                    f"({len(conformance.get('issues', ())) or 1} issue(s)) "
                    "— rejected before simulation repair",
                    flush=True,
                )
                return None
            if planned_candidate != cand_sysml:
                cand_sysml = planned_candidate
                self._sync_model_text(candidate, cand_sysml)
        cand_syntax = check_syntax(cand_sysml)
        cand_sim = self._run_simulation(cand_sysml, candidate.name)
        candidate_eval = self.evaluator.evaluate(
            config=DesignConfiguration(name="candidate", parameters={}),
            model=candidate,
            dse_config=dse_best_config,
            syntax_result=cand_syntax,
            sim_result=cand_sim,
            requirements=requirements,
        )
        delta = candidate_eval.weighted_total - rule_score
        delta_str = f"{delta:+.3f}"
        if connectivity_floor:
            cur_connects = self._count_connects(current_sysml)
            cand_connects = self._count_connects(cand_sysml)
            if cand_connects < cur_connects:
                print(
                    f"  ⚠ Refinement dropped connectivity "
                    f"({cur_connects} → {cand_connects} connects) — "
                    f"rejected to preserve resolved variation wiring",
                    flush=True,
                )
                return None
        if candidate_eval.weighted_total >= rule_score - 0.05:
            print(
                f"  ✓ Refinement accepted  "
                f"rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f} "
                f"({delta_str})",
                flush=True,
            )
            # ── Simulation inner loop ── re-run simulation on the accepted
            # candidate and attempt up to MAX_SIM_INNER_ITERS targeted fixes
            # before handing the model back to the outer loop.
            refined = self._sim_refinement_loop(
                candidate, requirements, max_iters=3
            )
            if getattr(refined, "metadata", None) is None:
                refined.metadata = {}
            refined.metadata["_accepted_refinement_rule_score"] = (
                candidate_eval.weighted_total
            )
            return refined
        print(
            f"  ⚠ Refinement regression detected "
            f"(rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f}), "
            f"keeping current model",
            flush=True,
        )
        return None

    def _attempt_refinement(
        self,
        *,
        current_model: SysMLModel,
        current_sysml: str,
        eval_result,
        cot_eval,
        persistent: List[str],
        mcts_constraints: str,
        sim_issues: List[str],
        requirements: List[str],
        rule_score: float,
        dse_best_config: Optional[DesignConfiguration],
        connectivity_floor: bool,
    ) -> SysMLModel:
        """One LLM refinement step: build the feedback prompt, generate a
        candidate (surgical first, full rewrite fallback), and adopt it only
        when the regression and connectivity guards accept it.  Returns the
        model to carry into the next iteration."""
        print(f"\n  ⟳  Refining model …", flush=True)
        refinement_feedback = self._build_refinement_feedback(
            eval_result,
            cot_eval.final_answer if cot_eval else "",
            persistent_issues=persistent,
            mcts_constraints=mcts_constraints,
            sim_issues=sim_issues,
        )
        candidate = self._generate_refinement_candidate(
            current_model, current_sysml, eval_result,
            refinement_feedback, requirements,
        )
        if candidate is None:
            return current_model
        accepted = self._accept_refinement_candidate(
            candidate=candidate,
            current_sysml=current_sysml,
            rule_score=rule_score,
            dse_best_config=dse_best_config,
            requirements=requirements,
            connectivity_floor=connectivity_floor,
        )
        return accepted if accepted is not None else current_model

    def _iterative_refinement(
        self,
        model: SysMLModel,
        requirements: List[str],
        dse_best_config: Optional[DesignConfiguration] = None,
        connectivity_floor: bool = False,
    ) -> tuple[SysMLModel, float, Any]:
        """Phase 4-5: Evaluate and iteratively refine the design.

        Key improvements over the naive version:

        P0 — Best-model tracking: ``best_model`` is updated whenever the
             blended score improves; the loop always returns the peak-scoring
             model, not the last one.

        P0 — Regression guard: a refined candidate is only accepted when its
             rule-based score does not fall more than 5 pp below the current
             model's score.  If it does, the current model is kept and a
             warning is printed.

        P1 — LLM-guided refinement with no explicit issues: when the rule
             evaluator reports zero issues but the LLM returned non-empty
             feedback (and the score is still below threshold), refinement is
             still triggered.  This avoids silent stalls.

        P1 — Persistent issue escalation: issues that recur across iterations
             are flagged with ``[PERSISTENT]`` in the refinement prompt so the
             LLM can prioritise them.

        P1 — MCTS grounding: architectural decisions from Phase 3 (redundancy
             level, frequency, protocol, topology, sensor count) are prepended
             to every refinement prompt so the LLM implements them rather than
             guessing.

        P2 — Skip expensive LLM call when rule_score already meets the
             quality threshold — the blended score would pass anyway.

        P2 — Configurable blend weights via ``self.rule_weight`` /
             ``self.llm_weight`` (set in ``__init__``).
        """
        current_model = model
        best_score = 0.0
        best_model = model
        best_sim_result: Any = None          # tracks sim matching best_model
        last_sim_result: Any = None          # most recent sim result
        seen_issues: Dict[str, int] = {}  # issue text → occurrence count

        # Pre-compute MCTS constraint text once — same for every iteration
        mcts_constraints = (
            _build_dse_design_constraints(dse_best_config)
            if dse_best_config else ""
        )
        if self.verbose and mcts_constraints:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] MCTS constraints injected into every refinement prompt")
            print(f"  {'─'*60}")
            print(mcts_constraints)

        for iteration in range(self.max_iterations):
            self.state.iteration = iteration + 1

            # ── Step 0: Syntax gate — fix errors before evaluation ────────
            current_sysml = get_sysml_text(current_model)
            current_sysml, fixed_model, syntax_result = self._syntax_gate(
                current_sysml, current_model, requirements, max_attempts=3
            )
            if fixed_model is not None:
                current_model = fixed_model

            # ── Step 0.5: Connect audit — remove type/direction-invalid connects ─
            current_sysml, current_model = self._connect_audit_step(
                current_sysml, current_model
            )

            # ── Behavioral simulation (runs before eval to feed into score) ─
            sim_result = self._run_simulation(current_sysml, current_model.name)
            sim_issues = self._format_sim_issues(sim_result, requirements=requirements)

            # ── Rule-based evaluation (pass cached syntax + sim results;
            #    requirements enable requirement-derived dimension weights) ──
            eval_result = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}",
                    parameters={},
                ),
                model=current_model,
                dse_config=dse_best_config,
                syntax_result=syntax_result,
                sim_result=sim_result,
                requirements=requirements,
            )
            rule_score = eval_result.weighted_total
            # ── Verification-readiness audit (static matrix projection) ──────
            # Runs the SAME logic the final verification matrix uses, with no
            # execution results: requirements that would land `unassigned` (no
            # tier anchors them at all — execution-independent) become refinement
            # issues NOW, while the LLM is still in the loop. Advisory: they ride
            # along in refinement prompts and get ONE bounded surgical anchor
            # pass at the quality gate; they never block early exit on their own.
            verify_gaps = self._verification_gap_issues(
                current_sysml, current_model.name)
            if verify_gaps and isinstance(eval_result.issues, list):
                eval_result.issues.extend(verify_gaps)
            # How much the pass/fail verdict depends on the weighting at all —
            # sampled over the weight simplex (answers "would another weighting
            # flip the outcome?").  Defensive: test doubles may not provide it.
            _rob_fn = getattr(self.evaluator, "verdict_robustness", None)
            verdict_rob = _rob_fn(eval_result) if callable(_rob_fn) else None

            # ── LLM evaluation (skip when rule score already sufficient OR
            #    when a [VETO] fired in the rule evaluator) ─────────────────
            score, llm_overall, cot_eval, veto_fired = self._blended_iteration_score(
                rule_score, eval_result, current_model, requirements
            )

            self.state.evaluation_history.append({
                "iteration": iteration + 1,
                "score": score,
                "rule_score": rule_score,
                "llm_score": llm_overall,
                "issues": eval_result.issues,
                "sim_score": sim_result.reachability_score,
                "sim_passed": len(sim_result.passed_scenarios()),
                "sim_total": len(sim_result.scenario_results),
                "weights_used": getattr(eval_result, "weights_used", {}),
                "verdict_robustness": verdict_rob,
            })
            if verdict_rob is not None:
                print(f"  Verdict robustness over the weight simplex: "
                      f"{verdict_rob:.0%} of sampled weightings agree", flush=True)

            # ── Always-visible iteration summary ─────────────────────────
            self._print_iteration_summary(
                iteration=iteration + 1,
                score=score,
                rule_score=rule_score,
                llm_overall=llm_overall,
                eval_result=eval_result,
                sim_result=sim_result,
                veto_fired=veto_fired,
                syntax_result=syntax_result,
            )

            if self.verbose and cot_eval:
                cot_scores = cot_eval.get_scores() or {}
                if cot_scores:
                    print("  [DEBUG] LLM sub-scores: "
                          + ", ".join(f"{k}={v:.2f}" for k, v in cot_scores.items()))

            # ── P0: Best-model tracking ───────────────────────────────────
            last_sim_result = sim_result
            if score > best_score:
                best_score = score
                best_model = current_model
                best_sim_result = sim_result

            # ── Early exit ────────────────────────────────────────────────
            _force_llm_refinement = False   # set True when surgical fix fails
            if score >= self.quality_threshold:
                # Only exit if behavioral simulation, reachability, and sema
                # are all clean.  A high rule-score can coexist with state-machine
                # failures or connectivity gaps — those must be resolved first.
                behavioral_ok, reachability_ok, sema_ok = self._early_exit_gates(
                    sim_result, syntax_result, requirements, current_model
                )

                if behavioral_ok and reachability_ok and sema_ok:
                    # ── One bounded verification-anchor pass ──────────────
                    # Quality is met, but the static audit predicts unassigned
                    # matrix rows. Exactly ONE surgical pass scoped to those
                    # issues (gates: syntax, connects preserved, requirement-def
                    # set frozen, satisfy links may not shrink). Accepted only
                    # if gaps actually shrink AND nothing regresses (local sim +
                    # rule score re-checked, no LLM cost). Success or not, we
                    # return afterwards — anchors are advisory, never a loop.
                    current_model, sim_result, _ = self._verification_anchor_pass(
                        current_model=current_model,
                        sim_result=sim_result,
                        rule_score=rule_score,
                        verify_gaps=verify_gaps,
                        requirements=requirements,
                        dse_best_config=dse_best_config,
                    )
                    print(f"  ✓ Quality threshold {self.quality_threshold} reached",
                          flush=True)
                    return current_model, score, sim_result

                # Score met but hard failures remain — one targeted fix pass
                issues_desc = ", ".join(filter(None, [
                    "behavioral" if not behavioral_ok else "",
                    "reachability" if not reachability_ok else "",
                    "sema" if not sema_ok else "",
                ]))
                print(
                    f"  ~ Quality threshold met (score={score:.3f}) "
                    f"but {issues_desc} issues remain — forcing sim fix pass",
                    flush=True,
                )
                current_model = self._sim_refinement_loop(
                    current_model, requirements, max_iters=2
                )

                # Re-check after surgical fix
                resolved, current_model, score, sim_result = (
                    self._resolve_after_forced_fix(
                        current_model, requirements, dse_best_config,
                        iteration, score,
                    )
                )
                last_sim_result = sim_result
                if resolved:
                    return current_model, score, sim_result

                # Surgical fix insufficient
                remaining = self.max_iterations - iteration - 1
                if remaining == 0:
                    print(
                        f"  ⚠ Surgical fix insufficient — no iterations remaining, "
                        f"returning the improved terminal model for final re-evaluation",
                        flush=True,
                    )
                    return current_model, score, sim_result

                sim_issues = self._format_sim_issues(sim_result, requirements=requirements)
                failing_count = len(sim_result.failed_scenarios())
                escalation_reason = (
                    f"{failing_count} scenario(s) still failing"
                    if failing_count
                    else "behavioral/semantic gates still unmet"
                )
                print(
                    f"  ⚠ Surgical fix insufficient ({escalation_reason}) "
                    f"— escalating to LLM refinement ({remaining} iteration(s) remaining)",
                    flush=True,
                )
                _force_llm_refinement = True   # trigger LLM refinement below

            # ── P1: Persistent issue tracking ─────────────────────────────
            for issue in eval_result.issues:
                seen_issues[issue] = seen_issues.get(issue, 0) + 1
            persistent = [iss for iss, cnt in seen_issues.items() if cnt > 1]

            # ── P1: Refinement trigger ────────────────────────────────────
            has_issues = bool(eval_result.issues) or _force_llm_refinement
            has_llm_feedback = cot_eval is not None and bool(cot_eval.final_answer)

            if has_issues or has_llm_feedback:
                current_model = self._attempt_refinement(
                    current_model=current_model,
                    current_sysml=current_sysml,
                    eval_result=eval_result,
                    cot_eval=cot_eval,
                    persistent=persistent,
                    mcts_constraints=mcts_constraints,
                    sim_issues=sim_issues,
                    requirements=requirements,
                    rule_score=rule_score,
                    dse_best_config=dse_best_config,
                    connectivity_floor=connectivity_floor,
                )
                accepted_score = (
                    getattr(current_model, "metadata", {}) or {}
                ).pop("_accepted_refinement_rule_score", None)
                if (
                    isinstance(accepted_score, (int, float))
                    and float(accepted_score) > best_score
                ):
                    # The last allowed iteration has no next pass in which to
                    # promote an accepted candidate. Record it immediately,
                    # paired with simulation from the exact returned text.
                    candidate_sim = self._run_simulation(
                        get_sysml_text(current_model), current_model.name
                    )
                    best_model = current_model
                    best_score = float(accepted_score)
                    best_sim_result = candidate_sim
                    last_sim_result = candidate_sim

        return best_model, best_score, best_sim_result or last_sim_result

    # ------------------------------------------------------------------
    # Phase 3.5: SITL-L1 refinement loop
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sitl_feedback(items: List[Dict[str, Any]]) -> str:
        """Format unresolved SITL parameter gaps as a refinement prompt section."""
        lines = [
            "SITL parameter-mapping gaps (ArduPilot L1):",
            "  Each requirement below maps to an ArduPilot parameter that cannot be",
            "  derived because the model lacks the guard/attribute it reads from.",
            "  Add the missing element to the named part using valid SysML v2 syntax.",
            "",
        ]
        for i in items:
            lines.append(f"- {i['message']}")
        return "\n".join(lines)

    def _sitl_refinement_loop(
        self,
        model: SysMLModel,
        requirements: List[str],
        base_score: float,
        base_sim: Any,
        max_iters: int = 2,
    ) -> Tuple[SysMLModel, float, Any]:
        """Feed unresolved ArduPilot parameter mappings back to the design LLM.

        An unresolved mapping means the model genuinely lacks a guard/attribute
        a requirement needs (the tooling-side source of 'unresolved' was removed
        by resolving AST-matched thresholds).  Each pass: run the L1 mapping →
        if unresolved, build feedback → refine → accept only when there is no
        syntax/sim/score regression.  Terminates on clean L1, no progress, or
        regression.  Returns (model, score, sim) for the final accepted model.
        """
        from ..sitl.requirement_linker import RequirementLinker

        current, cur_score, cur_sim = model, base_score, base_sim
        last_sig = None

        print(f"\n  {'─'*62}", flush=True)
        print(f"  ▶  SITL-L1 refinement loop  (max {max_iters} pass"
              f"{'es' if max_iters > 1 else ''})")

        for it in range(max_iters):
            linker = RequirementLinker(current, llm=self.llm, verbose=self.verbose)
            items = linker.unresolved_feedback()

            if not items:
                print(f"  │  Pass {it+1}/{max_iters}  ✓ all SITL parameters resolved")
                print(f"  └─ SITL-L1 clean", flush=True)
                break

            print(f"  │  Pass {it+1}/{max_iters}  {len(items)} unresolved parameter(s):",
                  flush=True)
            for i in items:
                print(f"  │    ✗ {i['req_id']} → {i['param']} ({i['kind']})")

            sig = frozenset((i["req_id"], i["param"]) for i in items)
            if sig == last_sig:
                print(f"  └─ ⚠ no progress (same unresolved set) — stopping", flush=True)
                break
            last_sig = sig

            refine_result = self.design_agent.run({
                "system_name": current.name,
                "requirements": requirements,
                "existing_model": current,
                "refinement_feedback": self._build_sitl_feedback(items),
                "refinement_issues": [i["message"] for i in items],
                "verbose": self.verbose,
            })
            if not (refine_result.success
                    and isinstance(refine_result.output, _SysMLModelTypes)):
                print(f"  └─ ⚠ refinement produced no usable model — stopping", flush=True)
                break

            candidate = refine_result.output
            # Same-basis comparison: evaluate the candidate with its own fresh
            # syntax + sim results (mirrors the regression-check fix elsewhere).
            cand_sysml = get_sysml_text(candidate)
            cand_syntax = check_syntax(cand_sysml)
            cand_sim = self._run_simulation(cand_sysml, candidate.name)
            cand_eval = self.evaluator.evaluate(
                config=DesignConfiguration(name="sitl_candidate", parameters={}),
                model=candidate,
                syntax_result=cand_syntax,
                sim_result=cand_sim,
                requirements=requirements,
            )
            if cand_eval.weighted_total < cur_score - 0.05:
                print(f"  └─ ⚠ regression (score {cur_score:.3f} → "
                      f"{cand_eval.weighted_total:.3f}) — keeping previous model",
                      flush=True)
                break

            print(f"  │  ✓ accepted  score {cur_score:.3f} → "
                  f"{cand_eval.weighted_total:.3f}", flush=True)
            current, cur_score, cur_sim = candidate, cand_eval.weighted_total, cand_sim

        print(f"  {'─'*62}", flush=True)
        return current, cur_score, cur_sim

    def _print_iteration_summary(
        self,
        iteration: int,
        score: float,
        rule_score: float,
        llm_overall,
        eval_result,
        sim_result: "SimulationResult",
        veto_fired: bool,
        syntax_result=None,
    ) -> None:
        """Print a self-contained, always-visible summary block for one iteration."""
        W = 62
        bar_w = 30

        def bar(v: float) -> str:
            filled = round(v * bar_w)
            return "█" * filled + "░" * (bar_w - filled)

        def score_icon(v: float) -> str:
            if v >= 0.85: return "✓"
            if v >= 0.65: return "~"
            return "✗"

        llm_str = f"{llm_overall:.3f}" if llm_overall is not None else " N/A "
        veto_tag = "  ← VETO" if veto_fired else ""

        print(f"\n  {'─'*W}", flush=True)
        print(f"  Iteration {iteration}  │  score={score:.3f}  "
              f"rule={rule_score:.3f}  llm={llm_str}{veto_tag}")
        print(f"  {'─'*W}")

        # ── Syntax check block ───────────────────────────────────────────
        if syntax_result is not None:
            if not syntax_result.has_errors:
                print(f"  [SYNTAX]  ✓ no errors")
            else:
                n_p = len(syntax_result.parser_errors)
                n_s = len(syntax_result.sema_errors)
                print(f"  [SYNTAX]  ✗ {n_p} parser  {n_s} sema")
                for e in (syntax_result.parser_errors + syntax_result.sema_errors)[:5]:
                    print(f"    L{e['line']:>3}: {e['message'][:W-10]}")
            print(f"  {'-'*W}")

        # ── Dimension scores table ────────────────────────────────────────
        dim_scores = eval_result.criteria_scores or {}
        dim_labels = {
            "syntactic_validity":       "Syntactic valid ",
            "requirement_coverage":     "Req coverage    ",
            "structural_completeness":  "Struct complete ",
            "behavioral_verification":  "Behav verificat ",
            "safety_assurance":         "Safety assurance",
            "interface_quality":        "Interface quality",
            "mcts_fidelity":            "MCTS fidelity   ",
        }
        for key, label in dim_labels.items():
            v = dim_scores.get(key, None)
            if v is None:
                continue
            icon = score_icon(v)
            print(f"  {icon} {label}  {bar(v)}  {v:.3f}")

        # ── Structural reachability block ─────────────────────────────────
        sim_passed = len(sim_result.passed_scenarios())
        sim_total  = len(sim_result.scenario_results)
        sim_score  = sim_result.reachability_score
        sim_icon   = score_icon(sim_score)
        print(f"  {'-'*W}")
        print(f"  [STRUCTURAL]  {sim_icon} {sim_score:.3f}  ({sim_passed}/{sim_total} scenarios)")

        # Show paths for safety/emergency passing scenarios
        for r in sim_result.passed_scenarios():
            if "safety" in r.tags or "emergency" in r.tags:
                path_parts = [n for n in r.path if "." not in n]
                path_str = " → ".join(path_parts) if path_parts else "(direct)"
                print(f"    ✓ [{'/'.join(r.tags):<20}] {r.scenario_name}")
                print(f"       {path_str[:W-7]}")

        # Show all passing nominal scenarios (compact, one line each)
        nominal_passed = [r for r in sim_result.passed_scenarios()
                          if "safety" not in r.tags and "emergency" not in r.tags]
        if nominal_passed:
            print(f"    ✓ nominal ({len(nominal_passed)} passed): "
                  + ", ".join(r.scenario_name[:20] for r in nominal_passed[:4])
                  + ("…" if len(nominal_passed) > 4 else ""))

        # Show failed scenarios with reason
        for r in sim_result.failed_scenarios():
            tgts = ", ".join(r.unreachable_targets) or "?"
            print(f"    ✗ {r.scenario_name}  →  can't reach: {tgts}")
            for w in r.warnings:
                print(f"      ⚠ {w}")

        # ── Behavioral simulation (state machine) block ───────────────────
        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.extracted_sm_count > 0:
            b_passed = br.passed_count()
            b_total  = len(br.scenario_results)
            b_icon   = score_icon(br.sim_score)
            print(f"  {'-'*W}")
            print(f"  [BEHAVIORAL]  {b_icon} {br.sim_score:.3f}  "
                  f"({b_passed}/{b_total} state machines)  "
                  f"extracted: {br.extracted_sm_count}")
            for sr in br.scenario_results:
                icon = "✓" if sr.passed else "✗"
                # Compact trigger line
                trig = ""
                if sr.trigger_value is not None:
                    trig = f"  val={sr.trigger_value}"
                elif sr.trigger_step is not None:
                    trig = f"  step={sr.trigger_step}"
                # Show timeline entries (guard drive + entry action)
                key_lines = [l for l in sr.timeline
                             if "Driving" in l or "Flipping" in l
                             or "entry action" in l or "at trigger" in l]
                print(f"    {icon} {sr.state_machine}{trig}")
                for tl in key_lines[:3]:
                    print(f"       {tl.strip()[:W-7]}")
                for v in sr.violations:
                    print(f"       ⚠ {v[:W-7]}")

        # ── Issues (always shown) ─────────────────────────────────────────
        if eval_result.issues:
            print(f"  {'-'*W}")
            print(f"  Issues ({len(eval_result.issues)}):")
            persistent_set = set()
            for iss in eval_result.issues:
                tag = "  ← PERSISTENT" if iss in persistent_set else ""
                display = iss.lstrip("[VETO] ").lstrip("[SIM] ")
                print(f"    • {display[:W-4]}{tag}")

        # ── Recommendations (top 3) ───────────────────────────────────────
        if eval_result.recommendations:
            print(f"  {'-'*W}")
            print(f"  Recommendations (top {min(3, len(eval_result.recommendations))}):")
            for rec in eval_result.recommendations[:3]:
                words = rec.split()
                line, lines_out = "", []
                for w in words:
                    if len(line) + len(w) + 1 > W - 6:
                        lines_out.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    lines_out.append(line)
                for i, l in enumerate(lines_out):
                    prefix = "    → " if i == 0 else "       "
                    print(f"{prefix}{l}")
        print(f"  {'─'*W}", flush=True)

    # ------------------------------------------------------------------
    # Simulation inner refinement loop
    # ------------------------------------------------------------------

    def _port_fixer_fallback(
        self,
        sysml: str,
        directory,
        failed_payload: List[Dict],
        isolated_parts,
        current: SysMLModel,
    ):
        """connectivity_fixer found no usable connects — fall back to
        port_fixer: add missing port declarations on part defs, then retry
        connectivity once with the widened port directory.

        Returns ``(sysml, merge, outcome)`` where outcome is:
          "merged" — a validated connect merge is ready to apply;
          "retry"  — intermediate text persisted, caller continues next pass;
          "stop"   — unrecoverable, caller breaks out of the loop.
        """
        print(f"  │  ⚠ no valid connections — trying port fixer …",
              flush=True)
        port_defs  = collect_port_defs(sysml)
        port_prompt = build_port_fix_prompt(
            directory, port_defs, failed_payload
        )
        try:
            port_raw = self.llm.chat(
                port_prompt, system_prompt=_PORT_FIX_SYSTEM
            )
        except Exception as exc:
            print(f"  │  ✗ LLM error in port fixer: {exc} — stopping",
                  flush=True)
            return sysml, None, "stop"

        port_items = extract_port_additions(port_raw)
        port_val   = validate_port_additions(
            port_items, directory, port_defs
        )
        for item in port_val.accepted:
            print(f"  │    + {item.part_def}: {item.to_sysml()}",
                  flush=True)
        for raw_line, reason in port_val.rejected:
            print(f"  │    ✗ port rejected: {raw_line}  — {reason}",
                  flush=True)

        if not port_val.accepted:
            print(f"  │  ⚠ no valid ports to add — stopping",
                  flush=True)
            return sysml, None, "stop"

        port_merge = merge_port_additions(sysml, port_val.accepted)
        sysml = port_merge.merged_text
        print(
            f"  │  ✓ added {port_merge.n_added} port(s): "
            f"{', '.join(port_merge.added_descriptions)}",
            flush=True,
        )

        # Rebuild directory with the new ports and retry connectivity.
        directory = build_port_directory(sysml)
        existing  = parse_connects(sysml)
        try:
            raw2 = self.llm.chat(
                build_connectivity_prompt(
                    directory, existing, failed_payload,
                    isolated_parts=isolated_parts,
                ),
                system_prompt=_CONNECTIVITY_FIX_SYSTEM,
            )
        except Exception as exc:
            print(f"  │  ✗ LLM error in post-port connect: {exc}",
                  flush=True)
            # Persist the port additions; let next pass try connects.
            self._sync_model_text(current, sysml)
            return sysml, None, "retry"

        cand_lines2 = extract_connect_lines(raw2)
        validation2 = validate_connects(cand_lines2, directory, existing)
        for stmt in validation2.accepted:
            print(f"  │    + {stmt.to_sysml()}", flush=True)
        for line, reason in validation2.rejected:
            print(f"  │    ✗ rejected: {line}  — {reason}", flush=True)

        if not validation2.accepted:
            print(
                f"  │  ⚠ no valid connections after port fix"
                f" — persisting ports for next pass",
                flush=True,
            )
            self._sync_model_text(current, sysml)
            return sysml, None, "retry"

        return sysml, merge_connects(sysml, validation2.accepted), "merged"

    def _finalize_sim_loop(self, current: SysMLModel, max_iters: int) -> SysMLModel:
        """Exit path of the simulation inner loop: re-simulate once and, when
        scenarios are still unreachable, attach a [SIM-WARNING] summary to the
        model metadata so downstream phases (and the run report) surface it."""
        final_sysml = get_sysml_text(current)
        final_sim = self._run_simulation(final_sysml, current.name)
        remaining = final_sim.failed_scenarios()

        if remaining:
            warning_lines = [
                f"[SIM-WARNING] {len(remaining)} scenario(s) still unreachable "
                f"after {max_iters} simulation refinement pass(es):"
            ]
            for r in remaining:
                tgts = ", ".join(r.unreachable_targets) or "?"
                warning_lines.append(
                    f"  • {r.scenario_name}: '{tgts}' unreachable"
                )
            warning_text = "\n".join(warning_lines)

            # Attach warning to model metadata
            if not hasattr(current, "metadata") or current.metadata is None:
                current.metadata = {}
            current.metadata["sim_warnings"] = warning_text

            print(f"  └─ ⚠  Simulation warnings attached to model:", flush=True)
            for line in warning_lines:
                print(f"       {line}")
        else:
            print(f"  └─ Simulation fully resolved ✓", flush=True)

        print(f"  {'─'*62}", flush=True)
        return current

    @staticmethod
    def _simulation_quality_key(sim_result: SimulationResult) -> tuple:
        """Lexicographic evidence key for transactional connectivity edits."""
        behavioral = getattr(sim_result, "behavioral_result", None)
        behavioral_passed = (
            len(behavioral.passed_scenarios())
            if behavioral is not None
            and callable(getattr(behavioral, "passed_scenarios", None))
            else 0
        )
        return (
            len(sim_result.passed_scenarios()),
            float(getattr(sim_result, "reachability_score", 0.0) or 0.0),
            -len(getattr(sim_result, "isolated_parts", ()) or ()),
            behavioral_passed,
        )

    def _record_rejected_connectivity_edit(
        self,
        model: SysMLModel,
        *,
        source: str,
        before: SimulationResult,
        after: SimulationResult,
    ) -> None:
        metadata = getattr(model, "metadata", None)
        if metadata is None:
            model.metadata = {}
            metadata = model.metadata
        metadata.setdefault("rejected_connectivity_repairs", []).append({
            "source": source,
            "reason": "no_simulation_quality_improvement",
            "before": {
                "passed": len(before.passed_scenarios()),
                "total": len(before.scenario_results),
                "reachability": before.reachability_score,
            },
            "after": {
                "passed": len(after.passed_scenarios()),
                "total": len(after.scenario_results),
                "reachability": after.reachability_score,
            },
        })

    def _sim_refinement_loop(
        self,
        model: SysMLModel,
        requirements: List[str],
        max_iters: int = 3,
    ) -> SysMLModel:
        """
        After the main LLM refinement is accepted, run simulation on the
        candidate and attempt targeted connectivity fixes.

        Loop:
          1. Run simulation → collect failed scenarios
          2. If all pass → return immediately
          3. Build a sim-only feedback prompt → call DesignAgent for a fix
          4. If fix accepted (no regression) → update candidate and continue
          5. After max_iters with persistent failures → attach [SIM-WARNING]
             to model metadata and return with warning printed

        Returns the best candidate (may still have sim warnings attached).
        """
        current = model
        persistent_sim_issues: Dict[str, int] = {}   # issue text → occurrence count

        print(f"\n  {'─'*62}", flush=True)
        print(f"  ▶  Simulation inner loop  (max {max_iters} pass{'es' if max_iters>1 else ''})")

        # ── Behavioral transition fix (run once before connectivity loop) ──
        # When a state machine got stuck mid-chain (Layer-3 mode machine
        # incomplete), repair transition source/target via the surgical
        # transition fixer.  This complements connectivity_fixer which only
        # handles port-level reachability, not state-machine semantics.
        current = self._fix_stuck_transitions(current)

        raw_plan = (
            getattr(current, "metadata", None) or {}
        ).get("whole_model_generation_plan")
        if isinstance(raw_plan, Mapping):
            from ..prototyping.generation_plan import (
                ModelGenerationPlan,
                apply_generation_plan,
            )
            from ..prototyping.structural_obligations import (
                validate_structural_obligations,
            )

            plan = ModelGenerationPlan.from_dict(raw_plan)
            before_text = get_sysml_text(current)
            before_report = validate_structural_obligations(
                before_text,
                plan.structural_obligations,
                model_name=current.name,
            )
            repaired_text, conformance = apply_generation_plan(
                before_text,
                plan,
            )
            after_report = validate_structural_obligations(
                repaired_text,
                plan.structural_obligations,
                model_name=current.name,
            )
            before_passed = {
                item["obligation_id"]
                for item in before_report["results"]
                if item["status"] == "PASS"
            }
            after_passed = {
                item["obligation_id"]
                for item in after_report["results"]
                if item["status"] == "PASS"
            }
            fixed_set_preserved = before_passed <= after_passed
            syntax_ok = not check_syntax(
                repaired_text,
                fail_closed=True,
                filter_stdlib_diagnostics=True,
            ).has_errors
            behavior_preserved = True
            if repaired_text != before_text and syntax_ok:
                from .verification_audit import behavioral_result_regressed

                behavior_preserved = not behavioral_result_regressed(
                    self._run_simulation(before_text, current.name),
                    self._run_simulation(repaired_text, current.name),
                )
            if (
                repaired_text != before_text
                and fixed_set_preserved
                and syntax_ok
                and behavior_preserved
            ):
                self._sync_model_text(current, repaired_text)
                print(
                    "  │  ✓ restored plan-authorized ports/connections; "
                    f"fixed obligations remain {len(after_passed)}/"
                    f"{len(after_report['results'])}",
                    flush=True,
                )
            elif repaired_text != before_text:
                conformance["status"] = "FAIL"
                conformance.setdefault("issues", []).append(
                    "plan-authorized repair rejected: syntax, behavior, or "
                    "frozen structural obligation regression"
                )
                after_report = before_report

            if getattr(current, "metadata", None) is None:
                current.metadata = {}
            current.metadata["generation_plan_conformance"] = conformance
            current.metadata["structural_obligation_report"] = after_report
            if (
                conformance.get("status") == "PASS"
                and after_report.get("status") == "PASS"
            ):
                print(
                    "  └─ Frozen requirement structural obligations fully "
                    "resolved ✓; heuristic role scenarios remain advisory",
                    flush=True,
                )
                return current

            current.metadata["structural_repair_blocked"] = {
                "reason": (
                    "no plan-authorized structural repair can satisfy the "
                    "terminal plan"
                ),
                "generation_plan_conformance": conformance,
                "structural_obligation_report": after_report,
            }
            print(
                "  └─ ⚠ structural repair blocked: remaining issue requires "
                "a validated plan revision; unrestricted port/connect "
                "generation was not invoked",
                flush=True,
            )
            return current

        for sim_iter in range(max_iters):
            sysml = get_sysml_text(current)
            baseline_sim = self._run_simulation(sysml, current.name)
            # Deterministic port-DIRECTION fix BEFORE simulating (no LLM): widen direction-blocking
            # ports so existing connects are traversable as written — resolves 'connected but signal
            # direction may be wrong' cheaply, so only genuinely-missing connections reach the LLM
            # step below (avoids escalating direction errors to slow LLM refinement). Idempotent.
            from ..simulation.direction_fixer import fix_signal_directions
            direction_candidate, _n_dir, _dir_names = fix_signal_directions(sysml)
            if _n_dir:
                direction_sim = self._run_simulation(
                    direction_candidate, current.name
                )
                if (
                    self._simulation_quality_key(direction_sim)
                    > self._simulation_quality_key(baseline_sim)
                ):
                    sysml = direction_candidate
                    baseline_sim = direction_sim
                    print(
                        f"  │  ⟳  direction fix (deterministic): widened "
                        f"{_n_dir} port(s) → inout: {', '.join(_dir_names)}; "
                        "simulation improved",
                        flush=True,
                    )
                    self._sync_model_text(current, sysml)
                else:
                    self._record_rejected_connectivity_edit(
                        current,
                        source="DETERMINISTIC_DIRECTION_WIDENING",
                        before=baseline_sim,
                        after=direction_sim,
                    )
                    print(
                        "  │  ↩ rejected deterministic direction widening: "
                        "simulation did not improve",
                        flush=True,
                    )
            sim_result = baseline_sim
            failed = sim_result.failed_scenarios()

            # ── Print this pass's result ───────────────────────────────
            passed  = len(sim_result.passed_scenarios())
            total   = len(sim_result.scenario_results)
            status  = "✓ all pass" if not failed else f"✗ {len(failed)} failing"
            print(
                f"  │  Pass {sim_iter+1}/{max_iters}  sim={sim_result.reachability_score:.3f} "
                f"[{passed}/{total}]  {status}",
                flush=True,
            )

            if not failed:
                print(f"  └─ Simulation fully resolved ✓", flush=True)
                return current

            # ── Persistent tracking ────────────────────────────────────
            sim_issues = self._format_sim_issues(sim_result, requirements=requirements)
            for iss in sim_issues:
                persistent_sim_issues[iss] = persistent_sim_issues.get(iss, 0) + 1
            persistent = [
                iss for iss, cnt in persistent_sim_issues.items() if cnt > 1
            ]

            # ── Print isolated parts (highest priority) ────────────────
            if sim_result.isolated_parts:
                print(f"  │  ⚠ ISOLATED PARTS ({len(sim_result.isolated_parts)}) — "
                      f"no connect statements: "
                      f"{', '.join(sim_result.isolated_parts)}")

            # ── Print failed scenarios ─────────────────────────────────
            for r in failed:
                tgts = ", ".join(r.unreachable_targets) or "?"
                p_tag = "  [PERSISTENT]" if any(
                    r.scenario_name in iss for iss in persistent
                ) else ""
                print(f"  │    ✗ {r.scenario_name} → can't reach: {tgts}{p_tag}")
                for w in r.warnings:
                    print(f"  │      ⚠ {w}")

            # ── Deterministic missing-connect fix (no LLM) ─────────────
            # For each failed scenario the design is often just missing a same-name/type out→in
            # connect (e.g. payloadStatus payload→flightController). Add those deterministically
            # (validated: type/direction/single-driver) BEFORE spending an LLM call. Resolves the
            # common churn cheaply; only genuinely-ambiguous gaps reach the LLM below.
            from ..simulation.direction_fixer import fix_missing_connects
            _fp = [{"src": _scenario_src_instance(r.scenario_name),
                    "tgts": list(r.unreachable_targets)} for r in failed]
            _mc_text, _n_mc, _mc_lines = fix_missing_connects(sysml, _fp)
            if _n_mc:
                print(f"  │  ⟳  connect fix (deterministic): added {_n_mc} — "
                      f"{'; '.join(_mc_lines)}", flush=True)
                candidate_sim = self._run_simulation(_mc_text, current.name)
                if (
                    self._simulation_quality_key(candidate_sim)
                    > self._simulation_quality_key(sim_result)
                ):
                    sysml = _mc_text
                    self._sync_model_text(current, sysml)
                    sim_result = candidate_sim
                    failed = sim_result.failed_scenarios()
                else:
                    self._record_rejected_connectivity_edit(
                        current,
                        source="DETERMINISTIC_MISSING_CONNECT",
                        before=sim_result,
                        after=candidate_sim,
                    )
                    print(
                        "  │  ↩ rejected deterministic connect edit: "
                        "simulation did not improve",
                        flush=True,
                    )
                if not failed and not sim_result.isolated_parts:
                    continue                      # resolved deterministically → skip the LLM step

            if sim_iter == max_iters - 1:
                # Last pass — no more LLM calls, attach warning and exit
                break

            # ── Surgical connectivity fix ──────────────────────────────
            # Feed ONLY a compact assembly context (port directory + existing
            # connects + failed scenarios) instead of the whole model.  The
            # LLM may return only `connect` lines; each is then validated
            # programmatically (no fabricated ports, correct direction, type
            # match, single-driver in-ports) before merging.
            print(f"  │  ⟳  Fixing connectivity (surgical) …", flush=True)

            directory = build_port_directory(sysml)
            existing  = parse_connects(sysml)
            failed_payload = [
                {
                    "name": r.scenario_name,
                    "src":  _scenario_src_instance(r.scenario_name),
                    "tgts": list(r.unreachable_targets),
                }
                for r in failed
            ]
            conn_prompt = build_connectivity_prompt(
                directory, existing, failed_payload,
                isolated_parts=sim_result.isolated_parts,
            )

            try:
                raw = self.llm.chat(conn_prompt, system_prompt=_CONNECTIVITY_FIX_SYSTEM)
            except Exception as exc:
                print(f"  │  ✗ LLM error: {exc} — keeping candidate", flush=True)
                continue

            cand_lines = extract_connect_lines(raw)
            validation = validate_connects(cand_lines, directory, existing)

            for stmt in validation.accepted:
                print(f"  │    + {stmt.to_sysml()}", flush=True)
            for line, reason in validation.rejected:
                print(f"  │    ✗ rejected: {line}  — {reason}", flush=True)

            if not validation.accepted:
                pre_fallback_text = sysml
                sysml, merge, outcome = self._port_fixer_fallback(
                    sysml, directory, failed_payload,
                    sim_result.isolated_parts, current,
                )
                if outcome == "stop":
                    break
                if outcome == "retry":
                    candidate_sim = self._run_simulation(sysml, current.name)
                    if (
                        self._simulation_quality_key(candidate_sim)
                        > self._simulation_quality_key(sim_result)
                    ):
                        self._sync_model_text(current, sysml)
                    else:
                        self._sync_model_text(current, pre_fallback_text)
                        self._record_rejected_connectivity_edit(
                            current,
                            source="PORT_ONLY_FALLBACK",
                            before=sim_result,
                            after=candidate_sim,
                        )
                    continue
            else:
                merge = merge_connects(sysml, validation.accepted)

            # Type/direction validity is necessary but not sufficient. Commit the
            # edit only when it improves actual reachability/behavior evidence.
            candidate_sim = self._run_simulation(merge.merged_text, current.name)
            if (
                self._simulation_quality_key(candidate_sim)
                > self._simulation_quality_key(sim_result)
            ):
                self._sync_model_text(current, merge.merged_text)
                print(
                    f"  │  ✓ added {merge.n_added} validated connection(s); "
                    "simulation improved",
                    flush=True,
                )
            else:
                self._record_rejected_connectivity_edit(
                    current,
                    source="LLM_CONNECTIVITY_REPAIR",
                    before=sim_result,
                    after=candidate_sim,
                )
                print(
                    f"  │  ↩ rejected {merge.n_added} validated connection(s): "
                    "simulation did not improve",
                    flush=True,
                )

        # ── Exited loop with persistent sim failures ───────────────────
        return self._finalize_sim_loop(current, max_iters)

    # ------------------------------------------------------------------
    # Connect audit step
    # ------------------------------------------------------------------

    def _connect_audit_step(
        self,
        sysml_text: str,
        model: SysMLModel,
    ) -> Tuple[str, SysMLModel]:
        """
        Programmatically audit every existing ``connect`` statement using
        the same five rules as connectivity_fixer.  Invalid connects are
        removed from the text so downstream simulation and
        connectivity_fixer see a clean model and can propose correct
        replacements.

        Runs in < 1 ms (pure regex + dict lookups, no LLM call).
        """
        result = audit_connects(sysml_text)

        if not result.has_violations:
            return sysml_text, model

        W = 62
        print(f"\n  ┌─ [CONNECT-AUDIT]  {result.n_removed} invalid connect(s) removed",
              flush=True)
        for v in result.violations:
            print(f"  │  ✗ {v.summary()}", flush=True)
        print(f"  └─ cleaned text passed to simulation", flush=True)

        # Persist cleaned text into model metadata
        self._sync_model_text(model, result.cleaned_text)

        return result.cleaned_text, model

    def _fix_stuck_transitions(
        self, model: SysMLModel, max_rounds: int = 3
    ) -> SysMLModel:
        """
        Iterative surgical mode-machine repair.

        Each round:
          1. Run behavioral simulation to find "stuck at X" violations.
          2. For every stuck state machine, ask the LLM (narrow context only)
             to correct the wrong transition source(s).
          3. Validate and merge accepted fixes; re-run simulation.
          4. Stop when all state machines pass, no more stuck machines are
             found, no LLM fix was accepted (dead end), or max_rounds reached.

        Supports both guard-based (enum_eq) and accept-triggered mode machines.
        """
        sysml = get_sysml_text(model)

        _STUCK_RE = re.compile(
            r"only traversed (\d+)/(\d+) (?:accept )?transitions"
            r" — stuck at '([^']+)'"
        )

        def _collect_stuck(br) -> List[Tuple[str, str, int, int]]:
            """Return (sm_name, stuck_state, fired, expected) for every stuck SM."""
            out: List[Tuple[str, str, int, int]] = []
            if br is None:
                return out
            for sr in br.scenario_results:
                if sr.passed:
                    continue
                for v in sr.violations:
                    m = _STUCK_RE.search(v)
                    if m:
                        out.append((sr.state_machine, m.group(3),
                                    int(m.group(1)), int(m.group(2))))
                        break
            return out

        # Initial simulation
        sim_result = self._run_simulation(sysml, model.name)
        br = sim_result.behavioral_result
        if br is None or br.extracted_sm_count == 0:
            return model

        stuck = _collect_stuck(br)
        if not stuck:
            return model

        for rnd in range(1, max_rounds + 1):
            # Snapshot total fired count BEFORE this round's repairs so we can
            # detect genuine progress after re-simulation.
            prev_fired_total = sum(f for _, _, f, _ in stuck)

            print(
                f"  │  ⟳  Fixing stuck mode machine(s) "
                f"— round {rnd}/{max_rounds} "
                f"({len(stuck)} stuck) …",
                flush=True,
            )

            any_accepted = False

            for sm_name, stuck_state, fired, expected in stuck:
                info = build_state_machine_summary(sysml, sm_name)
                if info is None:
                    print(f"  │    ✗ '{sm_name}' summary unavailable", flush=True)
                    continue

                prompt = build_transition_prompt(info, stuck_state, fired, expected)
                try:
                    raw = self.llm.chat(prompt, system_prompt=_TRANSITION_FIX_SYSTEM)
                except Exception as exc:
                    print(f"  │    ✗ LLM error: {exc}", flush=True)
                    continue

                lines = extract_transition_lines(raw)
                validation = validate_transitions(lines, info)
                for stmt in validation.accepted:
                    print(
                        f"  │    + {stmt.name}: first {stmt.source}"
                        f" → then {stmt.target}",
                        flush=True,
                    )
                for line, reason in validation.rejected:
                    print(
                        f"  │    ✗ rejected: {' '.join(line.split())[:60]}"
                        f"  — {reason}",
                        flush=True,
                    )

                if not validation.accepted:
                    continue

                merge = merge_transitions(sysml, validation.accepted)
                sysml = merge.merged_text
                self._sync_model_text(model, sysml)
                print(
                    f"  │  ✓ repaired {merge.n_replaced} transition(s)"
                    f" in '{sm_name}'",
                    flush=True,
                )
                any_accepted = True

            if not any_accepted:
                print(f"  │  ⚠ no fix accepted — stopping transition repair",
                      flush=True)
                break

            # Re-simulate to check progress
            sim_result = self._run_simulation(sysml, model.name)
            br = sim_result.behavioral_result
            stuck = _collect_stuck(br)

            if not stuck:
                print(f"  │  ✓ all mode machines resolved after round {rnd}",
                      flush=True)
                break

            # Compare against the snapshot taken before this round's repairs.
            # If total fired count didn't increase, the fix made no progress.
            new_fired_total = sum(f for _, _, f, _ in stuck)
            if new_fired_total <= prev_fired_total:
                print(
                    f"  │  ⚠ no progress in round {rnd}"
                    f" ({prev_fired_total} → {new_fired_total} fired) — stopping",
                    flush=True,
                )
                break

        return model

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_model_text(model: SysMLModel, text: str) -> None:
        """Write *text* to model.metadata["last_sysml_text"] so downstream
        phases (evaluation, simulation, artifacts) read the updated source."""
        meta = getattr(model, "metadata", None)
        if meta is None:
            object.__setattr__(model, "metadata", {})
            meta = model.metadata
        meta["last_sysml_text"] = text

    def _tier0_deterministic_fixes(
        self,
        working_sysml: str,
        working_model: SysMLModel,
        latest_result: SyntaxCheckResult,
    ) -> Tuple[str, SyntaxCheckResult, List[Dict], bool]:
        """Tier 0 of the syntax gate: deterministic, LLM-free fixes.

        Applies, in order: `readonly` stripping, reserved-keyword item-name
        quoting, Levenshtein distance-1 typo correction, and missing guard
        attribute injection.  Returns ``(sysml, result, lev_hints, resolved)``
        where ``resolved`` means all errors are gone and the LLM loop can be
        skipped; ``lev_hints`` carries distance-2 suggestions for the Tier 1
        LLM prompt.
        """
        lev_hints: List[Dict] = []   # distance-2 suggestions for the LLM prompt

        # ── strip `readonly` before attribute ────────────────────────────────
        # syside rejects `readonly attribute X : ...`; idiomatic SysML v2 uses
        # plain `attribute`.  Strip deterministically — no LLM needed.
        if latest_result.parser_errors:
            stripped = _strip_readonly_keyword(working_sysml)
            if stripped != working_sysml:
                re_checked = check_syntax(stripped)
                if re_checked.total_errors() < latest_result.total_errors():
                    n_fixed = latest_result.total_errors() - re_checked.total_errors()
                    working_sysml = stripped
                    print(
                        f"\n  ┌─ [RO-FIX]  {n_fixed} `readonly` modifier(s) stripped"
                        f" — no LLM needed",
                        flush=True,
                    )
                    self._sync_model_text(working_model, working_sysml)
                    latest_result = re_checked
                    if not latest_result.has_errors:
                        print(f"  └─ [RO-FIX]  ✓ all errors resolved", flush=True)
                        return working_sysml, latest_result, lev_hints, True
                    print(
                        f"  └─ [RO-FIX]  {latest_result.total_errors()} error(s) remain"
                        f" — continuing",
                        flush=True,
                    )

        # ── SysML keyword quoting ────────────────────────────────────────────
        # `inout/in/out item <keyword> :` where <keyword> is a SysML reserved
        # word causes a parser error ("Unexpected 'item'").  Fix deterministically
        # by quoting the offending name — no LLM needed.
        if latest_result.parser_errors:
            working_sysml = _fix_keyword_item_names(working_sysml)
            re_checked = check_syntax(working_sysml)
            if re_checked.total_errors() < latest_result.total_errors():
                n_fixed = latest_result.total_errors() - re_checked.total_errors()
                print(
                    f"\n  ┌─ [KW-FIX]  {n_fixed} reserved-keyword item name(s) quoted"
                    f" — no LLM needed",
                    flush=True,
                )
                self._sync_model_text(working_model, working_sysml)
                latest_result = re_checked
                if not latest_result.has_errors:
                    print(f"  └─ [KW-FIX]  ✓ all errors resolved", flush=True)
                    return working_sysml, latest_result, lev_hints, True
                print(
                    f"  └─ [KW-FIX]  {latest_result.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )

        # ── Levenshtein quick-fix ────────────────────────────────────────────
        if latest_result.sema_errors:
            lev = try_fix_sema_errors(working_sysml, latest_result.sema_errors)

            if lev.auto_fixed:
                n_fixed = len(lev.auto_fixed)
                print(
                    f"\n  ┌─ [LEV-FIX]  {n_fixed} typo(s) auto-corrected"
                    f" (distance=1, no LLM needed):",
                    flush=True,
                )
                for e in lev.auto_fixed:
                    import re as _re
                    _wrong = _re.search(r"named '([^']+)'", e['message'])
                    wrong_name = _wrong.group(1) if _wrong else "?"
                    print(
                        f"  │  L{e['line']:>3}: '{wrong_name}'"
                        f"  →  '{e['_suggestion']}'",
                        flush=True,
                    )

                # Re-check after applying Levenshtein fixes
                working_sysml = lev.fixed_text
                re_checked    = check_syntax(working_sysml)

                # Update model metadata so downstream reads the fixed text
                self._sync_model_text(working_model, working_sysml)

                if not re_checked.has_errors:
                    print(
                        f"  └─ [LEV-FIX]  ✓ all errors resolved"
                        f" — LLM fix loop skipped",
                        flush=True,
                    )
                    return working_sysml, re_checked, lev_hints, True

                print(
                    f"  └─ [LEV-FIX]  {re_checked.total_errors()} error(s) remain"
                    f" — continuing to LLM fix loop",
                    flush=True,
                )
                latest_result = re_checked

            # Collect distance-2 hints for the LLM prompt
            lev_hints = lev.hints

        # ── undeclared guard attribute injection ─────────────────────────────
        # sema error "No Feature named 'X' found" where X appears in a state
        # machine guard → inject `attribute X : Real/Boolean = <default>;`
        # into the owner part def.  No LLM needed — purely programmatic.
        if latest_result.sema_errors:
            working_sysml, n_injected = _inject_missing_guard_attrs(
                working_sysml, latest_result.sema_errors
            )
            if n_injected:
                re_checked = check_syntax(working_sysml)
                self._sync_model_text(working_model, working_sysml)
                print(
                    f"\n  ┌─ [ATTR-INJ]  {n_injected} missing guard attribute(s)"
                    f" injected — no LLM needed",
                    flush=True,
                )
                if not re_checked.has_errors:
                    print(f"  └─ [ATTR-INJ]  ✓ all errors resolved", flush=True)
                    return working_sysml, re_checked, lev_hints, True
                print(
                    f"  └─ [ATTR-INJ]  {re_checked.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )
                latest_result = re_checked

        return working_sysml, latest_result, lev_hints, False

    def _tier1_fix_chunk(
        self,
        working_sysml: str,
        chunk,
        latest_result: SyntaxCheckResult,
        lev_hints: List[Dict],
        attempt: int,
        model_total_lines: int,
    ) -> Tuple[str, bool]:
        """One surgical LLM fix for a single error block (Tier 1).

        Builds the minimal-context prompt (attaching distance-2 Levenshtein
        hints on the first attempt only), calls the LLM, and merges the
        returned block.  Returns ``(sysml, merged)``; on LLM error or merge
        rejection the text is returned unchanged.
        """
        # 构建 prompt；首次调用时把 d=2 Lev 建议附到所属块
        prompt = build_fix_prompt(chunk)
        if lev_hints and attempt == 0:
            chunk_hints = [
                h for h in lev_hints
                if chunk.start_line <= h.get('line', 0) <= chunk.end_line
            ]
            hint_block = format_hints_for_llm(chunk_hints)
            if hint_block:
                prompt += "\n\n" + hint_block

        chunk_lines   = chunk.end_line - chunk.start_line + 1
        prompt_lines  = len(prompt.splitlines())
        print(
            f"  ║\n  ║  ┌─ block '{chunk.block_name}'"
            f"  lines {chunk.start_line}–{chunk.end_line}"
            f"  ({chunk_lines} lines extracted / {model_total_lines} total)",
            flush=True,
        )
        for e in chunk.errors:
            tag = "parser" if e in latest_result.parser_errors else "sema"
            print(
                f"  ║  │  [{tag}] L{e['line']:>3}: {e['message']}",
                flush=True,
            )
        print(
            f"  ║  │  prompt: {prompt_lines} lines"
            f"  (compressed {model_total_lines}→{prompt_lines} lines,"
            f" {100 * prompt_lines // max(model_total_lines, 1)}% of model)",
            flush=True,
        )
        print(f"  ║  │  ↳ calling LLM …", flush=True)

        t0 = time.perf_counter()
        try:
            raw_fix = self.llm.chat(
                prompt, system_prompt=_SURGICAL_FIX_SYSTEM
            )
        except Exception as exc:
            print(f"  ║  │  ✗ LLM error: {exc}", flush=True)
            return working_sysml, False
        elapsed = time.perf_counter() - t0

        # 显示 LLM 返回的前几行（去除围栏后）
        preview_lines = strip_code_fences(raw_fix).splitlines()
        n_resp = len(preview_lines)
        print(
            f"  ║  │  ↳ response: {n_resp} lines  ⏱ {elapsed:.1f}s",
            flush=True,
        )
        for pl in preview_lines[:4]:
            print(f"  ║  │     {pl}", flush=True)
        if n_resp > 4:
            print(f"  ║  │     … ({n_resp - 4} more lines)", flush=True)

        merge = merge_fixed_chunk(working_sysml, chunk, raw_fix)
        if merge.success:
            status = f"Δlines={merge.line_delta:+d}"
            if merge.warning:
                print(f"  ║  └─ ⚠  merged  {status}  {merge.warning}", flush=True)
            else:
                print(f"  ║  └─ ✓  merged  {status}", flush=True)
            return merge.merged_text, True
        print(f"  ║  └─ ✗  merge rejected — {merge.warning}", flush=True)
        return working_sysml, False

    def _syntax_gate(
        self,
        sysml_text: str,
        current_model: SysMLModel,
        requirements: List[str],
        max_attempts: int = 3,
    ) -> Tuple[str, Optional[SysMLModel], SyntaxCheckResult]:
        """
        Syntax pre-check gate — two-tier fix strategy.

        Tier 0 (Levenshtein, < 1 ms)
            Applied first when sema errors exist.  Single-edit (distance=1)
            typos in feature / type / instance names are corrected directly
            in the text without calling the LLM.  Distance-2 near-misses are
            collected as hints and injected into the LLM prompt (Tier 1).
            If Tier 0 resolves ALL errors, the LLM loop is skipped entirely.

        Tier 1 (LLM, up to max_attempts rounds)
            Runs only when Tier 0 leaves errors unresolved (parser errors,
            unresolvable sema errors, etc.).  Each round re-checks with syside
            and stops as soon as the model is error-free.

        Returns:
            (final_sysml, fixed_model_or_None, syntax_result)
            fixed_model_or_None is set only when the text was actually changed.
        """
        result = check_syntax(sysml_text)

        if not result.has_errors:
            print(f"  ✓ [SYNTAX]  no errors  (syside: 0 parser, 0 sema)", flush=True)
            return sysml_text, None, result

        working_sysml = sysml_text
        working_model = current_model
        latest_result = result
        # ── Tier 0: deterministic fixes (RO-FIX / KW-FIX / LEV-FIX / ATTR-INJ) ─
        working_sysml, latest_result, lev_hints, resolved = (
            self._tier0_deterministic_fixes(working_sysml, working_model, latest_result)
        )
        if resolved:
            return working_sysml, working_model, latest_result

        # ── Tier 1: 外科式 LLM 修复 ──────────────────────────────────────────
        # 只传错误块（~15 行）+ 精简声明摘要，而不是整个模型（~200 行）。
        # 多个错误块逆序处理，保证行号不因前面的合并而漂移。
        model_total_lines = len(working_sysml.splitlines())

        for attempt in range(max_attempts):
            all_errors = latest_result.parser_errors + latest_result.sema_errors
            n = len(all_errors)

            print(
                f"\n  ╔═ [SYNTAX-GATE] attempt {attempt + 1}/{max_attempts}"
                f" ─── {n} error(s)  ({latest_result.short_summary()})",
                flush=True,
            )
            for e in all_errors[:8]:
                tag = "parser" if e in latest_result.parser_errors else "sema"
                print(f"  ║  [{tag}] L{e['line']:>3}: {e['message']}", flush=True)
            if n > 8:
                print(f"  ║  … and {n - 8} more", flush=True)

            if attempt == max_attempts - 1:
                print(
                    f"  ╚═ ⚠  errors persist after {max_attempts} attempt(s)"
                    f" — proceeding with degraded syntactic_validity score",
                    flush=True,
                )
                changed = working_sysml != sysml_text or working_model is not current_model
                return working_sysml, (working_model if changed else None), latest_result

            # 按语法块分组，提取最小错误上下文
            chunks = extract_error_context(working_sysml, all_errors)
            print(
                f"  ║\n  ║  ▸ {n} error(s) → {len(chunks)} block(s)"
                f"  [model: {model_total_lines} lines total]",
                flush=True,
            )

            # 逆序遍历，晚出现的块先修，避免行号漂移
            for chunk in sorted(chunks, key=lambda c: c.start_line, reverse=True):
                working_sysml, merged = self._tier1_fix_chunk(
                    working_sysml, chunk, latest_result, lev_hints,
                    attempt, model_total_lines,
                )
                if merged:
                    model_total_lines = len(working_sysml.splitlines())

            lev_hints = []   # d=2 建议只在首次 LLM 调用时传递

            # 更新 model metadata，让后续流程读到最新文本
            self._sync_model_text(working_model, working_sysml)

            print(f"  ║\n  ║  re-checking syntax …", flush=True)
            latest_result = check_syntax(working_sysml)

            if not latest_result.has_errors:
                print(
                    f"  ╚═ ✓  all errors resolved"
                    f" after {attempt + 1} fix attempt(s)",
                    flush=True,
                )
                return working_sysml, working_model, latest_result

            print(
                f"  ╚═ {latest_result.total_errors()} error(s) remain"
                f" after attempt {attempt + 1}"
                f" — retrying …",
                flush=True,
            )

        # Should not reach here, but safety fallback
        return working_sysml, working_model, latest_result

    def _run_simulation(self, sysml_text: str, model_name: str) -> SimulationResult:
        """Run behavioral reachability simulation on raw SysML text."""
        try:
            return self.sim_validator.validate(sysml_text, model_name=model_name)
        except Exception as e:
            from ..simulation.validator import SimulationResult
            r = SimulationResult(model_name=model_name)
            r.issues.append(f"Simulation error: {e}")
            return r

    def _format_sim_issues(self, sim_result: SimulationResult,
                            requirements: Optional[List[str]] = None) -> List[str]:
        """Convert failed simulation scenarios into LLM-readable issue strings."""
        issues: List[str] = []

        # Isolated parts are the most actionable issue — report first
        if sim_result.isolated_parts:
            issues.append(
                f"ISOLATED PARTS — the following parts have zero connect statements "
                f"and are architecturally dead (no signal in or out): "
                f"{', '.join(sim_result.isolated_parts)}. "
                f"For each, add `connect <part>.<outPort> to <target>.<inPort>;` "
                f"in the system assembly section."
            )

        for r in sim_result.failed_scenarios():
            tgt = ", ".join(r.unreachable_targets) if r.unreachable_targets else "unknown"
            entry = r.scenario_name.split("_to_")[0] if "_to_" in r.scenario_name else "?"
            # Annotate if the source itself is isolated (helps LLM prioritise)
            isolated_tag = (
                " [entry part is isolated — no connections at all]"
                if entry in sim_result.isolated_parts else ""
            )
            issues.append(
                f"Scenario '{r.scenario_name}': no signal path from '{entry}' to '{tgt}'."
                f"{isolated_tag} "
                + (r.issues[0] if r.issues else "")
            )
        for rec in sim_result.recommendations:
            if "only input ports" in rec or "Isolated" in rec:
                issues.append(rec)

        # ── Behavioral state machine violations ───────────────────────────────
        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.extracted_sm_count > 0:
            for sr in br.scenario_results:
                if not sr.passed:
                    for v in sr.violations:
                        issues.append(
                            f"STATE MACHINE '{sr.state_machine}': {v} "
                            f"Fix: verify guard thresholds in the state def match "
                            f"the corresponding SAFE requirement value."
                        )

        # ── Missing state machines (Solution B) ──────────────────────────────
        # When SAFE requirements exist but no state def blocks were extracted,
        # the LLM failed to generate fault-handling behaviour — flag it.
        safe_reqs = [r for r in (requirements or []) if "-SAFE-" in r or "SAFE" in r.upper()[:10]]
        if safe_reqs and (br is None or br.extracted_sm_count == 0):
            sample = "; ".join(safe_reqs[:3])
            issues.append(
                f"MISSING STATE MACHINES — {len(safe_reqs)} safety requirement(s) found "
                f"but no `state def` blocks were extracted from the model. "
                f"You MUST add `state def` blocks inside the relevant PartDefinition(s) "
                f"with guard-based transitions for each fault condition. "
                f"Relevant requirements: {sample}"
            )

        return issues

    def _print_final_sim(self, sim_result: SimulationResult) -> None:
        """Full simulation report printed at the end of Phase 6."""
        total  = len(sim_result.scenario_results)
        passed = len(sim_result.passed_scenarios())
        failed = sim_result.failed_scenarios()
        W = 62

        def bar(v: float, w: int = 30) -> str:
            filled = round(v * w)
            return "█" * filled + "░" * (w - filled)

        print(f"\n  ╔{'═'*W}╗")
        print(f"  ║  FINAL SIMULATION REPORT  ─  {sim_result.model_name:<29}║")
        print(f"  ╠{'═'*W}╣")
        print(f"  ║  Reachability score : {sim_result.reachability_score:.3f}  "
              f"{bar(sim_result.reachability_score, 20)}  "
              f"{passed}/{total} scenarios{' '*(4-len(str(total)))}║")
        print(f"  ║  Graph              : {sim_result.num_parts} parts · "
              f"{sim_result.num_ports} ports · "
              f"{sim_result.num_connections} connections{' '*10}║")
        if sim_result.isolated_parts:
            iso_str = f"  ║  ⚠ Isolated parts   : {', '.join(sim_result.isolated_parts)}"
            print(f"{iso_str:<{W+4}}║")
        print(f"  ╠{'═'*W}╣")

        # Per-scenario table
        for r in sim_result.scenario_results:
            icon  = "✓" if r.passed else "✗"
            tags  = "|".join(r.tags)
            name  = r.scenario_name[:32]
            path  = (" → ".join(r.path[:3]) + ("…" if len(r.path) > 3 else "")) if r.path else "—"
            path  = path[:26]
            print(f"  ║  {icon} [{tags:<15}] {name:<33}║")
            print(f"  ║      path: {path:<50}║")
            for iss in r.issues[:2]:
                print(f"  ║      ! {iss[:54]:<54}║")

        print(f"  ╠{'═'*W}╣")
        if sim_result.recommendations and \
                sim_result.recommendations[0] != \
                "All scenarios passed — model connectivity is structurally sound":
            print(f"  ║  Recommendations:{'  '*(W//2-9)}║")
            for rec in sim_result.recommendations[:3]:
                words = rec.split()
                line = ""
                for w in words:
                    if len(line) + len(w) + 1 > W - 6:
                        print(f"  ║    • {line:<{W-6}}║")
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    print(f"  ║    • {line:<{W-6}}║")
        else:
            print(f"  ║  ✓ All scenarios passed — connectivity is sound {'':>10}║")
        print(f"  ╚{'═'*W}╝", flush=True)

        # ── Behavioral simulation (state machine) detail report ───────────
        br = getattr(sim_result, "behavioral_result", None)
        if br is not None and br.extracted_sm_count > 0:
            b_passed = br.passed_count()
            b_total  = len(br.scenario_results)
            print(f"\n  ╔{'═'*W}╗")
            print(f"  ║  BEHAVIORAL SIMULATION  ─  State Machine Execution"
                  f"{' '*(W-50)}║")
            print(f"  ║  Extracted {br.extracted_sm_count} state machine(s)   "
                  f"Score: {br.sim_score:.2%}   ({b_passed}/{b_total} passed)"
                  f"{' '*(W-58+len(str(br.extracted_sm_count)))}║")
            print(f"  ╠{'═'*W}╣")
            for sr in br.scenario_results:
                icon = "✓" if sr.passed else "✗"
                name = sr.state_machine[:38]
                print(f"  ║  {icon}  {name:<58}║")
                for tl in sr.timeline:
                    line = tl.strip()[:W-6]
                    print(f"  ║      {line:<{W-4}}║")
                for v in sr.violations:
                    line = f"⚠ {v}"[:W-6]
                    print(f"  ║      {line:<{W-4}}║")
                print(f"  ║  {'·'*W}║")
            print(f"  ╚{'═'*W}╝", flush=True)

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
