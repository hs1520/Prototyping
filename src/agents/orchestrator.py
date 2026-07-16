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
from typing import Any, Dict, List, Optional, Tuple

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

        self.state = PrototypingState(
            system_name=system_name,
            system_description=system_description,
        )

        print(f"\n{'='*60}")
        print(f"[generate]  {system_name}")
        print(f"{'='*60}\n")

        # ── Phase 1: Requirements Extraction ─────────────────────────────────
        print("Phase 1: Requirements Extraction")
        print("-" * 40)
        requirements = self._extract_requirements(
            system_name, system_description, additional_requirements or []
        )
        self.state.requirements = requirements
        print(f"  ✓ Extracted {len(requirements)} requirements\n")

        # ── Phase 2: Initial Design Generation ───────────────────────────────
        print("Phase 2: Initial Design Generation")
        print("-" * 40)
        model = self._generate_initial_design(
            system_name, requirements,
            parse_strict=parse_strict,
            platform_profile=platform_profile,
        )
        self.state.current_model = model
        print(f"  ✓ Generated model with {len(model.part_definitions)} part definitions\n")

        # ── Phase 3: Iterative Refinement (no MCTS) ───────────────────────────
        print("Phase 3: Iterative Refinement")
        print("-" * 40)
        final_model, final_score, final_sim = self._iterative_refinement(
            model, requirements, dse_best_config=None
        )
        self.state.current_model = final_model
        print(f"  ✓ Final design score: {final_score:.3f}\n")

        # ── Phase 3.5: SITL-L1 refinement (only when targeting a platform) ────
        # Feed unresolved ArduPilot-parameter mappings (= model genuinely
        # missing a guard/attribute a requirement needs) back to the design
        # LLM.  Cheap & deterministic (no SITL process launch); L2 stays
        # terminal.
        if platform_profile is not None:
            final_model, final_score, final_sim = self._sitl_refinement_loop(
                final_model, requirements, final_score, final_sim, max_iters=2
            )
            self.state.current_model = final_model

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

        final_sysml = get_sysml_text(final_model)
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
            "platform_profile":   platform_profile,
            "llm_usage":          ledger.as_dict() if ledger is not None else None,
        }

    # ---------------------------------------------------------------------- #
    #  Stage 2 — Design Space Exploration on a validated model                #
    # ---------------------------------------------------------------------- #

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

        # Recommendation metadata is run-scoped. Reusing an Orchestrator must never
        # let a prior run's physical design leak into a new authority decision.
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
                        if not hasattr(final_model, "metadata") or final_model.metadata is None:
                            object.__setattr__(final_model, "metadata", {})
                        final_model.metadata["last_sysml_text"] = injected
                        print("  [realization] injected RealizationPackage into final model")
                except Exception as e:
                    print(f"  ⚠ realization model injection skipped ({e})")
        else:
            print("  ⚠ no recommended design / realization skipped")

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

        # ── Summary ───────────────────────────────────────────────────────────
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

        final_sysml = get_sysml_text(final_model)
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
            "model":              final_model,
            "model_sysml":        final_sysml,
            "model_summary":      final_model.get_summary(),
            "final_score":        final_score,
            "iterations":         self.state.iteration,
            "evaluation_history": combined_history,
            "simulation_result":  final_sim,
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
        result = self.design_agent.run(task)
        if result.success and isinstance(result.output, _SysMLModelTypes):
            model = result.output
        else:
            model = build_lite_model("", model_name=system_name)

        untraced = result.metadata.get("untraced_requirements", [])
        if untraced:
            print(f"  ⚠ {len(untraced)} requirement(s) could not be matched to any component: "
                  f"{', '.join(untraced)}")

        self.requirements_agent.create_sysml_requirements(requirements, model)
        return model

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
                if not hasattr(model, "metadata") or model.metadata is None:
                    object.__setattr__(model, "metadata", {})
                model.metadata["last_sysml_text"] = text
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
            if not hasattr(model, "metadata") or model.metadata is None:
                object.__setattr__(model, "metadata", {})
            model.metadata["last_sysml_text"] = text
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
            return verification_gap_issues(sysml_text, model_name)
        except Exception:
            return []

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
            veto_fired = any(
                str(iss).startswith("[VETO]") for iss in eval_result.issues
            )

            if rule_score >= self.quality_threshold:
                score = rule_score
                llm_overall = None
                cot_eval = None
            elif veto_fired:
                score = rule_score
                llm_overall = None
                cot_eval = None
            else:
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
                reachability_ok = not sim_result.failed_scenarios()
                sema_ok = syntax_result is None or not syntax_result.has_errors

                if behavioral_ok and reachability_ok and sema_ok:
                    # ── One bounded verification-anchor pass ──────────────
                    # Quality is met, but the static audit predicts unassigned
                    # matrix rows. Exactly ONE surgical pass scoped to those
                    # issues (gates: syntax, connects preserved, requirement-def
                    # set frozen, satisfy links may not shrink). Accepted only
                    # if gaps actually shrink AND nothing regresses (local sim +
                    # rule score re-checked, no LLM cost). Success or not, we
                    # return afterwards — anchors are advisory, never a loop.
                    if verify_gaps and self.use_surgical_refinement:
                        print(f"  ~ Quality met, but {len(verify_gaps)} requirement(s) "
                              f"would be UNASSIGNED in the verification matrix — "
                              f"one surgical anchor pass", flush=True)
                        from .surgical_refiner import attempt_surgical_refinement
                        anchored = attempt_surgical_refinement(
                            llm=self.llm,
                            model_text=get_sysml_text(current_model),
                            issues=verify_gaps,
                            verbose=self.verbose,
                        )
                        if anchored is not None:
                            anchor_model = build_lite_model(
                                anchored.merged_text, model_name=current_model.name)
                            anchor_sim = self._run_simulation(
                                anchored.merged_text, current_model.name)
                            anchor_eval = self.evaluator.evaluate(
                                config=DesignConfiguration(
                                    name="anchor_pass", parameters={}),
                                model=anchor_model,
                                dse_config=dse_best_config,
                                syntax_result=check_syntax(anchored.merged_text),
                                sim_result=anchor_sim,
                                requirements=requirements,
                            )
                            remaining = self._verification_gap_issues(
                                anchored.merged_text, current_model.name)
                            regressed = (
                                bool(anchor_sim.failed_scenarios())
                                or anchor_eval.weighted_total < rule_score - 0.05
                            )
                            if not regressed and len(remaining) < len(verify_gaps):
                                current_model = anchor_model
                                sim_result = anchor_sim
                                print(f"  ✓ Anchor pass accepted: verification gaps "
                                      f"{len(verify_gaps)} → {len(remaining)}", flush=True)
                            else:
                                print("  ⚠ Anchor pass rejected (no gap reduction or "
                                      "regression) — keeping the original model", flush=True)
                        else:
                            print("  ⚠ Anchor pass not applicable (LLM output failed "
                                  "the surgical gates)", flush=True)
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
                _sysml_after = get_sysml_text(current_model)
                sim_result = self._run_simulation(_sysml_after, current_model.name)
                last_sim_result = sim_result

                if not sim_result.failed_scenarios():
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
                        syntax_result=syntax_result,
                        sim_result=sim_result,
                        requirements=requirements,
                    )
                    score = eval_after.weighted_total
                    return current_model, score, sim_result

                # Surgical fix insufficient
                remaining = self.max_iterations - iteration - 1
                if remaining == 0:
                    print(
                        f"  ⚠ Surgical fix insufficient — no iterations remaining, "
                        f"returning best model",
                        flush=True,
                    )
                    return best_model, best_score, best_sim_result or sim_result

                sim_issues = self._format_sim_issues(sim_result, requirements=requirements)
                print(
                    f"  ⚠ Surgical fix insufficient "
                    f"({len(sim_result.failed_scenarios())} scenario(s) still failing) "
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
                print(f"\n  ⟳  Refining model …", flush=True)
                refinement_feedback = self._build_refinement_feedback(
                    eval_result,
                    cot_eval.final_answer if cot_eval else "",
                    persistent_issues=persistent,
                    mcts_constraints=mcts_constraints,
                    sim_issues=sim_issues,
                )

                # ── Surgical refinement first: the LLM returns only the blocks
                #    it changes; the merge is syntax-gated and cannot shed
                #    connects on untouched components (prevention, not the
                #    after-the-fact rejection the full rewrite needs). Falls
                #    back to the legacy whole-model rewrite on any failure.
                candidate = None
                if self.use_surgical_refinement:
                    from .surgical_refiner import attempt_surgical_refinement
                    surgical = attempt_surgical_refinement(
                        llm=self.llm,
                        model_text=current_sysml,
                        issues=eval_result.issues + eval_result.recommendations,
                        feedback=refinement_feedback,
                        verbose=self.verbose,
                    )
                    if surgical is not None:
                        print(f"  ✓ Surgical refinement: {surgical.summary()}",
                              flush=True)
                        candidate = build_lite_model(
                            surgical.merged_text, model_name=current_model.name
                        )
                    else:
                        print("  ⚠ Surgical refinement not applicable — "
                              "falling back to full rewrite", flush=True)

                if candidate is None:
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

                if candidate is not None:
                    # ── P0: Regression prevention ─────────────────────────
                    # Evaluate the candidate with the SAME inputs as rule_score
                    # (sim + syntax + mcts_config).  Omitting sim_result makes
                    # behavioral_verification fall back to 1.0 and omitting
                    # mcts_config changes the weight denominator — both bias
                    # the comparison toward accepting the candidate.
                    # check_syntax and _run_simulation are local (no LLM cost).
                    cand_sysml = get_sysml_text(candidate)
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
                    # ── Connectivity-regression guard (variation DSE) ──────
                    # The resolved variation model arrives fully wired; a full
                    # LLM rewrite tends to drop connects on the converted
                    # components, isolating them (reachability → 0).  Reject any
                    # candidate that sheds connect statements relative to the
                    # model it was refined from — connectivity must not regress.
                    connectivity_ok = True
                    if connectivity_floor:
                        cur_connects = self._count_connects(current_sysml)
                        cand_connects = self._count_connects(cand_sysml)
                        if cand_connects < cur_connects:
                            connectivity_ok = False
                            print(
                                f"  ⚠ Refinement dropped connectivity "
                                f"({cur_connects} → {cand_connects} connects) — "
                                f"rejected to preserve resolved variation wiring",
                                flush=True,
                            )
                    if connectivity_ok and candidate_eval.weighted_total >= rule_score - 0.05:
                        print(
                            f"  ✓ Refinement accepted  "
                            f"rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f} "
                            f"({delta_str})",
                            flush=True,
                        )
                        # ── Simulation inner loop ──────────────────────────
                        # Re-run simulation on the accepted candidate and
                        # attempt up to MAX_SIM_INNER_ITERS targeted fixes
                        # before handing the model back to the outer loop.
                        candidate = self._sim_refinement_loop(
                            candidate, requirements, max_iters=3
                        )
                        current_model = candidate
                    elif connectivity_ok:
                        print(
                            f"  ⚠ Refinement regression detected "
                            f"(rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f}), "
                            f"keeping current model",
                            flush=True,
                        )

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

        for sim_iter in range(max_iters):
            sysml = get_sysml_text(current)
            # Deterministic port-DIRECTION fix BEFORE simulating (no LLM): widen direction-blocking
            # ports so existing connects are traversable as written — resolves 'connected but signal
            # direction may be wrong' cheaply, so only genuinely-missing connections reach the LLM
            # step below (avoids escalating direction errors to slow LLM refinement). Idempotent.
            from ..simulation.direction_fixer import fix_signal_directions
            sysml, _n_dir, _dir_names = fix_signal_directions(sysml)
            if _n_dir:
                print(f"  │  ⟳  direction fix (deterministic): widened {_n_dir} port(s) → inout: "
                      f"{', '.join(_dir_names)}", flush=True)
                if not getattr(current, "metadata", None):
                    object.__setattr__(current, "metadata", {})
                current.metadata["last_sysml_text"] = sysml
            sim_result = self._run_simulation(sysml, current.name)
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
                sysml = _mc_text
                if not getattr(current, "metadata", None):
                    object.__setattr__(current, "metadata", {})
                current.metadata["last_sysml_text"] = sysml
                sim_result = self._run_simulation(sysml, current.name)
                failed = sim_result.failed_scenarios()
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
                # connectivity_fixer found no usable connects — fall back to
                # port_fixer: add missing port declarations on part defs so a
                # second connectivity pass can wire them up.
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
                    break

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
                    break

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
                            isolated_parts=sim_result.isolated_parts,
                        ),
                        system_prompt=_CONNECTIVITY_FIX_SYSTEM,
                    )
                except Exception as exc:
                    print(f"  │  ✗ LLM error in post-port connect: {exc}",
                          flush=True)
                    # Persist the port additions; let next pass try connects.
                    meta = getattr(current, "metadata", None)
                    if meta is None:
                        object.__setattr__(current, "metadata", {})
                        meta = current.metadata
                    meta["last_sysml_text"] = sysml
                    continue

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
                    meta = getattr(current, "metadata", None)
                    if meta is None:
                        object.__setattr__(current, "metadata", {})
                        meta = current.metadata
                    meta["last_sysml_text"] = sysml
                    continue

                merge = merge_connects(sysml, validation2.accepted)
            else:
                merge = merge_connects(sysml, validation.accepted)

            # Update the stored text so the next pass's simulation sees the fix.
            meta = getattr(current, "metadata", None)
            if meta is None:
                object.__setattr__(current, "metadata", {})
                meta = current.metadata
            meta["last_sysml_text"] = merge.merged_text
            print(f"  │  ✓ added {merge.n_added} validated connection(s)", flush=True)

        # ── Exited loop with persistent sim failures ───────────────────
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
        meta = getattr(model, "metadata", None)
        if meta is None:
            object.__setattr__(model, "metadata", {})
            meta = model.metadata
        meta["last_sysml_text"] = result.cleaned_text

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
                meta = getattr(model, "metadata", None)
                if meta is None:
                    object.__setattr__(model, "metadata", {})
                    meta = model.metadata
                meta["last_sysml_text"] = sysml
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

    @staticmethod
    def _build_sim_only_feedback(
        sim_result: SimulationResult,
        persistent_issues: Optional[List[str]] = None,
    ) -> str:
        """
        Build a targeted, simulation-only refinement prompt.
        Focuses exclusively on missing port connections — no evaluator noise.
        """
        failed = sim_result.failed_scenarios()
        lines = [
            "CONNECTIVITY FIX ONLY — do not change any other part of the model.",
            "",
            "The following operational scenarios have no directed signal path.",
            "For each, add the missing `connect <partA>.<portA> to <partB>.<portB>;`",
            "statement(s) in the system assembly section of the package.",
        ]

        # ── Isolated parts section (highest priority) ──────────────────────
        if sim_result.isolated_parts:
            lines += [
                "",
                "ISOLATED PARTS (critical — fix these first):",
                "  These parts exist in the model but have ZERO connect statements.",
                "  They cannot send or receive any signal and will fail every scenario.",
                "  For each part, add at least one outgoing and one incoming connection:",
            ]
            for p in sim_result.isolated_parts:
                lines.append(f"  • {p}: connect {p}.<outPort> to <target>.<inPort>;")

        lines += ["", "Failed scenarios:"]
        for r in failed:
            tgts = ", ".join(r.unreachable_targets) or "?"
            entry = r.scenario_name.split("_to_")[0] if "_to_" in r.scenario_name else "?"
            isolated_note = (
                "  [entry part is isolated — add connections to it first]"
                if entry in sim_result.isolated_parts else ""
            )
            lines.append(f"  • Scenario '{r.scenario_name}'{isolated_note}")
            lines.append(f"    Signal must flow from '{entry}' to '{tgts}'.")
            lines.append(f"    Check: are the relevant out-port → in-port connections present?")
            if r.warnings:
                for w in r.warnings:
                    lines.append(f"    ⚠ {w}")
        if persistent_issues:
            lines += [
                "",
                "Persistent (appeared in multiple passes — highest priority):",
            ]
            for iss in persistent_issues:
                lines.append(f"  • [PERSISTENT] {iss}")
        lines += [
            "",
            "Rules:",
            "  - ONLY add connect statements or add/fix port declarations.",
            "  - Do NOT change part def structure, attributes, or requirements.",
            "  - connect syntax: connect <instanceA>.<portA> to <instanceB>.<portB>;",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

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
        lev_hints: List[Dict] = []   # distance-2 suggestions for the LLM prompt

        # ── Tier 0-pre: strip `readonly` before attribute ────────────────────
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
                    meta = getattr(working_model, "metadata", None)
                    if meta is None:
                        object.__setattr__(working_model, "metadata", {})
                        meta = working_model.metadata
                    meta["last_sysml_text"] = working_sysml
                    latest_result = re_checked
                    if not latest_result.has_errors:
                        print(f"  └─ [RO-FIX]  ✓ all errors resolved", flush=True)
                        return working_sysml, working_model, latest_result
                    print(
                        f"  └─ [RO-FIX]  {latest_result.total_errors()} error(s) remain"
                        f" — continuing",
                        flush=True,
                    )

        # ── Tier 0-pre: SysML keyword quoting ────────────────────────────────
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
                meta = getattr(working_model, "metadata", None)
                if meta is None:
                    object.__setattr__(working_model, "metadata", {})
                    meta = working_model.metadata
                meta["last_sysml_text"] = working_sysml
                latest_result = re_checked
                if not latest_result.has_errors:
                    print(f"  └─ [KW-FIX]  ✓ all errors resolved", flush=True)
                    return working_sysml, working_model, latest_result
                print(
                    f"  └─ [KW-FIX]  {latest_result.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )

        # ── Tier 0: Levenshtein quick-fix ────────────────────────────────────
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
                meta = getattr(working_model, "metadata", None)
                if meta is None:
                    object.__setattr__(working_model, "metadata", {})
                    meta = working_model.metadata
                meta["last_sysml_text"] = working_sysml

                if not re_checked.has_errors:
                    print(
                        f"  └─ [LEV-FIX]  ✓ all errors resolved"
                        f" — LLM fix loop skipped",
                        flush=True,
                    )
                    return working_sysml, working_model, re_checked

                print(
                    f"  └─ [LEV-FIX]  {re_checked.total_errors()} error(s) remain"
                    f" — continuing to LLM fix loop",
                    flush=True,
                )
                latest_result = re_checked

            # Collect distance-2 hints for the LLM prompt
            lev_hints = lev.hints

        # ── Tier 0-post: undeclared guard attribute injection ─────────────────
        # sema error "No Feature named 'X' found" where X appears in a state
        # machine guard → inject `attribute X : Real/Boolean = <default>;`
        # into the owner part def.  No LLM needed — purely programmatic.
        if latest_result.sema_errors:
            working_sysml, n_injected = _inject_missing_guard_attrs(
                working_sysml, latest_result.sema_errors
            )
            if n_injected:
                re_checked = check_syntax(working_sysml)
                meta = getattr(working_model, "metadata", None)
                if meta is None:
                    object.__setattr__(working_model, "metadata", {})
                    meta = working_model.metadata
                meta["last_sysml_text"] = working_sysml
                print(
                    f"\n  ┌─ [ATTR-INJ]  {n_injected} missing guard attribute(s)"
                    f" injected — no LLM needed",
                    flush=True,
                )
                if not re_checked.has_errors:
                    print(f"  └─ [ATTR-INJ]  ✓ all errors resolved", flush=True)
                    return working_sysml, working_model, re_checked
                print(
                    f"  └─ [ATTR-INJ]  {re_checked.total_errors()} error(s) remain"
                    f" — continuing",
                    flush=True,
                )
                latest_result = re_checked

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
            n_merged = 0
            for chunk in sorted(chunks, key=lambda c: c.start_line, reverse=True):

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
                    continue
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
                    working_sysml = merge.merged_text
                    model_total_lines = len(working_sysml.splitlines())
                    n_merged += 1
                    status = f"Δlines={merge.line_delta:+d}"
                    if merge.warning:
                        print(f"  ║  └─ ⚠  merged  {status}  {merge.warning}", flush=True)
                    else:
                        print(f"  ║  └─ ✓  merged  {status}", flush=True)
                else:
                    print(f"  ║  └─ ✗  merge rejected — {merge.warning}", flush=True)

            lev_hints = []   # d=2 建议只在首次 LLM 调用时传递

            # 更新 model metadata，让后续流程读到最新文本
            meta = getattr(working_model, "metadata", None)
            if meta is None:
                object.__setattr__(working_model, "metadata", {})
                meta = working_model.metadata
            meta["last_sysml_text"] = working_sysml

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

    def _print_sim_result(self, sim_result: SimulationResult) -> None:
        """Compact simulation box (used in verbose mode inside iteration loop)."""
        total  = len(sim_result.scenario_results)
        passed = len(sim_result.passed_scenarios())
        failed = sim_result.failed_scenarios()
        print(f"\n  ┌─ Simulation: {sim_result.model_name}")
        print(f"  │  Score {sim_result.reachability_score:.3f}  "
              f"({passed}/{total} scenarios passed)")
        print(f"  │  Graph: {sim_result.num_parts} parts · "
              f"{sim_result.num_ports} ports · "
              f"{sim_result.num_connections} connections")
        if sim_result.isolated_parts:
            print(f"  │  ⚠ ISOLATED ({len(sim_result.isolated_parts)}): "
                  f"{', '.join(sim_result.isolated_parts)}")
        if failed:
            print(f"  │  Failed scenarios:")
            for r in failed:
                tgts = ", ".join(r.unreachable_targets) or "?"
                print(f"  │    ✗ {r.scenario_name}  (can't reach: {tgts})")
                for w in r.warnings:
                    print(f"  │      ⚠ {w}")
        else:
            print(f"  │  All scenarios passed ✓")
        for rec in sim_result.recommendations:
            if rec != "All scenarios passed — model connectivity is structurally sound":
                print(f"  │  • {rec}")
        print(f"  └{'─'*55}")

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
