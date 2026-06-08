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

from .base_agent import AgentMessage
from .design_agent import DesignAgent
from .requirements_agent import RequirementsAgent
from ..dse.design_space import DesignConfiguration, DesignParameter, DesignSpace, ParameterType
from ..dse.evaluator import DesignEvaluator
from ..dse.mcts import MCTSDesignExplorer
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
        depth, end = 0, brace
        for i in range(brace, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
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
    messages: List[AgentMessage] = field(default_factory=list)


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
    ):
        self.llm = llm
        self.rag = rag_retriever
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.rule_weight = rule_weight
        self.llm_weight = llm_weight
        self.verbose = verbose

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
            model, requirements, mcts_best_config=None
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
        print(f"  Simulation reachability:  {final_sim.reachability_score:.3f} "
              f"({len(final_sim.passed_scenarios())}/{len(final_sim.scenario_results)} scenarios)")
        print(f"  Part definitions: {len(final_model.part_definitions)}")
        print(f"  Requirements:     {len(requirements)}")
        if sim_warnings:
            print()
            for line in sim_warnings.splitlines():
                print(f"  {line}")
        print(f"{'='*60}\n")

        final_sysml = (
            (getattr(final_model, "metadata", None) or {}).get("last_sysml_text")
            or final_model.to_sysml_text()
        )
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
        Run MCTS Design Space Exploration on a validated model.

        Takes the output of generate() as input.  Explores the parameter
        space, injects the winning configuration into the model, then runs
        a final refinement pass to implement those architectural decisions.

        Pipeline
        ────────
        Phase 3  MCTS exploration + programmatic injection + grounding pass
        Phase 4-5  Iterative Refinement (with MCTS constraints in prompt)
        Phase 6  Behavioral Reachability Simulation

        Parameters
        ----------
        generate_result   Dict returned by generate().
        mcts_iterations   Number of MCTS simulation steps.
        mcts_seed         Random seed (None = non-deterministic).
        mcts_patience     Early-stop patience (steps without improvement).

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

        # ── Phase 3: MCTS ─────────────────────────────────────────────────────
        print("Phase 3: Design Space Exploration (MCTS)")
        print("-" * 40)
        design_space, best_config, pareto_front = self._explore_design_space(
            model, mcts_iterations, requirements,
            random_seed=mcts_seed,
            patience=mcts_patience,
        )
        self.state.design_space = design_space

        # Inject MCTS decisions into the model programmatically
        self._apply_best_config_to_model(best_config, model)
        self._apply_inject_attrs_to_sysml_text(model)
        self._apply_inject_protocol_to_sysml_text(model, best_config)
        self._apply_inject_sensor_count_to_sysml_text(model, best_config)
        # Structural grounding: LLM-driven pass for redundancy/state-machine
        self._mcts_structural_grounding_pass(model, best_config)
        self._print_exploration_summary(design_space, best_config, pareto_front)

        # ── Phase 4-5: Refinement with MCTS constraints ───────────────────────
        print("Phase 4-5: Iterative Refinement (MCTS-grounded)")
        print("-" * 40)
        final_model, final_score, final_sim = self._iterative_refinement(
            model, requirements, mcts_best_config=best_config
        )
        self.state.current_model = final_model
        print(f"  ✓ Final design score: {final_score:.3f}\n")

        # ── Phase 6: Simulation ───────────────────────────────────────────────
        # Simulation already ran in Phase 4-5 — reuse the result.
        print("Phase 6: Behavioral Reachability Simulation", flush=True)
        print("-" * 40)
        self._print_final_sim(final_sim)

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
        print(f"{'='*60}\n")

        final_sysml = (
            (getattr(final_model, "metadata", None) or {}).get("last_sysml_text")
            or final_model.to_sysml_text()
        )
        return {
            # ── Fields inherited / updated from generate() ────────────────────
            **generate_result,
            "model":              final_model,
            "model_sysml":        final_sysml,
            "model_summary":      final_model.get_summary(),
            "final_score":        final_score,
            "iterations":         self.state.iteration,
            "evaluation_history": self.state.evaluation_history,
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
        }

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

    def _explore_design_space(
        self,
        model: SysMLModel,
        mcts_iterations: int,
        requirements: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> Tuple[DesignSpace, DesignConfiguration, List[DesignConfiguration]]:
        """Phase 3: Define and explore the design space using MCTS.

        Returns (DesignSpace, best_config, pareto_front).
        """
        requirements = requirements or []
        design_space = self._define_design_space(model, requirements)
        self._add_inter_parameter_constraints(design_space)

        # 方案 B: score each config against real requirement bounds
        def evaluate_config(config: DesignConfiguration) -> Dict[str, float]:
            return self._score_config_against_requirements(config, requirements)

        explorer = MCTSDesignExplorer(
            design_space=design_space,
            evaluation_function=evaluate_config,
            max_depth=4,
            random_seed=random_seed,
        )
        best_config = explorer.search(
            num_iterations=mcts_iterations,
            patience=patience,
        )
        # Track exploration diagnostics in design space metadata
        design_space.objective_weights = {
            "iterations_run": float(explorer.iterations_run),
            "early_stopped": 1.0 if explorer.early_stopped else 0.0,
        }
        pareto_front = sorted(
            design_space.get_pareto_front(),
            key=lambda c: c.overall_score,
            reverse=True,
        )
        return design_space, best_config, pareto_front

    @staticmethod
    def _add_inter_parameter_constraints(space: DesignSpace) -> None:
        """Register typical engineering constraints linking the parameters."""

        def triple_redundancy_needs_sensors(p: Dict[str, Any]) -> bool:
            # Triple modular redundancy needs ≥3 sensors to be coherent
            return not (p.get("redundancy_level") == "triple"
                        and int(p.get("num_sensors", 0)) < 3)

        def dual_redundancy_needs_sensors(p: Dict[str, Any]) -> bool:
            return not (p.get("redundancy_level") == "dual"
                        and int(p.get("num_sensors", 0)) < 2)

        def centralised_caps_sensors(p: Dict[str, Any]) -> bool:
            # Centralised control struggles past 4 sensor inputs
            return not (p.get("distributed_control") is False
                        and int(p.get("num_sensors", 0)) > 5)

        space.add_constraint(triple_redundancy_needs_sensors)
        space.add_constraint(dual_redundancy_needs_sensors)
        space.add_constraint(centralised_caps_sensors)

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

        print(f"  ✓ Explored {summary['configurations_evaluated']} configurations "
              f"in {iters} iteration(s){' (early-stopped)' if early else ''}")
        print(f"  ✓ Pareto front size: {summary['pareto_front_size']}")

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
        print(f"  ✓ Best config applied to model: {best_params}\n")

    # ------------------------------------------------------------------
    # 方案 A: Requirements-driven design space generation
    # ------------------------------------------------------------------

    def _define_design_space(
        self,
        model: SysMLModel,
        requirements: Optional[List[str]] = None,
    ) -> DesignSpace:
        """Build a system-specific design space derived from requirements and model."""
        requirements = requirements or []
        space = DesignSpace(name=f"{model.name}_DesignSpace")

        # ── Redundancy (SAFE-driven) ───────────────────────────────────
        safe_count = sum(1 for r in requirements if "-SAFE-" in r)
        if safe_count == 0:
            redundancy_choices = ["none"]
        elif safe_count == 1:
            redundancy_choices = ["none", "dual"]
        else:
            redundancy_choices = ["none", "dual", "triple"]

        # Safety-driven default: start MCTS from the most appropriate redundancy
        # level given the number of SAFE requirements.  Starting from "none" traps
        # the search in the safety-floor zone (overall ≤ 0.30) before it can
        # discover higher-redundancy configurations — especially when the
        # triple_redundancy_needs_sensors constraint requires ≥ 3 sensors first.
        if safe_count >= 3:
            default_redundancy = "triple"
        elif safe_count >= 1:
            default_redundancy = "dual"
        else:
            default_redundancy = "none"

        space.add_parameter(DesignParameter(
            name="redundancy_level",
            param_type=ParameterType.CATEGORICAL,
            default_value=default_redundancy,
            choices=redundancy_choices,
            description=f"Hardware redundancy level (derived from {safe_count} SAFE requirement(s))",
        ))

        # ── Communication protocol (INTF-driven) ──────────────────────
        all_req_text = " ".join(requirements)
        intf_text = " ".join(r for r in requirements if "-INTF-" in r).upper()
        # Expanded known-protocols list, including drone/aerospace standards
        known_protocols = [
            "CAN", "ETHERNET", "SPI", "I2C",
            "MAVLINK",                      # drone ground-control
            "ROS2",                         # robotic middleware
            "WIRELESS",
            "MODBUS", "PROFINET",
            "ASTM",                         # ASTM F3411-22 Remote ID (drone)
            "ADSB", "ADS-B",                # Automatic Dependent Surveillance
            "UAVCAN", "DRONECAN",           # drone CAN variants
            "OPENAPI", "REST",              # web/cloud interfaces
        ]
        detected = [p for p in known_protocols if p in intf_text]

        # Domain-adaptive fallback: when no protocol is found in INTF reqs,
        # use system-name and requirement text to pick sensible defaults.
        if len(detected) < 2:
            combined_lower = (model.name + " " + all_req_text).lower()
            # Ordered most-specific first so industrial + robotic systems
            # (e.g. a CNC arm) don't fall into the generic "robot→ROS2" bucket.
            _DRONE_KWS    = {"drone", "uav", "aerial", "quadcopter", "rotor",
                             "flight", "autopilot"}
            _INDUSTRIAL_KWS = {"factory", "plc", "industrial", "cnc", "conveyor",
                               "scada", "fieldbus"}
            _ROBOT_KWS    = {"ros2", "ros ", "manipulator", "mobile robot"}
            if any(kw in combined_lower for kw in _DRONE_KWS):
                domain_fallback = ["MAVLink", "Ethernet"]
            elif any(kw in combined_lower for kw in _INDUSTRIAL_KWS):
                domain_fallback = ["CAN", "Modbus"]
            elif any(kw in combined_lower for kw in _ROBOT_KWS):
                domain_fallback = ["ROS2", "Ethernet"]
            else:
                domain_fallback = ["CAN", "Ethernet"]
            protocol_choices = (detected + domain_fallback)[:4]
        else:
            protocol_choices = detected[:4]

        # Normalise to mixed-case display names
        _display = {
            "ETHERNET": "Ethernet", "MAVLINK": "MAVLink", "ROS2": "ROS2",
            "WIRELESS": "Wireless", "MODBUS": "Modbus", "PROFINET": "PROFINET",
            "ASTM": "ASTM", "ADSB": "ADSB", "ADS-B": "ADS-B",
            "UAVCAN": "UAVCAN", "DRONECAN": "DroneCAN",
        }
        protocol_choices = [_display.get(p, p) for p in dict.fromkeys(protocol_choices)]
        # First detected protocol is the best default; else first fallback
        default_protocol = (
            _display.get(detected[0], detected[0]) if detected else protocol_choices[0]
        )
        space.add_parameter(DesignParameter(
            name="communication_protocol",
            param_type=ParameterType.CATEGORICAL,
            default_value=default_protocol,
            choices=protocol_choices,
            description="Communication protocol between components (derived from INTF requirements)",
        ))

        # ── Control frequency (PERF-driven) ───────────────────────────
        perf_nums = []
        for req in requirements:
            if "-PERF-" not in req:
                continue
            body = req.split(":", 1)[-1]
            # Match numbers followed by Hz / kHz / frequency-related units
            hz_matches = re.findall(
                r'\b(\d+(?:\.\d+)?)\s*(?:hz|khz|kHz|Hz|KHz)\b', body, re.IGNORECASE
            )
            perf_nums.extend(float(m) for m in hz_matches)
        if perf_nums:
            target_hz = max(perf_nums)
            # Allow exploring half to double the stated requirement
            min_hz = max(1.0, target_hz * 0.5)
            max_hz = target_hz * 2.0
            default_hz = target_hz
        else:
            min_hz, max_hz, default_hz = 10.0, 1000.0, 100.0
        space.add_parameter(DesignParameter(
            name="control_frequency_hz",
            param_type=ParameterType.CONTINUOUS,
            default_value=default_hz,
            min_value=min_hz,
            max_value=max_hz,
            unit="Hz",
            description="Main control loop frequency (derived from PERF requirements)",
        ))

        # ── Distributed vs. centralised (structural, from part count) ─
        part_count = len(model.part_definitions)
        space.add_parameter(DesignParameter(
            name="distributed_control",
            param_type=ParameterType.BOOLEAN,
            default_value=(part_count > 4),   # lean distributed if already complex
            description="Distributed vs. centralised control (based on model complexity)",
        ))

        # ── Sensor count (PERF + model-driven) ────────────────────────
        # Count parts whose name contains sensor-like keywords
        sensor_kws = {"sensor", "detector", "monitor", "camera", "lidar", "imu", "gps"}
        existing_sensors = sum(
            1 for p in model.part_definitions
            if any(kw in p.name.lower() for kw in sensor_kws)
        )
        min_sensors = max(1, existing_sensors)
        max_sensors = max(6, existing_sensors + 3)
        sensor_choices = list(range(min_sensors, max_sensors + 1))

        # Ensure the default sensor count satisfies the inter-parameter constraint
        # (triple_redundancy_needs_sensors requires num_sensors ≥ 3 for TMR,
        # dual_redundancy_needs_sensors requires num_sensors ≥ 2 for dual).
        # This prevents the root MCTS node from being infeasible-at-first-step
        # when the safety-driven default redundancy is triple or dual.
        if default_redundancy == "triple":
            raw_default_sensors = max(3, min_sensors)
        elif default_redundancy == "dual":
            raw_default_sensors = max(2, min_sensors)
        else:
            raw_default_sensors = max(1, existing_sensors) if existing_sensors else 2
        default_sensors = min(max_sensors, max(min_sensors, raw_default_sensors))

        space.add_parameter(DesignParameter(
            name="num_sensors",
            param_type=ParameterType.DISCRETE,
            default_value=default_sensors,
            choices=sensor_choices,
            description=f"Number of sensor units (model has {existing_sensors} sensor-like parts)",
        ))

        return space

    # ------------------------------------------------------------------
    # 方案 B: Requirement-bound scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_config_against_requirements(
        config: DesignConfiguration,
        requirements: List[str],
    ) -> Dict[str, float]:
        """Score a DesignConfiguration against extracted requirement bounds.

        Replaces the generic parameter-count heuristic so MCTS explores
        a semantically grounded fitness landscape.
        """
        scores: Dict[str, float] = {}

        # ── PERF satisfaction ──────────────────────────────────────────
        perf_reqs = [r for r in requirements if "-PERF-" in r]
        if perf_reqs:
            hz_values: List[float] = []
            for req in perf_reqs:
                body = req.split(":", 1)[-1]
                hz_matches = re.findall(
                    r'\b(\d+(?:\.\d+)?)\s*(?:hz|kHz|Hz|KHz)\b', body, re.IGNORECASE
                )
                hz_values.extend(float(m) for m in hz_matches)
            freq = float(config.parameters.get("control_frequency_hz", 100.0))
            if hz_values:
                target = max(hz_values)
                scores["perf_satisfaction"] = min(1.0, freq / max(target, 1.0))
            else:
                # No frequency bound found — score neutral
                scores["perf_satisfaction"] = 0.7
        else:
            scores["perf_satisfaction"] = 1.0

        # ── Safety margin ──────────────────────────────────────────────
        safe_count = sum(1 for r in requirements if "-SAFE-" in r)
        redundancy_map = {"none": 0, "dual": 1, "triple": 2}
        redundancy = redundancy_map.get(
            str(config.parameters.get("redundancy_level", "none")), 0
        )
        if safe_count == 0:
            scores["safety_margin"] = 1.0
        else:
            needed = min(2, safe_count)
            raw = min(1.0, (redundancy + 0.1) / (needed + 0.1))
            # Hard penalty: when ≥ 3 SAFE requirements exist, redundancy=none is
            # architecturally unacceptable — clamp safety_margin to near-zero so
            # MCTS consistently selects dual or triple redundancy.
            if safe_count >= 3 and redundancy == 0:
                raw = 0.01
            scores["safety_margin"] = raw

        # ── Protocol match ─────────────────────────────────────────────
        intf_reqs = [r for r in requirements if "-INTF-" in r]
        protocol = str(config.parameters.get("communication_protocol", "")).upper()
        if intf_reqs:
            intf_text = " ".join(intf_reqs).upper()
            all_req_text = " ".join(requirements).upper()
            if protocol and protocol in intf_text:
                # Protocol explicitly named in INTF requirements
                scores["protocol_match"] = 1.0
            elif protocol and protocol in all_req_text:
                # Protocol mentioned somewhere in requirements (not just INTF)
                scores["protocol_match"] = 0.7
            elif protocol in ("MAVLINK", "ROS2", "UAVCAN", "DRONECAN", "ASTM"):
                # Domain-appropriate protocol for aerial/robotic systems;
                # not penalised as heavily as generic bus protocols
                scores["protocol_match"] = 0.65
            else:
                scores["protocol_match"] = 0.4
        else:
            scores["protocol_match"] = 0.8

        # ── Structural simplicity (cost proxy) ────────────────────────
        sensor_count = int(config.parameters.get("num_sensors", 3))
        all_sensor_choices = [1, 2, 3, 4, 5, 6]
        max_s = max(all_sensor_choices)
        scores["simplicity"] = 1.0 - (sensor_count - 1) / max(max_s - 1, 1)

        return scores

    # ------------------------------------------------------------------
    # 方案 C: Apply MCTS best config back to the SysMLModel
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_best_config_to_model(
        best_config: DesignConfiguration,
        model: SysMLModel,
    ) -> None:
        """Write MCTS winning parameter decisions into the SysMLModel in-place.

        Only updates elements that are clearly mapped — no structural changes
        that could invalidate satisfy links or existing part defs.
        """
        params = best_config.parameters
        if not params:
            return

        # 1. Control frequency → add/update ONLY the dedicated controlFrequency
        #    attribute in the primary controller part.
        #    IMPORTANT: do NOT touch telemetryRate, gnssRate, updateRate, or any
        #    other domain-specific rate attribute — those are set by requirements.
        freq = params.get("control_frequency_hz")
        if freq is not None:
            freq_str = str(round(float(freq), 2))
            # Identify the "main controller" part: most FUNC/PERF satisfy links.
            _CTRL_KWS = {"controller", "flight", "control", "nav", "autopilot"}
            ctrl_part = None
            best_ctrl_score = -1
            for part in model.part_definitions:
                score = sum(
                    1 for sr in part.satisfy_relationships
                    if sr.target and any(
                        cat in (sr.target.name or "").upper()
                        for cat in ("FUNC", "PERF")
                    )
                ) + (2 if any(kw in part.name.lower() for kw in _CTRL_KWS) else 0)
                if score > best_ctrl_score:
                    best_ctrl_score = score
                    ctrl_part = part
            if ctrl_part is not None:
                # Update or inject the `controlFrequency` attribute.
                _CF_KWS = {"controlfrequency", "controlfreq", "loopfrequency",
                           "samplingfrequency"}
                matched = False
                for attr in ctrl_part.attributes:
                    if attr.name.lower() in _CF_KWS:
                        attr.default_value = freq_str
                        matched = True
                        break
                if not matched:
                    # Inject as new attribute via metadata note (actual text
                    # injection happens in the refinement prompt instruction)
                    if not hasattr(model, "metadata") or model.metadata is None:
                        object.__setattr__(model, "metadata", {})
                    model.metadata.setdefault("mcts_inject_attrs", []).append(
                        {
                            "part": ctrl_part.name,
                            "attr": "controlFrequency",
                            "value": freq_str,
                            "unit": "Hz",
                        }
                    )

        # 2. Redundancy level → doc annotation on model description
        redundancy = str(params.get("redundancy_level", "none"))
        if redundancy != "none":
            tag = f"redundancy={redundancy}"
            if model.description and tag not in model.description:
                model.description = f"{model.description} [{tag}]"
            elif not model.description:
                model.description = f"[{tag}]"

        # 3. Communication protocol → typed ports
        #    Replace generic DataPort / RfPort with the protocol signal type on
        #    all DATA-carrying ports.  PowerPort is left unchanged (it carries
        #    electrical power, not protocol data).
        _POWER_PORT_NAMES = {"powerport", "power", "powerout", "powerin"}
        protocol = params.get("communication_protocol")
        if protocol:
            # Sanitize: strip hyphens, spaces, etc. so "ADS-B" → "ADSB" (valid SysML identifier)
            _proto_id = re.sub(r"[^A-Za-z0-9]", "", str(protocol))
            protocol_type_name = f"{_proto_id}Signal"
            for part in model.part_definitions:
                for port in part.ports:
                    existing = (port.type_ref.name or "") if port.type_ref else ""
                    # Skip ports whose name or type clearly belong to power domain
                    if existing.lower() in _POWER_PORT_NAMES:
                        continue
                    if port.name.lower() in _POWER_PORT_NAMES:
                        continue
                    # Replace generic or absent types; leave already-specific ones alone
                    generic_types = {"dataport", "data", "rfport", "rf", ""}
                    if existing.lower() in generic_types:
                        port.type_ref = ElementRef(name=protocol_type_name)

        # 4. Recommended sensor count → model-level metadata
        num_sensors = params.get("num_sensors")
        if num_sensors is not None:
            if not hasattr(model, "metadata") or model.metadata is None:
                object.__setattr__(model, "metadata", {})
            model.metadata["recommended_sensor_count"] = int(num_sensors)
            model.metadata["mcts_best_config"] = best_config.name

    @staticmethod
    def _apply_inject_attrs_to_sysml_text(model: SysMLModel) -> None:
        """Inject attributes from ``mcts_inject_attrs`` directly into the
        stored SysML text (``model.metadata["last_sysml_text"]``).

        When :meth:`_apply_best_config_to_model` wants to add an attribute
        (e.g. ``controlFrequency``) but the part already has no matching
        attribute object to update, it records the injection request in
        ``model.metadata["mcts_inject_attrs"]``.  This method applies those
        requests at the *text* level so the refinement LLM receives a model
        that already contains the desired attribute rather than having to infer
        it from a prose instruction (which it often ignores).

        Each entry in ``mcts_inject_attrs`` is a dict:
            {"part": str, "attr": str, "value": str, "unit": str}
        """
        meta = getattr(model, "metadata", None) or {}
        inject_attrs = meta.get("mcts_inject_attrs", [])
        sysml_text = meta.get("last_sysml_text", "")

        if not inject_attrs or not sysml_text:
            return

        from .design_agent import DesignAgent
        result = sysml_text
        applied: List[str] = []

        for entry in inject_attrs:
            part_name = entry.get("part", "")
            attr_name = entry.get("attr", "")
            value = entry.get("value", "")
            unit = entry.get("unit", "")

            if not part_name or not attr_name:
                continue

            # Skip if the attribute already appears in the text
            already_re = re.compile(
                rf"\battribute\s+{re.escape(attr_name)}\s*:", re.IGNORECASE
            )
            if already_re.search(result):
                continue

            # Locate the target part def block
            part_def_re = re.compile(
                rf"\bpart\s+def\s+{re.escape(part_name)}\s*\{{"
            )
            m_part = part_def_re.search(result)
            if not m_part:
                continue

            brace_open = result.index("{", m_part.start())
            closing = DesignAgent._find_block_end(result, brace_open)
            if closing == -1:
                continue

            # Build the attribute declaration
            unit_suffix = f" [{unit}]" if unit else ""
            attr_line = f"attribute {attr_name} : Real = {value}{unit_suffix};"

            # Inject before the closing `}` of the part def
            result = (
                result[:closing]
                + "\n        // (controlFrequency injected by MCTS pipeline)\n"
                + "        " + attr_line + "\n    "
                + result[closing:]
            )
            applied.append(f"{part_name}.{attr_name}={value}")

        if applied:
            model.metadata["last_sysml_text"] = result
            model.metadata["mcts_injected_attrs_applied"] = applied

    @staticmethod
    def _apply_inject_protocol_to_sysml_text(
        model: SysMLModel,
        best_config: DesignConfiguration,
    ) -> None:
        """Replace generic ``DataPort`` / ``RFPort`` type annotations with the
        MCTS-selected protocol signal type directly in
        ``model.metadata["last_sysml_text"]``.

        ``_apply_best_config_to_model()`` updates ``port.type_ref`` on in-memory
        model objects, but those changes are never written back to the stored SysML
        text.  This method applies the identical substitution at the text level so
        the final export is consistent with the MCTS architectural decision —
        **even when the refinement loop is skipped because the initial model already
        meets the quality threshold**.

        Steps:
            1. Inject ``port def {Proto}Signal;`` at the package level (if absent).
            2. Replace every directed-port type annotation that currently uses
               ``DataPort``, ``RFPort``, or ``RfPort`` with ``{Proto}Signal``.
               Ports whose *name* contains "power" or "pwr" are left unchanged.
        """
        meta = getattr(model, "metadata", None) or {}
        sysml_text = meta.get("last_sysml_text", "")
        if not sysml_text:
            return

        protocol = str(best_config.parameters.get("communication_protocol", ""))
        if not protocol or protocol.lower() == "none":
            return

        # Sanitize: "ADS-B" → "ADSB"  (strip chars invalid in SysML identifiers)
        proto_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
        if not proto_id:
            return

        signal_type = f"{proto_id}Signal"

        # ── 1. Add port def at package level if not already present ───────
        if not re.search(
            rf"\bport\s+def\s+{re.escape(signal_type)}\b", sysml_text
        ):
            pkg_open_re = re.compile(r"(\bpackage\s+\w+\s*\{)")
            sysml_text = pkg_open_re.sub(
                rf"\1\n    port def {signal_type};",
                sysml_text,
                count=1,
            )

        # ── 2. Replace directed port type annotations ─────────────────────
        # Matches `in port foo : DataPort` / `out port bar : RFPort` etc.
        # Does NOT match `port def DataPort { ... }` — those lack a direction
        # keyword before `port`, so the alternation `(?:in|out|inout)` prevents
        # them from being matched.
        port_usage_re = re.compile(
            r"\b((?:in|out|inout)\s+port\s+(\w+)\s*:\s*)"
            r"(DataPort|RFPort|RfPort)\b",
            re.IGNORECASE,
        )

        # Electrical-power port names: Supply / In / Out / Bus / Rail / Link / Pwr
        # Status / monitoring ports (powerStatus, pwrStatus) carry *data*, not
        # electricity — they must be replaced with the protocol signal type.
        _PWR_EXACT = re.compile(
            r"\b(power|pwr)(supply|in|out|bus|rail|link|feed|connector|line)\b",
            re.IGNORECASE,
        )

        def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
            port_name: str = m.group(2)
            pn_lower = port_name.lower()
            # Exclude only TRUE electrical power ports (e.g. powerIn, powerOut,
            # powerSupply, pwrBus).  "powerStatus" and "pwrStatus" are data ports
            # and must be replaced.
            if _PWR_EXACT.search(pn_lower) or pn_lower in ("power", "pwr"):
                return m.group(0)   # leave electrical power ports unchanged
            return f"{m.group(1)}{signal_type}"

        result = port_usage_re.sub(_replace, sysml_text)

        # ── 3. Remove stale generic port def declarations ─────────────────
        # After replacing all usages, `port def DataPort;` / `port def RfPort;`
        # etc. become unreferenced dead declarations.  Strip them so the final
        # SysML text doesn't contain unused generic defs that confuse the scorer.
        for _generic_def in ("DataPort", "RFPort", "RfPort", "GenericPort"):
            # Use MULTILINE + ^ so the leading [ \t]* only eats the line's own
            # indentation, not blank separator lines between declarations.
            result = re.sub(
                rf"^[ \t]*\bport\s+def\s+{_generic_def}\s*;[ \t]*\n?",
                "",
                result,
                flags=re.IGNORECASE | re.MULTILINE,
            )

        if result != sysml_text:
            model.metadata["last_sysml_text"] = result
            model.metadata["mcts_injected_protocol_signal"] = signal_type

    @staticmethod
    def _apply_inject_sensor_count_to_sysml_text(
        model: SysMLModel,
        best_config: DesignConfiguration,
    ) -> None:
        """Programmatically add extra part usages to reach the MCTS sensor count target.

        MCTS may decide ``num_sensors=3`` but the generated model typically has only
        one sensor-like part def (e.g. PerceptionSystem).  This method adds additional
        part usage declarations in the assembly section so the instantiation count
        matches the MCTS decision.

        Example injection (num_sensors=3, existing=1):
            part sensorUnit2 : PerceptionSystem;
            part sensorUnit3 : PerceptionSystem;

        The injection is purely additive — no structural SysML is changed.
        """
        meta = getattr(model, "metadata", None) or {}
        sysml_text = meta.get("last_sysml_text", "")
        if not sysml_text:
            return

        target = int(best_config.parameters.get("num_sensors", 0))
        if target <= 1:
            return

        # Identify the primary sensor-like part def name
        _SENSOR_KWS = {"sensor", "perception", "detector", "camera",
                       "lidar", "imu", "gps", "radar"}
        sensor_part_name: Optional[str] = None
        for part in model.part_definitions:
            if any(kw in part.name.lower() for kw in _SENSOR_KWS):
                sensor_part_name = part.name
                break
        if sensor_part_name is None:
            return  # no sensor-like part found — nothing to do

        # Count existing part usages of this type in the assembly section.
        # Pattern: `part <varName> : <SensorPartName>;` (not `part def ...`)
        usage_re = re.compile(
            rf"\bpart\s+(?!def\b)\w+\s*:\s*{re.escape(sensor_part_name)}\s*;",
            re.IGNORECASE,
        )
        existing = len(usage_re.findall(sysml_text))
        if existing >= target:
            return  # already meets or exceeds target

        # Find injection point: right after the last `part <name> : <Type>;` line
        # (any part usage, not just sensor ones) — before the first connect.
        any_usage_re = re.compile(r"\bpart\s+(?!def\b)\w+\s*:\s*\w+\s*;")
        last_usage_end = 0
        for m in any_usage_re.finditer(sysml_text):
            last_usage_end = m.end()
        if last_usage_end == 0:
            return  # no assembly section found

        # ── Discover the primary sensor's instance name and its connections ──
        # Pattern: `part <instanceName> : <SensorPartName>;`
        primary_usage_re = re.compile(
            rf"\bpart\s+(\w+)\s*:\s*{re.escape(sensor_part_name)}\s*;",
            re.IGNORECASE,
        )
        primary_match = primary_usage_re.search(sysml_text)
        primary_instance = primary_match.group(1) if primary_match else None

        # Find out-port names on the sensor part def (e.g. sensorStatus, navData)
        # that the primary instance connects OUTWARD (source side of a connect).
        # We build a set of already-occupied target port keys to avoid fan-in.
        # SysML v2 connect uses dot notation (per official examples corpus).
        connect_re = re.compile(
            r"\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)\.(\w+)\s*;",
            re.IGNORECASE,
        )
        occupied_targets: set = set()
        primary_out_ports: list = []
        if primary_instance:
            for cm in connect_re.finditer(sysml_text):
                src_inst, src_port, tgt_inst, tgt_port = cm.groups()
                tgt_key = f"{tgt_inst}.{tgt_port}"
                occupied_targets.add(tgt_key)
                if src_inst.lower() == primary_instance.lower():
                    primary_out_ports.append((src_port, tgt_inst, tgt_port))

        # ── Build additional part usages + safe connect stubs ─────────────────
        # Rule: only inject a sensorUnitN if at least one connection can be
        # wired for it.  A unit with zero connections is an isolated node —
        # a real design defect that lowers the reachability score.
        new_lines: list = []
        new_connects: list = []
        actually_injected = 0

        for i in range(existing + 1, target + 1):
            unit_name = f"sensorUnit{i}"
            unit_connects: list = []

            # For each out-port the primary sensor exposes, try to add a connect
            # for this redundant instance.  Skip if the same target is already
            # occupied (fan-in guard) — those cases need a voting intermediary
            # that is beyond the scope of programmatic injection.
            for src_port, tgt_inst, tgt_port in primary_out_ports:
                tgt_key = f"{tgt_inst}.{tgt_port}"
                if tgt_key not in occupied_targets:
                    unit_connects.append(
                        f"    connect {unit_name}.{src_port} to {tgt_inst}.{tgt_port};"
                    )
                    occupied_targets.add(tgt_key)

            if not unit_connects:
                # All target ports already occupied (fan-in guard) — skip
                # injection rather than creating an isolated node.
                continue

            new_lines.append(f"    part {unit_name} : {sensor_part_name};")
            new_connects.extend(unit_connects)
            actually_injected += 1

        if not new_lines:
            return   # nothing to inject

        # Inject part usages after the last existing part usage
        usage_block = "\n" + "\n".join(new_lines)
        result = sysml_text[:last_usage_end] + usage_block + sysml_text[last_usage_end:]

        # Inject connect stubs just before the closing `}` of the package
        if new_connects:
            connect_block = (
                "\n    // MCTS-injected redundant sensor connects (fan-in-safe only):\n"
                + "\n".join(new_connects)
                + "\n"
            )
            last_brace = result.rfind("}")
            if last_brace != -1:
                result = result[:last_brace] + connect_block + result[last_brace:]

        model.metadata["last_sysml_text"] = result
        model.metadata["mcts_injected_sensor_units"] = actually_injected

    def _mcts_structural_grounding_pass(
        self,
        model: SysMLModel,
        best_config: DesignConfiguration,
    ) -> None:
        """Run a focused, unconditional LLM call to implement structural MCTS decisions.

        Specifically handles ``redundancy_level`` — the only MCTS decision that
        requires LLM to generate new SysML structure (a triple/dual-channel state def).
        Unlike the refinement loop this pass is **not gated by the quality threshold**:
        it always runs when MCTS selected a non-trivial redundancy level, regardless of
        how high the initial model scored.

        The call uses a minimal, single-task prompt so the LLM cannot drift into
        unrelated changes.  A regression guard reverts the result if part defs were
        dropped or the expected redundancy structure is absent.
        """
        redundancy = str(best_config.parameters.get("redundancy_level", "none"))
        if redundancy == "none":
            return

        meta = getattr(model, "metadata", None) or {}
        sysml_text = meta.get("last_sysml_text", "")
        if not sysml_text:
            return

        # ── Already-implemented check ──────────────────────────────────────
        _TRIPLE_SIGNALS = re.compile(
            r"TripleChannel|redundancyChannels\s*:\s*Integer\s*=\s*3"
            r"|ChannelA\b.*ChannelB\b.*ChannelC\b",
            re.DOTALL,
        )
        _DUAL_SIGNALS = re.compile(
            r"DualChannel|redundancyChannels\s*:\s*Integer\s*=\s*2"
        )
        if redundancy == "triple" and _TRIPLE_SIGNALS.search(sysml_text):
            if self.verbose:
                print("  [DEBUG] MCTS Grounding — triple redundancy already present, skipping")
            return
        if redundancy == "dual" and _DUAL_SIGNALS.search(sysml_text):
            if self.verbose:
                print("  [DEBUG] MCTS Grounding — dual redundancy already present, skipping")
            return

        # ── Identify the safety/monitor part def to inject into ───────────
        _SAFETY_KWS = {"safety", "monitor", "fault", "health"}
        target_part: Optional[str] = None
        for part in model.part_definitions:
            if any(kw in part.name.lower() for kw in _SAFETY_KWS):
                target_part = part.name
                break
        if target_part is None and model.part_definitions:
            target_part = model.part_definitions[-1].name  # last-resort fallback
        if target_part is None:
            return

        if self.verbose:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] MCTS Structural Grounding Pass")
            print(f"  {'─'*60}")
            print(f"  Adding {redundancy} redundancy → {target_part}")

        # ── Single focused LLM call ────────────────────────────────────────
        grounding_result = self.design_agent.cot.mcts_structural_grounding(
            redundancy_level=redundancy,
            target_part=target_part,
            sysml_text=sysml_text,
        )

        if not grounding_result.extracted_sysml:
            if self.verbose:
                print("  ⚠ MCTS Grounding — no SysML extracted, reverting")
            return

        from .design_agent import DesignAgent
        fixed, _ = DesignAgent._fix_doc_syntax(grounding_result.extracted_sysml)

        # ── Regression guard: part def count must not drop ─────────────────
        orig_parts = len(re.findall(r"\bpart\s+def\s+\w+", sysml_text))
        new_parts  = len(re.findall(r"\bpart\s+def\s+\w+", fixed))
        if new_parts < orig_parts:
            if self.verbose:
                print(f"  ⚠ MCTS Grounding — part def count dropped "
                      f"({orig_parts} → {new_parts}), reverting")
            return

        # ── Verify the redundancy structure was actually added ─────────────
        _TRIPLE_CHECK = re.compile(r"ChannelA|TripleChannel|redundancyChannels")
        _DUAL_CHECK   = re.compile(r"DualChannel|redundancyChannels")
        if redundancy == "triple" and not _TRIPLE_CHECK.search(fixed):
            if self.verbose:
                print("  ⚠ MCTS Grounding — LLM did not add triple structure, reverting")
            return
        if redundancy == "dual" and not _DUAL_CHECK.search(fixed):
            if self.verbose:
                print("  ⚠ MCTS Grounding — LLM did not add dual structure, reverting")
            return

        # ── Accept ────────────────────────────────────────────────────────
        model.metadata["last_sysml_text"] = fixed
        model.metadata["mcts_grounding_applied"] = redundancy
        if self.verbose:
            print(f"  ✓ MCTS Grounding — {redundancy} redundancy added to {target_part}")

    def _iterative_refinement(
        self,
        model: SysMLModel,
        requirements: List[str],
        mcts_best_config: Optional[DesignConfiguration] = None,
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
            self._build_mcts_design_constraints(mcts_best_config)
            if mcts_best_config else ""
        )
        if self.verbose and mcts_constraints:
            print(f"\n  {'─'*60}")
            print(f"  [DEBUG] MCTS constraints injected into every refinement prompt")
            print(f"  {'─'*60}")
            print(mcts_constraints)

        for iteration in range(self.max_iterations):
            self.state.iteration = iteration + 1

            # ── Step 0: Syntax gate — fix errors before evaluation ────────
            current_sysml = (
                (getattr(current_model, "metadata", None) or {}).get("last_sysml_text")
                or current_model.to_sysml_text()
            )
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

            # ── Rule-based evaluation (pass cached syntax + sim results) ──
            eval_result = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}",
                    parameters={},
                ),
                model=current_model,
                mcts_config=mcts_best_config,
                syntax_result=syntax_result,
                sim_result=sim_result,
            )
            rule_score = eval_result.weighted_total

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
            })

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
                _sysml_after = (
                    (getattr(current_model, "metadata", None) or {}).get("last_sysml_text")
                    or current_model.to_sysml_text()
                )
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
                        syntax_result=syntax_result,
                        sim_result=sim_result,
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
                refine_result = self.design_agent.run({
                    "system_name": current_model.name,
                    "requirements": requirements,
                    "existing_model": current_model,
                    "refinement_feedback": refinement_feedback,
                    "refinement_issues": eval_result.issues + eval_result.recommendations,
                    "verbose": self.verbose,
                })
                if refine_result.success and isinstance(refine_result.output, _SysMLModelTypes):
                    candidate = refine_result.output
                    # ── P0: Regression prevention ─────────────────────────
                    candidate_eval = self.evaluator.evaluate(
                        config=DesignConfiguration(name="candidate", parameters={}),
                        model=candidate,
                    )
                    delta = candidate_eval.weighted_total - rule_score
                    delta_str = f"{delta:+.3f}"
                    if candidate_eval.weighted_total >= rule_score - 0.05:
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
                    else:
                        print(
                            f"  ⚠ Refinement regression detected "
                            f"(rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f}), "
                            f"keeping current model",
                            flush=True,
                        )

        return best_model, best_score, best_sim_result or last_sim_result

    @staticmethod
    def _build_mcts_design_constraints(best_config: DesignConfiguration) -> str:
        """Translate MCTS best-config parameters into concrete SysML implementation guidance.

        Each parameter is mapped to the specific SysML construct the LLM must
        produce, so refinement rounds are grounded in the architectural decisions
        that MCTS made rather than guessing from scratch.
        """
        params = best_config.parameters
        if not params:
            return ""

        lines = [
            "MCTS Architectural Decisions"
            " (these must be faithfully implemented in the SysML model):"
        ]

        # Redundancy level → state def structure (canonical SysML v2 syntax)
        redundancy = str(params.get("redundancy_level", "none"))
        if redundancy == "triple":
            lines.append(
                "  • redundancy_level=triple  →  add a state def implementing 2-of-3 "
                "majority voting.  Use canonical SysML v2 syntax — `transition <name> "
                "first <state> if <guard> then <state>;` (NOT `from/to/when`, NOT `->`). "
                "Declare `action def emergencyStop {}` at the part-def top level (NOT "
                "inline inside an entry); reference it via `entry action stop : "
                "emergencyStop;` from the failsafe state.  Ground guard names by "
                "declaring matching Boolean attributes (channelAFailed / channelBFailed "
                "/ channelCFailed)."
            )
        elif redundancy == "dual":
            lines.append(
                "  • redundancy_level=dual    →  add a state def with primary/backup "
                "channels.  Use canonical SysML v2 syntax — `transition <name> first "
                "<state> if <guard> then <state>;` (NOT `from/to/when`, NOT `->`). "
                "Declare `action def emergencyStop {}` at the part-def top level (NOT "
                "inline); reference it via `entry action stop : emergencyStop;` from "
                "the failsafe state.  Ground guard names with Boolean attributes "
                "(primaryChannelFailed / backupChannelFailed)."
            )

        # Control frequency → numeric attribute (ONLY in main controller, nowhere else)
        freq = params.get("control_frequency_hz")
        if freq is not None:
            lines.append(
                f"  • control_frequency_hz={freq}  →  ADD (or update) exactly ONE "
                f"attribute named `controlFrequency : Real = {freq} [Hz]` in the main "
                f"flight-controller / autopilot part def.\n"
                f"    ✗ DO NOT change `telemetryRate`, `gnssRate`, `updateRate`, "
                f"`sampleRate`, or any other existing rate/frequency attribute — "
                f"those values come from requirements and must stay unchanged."
            )

        # Communication protocol → port type replacement
        protocol = params.get("communication_protocol")
        if protocol and str(protocol).lower() not in ("none", ""):
            # Sanitize: strip hyphens/spaces so "ADS-B" → "ADSB" (valid SysML identifier)
            _proto_id = re.sub(r"[^A-Za-z0-9]", "", str(protocol))
            lines.append(
                f"  • communication_protocol={protocol}  →  THREE mandatory steps:\n"
                f"    1. Add `port def {_proto_id}Signal;` at the package level.\n"
                f"    2. Change EVERY port currently typed as `DataPort` or `RfPort` "
                f"to `{_proto_id}Signal` (e.g. `in port gnssIn : {_proto_id}Signal;`).\n"
                f"    3. Leave `PowerPort`-typed ports unchanged — they carry "
                f"electrical power, not protocol data.\n"
                f"    ✗ Do NOT keep any `DataPort` or `RfPort` in the final model."
            )

        # Distributed vs centralised
        distributed = params.get("distributed_control")
        if distributed is True:
            lines.append(
                "  • distributed_control=True  →  split control logic across dedicated "
                "part defs — do NOT centralise into a single monolithic block"
            )
        elif distributed is False:
            lines.append(
                "  • distributed_control=False →  use a single centralised controller "
                "part def that owns all decision logic"
            )

        # Sensor count → part defs / part usages
        num_sensors = params.get("num_sensors")
        if num_sensors is not None:
            lines.append(
                f"  • num_sensors={num_sensors}              →  include exactly "
                f"{num_sensors} sensor-related part def(s) or part usage(s)"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Always-visible iteration output
    # ------------------------------------------------------------------

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
            sysml = (
                (getattr(current, "metadata", None) or {}).get("last_sysml_text")
                or current.to_sysml_text()
            )
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
        final_sysml = (
            (getattr(current, "metadata", None) or {}).get("last_sysml_text")
            or current.to_sysml_text()
        )
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
        sysml = (
            (getattr(model, "metadata", None) or {}).get("last_sysml_text")
            or model.to_sysml_text()
        )

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

