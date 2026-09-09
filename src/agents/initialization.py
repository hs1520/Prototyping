"""InitializationMixin extracted from the orchestrator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from .design_agent import DesignAgent
from .typed_plan_generation import DEFAULT_MAXIMUM_PLAN_ATTEMPTS
from .orchestrator_support import PrototypingState
from .requirements_agent import RequirementsAgent
from .refinement import RefinementClosure
from ..dse.evaluator import DesignEvaluator
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..simulation.validator import SimulationValidator


class InitializationMixin:
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
        use_deterministic_fixers: bool = True,
        design_generation_mode: str = "multistep",
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
        enable_plan_revision: bool = True,
    ):
        self._init_pipeline_state()
        self.llm = llm
        # Bounded plan revision when refinement reports
        # `structural_repair_blocked` (see agents/plan_revision.py). An
        # ablation arm may disable it to reproduce the frozen-plan deadlock.
        self.enable_plan_revision = enable_plan_revision
        self.rag = rag_retriever
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.rule_weight = rule_weight
        self.llm_weight = llm_weight
        self.verbose = verbose
        self.use_variation_dse = use_variation_dse
        self.realization_inject = realization_inject
        self.last_recommended_design = None
        self.last_pareto_designs = []
        self.last_recommended_bindings = {}
        self.last_recommended_realizable = None
        self.last_realizable_front_count = None
        self.last_recommended_by = None
        self.last_weight_sensitivity = None
        self.last_recommended_estimator_feasible = None
        self.last_recommendation_status = None
        self.last_recommendable_front_count = None
        self.last_exploratory_design = None
        self.last_exploratory_pareto_alternatives = []
        self.last_constraint_counts = {}
        self.last_search_coverage = {}
        self.last_variation_proposal_source = None
        self.last_functional_closure = None
        self.last_verification_anchor_attempts: List[Dict[str, Any]] = []
        self.last_namespace_repair_attempts: List[Dict[str, Any]] = []
        self.last_unplanned_connect_removal_attempts: List[Dict[str, Any]] = []
        self.last_response_conformance_repair_attempts: List[Dict[str, Any]] = []
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
        # F1: catalog-grid estimator calibration, applied only around the search
        # (the injected SysML calc defs and Phase 8's estimator_value column keep
        # the documented textbook constants). Provenance is recorded.
        self.use_estimator_calibration = estimator_calibration
        self.last_estimator_calibration = None
        # Phase 9 is default-off.  A published authority requires the dedicated
        # realization driver, which freezes the Phase 8 result and then runs
        # Gazebo -> SITL in one locked run bundle.  Explicit modes are a
        # best-effort developer seam and do not upgrade datasheet CLOSED.
        self.phase9_hifi = phase9_hifi
        # Refinement asks the LLM for block-level replacements first (surgical:
        # untouched blocks cannot lose connects, output ~10x smaller) and only
        # falls back to the legacy whole-model rewrite when no valid merge is
        # produced. Disable to force the legacy path (tests / A-B comparison).
        self.use_surgical_refinement = use_surgical_refinement
        # Deterministic, LLM-free fixes (Tier-0 syntax fixes; connectivity
        # direction-widening and missing-connect injection) run before any LLM
        # repair. Disable to force LLM-only repair - the ablation that measures
        # what the fixer chain saves in calls, failures, and score.
        self.use_deterministic_fixers = use_deterministic_fixers

        self.requirements_agent = RequirementsAgent(llm, rag_retriever)
        self.design_agent = DesignAgent(
            llm,
            rag_retriever,
            maximum_plan_attempts=maximum_plan_attempts,
            generation_mode=design_generation_mode,
        )
        # Pass quality_threshold so the evaluator's veto cap is consistent with
        # the orchestrator's refinement gate (cap = threshold − 0.05).
        self.evaluator = DesignEvaluator(quality_threshold=quality_threshold)
        self.cot = ChainOfThoughtPrompter(llm)
        self.sim_validator = SimulationValidator()
        self.refinement_closure = RefinementClosure(self)

        self.state: Optional[PrototypingState] = None
