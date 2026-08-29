"""
Design Agent for MBSE prototyping.

Specializes in generating and refining SysML v2 design models
from requirements using Chain of Thought prompting and RAG.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .base_agent import AgentResult, BaseAgent
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.lite_model import build_lite_model
from .typed_plan_generation import (
    DEFAULT_MAXIMUM_PLAN_ATTEMPTS,
    TypedPlanGeneration,
    TypedPlanRequest,
)
from .model_authoring import AuthoringRequest, ModelAuthoring
from .generated_model_admission import (
    GeneratedModelAdmission,
    ModelAdmissionRequest,
)
from .refinement_authoring import RefinementAuthoring, RefinementRequest


#: Initial-generation modes. "multistep" is the production 5-step pipeline;
#: "single_shot" is the pre-multistep legacy path (one prompt → whole model),
#: retained as an ablation arm so the contribution of structured decomposition
#: is measured rather than assumed.
GENERATION_MODES = ("multistep", "single_shot")


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


    def __init__(
        self,
        llm: LLMInterface,
        rag_retriever: Optional[RAGRetriever] = None,
        *,
        allow_legacy_architecture_plan: bool = False,
        maximum_plan_attempts: int = DEFAULT_MAXIMUM_PLAN_ATTEMPTS,
        generation_mode: str = "multistep",
    ):
        super().__init__("DesignAgent", llm, rag_retriever)
        self.cot = ChainOfThoughtPrompter(llm)
        self.allow_legacy_architecture_plan = bool(
            allow_legacy_architecture_plan
        )
        if int(maximum_plan_attempts) < 1:
            raise ValueError("maximum_plan_attempts must be at least 1")
        self.maximum_plan_attempts = int(maximum_plan_attempts)
        if generation_mode not in GENERATION_MODES:
            raise ValueError(
                f"unknown generation_mode {generation_mode!r}; "
                f"must be one of {GENERATION_MODES}"
            )
        self.generation_mode = generation_mode

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

        refinement = RefinementAuthoring(
            self.cot,
            lambda query: self.get_augmented_context(
                query,
                include_official_sysml=True,
                allowed_extensions=(".sysml",),
            ),
        )
        if is_refinement:
            cot_result = refinement.refine(RefinementRequest(
                existing_model=existing_model,
                feedback=feedback,
                issues=refinement_issues,
                skip_rag=skip_rag,
                verbose=verbose,
            )).response
            generation_metadata = {}
        elif self.generation_mode == "single_shot":
            # Ablation arm — one prompt produces the whole model. No typed plan
            # exists, so plan-conformance / structural / semantic-fidelity gates
            # downstream are inapplicable (they record None, not PASS).
            cot_result, generation_metadata = self._single_shot_generate(
                system_name=system_name,
                requirements=requirements,
                context=context,
                verbose=verbose,
            )
        else:
            # Generation mode — multi-step pipeline. Each SysML-authoring step
            # issues its own request; typed planning owns its separate bounded
            # correction protocol.
            cot_result, generation_metadata = self._multistep_generate(
                system_name=system_name,
                requirements=requirements,
                context=context,
                verbose=verbose,
                platform_profile=task.get("platform_profile"),
                semantic_guidance_by_step=task.get(
                    "semantic_guidance_by_step"
                ),
                ag_behavior_obligation_plan=task.get(
                    "ag_behavior_obligation_plan"
                ),
                allow_legacy_architecture_plan=bool(
                    task.get(
                        "allow_legacy_architecture_plan",
                        self.allow_legacy_architecture_plan,
                    )
                ),
            )

        admitted = GeneratedModelAdmission(
            build_lite_model,
            refinement,
        ).accept(ModelAdmissionRequest(
            response=cot_result,
            system_name=system_name,
            requirements=requirements,
            generation_metadata=generation_metadata,
            is_refinement=is_refinement,
            verbose=verbose,
        ))
        cot_result = admitted.response
        model = admitted.model
        generation_metadata = dict(admitted.metadata)
        parse_diagnostics = list(admitted.parse_diagnostics)
        untraced = list(admitted.untraced_requirements)

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


    def _single_shot_generate(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        verbose: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Single-prompt whole-model generation (the pre-multistep legacy path).

        Reuses ``ChainOfThoughtPrompter.generate_design`` verbatim so the
        ablation compares against the pipeline's own historical single-shot
        behaviour, not a prompt written for the experiment.  Admission, the
        syntax gate, refinement, and evaluation downstream are unchanged.
        """
        if verbose:
            print("\n  [DEBUG] single-shot generation — one prompt, no typed plan")
        response = self.cot.generate_design(
            system_name=system_name,
            requirements=requirements,
            context=context,
        )
        metadata: Dict[str, Any] = {
            "generation_mode": "single_shot",
            "generation_steps_completed": 1,
            "degraded_steps": [],
            "total_thought_steps": len(response.thought_steps),
        }
        return response, metadata

    def _multistep_generate(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        verbose: bool = False,
        platform_profile=None,
        semantic_guidance_by_step: Optional[Mapping[str, str]] = None,
        ag_behavior_obligation_plan: Optional[Mapping[str, Any]] = None,
        allow_legacy_architecture_plan: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        5-step generation pipeline:
          1. Architecture Decomposition  (typed whole-model JSON plan)
          2. Part Definitions            (SysML structural fragment)
          3. Interface Definitions       (SysML interface fragment)
          4. Behavioral Model            (SysML behavioral fragment)
          5. Integration / Assembly      (complete SysML package)

        Each step issues its own RAG query targeting the SysML constructs most
        relevant to that step, rather than sharing one generic upfront query.

        Returns (final_CoTResult, generation_metadata).
        final_CoTResult.extracted_sysml is the assembled model.
        generation_metadata carries per-step diagnostics.
        """
        metadata: Dict[str, Any] = {
            "generation_mode": "multistep",
            "generation_steps_completed": 0,
            "degraded_steps": [],
        }
        guidance = {
            str(key): str(value)
            for key, value in dict(semantic_guidance_by_step or {}).items()
            if str(value).strip()
        }
        from ..prototyping.ag_behavior_plan import BehaviorObligationPlan

        behavior_plan = (
            BehaviorObligationPlan.from_dict(ag_behavior_obligation_plan)
            if ag_behavior_obligation_plan is not None else None
        )

        # --- Step 1: Architecture Decomposition ---
        typed_plan = TypedPlanGeneration(
            self.cot,
            maximum_attempts=self.maximum_plan_attempts,
        ).generate(TypedPlanRequest(
            system_name=system_name,
            requirements=requirements,
            context=context,
            semantic_guidance=guidance.get("architecture", ""),
            behavior_plan=behavior_plan,
            allow_legacy_plan=allow_legacy_architecture_plan,
            verbose=verbose,
        ))
        metadata.update(typed_plan.metadata)
        step1 = typed_plan.response
        architecture_text = typed_plan.architecture_text
        generation_plan = typed_plan.plan

        authoring = ModelAuthoring(
            self.cot,
            lambda query: self.get_augmented_context(
                query,
                include_official_sysml=True,
                allowed_extensions=(".sysml",),
            ),
        ).generate(AuthoringRequest(
            system_name=system_name,
            architecture_text=architecture_text,
            requirements=requirements,
            generation_plan=generation_plan,
            base_context=context,
            semantic_guidance=guidance,
            platform_profile=platform_profile,
            behavior_plan=behavior_plan,
            verbose=verbose,
        ))
        authoring_metadata = dict(authoring.metadata)
        authoring_degraded = list(
            authoring_metadata.pop("degraded_steps", ())
        )
        metadata.update(authoring_metadata)
        metadata["degraded_steps"].extend(authoring_degraded)
        metadata["generation_steps_completed"] = 5
        metadata["total_thought_steps"] = (
            len(step1.thought_steps)
            + authoring.thought_steps
        )
        return authoring.response, metadata
