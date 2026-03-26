"""
Multi-agent Orchestrator for MBSE prototyping.

Coordinates the different specialized agents to implement the
full rapid prototyping pipeline, combining forward and backward
inference with design space exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_agent import AgentMessage, AgentResult, BaseAgent
from .design_agent import DesignAgent
from .requirements_agent import RequirementsAgent
from ..dse.design_space import DesignConfiguration, DesignParameter, DesignSpace, ParameterType
from ..dse.evaluator import DesignEvaluator
from ..dse.mcts import MCTSDesignExplorer
from ..llm.chain_of_thought import ChainOfThoughtPrompter
from ..llm.interface import LLMInterface
from ..rag.retriever import RAGRetriever
from ..sysml.model import SysMLModel


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
    ):
        self.llm = llm
        self.rag = rag_retriever
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations

        # Initialize specialized agents
        self.requirements_agent = RequirementsAgent(llm, rag_retriever)
        self.design_agent = DesignAgent(llm, rag_retriever)
        self.evaluator = DesignEvaluator()
        self.cot = ChainOfThoughtPrompter(llm)

        self.state: Optional[PrototypingState] = None

    def prototype(
        self,
        system_name: str,
        system_description: str,
        additional_requirements: Optional[List[str]] = None,
        mcts_iterations: int = 50,
    ) -> Dict[str, Any]:
        """
        Run the complete prototyping pipeline.

        Returns a dictionary with the final model, requirements, evaluation,
        and design space exploration results.
        """
        self.state = PrototypingState(
            system_name=system_name,
            system_description=system_description,
        )

        print(f"\n{'='*60}")
        print(f"Starting AI-assisted prototyping for: {system_name}")
        print(f"{'='*60}\n")

        # Phase 1: Requirements Extraction
        print("Phase 1: Requirements Extraction")
        print("-" * 40)
        requirements = self._extract_requirements(
            system_description, additional_requirements or []
        )
        self.state.requirements = requirements
        print(f"  ✓ Extracted {len(requirements)} requirements\n")

        # Phase 2: Initial Design Generation
        print("Phase 2: Initial Design Generation")
        print("-" * 40)
        model = self._generate_initial_design(system_name, requirements)
        self.state.current_model = model
        print(f"  ✓ Generated model with {len(model.blocks)} blocks\n")

        # Phase 3: Design Space Exploration
        print("Phase 3: Design Space Exploration (MCTS)")
        print("-" * 40)
        design_space = self._explore_design_space(model, mcts_iterations)
        self.state.design_space = design_space
        summary = design_space.get_summary()
        print(f"  ✓ Explored {summary['configurations_evaluated']} configurations")
        print(f"  ✓ Pareto front size: {summary['pareto_front_size']}\n")

        # Phase 4-5: Iterative Refinement
        print("Phase 4-5: Iterative Evaluation and Refinement")
        print("-" * 40)
        final_model, final_score = self._iterative_refinement(model, requirements)
        self.state.current_model = final_model
        print(f"  ✓ Final design score: {final_score:.3f}\n")

        # Compile final results
        result = {
            "system_name": system_name,
            "requirements": requirements,
            "model": final_model,
            "model_sysml": final_model.to_sysml_text(),
            "model_summary": final_model.get_summary(),
            "design_space_summary": design_space.get_summary(),
            "final_score": final_score,
            "iterations": self.state.iteration,
            "evaluation_history": self.state.evaluation_history,
        }

        print(f"{'='*60}")
        print("Prototyping Complete!")
        print(f"  Final score: {final_score:.3f}")
        print(f"  Blocks: {len(final_model.blocks)}")
        print(f"  Requirements: {len(requirements)}")
        print(f"{'='*60}\n")

        return result

    def _extract_requirements(
        self,
        description: str,
        additional: List[str],
    ) -> List[str]:
        """Phase 1: Extract requirements using the RequirementsAgent."""
        result = self.requirements_agent.run({
            "system_description": description,
        })
        requirements = result.output if result.success else []

        # Add any manually provided requirements
        requirements.extend(additional)

        # Validate
        validation = self.requirements_agent.validate_requirements(requirements)
        if validation["warnings"]:
            for warning in validation["warnings"][:3]:  # Show max 3 warnings
                print(f"  ⚠ {warning}")

        return requirements

    def _generate_initial_design(
        self,
        system_name: str,
        requirements: List[str],
    ) -> SysMLModel:
        """Phase 2: Generate initial SysML v2 design."""
        result = self.design_agent.run({
            "system_name": system_name,
            "requirements": requirements,
        })
        if result.success and isinstance(result.output, SysMLModel):
            model = result.output
        else:
            # Fallback: create minimal model
            model = SysMLModel(
                name=system_name,
                description=f"Prototype design for {system_name}",
            )
        # Add requirements to model
        self.requirements_agent.create_sysml_requirements(requirements, model)
        return model

    def _explore_design_space(
        self,
        model: SysMLModel,
        mcts_iterations: int,
    ) -> DesignSpace:
        """Phase 3: Define and explore the design space using MCTS."""
        design_space = self._define_design_space(model)

        def evaluate_config(config: DesignConfiguration) -> Dict[str, float]:
            return self.evaluator.simple_score(config)

        explorer = MCTSDesignExplorer(
            design_space=design_space,
            evaluation_function=evaluate_config,
            max_depth=4,
            random_seed=42,
        )
        best_config = explorer.search(num_iterations=mcts_iterations)
        return design_space

    def _define_design_space(self, model: SysMLModel) -> DesignSpace:
        """Define the design space based on the model's structure."""
        space = DesignSpace(name=f"{model.name}_DesignSpace")

        # Add architectural choice parameters
        space.add_parameter(DesignParameter(
            name="redundancy_level",
            param_type=ParameterType.CATEGORICAL,
            default_value="none",
            choices=["none", "dual", "triple"],
            description="Level of hardware redundancy",
        ))
        space.add_parameter(DesignParameter(
            name="communication_protocol",
            param_type=ParameterType.CATEGORICAL,
            default_value="CAN",
            choices=["CAN", "Ethernet", "SPI", "I2C", "wireless"],
            description="Communication protocol between components",
        ))
        space.add_parameter(DesignParameter(
            name="control_frequency_hz",
            param_type=ParameterType.CONTINUOUS,
            default_value=100.0,
            min_value=10.0,
            max_value=1000.0,
            unit="Hz",
            description="Main control loop frequency",
        ))
        space.add_parameter(DesignParameter(
            name="distributed_control",
            param_type=ParameterType.BOOLEAN,
            default_value=False,
            description="Whether to use distributed vs centralized control",
        ))
        space.add_parameter(DesignParameter(
            name="num_sensors",
            param_type=ParameterType.DISCRETE,
            default_value=3,
            choices=[1, 2, 3, 4, 5, 6],
            description="Number of sensor units",
        ))

        return space

    def _iterative_refinement(
        self,
        model: SysMLModel,
        requirements: List[str],
    ) -> tuple[SysMLModel, float]:
        """Phase 4-5: Evaluate and iteratively refine the design."""
        current_model = model
        best_score = 0.0

        for iteration in range(self.max_iterations):
            self.state.iteration = iteration + 1

            # Evaluate current design
            eval_result = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}",
                    parameters={},
                ),
                model=current_model,
            )
            score = eval_result.weighted_total
            self.state.evaluation_history.append({
                "iteration": iteration + 1,
                "score": score,
                "issues": eval_result.issues,
            })

            print(f"  Iteration {iteration+1}: score={score:.3f}")

            if score > best_score:
                best_score = score
                best_model = current_model

            if score >= self.quality_threshold:
                print(f"  ✓ Quality threshold {self.quality_threshold} reached")
                return best_model, best_score

            # Also get LLM evaluation
            cot_eval = self.cot.evaluate_design(
                model_text=current_model.to_sysml_text(),
                requirements=requirements,
            )

            if eval_result.issues:
                # Refine based on issues
                refine_result = self.design_agent.run({
                    "system_name": current_model.name,
                    "requirements": requirements,
                    "existing_model": current_model,
                    "refinement_feedback": cot_eval.final_answer,
                })
                if refine_result.success and isinstance(refine_result.output, SysMLModel):
                    current_model = refine_result.output

        return best_model if best_score > 0 else current_model, best_score
