"""
Multi-agent Orchestrator for MBSE prototyping.

Coordinates the different specialized agents to implement the
full rapid prototyping pipeline, combining forward and backward
inference with design space exploration.
"""

from __future__ import annotations

import re
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
        self.evaluator = DesignEvaluator()
        self.cot = ChainOfThoughtPrompter(llm)

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
            system_name, system_description, additional_requirements or []
        )
        self.state.requirements = requirements
        print(f"  ✓ Extracted {len(requirements)} requirements\n")

        # Phase 2: Initial Design Generation
        print("Phase 2: Initial Design Generation")
        print("-" * 40)
        model = self._generate_initial_design(system_name, requirements, parse_strict=parse_strict)
        self.state.current_model = model
        print(f"  ✓ Generated model with {len(model.part_definitions)} part definitions\n")

        # Phase 3: Design Space Exploration
        print("Phase 3: Design Space Exploration (MCTS)")
        print("-" * 40)
        design_space, best_config, pareto_front = self._explore_design_space(
            model, mcts_iterations, requirements,
            random_seed=mcts_seed,
            patience=mcts_patience,
        )
        self.state.design_space = design_space
        # 方案 C: apply winning configuration back to the model
        self._apply_best_config_to_model(best_config, model)
        self._print_exploration_summary(design_space, best_config, pareto_front)

        # Phase 4-5: Iterative Refinement
        print("Phase 4-5: Iterative Evaluation and Refinement")
        print("-" * 40)
        final_model, final_score = self._iterative_refinement(
            model, requirements, mcts_best_config=best_config
        )
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
            "design_space_parameters": [
                {"name": p.name, "type": p.param_type.value,
                 "choices": p.choices, "description": p.description}
                for p in design_space.parameters
            ],
            "best_config": best_config.parameters,
            "pareto_alternatives": [
                {
                    "name": c.name,
                    "parameters": dict(c.parameters),
                    "scores": dict(c.scores),
                    "overall_score": round(c.overall_score, 4),
                }
                for c in pareto_front
            ],
            "final_score": final_score,
            "iterations": self.state.iteration,
            "evaluation_history": self.state.evaluation_history,
        }

        print(f"{'='*60}")
        print("Prototyping Complete!")
        print(f"  Final score: {final_score:.3f}")
        print(f"  Part definitions: {len(final_model.part_definitions)}")
        print(f"  Requirements: {len(requirements)}")
        print(f"{'='*60}\n")

        return result

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

        llm_requirements = result.output if result.output else []

        # Merge: manual requirements take precedence; duplicates (by ID or text) are dropped;
        # same-ID-different-content LLM requirements are reassigned to a new ID with a warning.
        requirements, merge_conflicts = self.requirements_agent.merge_requirements(llm_requirements, additional)
        for warning in merge_conflicts:
            print(f"  ⚠ {warning}")

        # Validate merged set
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
    ) -> SysMLModel:
        """Phase 2: Generate initial SysML v2 design."""
        task = {
            "system_name": system_name,
            "requirements": requirements,
            "parse_strict": (parse_strict if parse_strict is not None else False),
            "verbose": self.verbose,
        }
        result = self.design_agent.run(task)
        if result.success and isinstance(result.output, SysMLModel):
            model = result.output
        else:
            model = SysMLModel(
                name=system_name,
                description=f"Prototype design for {system_name}",
            )

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
        space.add_parameter(DesignParameter(
            name="redundancy_level",
            param_type=ParameterType.CATEGORICAL,
            default_value="none",
            choices=redundancy_choices,
            description=f"Hardware redundancy level (derived from {safe_count} SAFE requirement(s))",
        ))

        # ── Communication protocol (INTF-driven) ──────────────────────
        intf_text = " ".join(r for r in requirements if "-INTF-" in r).upper()
        known_protocols = ["CAN", "ETHERNET", "SPI", "I2C", "MAVLINK",
                           "ROS2", "WIRELESS", "MODBUS", "PROFINET"]
        detected = [p for p in known_protocols if p in intf_text]
        # Always offer at least two options so MCTS has something to explore
        protocol_choices = detected if len(detected) >= 2 else (detected + ["CAN", "Ethernet"])[:4]
        # Normalise to mixed-case display names
        _display = {"ETHERNET": "Ethernet", "MAVLINK": "MAVLink", "ROS2": "ROS2",
                    "WIRELESS": "wireless", "MODBUS": "Modbus", "PROFINET": "PROFINET"}
        protocol_choices = [_display.get(p, p) for p in dict.fromkeys(protocol_choices)]
        default_protocol = protocol_choices[0] if detected else "CAN"
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
        space.add_parameter(DesignParameter(
            name="num_sensors",
            param_type=ParameterType.DISCRETE,
            default_value=max(1, existing_sensors) if existing_sensors else 2,
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
            scores["safety_margin"] = min(1.0, (redundancy + 0.1) / (needed + 0.1))

        # ── Protocol match ─────────────────────────────────────────────
        intf_reqs = [r for r in requirements if "-INTF-" in r]
        protocol = str(config.parameters.get("communication_protocol", "")).upper()
        if intf_reqs:
            intf_text = " ".join(intf_reqs).upper()
            scores["protocol_match"] = 1.0 if (protocol and protocol in intf_text) else 0.4
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

        # 1. Numeric attribute values — update frequency / rate attributes
        freq = params.get("control_frequency_hz")
        if freq is not None:
            freq_str = str(round(float(freq), 2))
            _freq_kws = {"frequency", "freq", "rate", "hz", "sample"}
            for part in model.part_definitions:
                for attr in part.attributes:
                    if any(kw in attr.name.lower() for kw in _freq_kws):
                        attr.default_value = freq_str

        # 2. Redundancy level → doc annotation on model description
        redundancy = str(params.get("redundancy_level", "none"))
        if redundancy != "none":
            tag = f"redundancy={redundancy}"
            if model.description and tag not in model.description:
                model.description = f"{model.description} [{tag}]"
            elif not model.description:
                model.description = f"[{tag}]"

        # 3. Communication protocol → default type on undeclared ports
        protocol = params.get("communication_protocol")
        if protocol:
            protocol_type_name = f"{protocol}Signal"
            for part in model.part_definitions:
                for port in part.ports:
                    if port.type_ref is None or not port.type_ref.name:
                        port.type_ref = ElementRef(name=protocol_type_name)

        # 4. Recommended sensor count → model-level metadata
        num_sensors = params.get("num_sensors")
        if num_sensors is not None:
            if not hasattr(model, "metadata") or model.metadata is None:
                object.__setattr__(model, "metadata", {})
            model.metadata["recommended_sensor_count"] = int(num_sensors)
            model.metadata["mcts_best_config"] = best_config.name

    def _iterative_refinement(
        self,
        model: SysMLModel,
        requirements: List[str],
        mcts_best_config: Optional[DesignConfiguration] = None,
    ) -> tuple[SysMLModel, float]:
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

            # ── Rule-based evaluation ─────────────────────────────────────
            eval_result = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}",
                    parameters={},
                ),
                model=current_model,
            )
            rule_score = eval_result.weighted_total

            # ── LLM evaluation (skip when rule score already sufficient) ──
            # Avoids an expensive API call when it can't change the outcome.
            if rule_score >= self.quality_threshold:
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
            })

            print(
                f"  Iteration {iteration+1}: score={score:.3f} "
                f"(rule={rule_score:.3f}, "
                f"llm={llm_overall if llm_overall is not None else 'N/A'})"
            )

            # ── P0: Best-model tracking ───────────────────────────────────
            if score > best_score:
                best_score = score
                best_model = current_model

            # ── Early exit ────────────────────────────────────────────────
            if score >= self.quality_threshold:
                print(f"  ✓ Quality threshold {self.quality_threshold} reached")
                return current_model, score

            # ── P1: Persistent issue tracking ─────────────────────────────
            for issue in eval_result.issues:
                seen_issues[issue] = seen_issues.get(issue, 0) + 1
            persistent = [iss for iss, cnt in seen_issues.items() if cnt > 1]

            # ── P1: Refinement trigger ────────────────────────────────────
            # Fire when explicit issues exist OR when LLM provided substantive
            # feedback (covers the empty-issues / below-threshold case).
            has_issues = bool(eval_result.issues)
            has_llm_feedback = cot_eval is not None and bool(cot_eval.final_answer)

            if self.verbose and eval_result.issues:
                print(f"  [DEBUG] Rule issues ({len(eval_result.issues)}):")
                for iss in eval_result.issues:
                    marker = "  [PERSISTENT]" if iss in persistent else ""
                    print(f"    • {iss}{marker}")
            if self.verbose and eval_result.recommendations:
                print(f"  [DEBUG] Recommendations ({len(eval_result.recommendations)}):")
                for rec in eval_result.recommendations:
                    print(f"    → {rec}")
            if self.verbose and cot_eval:
                cot_scores = cot_eval.get_scores() or {}
                if cot_scores:
                    print(f"  [DEBUG] LLM sub-scores: "
                          + ", ".join(f"{k}={v:.2f}" for k, v in cot_scores.items()))

            if has_issues or has_llm_feedback:
                refinement_feedback = self._build_refinement_feedback(
                    eval_result,
                    cot_eval.final_answer if cot_eval else "",
                    persistent_issues=persistent,
                    mcts_constraints=mcts_constraints,
                )
                refine_result = self.design_agent.run({
                    "system_name": current_model.name,
                    "requirements": requirements,
                    "existing_model": current_model,
                    "refinement_feedback": refinement_feedback,
                    "refinement_issues": eval_result.issues + eval_result.recommendations,
                    "verbose": self.verbose,
                })
                if refine_result.success and isinstance(refine_result.output, SysMLModel):
                    candidate = refine_result.output
                    # ── P0: Regression prevention ─────────────────────────
                    # Accept the candidate only if its rule-based score does
                    # not regress by more than 5 percentage points.
                    candidate_eval = self.evaluator.evaluate(
                        config=DesignConfiguration(name="candidate", parameters={}),
                        model=candidate,
                    )
                    if candidate_eval.weighted_total >= rule_score - 0.05:
                        if self.verbose:
                            print(
                                f"  [DEBUG] Regression check: "
                                f"rule {rule_score:.3f} → {candidate_eval.weighted_total:.3f}"
                                f"  ✓ accepted"
                            )
                        current_model = candidate
                    else:
                        print(
                            f"  ⚠ Refinement regression detected "
                            f"(rule: {rule_score:.3f} → {candidate_eval.weighted_total:.3f}), "
                            f"keeping current model"
                        )

        return best_model, best_score

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

        # Redundancy level → state def structure
        redundancy = str(params.get("redundancy_level", "none"))
        if redundancy == "triple":
            lines.append(
                "  • redundancy_level=triple  →  add a state def with three independent "
                "channels (states: nominal, fault1, fault2); transition to failsafe when "
                "≥2 channels report failure; include action def emergencyStop {}"
            )
        elif redundancy == "dual":
            lines.append(
                "  • redundancy_level=dual    →  add a state def with dual-channel "
                "redundancy (states: nominal, faultPrimary); transition to failsafe on "
                "primary channel failure; include action def emergencyStop {}"
            )

        # Control frequency → numeric attribute
        freq = params.get("control_frequency_hz")
        if freq is not None:
            lines.append(
                f"  • control_frequency_hz={freq}  →  the main controller part def must "
                f"have  attribute controlFrequency : Real = {freq} [Hz]"
            )

        # Communication protocol → port types
        protocol = params.get("communication_protocol")
        if protocol and str(protocol).lower() not in ("none", ""):
            lines.append(
                f"  • communication_protocol={protocol}  →  all inter-component ports "
                f"must carry type  {protocol}Signal  "
                f"(e.g.  port commandIn : {protocol}Signal {{ in; }})"
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

    @staticmethod
    def _build_refinement_feedback(
        eval_result: Any,
        cot_feedback: str,
        persistent_issues: Optional[List[str]] = None,
        mcts_constraints: str = "",
    ) -> str:
        """Combine evaluator issues with LLM feedback into a refinement-oriented summary.

        Args:
            eval_result:        Rule-based evaluation result (issues + recommendations).
            cot_feedback:       LLM chain-of-thought final answer (may be empty).
            persistent_issues:  Issues that have appeared in more than one iteration;
                                these are flagged with [PERSISTENT] so the LLM can
                                prioritise them.
            mcts_constraints:   Architectural decisions from MCTS that the LLM must
                                implement in the SysML model (prepended so the LLM
                                sees them before anything else).
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

