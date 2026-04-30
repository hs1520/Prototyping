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
        # Programmatically inject any attributes that MCTS decided to add but
        # which don't yet exist in the model object — directly into the stored
        # SysML text so the refinement LLM receives a complete starting point.
        self._apply_inject_attrs_to_sysml_text(model)
        # Programmatically replace generic DataPort / RFPort annotations with
        # the MCTS-selected protocol signal type in the stored SysML text.
        # This ensures the export is consistent even when refinement is skipped
        # (e.g. the initial model already scores above the quality threshold).
        self._apply_inject_protocol_to_sysml_text(model, best_config)
        # Layer 1: programmatically inject extra sensor part usages so that the
        # MCTS-selected num_sensors count is reflected in the assembly section
        # even when the refinement loop is skipped.
        self._apply_inject_sensor_count_to_sysml_text(model, best_config)
        # Layer 2: dedicated (quality-gate-bypassing) LLM call that adds the
        # hardware-redundancy structure (TMR channels, voting logic, failsafe
        # state) mandated by the MCTS redundancy_level decision.
        self._mcts_structural_grounding_pass(model, best_config)
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
            # Prefer the original LLM-generated text (with state defs, port def
            # bodies, and MCTS-injected attrs intact) over the Syside round-trip
            # reconstruction, which loses state defs and port def content.
            "model_sysml": (
                (getattr(final_model, "metadata", None) or {}).get("last_sysml_text")
                or final_model.to_sysml_text()
            ),
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
        connect_re = re.compile(
            r"\bconnect\s+(\w+)::(\w+)\s+to\s+(\w+)::(\w+)\s*;",
            re.IGNORECASE,
        )
        occupied_targets: set = set()
        primary_out_ports: list = []
        if primary_instance:
            for cm in connect_re.finditer(sysml_text):
                src_inst, src_port, tgt_inst, tgt_port = cm.groups()
                tgt_key = f"{tgt_inst}::{tgt_port}"
                occupied_targets.add(tgt_key)
                if src_inst.lower() == primary_instance.lower():
                    primary_out_ports.append((src_port, tgt_inst, tgt_port))

        # ── Build additional part usages + safe connect stubs ─────────────────
        new_lines: list = []
        new_connects: list = []
        for i in range(existing + 1, target + 1):
            unit_name = f"sensorUnit{i}"
            new_lines.append(f"    part {unit_name} : {sensor_part_name};")

            # For each out-port the primary sensor exposes, try to add a connect
            # for this redundant instance.  Skip if the same target is already
            # occupied (fan-in guard) — those cases need a voting intermediary
            # that is beyond the scope of programmatic injection.
            for src_port, tgt_inst, tgt_port in primary_out_ports:
                tgt_key = f"{tgt_inst}::{tgt_port}"
                if tgt_key not in occupied_targets:
                    new_connects.append(
                        f"    connect {unit_name}::{src_port} to {tgt_inst}::{tgt_port};"
                    )
                    occupied_targets.add(tgt_key)  # mark so next unit doesn't reuse it
                # else: fan-in risk — omit; a TMR voter/aggregator is needed

        # Inject part usages after the last existing part usage
        usage_block = "\n" + "\n".join(new_lines)
        result = sysml_text[:last_usage_end] + usage_block + sysml_text[last_usage_end:]

        # Inject any safe connect stubs just before the closing `}` of the package
        if new_connects:
            connect_block = (
                "\n    // MCTS-injected redundant sensor connects (fan-in-safe only):\n"
                + "\n".join(new_connects)
                + "\n"
            )
            # Insert before the very last `}` in the file (package close)
            last_brace = result.rfind("}")
            if last_brace != -1:
                result = result[:last_brace] + connect_block + result[last_brace:]

        model.metadata["last_sysml_text"] = result
        model.metadata["mcts_injected_sensor_units"] = target - existing

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
            # Pass mcts_best_config so Dim-2 (mcts_fidelity) can verify that
            # all MCTS architectural decisions are present in the SysML text.
            eval_result = self.evaluator.evaluate(
                config=DesignConfiguration(
                    name=f"iteration_{iteration}",
                    parameters={},
                ),
                model=current_model,
                mcts_config=mcts_best_config,
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

