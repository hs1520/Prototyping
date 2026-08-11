"""Board-driven knowledge sources for the complete generation pipeline."""
from __future__ import annotations

from typing import Any, Callable, Mapping
from ..prototyping.action_effects import LEGACY_AUDIT, parse_action_effects
from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import get_sysml_text
from .pipeline_records import GenerationContext



_ORCH = "Orchestrator"
_ASSUR = "AssuranceAgent"

class GenerationPipelineMixin:
    def _publish_phase(
        self,
        context: GenerationContext,
        topic: str,
        phase: Callable[[GenerationContext], Any],
    ) -> Any:
        from ..prototyping.blackboard import RecordType

        result = phase(context)
        self._runtime_board.publish(
            RecordType.RESULT,
            topic,
            "PipelineKnowledgeSource",
            {"phase": topic, "status": "COMPLETED"},
            # A phase-completion record states that a step of the process ran,
            # not a property of the model it ran against. Committing a model
            # afterwards does not make it untrue, so it must not expire with the
            # revision -- otherwise every topic published before a commit would
            # vanish and the chain would stall halfway.
            revision_bound=False,
        )
        return result

    def _generation_sources(self, context: GenerationContext):
        from ..prototyping.controller import KnowledgeSource

        # The agent role is declared per source, not derived from the name. It
        # used to be `"Orchestrator" if "ag_" not in name else "AssuranceAgent"`,
        # which happened to be right for these twenty-one and is wrong for any
        # future name that merely contains those two characters. The role reaches
        # the run artefacts on every activation record, so a misclassification
        # would land in the evidence rather than staying in the code.
        phases = (
            ("requirements_input", _ORCH, ("pipeline.request",), "phase.requirements.ready", self._phase_requirements),
            ("collaboration_board", _ORCH, ("phase.requirements.ready",), "phase.board.ready", self._phase_board),
            ("ag_generation_planning", _ASSUR, ("phase.board.ready",), "phase.ag_plan.ready", self._phase_ag_plan),
            ("design_handoff", _ORCH, ("phase.ag_plan.ready",), "phase.design_handoff.ready", self._phase_design_handoff),
            ("initial_design_generation", _ORCH, ("phase.design_handoff.ready",), "phase.initial_design.ready", self._phase_initial_design),
            ("iterative_refinement", _ORCH, ("phase.initial_design.ready",), "phase.refinement.ready", self._phase_refinement),
            ("sitl_refinement", _ORCH, ("phase.refinement.ready",), "phase.sitl_refinement.ready", self._phase_sitl_refinement),
            ("functional_closure", _ORCH, ("phase.sitl_refinement.ready",), "phase.functional_closure.ready", self._phase_functional_closure),
            ("terminal_plan_enforcement", _ORCH, ("phase.functional_closure.ready",), "phase.terminal_plan.ready", self._phase_terminal_plan),
            ("pre_ag_simulation", _ASSUR, ("phase.terminal_plan.ready",), "phase.pre_ag_simulation.ready", self._phase_pre_ag_simulation),
            ("ag_contract_reconciliation", _ASSUR, ("phase.pre_ag_simulation.ready",), "phase.ag_reconciliation.ready", self._phase_ag_reconciliation),
            ("terminal_model_commit", _ORCH, ("phase.ag_reconciliation.ready",), "phase.terminal_commit.ready", self._phase_terminal_commit),
            ("verification_planning", _ORCH, ("phase.terminal_commit.ready",), "phase.verification.ready", self._phase_verification),
            ("ag_semantic_assurance", _ASSUR, ("phase.verification.ready",), "phase.assurance.ready", self._phase_assurance),
            ("collaboration_artifacts", _ORCH, ("phase.assurance.ready",), "phase.collaboration.ready", self._phase_collaboration_artifacts),
            ("terminal_snapshot_sync", _ORCH, ("phase.collaboration.ready",), "phase.terminal_snapshot.ready", self._phase_terminal_snapshot),
            ("ag_non_degradation", _ASSUR, ("phase.terminal_snapshot.ready",), "phase.ag_non_degradation.ready", self._phase_ag_non_degradation),
            ("structural_qualification", _ORCH, ("phase.ag_non_degradation.ready",), "phase.structural.ready", self._phase_structural),
            ("semantic_qualification", _ORCH, ("phase.structural.ready",), "phase.semantic.ready", self._phase_semantic),
            ("model_qualification", _ORCH, ("phase.semantic.ready",), "phase.qualification.ready", self._phase_qualification),
            ("generation_reporting", _ORCH, ("phase.qualification.ready",), "phase.generation.complete", self._phase_reporting),
        )
        sources = []
        for name, role, preconditions, output, phase in reversed(phases):
            sources.append(KnowledgeSource(
                name=name,
                agent_role=role,
                precondition_topics=preconditions,
                output_topics=(output,),
                activate=lambda p=phase, o=output: self._publish_phase(
                    context, o, p
                ),
            ))
        return sources

    def _phase_requirements(self, c: GenerationContext) -> None:
        print("Phase 1: Requirements Input")
        print("-" * 40)
        if c.frozen_requirements is not None:
            if c.additional_requirements:
                raise ValueError(
                    "frozen_requirements and additional_requirements are mutually "
                    "exclusive in a controlled run"
                )
            c.requirements = self._use_frozen_requirements(c.frozen_requirements)
            print(f"  ✓ Loaded {len(c.requirements)} frozen requirements")
        else:
            c.requirements = self._extract_requirements(
                c.system_name, c.system_description, c.additional_requirements
            )
            print(f"  ✓ Extracted {len(c.requirements)} requirements")
        self.state.requirements = c.requirements

    def _phase_board(self, c: GenerationContext) -> None:
        # Board first: A/G planning makes real LLM decisions, so it is an Agent
        # task and needs a typed task, an envelope, and an archived session like
        # every other one. Freezing the plan before the board existed left those
        # turns unrecorded, which §5.3 does not permit.
        self._open_collaboration_board(c.system_name, c.requirements)

    def _phase_ag_plan(self, c: GenerationContext) -> None:
        self._active_ag_generation_plan = self._prepare_ag_guided_generation(
            c.requirements
        )

    def _phase_design_handoff(self, c: GenerationContext) -> None:
        self._prepare_design_handoff(c.system_name, c.requirements)
        print()

    def _phase_initial_design(self, c: GenerationContext) -> None:
        print("Phase 2: Initial Design Generation")
        print("-" * 40)
        c.model = self._generate_initial_design(
            c.system_name, c.requirements,
            parse_strict=c.parse_strict,
            platform_profile=c.platform_profile,
        )
        self.state.current_model = c.model
        raw_model_plan = (getattr(c.model, "metadata", None) or {}).get(
            "whole_model_generation_plan"
        )
        if isinstance(raw_model_plan, Mapping):
            self._active_model_generation_plan = dict(raw_model_plan)
        print(
            f"  ✓ Generated model with {len(c.model.part_definitions)} "
            "part definitions\n"
        )

    def _phase_refinement(self, c: GenerationContext) -> None:
        print("Phase 3: Iterative Refinement")
        print("-" * 40)
        c.final_model, c.final_score, c.final_sim = self._iterative_refinement(
            c.model, c.requirements, dse_best_config=None
        )

    def _phase_sitl_refinement(self, c: GenerationContext) -> None:
        # ── Phase 3.5: SITL-L1 refinement (only when targeting a platform) ────
        # Feed unresolved ArduPilot-parameter mappings (= model genuinely
        # missing a guard/attribute a requirement needs) back to the design
        # LLM.  Cheap & deterministic (no SITL process launch); L2 stays
        # terminal.
        if c.platform_profile is not None:
            c.final_model, c.final_score, c.final_sim = self._sitl_refinement_loop(
                c.final_model, c.requirements, c.final_score, c.final_sim,
                max_iters=2,
            )

    def _phase_functional_closure(self, c: GenerationContext) -> None:
        # Terminal model mutation: runs after ordinary and optional SITL-L1
        # refinement so no later LLM rewrite can overwrite functional closure.
        c.final_model, c.final_score, c.final_sim = self._functional_closure_pass(
            c.final_model, c.final_sim, c.final_score, c.requirements,
            dse_best_config=None, max_iters=2,
        )
        c.pre_terminal_score = c.final_score

    def _phase_terminal_plan(self, c: GenerationContext) -> None:
        # First close the ordinary generated model, then snapshot it.  Terminal
        # A/G binding is evaluated as a separate transaction so it cannot hide
        # damage to the executable architecture behind an aggregate score.
        self._restore_generation_plan_metadata(c.final_model)
        c.pre_ag_sysml, c.generation_plan_conformance = (
            self._enforce_terminal_generation_plan(
                c.final_model, get_sysml_text(c.final_model)
            )
        )

    def _phase_pre_ag_simulation(self, c: GenerationContext) -> None:
        c.pre_ag_sim = self._run_simulation(c.pre_ag_sysml, c.system_name)

    def _phase_ag_reconciliation(self, c: GenerationContext) -> None:
        c.final_sysml = self._reconcile_guided_ag_contract_layer(
            c.pre_ag_sysml, c.requirements
        )

    #: Scoped to one chain while the profile is proven end to end.  Widening
    #: this is a deliberate act: every added chain changes more generated text.
    _ACTION_EFFECT_CHAINS = ("REQ_SAFE_005",)

    def _derive_action_effects(self) -> tuple:
        """The planned response chains, from the specs this run actually decided.

        `LLM_DECIDED_SPEC` re-decides every name, so the static chain library is
        a template and not the authority: the archived models call the arbiter's
        response `setParachuteDeploymentCommandAndParachuteResponseSelected`
        where the library says `setParachuteResponseSelectedAndIssue...`.
        Deriving from the library instead of the run's own specs resolves to
        nothing.
        """
        from ..prototyping.action_effects import derive_action_effects

        ag_plan = self._active_ag_generation_plan or {}
        specs = ag_plan.get("specs") or ()
        model_plan = self._active_model_generation_plan
        components = (
            model_plan.get("components") or ()
            if isinstance(model_plan, Mapping) else ()
        )
        connections = (
            model_plan.get("connections") or ()
            if isinstance(model_plan, Mapping) else ()
        )
        effects: list = []
        for spec in specs:
            if getattr(spec, "source_requirement", None) not in (
                self._ACTION_EFFECT_CHAINS
            ):
                continue
            effects.extend(
                derive_action_effects(spec, components, connections)
            )
        return tuple(effects)

    def _phase_terminal_commit(self, c: GenerationContext) -> None:
        self._materialize_action_effects(c)
        self._commit_terminal_model(
            c.final_sysml, producer="Orchestrator.generate"
        )
        self._ensure_terminal_ready()

    def _materialize_action_effects(self, c: GenerationContext) -> None:
        """Write the planned send and type edge, and keep it only if it holds.

        Applied here because every other writer of behaviour text has already
        run: `materialize_owned_behavior_obligations` re-renders a state machine
        with bare entry actions, and anything written before it is replaced.
        The result is kept only when the syntax check does not get worse, which
        is what bounds a rewrite that names elements the plan believes exist.
        """
        from ..prototyping.action_effects import materialize_action_effects

        self.last_action_effects = self._derive_action_effects()
        if not self.last_action_effects:
            return
        candidate, written = materialize_action_effects(
            c.final_sysml, self.last_action_effects
        )
        if not written or candidate == c.final_sysml:
            return
        before = check_syntax(c.final_sysml).total_errors()
        after = check_syntax(candidate).total_errors()
        if after > before:
            return
        c.final_sysml = candidate

    def _phase_verification(self, c: GenerationContext) -> None:
        c.verification_plan = self._run_verification_handoff()

    def _phase_assurance(self, c: GenerationContext) -> None:
        from ..prototyping.experiment_arms import RevisedExperimentArm

        if (
            self.revised_experiment_arm
            is RevisedExperimentArm.SEMANTIC_ASSURANCE
        ):
            c.assurance_artifacts = self._build_ag_trace(c.final_sysml)

    def _phase_collaboration_artifacts(self, c: GenerationContext) -> None:
        c.collaboration_artifacts = self._build_collaboration_artifacts(
            c.final_sysml
        )
        if c.verification_plan is not None:
            c.collaboration_artifacts["verification_plan"] = c.verification_plan
        if c.assurance_artifacts:
            c.collaboration_artifacts.update(c.assurance_artifacts)
            revised = c.collaboration_artifacts["revised_experiment"]
            revised.update({
                "runtime_assurance_status": (
                    "PASS"
                    if c.assurance_artifacts["ag_contract_graph"]["verdict"] == "PASS"
                    and c.assurance_artifacts["pattern_conformance_report"]["verdict"] == "PASS"
                    else "FAILED_OR_INCOMPLETE"
                ),
                "formal_ag_proof": False,
                "physical_verification": False,
            })
        c.final_sysml = c.collaboration_artifacts.pop(
            "_terminal_model_sysml", c.final_sysml
        )

    def _phase_terminal_snapshot(self, c: GenerationContext) -> None:
        self._verify_terminal_functional_closure(
            c.final_sysml, c.system_name
        )
        (
            c.final_model, c.final_score, c.final_sim, c.terminal_consistency
        ) = self._synchronize_terminal_snapshot(
            c.final_model, c.final_sysml, c.requirements,
            prior_score=c.pre_terminal_score, dse_best_config=None,
        )
        self._audit_action_semantics(c)

    def _audit_action_semantics(self, c: GenerationContext) -> None:
        """Record what the committed actions do.  Advisory: it changes no verdict.

        Run at the tail of the terminal snapshot rather than as its own
        knowledge source: the audit is read-only, and a new source would change
        the 21/5 phase-and-role counts that the pipeline's own tests pin and the
        write-up states.  A profile that gates on this evidence should add a
        real phase deliberately, and revise those counts with it.
        """
        from ..dse.functional_behavior import functional_behavior_diagnosis
        from ..prototyping.action_semantics import analyze_action_semantics

        plan = self._active_model_generation_plan
        effects = parse_action_effects(
            (plan or {}).get("action_effects") if isinstance(plan, Mapping)
            else None
        ) or getattr(self, "last_action_effects", ())
        report = analyze_action_semantics(
            c.final_sysml,
            requirements=c.requirements,
            action_effect_plan=effects,
            profile=getattr(self, "action_semantics_profile", LEGACY_AUDIT),
        )
        payload = report.to_dict()
        # Merged here rather than inside the audit: the strict reading lives in
        # `dse`, and having `prototyping` call it would close a package cycle.
        payload["functional_behaviour_diagnosis"] = functional_behavior_diagnosis(
            c.final_sysml, list(c.requirements or ())
        )
        self.last_action_semantics_audit = payload

    def _phase_ag_non_degradation(self, c: GenerationContext) -> None:
        self.last_ag_non_degradation = None
        if self._active_ag_generation_plan is not None:
            from ..prototyping.ag_quality_gate import build_ag_non_degradation_report
            self.last_ag_non_degradation = build_ag_non_degradation_report(
                c.pre_ag_sim, c.final_sim
            )

    def _phase_structural(self, c: GenerationContext) -> None:
        c.structural_obligation_report = self._validate_terminal_structural_obligations(
            c.final_model, c.final_sysml, c.system_name
        )

    def _phase_semantic(self, c: GenerationContext) -> None:
        c.semantic_fidelity_report = self._validate_terminal_semantic_obligations(
            c.final_model, c.final_sysml, c.system_name
        )

    def _phase_qualification(self, c: GenerationContext) -> None:
        from ..prototyping.model_qualification import build_model_qualification
        c.model_qualification = build_model_qualification(
            model_text=c.final_sysml,
            requirements=c.requirements,
            syntax_result=check_syntax(
                c.final_sysml, fail_closed=True,
                filter_stdlib_diagnostics=False,
            ),
            simulation_result=c.final_sim,
            terminal_consistency=c.terminal_consistency,
            structural_obligation_report=c.structural_obligation_report,
            semantic_fidelity_report=c.semantic_fidelity_report,
            generation_plan_conformance=c.generation_plan_conformance,
            ag_contract_graph=c.collaboration_artifacts.get("ag_contract_graph"),
            pattern_conformance_report=c.collaboration_artifacts.get(
                "pattern_conformance_report"
            ),
            ag_binding_report=self.last_ag_binding_report,
            ag_non_degradation=self.last_ag_non_degradation,
            ag_expected=self._active_ag_generation_plan is not None,
            generation_plan_expected=self._active_model_generation_plan is not None,
            semantic_fidelity_expected=bool(
                (c.semantic_fidelity_report or {}).get("total")
            ),
        )
        self.state.current_model = c.final_model
        print(f"  ✓ Final design score: {c.final_score:.3f}\n")

    def _phase_reporting(self, c: GenerationContext) -> None:
        print("Phase 4: Behavioral Reachability Simulation", flush=True)
        print("-" * 40)
        self._print_final_sim(c.final_sim)
        self._print_generation_summary(
            c.final_model, c.final_score, c.final_sim,
            c.model_qualification, c.requirements,
        )
        ledger = getattr(self.llm, "ledger", None)
        c.result = {
            "system_name": c.system_name,
            "requirements": c.requirements,
            "model": c.final_model,
            "model_sysml": c.final_sysml,
            "model_summary": c.final_model.get_summary(),
            "final_score": c.final_score,
            "iterations": self.state.iteration,
            "evaluation_history": self.state.evaluation_history,
            "simulation_result": c.final_sim,
            "terminal_consistency": c.terminal_consistency,
            "model_qualification": c.model_qualification,
            "model_acceptance_status": c.model_qualification["status"],
            "whole_model_generation_plan": c.final_model.metadata.get("whole_model_generation_plan"),
            "generation_plan_conformance": c.final_model.metadata.get("generation_plan_conformance"),
            "step1_plan_attempts": c.final_model.metadata.get("step1_plan_attempts"),
            "step1_plan_retries": c.final_model.metadata.get("step1_plan_retries", 0),
            "structural_obligation_report": c.structural_obligation_report,
            "semantic_fidelity_report": c.semantic_fidelity_report,
            "ag_binding_report": self.last_ag_binding_report,
            "ag_non_degradation": self.last_ag_non_degradation,
            "action_semantics_audit": self.last_action_semantics_audit,
            "functional_closure": dict(self.last_functional_closure or {}),
            "verification_anchor_attempts": list(self.last_verification_anchor_attempts),
            "requirement_semantic_analysis": dict(self.last_requirement_semantic_analysis or {}),
            "requirement_input": dict(self.last_requirement_input or {}),
            **c.collaboration_artifacts,
            "platform_profile": c.platform_profile,
            "llm_usage": ledger.as_dict() if ledger is not None else None,
        }

    def _print_generation_summary(
        self, final_model, final_score, final_sim, model_qualification,
        requirements,
    ) -> None:
        sim_warnings = (getattr(final_model, "metadata", None) or {}).get(
            "sim_warnings", ""
        )
        print(f"{'='*60}")
        print("Generation Complete!")
        print(f"  Final score:              {final_score:.3f}")
        print(f"  Model qualification:      {model_qualification['status']}")
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
                f"  Simulation reachability:  {final_sim.reachability_score:.3f} "
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
