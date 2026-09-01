"""Board-driven knowledge sources for the complete generation pipeline."""
from __future__ import annotations

from typing import Any, Callable, Mapping
from ..prototyping.action_effects import LEGACY_AUDIT
from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import get_sysml_text
from .planned_action_lifecycle import (
    observe_terminal_actions,
    prepare_planned_actions,
)
from .pipeline_records import GenerationContext
from .refinement import ModelRevision, RefinementClosureRequest



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
        c.refined_revision = self.refinement_closure.refine(
            RefinementClosureRequest(
                base=ModelRevision.capture(c.model),
                requirements=tuple(c.requirements),
            )
        )
        c.final_model, c.final_score, c.final_sim = (
            c.refined_revision.materialize()
        )
        self._attempt_frozen_plan_revision(c)

    def _attempt_frozen_plan_revision(self, c: GenerationContext) -> None:
        """The one path back from a plan-frozen structural deadlock.

        Refinement's `structural_repair_blocked` verdict names its own remedy
        — "requires a validated plan revision" — and until this method that
        remedy had no code path: the plan was authored once, frozen, and
        every downstream repair was bounded by it, so a wrong plan doomed
        the run to idle iterations and NOT_QUALIFIED (run 2026-08-31,
        265k tokens). Bounded sequence, once per run:

        1. `PlanRevision` re-enters the typed-plan LLM protocol with the
           frozen plan as repair base and the blockage as the authorizing
           issues; the revision is validated by the full current plan
           validator set and diff-gated to issue-named changes only.
        2. The revised plan is applied to the committed text and accepted
           only if the unsatisfied obligation set STRICTLY shrinks and
           syntax holds — otherwise everything is rolled back.
        3. On acceptance, one further bounded refinement pass runs under
           the revised plan.
        """
        metadata = getattr(c.final_model, "metadata", None) or {}
        blocked = metadata.get("structural_repair_blocked")
        if not isinstance(blocked, Mapping):
            return
        if not getattr(self, "enable_plan_revision", True):
            return
        raw_plan = metadata.get("whole_model_generation_plan")
        if not isinstance(raw_plan, Mapping):
            raw_plan = self._active_model_generation_plan
        if not isinstance(raw_plan, Mapping):
            return

        print("  ⟳ structural repair blocked — attempting bounded plan "
              "revision", flush=True)
        from ..prototyping.generation_plan import apply_generation_plan
        from ..prototyping.structural_obligations import (
            validate_structural_obligations,
        )
        from ..utils.sysml_text_utils import set_sysml_text
        from .plan_revision import PlanRevision, PlanRevisionRequest

        outcome = PlanRevision(self.cot).revise(PlanRevisionRequest(
            system_name=c.system_name,
            requirements=tuple(c.requirements),
            frozen_plan=dict(raw_plan),
            blocked=dict(blocked),
            verbose=self.verbose,
        ))
        record = dict(outcome.record)
        if outcome.plan is None:
            record.setdefault("applied", False)
            c.final_model.metadata["plan_revision"] = record
            self._publish_pipeline_state("plan_revision", record)
            print(
                f"  ✗ plan revision {record.get('status')} — the frozen "
                "plan stands; downstream gates will report the blockage",
                flush=True,
            )
            return

        before_report = blocked.get("structural_obligation_report") or {}
        before_unsatisfied = {
            str(item.get("obligation_id"))
            for item in before_report.get("results", ())
            if isinstance(item, Mapping) and item.get("status") != "PASS"
        }
        text = get_sysml_text(c.final_model)
        revised_text, conformance = apply_generation_plan(
            text, outcome.plan
        )
        # A mechanically-fixable doc spelling in the underlying text must
        # not veto the revision: s0v11's acceptance check read
        # syntax_ok=False from a doc "..." a late writer left behind, and
        # the revision was rejected for a defect it never caused.
        from ..sysml.text_normalization import fix_doc_syntax
        revised_text, _n = fix_doc_syntax(revised_text)
        after_report = validate_structural_obligations(
            revised_text,
            outcome.plan.structural_obligations,
            model_name=c.final_model.name,
        )
        after_unsatisfied = {
            str(item.get("obligation_id"))
            for item in after_report.get("results", ())
            if isinstance(item, Mapping) and item.get("status") != "PASS"
        }
        syntax_ok = not check_syntax(
            revised_text,
            fail_closed=True,
            filter_stdlib_diagnostics=False,
        ).has_errors
        monotonic = (
            after_unsatisfied < before_unsatisfied
            if before_unsatisfied
            else not after_unsatisfied
        )
        record.update({
            "unsatisfied_before": sorted(before_unsatisfied),
            "unsatisfied_after": sorted(after_unsatisfied),
            "syntax_ok": syntax_ok,
            "monotonic": monotonic,
        })
        if not (monotonic and syntax_ok):
            record["status"] = "REJECTED_NOT_MONOTONIC"
            record["applied"] = False
            c.final_model.metadata["plan_revision"] = record
            self._publish_pipeline_state("plan_revision", record)
            print(
                "  ✗ plan revision rejected: applying it does not strictly "
                "shrink the unsatisfied obligation set "
                f"({len(before_unsatisfied)} -> {len(after_unsatisfied)}, "
                f"syntax_ok={syntax_ok}); the frozen plan stands",
                flush=True,
            )
            return

        record["applied"] = True
        revised_payload = outcome.plan.to_dict()
        set_sysml_text(c.final_model, revised_text)
        c.final_model.metadata["whole_model_generation_plan"] = (
            revised_payload
        )
        c.final_model.metadata["plan_revision"] = record
        c.final_model.metadata["generation_plan_conformance"] = conformance
        c.final_model.metadata["structural_obligation_report"] = after_report
        c.final_model.metadata.pop("structural_repair_blocked", None)
        c.final_model.metadata.pop("refinement_short_circuit", None)
        self._active_model_generation_plan = dict(revised_payload)
        self._publish_pipeline_state("plan_revision", record)
        print(
            "  ✓ plan revision accepted: unsatisfied obligations "
            f"{len(before_unsatisfied)} -> {len(after_unsatisfied)}; "
            "re-running bounded refinement under the revised plan",
            flush=True,
        )
        c.refined_revision = self.refinement_closure.refine(
            RefinementClosureRequest(
                base=ModelRevision.capture(c.final_model),
                requirements=tuple(c.requirements),
            )
        )
        c.final_model, c.final_score, c.final_sim = (
            c.refined_revision.materialize()
        )

    def _phase_sitl_refinement(self, c: GenerationContext) -> None:
        # ── Phase 3.5: SITL-L1 refinement (only when targeting a platform) ────
        # Feed unresolved ArduPilot-parameter mappings (= model genuinely
        # missing a guard/attribute a requirement needs) back to the design
        # LLM.  Cheap & deterministic (no SITL process launch); L2 stays
        # terminal.
        if c.refined_revision is None:
            raise RuntimeError("parameter projection requires refined revision")
        c.projected_revision = self.refinement_closure.project_parameters(
            c.refined_revision,
            c.platform_profile,
        )
        c.final_model, c.final_score, c.final_sim = (
            c.projected_revision.materialize()
        )

    def _phase_functional_closure(self, c: GenerationContext) -> None:
        # Terminal model mutation: runs after ordinary and optional SITL-L1
        # refinement so no later LLM rewrite can overwrite functional closure.
        if c.projected_revision is None:
            raise RuntimeError("functional closure requires parameter projection")
        c.refinement_closure_outcome = self.refinement_closure.close(
            c.projected_revision,
        )
        c.final_model, c.final_score, c.final_sim = (
            c.refinement_closure_outcome.materialize()
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
        c.pre_ag_sim = self.refinement_closure.simulate(
            c.pre_ag_sysml, c.system_name
        )

    def _phase_ag_reconciliation(self, c: GenerationContext) -> None:
        # Redundant redeclarations of inherited ports are semantically inert
        # but each one costs a namespace-distinguishability warning at the
        # zero-warning terminal qualification (run 00e4d333: ten of them).
        from ..sysml.text_normalization import strip_redundant_inherited_ports
        c.pre_ag_sysml, n_stripped = strip_redundant_inherited_ports(
            c.pre_ag_sysml
        )
        if n_stripped:
            print(
                f"  ⟳ stripped {n_stripped} redundant inherited port "
                "redeclaration(s) — no LLM needed",
                flush=True,
            )
        c.final_sysml = self._reconcile_guided_ag_contract_layer(
            c.pre_ag_sysml, c.requirements
        )

    def _phase_terminal_commit(self, c: GenerationContext) -> None:
        # Terminal deterministic normalisation, regardless of which exit
        # path produced the text: a late writer (surgical merge, AG layer)
        # can introduce mechanically-fixable spellings AFTER the last
        # in-loop syntax gate ran. Measured on s0v11: a justification
        # doc "..." written mid-refinement reached qualification as a
        # parser error the assembly-time fixer never saw.
        from ..sysml.text_normalization import fix_doc_syntax
        c.final_sysml, n_doc_fixed = fix_doc_syntax(c.final_sysml)
        if n_doc_fixed:
            print(f"  ⟳ terminal normalisation: fixed {n_doc_fixed} "
                  "doc spelling(s) — no LLM needed", flush=True)
        c.planned_action_preparation = prepare_planned_actions(
            c.final_sysml,
            model_plan=self._active_model_generation_plan,
            ag_plan=self._active_ag_generation_plan,
        )
        self._publish_pipeline_state(
            "planned_action_preparation", c.planned_action_preparation
        )
        c.final_sysml = c.planned_action_preparation.model_text
        self._commit_terminal_model(
            c.final_sysml, producer="Orchestrator.generate"
        )
        self._ensure_terminal_ready()

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
        self.refinement_closure.verify_terminal(
            c.final_sysml, c.system_name
        )
        (
            c.final_model, c.final_score, c.final_sim, c.terminal_consistency
        ) = self._synchronize_terminal_snapshot(
            c.final_model, c.final_sysml, c.requirements,
            prior_score=c.pre_terminal_score, dse_best_config=None,
        )
        if c.planned_action_preparation is None:
            raise RuntimeError(
                "terminal snapshot requires planned-action preparation"
            )
        c.planned_action_observation = observe_terminal_actions(
            c.planned_action_preparation,
            c.final_sysml,
            requirements=c.requirements,
            profile=getattr(self, "action_semantics_profile", LEGACY_AUDIT),
        )
        self._publish_pipeline_state(
            "planned_action_observation", c.planned_action_observation
        )

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
            "plan_revision": c.final_model.metadata.get("plan_revision"),
            "structural_obligation_report": c.structural_obligation_report,
            "semantic_fidelity_report": c.semantic_fidelity_report,
            "ag_binding_report": self.last_ag_binding_report,
            "ag_non_degradation": self.last_ag_non_degradation,
            "action_semantics_audit": (
                c.planned_action_observation.to_artifact_dict()
                if c.planned_action_observation is not None else None
            ),
            "functional_closure": dict(self.last_functional_closure or {}),
            "verification_anchor_attempts": list(self.last_verification_anchor_attempts),
            "plan_conformance_rejections": list(self.last_plan_conformance_rejections),
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
        from .summary_rendering import (
            runtime_footer_lines,
            simulation_summary_lines,
        )

        print(f"{'='*60}")
        print("Generation Complete!")
        print(f"  Final score:              {final_score:.3f}")
        print(f"  Model qualification:      {model_qualification['status']}")
        for line in simulation_summary_lines(final_sim):
            print(line)
        print(f"  Part definitions: {len(final_model.part_definitions)}")
        print(f"  Requirements:     {len(requirements)}")
        for line in runtime_footer_lines(
            final_model, self.llm, verbose=self.verbose
        ):
            print(line)
        print(f"{'='*60}\n")
